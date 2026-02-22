import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import structlog
from PIL import Image, ImageDraw, ImageFont
from simple_parsing import parse

from tetris_core import MCTSConfig, evaluate_model, evaluate_model_without_nn
from tetris_bot.constants import BENCHMARKS_DIR, PARALLEL_ONNX_FILENAME

logger = structlog.get_logger()


@dataclass
class ScriptArgs:
    model_path: Path = BENCHMARKS_DIR / "models" / PARALLEL_ONNX_FILENAME
    use_dummy_network: bool = True  # Run bootstrap MCTS without loading an ONNX model
    num_games: int = 60  # Number of games per configuration
    simulations: list[int] = field(  # MCTS simulations per move
        default_factory=lambda: [50, 100, 200, 500, 1000, 2000, 4000]
    )
    max_placements: int = 50  # Maximum placements per game
    seed_start: int = 42  # Starting seed
    mcts_seed: int = 123  # Deterministic MCTS seed for reproducibility
    death_penalty: float = 10.0
    overhang_penalty_weight: float = 75.0
    num_workers: int = 7  # Parallel workers for evaluation
    add_noise: bool = False
    dirichlet_alpha: float = 0.02
    dirichlet_epsilon: float = 0.25
    output_plot: Path = BENCHMARKS_DIR / "simple_evaluate_reuse_vs_no_reuse.png"
    cache_path: Path = BENCHMARKS_DIR / "simple_evaluate_reuse_vs_no_reuse_cache.json"
    reuse_cached_results: bool = (
        True  # Reuse cache when eval args match to skip expensive reruns
    )
    force_recompute: bool = False  # Ignore cache and rerun evaluations
    plot_only: bool = False  # Rebuild plot from cache without running evaluations


def run_config(
    args: ScriptArgs,
    num_simulations: int,
    reuse_tree: bool,
) -> dict:
    config = MCTSConfig()
    config.num_simulations = num_simulations
    config.max_placements = args.max_placements
    config.seed = args.mcts_seed
    config.death_penalty = args.death_penalty
    config.overhang_penalty_weight = args.overhang_penalty_weight
    config.reuse_tree = reuse_tree
    config.dirichlet_alpha = args.dirichlet_alpha
    config.dirichlet_epsilon = args.dirichlet_epsilon

    seeds = list(range(args.seed_start, args.seed_start + args.num_games))

    start = time.perf_counter()

    if args.use_dummy_network:
        result = evaluate_model_without_nn(
            seeds=seeds,
            config=config,
            max_placements=args.max_placements,
            num_workers=args.num_workers,
            add_noise=args.add_noise,
        )
    else:
        result = evaluate_model(
            model_path=str(args.model_path),
            seeds=seeds,
            config=config,
            max_placements=args.max_placements,
            num_workers=args.num_workers,
            add_noise=args.add_noise,
        )

    elapsed = time.perf_counter() - start

    games_per_sec = args.num_games / elapsed if elapsed > 0 else 0
    avg_tree_nodes = getattr(result, "avg_tree_nodes", None)
    avg_tree_nodes = float(avg_tree_nodes) if avg_tree_nodes is not None else None

    return {
        "num_simulations": num_simulations,
        "reuse_tree": reuse_tree,
        "elapsed_sec": elapsed,
        "games_per_sec": games_per_sec,
        "avg_attack": result.avg_attack,
        "avg_tree_nodes": avg_tree_nodes,
        "num_games": result.num_games,
        "avg_moves": result.avg_moves,
        "max_attack": result.max_attack,
    }


def _coord(
    value: float,
    value_min: float,
    value_max: float,
    pixel_min: float,
    pixel_max: float,
) -> int:
    if value_max == value_min:
        return int((pixel_min + pixel_max) / 2)
    ratio = (value - value_min) / (value_max - value_min)
    return int(pixel_min + ratio * (pixel_max - pixel_min))


def create_plot(results: list[dict], output_path: Path) -> None:
    width = 980
    height = 1220
    left = 95
    right = 40
    top = 80
    bottom = 80
    panel_gap = 70
    num_panels = 3
    plot_width = width - left - right
    plot_height = int((height - top - bottom - panel_gap * (num_panels - 1)) / num_panels)
    panel_tops = [top + i * (plot_height + panel_gap) for i in range(num_panels)]

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    grouped_attack: dict[bool, dict[int, float]] = {False: {}, True: {}}
    grouped_gps: dict[bool, dict[int, float]] = {False: {}, True: {}}
    grouped_tree_nodes: dict[bool, dict[int, float]] = {False: {}, True: {}}
    for row in results:
        reuse_tree = bool(row["reuse_tree"])
        num_simulations = int(row["num_simulations"])
        grouped_attack[reuse_tree][num_simulations] = float(row["avg_attack"])
        grouped_gps[reuse_tree][num_simulations] = float(row["games_per_sec"])
        grouped_tree_nodes[reuse_tree][num_simulations] = float(row["avg_tree_nodes"])

    sim_values = sorted({int(row["num_simulations"]) for row in results})

    x_pos: dict[int, int] = {}
    for i, sim in enumerate(sim_values):
        x_pos[sim] = _coord(
            i,
            0,
            max(1, len(sim_values) - 1),
            left,
            left + plot_width,
        )

    def draw_series(
        panel_top: int,
        y_min: float,
        y_max: float,
        grouped: dict[bool, dict[int, float]],
        reuse_tree: bool,
        color: str,
    ) -> None:
        points: list[tuple[int, int]] = []
        for sim in sim_values:
            if sim not in grouped[reuse_tree] or sim not in x_pos:
                continue
            x = x_pos[sim]
            y = _coord(
                grouped[reuse_tree][sim],
                y_min,
                y_max,
                panel_top + plot_height,
                panel_top,
            )
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=color)

    def draw_panel(
        panel_top: int,
        grouped: dict[bool, dict[int, float]],
        metric_key: str,
        panel_title: str,
        y_label: str,
        show_x_labels: bool,
        tick_decimals: int = 2,
    ) -> None:
        values = [float(row[metric_key]) for row in results]
        value_min = min(values)
        value_max = max(values)
        padding = max(0.05, (value_max - value_min) * 0.1)
        y_min = max(0.0, value_min - padding)
        y_max = value_max + padding

        draw.rectangle(
            [left, panel_top, left + plot_width, panel_top + plot_height],
            outline="#222",
        )
        for tick in range(6):
            y_val = y_min + (y_max - y_min) * (tick / 5)
            y = _coord(y_val, y_min, y_max, panel_top + plot_height, panel_top)
            draw.line([(left, y), (left + plot_width, y)], fill="#e0e0e0", width=1)
            draw.text(
                (left - 75, y - 8),
                f"{y_val:.{tick_decimals}f}",
                fill="#111",
                font=font,
            )

        for sim in sim_values:
            x = x_pos[sim]
            draw.line(
                [(x, panel_top), (x, panel_top + plot_height)], fill="#f0f0f0", width=1
            )
            if show_x_labels:
                label = str(sim)
                label_w = draw.textlength(label, font=font)
                draw.text(
                    (x - label_w / 2, panel_top + plot_height + 10),
                    label,
                    fill="#111",
                    font=font,
                )

        draw_series(panel_top, y_min, y_max, grouped, False, "#c62828")
        draw_series(panel_top, y_min, y_max, grouped, True, "#2e7d32")

        title_w = draw.textlength(panel_title, font=font)
        draw.text(
            (left + (plot_width - title_w) / 2, panel_top - 24),
            panel_title,
            fill="#111",
            font=font,
        )
        draw.text((10, panel_top + plot_height / 2), y_label, fill="#111", font=font)

    draw_panel(
        panel_top=panel_tops[0],
        grouped=grouped_attack,
        metric_key="avg_attack",
        panel_title="Average Attack vs Simulations",
        y_label="Avg Attack",
        show_x_labels=False,
        tick_decimals=2,
    )
    draw_panel(
        panel_top=panel_tops[1],
        grouped=grouped_gps,
        metric_key="games_per_sec",
        panel_title="Games/Sec vs Simulations",
        y_label="Games/Sec",
        show_x_labels=False,
        tick_decimals=2,
    )
    draw_panel(
        panel_top=panel_tops[2],
        grouped=grouped_tree_nodes,
        metric_key="avg_tree_nodes",
        panel_title="Average Tree Nodes vs Simulations",
        y_label="Avg Tree Nodes",
        show_x_labels=True,
        tick_decimals=1,
    )

    draw.rectangle((left, 24, left + 18, 34), fill="#c62828")
    draw.text((left + 25, 22), "No tree reuse", fill="#111", font=font)
    draw.rectangle((left + 150, 24, left + 168, 34), fill="#2e7d32")
    draw.text((left + 175, 22), "With tree reuse", fill="#111", font=font)

    main_title = "Tree Reuse Comparison Across Simulations"
    main_title_w = draw.textlength(main_title, font=font)
    draw.text(((width - main_title_w) / 2, 6), main_title, fill="#111", font=font)

    x_label = "MCTS Simulations"
    x_label_w = draw.textlength(x_label, font=font)
    draw.text(((width - x_label_w) / 2, height - 30), x_label, fill="#111", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def print_results_table(results: list[dict]) -> None:
    by_sim = {
        int(sim): {} for sim in sorted({int(row["num_simulations"]) for row in results})
    }
    for row in results:
        by_sim[int(row["num_simulations"])][bool(row["reuse_tree"])] = {
            "avg_attack": float(row["avg_attack"]),
            "games_per_sec": float(row["games_per_sec"]),
            "avg_tree_nodes": float(row["avg_tree_nodes"]),
        }

    print(
        f"{'Simulations':>12}  {'No Reuse Atk':>12}  {'With Reuse Atk':>14}  {'No Reuse G/s':>12}  {'With Reuse G/s':>14}  {'No Reuse Nodes':>14}  {'With Reuse Nodes':>16}"
    )
    print("-" * 116)
    for sim, entries in by_sim.items():
        no_reuse_attack = entries.get(False, {}).get("avg_attack", float("nan"))
        with_reuse_attack = entries.get(True, {}).get("avg_attack", float("nan"))
        no_reuse_gps = entries.get(False, {}).get("games_per_sec", float("nan"))
        with_reuse_gps = entries.get(True, {}).get("games_per_sec", float("nan"))
        no_reuse_nodes = entries.get(False, {}).get("avg_tree_nodes", float("nan"))
        with_reuse_nodes = entries.get(True, {}).get("avg_tree_nodes", float("nan"))
        print(
            f"{sim:>12}  {no_reuse_attack:>12.2f}  {with_reuse_attack:>14.2f}  {no_reuse_gps:>12.2f}  {with_reuse_gps:>14.2f}  {no_reuse_nodes:>14.1f}  {with_reuse_nodes:>16.1f}"
        )


def normalize_results(results: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for row in results:
        normalized_row = dict(row)
        if normalized_row.get("avg_tree_nodes") is not None:
            normalized_row["avg_tree_nodes"] = float(normalized_row["avg_tree_nodes"])
        normalized.append(normalized_row)
    return sorted(
        normalized,
        key=lambda r: (int(r["num_simulations"]), bool(r["reuse_tree"])),
    )


def has_tree_node_metrics(results: list[dict]) -> bool:
    return all(row.get("avg_tree_nodes") is not None for row in results)


def build_eval_config(args: ScriptArgs, simulations: list[int]) -> dict:
    return {
        "use_dummy_network": bool(args.use_dummy_network),
        "model_path": None if args.use_dummy_network else str(args.model_path),
        "num_games": int(args.num_games),
        "simulations": [int(s) for s in simulations],
        "max_placements": int(args.max_placements),
        "seed_start": int(args.seed_start),
        "mcts_seed": int(args.mcts_seed),
        "death_penalty": float(args.death_penalty),
        "overhang_penalty_weight": float(args.overhang_penalty_weight),
        "num_workers": int(args.num_workers),
        "add_noise": bool(args.add_noise),
        "dirichlet_alpha": float(args.dirichlet_alpha),
        "dirichlet_epsilon": float(args.dirichlet_epsilon),
    }


def load_cache(cache_path: Path) -> dict | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Failed to read cache; ignoring",
            path=str(cache_path),
            error=str(exc),
        )
        return None
    if not isinstance(payload, dict):
        logger.warning("Invalid cache format; ignoring", path=str(cache_path))
        return None
    if not isinstance(payload.get("results"), list):
        logger.warning("Cache missing results; ignoring", path=str(cache_path))
        return None
    return payload


def save_cache(cache_path: Path, eval_config: dict, results: list[dict]) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_config": eval_config,
        "results": normalize_results(results),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2) + "\n")


def main(args: ScriptArgs) -> None:
    simulations = sorted(set(args.simulations))
    if not simulations:
        raise ValueError("simulations cannot be empty")

    eval_config = build_eval_config(args, simulations)
    cached_payload = load_cache(args.cache_path)

    all_results: list[dict] | None = None
    if args.plot_only:
        if cached_payload is None:
            raise FileNotFoundError(
                f"plot_only=true but cache file not found or invalid: {args.cache_path}"
            )
        all_results = normalize_results(cached_payload["results"])
        if not has_tree_node_metrics(all_results):
            raise ValueError(
                "Cached results do not contain avg_tree_nodes. "
                "Run once with --force_recompute true after rebuilding tetris_core."
            )
        logger.info("Loaded cached results (plot_only)", path=str(args.cache_path))
    elif (
        args.reuse_cached_results
        and not args.force_recompute
        and cached_payload is not None
        and cached_payload.get("eval_config") == eval_config
    ):
        all_results = normalize_results(cached_payload["results"])
        if has_tree_node_metrics(all_results):
            logger.info("Loaded cached results", path=str(args.cache_path))
        else:
            logger.info(
                "Cache missing avg_tree_nodes; recomputing evaluations",
                path=str(args.cache_path),
            )
            all_results = None

    if all_results is None:
        if not args.use_dummy_network and not args.model_path.exists():
            raise FileNotFoundError(f"Model not found: {args.model_path}")

        all_results = []
        for num_simulations in simulations:
            for reuse_tree in [False, True]:
                logger.info(
                    "Running evaluation",
                    num_simulations=num_simulations,
                    reuse_tree=reuse_tree,
                    num_games=args.num_games,
                    num_workers=args.num_workers,
                )
                all_results.append(
                    run_config(
                        args=args,
                        num_simulations=num_simulations,
                        reuse_tree=reuse_tree,
                    )
                )
        all_results = normalize_results(all_results)
        if not has_tree_node_metrics(all_results):
            raise RuntimeError(
                "avg_tree_nodes not available from tetris_core.EvalResult. "
                "Rebuild the extension (for example: make build-dev) and rerun."
            )
        save_cache(args.cache_path, eval_config, all_results)
        logger.info("Saved cache", path=str(args.cache_path))

    print_results_table(all_results)
    create_plot(all_results, args.output_plot)
    logger.info("Saved plot", path=str(args.output_plot))


if __name__ == "__main__":
    args = parse(ScriptArgs)
    main(args)
