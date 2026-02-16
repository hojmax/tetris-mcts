from __future__ import annotations

import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import structlog
from PIL import Image, ImageDraw, ImageFont
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from simple_parsing import parse

from tetris_core import MCTSAgent, MCTSConfig, TetrisEnv
from tetris_mcts.config import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    CHECKPOINT_DIRNAME,
    CONFIG_FILENAME,
    LATEST_ONNX_FILENAME,
    NUM_ACTIONS,
)

logger = structlog.get_logger()
HOLD_ACTION_INDEX = NUM_ACTIONS - 1
WORKER_AGENT: MCTSAgent | None = None
WORKER_MAX_PLACEMENTS: int | None = None


@dataclass
class ScriptArgs:
    run_dir: Path  # Training run directory
    nn_value_weights: list[float] = field(default_factory=lambda: [0.005, 0.025, 0.075])
    num_games: int = 50
    seed_start: int = 1

    model_path: Path | None = None
    num_simulations: int | None = None
    c_puct: float | None = None
    max_placements: int | None = None
    overhang_penalty_weight: float | None = None
    mcts_seed: int | None = None
    workers: int | None = None
    show_progress: bool = True
    log_each_game: bool = False

    output_json: Path | None = None
    output_plot: Path | None = None


def validate_args(args: ScriptArgs) -> None:
    if args.num_games <= 0:
        raise ValueError(f"num_games must be > 0 (got {args.num_games})")
    if not args.nn_value_weights:
        raise ValueError("nn_value_weights cannot be empty")
    if args.seed_start < 0:
        raise ValueError(f"seed_start must be >= 0 (got {args.seed_start})")
    if args.workers is not None and args.workers <= 0:
        raise ValueError(f"workers must be > 0 (got {args.workers})")

    seen = set()
    for weight in args.nn_value_weights:
        if weight < 0.0:
            raise ValueError(f"nn_value_weight must be >= 0 (got {weight})")
        if weight in seen:
            raise ValueError(f"nn_value_weights contains duplicate value: {weight}")
        seen.add(weight)


def load_run_config_data(run_dir: Path) -> dict:
    config_path = run_dir / CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    config_data = json.loads(config_path.read_text())
    if not isinstance(config_data, dict):
        raise ValueError(f"Run config must be a JSON object: {config_path}")
    return config_data


def get_config_or_default(
    run_config_data: dict,
    key: str,
    default_value: int | float,
) -> int | float:
    value = run_config_data.get(key, default_value)
    if value is None:
        raise ValueError(f"Config value for {key} cannot be None")
    return value


def resolve_model_path(args: ScriptArgs) -> Path:
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = args.run_dir / CHECKPOINT_DIRNAME / LATEST_ONNX_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return model_path


def resolve_output_paths(args: ScriptArgs) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_base = args.run_dir / "analysis" / f"nn_value_weight_eval_{timestamp}"
    output_json = args.output_json or default_base.with_suffix(".json")
    output_plot = args.output_plot or default_base.with_suffix(".png")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    return output_json, output_plot


def build_mcts_config(
    num_simulations: int,
    c_puct: float,
    max_placements: int,
    overhang_penalty_weight: float,
    mcts_seed: int | None,
    nn_value_weight: float,
) -> MCTSConfig:
    config = MCTSConfig()
    config.num_simulations = num_simulations
    config.c_puct = c_puct
    config.max_placements = max_placements
    config.overhang_penalty_weight = overhang_penalty_weight
    config.visit_sampling_epsilon = 0.0
    config.seed = mcts_seed
    config.nn_value_weight = nn_value_weight
    return config


def initialize_worker_agent(
    model_path: str,
    num_simulations: int,
    c_puct: float,
    max_placements: int,
    overhang_penalty_weight: float,
    mcts_seed: int | None,
    nn_value_weight: float,
) -> None:
    global WORKER_AGENT
    global WORKER_MAX_PLACEMENTS

    config = build_mcts_config(
        num_simulations=num_simulations,
        c_puct=c_puct,
        max_placements=max_placements,
        overhang_penalty_weight=overhang_penalty_weight,
        mcts_seed=mcts_seed,
        nn_value_weight=nn_value_weight,
    )
    config.temperature = 0.0

    agent = MCTSAgent(config)
    if not agent.load_model(model_path):
        raise RuntimeError(f"Failed to load model in worker: {model_path}")

    WORKER_AGENT = agent
    WORKER_MAX_PLACEMENTS = max_placements


def evaluate_single_seed(
    seed: int,
) -> dict:
    if WORKER_AGENT is None or WORKER_MAX_PLACEMENTS is None:
        raise RuntimeError("Worker agent not initialized")

    env = TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, int(seed))
    placement_count = 0
    game_attack = 0
    game_lines = 0
    game_moves = 0

    while placement_count < WORKER_MAX_PLACEMENTS:
        if env.game_over:
            break

        result = WORKER_AGENT.select_action(
            env=env,
            add_noise=False,
            placement_count=placement_count,
        )
        if result is None:
            break

        action = int(result.action)
        attack = env.execute_action_index(action)
        if attack is None:
            raise RuntimeError(
                f"MCTS selected non-executable action for seed={seed}: {action}"
            )
        game_attack += int(attack)

        if action != HOLD_ACTION_INDEX:
            placement_count += 1
            game_moves += 1

        attack_result = env.get_last_attack_result()
        if attack_result is not None:
            game_lines += int(attack_result.lines_cleared)

    return {
        "seed": int(seed),
        "attack": int(game_attack),
        "moves": int(game_moves),
        "lines": int(game_lines),
    }


def aggregate_weight_results(nn_value_weight: float, game_rows: list[dict]) -> dict:
    game_rows.sort(key=lambda row: row["seed"])
    num_games = len(game_rows)
    total_attack = sum(int(row["attack"]) for row in game_rows)
    max_attack = max((int(row["attack"]) for row in game_rows), default=0)
    total_lines = sum(int(row["lines"]) for row in game_rows)
    max_lines = max((int(row["lines"]) for row in game_rows), default=0)
    total_moves = sum(int(row["moves"]) for row in game_rows)

    avg_attack = total_attack / num_games if num_games > 0 else 0.0
    avg_lines = total_lines / num_games if num_games > 0 else 0.0
    avg_moves = total_moves / num_games if num_games > 0 else 0.0
    attack_per_piece = total_attack / total_moves if total_moves > 0 else 0.0
    lines_per_piece = total_lines / total_moves if total_moves > 0 else 0.0

    attack_values = [row["attack"] for row in game_rows]
    attack_std = (
        float(statistics.pstdev(attack_values)) if len(attack_values) > 1 else 0.0
    )
    attack_sem = attack_std / math.sqrt(len(attack_values)) if attack_values else 0.0

    return {
        "nn_value_weight": float(nn_value_weight),
        "num_games": int(num_games),
        "total_attack": int(total_attack),
        "max_attack": int(max_attack),
        "total_lines": int(total_lines),
        "max_lines": int(max_lines),
        "total_moves": int(total_moves),
        "avg_attack": float(avg_attack),
        "avg_lines": float(avg_lines),
        "avg_moves": float(avg_moves),
        "attack_per_piece": float(attack_per_piece),
        "lines_per_piece": float(lines_per_piece),
        "attack_std": attack_std,
        "attack_sem": attack_sem,
        "game_results": game_rows,
    }


def evaluate_weight(
    model_path: Path,
    seeds: list[int],
    num_simulations: int,
    c_puct: float,
    max_placements: int,
    overhang_penalty_weight: float,
    mcts_seed: int | None,
    nn_value_weight: float,
    workers: int,
    show_progress: bool,
    log_each_game: bool,
) -> dict:
    effective_workers = max(1, min(workers, len(seeds)))
    game_rows = []

    progress = None
    task_id = None
    if show_progress:
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        progress.start()
        task_id = progress.add_task(
            description=f"nn_value_weight={nn_value_weight:g}", total=len(seeds)
        )

    try:
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            initializer=initialize_worker_agent,
            initargs=(
                str(model_path),
                num_simulations,
                c_puct,
                max_placements,
                overhang_penalty_weight,
                mcts_seed,
                nn_value_weight,
            ),
        ) as executor:
            futures = {
                executor.submit(
                    evaluate_single_seed,
                    seed,
                ): seed
                for seed in seeds
            }
            for future in as_completed(futures):
                row = future.result()
                game_rows.append(row)

                if progress is not None and task_id is not None:
                    progress.update(task_id, advance=1)

                if log_each_game:
                    logger.info(
                        "Completed game",
                        nn_value_weight=nn_value_weight,
                        seed=row["seed"],
                        attack=row["attack"],
                        lines=row["lines"],
                        moves=row["moves"],
                    )
    finally:
        if progress is not None:
            progress.stop()

    return aggregate_weight_results(nn_value_weight, game_rows)


def text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> tuple[float, float]:
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    return x1 - x0, y1 - y0


def create_plot(results: list[dict], output_path: Path) -> None:
    rows = sorted(results, key=lambda row: row["nn_value_weight"])
    width = 980
    height = 620
    margin_left = 100
    margin_right = 40
    margin_top = 70
    margin_bottom = 110
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    y_values = [float(row["avg_attack"]) for row in rows]
    y_min = min(y_values)
    y_max = max(y_values)
    if y_max == y_min:
        y_max = y_max + 1.0
        y_min = max(0.0, y_min - 1.0)
    else:
        padding = (y_max - y_min) * 0.15
        y_min = max(0.0, y_min - padding)
        y_max = y_max + padding

    def x_coord(index: int) -> int:
        if len(rows) == 1:
            return margin_left + plot_width // 2
        return int(round(margin_left + index * (plot_width / (len(rows) - 1))))

    def y_coord(value: float) -> int:
        normalized = (value - y_min) / (y_max - y_min)
        return int(round(margin_top + plot_height * (1.0 - normalized)))

    for i in range(6):
        frac = i / 5
        y = int(round(margin_top + plot_height * frac))
        value = y_max - (y_max - y_min) * frac
        draw.line(
            [(margin_left, y), (margin_left + plot_width, y)],
            fill=(225, 225, 225),
            width=1,
        )
        tick_label = f"{value:.1f}"
        tw, th = text_size(draw, tick_label, font)
        draw.text(
            (margin_left - 10 - tw, y - th / 2), tick_label, fill="black", font=font
        )

    draw.line(
        [(margin_left, margin_top), (margin_left, margin_top + plot_height)],
        fill="black",
        width=2,
    )
    draw.line(
        [
            (margin_left, margin_top + plot_height),
            (margin_left + plot_width, margin_top + plot_height),
        ],
        fill="black",
        width=2,
    )

    points = [
        (x_coord(i), y_coord(float(row["avg_attack"]))) for i, row in enumerate(rows)
    ]
    if len(points) > 1:
        draw.line(points, fill=(39, 91, 166), width=3)

    for i, row in enumerate(rows):
        x, y = points[i]
        std = float(row["attack_std"])
        if std > 0:
            y_low = y_coord(float(row["avg_attack"]) - std)
            y_high = y_coord(float(row["avg_attack"]) + std)
            draw.line([(x, y_low), (x, y_high)], fill=(39, 91, 166), width=2)
            draw.line([(x - 5, y_low), (x + 5, y_low)], fill=(39, 91, 166), width=2)
            draw.line([(x - 5, y_high), (x + 5, y_high)], fill=(39, 91, 166), width=2)

        r = 5
        draw.ellipse(
            [(x - r, y - r), (x + r, y + r)], fill=(218, 64, 82), outline="black"
        )

        x_label = f"{row['nn_value_weight']:g}"
        tw, th = text_size(draw, x_label, font)
        draw.text(
            (x - tw / 2, margin_top + plot_height + 12),
            x_label,
            fill="black",
            font=font,
        )

        y_label = f"{row['avg_attack']:.2f}"
        yw, yh = text_size(draw, y_label, font)
        draw.text((x - yw / 2, y - yh - 10), y_label, fill="black", font=font)

    title = "Average Attack vs nn_value_weight"
    tw, th = text_size(draw, title, font)
    draw.text(((width - tw) / 2, 20), title, fill="black", font=font)

    x_axis_label = "nn_value_weight"
    xw, xh = text_size(draw, x_axis_label, font)
    draw.text(
        (margin_left + (plot_width - xw) / 2, height - 35),
        x_axis_label,
        fill="black",
        font=font,
    )

    y_axis_label = "Average attack"
    draw.text((20, margin_top - 25), y_axis_label, fill="black", font=font)

    image.save(output_path)


def main(args: ScriptArgs) -> None:
    validate_args(args)

    run_dir = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run path is not a directory: {run_dir}")

    run_config_data = load_run_config_data(run_dir)
    model_path = resolve_model_path(args)
    output_json, output_plot = resolve_output_paths(args)

    default_config = {
        "num_simulations": 1000,
        "c_puct": 1.5,
        "max_placements": 50,
        "overhang_penalty_weight": 5.0,
        "eval_mcts_seed": 12345,
    }
    num_simulations = (
        args.num_simulations
        if args.num_simulations is not None
        else int(
            get_config_or_default(
                run_config_data,
                "num_simulations",
                default_config["num_simulations"],
            )
        )
    )
    c_puct = (
        args.c_puct
        if args.c_puct is not None
        else float(
            get_config_or_default(
                run_config_data,
                "c_puct",
                default_config["c_puct"],
            )
        )
    )
    max_placements = (
        args.max_placements
        if args.max_placements is not None
        else int(
            get_config_or_default(
                run_config_data,
                "max_placements",
                default_config["max_placements"],
            )
        )
    )
    overhang_penalty_weight = (
        args.overhang_penalty_weight
        if args.overhang_penalty_weight is not None
        else float(
            get_config_or_default(
                run_config_data,
                "overhang_penalty_weight",
                default_config["overhang_penalty_weight"],
            )
        )
    )
    mcts_seed = (
        args.mcts_seed
        if args.mcts_seed is not None
        else int(
            get_config_or_default(
                run_config_data,
                "eval_mcts_seed",
                default_config["eval_mcts_seed"],
            )
        )
    )
    default_workers = int(get_config_or_default(run_config_data, "num_workers", 4))
    effective_workers = args.workers if args.workers is not None else default_workers
    effective_workers = min(effective_workers, args.num_games)
    if effective_workers <= 0:
        raise ValueError(
            "effective_workers must be > 0; check workers/num_games settings"
        )
    seeds = list(range(args.seed_start, args.seed_start + args.num_games))

    logger.info(
        "Starting nn_value_weight evaluation sweep",
        run_dir=str(run_dir),
        model_path=str(model_path),
        nn_value_weights=args.nn_value_weights,
        num_games=args.num_games,
        seed_start=args.seed_start,
        num_simulations=num_simulations,
        c_puct=c_puct,
        max_placements=max_placements,
        overhang_penalty_weight=overhang_penalty_weight,
        mcts_seed=mcts_seed,
        workers=effective_workers,
        show_progress=args.show_progress,
        log_each_game=args.log_each_game,
    )

    results = []
    for nn_value_weight in args.nn_value_weights:
        logger.info("Evaluating weight", nn_value_weight=nn_value_weight)
        result_row = evaluate_weight(
            model_path=model_path,
            seeds=seeds,
            num_simulations=num_simulations,
            c_puct=c_puct,
            max_placements=max_placements,
            overhang_penalty_weight=overhang_penalty_weight,
            mcts_seed=mcts_seed,
            nn_value_weight=nn_value_weight,
            workers=effective_workers,
            show_progress=args.show_progress,
            log_each_game=args.log_each_game,
        )
        results.append(result_row)
        logger.info(
            "Completed weight evaluation",
            nn_value_weight=nn_value_weight,
            avg_attack=result_row["avg_attack"],
            attack_std=result_row["attack_std"],
            max_attack=result_row["max_attack"],
        )

    results_by_weight = {
        f"{row['nn_value_weight']:g}": row["avg_attack"] for row in results
    }
    best_row = max(results, key=lambda row: row["avg_attack"])

    payload = {
        "created_at": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "num_games": args.num_games,
        "seed_start": args.seed_start,
        "seeds": seeds,
        "base_mcts_config": {
            "num_simulations": num_simulations,
            "c_puct": c_puct,
            "max_placements": max_placements,
            "overhang_penalty_weight": overhang_penalty_weight,
            "mcts_seed": mcts_seed,
        },
        "nn_value_weights": [float(value) for value in args.nn_value_weights],
        "results": results,
        "summary": {
            "avg_attack_by_nn_value_weight": results_by_weight,
            "best_nn_value_weight": float(best_row["nn_value_weight"]),
            "best_avg_attack": float(best_row["avg_attack"]),
        },
    }

    output_json.write_text(json.dumps(payload, indent=2))
    create_plot(results=results, output_path=output_plot)

    logger.info(
        "Finished nn_value_weight evaluation sweep",
        output_json=str(output_json),
        output_plot=str(output_plot),
        best_nn_value_weight=payload["summary"]["best_nn_value_weight"],
        best_avg_attack=payload["summary"]["best_avg_attack"],
    )


if __name__ == "__main__":
    main(parse(ScriptArgs))
