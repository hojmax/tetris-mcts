from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import structlog
from PIL import Image, ImageDraw, ImageFont
from simple_parsing import parse

from tetris_core import MCTSConfig, evaluate_model
from tetris_mcts.constants import (
    CHECKPOINT_DIRNAME,
    CONFIG_FILENAME,
    LATEST_ONNX_FILENAME,
)
from tetris_mcts.ml.artifacts import assert_rust_inference_artifacts

logger = structlog.get_logger()

SWEEPABLE_FLOAT_PARAMS = {
    "q_scale",
    "nn_value_weight",
    "c_puct",
    "death_penalty",
    "overhang_penalty_weight",
    "temperature",
    "dirichlet_alpha",
    "dirichlet_epsilon",
}


@dataclass
class ScriptArgs:
    run_dir: Path  # Training run to evaluate
    sweep_param: str = "q_scale"  # MCTSConfig field to sweep
    sweep_values: list[float] = field(
        default_factory=lambda: [2.0, 4.0, 8.0, 16.0, 32.0]
    )  # Values to sweep
    include_minmax: bool = (
        False  # Include min-max Q normalization baseline (q_scale sweep only)
    )
    num_games: int = 50  # Games per sweep value
    seed_start: int = 1  # First env seed

    model_path: Path | None = None  # Override model path
    num_simulations: int | None = None  # Override from run config
    c_puct: float | None = None  # Override from run config
    max_placements: int | None = None  # Override from run config
    overhang_penalty_weight: float | None = None  # Override from run config
    nn_value_weight: float | None = None  # Override from run config
    q_scale: float | None = None  # Override from run config
    mcts_seed: int | None = None  # Override from run config

    output_json: Path | None = None  # Output JSON path
    output_plot: Path | None = None  # Output PNG path


def validate_args(args: ScriptArgs) -> None:
    if args.num_games <= 0:
        raise ValueError(f"num_games must be > 0 (got {args.num_games})")
    if not args.sweep_values:
        raise ValueError("sweep_values cannot be empty")
    if args.seed_start < 0:
        raise ValueError(f"seed_start must be >= 0 (got {args.seed_start})")
    if args.sweep_param not in SWEEPABLE_FLOAT_PARAMS:
        raise ValueError(
            f"sweep_param must be one of {sorted(SWEEPABLE_FLOAT_PARAMS)}, "
            f"got {args.sweep_param!r}"
        )
    if args.include_minmax and args.sweep_param != "q_scale":
        raise ValueError(
            "include_minmax is only supported when sweep_param='q_scale', "
            f"got sweep_param={args.sweep_param!r}"
        )

    seen: set[float] = set()
    for value in args.sweep_values:
        if not math.isfinite(value):
            raise ValueError(f"sweep_values must be finite (got {value})")
        if value in seen:
            raise ValueError(f"sweep_values contains duplicate: {value}")
        seen.add(value)


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    data = json.loads(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Run config must be a JSON object: {config_path}")
    return data


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
    default_base = args.run_dir / "analysis" / f"{args.sweep_param}_sweep_{timestamp}"
    output_json = args.output_json or default_base.with_suffix(".json")
    output_plot = args.output_plot or default_base.with_suffix(".png")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    return output_json, output_plot


def resolve_config_value(
    cli_override: int | float | None,
    run_config: dict,
    key: str,
    default: int | float,
) -> int | float:
    if cli_override is not None:
        return cli_override
    value = run_config.get(key, default)
    if value is None:
        raise ValueError(f"Config value for {key} cannot be None")
    return value


def build_base_mcts_config(
    args: ScriptArgs,
    run_config: dict,
) -> MCTSConfig:
    config = MCTSConfig()
    config.num_simulations = int(
        resolve_config_value(args.num_simulations, run_config, "num_simulations", 1000)
    )
    config.c_puct = float(resolve_config_value(args.c_puct, run_config, "c_puct", 1.5))
    config.max_placements = int(
        resolve_config_value(args.max_placements, run_config, "max_placements", 50)
    )
    config.overhang_penalty_weight = float(
        resolve_config_value(
            args.overhang_penalty_weight,
            run_config,
            "overhang_penalty_weight",
            5.0,
        )
    )
    config.nn_value_weight = float(
        resolve_config_value(args.nn_value_weight, run_config, "nn_value_weight", 0.01)
    )
    config.q_scale = float(
        resolve_config_value(args.q_scale, run_config, "q_scale", 8.0)
    )
    config.visit_sampling_epsilon = 0.0
    config.temperature = 0.0

    mcts_seed = resolve_config_value(
        args.mcts_seed, run_config, "eval_mcts_seed", 12345
    )
    config.seed = int(mcts_seed)
    return config


def build_sweep_mcts_config(
    args: ScriptArgs,
    run_config: dict,
    sweep_value: float,
) -> MCTSConfig:
    config = build_base_mcts_config(args, run_config)

    if args.sweep_param == "q_scale":
        config.q_scale = sweep_value
    else:
        setattr(config, args.sweep_param, sweep_value)

    return config


def build_minmax_mcts_config(
    args: ScriptArgs,
    run_config: dict,
) -> MCTSConfig:
    config = build_base_mcts_config(args, run_config)
    config.q_scale = None
    return config


def aggregate_eval_result(
    label: str,
    sweep_value: float | None,
    seeds: list[int],
    result: object,
) -> dict:
    game_rows = [
        {"seed": seeds[i], "attack": int(attack), "moves": int(moves)}
        for i, (attack, moves) in enumerate(result.game_results)  # type: ignore[attr-defined]
    ]
    attack_values = [row["attack"] for row in game_rows]
    attack_std = (
        float(statistics.pstdev(attack_values)) if len(attack_values) > 1 else 0.0
    )
    attack_sem = attack_std / math.sqrt(len(attack_values)) if attack_values else 0.0

    return {
        "label": label,
        "sweep_value": sweep_value,
        "num_games": int(result.num_games),  # type: ignore[attr-defined]
        "avg_attack": float(result.avg_attack),  # type: ignore[attr-defined]
        "max_attack": int(result.max_attack),  # type: ignore[attr-defined]
        "avg_lines": float(result.avg_lines),  # type: ignore[attr-defined]
        "max_lines": int(result.max_lines),  # type: ignore[attr-defined]
        "avg_moves": float(result.avg_moves),  # type: ignore[attr-defined]
        "attack_per_piece": float(result.attack_per_piece),  # type: ignore[attr-defined]
        "lines_per_piece": float(result.lines_per_piece),  # type: ignore[attr-defined]
        "attack_std": attack_std,
        "attack_sem": attack_sem,
        "game_results": game_rows,
    }


def evaluate_sweep_value(
    model_path: Path,
    seeds: list[int],
    args: ScriptArgs,
    run_config: dict,
    sweep_value: float,
) -> dict:
    config = build_sweep_mcts_config(args, run_config, sweep_value)
    result = evaluate_model(
        model_path=str(model_path),
        seeds=[int(s) for s in seeds],
        config=config,
        max_placements=config.max_placements,
    )
    return aggregate_eval_result(
        label=f"{args.sweep_param}={sweep_value:g}",
        sweep_value=sweep_value,
        seeds=seeds,
        result=result,
    )


def evaluate_minmax(
    model_path: Path,
    seeds: list[int],
    args: ScriptArgs,
    run_config: dict,
) -> dict:
    config = build_minmax_mcts_config(args, run_config)
    result = evaluate_model(
        model_path=str(model_path),
        seeds=[int(s) for s in seeds],
        config=config,
        max_placements=config.max_placements,
    )
    return aggregate_eval_result(
        label="minmax",
        sweep_value=None,
        seeds=seeds,
        result=result,
    )


def text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> tuple[float, float]:
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    return x1 - x0, y1 - y0


def create_plot(
    results: list[dict],
    sweep_param: str,
    output_path: Path,
) -> None:
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

    # Order: minmax first (if present), then tanh values sorted ascending
    minmax_rows = [r for r in results if r["sweep_value"] is None]
    tanh_rows = sorted(
        [r for r in results if r["sweep_value"] is not None],
        key=lambda r: r["sweep_value"],
    )
    ordered = minmax_rows + tanh_rows

    y_values = [float(row["avg_attack"]) for row in ordered]
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
        if len(ordered) == 1:
            return margin_left + plot_width // 2
        return int(round(margin_left + index * (plot_width / (len(ordered) - 1))))

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
        (x_coord(i), y_coord(float(row["avg_attack"]))) for i, row in enumerate(ordered)
    ]

    # Draw lines between tanh points only (skip minmax for the connecting line)
    tanh_points = [
        points[i] for i, row in enumerate(ordered) if row["sweep_value"] is not None
    ]
    if len(tanh_points) > 1:
        draw.line(tanh_points, fill=(39, 91, 166), width=3)

    # Draw minmax horizontal reference line if present
    if minmax_rows:
        minmax_y = y_coord(float(minmax_rows[0]["avg_attack"]))
        draw.line(
            [(margin_left, minmax_y), (margin_left + plot_width, minmax_y)],
            fill=(180, 60, 60),
            width=1,
        )

    for i, row in enumerate(ordered):
        x, y = points[i]
        std = float(row["attack_std"])
        if std > 0:
            y_low = y_coord(float(row["avg_attack"]) - std)
            y_high = y_coord(float(row["avg_attack"]) + std)
            color = (180, 60, 60) if row["sweep_value"] is None else (39, 91, 166)
            draw.line([(x, y_low), (x, y_high)], fill=color, width=2)
            draw.line([(x - 5, y_low), (x + 5, y_low)], fill=color, width=2)
            draw.line([(x - 5, y_high), (x + 5, y_high)], fill=color, width=2)

        r = 5
        fill_color = (180, 180, 180) if row["sweep_value"] is None else (218, 64, 82)
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=fill_color, outline="black")

        x_label = row["label"]
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

    title = f"Average Attack vs {sweep_param} (Q normalization)"
    tw, th = text_size(draw, title, font)
    draw.text(((width - tw) / 2, 20), title, fill="black", font=font)

    xw, xh = text_size(draw, sweep_param, font)
    draw.text(
        (margin_left + (plot_width - xw) / 2, height - 35),
        sweep_param,
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

    run_config = load_run_config(run_dir)
    model_path = resolve_model_path(args)
    assert_rust_inference_artifacts(model_path)
    output_json, output_plot = resolve_output_paths(args)
    seeds = list(range(args.seed_start, args.seed_start + args.num_games))

    logger.info(
        "Starting MCTS config sweep",
        sweep_param=args.sweep_param,
        sweep_values=args.sweep_values,
        include_minmax=args.include_minmax,
        run_dir=str(run_dir),
        model_path=str(model_path),
        num_games=args.num_games,
        seed_start=args.seed_start,
    )

    results: list[dict] = []

    # Run min-max baseline first if requested
    if args.include_minmax:
        logger.info("Evaluating min-max Q normalization baseline")
        minmax_result = evaluate_minmax(
            model_path=model_path,
            seeds=seeds,
            args=args,
            run_config=run_config,
        )
        results.append(minmax_result)
        logger.info(
            "Completed min-max baseline",
            avg_attack=minmax_result["avg_attack"],
            attack_std=minmax_result["attack_std"],
            max_attack=minmax_result["max_attack"],
        )

    # Run tanh sweep values
    for sweep_value in args.sweep_values:
        logger.info(
            "Evaluating sweep value",
            sweep_param=args.sweep_param,
            sweep_value=sweep_value,
        )
        result = evaluate_sweep_value(
            model_path=model_path,
            seeds=seeds,
            args=args,
            run_config=run_config,
            sweep_value=sweep_value,
        )
        results.append(result)
        logger.info(
            "Completed sweep value",
            sweep_param=args.sweep_param,
            sweep_value=sweep_value,
            avg_attack=result["avg_attack"],
            attack_std=result["attack_std"],
            max_attack=result["max_attack"],
        )

    best_row = max(results, key=lambda row: row["avg_attack"])

    base_config = build_base_mcts_config(args, run_config)

    payload = {
        "created_at": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "sweep_param": args.sweep_param,
        "include_minmax": args.include_minmax,
        "num_games": args.num_games,
        "seed_start": args.seed_start,
        "seeds": seeds,
        "base_mcts_config": {
            "num_simulations": base_config.num_simulations,
            "c_puct": base_config.c_puct,
            "max_placements": base_config.max_placements,
            "overhang_penalty_weight": base_config.overhang_penalty_weight,
            "nn_value_weight": base_config.nn_value_weight,
            "q_scale": base_config.q_scale,
            "mcts_seed": base_config.seed,
        },
        "sweep_values": [float(v) for v in args.sweep_values],
        "results": results,
        "summary": {
            "avg_attack_by_label": {row["label"]: row["avg_attack"] for row in results},
            "best_label": best_row["label"],
            "best_avg_attack": float(best_row["avg_attack"]),
        },
    }

    output_json.write_text(json.dumps(payload, indent=2))
    create_plot(results=results, sweep_param=args.sweep_param, output_path=output_plot)

    logger.info(
        "Finished MCTS config sweep",
        sweep_param=args.sweep_param,
        output_json=str(output_json),
        output_plot=str(output_plot),
        best_label=payload["summary"]["best_label"],
        best_avg_attack=payload["summary"]["best_avg_attack"],
    )


if __name__ == "__main__":
    main(parse(ScriptArgs))
