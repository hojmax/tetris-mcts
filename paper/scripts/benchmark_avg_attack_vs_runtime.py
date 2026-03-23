from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from simple_parsing import parse

from tetris_core.tetris_core import MCTSConfig, evaluate_model
from tetris_bot.constants import BENCHMARKS_DIR, PROJECT_ROOT
from tetris_bot.ml.artifacts import assert_rust_inference_artifacts
from tetris_bot.scripts.inspection.optimize_machine import (
    machine_profile,
    machine_type_fingerprint,
)
from tetris_bot.scripts.utils.eval_utils import compute_attack_stats
from tetris_bot.scripts.utils.run_search_config import (
    default_checkpoint_path,
    default_model_path,
    load_effective_self_play_config,
)

logger = structlog.get_logger()

_DEFAULT_SIMULATION_BUDGETS = [1, 100, 500, 1_000, 2_000, 4_000, 8_000]
_DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "paper" / "results" / "avg_attack_vs_runtime"
_OPTIMIZE_CACHE_DIR = BENCHMARKS_DIR / "profiles" / "optimize_cache"
_OPTIMIZED_WORKERS_ENV_VAR = "TETRIS_OPT_NUM_WORKERS"


@dataclass
class ScriptArgs:
    run_dir: Path
    label: str | None = None
    entry_name: str | None = None
    model_path: Path | None = None
    checkpoint_path: Path | None = None
    simulations: list[int] = field(
        default_factory=lambda: list(_DEFAULT_SIMULATION_BUDGETS)
    )
    num_games: int = 20
    seed_start: int = 42
    num_workers: int = 0
    mcts_seed: int = 12_345
    output_root: Path = _DEFAULT_RESULTS_ROOT
    color: str | None = None
    sort_order: int = 0


def validate_args(args: ScriptArgs) -> None:
    if not args.run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {args.run_dir}")
    if not args.run_dir.is_dir():
        raise NotADirectoryError(f"run_dir must be a directory: {args.run_dir}")
    if args.num_games <= 0:
        raise ValueError(f"num_games must be > 0 (got {args.num_games})")
    if args.seed_start < 0:
        raise ValueError(f"seed_start must be >= 0 (got {args.seed_start})")
    if args.num_workers < 0:
        raise ValueError(f"num_workers must be >= 0 (got {args.num_workers})")
    if args.mcts_seed < 0:
        raise ValueError(f"mcts_seed must be >= 0 (got {args.mcts_seed})")
    if not args.simulations:
        raise ValueError("simulations cannot be empty")

    seen: set[int] = set()
    for simulations in args.simulations:
        if simulations <= 0:
            raise ValueError(f"simulation budgets must be > 0 (got {simulations})")
        if simulations in seen:
            raise ValueError(f"duplicate simulation budget: {simulations}")
        seen.add(simulations)


def parse_positive_int(raw_value: str, *, label: str) -> int:
    try:
        parsed = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{label} must be an integer (got {raw_value!r})") from error
    if parsed <= 0:
        raise ValueError(f"{label} must be > 0 (got {parsed})")
    return parsed


def optimized_worker_env_cache_path() -> Path:
    fingerprint = machine_type_fingerprint(machine_profile())
    return _OPTIMIZE_CACHE_DIR / f"{fingerprint}.env"


def resolve_num_workers(
    requested_workers: int,
    *,
    default_workers: int,
) -> tuple[int, str, str | None]:
    if requested_workers > 0:
        return requested_workers, "cli", None

    raw_env_value = os.getenv(_OPTIMIZED_WORKERS_ENV_VAR)
    if raw_env_value is not None and raw_env_value.strip() != "":
        return (
            parse_positive_int(raw_env_value.strip(), label=_OPTIMIZED_WORKERS_ENV_VAR),
            "environment",
            None,
        )

    cache_path = optimized_worker_env_cache_path()
    if cache_path.exists():
        for raw_line in cache_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key == _OPTIMIZED_WORKERS_ENV_VAR:
                return (
                    parse_positive_int(
                        value.strip(),
                        label=f"{cache_path}:{_OPTIMIZED_WORKERS_ENV_VAR}",
                    ),
                    "optimize_cache",
                    str(cache_path),
                )

    return max(2, default_workers), "config", None


def resolve_model_path(run_dir: Path, model_path_override: Path | None) -> Path:
    model_path = (
        model_path_override
        if model_path_override is not None
        else default_model_path(run_dir)
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return model_path.resolve()


def resolve_checkpoint_path(
    run_dir: Path,
    checkpoint_path_override: Path | None,
) -> Path | None:
    checkpoint_path = (
        checkpoint_path_override
        if checkpoint_path_override is not None
        else default_checkpoint_path(run_dir)
    )
    if not checkpoint_path.exists():
        logger.info(
            "Checkpoint not found; using static run config only",
            checkpoint_path=str(checkpoint_path),
        )
        return None
    return checkpoint_path.resolve()


def default_label(run_dir: Path, model_path: Path) -> str:
    default_model = default_model_path(run_dir)
    if model_path == default_model:
        return run_dir.name
    return f"{run_dir.name} ({model_path.stem})"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._").lower()
    if slug == "":
        raise ValueError(f"Could not derive a valid file slug from {value!r}")
    return slug


def resolve_entry_name(args: ScriptArgs, run_dir: Path, model_path: Path) -> str:
    if args.entry_name is not None:
        return slugify(args.entry_name)
    return slugify(f"{run_dir.name}-{model_path.stem}")


def build_eval_config(
    effective_self_play_config: dict[str, Any],
    *,
    simulations: int,
    mcts_seed: int,
) -> MCTSConfig:
    config = MCTSConfig()
    config.num_simulations = simulations
    config.c_puct = float(effective_self_play_config["c_puct"])
    config.temperature = 0.0
    config.dirichlet_alpha = float(
        effective_self_play_config.get("dirichlet_alpha", config.dirichlet_alpha)
    )
    config.dirichlet_epsilon = float(
        effective_self_play_config.get("dirichlet_epsilon", config.dirichlet_epsilon)
    )
    config.visit_sampling_epsilon = 0.0
    config.max_placements = int(effective_self_play_config["max_placements"])
    config.q_scale = (
        float(effective_self_play_config["q_scale"])
        if effective_self_play_config.get("use_tanh_q_normalization", True)
        and effective_self_play_config.get("q_scale") is not None
        else None
    )
    config.reuse_tree = bool(effective_self_play_config.get("reuse_tree", True))
    config.use_parent_value_for_unvisited_q = bool(
        effective_self_play_config.get("use_parent_value_for_unvisited_q", False)
    )
    config.nn_value_weight = float(
        effective_self_play_config.get("nn_value_weight", config.nn_value_weight)
    )
    config.death_penalty = float(
        effective_self_play_config.get("death_penalty", config.death_penalty)
    )
    config.overhang_penalty_weight = float(
        effective_self_play_config.get(
            "overhang_penalty_weight",
            config.overhang_penalty_weight,
        )
    )
    config.seed = mcts_seed
    return config


def evaluate_simulation_budget(
    *,
    model_path: Path,
    seeds: list[int],
    simulations: int,
    num_workers: int,
    effective_self_play_config: dict[str, Any],
    mcts_seed: int,
) -> dict[str, object]:
    config = build_eval_config(
        effective_self_play_config,
        simulations=simulations,
        mcts_seed=mcts_seed,
    )
    start_time = time.perf_counter()
    result = evaluate_model(
        model_path=str(model_path),
        seeds=seeds,
        config=config,
        max_placements=config.max_placements,
        num_workers=num_workers,
        add_noise=False,
    )
    total_time_sec = time.perf_counter() - start_time
    num_games = int(result.num_games)  # type: ignore[attr-defined]
    if num_games <= 0:
        raise RuntimeError(
            "Evaluation completed zero games; cannot benchmark runtime vs attack"
        )

    game_results = [
        {"seed": seeds[index], "attack": int(attack), "moves": int(moves)}
        for index, (attack, moves) in enumerate(result.game_results)  # type: ignore[attr-defined]
    ]
    total_moves = sum(int(row["moves"]) for row in game_results)
    attack_std, attack_sem = compute_attack_stats(seeds, result)
    avg_time_per_game_sec = total_time_sec / num_games
    avg_time_per_move_sec = total_time_sec / total_moves if total_moves > 0 else 0.0

    return {
        "simulations": simulations,
        "num_games": num_games,
        "avg_attack": float(result.avg_attack),  # type: ignore[attr-defined]
        "avg_lines": float(result.avg_lines),  # type: ignore[attr-defined]
        "avg_moves": float(result.avg_moves),  # type: ignore[attr-defined]
        "max_attack": int(result.max_attack),  # type: ignore[attr-defined]
        "max_lines": int(result.max_lines),  # type: ignore[attr-defined]
        "attack_per_piece": float(result.attack_per_piece),  # type: ignore[attr-defined]
        "lines_per_piece": float(result.lines_per_piece),  # type: ignore[attr-defined]
        "attack_std": attack_std,
        "attack_sem": attack_sem,
        "total_time_sec": total_time_sec,
        "total_moves": total_moves,
        "avg_runtime_ms": avg_time_per_game_sec * 1_000,
        "avg_time_per_move_ms": avg_time_per_move_sec * 1_000,
        "games_per_second": (num_games / total_time_sec if total_time_sec > 0 else 0.0),
        "moves_per_second": total_moves / total_time_sec if total_time_sec > 0 else 0.0,
        "game_results": game_results,
    }


def build_summary_payload(
    *,
    args: ScriptArgs,
    label: str,
    entry_name: str,
    run_dir: Path,
    model_path: Path,
    checkpoint_path: Path | None,
    runtime_backend: str,
    num_workers: int,
    num_workers_source: str,
    num_workers_cache_path: str | None,
    effective_self_play_config: dict[str, Any],
    points: list[dict[str, object]],
) -> dict[str, object]:
    representative_config = build_eval_config(
        effective_self_play_config,
        simulations=int(points[0]["simulations"]),
        mcts_seed=args.mcts_seed,
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entry_name": entry_name,
        "label": label,
        "run_dir": str(run_dir.resolve()),
        "model_path": str(model_path.resolve()),
        "checkpoint_path": (
            str(checkpoint_path.resolve()) if checkpoint_path is not None else None
        ),
        "runtime_backend": runtime_backend,
        "num_games": args.num_games,
        "seed_start": args.seed_start,
        "seeds": list(range(args.seed_start, args.seed_start + args.num_games)),
        "num_workers": num_workers,
        "num_workers_source": num_workers_source,
        "num_workers_cache_path": num_workers_cache_path,
        "mcts_seed": args.mcts_seed,
        "plot": {
            "color": args.color,
            "sort_order": args.sort_order,
        },
        "base_config": {
            "c_puct": representative_config.c_puct,
            "temperature": representative_config.temperature,
            "dirichlet_alpha": representative_config.dirichlet_alpha,
            "dirichlet_epsilon": representative_config.dirichlet_epsilon,
            "visit_sampling_epsilon": representative_config.visit_sampling_epsilon,
            "max_placements": representative_config.max_placements,
            "q_scale": representative_config.q_scale,
            "reuse_tree": representative_config.reuse_tree,
            "use_parent_value_for_unvisited_q": (
                representative_config.use_parent_value_for_unvisited_q
            ),
            "nn_value_weight": representative_config.nn_value_weight,
            "death_penalty": representative_config.death_penalty,
            "overhang_penalty_weight": representative_config.overhang_penalty_weight,
            "add_noise": False,
        },
        "points": points,
    }


def write_summary(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse(ScriptArgs)
    validate_args(args)

    run_dir = args.run_dir.resolve()
    model_path = resolve_model_path(run_dir, args.model_path)
    checkpoint_path = resolve_checkpoint_path(run_dir, args.checkpoint_path)
    assert_rust_inference_artifacts(model_path)

    label = args.label or default_label(run_dir, model_path)
    entry_name = resolve_entry_name(args, run_dir, model_path)
    output_dir = args.output_root.resolve() / entry_name
    summary_path = output_dir / "summary.json"

    effective_self_play_config = load_effective_self_play_config(
        run_dir, checkpoint_path
    )
    default_workers = int(effective_self_play_config.get("num_workers", 2))
    num_workers, num_workers_source, num_workers_cache_path = resolve_num_workers(
        args.num_workers,
        default_workers=default_workers,
    )

    seeds = list(range(args.seed_start, args.seed_start + args.num_games))
    runtime_backend = os.getenv("TETRIS_NN_BACKEND", "tract(default)")
    logger.info(
        "Benchmarking runtime vs attack",
        label=label,
        entry_name=entry_name,
        run_dir=str(run_dir),
        model_path=str(model_path),
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        simulations=args.simulations,
        num_games=args.num_games,
        num_workers=num_workers,
        num_workers_source=num_workers_source,
        runtime_backend=runtime_backend,
    )

    points: list[dict[str, object]] = []
    for simulations in args.simulations:
        logger.info(
            "Evaluating simulation budget",
            entry_name=entry_name,
            simulations=simulations,
        )
        point = evaluate_simulation_budget(
            model_path=model_path,
            seeds=seeds,
            simulations=simulations,
            num_workers=num_workers,
            effective_self_play_config=effective_self_play_config,
            mcts_seed=args.mcts_seed,
        )
        points.append(point)
        logger.info(
            "Completed simulation budget",
            entry_name=entry_name,
            simulations=simulations,
            avg_attack=point["avg_attack"],
            avg_runtime_ms=point["avg_runtime_ms"],
        )

    summary_payload = build_summary_payload(
        args=args,
        label=label,
        entry_name=entry_name,
        run_dir=run_dir,
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        runtime_backend=runtime_backend,
        num_workers=num_workers,
        num_workers_source=num_workers_source,
        num_workers_cache_path=num_workers_cache_path,
        effective_self_play_config=effective_self_play_config,
        points=points,
    )
    write_summary(summary_path, summary_payload)
    logger.info("Wrote runtime vs attack benchmark summary", path=str(summary_path))


if __name__ == "__main__":
    main()
