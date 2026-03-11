from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import structlog
from simple_parsing import parse

from tetris_core.tetris_core import (
    MCTSConfig,
    evaluate_model,
    evaluate_model_without_nn,
)
from tetris_bot.constants import BENCHMARKS_DIR, PARALLEL_ONNX_FILENAME
from tetris_bot.ml.config import SelfPlayConfig

logger = structlog.get_logger()
_DEFAULT_SELF_PLAY = SelfPlayConfig()


def default_worker_candidates() -> list[int]:
    cpu_count = max(2, os.cpu_count() or 1)
    presets = [4, 8, 12, 16, 24, 32, 40, 48, 56, 64]
    candidates = [value for value in presets if 1 < value <= cpu_count]
    if cpu_count not in candidates:
        candidates.append(cpu_count)
    return sorted(set(candidates))


@dataclass
class ScriptArgs:
    """Benchmark game-generation throughput across worker counts."""

    model_path: Path = BENCHMARKS_DIR / "models" / PARALLEL_ONNX_FILENAME
    use_dummy_network: bool = True
    worker_candidates: list[int] = field(default_factory=default_worker_candidates)
    num_games: int = 40
    num_repeats: int = 3
    simulations: int = 2000
    max_placements: int = 50
    seed_start: int = 42
    mcts_seed: int = 123
    c_puct: float = _DEFAULT_SELF_PLAY.c_puct
    reuse_tree: bool = True
    add_noise: bool = False
    q_scale: float | None = None
    death_penalty: float = 5.0
    overhang_penalty_weight: float = 5.0
    dirichlet_alpha: float = 0.02
    dirichlet_epsilon: float = 0.25
    output_json: Path = BENCHMARKS_DIR / "worker_sweep.json"


@dataclass
class RunResult:
    num_workers: int
    repeat_idx: int
    elapsed_sec: float
    games_per_sec: float
    moves_per_sec: float
    avg_attack: float
    avg_moves: float
    max_attack: int


@dataclass
class WorkerSummary:
    num_workers: int
    repeats: int
    games_per_sec_median: float
    games_per_sec_mean: float
    games_per_sec_min: float
    games_per_sec_max: float
    moves_per_sec_median: float
    elapsed_sec_median: float


def validate_args(args: ScriptArgs) -> list[int]:
    if args.num_games <= 0:
        raise ValueError(f"num_games must be > 0 (got {args.num_games})")
    if args.num_repeats <= 0:
        raise ValueError(f"num_repeats must be > 0 (got {args.num_repeats})")
    if args.simulations <= 0:
        raise ValueError(f"simulations must be > 0 (got {args.simulations})")
    if args.max_placements <= 0:
        raise ValueError(f"max_placements must be > 0 (got {args.max_placements})")
    if not args.worker_candidates:
        raise ValueError("worker_candidates cannot be empty")

    unique_workers = sorted(set(args.worker_candidates))
    if any(value <= 1 for value in unique_workers):
        raise ValueError(
            f"all worker_candidates must be > 1 for parallel evaluation: {unique_workers}"
        )

    if not args.use_dummy_network and not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    return unique_workers


def build_mcts_config(args: ScriptArgs) -> MCTSConfig:
    config = MCTSConfig()
    config.num_simulations = args.simulations
    config.max_placements = args.max_placements
    config.seed = args.mcts_seed
    config.c_puct = args.c_puct
    config.reuse_tree = args.reuse_tree
    config.q_scale = args.q_scale
    config.death_penalty = args.death_penalty
    config.overhang_penalty_weight = args.overhang_penalty_weight
    config.dirichlet_alpha = args.dirichlet_alpha
    config.dirichlet_epsilon = args.dirichlet_epsilon
    return config


def evaluate_once(
    *,
    args: ScriptArgs,
    num_workers: int,
    repeat_idx: int,
    config: MCTSConfig,
) -> RunResult:
    seed_offset = repeat_idx * args.num_games
    seeds = list(
        range(
            args.seed_start + seed_offset,
            args.seed_start + seed_offset + args.num_games,
        )
    )

    start = time.perf_counter()
    if args.use_dummy_network:
        result = evaluate_model_without_nn(
            seeds=seeds,
            config=config,
            max_placements=args.max_placements,
            num_workers=num_workers,
            add_noise=args.add_noise,
        )
    else:
        result = evaluate_model(
            model_path=str(args.model_path),
            seeds=seeds,
            config=config,
            max_placements=args.max_placements,
            num_workers=num_workers,
            add_noise=args.add_noise,
        )
    elapsed = time.perf_counter() - start

    total_moves = float(result.avg_moves) * float(result.num_games)
    games_per_sec = float(result.num_games) / elapsed if elapsed > 0 else 0.0
    moves_per_sec = total_moves / elapsed if elapsed > 0 else 0.0

    return RunResult(
        num_workers=num_workers,
        repeat_idx=repeat_idx,
        elapsed_sec=elapsed,
        games_per_sec=games_per_sec,
        moves_per_sec=moves_per_sec,
        avg_attack=float(result.avg_attack),
        avg_moves=float(result.avg_moves),
        max_attack=int(result.max_attack),
    )


def summarize(runs: list[RunResult], workers: list[int]) -> list[WorkerSummary]:
    summaries: list[WorkerSummary] = []
    for num_workers in workers:
        matching = [row for row in runs if row.num_workers == num_workers]
        gps = [row.games_per_sec for row in matching]
        mps = [row.moves_per_sec for row in matching]
        elapsed = [row.elapsed_sec for row in matching]

        summaries.append(
            WorkerSummary(
                num_workers=num_workers,
                repeats=len(matching),
                games_per_sec_median=statistics.median(gps),
                games_per_sec_mean=statistics.fmean(gps),
                games_per_sec_min=min(gps),
                games_per_sec_max=max(gps),
                moves_per_sec_median=statistics.median(mps),
                elapsed_sec_median=statistics.median(elapsed),
            )
        )
    return summaries


def print_summary_table(summaries: list[WorkerSummary]) -> None:
    print(
        f"{'Workers':>8}  {'Median G/s':>11}  {'Mean G/s':>9}  {'Min G/s':>9}  {'Max G/s':>9}  {'Median M/s':>11}  {'Median sec':>11}"
    )
    print("-" * 86)
    for row in summaries:
        print(
            f"{row.num_workers:>8}  "
            f"{row.games_per_sec_median:>11.3f}  "
            f"{row.games_per_sec_mean:>9.3f}  "
            f"{row.games_per_sec_min:>9.3f}  "
            f"{row.games_per_sec_max:>9.3f}  "
            f"{row.moves_per_sec_median:>11.1f}  "
            f"{row.elapsed_sec_median:>11.3f}"
        )


def main(args: ScriptArgs) -> None:
    workers = validate_args(args)
    config = build_mcts_config(args)

    logger.info(
        "Starting worker sweep",
        use_dummy_network=args.use_dummy_network,
        model_path=None if args.use_dummy_network else str(args.model_path),
        worker_candidates=workers,
        num_games=args.num_games,
        num_repeats=args.num_repeats,
        simulations=args.simulations,
    )

    runs: list[RunResult] = []
    for num_workers in workers:
        logger.info(
            "Evaluating worker setting",
            num_workers=num_workers,
            repeats=args.num_repeats,
        )
        for repeat_idx in range(args.num_repeats):
            run = evaluate_once(
                args=args,
                num_workers=num_workers,
                repeat_idx=repeat_idx,
                config=config,
            )
            runs.append(run)
            logger.info(
                "Completed benchmark run",
                num_workers=run.num_workers,
                repeat_idx=run.repeat_idx,
                games_per_sec=f"{run.games_per_sec:.3f}",
                elapsed_sec=f"{run.elapsed_sec:.3f}",
            )

    summaries = summarize(runs, workers)
    ranked = sorted(
        summaries,
        key=lambda row: (row.games_per_sec_median, row.games_per_sec_mean),
        reverse=True,
    )
    best = ranked[0]

    print("\n=== Worker Sweep Summary ===")
    print_summary_table(summaries)
    print(
        "\nBest setting: "
        f"workers={best.num_workers} "
        f"median_games_per_sec={best.games_per_sec_median:.3f} "
        f"mean_games_per_sec={best.games_per_sec_mean:.3f}"
    )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host_cpu_count": os.cpu_count(),
        "config": {
            "use_dummy_network": args.use_dummy_network,
            "model_path": None if args.use_dummy_network else str(args.model_path),
            "worker_candidates": workers,
            "num_games": args.num_games,
            "num_repeats": args.num_repeats,
            "simulations": args.simulations,
            "max_placements": args.max_placements,
            "seed_start": args.seed_start,
            "mcts_seed": args.mcts_seed,
            "c_puct": args.c_puct,
            "reuse_tree": args.reuse_tree,
            "add_noise": args.add_noise,
            "q_scale": args.q_scale,
            "death_penalty": args.death_penalty,
            "overhang_penalty_weight": args.overhang_penalty_weight,
            "dirichlet_alpha": args.dirichlet_alpha,
            "dirichlet_epsilon": args.dirichlet_epsilon,
        },
        "runs": [asdict(row) for row in runs],
        "summaries": [asdict(row) for row in summaries],
        "ranking": [asdict(row) for row in ranked],
        "best": asdict(best),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("Saved worker sweep results", output=str(args.output_json))


if __name__ == "__main__":
    main(parse(ScriptArgs))
