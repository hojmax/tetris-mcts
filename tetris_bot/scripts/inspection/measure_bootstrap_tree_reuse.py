"""Measure tree reuse metrics for bootstrap (no-NN) MCTS games.

Default settings match the requested quick check:
- 4000 simulations
- 10 games
- 50 max placements

The script reports per-game tree reuse stats and traversal outcome fractions
(expansion/terminal/horizon) across simulations.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import structlog
from simple_parsing import parse

from tetris_core import MCTSAgent, MCTSConfig

logger = structlog.get_logger()


@dataclass
class ScriptArgs:
    """Run bootstrap tree-reuse measurements over multiple games."""

    num_games: int = 10
    simulations: int = 4000
    max_placements: int = 50
    num_workers: int = 0  # 0 => auto (all available CPU cores)
    add_noise: bool = True
    output_json: Path = Path("benchmarks/bootstrap_tree_reuse.json")


@dataclass
class GameMetric:
    game_idx: int
    total_attack: int
    num_moves: int
    tree_reuse_hits: int
    tree_reuse_misses: int
    tree_reuse_rate: float
    tree_reuse_carry_fraction: float
    traversal_total: int
    traversal_expansions: int
    traversal_terminal_ends: int
    traversal_horizon_ends: int
    traversal_expansion_fraction: float
    traversal_terminal_fraction: float
    traversal_horizon_fraction: float


def build_bootstrap_config(args: ScriptArgs) -> MCTSConfig:
    """Create no-NN bootstrap config with the same key defaults as training."""
    return build_bootstrap_config_from_values(
        simulations=args.simulations,
        max_placements=args.max_placements,
    )


def build_bootstrap_config_from_values(
    simulations: int,
    max_placements: int,
) -> MCTSConfig:
    """Create no-NN bootstrap config with training-aligned defaults."""
    config = MCTSConfig()
    config.num_simulations = simulations
    config.max_placements = max_placements
    config.reuse_tree = True

    # Keep bootstrap behavior aligned with training no-NN rollout defaults.
    config.q_scale = None
    config.dirichlet_alpha = 0.02
    config.death_penalty = 5.0
    config.overhang_penalty_weight = 5.0

    return config


def run_games_chunk(
    start_game_idx: int,
    num_games: int,
    simulations: int,
    max_placements: int,
    add_noise: bool,
) -> list[GameMetric]:
    """Run a contiguous chunk of games and return per-game metrics."""
    config = build_bootstrap_config_from_values(
        simulations=simulations,
        max_placements=max_placements,
    )
    agent = MCTSAgent(config)

    metrics: list[GameMetric] = []
    for offset in range(num_games):
        game_idx = start_game_idx + offset
        result = agent.play_game(
            max_placements=max_placements,
            add_noise=add_noise,
        )
        if result is None:
            continue

        hits = int(result.tree_reuse_hits)
        misses = int(result.tree_reuse_misses)
        total = hits + misses

        metrics.append(
            GameMetric(
                game_idx=game_idx,
                total_attack=int(result.total_attack),
                num_moves=int(result.num_moves),
                tree_reuse_hits=hits,
                tree_reuse_misses=misses,
                tree_reuse_rate=(hits / total) if total > 0 else 0.0,
                tree_reuse_carry_fraction=float(result.tree_reuse_carry_fraction),
                traversal_total=int(result.traversal_total),
                traversal_expansions=int(result.traversal_expansions),
                traversal_terminal_ends=int(result.traversal_terminal_ends),
                traversal_horizon_ends=int(result.traversal_horizon_ends),
                traversal_expansion_fraction=float(result.traversal_expansion_fraction),
                traversal_terminal_fraction=float(result.traversal_terminal_fraction),
                traversal_horizon_fraction=float(result.traversal_horizon_fraction),
            )
        )

    return metrics


def summarize(metrics: list[GameMetric]) -> dict[str, float | int]:
    if not metrics:
        return {
            "tree_reuse_hits": 0,
            "tree_reuse_misses": 0,
            "tree_reuse_rate": 0.0,
            "tree_reuse_carry_fraction_weighted_by_hits": 0.0,
            "tree_reuse_carry_fraction_mean_per_game": 0.0,
            "tree_reuse_carry_fraction_min": 0.0,
            "tree_reuse_carry_fraction_max": 0.0,
            "traversal_total": 0,
            "traversal_expansions": 0,
            "traversal_terminal_ends": 0,
            "traversal_horizon_ends": 0,
            "traversal_expansion_fraction": 0.0,
            "traversal_terminal_fraction": 0.0,
            "traversal_horizon_fraction": 0.0,
            "avg_attack": 0.0,
            "avg_moves": 0.0,
        }

    total_hits = sum(row.tree_reuse_hits for row in metrics)
    total_misses = sum(row.tree_reuse_misses for row in metrics)
    total_reuse_events = total_hits + total_misses
    traversal_total = sum(row.traversal_total for row in metrics)
    traversal_expansions = sum(row.traversal_expansions for row in metrics)
    traversal_terminal_ends = sum(row.traversal_terminal_ends for row in metrics)
    traversal_horizon_ends = sum(row.traversal_horizon_ends for row in metrics)

    weighted_carry = (
        sum(row.tree_reuse_carry_fraction * row.tree_reuse_hits for row in metrics)
        / total_hits
        if total_hits > 0
        else 0.0
    )

    carries = [row.tree_reuse_carry_fraction for row in metrics]

    return {
        "tree_reuse_hits": total_hits,
        "tree_reuse_misses": total_misses,
        "tree_reuse_rate": (
            total_hits / total_reuse_events if total_reuse_events > 0 else 0.0
        ),
        "tree_reuse_carry_fraction_weighted_by_hits": weighted_carry,
        "tree_reuse_carry_fraction_mean_per_game": statistics.fmean(carries),
        "tree_reuse_carry_fraction_min": min(carries),
        "tree_reuse_carry_fraction_max": max(carries),
        "traversal_total": traversal_total,
        "traversal_expansions": traversal_expansions,
        "traversal_terminal_ends": traversal_terminal_ends,
        "traversal_horizon_ends": traversal_horizon_ends,
        "traversal_expansion_fraction": (
            traversal_expansions / traversal_total if traversal_total > 0 else 0.0
        ),
        "traversal_terminal_fraction": (
            traversal_terminal_ends / traversal_total if traversal_total > 0 else 0.0
        ),
        "traversal_horizon_fraction": (
            traversal_horizon_ends / traversal_total if traversal_total > 0 else 0.0
        ),
        "avg_attack": statistics.fmean(row.total_attack for row in metrics),
        "avg_moves": statistics.fmean(row.num_moves for row in metrics),
    }


def print_summary(metrics: list[GameMetric], aggregate: dict[str, float | int]) -> None:
    print("\n=== Bootstrap tree reuse summary ===")
    print(f"Games completed: {len(metrics)}")
    print(
        "Tree reuse: "
        f"hits={aggregate['tree_reuse_hits']} "
        f"misses={aggregate['tree_reuse_misses']} "
        f"rate={float(aggregate['tree_reuse_rate']):.4f}"
    )
    print(
        "Carry fraction: "
        f"weighted_by_hits={float(aggregate['tree_reuse_carry_fraction_weighted_by_hits']):.4f} "
        f"mean_per_game={float(aggregate['tree_reuse_carry_fraction_mean_per_game']):.4f} "
        f"min={float(aggregate['tree_reuse_carry_fraction_min']):.4f} "
        f"max={float(aggregate['tree_reuse_carry_fraction_max']):.4f}"
    )
    print(
        "Traversal outcomes: "
        f"expansion={float(aggregate['traversal_expansion_fraction']):.4f} "
        f"terminal={float(aggregate['traversal_terminal_fraction']):.4f} "
        f"horizon={float(aggregate['traversal_horizon_fraction']):.4f}"
    )
    print(
        "Game quality: "
        f"avg_attack={float(aggregate['avg_attack']):.2f} "
        f"avg_moves={float(aggregate['avg_moves']):.2f}"
    )

    if metrics:
        print("\nPer-game tree reuse carry fraction")
        print(
            f"{'Game':>4}  {'Atk':>5}  {'Moves':>5}  {'Hits':>5}  {'Miss':>5}  "
            f"{'Reuse':>6}  {'Carry':>6}  {'Expand':>6}  {'Term':>6}  {'Horiz':>6}"
        )
        for row in metrics:
            print(
                f"{row.game_idx:>4}  {row.total_attack:>5}  {row.num_moves:>5}  "
                f"{row.tree_reuse_hits:>5}  {row.tree_reuse_misses:>5}  "
                f"{row.tree_reuse_rate:>6.3f}  {row.tree_reuse_carry_fraction:>6.3f}  "
                f"{row.traversal_expansion_fraction:>6.3f}  "
                f"{row.traversal_terminal_fraction:>6.3f}  "
                f"{row.traversal_horizon_fraction:>6.3f}"
            )


def main(args: ScriptArgs) -> None:
    if args.num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if args.num_games <= 0:
        raise ValueError("num_games must be > 0")

    # Validate bootstrap config values early.
    _ = build_bootstrap_config(args)
    requested_workers = (
        args.num_workers if args.num_workers > 0 else (os.cpu_count() or 1)
    )
    worker_count = min(requested_workers, args.num_games)

    logger.info(
        "Running bootstrap tree reuse measurement",
        num_games=args.num_games,
        simulations=args.simulations,
        max_placements=args.max_placements,
        num_workers=worker_count,
        add_noise=args.add_noise,
    )

    metrics: list[GameMetric] = []
    if worker_count == 1:
        metrics = run_games_chunk(
            start_game_idx=0,
            num_games=args.num_games,
            simulations=args.simulations,
            max_placements=args.max_placements,
            add_noise=args.add_noise,
        )
    else:
        base_chunk = args.num_games // worker_count
        remainder = args.num_games % worker_count
        chunk_sizes = [
            base_chunk + (1 if worker_idx < remainder else 0)
            for worker_idx in range(worker_count)
        ]

        next_start_idx = 0
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = []
            for chunk_size in chunk_sizes:
                futures.append(
                    executor.submit(
                        run_games_chunk,
                        next_start_idx,
                        chunk_size,
                        args.simulations,
                        args.max_placements,
                        args.add_noise,
                    )
                )
                next_start_idx += chunk_size

            for future in futures:
                metrics.extend(future.result())

        metrics.sort(key=lambda row: row.game_idx)

    aggregate = summarize(metrics)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "num_games": args.num_games,
            "num_simulations": args.simulations,
            "max_placements": args.max_placements,
            "num_workers_requested": requested_workers,
            "num_workers_used": worker_count,
            "add_noise": args.add_noise,
            "reuse_tree": True,
            "q_scale": None,
            "dirichlet_alpha": 0.02,
            "death_penalty": 5.0,
            "overhang_penalty_weight": 5.0,
        },
        "num_games_completed": len(metrics),
        "aggregate": aggregate,
        "per_game": [asdict(row) for row in metrics],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    logger.info("Saved results", output=str(args.output_json))

    print_summary(metrics, aggregate)
    print(f"\nJSON output: {args.output_json}")


if __name__ == "__main__":
    main(parse(ScriptArgs))
