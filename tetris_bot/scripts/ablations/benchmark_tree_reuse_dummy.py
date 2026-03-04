"""Benchmark dummy-network (no NN) MCTS with and without tree reuse.

Runs evaluate_model_without_nn for each combination of
(num_simulations, reuse_tree) on fixed seeds and reports avg attack,
std, and attack/piece.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import structlog
from simple_parsing import parse

from tetris_core.tetris_core import MCTSConfig, evaluate_model_without_nn

logger = structlog.get_logger()


@dataclass
class ScriptArgs:
    """Benchmark tree reuse vs no-reuse for dummy-network MCTS."""

    num_simulations_list: list[int] = field(
        default_factory=lambda: [1000, 2000, 4000]
    )  # Simulation counts to benchmark
    num_seeds: int = 20  # Number of fixed seeds (0..N)
    seed_start: int = 0  # First seed
    max_placements: int = 50  # Max placements per game (matches training default)
    output_json: Path = Path(
        "benchmarks/tree_reuse_dummy.jsonl"
    )  # Output path (relative to cwd)


def build_config(num_simulations: int, reuse_tree: bool) -> MCTSConfig:
    config = MCTSConfig()
    config.num_simulations = num_simulations
    config.reuse_tree = reuse_tree
    # Match training bootstrap settings (game_generator.rs build_rollout_config):
    # bootstrap mode forces q_scale=None (min-max normalization, not tanh)
    config.q_scale = None
    config.dirichlet_alpha = 0.02  # training default (not MCTSConfig default 0.15)
    # death_penalty: negative signal for topout, discoverable with 1-2 levels of lookahead
    config.death_penalty = 5.0
    # overhang_penalty_weight: penalizes holes covered by pieces during search,
    # teaches flat stacking without needing deep line-clear lookahead
    config.overhang_penalty_weight = 5.0
    return config


def run_condition(
    num_simulations: int,
    reuse_tree: bool,
    seeds: list[int],
    max_placements: int,
) -> dict:
    config = build_config(num_simulations, reuse_tree)
    logger.info(
        "Running condition",
        num_simulations=num_simulations,
        reuse_tree=reuse_tree,
        num_seeds=len(seeds),
    )
    result = evaluate_model_without_nn(
        seeds=seeds,
        config=config,
        max_placements=max_placements,
        add_noise=True,
    )
    attacks = [int(attack) for attack, _moves in result.game_results]
    moves_list = [int(moves) for _attack, moves in result.game_results]
    return {
        "num_simulations": num_simulations,
        "reuse_tree": reuse_tree,
        "avg_attack": result.avg_attack,
        "std_attack": statistics.pstdev(attacks) if len(attacks) > 1 else 0.0,
        "max_attack": result.max_attack,
        "avg_moves": result.avg_moves,
        "attack_per_piece": result.attack_per_piece,
        "per_seed": [
            {"seed": seeds[i], "attack": attacks[i], "moves": moves_list[i]}
            for i in range(len(seeds))
        ],
    }


def print_table(rows: list[dict]) -> None:
    header = f"{'Sims':>6}  {'Reuse':>5}  {'AvgAtk':>8}  {'Std':>7}  {'Max':>5}  {'Atk/Pc':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['num_simulations']:>6}  "
            f"{'yes' if r['reuse_tree'] else 'no':>5}  "
            f"{r['avg_attack']:>8.2f}  "
            f"{r['std_attack']:>7.2f}  "
            f"{r['max_attack']:>5}  "
            f"{r['attack_per_piece']:>7.4f}"
        )


def main(args: ScriptArgs) -> None:
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    logger.info(
        "Starting benchmark",
        seeds=seeds,
        num_simulations_list=args.num_simulations_list,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for num_simulations in args.num_simulations_list:
        for reuse_tree in [False, True]:
            row = run_condition(num_simulations, reuse_tree, seeds, args.max_placements)
            rows.append(row)
            logger.info(
                "Condition done",
                num_simulations=num_simulations,
                reuse_tree=reuse_tree,
                avg_attack=f"{row['avg_attack']:.2f}",
                std=f"{row['std_attack']:.2f}",
                attack_per_piece=f"{row['attack_per_piece']:.4f}",
            )

    with args.output_json.open("w") as f:
        meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_seeds": args.num_seeds,
            "seed_start": args.seed_start,
            "max_placements": args.max_placements,
        }
        f.write(json.dumps(meta) + "\n")
        for row in rows:
            f.write(json.dumps(row) + "\n")

    logger.info("Results saved", path=str(args.output_json))
    print("\n=== Dummy-network MCTS: tree reuse benchmark ===\n")
    print_table(rows)


if __name__ == "__main__":
    args = parse(ScriptArgs)
    main(args)
