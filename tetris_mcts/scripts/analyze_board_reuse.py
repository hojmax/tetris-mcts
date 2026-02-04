"""
Analyze board state reuse during MCTS to estimate CNN caching benefit.

Instruments MCTS search to track:
1. How many unique board states are encountered
2. How many times each board is seen
3. Potential CNN cache hit rate

This helps determine if board-level CNN caching would provide speedup.
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import numpy as np
from simple_parsing import parse

from tetris_mcts.config import PROJECT_ROOT
import tetris_core

logger = structlog.get_logger()


@dataclass
class AnalysisArgs:
    """Arguments for board reuse analysis."""

    model_path: Path = PROJECT_ROOT / "parallel.onnx"
    num_games: int = 5
    simulations: int = 400
    seed: int = 42


def board_to_key(board: np.ndarray) -> bytes:
    """Convert board array to hashable key."""
    return board.tobytes()


def analyze_board_reuse(args: AnalysisArgs) -> None:
    """Run MCTS games and track board state reuse."""

    # Unfortunately, we can't easily instrument the Rust MCTS code from Python
    # to track board hashes during tree search. We'd need to modify the Rust code.
    #
    # Instead, let's estimate by:
    # 1. Running games with MCTS
    # 2. Tracking unique boards from training examples
    # 3. This gives a lower bound (actual MCTS tree has more node expansions)

    logger.info(
        "Starting board reuse analysis",
        num_games=args.num_games,
        simulations=args.simulations,
    )

    env = tetris_core.TetrisEnv(width=10, height=20)
    agent = tetris_core.MCTSAgent(str(args.model_path), seed=args.seed)

    config = tetris_core.MCTSConfig(
        num_simulations=args.simulations,
        c_puct=1.5,
        temperature=1.0,
        dirichlet_alpha=0.15,
        dirichlet_epsilon=0.25,
    )

    # Track board occurrences
    board_counts: dict[bytes, int] = defaultdict(int)
    total_nn_calls = 0

    for game_idx in range(args.num_games):
        env.reset()
        move_count = 0

        logger.info("Starting game", game_idx=game_idx)

        while not env.game_over:
            # Get MCTS policy
            policy = agent.get_mcts_policy(env, config)

            # Track this board state
            board_key = board_to_key(env.get_board())
            board_counts[board_key] += 1

            # Estimate NN calls: one per simulation (this is a lower bound)
            total_nn_calls += args.simulations

            # Sample action and step
            action_mask = agent.get_action_mask(env)
            masked_policy = policy * action_mask
            masked_policy = masked_policy / masked_policy.sum()

            action = np.random.choice(len(policy), p=masked_policy)
            env.step(action)

            move_count += 1
            if move_count > 100:
                break

        logger.info(
            "Game finished",
            game_idx=game_idx,
            moves=move_count,
            attack=env.get_attack(),
        )

    # Analyze results
    unique_boards = len(board_counts)
    total_board_accesses = sum(board_counts.values())
    duplicate_accesses = total_board_accesses - unique_boards
    hit_rate = duplicate_accesses / total_board_accesses if total_board_accesses > 0 else 0

    logger.info("=== Board Reuse Analysis ===")
    logger.info(
        "Board statistics",
        unique_boards=unique_boards,
        total_accesses=total_board_accesses,
        duplicate_accesses=duplicate_accesses,
        hit_rate_pct=hit_rate * 100,
    )

    # Count distribution
    reuse_counts = list(board_counts.values())
    boards_seen_once = sum(1 for c in reuse_counts if c == 1)
    boards_seen_multiple = unique_boards - boards_seen_once
    max_reuse = max(reuse_counts) if reuse_counts else 0

    logger.info(
        "Reuse distribution",
        boards_seen_once=boards_seen_once,
        boards_seen_multiple=boards_seen_multiple,
        max_reuse=max_reuse,
    )

    logger.info(
        "Important note",
        message="This is a LOWER BOUND estimate. Actual MCTS tree has many more node "
        "expansions (one per simulation per move). To get accurate data, we'd need to "
        "instrument the Rust MCTS code to track board hashes during tree search.",
    )

    # Estimate best-case speedup
    if hit_rate > 0:
        # Assume CNN is 50% of forward pass time
        cnn_fraction = 0.5
        theoretical_speedup = 1 / (1 - hit_rate * cnn_fraction)
        logger.info(
            "Theoretical speedup (if CNN is 50% of inference time)",
            max_speedup=f"{theoretical_speedup:.2f}x",
        )
    else:
        logger.info("No board reuse detected - caching would not help")


def main() -> None:
    args = parse(AnalysisArgs)

    if not args.model_path.exists():
        logger.error("Model not found", path=args.model_path)
        logger.info(
            "Export a model first",
            hint="python tetris_mcts/scripts/train.py --training.total-steps 100",
        )
        return

    analyze_board_reuse(args)


if __name__ == "__main__":
    main()
