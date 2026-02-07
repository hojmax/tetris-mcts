"""
Evaluate a trained model and save game replays to JSONL format.

Uses the Rust evaluate_model function for fast evaluation.
"""

from dataclasses import dataclass
from pathlib import Path

import structlog
from simple_parsing import parse

from tetris_core import MCTSConfig, evaluate_model
from tetris_mcts.config import OUTPUTS_DIR, TrainingConfig

logger = structlog.get_logger()

DEFAULT_TRAINING_CONFIG = TrainingConfig()


@dataclass
class EvalArgs:
    model_path: Path  # Path to ONNX model
    output_path: Path = OUTPUTS_DIR / "replays" / "replays.jsonl"
    num_games: int = 10  # Number of games to play
    max_moves: int = DEFAULT_TRAINING_CONFIG.max_moves  # Maximum moves per game
    simulations: int = 100  # MCTS simulations per move
    mcts_seed: int = (
        DEFAULT_TRAINING_CONFIG.eval_mcts_seed
    )  # MCTS RNG seed for deterministic evaluation


def main(args: EvalArgs) -> None:
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting evaluation",
        model=str(args.model_path),
        output=str(args.output_path),
        num_games=args.num_games,
        simulations=args.simulations,
        mcts_seed=args.mcts_seed,
    )

    # Create MCTS config
    config = MCTSConfig()
    config.num_simulations = args.simulations
    config.max_moves = args.max_moves
    config.seed = args.mcts_seed

    # Generate seeds (sequential for reproducibility)
    seeds = list(range(args.num_games))

    # Run evaluation and save replays
    result = evaluate_model(
        model_path=str(args.model_path),
        seeds=seeds,
        config=config,
        max_moves=args.max_moves,
        output_path=str(args.output_path),
    )

    logger.info(
        "Evaluation complete",
        num_games=result.num_games,
        avg_attack=round(result.avg_attack, 2),
        avg_lines=round(result.avg_lines, 2),
        max_attack=result.max_attack,
        max_lines=result.max_lines,
        avg_moves=round(result.avg_moves, 1),
        attack_per_piece=round(result.attack_per_piece, 3),
        lines_per_piece=round(result.lines_per_piece, 3),
    )


if __name__ == "__main__":
    args = parse(EvalArgs)
    main(args)
