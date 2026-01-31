"""
Evaluate a trained model and save game replays to JSONL format.

Uses the Rust evaluate_and_save function for fast evaluation.
"""

from dataclasses import dataclass
from pathlib import Path

import structlog
from simple_parsing import parse

from tetris_core import MCTSConfig, evaluate_and_save

logger = structlog.get_logger()


@dataclass
class EvalArgs:
    model_path: Path  # Path to ONNX model
    output_path: Path = Path(__file__).parent.parent.parent / "outputs" / "replays" / "replays.jsonl"
    num_games: int = 10  # Number of games to play
    max_moves: int = 100  # Maximum moves per game
    simulations: int = 100  # MCTS simulations per move


def main(args: EvalArgs) -> None:
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting evaluation",
        model=str(args.model_path),
        output=str(args.output_path),
        num_games=args.num_games,
        simulations=args.simulations,
    )

    # Create MCTS config
    config = MCTSConfig()
    config.num_simulations = args.simulations

    # Generate seeds (sequential for reproducibility)
    seeds = list(range(args.num_games))

    # Run evaluation and save replays
    result = evaluate_and_save(
        model_path=str(args.model_path),
        output_path=str(args.output_path),
        seeds=seeds,
        config=config,
        max_moves=args.max_moves,
    )

    logger.info(
        "Evaluation complete",
        num_games=result.num_games,
        avg_attack=round(result.avg_attack, 2),
        max_attack=result.max_attack,
        avg_moves=round(result.avg_moves, 1),
        attack_per_piece=round(result.attack_per_piece, 3),
    )


if __name__ == "__main__":
    args = parse(EvalArgs)
    main(args)
