import time
from dataclasses import dataclass
from pathlib import Path

import structlog
from simple_parsing import parse, field

from tetris_core import MCTSConfig, evaluate_model, evaluate_model_without_nn
from rich import print as rprint

logger = structlog.get_logger()


@dataclass
class ScriptArgs:
    model_path: Path = Path(
        "/Users/axelhojmark/Desktop/v37/checkpoints/parallel.onnx"
    )  # ONNX model
    use_dummy_network: bool = True  # Run bootstrap MCTS without loading an ONNX model
    reuse_tree: bool = False
    num_games: int = 60  # Number of games per configuration
    simulations: int = 4000  # MCTS simulations per move
    max_placements: int = 50  # Maximum placements per game
    seed_start: int = 42  # Starting seed
    mcts_seed: int = 123  # Deterministic MCTS seed for reproducibility
    death_penalties: list[int] = field(default_factory=lambda: [10])
    overhang_penalty_weights: list[float] = field(
        # Best so far: 75
        default_factory=lambda: [75]
    )
    num_workers: int = 7
    add_noise: bool = True
    dirichlet_alpha: float = 0.02
    dirichlet_epsilon: float = 0.25


def run_config(
    args: ScriptArgs, death_penalty: int, overhang_penalty_weight: int
) -> dict:
    config = MCTSConfig()
    config.num_simulations = args.simulations
    config.max_placements = args.max_placements
    config.seed = args.mcts_seed
    config.death_penalty = death_penalty
    config.overhang_penalty_weight = overhang_penalty_weight
    config.reuse_tree = args.reuse_tree
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
        )

    elapsed = time.perf_counter() - start

    games_per_sec = args.num_games / elapsed if elapsed > 0 else 0

    return {
        "elapsed_sec": elapsed,
        "games_per_sec": games_per_sec,
        "avg_attack": result.avg_attack,
        "num_games": result.num_games,
        "avg_moves": result.avg_moves,
        "max_attack": result.max_attack,
    }


def main(args: ScriptArgs) -> None:
    print("death_penalty | overhang_penalty_weight | avg_attack")
    print("------------- | ----------------------- | ----------")
    for death_penalty in args.death_penalties:
        for overhang_penalty_weight in args.overhang_penalty_weights:
            result = run_config(args, death_penalty, overhang_penalty_weight)
            avg_attack = result["avg_attack"]
            rprint(
                f"        {death_penalty:.2f} |                    {overhang_penalty_weight:.2f} |       {avg_attack:.2f}"
            )


if __name__ == "__main__":
    args = parse(ScriptArgs)
    main(args)
