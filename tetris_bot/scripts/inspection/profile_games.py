from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import structlog
from simple_parsing import parse

from tetris_core import MCTSConfig, evaluate_model, evaluate_model_without_nn
from tetris_bot.constants import (
    BENCHMARKS_DIR,
    PARALLEL_ONNX_FILENAME,
)
from tetris_bot.ml.config import SelfPlayConfig

logger = structlog.get_logger()
_DEFAULT_SELF_PLAY = SelfPlayConfig()


@dataclass
class ProfileArgs:
    """Profile MCTS game generation performance with fixed seeds."""

    model_path: Path = BENCHMARKS_DIR / "models" / PARALLEL_ONNX_FILENAME
    use_dummy_network: bool = False  # Run bootstrap MCTS without loading an ONNX model
    num_games: int = 10  # Number of games to profile
    simulations: int = 100  # MCTS simulations per move
    seed_start: int = 42  # Starting seed for deterministic games
    c_puct: float = _DEFAULT_SELF_PLAY.c_puct  # PUCT exploration constant
    mcts_seed: int | None = None  # Optional deterministic MCTS RNG seed
    max_placements: int = _DEFAULT_SELF_PLAY.max_placements  # Maximum placements per game
    output: Path = BENCHMARKS_DIR / "profile_results.jsonl"  # Output JSONL file


def main(args: ProfileArgs) -> None:
    if not args.use_dummy_network and not args.model_path.exists():
        logger.error("Model not found", path=str(args.model_path))
        return

    evaluation_mode = (
        "dummy_no_network_uniform" if args.use_dummy_network else "onnx_network"
    )
    logger.info(
        "Starting performance profiling",
        evaluation_mode=evaluation_mode,
        model=str(args.model_path) if not args.use_dummy_network else None,
        num_games=args.num_games,
        simulations=args.simulations,
        seed_start=args.seed_start,
    )

    config = MCTSConfig()
    config.num_simulations = args.simulations
    config.c_puct = args.c_puct
    config.max_placements = args.max_placements
    config.seed = args.mcts_seed

    seeds = list(range(args.seed_start, args.seed_start + args.num_games))

    logger.info("Starting game evaluation", seeds=seeds)
    start_time = time.perf_counter()

    if args.use_dummy_network:
        result = evaluate_model_without_nn(
            seeds=seeds,
            config=config,
            max_placements=args.max_placements,
        )
    else:
        result = evaluate_model(
            model_path=str(args.model_path),
            seeds=seeds,
            config=config,
            max_placements=args.max_placements,
        )

    end_time = time.perf_counter()
    total_time = end_time - start_time

    logger.info(
        "Profiling complete",
        total_time_sec=f"{total_time:.3f}",
        games=result.num_games,
        avg_moves=f"{result.avg_moves:.1f}",
        avg_attack=f"{result.avg_attack:.1f}",
    )

    total_moves = int(result.avg_moves * result.num_games)
    total_attack = int(result.avg_attack * result.num_games)
    avg_time_per_game = total_time / result.num_games if result.num_games > 0 else 0.0
    avg_time_per_move = total_time / total_moves if total_moves > 0 else 0
    moves_per_second = total_moves / total_time if total_time > 0 else 0
    games_per_second = result.num_games / total_time if total_time > 0 else 0

    if result.num_games == 0:
        logger.warning(
            "No games completed during profiling",
            note="This usually means model loading/inference failed for all seeds.",
        )

    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Total time:          {total_time:.3f}s")
    print(f"Games completed:     {result.num_games}")
    print(f"Total moves:         {total_moves}")
    print(f"Total attack:        {total_attack}")
    print()
    print(f"Avg time/game:       {avg_time_per_game:.3f}s")
    print(f"Avg time/move:       {avg_time_per_move * 1000:.1f}ms")
    print(f"Avg moves/game:      {result.avg_moves:.1f}")
    print(f"Avg attack/game:     {result.avg_attack:.1f}")
    print(f"Max attack:          {result.max_attack}")
    print(f"Attack/piece:        {result.attack_per_piece:.3f}")
    print()
    print(f"Throughput:          {moves_per_second:.1f} moves/sec")
    print(f"                     {games_per_second:.2f} games/sec")
    print("=" * 60)

    # Save results to JSONL file
    timestamp = datetime.now().isoformat()

    profile_data = {
        "timestamp": timestamp,
        "model_path": None if args.use_dummy_network else str(args.model_path),
        "evaluation_mode": evaluation_mode,
        "config": {
            "num_games": args.num_games,
            "simulations": args.simulations,
            "seed_start": args.seed_start,
            "c_puct": args.c_puct,
            "max_placements": args.max_placements,
            "evaluation_mode": "deterministic_argmax_no_dirichlet_noise",
            "use_dummy_network": args.use_dummy_network,
        },
        "timing": {
            "total_time_sec": total_time,
            "avg_time_per_game_sec": avg_time_per_game,
            "avg_time_per_move_ms": avg_time_per_move * 1000,
            "moves_per_second": moves_per_second,
            "games_per_second": games_per_second,
        },
        "results": {
            "num_games": result.num_games,
            "total_moves": total_moves,
            "total_attack": total_attack,
            "avg_moves": result.avg_moves,
            "avg_attack": result.avg_attack,
            "max_attack": result.max_attack,
            "attack_per_piece": result.attack_per_piece,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as f:
        f.write(json.dumps(profile_data) + "\n")

    logger.info("Results saved", output=str(args.output))


if __name__ == "__main__":
    args = parse(ProfileArgs)
    main(args)
