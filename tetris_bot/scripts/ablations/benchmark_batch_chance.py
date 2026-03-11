from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import structlog
from simple_parsing import parse

from tetris_core.tetris_core import (
    MCTSConfig,
    evaluate_model,
    evaluate_model_without_nn,
)
from tetris_bot.constants import BENCHMARKS_DIR, PARALLEL_ONNX_FILENAME, PROJECT_ROOT

logger = structlog.get_logger()


@dataclass
class ScriptArgs:
    """Benchmark batch chance node expansion vs lazy expansion."""

    model_path: Path = BENCHMARKS_DIR / "models" / PARALLEL_ONNX_FILENAME
    use_dummy_network: bool = False  # Run bootstrap MCTS without loading an ONNX model
    num_games: int = 5  # Number of games per configuration
    simulations: int = 1000  # MCTS simulations per move
    max_placements: int = 50  # Maximum placements per game
    seed_start: int = 42  # Starting seed
    mcts_seed: int = 123  # Deterministic MCTS seed for reproducibility
    warmup_games: int = 1  # Warmup games (discarded)


def run_config(args: ScriptArgs, batch_chance: bool, label: str) -> dict:
    config = MCTSConfig()
    config.num_simulations = args.simulations
    config.max_placements = args.max_placements
    config.seed = args.mcts_seed
    config.batch_chance_expansion = batch_chance

    seeds = list(range(args.seed_start, args.seed_start + args.num_games))

    start = time.perf_counter()

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

    elapsed = time.perf_counter() - start

    total_moves = int(result.avg_moves * result.num_games)
    total_sims = total_moves * args.simulations
    moves_per_sec = total_moves / elapsed if elapsed > 0 else 0
    sims_per_sec = total_sims / elapsed if elapsed > 0 else 0

    return {
        "label": label,
        "batch_chance_expansion": batch_chance,
        "elapsed_sec": elapsed,
        "num_games": result.num_games,
        "total_moves": total_moves,
        "total_sims": total_sims,
        "avg_attack": result.avg_attack,
        "avg_moves": result.avg_moves,
        "max_attack": result.max_attack,
        "moves_per_sec": moves_per_sec,
        "sims_per_sec": sims_per_sec,
    }


def main(args: ScriptArgs) -> None:
    if not args.use_dummy_network and not args.model_path.exists():
        logger.error("Model not found", path=str(args.model_path))
        return

    mode = (
        "bootstrap (no NN)"
        if args.use_dummy_network
        else f"NN ({args.model_path.name})"
    )
    logger.info(
        "Benchmark config",
        mode=mode,
        num_games=args.num_games,
        simulations=args.simulations,
        max_placements=args.max_placements,
        mcts_seed=args.mcts_seed,
    )

    # Warmup
    if args.warmup_games > 0:
        logger.info("Running warmup", warmup_games=args.warmup_games)
        warmup_args = ScriptArgs(
            model_path=args.model_path,
            use_dummy_network=args.use_dummy_network,
            num_games=args.warmup_games,
            simulations=args.simulations,
            max_placements=args.max_placements,
            seed_start=0,
            mcts_seed=args.mcts_seed,
            warmup_games=0,
        )
        run_config(warmup_args, False, "warmup")
        run_config(warmup_args, True, "warmup")

    # Baseline: lazy expansion
    logger.info("Running baseline (lazy expansion)")
    baseline = run_config(args, batch_chance=False, label="lazy")

    # Experiment: batch expansion
    logger.info("Running experiment (batch expansion)")
    experiment = run_config(args, batch_chance=True, label="batch")

    # Print comparison
    print("\n" + "=" * 70)
    print("BATCH CHANCE NODE EXPANSION BENCHMARK")
    print("=" * 70)
    print(f"Mode:            {mode}")
    print(f"Games:           {args.num_games}")
    print(f"Simulations:     {args.simulations}")
    print(f"Max placements:  {args.max_placements}")
    print(f"MCTS seed:       {args.mcts_seed}")
    print()

    for r in [baseline, experiment]:
        print(f"--- {r['label'].upper()} ---")
        print(f"  Time:          {r['elapsed_sec']:.3f}s")
        print(f"  Total moves:   {r['total_moves']}")
        print(f"  Avg attack:    {r['avg_attack']:.1f}")
        print(f"  Max attack:    {r['max_attack']}")
        print(f"  Moves/sec:     {r['moves_per_sec']:.1f}")
        print(f"  Sims/sec:      {r['sims_per_sec']:.0f}")
        print()

    speedup = (
        baseline["elapsed_sec"] / experiment["elapsed_sec"]
        if experiment["elapsed_sec"] > 0
        else 0
    )
    sims_speedup = (
        experiment["sims_per_sec"] / baseline["sims_per_sec"]
        if baseline["sims_per_sec"] > 0
        else 0
    )

    print("--- COMPARISON ---")
    print(f"  Wall-clock speedup:  {speedup:.3f}x")
    print(f"  Sims/sec speedup:    {sims_speedup:.3f}x")
    print(f"  Moves/sec (lazy):    {baseline['moves_per_sec']:.1f}")
    print(f"  Moves/sec (batch):   {experiment['moves_per_sec']:.1f}")

    # Verify game outcomes match (deterministic seed)
    if (
        baseline["avg_attack"] == experiment["avg_attack"]
        and baseline["total_moves"] == experiment["total_moves"]
    ):
        print("  Outcomes:            MATCH (deterministic)")
    else:
        print(
            f"  Outcomes:            DIFFER (lazy avg_attack={baseline['avg_attack']:.1f}, batch avg_attack={experiment['avg_attack']:.1f})"
        )

    print("=" * 70)

    output_path = PROJECT_ROOT / "benchmarks" / "batch_chance_results.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as f:
        for r in [baseline, experiment]:
            r["simulations"] = args.simulations
            r["max_placements"] = args.max_placements
            r["mcts_seed"] = args.mcts_seed
            r["mode"] = mode
            f.write(json.dumps(r) + "\n")

    logger.info("Results saved", output=str(output_path))


if __name__ == "__main__":
    args = parse(ScriptArgs)
    main(args)
