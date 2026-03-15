"""Empirical benchmark: current (fc_hidden=128) vs half (fc_hidden=64) network size.

Exports both model variants as split ONNX, then runs MCTS profiling back-to-back
with identical seeds and simulation counts to measure wall-clock throughput difference.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import structlog
import torch
from simple_parsing import parse

from tetris_core.tetris_core import MCTSConfig, evaluate_model
from tetris_bot.constants import BENCHMARKS_DIR
from tetris_bot.ml.config import NetworkConfig, SelfPlayConfig
from tetris_bot.ml.network import TetrisNet
from tetris_bot.ml.weights import export_split_models

logger = structlog.get_logger()
_DEFAULT_NETWORK = NetworkConfig()
_DEFAULT_SELF_PLAY = SelfPlayConfig()


@dataclass
class BenchmarkArgs:
    """Benchmark current vs half-size network inference speed."""

    num_games: int = 20  # Games per benchmark run
    simulations: int = 200  # MCTS simulations per move
    seed_start: int = 42  # Starting seed for deterministic games
    mcts_seed: int = 123  # Fixed MCTS RNG seed for reproducibility
    max_placements: int = _DEFAULT_SELF_PLAY.max_placements
    c_puct: float = _DEFAULT_SELF_PLAY.c_puct
    num_runs: int = 5  # Repeat each variant this many times
    output: Path = BENCHMARKS_DIR / "network_size_benchmark.jsonl"


def export_model(fc_hidden: int, label: str, output_dir: Path) -> Path:
    model_kwargs = _DEFAULT_NETWORK.to_model_kwargs()
    model_kwargs["fc_hidden"] = fc_hidden
    model = TetrisNet(**model_kwargs)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Created model",
        label=label,
        fc_hidden=fc_hidden,
        total_params=total_params,
    )

    onnx_path = output_dir / f"{label}.onnx"
    # Need dummy onnx file for the split loader path convention
    torch.save({}, onnx_path)
    export_split_models(model, onnx_path)
    logger.info("Exported split models", label=label, base_path=str(onnx_path))
    return onnx_path


def run_benchmark(
    model_path: Path,
    label: str,
    args: BenchmarkArgs,
) -> list[dict]:
    config = MCTSConfig()
    config.num_simulations = args.simulations
    config.c_puct = args.c_puct
    config.max_placements = args.max_placements
    config.seed = args.mcts_seed

    seeds = list(range(args.seed_start, args.seed_start + args.num_games))
    results = []

    for run_idx in range(args.num_runs):
        start = time.perf_counter()
        result = evaluate_model(
            model_path=str(model_path),
            seeds=seeds,
            config=config,
            max_placements=args.max_placements,
        )
        elapsed = time.perf_counter() - start

        total_moves = int(result.avg_moves * result.num_games)
        moves_per_sec = total_moves / elapsed if elapsed > 0 else 0
        ms_per_move = (elapsed / total_moves * 1000) if total_moves > 0 else 0

        run_data = {
            "label": label,
            "run": run_idx,
            "total_time_sec": elapsed,
            "num_games": result.num_games,
            "total_moves": total_moves,
            "avg_moves": result.avg_moves,
            "avg_attack": result.avg_attack,
            "moves_per_sec": moves_per_sec,
            "ms_per_move": ms_per_move,
        }
        results.append(run_data)
        logger.info(
            "Run complete",
            label=label,
            run=run_idx,
            elapsed=f"{elapsed:.3f}s",
            moves_per_sec=f"{moves_per_sec:.1f}",
            ms_per_move=f"{ms_per_move:.2f}",
        )

    return results


def main(args: BenchmarkArgs) -> None:
    output_dir = BENCHMARKS_DIR / "network_size_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export both models
    current_path = export_model(128, "current_128", output_dir)
    half_path = export_model(64, "half_64", output_dir)

    # Warm up: one throwaway run each to populate caches / JIT
    logger.info("Warming up current model...")
    warmup_config = MCTSConfig()
    warmup_config.num_simulations = args.simulations
    warmup_config.c_puct = args.c_puct
    warmup_config.max_placements = args.max_placements
    warmup_config.seed = args.mcts_seed
    evaluate_model(
        model_path=str(current_path),
        seeds=[0, 1, 2],
        config=warmup_config,
        max_placements=args.max_placements,
    )
    logger.info("Warming up half model...")
    evaluate_model(
        model_path=str(half_path),
        seeds=[0, 1, 2],
        config=warmup_config,
        max_placements=args.max_placements,
    )

    # Interleave runs to reduce systematic bias from thermal throttling etc.
    all_results = []
    for run_idx in range(args.num_runs):
        logger.info("Starting interleaved run pair", run=run_idx)

        # Current model
        current_results = run_benchmark(
            current_path,
            "current_128",
            BenchmarkArgs(
                num_games=args.num_games,
                simulations=args.simulations,
                seed_start=args.seed_start,
                mcts_seed=args.mcts_seed,
                max_placements=args.max_placements,
                c_puct=args.c_puct,
                num_runs=1,
                output=args.output,
            ),
        )
        all_results.extend(current_results)

        # Half model
        half_results = run_benchmark(
            half_path,
            "half_64",
            BenchmarkArgs(
                num_games=args.num_games,
                simulations=args.simulations,
                seed_start=args.seed_start,
                mcts_seed=args.mcts_seed,
                max_placements=args.max_placements,
                c_puct=args.c_puct,
                num_runs=1,
                output=args.output,
            ),
        )
        all_results.extend(half_results)

    # Save all results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    # Summary
    current_runs = [r for r in all_results if r["label"] == "current_128"]
    half_runs = [r for r in all_results if r["label"] == "half_64"]

    def summarize(runs: list[dict]) -> tuple[float, float]:
        speeds = [r["moves_per_sec"] for r in runs]
        avg = sum(speeds) / len(speeds)
        std = (sum((s - avg) ** 2 for s in speeds) / len(speeds)) ** 0.5
        return avg, std

    curr_avg, curr_std = summarize(current_runs)
    half_avg, half_std = summarize(half_runs)
    speedup = half_avg / curr_avg if curr_avg > 0 else 0

    print("\n" + "=" * 60)
    print("NETWORK SIZE BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Simulations per move:  {args.simulations}")
    print(f"Games per run:         {args.num_games}")
    print(f"Runs per variant:      {args.num_runs}")
    print(f"MCTS seed:             {args.mcts_seed}")
    print()
    print(f"current (fc_hidden=128): {curr_avg:.1f} +/- {curr_std:.1f} moves/sec")
    print(f"half    (fc_hidden=64):  {half_avg:.1f} +/- {half_std:.1f} moves/sec")
    print()
    print(f"Speedup: {speedup:.3f}x")
    print("=" * 60)


if __name__ == "__main__":
    args = parse(BenchmarkArgs)
    main(args)
