"""Benchmark raw Rust split-model inference for two gated-fusion configs.

This compares two model configurations by exporting randomly initialized split
models and timing direct Rust `predict_with_valid_actions(...)` calls on the
same fixed corpus of reachable Tetris states.

It reports two modes:
- `cold_no_cache`: board-embedding cache disabled, so the full conv/backbone +
  board-projection + heads path runs every prediction.
- `hot_warm_cache`: board-embedding cache enabled and warmed on the full state
  corpus first, so the timed passes mostly measure the uncached heads path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time

import structlog
from simple_parsing import parse

from tetris_core.tetris_core import (
    MCTSAgent,
    MCTSConfig,
    TetrisEnv,
    debug_get_action_mask,
)
from tetris_bot.constants import BENCHMARKS_DIR
from tetris_bot.ml.config import (
    NetworkConfig,
    default_network_config,
    default_self_play_config,
)
from tetris_bot.ml.network import TetrisNet
from tetris_bot.ml.weights import export_split_models, split_model_paths

logger = structlog.get_logger()
_DEFAULT_SELF_PLAY = default_self_play_config()
_LARGE_CONFIG = default_network_config().model_copy(update={"fc_hidden": 128})
_SMALL_CONFIG = default_network_config().model_copy(
    update={
        "trunk_channels": 8,
        "num_conv_residual_blocks": 2,
        "reduction_channels": 16,
        "fc_hidden": 128,
    }
)


@dataclass(frozen=True)
class VariantSpec:
    label: str
    network: NetworkConfig


@dataclass(frozen=True)
class ExportedVariant:
    label: str
    network: NetworkConfig
    onnx_path: Path
    total_params: int
    split_bytes: int
    conv_bytes: int
    heads_bytes: int
    fc_bytes: int


@dataclass(frozen=True)
class BenchmarkModeResult:
    label: str
    mode: str
    total_predictions: int
    elapsed_sec: float
    predictions_per_sec: float
    ms_per_prediction: float
    cache_hits: int
    cache_misses: int
    cache_size: int
    warmup_cache_hits: int
    warmup_cache_misses: int
    warmup_cache_size: int


@dataclass
class ScriptArgs:
    """Compare two split-model configs through direct Rust NN inference."""

    num_seeds: int = 128
    states_per_seed: int = 12
    timed_repeats: int = 6
    max_placements: int = _DEFAULT_SELF_PLAY.max_placements
    seed_start: int = 42
    output_root: Path = BENCHMARKS_DIR / "split_inference_config_compare"


def variant_specs() -> list[VariantSpec]:
    return [
        VariantSpec(label="large_16x3x32", network=_LARGE_CONFIG),
        VariantSpec(label="small_8x2x16", network=_SMALL_CONFIG),
    ]


def count_parameters(model: TetrisNet) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def export_variant(spec: VariantSpec, output_dir: Path) -> ExportedVariant:
    model = TetrisNet(**spec.network.to_model_kwargs())
    model.eval()

    onnx_path = output_dir / f"{spec.label}.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.touch()
    export_ok = export_split_models(model, onnx_path)
    if not export_ok:
        raise RuntimeError(f"Failed to export split models for {spec.label}")

    conv_path, heads_path, fc_path = split_model_paths(onnx_path)
    conv_bytes = conv_path.stat().st_size
    heads_bytes = heads_path.stat().st_size
    fc_bytes = fc_path.stat().st_size
    exported = ExportedVariant(
        label=spec.label,
        network=spec.network,
        onnx_path=onnx_path,
        total_params=count_parameters(model),
        split_bytes=conv_bytes + heads_bytes + fc_bytes,
        conv_bytes=conv_bytes,
        heads_bytes=heads_bytes,
        fc_bytes=fc_bytes,
    )
    logger.info(
        "Exported split model bundle",
        label=exported.label,
        total_params=exported.total_params,
        conv_bytes=exported.conv_bytes,
        heads_bytes=exported.heads_bytes,
        fc_bytes=exported.fc_bytes,
        onnx_path=str(exported.onnx_path),
    )
    return exported


def build_state_corpus(args: ScriptArgs) -> list[TetrisEnv]:
    states: list[TetrisEnv] = []
    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        env = TetrisEnv.with_seed(10, 20, seed)
        for step_idx in range(args.states_per_seed):
            action_mask = debug_get_action_mask(env)
            valid_actions = [
                idx for idx, is_valid in enumerate(action_mask) if is_valid
            ]
            if not valid_actions:
                break

            states.append(env.clone_state())
            chosen_idx = (seed * 31 + step_idx * 17) % len(valid_actions)
            chosen_action = valid_actions[chosen_idx]
            result = env.execute_action_index(chosen_action)
            if result is None:
                raise RuntimeError(
                    f"Deterministic action selection produced invalid action {chosen_action}"
                )
    if not states:
        raise RuntimeError("State corpus generation produced no states")
    logger.info(
        "Built fixed state corpus",
        num_states=len(states),
        num_seeds=args.num_seeds,
        states_per_seed=args.states_per_seed,
    )
    return states


def create_agent(model_path: Path, *, cache_enabled: bool) -> MCTSAgent:
    config = MCTSConfig()
    config.max_placements = _DEFAULT_SELF_PLAY.max_placements
    agent = MCTSAgent(config)
    if not agent.load_model(str(model_path)):
        raise RuntimeError(f"Failed to load model bundle: {model_path}")
    if not agent.set_board_cache_enabled(cache_enabled):
        raise RuntimeError(f"Failed to set board cache on agent: {model_path}")
    return agent


def run_predictions(
    agent: MCTSAgent,
    states: list[TetrisEnv],
    *,
    max_placements: int,
    repeats: int,
) -> int:
    total_predictions = 0
    for _ in range(repeats):
        for env in states:
            agent.predict_with_valid_actions(env, max_placements=max_placements)
            total_predictions += 1
    return total_predictions


def benchmark_mode(
    exported: ExportedVariant,
    states: list[TetrisEnv],
    *,
    max_placements: int,
    timed_repeats: int,
    mode: str,
    cache_enabled: bool,
    warm_cache_first: bool,
) -> BenchmarkModeResult:
    agent = create_agent(exported.onnx_path, cache_enabled=cache_enabled)
    initial_stats = agent.get_and_reset_cache_stats()
    if initial_stats is None:
        raise RuntimeError(f"Cache stats unavailable for {exported.label}")

    warmup_hits = 0
    warmup_misses = 0
    warmup_cache_size = 0
    if warm_cache_first:
        run_predictions(agent, states, max_placements=max_placements, repeats=1)
        warmup_stats = agent.get_and_reset_cache_stats()
        if warmup_stats is None:
            raise RuntimeError(f"Warmup cache stats unavailable for {exported.label}")
        warmup_hits, warmup_misses, warmup_cache_size = warmup_stats

    start = time.perf_counter()
    total_predictions = run_predictions(
        agent,
        states,
        max_placements=max_placements,
        repeats=timed_repeats,
    )
    elapsed_sec = time.perf_counter() - start

    stats = agent.get_and_reset_cache_stats()
    if stats is None:
        raise RuntimeError(f"Cache stats unavailable for {exported.label}")
    cache_hits, cache_misses, cache_size = stats
    predictions_per_sec = total_predictions / elapsed_sec
    result = BenchmarkModeResult(
        label=exported.label,
        mode=mode,
        total_predictions=total_predictions,
        elapsed_sec=elapsed_sec,
        predictions_per_sec=predictions_per_sec,
        ms_per_prediction=1000.0 * elapsed_sec / total_predictions,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        cache_size=cache_size,
        warmup_cache_hits=warmup_hits,
        warmup_cache_misses=warmup_misses,
        warmup_cache_size=warmup_cache_size,
    )
    logger.info(
        "Benchmark mode complete",
        label=result.label,
        mode=result.mode,
        total_predictions=result.total_predictions,
        elapsed_sec=f"{result.elapsed_sec:.3f}",
        predictions_per_sec=f"{result.predictions_per_sec:.1f}",
        ms_per_prediction=f"{result.ms_per_prediction:.4f}",
        cache_hits=result.cache_hits,
        cache_misses=result.cache_misses,
        cache_size=result.cache_size,
        warmup_cache_hits=result.warmup_cache_hits,
        warmup_cache_misses=result.warmup_cache_misses,
        warmup_cache_size=result.warmup_cache_size,
    )
    return result


def compare_mode(
    results_by_label: dict[str, BenchmarkModeResult],
    faster_label: str,
    slower_label: str,
) -> dict[str, float | str]:
    faster = results_by_label[faster_label]
    slower = results_by_label[slower_label]
    speedup = faster.predictions_per_sec / slower.predictions_per_sec
    slowdown_pct = 100.0 * (slower.ms_per_prediction / faster.ms_per_prediction - 1.0)
    return {
        "faster_label": faster.label,
        "slower_label": slower.label,
        "speedup_vs_slower": speedup,
        "slower_ms_per_prediction_pct_delta": slowdown_pct,
    }


def main(args: ScriptArgs) -> None:
    output_root = args.output_root
    model_dir = output_root / "models"
    output_root.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    states = build_state_corpus(args)
    exported_variants = [export_variant(spec, model_dir) for spec in variant_specs()]

    results: list[BenchmarkModeResult] = []
    for exported in exported_variants:
        results.append(
            benchmark_mode(
                exported,
                states,
                max_placements=args.max_placements,
                timed_repeats=args.timed_repeats,
                mode="cold_no_cache",
                cache_enabled=False,
                warm_cache_first=False,
            )
        )
        results.append(
            benchmark_mode(
                exported,
                states,
                max_placements=args.max_placements,
                timed_repeats=args.timed_repeats,
                mode="hot_warm_cache",
                cache_enabled=True,
                warm_cache_first=True,
            )
        )

    cold_results = {
        result.label: result for result in results if result.mode == "cold_no_cache"
    }
    hot_results = {
        result.label: result for result in results if result.mode == "hot_warm_cache"
    }
    large_label = exported_variants[0].label
    small_label = exported_variants[1].label

    summary = {
        "backend": os.environ.get("TETRIS_NN_BACKEND", "tract"),
        "args": {
            "num_seeds": args.num_seeds,
            "states_per_seed": args.states_per_seed,
            "timed_repeats": args.timed_repeats,
            "max_placements": args.max_placements,
            "seed_start": args.seed_start,
            "output_root": str(output_root),
        },
        "state_corpus": {
            "num_states": len(states),
        },
        "variants": [
            {
                "label": exported.label,
                "network": exported.network.model_dump(mode="json"),
                "onnx_path": str(exported.onnx_path),
                "total_params": exported.total_params,
                "split_bytes": exported.split_bytes,
                "conv_bytes": exported.conv_bytes,
                "heads_bytes": exported.heads_bytes,
                "fc_bytes": exported.fc_bytes,
            }
            for exported in exported_variants
        ],
        "results": [asdict(result) for result in results],
        "comparisons": {
            "cold_no_cache": compare_mode(
                cold_results,
                faster_label=small_label,
                slower_label=large_label,
            ),
            "hot_warm_cache": compare_mode(
                hot_results,
                faster_label=small_label,
                slower_label=large_label,
            ),
        },
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    logger.info(
        "Benchmark summary written",
        summary_path=str(summary_path),
        cold_small_preds_per_sec=(
            f"{cold_results[small_label].predictions_per_sec:.1f}"
        ),
        cold_large_preds_per_sec=(
            f"{cold_results[large_label].predictions_per_sec:.1f}"
        ),
        hot_small_preds_per_sec=(f"{hot_results[small_label].predictions_per_sec:.1f}"),
        hot_large_preds_per_sec=(f"{hot_results[large_label].predictions_per_sec:.1f}"),
    )


if __name__ == "__main__":
    main(parse(ScriptArgs))
