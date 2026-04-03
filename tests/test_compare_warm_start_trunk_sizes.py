from pathlib import Path

import pytest

from tetris_bot.ml.config import (
    NetworkConfig,
    TrainingConfig,
    default_network_config,
    default_training_config,
)
from tetris_bot.scripts.ablations.compare_warm_start_trunk_sizes import (
    BenchmarkAggregate,
    ScriptArgs,
    build_markdown_table,
    build_variant_network_config,
    build_variants,
    pareto_optimal_labels,
    resolve_trunk_channels,
)


def _source_config(network: NetworkConfig | None = None) -> TrainingConfig:
    config = default_training_config()
    config.network = (
        network
        if network is not None
        else default_network_config().model_copy(update={"fc_hidden": 128})
    )
    return config


def test_build_variants_defaults_to_half_source_and_double_source() -> None:
    args = ScriptArgs(source_run_dir=Path("/tmp/source"))

    variants = build_variants(_source_config(), args)

    assert [variant.label for variant in variants] == [
        "trunk8_blocks2_reduce16",
        "trunk16_blocks3_reduce32",
        "trunk32_blocks4_reduce64",
    ]
    assert [variant.num_conv_residual_blocks for variant in variants] == [2, 3, 4]
    assert [variant.matches_source for variant in variants] == [False, True, False]


def test_resolve_trunk_channels_requires_two_unique_values() -> None:
    args = ScriptArgs(
        source_run_dir=Path("/tmp/source"),
        trunk_channels=[12, 12],
    )

    with pytest.raises(ValueError, match="at least two unique trunk sizes"):
        resolve_trunk_channels(_source_config().network, args)


def test_build_variant_network_config_only_changes_trunk_and_reduction() -> None:
    source_network = default_network_config().model_copy(
        update={
            "trunk_channels": 16,
            "num_conv_residual_blocks": 4,
            "reduction_channels": 32,
            "fc_hidden": 192,
            "aux_hidden": 48,
            "num_fusion_blocks": 2,
            "conv_kernel_size": 5,
            "conv_padding": 2,
        }
    )
    variant = build_variants(
        _source_config(source_network),
        ScriptArgs(
            source_run_dir=Path("/tmp/source"),
            trunk_channels=[12, 16, 24],
        ),
    )[0]

    variant_network = build_variant_network_config(source_network, variant)

    assert variant_network.trunk_channels == 12
    assert variant_network.num_conv_residual_blocks == 3
    assert variant_network.reduction_channels == 24
    assert variant_network.fc_hidden == 192
    assert variant_network.aux_hidden == 48
    assert variant_network.num_fusion_blocks == 2
    assert variant_network.conv_kernel_size == 5
    assert variant_network.conv_padding == 2


def test_build_variants_accepts_explicit_residual_block_override() -> None:
    args = ScriptArgs(
        source_run_dir=Path("/tmp/source"),
        trunk_channels=[12, 16, 24],
        residual_block_counts=[1, 4, 6],
    )

    variants = build_variants(_source_config(), args)

    assert [variant.num_conv_residual_blocks for variant in variants] == [1, 4, 6]
    assert [variant.label for variant in variants] == [
        "trunk12_blocks1_reduce24",
        "trunk16_blocks4_reduce32",
        "trunk24_blocks6_reduce48",
    ]


def test_pareto_frontier_and_markdown_table_capture_tradeoff_metrics() -> None:
    aggregates = [
        BenchmarkAggregate(
            label="trunk8_blocks2_reduce16",
            trunk_channels=8,
            num_conv_residual_blocks=2,
            reduction_channels=16,
            num_parameters=1000,
            matches_source=False,
            output_run_dir="/tmp/trunk8",
            incumbent_onnx_path="/tmp/trunk8/incumbent.onnx",
            warm_start_avg_attack=18.0,
            warm_start_avg_moves=45.0,
            warm_start_num_steps=100,
            games_per_second_mean=3.2,
            games_per_second_std=0.1,
            moves_per_second_mean=140.0,
            moves_per_second_std=2.5,
            benchmark_avg_attack_mean=18.5,
            benchmark_avg_attack_std=0.3,
            benchmark_avg_moves_mean=44.0,
            benchmark_avg_moves_std=1.0,
            elapsed_sec_mean=6.0,
            elapsed_sec_std=0.2,
            speedup_vs_baseline=1.1,
            attack_delta_vs_baseline=-0.2,
            pareto_optimal=True,
        ),
        BenchmarkAggregate(
            label="trunk16_blocks3_reduce32",
            trunk_channels=16,
            num_conv_residual_blocks=3,
            reduction_channels=32,
            num_parameters=2000,
            matches_source=True,
            output_run_dir="/tmp/trunk16",
            incumbent_onnx_path="/tmp/trunk16/incumbent.onnx",
            warm_start_avg_attack=18.7,
            warm_start_avg_moves=46.0,
            warm_start_num_steps=100,
            games_per_second_mean=2.9,
            games_per_second_std=0.1,
            moves_per_second_mean=135.0,
            moves_per_second_std=2.0,
            benchmark_avg_attack_mean=18.7,
            benchmark_avg_attack_std=0.2,
            benchmark_avg_moves_mean=45.5,
            benchmark_avg_moves_std=1.0,
            elapsed_sec_mean=6.5,
            elapsed_sec_std=0.1,
            speedup_vs_baseline=1.0,
            attack_delta_vs_baseline=0.0,
            pareto_optimal=True,
        ),
        BenchmarkAggregate(
            label="trunk32_blocks4_reduce64",
            trunk_channels=32,
            num_conv_residual_blocks=4,
            reduction_channels=64,
            num_parameters=4000,
            matches_source=False,
            output_run_dir="/tmp/trunk32",
            incumbent_onnx_path="/tmp/trunk32/incumbent.onnx",
            warm_start_avg_attack=18.4,
            warm_start_avg_moves=44.0,
            warm_start_num_steps=100,
            games_per_second_mean=2.1,
            games_per_second_std=0.1,
            moves_per_second_mean=110.0,
            moves_per_second_std=3.0,
            benchmark_avg_attack_mean=18.2,
            benchmark_avg_attack_std=0.4,
            benchmark_avg_moves_mean=43.5,
            benchmark_avg_moves_std=1.2,
            elapsed_sec_mean=8.0,
            elapsed_sec_std=0.3,
            speedup_vs_baseline=0.72,
            attack_delta_vs_baseline=-0.5,
            pareto_optimal=False,
        ),
    ]

    frontier = pareto_optimal_labels(aggregates)
    table = build_markdown_table(aggregates)

    assert frontier == {
        "trunk8_blocks2_reduce16",
        "trunk16_blocks3_reduce32",
    }
    assert "| variant | trunk | blocks | reduce | params | warm-start attack |" in table
    assert "3.20 +/- 0.10" in table
    assert "1.100x" in table
    assert "| trunk32_blocks4_reduce64 | 32 | 4 | 64 | 4000 |" in table
    assert (
        "| trunk32_blocks4_reduce64 | 32 | 4 | 64 | 4000 | 18.40 | 18.20 +/- 0.40 | 2.10 +/- 0.10 | 110.0 +/- 3.0 | 0.720x | -0.50 | no |"
        in table
    )
