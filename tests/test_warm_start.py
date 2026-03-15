import json
from pathlib import Path

from tetris_bot.constants import CONFIG_FILENAME
from tetris_bot.ml.config import (
    NetworkConfig,
    OptimizerConfig,
    ReplayConfig,
    RunConfig,
    SelfPlayConfig,
    TrainingConfig,
    load_training_config_json,
)
from tetris_bot.scripts.warm_start import (
    build_output_config,
    compute_training_steps,
    has_better_validation_metric,
    optimized_worker_env_cache_path,
    resolve_eval_num_workers,
    warm_start_selection_metric,
)


def test_load_training_config_json_fills_missing_network_defaults(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / CONFIG_FILENAME
    config_path.write_text(
        json.dumps(
            {
                "network": {
                    "architecture": "gated_fusion",
                    "trunk_channels": 4,
                    "num_conv_residual_blocks": 1,
                    "reduction_channels": 8,
                    "fc_hidden": 128,
                    "conv_kernel_size": 3,
                    "conv_padding": 1,
                },
                "optimizer": {
                    "total_steps": 1000,
                    "batch_size": 512,
                },
                "self_play": {
                    "nn_value_weight": 0.01,
                    "nn_value_weight_cap": 1.0,
                    "death_penalty": 5.0,
                    "overhang_penalty_weight": 5.0,
                },
                "replay": {
                    "buffer_size": 10_000,
                },
                "run": {
                    "run_name": "v3",
                    "run_dir": "/tmp/example/v3",
                    "checkpoint_dir": "/tmp/example/v3/checkpoints",
                    "data_dir": "/tmp/example/v3",
                },
            }
        )
    )

    loaded = load_training_config_json(config_path)

    assert loaded.network.aux_hidden == NetworkConfig().aux_hidden
    assert loaded.network.num_fusion_blocks == NetworkConfig().num_fusion_blocks
    assert loaded.run.run_dir == Path("/tmp/example/v3")
    assert loaded.run.checkpoint_dir == Path("/tmp/example/v3/checkpoints")
    assert loaded.run.data_dir == Path("/tmp/example/v3")


def test_build_output_config_creates_resume_ready_incumbent_start(
    tmp_path: Path,
) -> None:
    source_config = TrainingConfig(
        network=NetworkConfig(),
        optimizer=OptimizerConfig(),
        self_play=SelfPlayConfig(
            nn_value_weight=0.01,
            nn_value_weight_cap=1.0,
            death_penalty=5.0,
            overhang_penalty_weight=5.0,
            bootstrap_without_network=True,
        ),
        replay=ReplayConfig(),
        run=RunConfig(run_name="v3"),
    )
    output_run_dir = tmp_path / "training_runs" / "v4"

    output_config = build_output_config(
        source_config,
        source_run_dir=tmp_path / "training_runs" / "v3",
        output_run_dir=output_run_dir,
    )

    assert output_config.self_play.nn_value_weight == 1.0
    assert output_config.self_play.death_penalty == 0.0
    assert output_config.self_play.overhang_penalty_weight == 0.0
    assert output_config.self_play.bootstrap_without_network is False
    assert output_config.run.run_name == "v4"
    assert output_config.run.run_dir == output_run_dir
    assert output_config.run.checkpoint_dir == output_run_dir / "checkpoints"
    assert output_config.run.data_dir == output_run_dir

    saved_config = json.loads((output_run_dir / CONFIG_FILENAME).read_text())
    assert saved_config["self_play"]["nn_value_weight"] == 1.0
    assert saved_config["self_play"]["death_penalty"] == 0.0
    assert saved_config["self_play"]["overhang_penalty_weight"] == 0.0


def test_compute_training_steps_rounds_up_epochs() -> None:
    assert compute_training_steps(900, batch_size=1024, epochs=20.0) == 18


def test_resolve_eval_num_workers_uses_machine_optimize_cache(
    tmp_path: Path,
) -> None:
    cache_path = optimized_worker_env_cache_path(tmp_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("TETRIS_OPT_NUM_WORKERS=12\n")

    resolution = resolve_eval_num_workers(
        0,
        default_workers=6,
        cache_dir=tmp_path,
    )

    assert resolution.num_workers == 12
    assert resolution.source == "optimize_cache"
    assert resolution.cache_path == str(cache_path)


def test_resolve_eval_num_workers_prefers_environment_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_path = optimized_worker_env_cache_path(tmp_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("TETRIS_OPT_NUM_WORKERS=12\n")
    monkeypatch.setenv("TETRIS_OPT_NUM_WORKERS", "10")

    resolution = resolve_eval_num_workers(
        0,
        default_workers=6,
        cache_dir=tmp_path,
    )

    assert resolution.num_workers == 10
    assert resolution.source == "environment"
    assert resolution.cache_path is None


def test_warm_start_selection_metric_weights_value_loss_by_quarter() -> None:
    assert warm_start_selection_metric(1.5, 10.0) == 4.0


def test_has_better_validation_metric_uses_fixed_selection_metric() -> None:
    best_record = {
        "val_selection_metric": 4.0,
        "val_policy_loss": 1.5,
        "val_value_loss": 10.0,
    }

    assert (
        has_better_validation_metric(
            {"policy_loss": 1.4, "value_loss": 9.9},
            best_record,
        )
        is True
    )
    assert (
        has_better_validation_metric(
            {"policy_loss": 1.55, "value_loss": 9.0},
            best_record,
        )
        is True
    )
    assert (
        has_better_validation_metric(
            {"policy_loss": 1.4, "value_loss": 10.8},
            best_record,
        )
        is False
    )
    assert (
        has_better_validation_metric(
            {"policy_loss": 1.6, "value_loss": 9.8},
            best_record,
        )
        is False
    )
