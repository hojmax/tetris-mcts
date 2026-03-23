import json
from pathlib import Path

import torch

from tetris_bot.constants import CONFIG_FILENAME
from tetris_bot.ml.config import (
    default_training_config,
    NetworkConfig,
    load_training_config_json,
)
from tetris_bot.ml.loss import RunningLossBalancer
from tetris_bot.ml.network import TetrisNet
from tetris_bot.run_setup import config_to_json
from tetris_bot.scripts.warm_start import (
    build_output_config,
    build_wandb_config,
    compute_training_steps,
    EvalWorkerResolution,
    has_better_eval_metric,
    load_offline_resume_checkpoint,
    offline_resume_checkpoint_path,
    optimized_worker_env_cache_path,
    resolve_eval_num_workers,
    save_offline_resume_checkpoint,
    ScriptArgs,
    validate_args,
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


def test_build_output_config_uses_current_repo_defaults_for_new_run(
    tmp_path: Path,
) -> None:
    default_config = default_training_config()
    output_run_dir = tmp_path / "training_runs" / "v4"

    output_config = build_output_config(
        source_run_dir=tmp_path / "training_runs" / "v3",
        output_run_dir=output_run_dir,
    )

    assert output_config.self_play.nn_value_weight == 1.0
    assert output_config.self_play.death_penalty == 0.0
    assert output_config.self_play.overhang_penalty_weight == 0.0
    assert output_config.self_play.bootstrap_without_network is False
    assert (
        output_config.self_play.num_simulations
        == default_config.self_play.num_simulations
    )
    assert output_config.self_play.num_workers == default_config.self_play.num_workers
    assert output_config.network.trunk_channels == default_config.network.trunk_channels
    assert (
        output_config.network.num_conv_residual_blocks
        == default_config.network.num_conv_residual_blocks
    )
    assert (
        output_config.network.reduction_channels
        == default_config.network.reduction_channels
    )
    assert output_config.network.fc_hidden == default_config.network.fc_hidden
    assert output_config.network.aux_hidden == default_config.network.aux_hidden
    assert (
        output_config.network.num_fusion_blocks
        == default_config.network.num_fusion_blocks
    )
    assert output_config.optimizer.batch_size == default_config.optimizer.batch_size
    assert (
        output_config.optimizer.learning_rate == default_config.optimizer.learning_rate
    )
    assert output_config.optimizer.weight_decay == default_config.optimizer.weight_decay
    assert output_config.replay.buffer_size == default_config.replay.buffer_size
    assert output_config.run.project_name == default_config.run.project_name
    assert (
        output_config.run.model_sync_interval_seconds
        == default_config.run.model_sync_interval_seconds
    )
    assert (
        output_config.run.checkpoint_interval_seconds
        == default_config.run.checkpoint_interval_seconds
    )
    assert (
        output_config.run.log_interval_seconds
        == default_config.run.log_interval_seconds
    )
    assert (
        output_config.run.save_interval_seconds
        == default_config.run.save_interval_seconds
    )
    assert output_config.run.run_name == "v4"
    assert output_config.run.run_dir == output_run_dir
    assert output_config.run.checkpoint_dir == output_run_dir / "checkpoints"
    assert output_config.run.data_dir == output_run_dir

    saved_config = json.loads((output_run_dir / CONFIG_FILENAME).read_text())
    assert (
        saved_config["network"]["trunk_channels"]
        == default_config.network.trunk_channels
    )
    assert (
        saved_config["optimizer"]["batch_size"] == default_config.optimizer.batch_size
    )
    assert saved_config["self_play"]["nn_value_weight"] == 1.0
    assert saved_config["self_play"]["death_penalty"] == 0.0
    assert saved_config["self_play"]["overhang_penalty_weight"] == 0.0


def test_validate_args_does_not_require_source_config_when_npz_exists(
    tmp_path: Path,
) -> None:
    source_run_dir = tmp_path / "training_runs" / "v5"
    source_run_dir.mkdir(parents=True)
    (source_run_dir / "training_data.npz").write_bytes(b"placeholder")

    validate_args(
        ScriptArgs(
            source_run_dir=source_run_dir,
            output_run_dir=tmp_path / "training_runs" / "v6",
            preload_to_gpu=False,
        )
    )


def test_build_wandb_config_includes_full_resolved_training_config(
    tmp_path: Path,
) -> None:
    source_run_dir = tmp_path / "training_runs" / "v5"
    output_run_dir = tmp_path / "training_runs" / "v6"
    resolved_output_config = build_output_config(
        source_run_dir=source_run_dir,
        output_run_dir=output_run_dir,
    )
    args = ScriptArgs(
        source_run_dir=source_run_dir,
        output_run_dir=output_run_dir,
        preload_to_gpu=False,
    )
    eval_worker_resolution = EvalWorkerResolution(num_workers=8, source="config")

    wandb_config = build_wandb_config(
        args=args,
        source_run_dir=source_run_dir,
        source_config_path=source_run_dir / CONFIG_FILENAME,
        source_training_data_path=source_run_dir / "training_data.npz",
        source_offline_resume_checkpoint=None,
        resolved_output_config=resolved_output_config,
        device_str="cpu",
        preload_mode="none",
        batch_size=resolved_output_config.optimizer.batch_size,
        learning_rate=resolved_output_config.optimizer.learning_rate,
        weight_decay=resolved_output_config.optimizer.weight_decay,
        grad_clip_norm=resolved_output_config.optimizer.grad_clip_norm,
        eval_worker_resolution=eval_worker_resolution,
    )

    serialized_output_config = json.loads(config_to_json(resolved_output_config))
    assert wandb_config["output_config"] == serialized_output_config
    assert wandb_config["training_config"] == serialized_output_config


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


def test_has_better_eval_metric_uses_fixed_selection_metric() -> None:
    best_record: dict[str, float | int | bool | str] = {
        "eval_selection_metric": 4.0,
        "eval_policy_loss": 1.5,
        "eval_value_loss": 10.0,
    }

    assert (
        has_better_eval_metric(
            {"policy_loss": 1.4, "value_loss": 9.9},
            best_record,
        )
        is True
    )
    assert (
        has_better_eval_metric(
            {"policy_loss": 1.55, "value_loss": 9.0},
            best_record,
        )
        is True
    )
    assert (
        has_better_eval_metric(
            {"policy_loss": 1.4, "value_loss": 10.8},
            best_record,
        )
        is False
    )


def test_validate_args_requires_offline_resume_checkpoint_when_requested(
    tmp_path: Path,
) -> None:
    source_run_dir = tmp_path / "training_runs" / "v5"
    source_run_dir.mkdir(parents=True)
    (source_run_dir / CONFIG_FILENAME).write_text(json.dumps({}))
    (source_run_dir / "training_data.npz").write_bytes(b"placeholder")

    args = ScriptArgs(
        source_run_dir=source_run_dir,
        output_run_dir=tmp_path / "training_runs" / "v6",
        resume_from_source_offline_state=True,
        preload_to_gpu=False,
    )

    try:
        validate_args(args)
    except FileNotFoundError as error:
        assert "offline warm-start resume checkpoint" in str(error)
    else:
        raise AssertionError(
            "validate_args should require an offline resume checkpoint"
        )


def test_offline_resume_checkpoint_round_trip_restores_optimizer_and_history(
    tmp_path: Path,
) -> None:
    checkpoint_path = offline_resume_checkpoint_path(tmp_path / "training_runs" / "v5")
    model = TetrisNet(**NetworkConfig().to_model_kwargs())
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    loss_balancer = RunningLossBalancer(window_size=8)
    loss_balancer.append(2.0, 8.0)
    loss_balancer.append(1.8, 7.2)

    boards = torch.randn(2, 1, 20, 10)
    aux = torch.randn(2, 80)
    policy_logits, value = model(boards, aux)
    loss = policy_logits.sum() + value.sum()
    loss.backward()
    optimizer.step()

    best_state_dict = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    best_record = {
        "round_index": 4,
        "step": 14064,
        "val_selection_metric": 4.25,
    }
    history = [{"round_index": 4, "step": 14064, "improved": True}]
    rng_state = {"seed": 123, "note": "test"}

    save_offline_resume_checkpoint(
        checkpoint_path,
        model=model,
        optimizer_state_dict=optimizer.state_dict(),
        best_state_dict=best_state_dict,
        best_record=best_record,
        history=history,
        total_steps=14064,
        rounds_completed=4,
        non_improving_rounds=2,
        current_value_loss_weight=0.25,
        loss_balancer_state=loss_balancer.state_dict(),
        rng_state=rng_state,
    )

    restored_model = TetrisNet(**NetworkConfig().to_model_kwargs())
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=5e-4)
    restored_loss_balancer = RunningLossBalancer(window_size=8)
    resumed = load_offline_resume_checkpoint(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        loss_balancer=restored_loss_balancer,
    )

    assert resumed.checkpoint_path == checkpoint_path
    assert resumed.total_steps == 14064
    assert resumed.rounds_completed == 4
    assert resumed.non_improving_rounds == 2
    assert resumed.current_value_loss_weight == 0.25
    assert resumed.best_record == best_record
    assert resumed.history == history
    assert resumed.rng_state == rng_state
    assert resumed.best_state_dict.keys() == best_state_dict.keys()
    assert restored_loss_balancer.state_dict() == loss_balancer.state_dict()
    restored_optimizer_state = restored_optimizer.state_dict()["state"]
    expected_optimizer_state = optimizer.state_dict()["state"]
    assert restored_optimizer_state.keys() == expected_optimizer_state.keys()
    for parameter_id, expected_state in expected_optimizer_state.items():
        restored_state = restored_optimizer_state[parameter_id]
        assert restored_state.keys() == expected_state.keys()
        for state_key, expected_value in expected_state.items():
            restored_value = restored_state[state_key]
            if isinstance(expected_value, torch.Tensor):
                assert torch.equal(restored_value, expected_value)
            else:
                assert restored_value == expected_value
