from __future__ import annotations

from pathlib import Path

import pytest

from tetris_bot.constants import DEFAULT_CONFIG_PATH, RUNTIME_OVERRIDES_FILENAME
from tetris_bot.ml.config import (
    ResolvedRuntimeOptimizerOverrides,
    ResolvedRuntimeOverrides,
    ResolvedRuntimeRunOverrides,
    ResolvedRuntimeSelfPlayOverrides,
    RuntimeOverrides,
    load_runtime_overrides,
    load_training_config,
    save_runtime_overrides,
)
from tetris_bot.ml.trainer import Trainer
from tetris_bot.ml.weights import save_checkpoint
from tetris_bot.run_setup import setup_run_directory
from tetris_bot.scripts.train import ScriptArgs, restore_trainer_from_checkpoint


def _make_config(tmp_path: Path):
    config = load_training_config(DEFAULT_CONFIG_PATH)
    return setup_run_directory(config, run_dir=tmp_path)


def _make_trainer(tmp_path: Path) -> Trainer:
    return Trainer(_make_config(tmp_path), device="cpu")


def test_setup_run_directory_writes_runtime_overrides_template(
    tmp_path: Path,
) -> None:
    _make_config(tmp_path)

    overrides = load_runtime_overrides(tmp_path / RUNTIME_OVERRIDES_FILENAME)

    assert overrides == RuntimeOverrides()


def test_trainer_reloads_runtime_overrides_and_reverts_to_defaults(
    tmp_path: Path,
) -> None:
    trainer = _make_trainer(tmp_path)
    overrides_path = tmp_path / RUNTIME_OVERRIDES_FILENAME
    config_default_add_noise = trainer.config.self_play.add_noise
    config_default_visit_sampling_epsilon = (
        trainer.config.self_play.visit_sampling_epsilon
    )

    save_runtime_overrides(
        RuntimeOverrides.model_validate(
            {
                "optimizer": {
                    "lr_multiplier": 0.1,
                    "grad_clip_norm": 3.0,
                    "weight_decay": 0.02,
                    "mirror_augmentation_probability": 0.25,
                },
                "run": {
                    "log_interval_seconds": 2.0,
                    "checkpoint_interval_seconds": 100.0,
                },
                "self_play": {
                    "add_noise": False,
                    "visit_sampling_epsilon": 0.0,
                },
            }
        ),
        overrides_path,
    )

    next_log_time_s, next_checkpoint_time_s = trainer._maybe_reload_runtime_overrides(
        now_s=500.0,
        next_log_time_s=130.0,
        next_checkpoint_time_s=11_100.0,
        force=True,
    )

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.00005)
    assert trainer.config.optimizer.grad_clip_norm == pytest.approx(3.0)
    assert trainer.optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.02)
    assert trainer.config.optimizer.mirror_augmentation_probability == pytest.approx(
        0.25
    )
    assert trainer.config.run.log_interval_seconds == pytest.approx(2.0)
    assert trainer.config.run.checkpoint_interval_seconds == pytest.approx(100.0)
    assert trainer.config.self_play.add_noise is False
    assert trainer.config.self_play.visit_sampling_epsilon == pytest.approx(0.0)
    assert next_log_time_s == pytest.approx(500.0)
    assert next_checkpoint_time_s == pytest.approx(500.0)

    save_runtime_overrides(RuntimeOverrides(), overrides_path)
    next_log_time_s, next_checkpoint_time_s = trainer._maybe_reload_runtime_overrides(
        now_s=520.0,
        next_log_time_s=next_log_time_s,
        next_checkpoint_time_s=next_checkpoint_time_s,
        force=True,
    )

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.0005)
    assert trainer.config.optimizer.grad_clip_norm == pytest.approx(10.0)
    assert trainer.optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.00005)
    assert trainer.config.optimizer.mirror_augmentation_probability == pytest.approx(
        0.5
    )
    assert trainer.config.run.log_interval_seconds == pytest.approx(10.0)
    assert trainer.config.run.checkpoint_interval_seconds == pytest.approx(10800.0)
    assert trainer.config.self_play.add_noise == config_default_add_noise
    assert trainer.config.self_play.visit_sampling_epsilon == pytest.approx(
        config_default_visit_sampling_epsilon
    )
    assert next_log_time_s == pytest.approx(520.0)
    assert next_checkpoint_time_s == pytest.approx(11_200.0)


def test_self_play_overrides_apply_to_generator_and_publish_snapshot(
    tmp_path: Path,
) -> None:
    """Final-phase use case: setting `add_noise=false` and
    `visit_sampling_epsilon=0` in `runtime_overrides.yaml` mutates the
    trainer's `self_play` config, pushes the live atomics into the
    GameGenerator, and re-publishes the SelfPlaySnapshot to R2 (mocked).
    """

    trainer = _make_trainer(tmp_path)
    overrides_path = tmp_path / RUNTIME_OVERRIDES_FILENAME

    captured_calls: list[dict[str, object]] = []

    class FakeGenerator:
        def update_search_overrides(
            self,
            *,
            add_noise: bool | None = None,
            visit_sampling_epsilon: float | None = None,
        ) -> None:
            captured_calls.append(
                {
                    "add_noise": add_noise,
                    "visit_sampling_epsilon": visit_sampling_epsilon,
                }
            )

    snapshot_publish_calls = 0

    def _record_publish() -> None:
        nonlocal snapshot_publish_calls
        snapshot_publish_calls += 1

    trainer._republish_self_play_snapshot_to_r2 = _record_publish  # type: ignore[method-assign]

    save_runtime_overrides(
        RuntimeOverrides.model_validate(
            {
                "self_play": {
                    "add_noise": False,
                    "visit_sampling_epsilon": 0.0,
                }
            }
        ),
        overrides_path,
    )

    trainer._maybe_reload_runtime_overrides(
        now_s=10.0,
        next_log_time_s=None,
        next_checkpoint_time_s=None,
        generator=FakeGenerator(),  # type: ignore[arg-type]
        force=True,
    )

    assert trainer.config.self_play.add_noise is False
    assert trainer.config.self_play.visit_sampling_epsilon == pytest.approx(0.0)
    assert captured_calls == [{"add_noise": False, "visit_sampling_epsilon": 0.0}]
    assert snapshot_publish_calls == 1


def test_self_play_overrides_validate_visit_sampling_epsilon_range(
    tmp_path: Path,
) -> None:
    trainer = _make_trainer(tmp_path)
    overrides_path = tmp_path / RUNTIME_OVERRIDES_FILENAME
    save_runtime_overrides(
        RuntimeOverrides.model_validate({"self_play": {"visit_sampling_epsilon": 1.5}}),
        overrides_path,
    )

    with pytest.raises(ValueError, match="visit_sampling_epsilon"):
        trainer._maybe_reload_runtime_overrides(
            now_s=0.0,
            next_log_time_s=None,
            next_checkpoint_time_s=None,
            force=True,
        )


def test_lr_multiplier_does_not_compound_across_scheduler_steps(
    tmp_path: Path,
) -> None:
    """Re-applying the multiplier on every scheduler step previously caused
    LR to decay geometrically (e.g. `1e-4 * 0.2^N`) past `lr_decay_steps`."""

    trainer = _make_trainer(tmp_path)
    # Push the scheduler past its decay window so LinearLR is in the
    # constant phase where each step is a no-op factor=1.0.
    trainer.align_scheduler_to_step(trainer.config.optimizer.lr_decay_steps + 1_000)
    base_lr_at_floor = float(trainer.optimizer.param_groups[0]["lr"])

    trainer._set_lr_multiplier(0.2)
    expected_effective_lr = base_lr_at_floor * 0.2
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(
        expected_effective_lr
    )

    for _ in range(200):
        trainer._step_scheduler()

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(
        expected_effective_lr
    )


def test_force_promote_next_candidate_latches_and_resets_file(
    tmp_path: Path,
) -> None:
    trainer = _make_trainer(tmp_path)
    overrides_path = tmp_path / RUNTIME_OVERRIDES_FILENAME
    save_runtime_overrides(
        RuntimeOverrides.model_validate(
            {"self_play": {"force_promote_next_candidate": True}}
        ),
        overrides_path,
    )

    trainer._maybe_reload_runtime_overrides(
        now_s=10.0,
        next_log_time_s=None,
        next_checkpoint_time_s=None,
        force=True,
    )

    assert trainer._force_promote_next_candidate is True
    persisted = load_runtime_overrides(overrides_path)
    assert persisted.self_play.force_promote_next_candidate is False


def test_force_promote_next_candidate_no_op_when_gating_disabled(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    config.self_play.use_candidate_gating = False
    trainer = Trainer(config, device="cpu")

    overrides_path = tmp_path / RUNTIME_OVERRIDES_FILENAME
    save_runtime_overrides(
        RuntimeOverrides.model_validate(
            {"self_play": {"force_promote_next_candidate": True}}
        ),
        overrides_path,
    )

    trainer._maybe_reload_runtime_overrides(
        now_s=10.0,
        next_log_time_s=None,
        next_checkpoint_time_s=None,
        force=True,
    )

    assert trainer._force_promote_next_candidate is False
    persisted = load_runtime_overrides(overrides_path)
    assert persisted.self_play.force_promote_next_candidate is False


def test_restore_trainer_restores_runtime_override_state_with_scheduler_restore(
    tmp_path: Path,
) -> None:
    trainer = _make_trainer(tmp_path)
    trainer.restore_runtime_override_state(
        ResolvedRuntimeOverrides(
            optimizer=ResolvedRuntimeOptimizerOverrides(
                lr_multiplier=0.1,
                grad_clip_norm=3.0,
                weight_decay=0.02,
                mirror_augmentation_probability=0.25,
            ),
            run=ResolvedRuntimeRunOverrides(
                log_interval_seconds=2.0,
                checkpoint_interval_seconds=100.0,
            ),
            self_play=ResolvedRuntimeSelfPlayOverrides(
                add_noise=False,
                visit_sampling_epsilon=0.0,
            ),
        ),
        lrs_already_scaled=False,
    )
    checkpoint = tmp_path / "latest.pt"
    save_checkpoint(
        trainer.model,
        trainer.ema_model,
        trainer.optimizer,
        trainer.scheduler,
        step=0,
        filepath=checkpoint,
        **trainer._runtime_override_checkpoint_state(),
    )

    restored = _make_trainer(tmp_path / "restored")
    restore_trainer_from_checkpoint(
        restored,
        ScriptArgs(config=tmp_path / "config.yaml", resume_dir=None),
        restored.config,
        checkpoint,
        incumbent_model_path=None,
    )

    assert restored.optimizer.param_groups[0]["lr"] == pytest.approx(0.00005)
    assert restored.config.optimizer.grad_clip_norm == pytest.approx(3.0)
    assert restored.optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.02)
    assert restored.config.optimizer.mirror_augmentation_probability == pytest.approx(
        0.25
    )
    assert restored.config.run.log_interval_seconds == pytest.approx(2.0)
    assert restored.config.run.checkpoint_interval_seconds == pytest.approx(100.0)
    assert restored.config.self_play.add_noise is False
    assert restored.config.self_play.visit_sampling_epsilon == pytest.approx(0.0)


def test_restore_trainer_restores_runtime_override_state_without_scheduler_restore(
    tmp_path: Path,
) -> None:
    trainer = _make_trainer(tmp_path)
    trainer.restore_runtime_override_state(
        ResolvedRuntimeOverrides(
            optimizer=ResolvedRuntimeOptimizerOverrides(
                lr_multiplier=0.1,
                grad_clip_norm=3.0,
                weight_decay=0.02,
                mirror_augmentation_probability=0.25,
            ),
            run=ResolvedRuntimeRunOverrides(
                log_interval_seconds=2.0,
                checkpoint_interval_seconds=100.0,
            ),
            self_play=ResolvedRuntimeSelfPlayOverrides(
                add_noise=False,
                visit_sampling_epsilon=0.0,
            ),
        ),
        lrs_already_scaled=False,
    )
    checkpoint = tmp_path / "latest.pt"
    save_checkpoint(
        trainer.model,
        trainer.ema_model,
        trainer.optimizer,
        trainer.scheduler,
        step=0,
        filepath=checkpoint,
        **trainer._runtime_override_checkpoint_state(),
    )

    restored = _make_trainer(tmp_path / "restored")
    restore_trainer_from_checkpoint(
        restored,
        ScriptArgs(
            config=tmp_path / "config.yaml",
            resume_dir=None,
            resume_restore_optimizer_scheduler=False,
        ),
        restored.config,
        checkpoint,
        incumbent_model_path=None,
    )

    assert restored.optimizer.param_groups[0]["lr"] == pytest.approx(0.00005)
    assert restored.config.optimizer.grad_clip_norm == pytest.approx(3.0)
    assert restored.optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.02)
    assert restored.config.optimizer.mirror_augmentation_probability == pytest.approx(
        0.25
    )
    assert restored.config.run.log_interval_seconds == pytest.approx(2.0)
    assert restored.config.run.checkpoint_interval_seconds == pytest.approx(100.0)
    assert restored.config.self_play.add_noise is False
    assert restored.config.self_play.visit_sampling_epsilon == pytest.approx(0.0)
