from __future__ import annotations

from pathlib import Path

import pytest

from tetris_bot.ml.config import (
    NetworkConfig,
    OptimizerConfig,
    ReplayConfig,
    RunConfig,
    SelfPlayConfig,
    TrainingConfig,
)
from tetris_bot.ml.trainer import CandidateGateSchedule, Trainer
from tetris_bot.scripts.train import ScriptArgs, restore_trainer_from_checkpoint


def _make_config(tmp_path: Path) -> TrainingConfig:
    checkpoint_dir = tmp_path / "checkpoints"
    data_dir = tmp_path / "data"
    return TrainingConfig(
        network=NetworkConfig(),
        optimizer=OptimizerConfig(),
        self_play=SelfPlayConfig(),
        replay=ReplayConfig(),
        run=RunConfig(
            run_dir=tmp_path,
            checkpoint_dir=checkpoint_dir,
            data_dir=data_dir,
            model_sync_interval_seconds=120.0,
            model_sync_failure_backoff_seconds=120.0,
            model_sync_max_interval_seconds=0.0,
        ),
    )


def _make_trainer(tmp_path: Path) -> Trainer:
    return Trainer(_make_config(tmp_path), device="cpu")


def test_candidate_gate_schedule_uses_eval_start_for_backoff(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path)
    schedule = trainer._initialize_candidate_gate_schedule(now_s=1_000.0)

    assert schedule.current_interval_seconds == 120.0
    assert schedule.failed_promotion_streak == 0
    assert schedule.next_export_time_s == 1_120.0

    trainer._update_candidate_gate_schedule_from_eval(
        schedule,
        evaluation_seconds=500.0,
        promoted=False,
        now_s=1_600.0,
    )

    assert schedule.failed_promotion_streak == 1
    assert schedule.current_interval_seconds == 240.0
    assert schedule.next_export_time_s == 1_340.0


def test_candidate_gate_schedule_resets_after_promotion(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path)
    schedule = CandidateGateSchedule(
        current_interval_seconds=360.0,
        failed_promotion_streak=2,
        next_export_time_s=0.0,
    )

    trainer._update_candidate_gate_schedule_from_eval(
        schedule,
        evaluation_seconds=100.0,
        promoted=True,
        now_s=2_400.0,
    )

    assert schedule.failed_promotion_streak == 0
    assert schedule.current_interval_seconds == 120.0
    assert schedule.next_export_time_s == 2_420.0


def test_restore_trainer_from_checkpoint_restores_candidate_gate_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trainer = _make_trainer(tmp_path)
    args = ScriptArgs(training=trainer.config, resume_dir=None)

    monkeypatch.setattr(
        "tetris_bot.scripts.train.load_checkpoint",
        lambda checkpoint, model, optimizer, scheduler: {
            "step": 77,
            "incumbent_uses_network": False,
            "incumbent_nn_value_weight": 0.3,
            "incumbent_death_penalty": 1.0,
            "incumbent_overhang_penalty_weight": 2.0,
            "incumbent_eval_avg_attack": 4.5,
            "candidate_gate_current_interval_seconds": 360.0,
            "candidate_gate_failed_promotion_streak": 2,
            "candidate_gate_next_export_delay_seconds": 45.0,
        },
    )

    restore_trainer_from_checkpoint(
        trainer=trainer,
        args=args,
        config=trainer.config,
        checkpoint=tmp_path / "latest.pt",
        incumbent_model_path=None,
    )

    assert trainer.step == 77
    assert trainer.initial_candidate_gate_interval_seconds == pytest.approx(360.0)
    assert trainer.initial_candidate_gate_failed_promotion_streak == 2
    assert trainer.initial_candidate_gate_next_export_delay_seconds == pytest.approx(
        45.0
    )
