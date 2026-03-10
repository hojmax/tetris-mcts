from pathlib import Path

from tetris_bot.ml.config import (
    NetworkConfig,
    OptimizerConfig,
    ReplayConfig,
    RunConfig,
    SelfPlayConfig,
    TrainingConfig,
)
from tetris_bot.ml.trainer import Trainer
from tetris_bot.ml.weights import save_checkpoint
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
        ),
    )


def _make_trainer(tmp_path: Path) -> tuple[Trainer, TrainingConfig]:
    config = _make_config(tmp_path)
    trainer = Trainer(config, device="cpu")
    return trainer, config


def test_restore_trainer_restores_search_penalties_from_checkpoint(
    tmp_path: Path,
) -> None:
    trainer, config = _make_trainer(tmp_path)
    checkpoint = tmp_path / "latest.pt"
    save_checkpoint(
        trainer.model,
        trainer.optimizer,
        trainer.scheduler,
        step=123,
        filepath=checkpoint,
        incumbent_uses_network=True,
        incumbent_nn_value_weight=1.0,
        incumbent_death_penalty=0.0,
        incumbent_overhang_penalty_weight=0.0,
        incumbent_eval_avg_attack=19.54,
    )

    # Mutate the config away from the checkpoint values so the restore is observable.
    config.self_play.nn_value_weight = 0.01
    config.self_play.death_penalty = 5.0
    config.self_play.overhang_penalty_weight = 5.0

    restore_trainer_from_checkpoint(
        trainer,
        ScriptArgs(training=config, resume_dir=None),
        config,
        checkpoint,
        incumbent_model_path=None,
    )

    assert config.self_play.nn_value_weight == 1.0
    assert config.self_play.death_penalty == 0.0
    assert config.self_play.overhang_penalty_weight == 0.0
    assert trainer.initial_incumbent_eval_avg_attack == 19.54


def test_restore_trainer_infers_zero_penalties_for_legacy_cap_checkpoint(
    tmp_path: Path,
) -> None:
    trainer, config = _make_trainer(tmp_path)
    checkpoint = tmp_path / "legacy.pt"
    save_checkpoint(
        trainer.model,
        trainer.optimizer,
        trainer.scheduler,
        step=456,
        filepath=checkpoint,
        incumbent_uses_network=True,
        incumbent_nn_value_weight=1.0,
        incumbent_eval_avg_attack=17.0,
    )

    config.self_play.nn_value_weight_cap = 1.0
    config.self_play.death_penalty = 5.0
    config.self_play.overhang_penalty_weight = 5.0

    restore_trainer_from_checkpoint(
        trainer,
        ScriptArgs(training=config, resume_dir=None),
        config,
        checkpoint,
        incumbent_model_path=None,
    )

    assert config.self_play.nn_value_weight == 1.0
    assert config.self_play.death_penalty == 0.0
    assert config.self_play.overhang_penalty_weight == 0.0
