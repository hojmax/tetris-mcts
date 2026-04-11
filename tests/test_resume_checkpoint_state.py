from pathlib import Path
from types import SimpleNamespace

import torch

import tetris_bot.ml.trainer as trainer_module
from tetris_bot.constants import DEFAULT_CONFIG_PATH
from tetris_bot.ml.config import TrainingConfig, load_training_config
from tetris_bot.ml.trainer import Trainer
from tetris_bot.ml.weights import save_checkpoint
from tetris_bot.scripts.train import ScriptArgs, restore_trainer_from_checkpoint


def _make_config(tmp_path: Path) -> TrainingConfig:
    config = load_training_config(DEFAULT_CONFIG_PATH)
    checkpoint_dir = tmp_path / "checkpoints"
    data_dir = tmp_path / "data"
    config.run = config.run.model_copy(
        update={
            "run_dir": tmp_path,
            "checkpoint_dir": checkpoint_dir,
            "data_dir": data_dir,
        }
    )
    return config


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
        trainer.ema_model,
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
        ScriptArgs(config=tmp_path / "config.yaml", resume_dir=None),
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
        trainer.ema_model,
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
        ScriptArgs(config=tmp_path / "config.yaml", resume_dir=None),
        config,
        checkpoint,
        incumbent_model_path=None,
    )

    assert config.self_play.nn_value_weight == 1.0
    assert config.self_play.death_penalty == 0.0
    assert config.self_play.overhang_penalty_weight == 0.0


def test_restore_trainer_can_use_latest_checkpoint_model_as_incumbent(
    tmp_path: Path,
) -> None:
    trainer, config = _make_trainer(tmp_path)
    config.self_play.use_candidate_gating = True
    checkpoint = tmp_path / "latest.pt"
    incumbent_path = tmp_path / "incumbent.onnx"
    incumbent_path.write_text("placeholder")
    save_checkpoint(
        trainer.model,
        trainer.ema_model,
        trainer.optimizer,
        trainer.scheduler,
        step=789,
        filepath=checkpoint,
        incumbent_uses_network=True,
        incumbent_nn_value_weight=1.0,
        incumbent_eval_avg_attack=42.0,
    )

    restore_trainer_from_checkpoint(
        trainer,
        ScriptArgs(
            config=tmp_path / "config.yaml",
            resume_dir=None,
            resume_latest_as_incumbent=True,
        ),
        config,
        checkpoint,
        incumbent_model_path=incumbent_path,
    )

    assert trainer.initial_incumbent_model_path is None
    assert trainer.initial_incumbent_eval_avg_attack == 0.0
    assert trainer.recompute_initial_incumbent_eval_avg_attack is True


def test_restore_trainer_skips_baseline_recompute_when_candidate_gating_disabled(
    tmp_path: Path,
) -> None:
    trainer, config = _make_trainer(tmp_path)
    config.self_play.use_candidate_gating = False
    checkpoint = tmp_path / "latest.pt"
    incumbent_path = tmp_path / "incumbent.onnx"
    incumbent_path.write_text("placeholder")
    save_checkpoint(
        trainer.model,
        trainer.ema_model,
        trainer.optimizer,
        trainer.scheduler,
        step=789,
        filepath=checkpoint,
        incumbent_uses_network=True,
        incumbent_nn_value_weight=1.0,
        incumbent_eval_avg_attack=42.0,
    )

    restore_trainer_from_checkpoint(
        trainer,
        ScriptArgs(
            config=tmp_path / "config.yaml",
            resume_dir=None,
            resume_latest_as_incumbent=True,
        ),
        config,
        checkpoint,
        incumbent_model_path=incumbent_path,
    )

    assert trainer.initial_incumbent_model_path is None
    assert trainer.initial_incumbent_eval_avg_attack == 0.0
    assert trainer.recompute_initial_incumbent_eval_avg_attack is False


def test_evaluate_starting_incumbent_avg_attack_uses_candidate_gate_settings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trainer, config = _make_trainer(tmp_path)
    config.self_play.num_workers = 7
    config.self_play.model_promotion_eval_games = 20
    config.self_play.nn_value_weight = 1.0
    config.self_play.nn_value_weight_cap = 1.0
    config.self_play.death_penalty = 5.0
    config.self_play.overhang_penalty_weight = 6.0
    config.self_play.visit_sampling_epsilon = 0.25
    config.self_play.mcts_seed = 123

    captured: dict[str, object] = {}

    def fake_evaluate_model(
        model_path: str,
        seeds: list[int],
        eval_config,
        max_placements: int,
        num_workers: int,
        add_noise: bool,
    ):
        captured["model_path"] = model_path
        captured["seeds"] = seeds
        captured["seed"] = eval_config.seed
        captured["visit_sampling_epsilon"] = eval_config.visit_sampling_epsilon
        captured["nn_value_weight"] = eval_config.nn_value_weight
        captured["death_penalty"] = eval_config.death_penalty
        captured["overhang_penalty_weight"] = eval_config.overhang_penalty_weight
        captured["max_placements"] = max_placements
        captured["num_workers"] = num_workers
        captured["add_noise"] = add_noise
        return SimpleNamespace(avg_attack=12.5, max_attack=20, num_games=len(seeds))

    monkeypatch.setattr(trainer_module, "evaluate_model", fake_evaluate_model)

    avg_attack = trainer._evaluate_starting_incumbent_avg_attack(
        tmp_path / "parallel.onnx"
    )

    assert avg_attack == 12.5
    assert captured["model_path"] == str(tmp_path / "parallel.onnx")
    assert captured["seeds"] == list(range(20))
    assert captured["seed"] == 0
    assert captured["visit_sampling_epsilon"] == 0.0
    assert captured["nn_value_weight"] == 1.0
    assert captured["death_penalty"] == 0.0
    assert captured["overhang_penalty_weight"] == 0.0
    assert captured["max_placements"] == config.self_play.max_placements
    assert captured["num_workers"] == 7
    assert captured["add_noise"] is False


def test_restore_trainer_recovers_ema_state_from_checkpoint(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.optimizer.ema_decay = 0.5
    trainer = Trainer(config, device="cpu")
    checkpoint = tmp_path / "ema.pt"
    ema_model = trainer.ema_model
    assert ema_model is not None

    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.add_(1.0)
        for parameter in ema_model.parameters():
            parameter.sub_(1.0)

    expected_ema_state = {
        name: tensor.clone() for name, tensor in ema_model.state_dict().items()
    }
    save_checkpoint(
        trainer.model,
        trainer.ema_model,
        trainer.optimizer,
        trainer.scheduler,
        step=12,
        filepath=checkpoint,
        incumbent_uses_network=True,
        incumbent_nn_value_weight=1.0,
        incumbent_eval_avg_attack=0.0,
    )

    restored = Trainer(config, device="cpu")
    restore_trainer_from_checkpoint(
        restored,
        ScriptArgs(config=tmp_path / "config.yaml", resume_dir=None),
        config,
        checkpoint,
        incumbent_model_path=None,
    )

    restored_ema_model = restored.ema_model
    assert restored_ema_model is not None
    assert all(
        torch.equal(restored_ema_model.state_dict()[name], expected_ema_state[name])
        for name in expected_ema_state
    )
