from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from tetris_bot.constants import BOARD_HEIGHT, BOARD_WIDTH, NUM_ACTIONS
from tetris_bot.ml.config import (
    NetworkConfig,
    OptimizerConfig,
    ReplayConfig,
    RunConfig,
    SelfPlayConfig,
    TrainingConfig,
)
from tetris_bot.ml.network import AUX_FEATURES
from tetris_bot.ml.replay_buffer import TrainingBatch
from tetris_bot.ml.trainer import Trainer
from tetris_bot.ml.weights import (
    load_checkpoint,
    load_optimizer_state_dict,
    save_checkpoint,
)


def _make_model() -> torch.nn.Linear:
    return torch.nn.Linear(8, 4)


def _make_training_config(tmp_path: Path) -> TrainingConfig:
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


def _make_training_batch(batch_size: int = 8) -> TrainingBatch:
    return TrainingBatch(
        boards=torch.zeros(batch_size, 1, BOARD_HEIGHT, BOARD_WIDTH),
        aux=torch.zeros(batch_size, AUX_FEATURES),
        policy_targets=torch.full(
            (batch_size, NUM_ACTIONS),
            1.0 / NUM_ACTIONS,
        ),
        value_targets=torch.zeros(batch_size),
        overhang_fields=torch.zeros(batch_size),
        masks=torch.ones(batch_size, NUM_ACTIONS, dtype=torch.bool),
    )


def _run_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    compiled_model: Any | None = None,
) -> None:
    active_model = compiled_model if compiled_model is not None else model
    inputs = torch.randn(32, 8)
    loss = active_model(inputs).sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def test_load_checkpoint_sanitizes_legacy_float_optimizer_steps(
    tmp_path: Path,
) -> None:
    model = _make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _run_step(model, optimizer)

    checkpoint = tmp_path / "legacy_float_step.pt"
    save_checkpoint(
        model,
        None,
        optimizer,
        scheduler=None,
        step=3,
        filepath=checkpoint,
    )

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    first_optimizer_state = next(iter(state["optimizer_state_dict"]["state"].values()))
    first_optimizer_state["step"] = float(first_optimizer_state["step"])
    torch.save(state, checkpoint)

    restored_model = _make_model()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    load_checkpoint(
        checkpoint,
        model=restored_model,
        ema_model=None,
        optimizer=restored_optimizer,
    )
    compiled_model = torch.compile(restored_model, backend="eager")

    assert all(
        torch.is_tensor(parameter_state["step"])
        for parameter_state in restored_optimizer.state.values()
        if parameter_state
    )
    _run_step(restored_model, restored_optimizer, compiled_model=compiled_model)


def test_load_optimizer_state_dict_sanitizes_legacy_float_steps() -> None:
    model = _make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _run_step(model, optimizer)

    optimizer_state_dict = optimizer.state_dict()
    first_optimizer_state = next(iter(optimizer_state_dict["state"].values()))
    first_optimizer_state["step"] = float(first_optimizer_state["step"])

    restored_model = _make_model()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    load_optimizer_state_dict(restored_optimizer, optimizer_state_dict)

    first_parameter = next(restored_model.parameters())
    compiled_model = torch.compile(restored_model, backend="eager")
    _run_step(restored_model, restored_optimizer, compiled_model=compiled_model)

    assert torch.is_tensor(restored_optimizer.state[first_parameter]["step"])


def test_train_step_sanitizes_live_float_optimizer_steps(tmp_path: Path) -> None:
    trainer = Trainer(_make_training_config(tmp_path), device="cpu")
    batch = _make_training_batch()

    trainer.train_step(batch, collect_metrics=False)

    first_parameter = next(iter(trainer.optimizer.state))
    trainer.optimizer.state[first_parameter]["step"] = float(
        trainer.optimizer.state[first_parameter]["step"]
    )

    trainer.train_step(batch, collect_metrics=False)

    assert torch.is_tensor(trainer.optimizer.state[first_parameter]["step"])


def test_train_step_updates_ema_weights(tmp_path: Path) -> None:
    config = _make_training_config(tmp_path)
    config.optimizer.ema_decay = 0.5
    trainer = Trainer(config, device="cpu")
    batch = _make_training_batch()

    ema_model = trainer.ema_model
    assert ema_model is not None
    ema_before = {
        name: tensor.clone() for name, tensor in ema_model.state_dict().items()
    }

    trainer.train_step(batch, collect_metrics=False)

    ema_after = ema_model.state_dict()
    assert any(
        not torch.equal(ema_after[name], ema_before[name]) for name in ema_before
    )
