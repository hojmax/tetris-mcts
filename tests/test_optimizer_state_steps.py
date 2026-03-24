from __future__ import annotations

from pathlib import Path

import torch

from tetris_bot.ml.weights import (
    load_checkpoint,
    load_optimizer_state_dict,
    save_checkpoint,
)


def _make_model() -> torch.nn.Linear:
    return torch.nn.Linear(8, 4)


def _run_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    compiled_model: torch.nn.Module | None = None,
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
    load_checkpoint(checkpoint, model=restored_model, optimizer=restored_optimizer)
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

    assert torch.is_tensor(
        restored_optimizer.state[first_parameter]["step"]
    )
