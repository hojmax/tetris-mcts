from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tetris_bot.constants import BOARD_HEIGHT, BOARD_WIDTH
from tetris_bot.ml.config import NetworkConfig
from tetris_bot.ml.network import AUX_FEATURES, TetrisNet
from tetris_bot.ml.weights import (
    AsyncCheckpointSaver,
    CheckpointSnapshot,
    capture_checkpoint_snapshot,
)


def _make_model() -> TetrisNet:
    return TetrisNet(**NetworkConfig().to_model_kwargs())


def test_capture_checkpoint_snapshot_clones_state_to_cpu() -> None:
    model = _make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    boards = torch.zeros(2, 1, BOARD_HEIGHT, BOARD_WIDTH)
    aux = torch.zeros(2, AUX_FEATURES)
    policy_logits, value = model(boards, aux)
    loss = policy_logits.square().mean() + value.square().mean()
    loss.backward()
    optimizer.step()

    snapshot = capture_checkpoint_snapshot(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        step=7,
        extra_checkpoint_state={"tag": "checkpoint"},
    )

    first_key = next(iter(snapshot.model_state_dict))
    first_snapshot_tensor = snapshot.model_state_dict[first_key]
    assert isinstance(first_snapshot_tensor, torch.Tensor)
    assert first_snapshot_tensor.device.type == "cpu"
    first_snapshot_copy = first_snapshot_tensor.clone()

    with torch.no_grad():
        first_param = next(model.parameters())
        first_param.add_(1.0)

    assert torch.equal(snapshot.model_state_dict[first_key], first_snapshot_copy)
    assert snapshot.extra_state == {"tag": "checkpoint"}

    assert snapshot.optimizer_state_dict is not None
    optimizer_states = snapshot.optimizer_state_dict["state"].values()
    optimizer_tensors = [
        value
        for state in optimizer_states
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ]
    assert optimizer_tensors
    assert all(tensor.device.type == "cpu" for tensor in optimizer_tensors)


def test_async_checkpoint_saver_flushes_completed_requests(tmp_path: Path) -> None:
    saved_steps: list[int] = []

    class FakeWeightManager:
        def save_snapshot(
            self,
            snapshot: CheckpointSnapshot,
            model_kwargs: dict[str, object],
            eval_metrics: dict[str, object] | None = None,
            export_for_rust: bool = True,
        ) -> dict[str, Path]:
            saved_steps.append(snapshot.step)
            return {
                "checkpoint": tmp_path / f"checkpoint_{snapshot.step}.pt",
                "metadata": tmp_path / "latest.json",
            }

    saver = AsyncCheckpointSaver(FakeWeightManager())  # type: ignore[arg-type]
    saver.submit(
        snapshot=CheckpointSnapshot(
            step=11,
            model_state_dict={},
            optimizer_state_dict=None,
            scheduler_state_dict=None,
            extra_state={},
        ),
        model_kwargs={"architecture": "gated_fusion"},
        export_for_rust=False,
    )

    saver.shutdown()

    assert saved_steps == [11]
    completed = saver.drain_completed()
    assert len(completed) == 1
    assert completed[0][0] == 11


def test_async_checkpoint_saver_propagates_worker_errors() -> None:
    class FailingWeightManager:
        def save_snapshot(
            self,
            snapshot: CheckpointSnapshot,
            model_kwargs: dict[str, object],
            eval_metrics: dict[str, object] | None = None,
            export_for_rust: bool = True,
        ) -> dict[str, Path]:
            raise RuntimeError(f"boom-{snapshot.step}")

    saver = AsyncCheckpointSaver(FailingWeightManager())  # type: ignore[arg-type]
    saver.submit(
        snapshot=CheckpointSnapshot(
            step=13,
            model_state_dict={},
            optimizer_state_dict=None,
            scheduler_state_dict=None,
            extra_state={},
        ),
        model_kwargs={"architecture": "gated_fusion"},
        export_for_rust=False,
    )

    with pytest.raises(RuntimeError, match="Asynchronous checkpoint save failed"):
        saver.shutdown()
