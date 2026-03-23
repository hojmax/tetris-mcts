from __future__ import annotations

from pathlib import Path

import numpy as np
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
from tetris_bot.ml.trainer import Trainer


class FakeReplayMirrorGenerator:
    def __init__(
        self,
        *,
        window_start: int,
        boards: np.ndarray,
        aux: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
        overhang_fields: np.ndarray,
        masks: np.ndarray,
    ) -> None:
        self.window_start = window_start
        self.window_end = window_start + int(boards.shape[0])
        self.boards = boards
        self.aux = aux
        self.policy_targets = policy_targets
        self.value_targets = value_targets
        self.overhang_fields = overhang_fields
        self.masks = masks
        self.snapshot_calls = 0
        self.delta_calls: list[int] = []

    def replay_buffer_snapshot(self) -> None:
        self.snapshot_calls += 1
        raise AssertionError(
            "replay_buffer_snapshot should not be used for mirror load"
        )

    def replay_buffer_delta(
        self,
        from_index: int,
        max_examples: int,
    ) -> tuple[
        int,
        int,
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        self.delta_calls.append(from_index)
        slice_start = max(from_index, self.window_start)
        slice_end = min(slice_start + max_examples, self.window_end)
        offset_start = slice_start - self.window_start
        offset_end = slice_end - self.window_start
        return (
            self.window_start,
            self.window_end,
            slice_start,
            self.boards[offset_start:offset_end],
            self.aux[offset_start:offset_end],
            self.policy_targets[offset_start:offset_end],
            self.value_targets[offset_start:offset_end],
            self.overhang_fields[offset_start:offset_end],
            self.masks[offset_start:offset_end],
        )


def _make_config(tmp_path: Path) -> TrainingConfig:
    checkpoint_dir = tmp_path / "checkpoints"
    data_dir = tmp_path / "data"
    return TrainingConfig(
        network=NetworkConfig(),
        optimizer=OptimizerConfig(),
        self_play=SelfPlayConfig(),
        replay=ReplayConfig(
            buffer_size=8,
            replay_mirror_delta_chunk_examples=2,
        ),
        run=RunConfig(
            run_dir=tmp_path,
            checkpoint_dir=checkpoint_dir,
            data_dir=data_dir,
        ),
    )


def test_load_replay_mirror_bootstraps_from_deltas_without_snapshot(
    tmp_path: Path,
) -> None:
    trainer = Trainer(_make_config(tmp_path), device="cpu")
    num_examples = 5
    boards = (
        np.arange(num_examples * BOARD_HEIGHT * BOARD_WIDTH, dtype=np.float32).reshape(
            num_examples, BOARD_HEIGHT * BOARD_WIDTH
        )
        % 2
    )
    aux = np.arange(num_examples * AUX_FEATURES, dtype=np.float32).reshape(
        num_examples, AUX_FEATURES
    )
    policy_targets = np.zeros((num_examples, NUM_ACTIONS), dtype=np.float32)
    value_targets = np.arange(num_examples, dtype=np.float32)
    overhang_fields = np.arange(num_examples, dtype=np.float32) + 0.5
    masks = np.zeros((num_examples, NUM_ACTIONS), dtype=np.float32)
    for row in range(num_examples):
        policy_targets[row, row] = 1.0
        masks[row, row] = 1.0

    generator = FakeReplayMirrorGenerator(
        window_start=7,
        boards=boards,
        aux=aux,
        policy_targets=policy_targets,
        value_targets=value_targets,
        overhang_fields=overhang_fields,
        masks=masks,
    )

    mirror = trainer._load_replay_mirror(generator)

    assert mirror is not None
    assert generator.snapshot_calls == 0
    assert generator.delta_calls == [0, 9, 11]
    assert mirror.count == num_examples
    assert mirror.write_pos == num_examples
    assert mirror.logical_end == 12
    assert torch.equal(
        mirror.boards[:num_examples],
        torch.from_numpy(boards)
        .reshape(num_examples, 1, BOARD_HEIGHT, BOARD_WIDTH)
        .to(torch.bool),
    )
    assert torch.equal(mirror.aux[:num_examples], torch.from_numpy(aux))
    assert torch.equal(
        mirror.policy_targets[:num_examples], torch.from_numpy(policy_targets)
    )
    assert torch.equal(
        mirror.value_targets[:num_examples], torch.from_numpy(value_targets)
    )
    assert torch.equal(
        mirror.overhang_fields[:num_examples], torch.from_numpy(overhang_fields)
    )
    assert torch.equal(
        mirror.masks[:num_examples], torch.from_numpy(masks).to(torch.bool)
    )
