from __future__ import annotations

from dataclasses import dataclass

import torch

from tetris_ml.constants import BOARD_HEIGHT, BOARD_WIDTH, NUM_ACTIONS
from tetris_ml.ml.network import AUX_FEATURES


@dataclass
class TrainingBatch:
    boards: torch.Tensor
    aux: torch.Tensor
    policy_targets: torch.Tensor
    value_targets: torch.Tensor
    overhang_fields: torch.Tensor
    masks: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.boards.shape[0])

    @property
    def device(self) -> torch.device:
        return self.boards.device

    def split(self, batch_size: int) -> list[TrainingBatch]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0 (got {batch_size})")
        batches: list[TrainingBatch] = []
        for start in range(0, self.size, batch_size):
            end = min(start + batch_size, self.size)
            batches.append(
                TrainingBatch(
                    boards=self.boards[start:end],
                    aux=self.aux[start:end],
                    policy_targets=self.policy_targets[start:end],
                    value_targets=self.value_targets[start:end],
                    overhang_fields=self.overhang_fields[start:end],
                    masks=self.masks[start:end],
                )
            )
        if not batches:
            raise ValueError("Cannot split empty staged batch")
        return batches


class CircularReplayMirror:
    """Pre-allocated circular buffer for device-resident replay mirror.

    All tensors are allocated once at full capacity. Incremental updates
    use in-place copy_() to avoid any new GPU allocations.
    """

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.boards = torch.zeros(capacity, 1, BOARD_HEIGHT, BOARD_WIDTH, device=device)
        self.aux = torch.zeros(capacity, AUX_FEATURES, device=device)
        self.policy_targets = torch.zeros(capacity, NUM_ACTIONS, device=device)
        self.value_targets = torch.zeros(capacity, device=device)
        self.overhang_fields = torch.zeros(capacity, device=device)
        self.masks = torch.zeros(capacity, NUM_ACTIONS, device=device)

        self.capacity = capacity
        self.count = 0
        self.write_pos = 0
        self.logical_end = 0

    @property
    def size(self) -> int:
        return self.count
