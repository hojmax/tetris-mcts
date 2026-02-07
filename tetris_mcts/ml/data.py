"""
Data Serialization and Dataset Management for Tetris AlphaZero

Handles:
- Saving/loading training data in NPZ format
- PyTorch Dataset for training
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from tetris_mcts.config import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    NUM_ACTIONS,
    NUM_PIECE_TYPES,
    QUEUE_SIZE,
    TrainingConfig,
)

MAX_MOVES = TrainingConfig().max_moves


@dataclass
class TrainingExample:
    """Single training example from self-play."""

    board: np.ndarray  # (20, 10) bool
    current_piece: int  # 0-6
    hold_piece: Optional[int]  # 0-6 or None
    hold_available: bool
    next_queue: list[int]  # List of piece types
    move_number: int
    policy_target: np.ndarray  # (734,) float32 - normalized policy probabilities
    value_target: float  # Cumulative lines cleared
    action_mask: np.ndarray  # (734,) bool


def save_training_data(
    examples: list[TrainingExample],
    filepath: str | Path,
) -> None:
    """
    Save training examples to NPZ format.

    Format:
        boards: (N, 20, 10) bool
        current_pieces: (N, 7) float32 one-hot
        hold_pieces: (N, 8) float32 one-hot + empty
        hold_available: (N,) bool
        next_queue: (N, 5, 7) float32 one-hot
        move_numbers: (N,) float32 normalized
        policy_targets: (N, 734) float32
        value_targets: (N,) float32
        action_masks: (N, 734) bool
    """
    n = len(examples)
    if n == 0:
        return

    # Allocate arrays
    boards = np.zeros((n, BOARD_HEIGHT, BOARD_WIDTH), dtype=bool)
    current_pieces = np.zeros((n, NUM_PIECE_TYPES), dtype=np.float32)
    hold_pieces = np.zeros((n, NUM_PIECE_TYPES + 1), dtype=np.float32)
    hold_available = np.zeros(n, dtype=bool)
    next_queue = np.zeros((n, QUEUE_SIZE, NUM_PIECE_TYPES), dtype=np.float32)
    move_numbers = np.zeros(n, dtype=np.float32)
    policy_targets = np.zeros((n, NUM_ACTIONS), dtype=np.float32)
    value_targets = np.zeros(n, dtype=np.float32)
    action_masks = np.zeros((n, NUM_ACTIONS), dtype=bool)

    for i, ex in enumerate(examples):
        boards[i] = ex.board

        # One-hot current piece
        current_pieces[i, ex.current_piece] = 1.0

        # One-hot hold piece
        if ex.hold_piece is not None:
            hold_pieces[i, ex.hold_piece] = 1.0
        else:
            hold_pieces[i, NUM_PIECE_TYPES] = 1.0  # Empty

        hold_available[i] = ex.hold_available

        # One-hot next queue
        for j, piece in enumerate(ex.next_queue[:QUEUE_SIZE]):
            next_queue[i, j, piece] = 1.0

        move_numbers[i] = ex.move_number / MAX_MOVES  # Normalize
        policy_targets[i] = ex.policy_target
        value_targets[i] = ex.value_target
        action_masks[i] = ex.action_mask

    np.savez_compressed(
        filepath,
        boards=boards,
        current_pieces=current_pieces,
        hold_pieces=hold_pieces,
        hold_available=hold_available,
        next_queue=next_queue,
        move_numbers=move_numbers,
        policy_targets=policy_targets,
        value_targets=value_targets,
        action_masks=action_masks,
    )


def load_training_data(filepath: str | Path) -> list[TrainingExample]:
    """Load training examples from NPZ format."""
    data = np.load(filepath)

    n = len(data["boards"])
    examples = []

    for i in range(n):
        # Decode one-hot current piece
        current_piece = int(np.argmax(data["current_pieces"][i]))

        # Decode one-hot hold piece
        hold_idx = int(np.argmax(data["hold_pieces"][i]))
        hold_piece = hold_idx if hold_idx < NUM_PIECE_TYPES else None

        # Decode one-hot next queue
        next_queue = []
        for j in range(QUEUE_SIZE):
            if np.any(data["next_queue"][i, j] > 0):
                next_queue.append(int(np.argmax(data["next_queue"][i, j])))

        examples.append(
            TrainingExample(
                board=data["boards"][i].astype(bool),
                current_piece=current_piece,
                hold_piece=hold_piece,
                hold_available=bool(data["hold_available"][i]),
                next_queue=next_queue,
                move_number=int(data["move_numbers"][i] * MAX_MOVES),
                policy_target=data["policy_targets"][i],
                value_target=float(data["value_targets"][i]),
                action_mask=data["action_masks"][i].astype(bool),
            )
        )

    return examples


class TetrisDataset(Dataset):
    """PyTorch Dataset for training."""

    def __init__(self, filepath: str | Path):
        """Load dataset from NPZ file."""
        data = np.load(filepath)

        self.boards = torch.tensor(data["boards"].astype(np.float32)).unsqueeze(
            1
        )  # (N, 1, 20, 10)

        # Build auxiliary features
        n = len(data["boards"])
        aux = np.concatenate(
            [
                data["current_pieces"],  # (N, 7)
                data["hold_pieces"],  # (N, 8)
                data["hold_available"].reshape(-1, 1).astype(np.float32),  # (N, 1)
                data["next_queue"].reshape(n, -1),  # (N, 35)
                data["move_numbers"].reshape(-1, 1),  # (N, 1)
            ],
            axis=1,
        )
        self.aux_features = torch.tensor(aux.astype(np.float32))

        self.policy_targets = torch.tensor(data["policy_targets"])
        self.value_targets = torch.tensor(data["value_targets"])
        self.action_masks = torch.tensor(data["action_masks"].astype(np.float32))

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        return (
            self.boards[idx],
            self.aux_features[idx],
            self.policy_targets[idx],
            self.value_targets[idx],
            self.action_masks[idx],
        )
