"""
Data Serialization and Dataset Management for Tetris AlphaZero

Handles:
- Saving/loading training data in NPZ format
- PyTorch Dataset for training
- Replay buffer management
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional
import os
import time
from dataclasses import dataclass

from .action_space import NUM_ACTIONS
from .network import (
    BOARD_HEIGHT, BOARD_WIDTH, NUM_PIECE_TYPES, QUEUE_SIZE,
    AUX_FEATURES, encode_state
)


@dataclass
class TrainingExample:
    """Single training example from self-play."""
    board: np.ndarray  # (20, 10) bool
    current_piece: int  # 0-6
    hold_piece: Optional[int]  # 0-6 or None
    hold_available: bool
    next_queue: list[int]  # List of piece types
    move_number: int
    policy_target: np.ndarray  # (734,) float32 - MCTS visit counts
    value_target: float  # Discounted cumulative attack
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

        move_numbers[i] = ex.move_number / 100.0  # Normalize
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

    n = len(data['boards'])
    examples = []

    for i in range(n):
        # Decode one-hot current piece
        current_piece = int(np.argmax(data['current_pieces'][i]))

        # Decode one-hot hold piece
        hold_idx = int(np.argmax(data['hold_pieces'][i]))
        hold_piece = hold_idx if hold_idx < NUM_PIECE_TYPES else None

        # Decode one-hot next queue
        next_queue = []
        for j in range(QUEUE_SIZE):
            if np.any(data['next_queue'][i, j] > 0):
                next_queue.append(int(np.argmax(data['next_queue'][i, j])))

        examples.append(TrainingExample(
            board=data['boards'][i].astype(bool),
            current_piece=current_piece,
            hold_piece=hold_piece,
            hold_available=bool(data['hold_available'][i]),
            next_queue=next_queue,
            move_number=int(data['move_numbers'][i] * 100),
            policy_target=data['policy_targets'][i],
            value_target=float(data['value_targets'][i]),
            action_mask=data['action_masks'][i].astype(bool),
        ))

    return examples


class TetrisDataset(Dataset):
    """PyTorch Dataset for training."""

    def __init__(self, filepath: str | Path):
        """Load dataset from NPZ file."""
        data = np.load(filepath)

        self.boards = torch.tensor(data['boards'].astype(np.float32)).unsqueeze(1)  # (N, 1, 20, 10)

        # Build auxiliary features
        n = len(data['boards'])
        aux = np.concatenate([
            data['current_pieces'],  # (N, 7)
            data['hold_pieces'],  # (N, 8)
            data['hold_available'].reshape(-1, 1).astype(np.float32),  # (N, 1)
            data['next_queue'].reshape(n, -1),  # (N, 35)
            data['move_numbers'].reshape(-1, 1),  # (N, 1)
        ], axis=1)
        self.aux_features = torch.tensor(aux.astype(np.float32))

        self.policy_targets = torch.tensor(data['policy_targets'])
        self.value_targets = torch.tensor(data['value_targets'])
        self.action_masks = torch.tensor(data['action_masks'].astype(np.float32))

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


class ReplayBuffer:
    """
    Circular replay buffer for training data.

    Stores raw arrays for efficient sampling.
    """

    def __init__(self, max_size: int = 100_000):
        self.max_size = max_size
        self.size = 0
        self.pos = 0

        # Pre-allocate arrays
        self.boards = np.zeros((max_size, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
        self.aux_features = np.zeros((max_size, AUX_FEATURES), dtype=np.float32)
        self.policy_targets = np.zeros((max_size, NUM_ACTIONS), dtype=np.float32)
        self.value_targets = np.zeros(max_size, dtype=np.float32)
        self.action_masks = np.zeros((max_size, NUM_ACTIONS), dtype=np.float32)

    def add(self, example: TrainingExample) -> None:
        """Add a single example to the buffer."""
        board_t, aux_t = encode_state(
            board=example.board,
            current_piece=example.current_piece,
            hold_piece=example.hold_piece,
            hold_available=example.hold_available,
            next_queue=example.next_queue,
            move_number=example.move_number,
        )

        self.boards[self.pos] = board_t.squeeze(0)
        self.aux_features[self.pos] = aux_t
        self.policy_targets[self.pos] = example.policy_target
        self.value_targets[self.pos] = example.value_target
        self.action_masks[self.pos] = example.action_mask.astype(np.float32)

        self.pos = (self.pos + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, examples: list[TrainingExample]) -> None:
        """Add multiple examples to the buffer."""
        for ex in examples:
            self.add(ex)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """
        Sample a random batch from the buffer.

        Returns:
            boards: (batch, 1, 20, 10)
            aux_features: (batch, 52)
            policy_targets: (batch, 734)
            value_targets: (batch,)
            action_masks: (batch, 734)
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.tensor(self.boards[indices]).unsqueeze(1),
            torch.tensor(self.aux_features[indices]),
            torch.tensor(self.policy_targets[indices]),
            torch.tensor(self.value_targets[indices]),
            torch.tensor(self.action_masks[indices]),
        )

    def __len__(self):
        return self.size

    def save(self, filepath: str | Path) -> None:
        """Save buffer contents to file."""
        np.savez_compressed(
            filepath,
            boards=self.boards[:self.size],
            aux_features=self.aux_features[:self.size],
            policy_targets=self.policy_targets[:self.size],
            value_targets=self.value_targets[:self.size],
            action_masks=self.action_masks[:self.size],
            pos=np.array([self.pos]),
        )

    def load(self, filepath: str | Path) -> None:
        """Load buffer contents from file."""
        data = np.load(filepath)
        n = len(data['boards'])

        self.boards[:n] = data['boards']
        self.aux_features[:n] = data['aux_features']
        self.policy_targets[:n] = data['policy_targets']
        self.value_targets[:n] = data['value_targets']
        self.action_masks[:n] = data['action_masks']
        self.size = n
        self.pos = int(data['pos'][0]) if n == self.max_size else n


class SharedReplayBuffer:
    """
    Replay buffer that reads from disk for multi-process training.

    Self-play process writes to data_dir/games_*.npz
    Training process reads and samples from them.
    """

    def __init__(self, data_dir: str | Path, max_files: int = 100):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_files = max_files
        self._cached_data = None
        self._cache_time = 0.0

    def add_game(self, examples: list[TrainingExample]) -> None:
        """Save a game's examples to a new file."""
        timestamp = int(time.time() * 1000)
        filepath = self.data_dir / f"game_{timestamp}.npz"
        save_training_data(examples, filepath)

        # Clean up old files if needed
        self._cleanup_old_files()

    def _cleanup_old_files(self) -> None:
        """Remove oldest files if over limit."""
        files = sorted(self.data_dir.glob("game_*.npz"))
        while len(files) > self.max_files:
            oldest = files.pop(0)
            oldest.unlink()

    def _refresh_cache(self) -> None:
        """Reload data from all files."""
        files = list(self.data_dir.glob("game_*.npz"))
        if not files:
            self._cached_data = None
            return

        all_data = {
            'boards': [],
            'aux_features': [],
            'policy_targets': [],
            'value_targets': [],
            'action_masks': [],
        }

        for f in files:
            try:
                data = np.load(f)

                # Build aux features from components
                n = len(data['boards'])
                aux = np.concatenate([
                    data['current_pieces'],
                    data['hold_pieces'],
                    data['hold_available'].reshape(-1, 1).astype(np.float32),
                    data['next_queue'].reshape(n, -1),
                    data['move_numbers'].reshape(-1, 1),
                ], axis=1)

                all_data['boards'].append(data['boards'].astype(np.float32))
                all_data['aux_features'].append(aux.astype(np.float32))
                all_data['policy_targets'].append(data['policy_targets'])
                all_data['value_targets'].append(data['value_targets'])
                all_data['action_masks'].append(data['action_masks'].astype(np.float32))
            except Exception as e:
                print(f"Warning: Failed to load {f}: {e}")

        if all_data['boards']:
            self._cached_data = {
                k: np.concatenate(v) for k, v in all_data.items()
            }
        else:
            self._cached_data = None

        self._cache_time = time.time()

    def sample(self, batch_size: int, cache_ttl: float = 30.0) -> Optional[tuple[torch.Tensor, ...]]:
        """
        Sample a batch from the buffer.

        Args:
            batch_size: Number of examples to sample
            cache_ttl: Seconds before refreshing the cache

        Returns:
            Tuple of tensors or None if buffer is empty
        """
        # Refresh cache if stale
        if self._cached_data is None or time.time() - self._cache_time > cache_ttl:
            self._refresh_cache()

        if self._cached_data is None:
            return None

        n = len(self._cached_data['boards'])
        if n == 0:
            return None

        indices = np.random.randint(0, n, size=min(batch_size, n))

        return (
            torch.tensor(self._cached_data['boards'][indices]).unsqueeze(1),
            torch.tensor(self._cached_data['aux_features'][indices]),
            torch.tensor(self._cached_data['policy_targets'][indices]),
            torch.tensor(self._cached_data['value_targets'][indices]),
            torch.tensor(self._cached_data['action_masks'][indices]),
        )

    def size(self) -> int:
        """Return approximate number of examples in buffer."""
        if self._cached_data is None:
            self._refresh_cache()
        if self._cached_data is None:
            return 0
        return len(self._cached_data['boards'])


if __name__ == "__main__":
    import tempfile

    # Test save/load
    print("Testing save/load...")
    examples = []
    for i in range(10):
        board = np.random.randint(0, 2, (BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8)
        examples.append(TrainingExample(
            board=board.astype(bool),
            current_piece=i % 7,
            hold_piece=(i + 1) % 7 if i % 2 == 0 else None,
            hold_available=i % 3 != 0,
            next_queue=[j % 7 for j in range(5)],
            move_number=i * 10,
            policy_target=np.random.dirichlet(np.ones(NUM_ACTIONS)).astype(np.float32),
            value_target=float(i),
            action_mask=np.random.randint(0, 2, NUM_ACTIONS).astype(bool),
        ))

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        filepath = f.name

    save_training_data(examples, filepath)
    loaded = load_training_data(filepath)

    assert len(loaded) == len(examples)
    print(f"Saved and loaded {len(loaded)} examples")

    # Test Dataset
    print("\nTesting Dataset...")
    dataset = TetrisDataset(filepath)
    print(f"Dataset size: {len(dataset)}")

    board, aux, policy, value, mask = dataset[0]
    print(f"Board shape: {board.shape}")
    print(f"Aux shape: {aux.shape}")
    print(f"Policy shape: {policy.shape}")
    print(f"Value: {value}")
    print(f"Mask shape: {mask.shape}")

    # Test DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print(f"\nBatch shapes: {[x.shape for x in batch]}")

    # Test ReplayBuffer
    print("\nTesting ReplayBuffer...")
    buffer = ReplayBuffer(max_size=100)
    buffer.add_batch(examples)
    print(f"Buffer size: {len(buffer)}")

    sample = buffer.sample(4)
    print(f"Sample shapes: {[x.shape for x in sample]}")

    # Clean up
    os.unlink(filepath)
    print("\nAll tests passed!")
