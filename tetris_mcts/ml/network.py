"""
Neural Network Architecture for Tetris AlphaZero

Input Representation:
- Board state: 20 x 10 binary (1 = filled, 0 = empty)
- Current piece: 7 (one-hot)
- Hold piece: 8 (one-hot, 7 pieces + empty)
- Hold available: 1 (binary)
- Next queue: 5 x 7 = 35 (one-hot per slot)
- Move number: 1 (normalized: move_idx / 100)

Total input: 200 + 7 + 8 + 1 + 35 + 1 = 252 features

Output:
- Policy head: 734 outputs (softmax over actions)
- Value head: 1 output (predicted cumulative attack)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

# Constants
BOARD_HEIGHT = 20
BOARD_WIDTH = 10
NUM_PIECE_TYPES = 7
QUEUE_SIZE = 5
MAX_MOVES = 100  # Maximum moves for move number normalization

# Total number of valid placement actions
# Covers x in [-3, 9], y in [-2, 19], rotation in [0, 3]
# Action space logic implemented in Rust (tetris_core/src/mcts/action_space.rs)
NUM_ACTIONS = 734

# Input feature sizes
BOARD_FEATURES = BOARD_HEIGHT * BOARD_WIDTH  # 200
CURRENT_PIECE_FEATURES = NUM_PIECE_TYPES  # 7
HOLD_PIECE_FEATURES = NUM_PIECE_TYPES + 1  # 8 (7 pieces + empty)
HOLD_AVAILABLE_FEATURES = 1
QUEUE_FEATURES = QUEUE_SIZE * NUM_PIECE_TYPES  # 35
MOVE_NUMBER_FEATURES = 1

AUX_FEATURES = (
    CURRENT_PIECE_FEATURES
    + HOLD_PIECE_FEATURES
    + HOLD_AVAILABLE_FEATURES
    + QUEUE_FEATURES
    + MOVE_NUMBER_FEATURES
)  # 52

TOTAL_FEATURES = BOARD_FEATURES + AUX_FEATURES  # 252


class TetrisNet(nn.Module):
    """
    AlphaZero-style network for Tetris.

    Architecture:
    - Conv layers process the board (20x10x1)
    - Aux features concatenated after flattening
    - Shared FC layer with LayerNorm
    - Separate policy and value heads
    """

    def __init__(
        self,
        conv_filters: list[int] = [2, 4],
        fc_hidden: int = 64,
        conv_kernel_size: int = 3,
        conv_padding: int = 1,
        num_actions: int = NUM_ACTIONS,
    ):
        super().__init__()

        self.num_actions = num_actions

        # Convolutional layers for board
        self.conv1 = nn.Conv2d(
            1, conv_filters[0], kernel_size=conv_kernel_size, padding=conv_padding
        )
        self.bn1 = nn.BatchNorm2d(conv_filters[0])
        self.conv2 = nn.Conv2d(
            conv_filters[0],
            conv_filters[1],
            kernel_size=conv_kernel_size,
            padding=conv_padding,
        )
        self.bn2 = nn.BatchNorm2d(conv_filters[1])

        # Flattened conv output size: 20 * 10 * 8 = 1,600
        conv_flat_size = BOARD_HEIGHT * BOARD_WIDTH * conv_filters[1]

        # Fully connected layer
        self.fc1 = nn.Linear(conv_flat_size + AUX_FEATURES, fc_hidden)
        self.ln1 = nn.LayerNorm(fc_hidden)

        # Policy head
        self.policy_head = nn.Linear(fc_hidden, num_actions)

        # Value head
        self.value_head = nn.Linear(fc_hidden, 1)

    def forward(
        self,
        board: torch.Tensor,
        aux_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            board: Shape (batch, 1, 20, 10) - binary board state
            aux_features: Shape (batch, 52) - auxiliary features

        Returns:
            policy_logits: Shape (batch, 734) - raw logits (caller should apply
                action mask before softmax to mask invalid actions)
            value: Shape (batch, 1) - predicted cumulative attack
        """
        # Conv layers
        x = F.relu(self.bn1(self.conv1(board)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Flatten conv output
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # (batch, 1600)

        # Concatenate with auxiliary features
        x = torch.cat([x, aux_features], dim=1)  # (batch, 1652)

        # Shared FC layer
        x = F.relu(self.ln1(self.fc1(x)))

        # Policy and value heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value

    def predict(
        self,
        board: torch.Tensor,
        aux_features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get policy probabilities and value.

        Args:
            board: Shape (batch, 1, 20, 10)
            aux_features: Shape (batch, 52)
            action_mask: Shape (batch, 734), 1 = valid, 0 = invalid

        Returns:
            policy: Shape (batch, 734) - softmax probabilities
            value: Shape (batch, 1) - predicted value
        """
        policy_logits, value = self.forward(board, aux_features)

        if action_mask is not None:
            # Mask invalid actions before softmax
            policy_logits = policy_logits.masked_fill(action_mask == 0, float("-inf"))

        policy = F.softmax(policy_logits, dim=-1)
        return policy, value


def encode_state(
    board: np.ndarray,
    current_piece: int,
    hold_piece: Optional[int],
    hold_available: bool,
    next_queue: list[int],
    move_number: int,
    max_moves: int = MAX_MOVES,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode game state into neural network input format.

    Args:
        board: 2D array (20, 10), 1 = filled, 0 = empty
        current_piece: Integer 0-6
        hold_piece: Integer 0-6 or None
        hold_available: Whether hold can be used this turn
        next_queue: List of piece types (0-6), up to 5 elements
        move_number: Current move number (0-indexed)
        max_moves: Maximum moves for normalization

    Returns:
        board_tensor: Shape (1, 20, 10) for CNN
        aux_tensor: Shape (52,) auxiliary features

    Raises:
        ValueError: If board dimensions are incorrect
    """
    # Validate board dimensions
    if board.shape != (BOARD_HEIGHT, BOARD_WIDTH):
        raise ValueError(
            f"Board shape must be ({BOARD_HEIGHT}, {BOARD_WIDTH}), got {board.shape}"
        )

    # Board: (1, 20, 10)
    board_tensor = board.astype(np.float32).reshape(1, BOARD_HEIGHT, BOARD_WIDTH)

    # Build auxiliary features
    aux = []

    # Current piece: one-hot (7)
    current_onehot = np.zeros(NUM_PIECE_TYPES, dtype=np.float32)
    current_onehot[current_piece] = 1.0
    aux.append(current_onehot)

    # Hold piece: one-hot (8) - 7 pieces + empty
    hold_onehot = np.zeros(HOLD_PIECE_FEATURES, dtype=np.float32)
    if hold_piece is not None:
        hold_onehot[hold_piece] = 1.0
    else:
        hold_onehot[NUM_PIECE_TYPES] = 1.0  # Empty slot
    aux.append(hold_onehot)

    # Hold available: binary (1)
    aux.append(np.array([1.0 if hold_available else 0.0], dtype=np.float32))

    # Next queue: one-hot per slot (5 x 7 = 35)
    queue_features = np.zeros((QUEUE_SIZE, NUM_PIECE_TYPES), dtype=np.float32)
    for i, piece in enumerate(next_queue[:QUEUE_SIZE]):
        queue_features[i, piece] = 1.0
    aux.append(queue_features.flatten())

    # Move number: normalized (1)
    normalized_move = move_number / max_moves
    aux.append(np.array([normalized_move], dtype=np.float32))

    aux_tensor = np.concatenate(aux)
    assert aux_tensor.shape == (AUX_FEATURES,), (
        f"Expected {AUX_FEATURES}, got {aux_tensor.shape}"
    )

    return board_tensor, aux_tensor


def encode_batch(states: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode a batch of states into tensors.

    Args:
        states: List of state dicts with keys:
            - board: (20, 10) array
            - current_piece: int
            - hold_piece: int or None
            - hold_available: bool
            - next_queue: list of ints
            - move_number: int
            - action_mask: (734,) array

    Returns:
        boards: (batch, 1, 20, 10)
        aux: (batch, 52)
        masks: (batch, 734)
    """
    boards = []
    aux_features = []
    masks = []

    for state in states:
        board_t, aux_t = encode_state(
            board=state["board"],
            current_piece=state["current_piece"],
            hold_piece=state.get("hold_piece"),
            hold_available=state.get("hold_available", True),
            next_queue=state.get("next_queue", []),
            move_number=state.get("move_number", 0),
        )
        boards.append(board_t)
        aux_features.append(aux_t)
        masks.append(state.get("action_mask", np.ones(NUM_ACTIONS, dtype=np.float32)))

    return (
        torch.tensor(np.stack(boards)),
        torch.tensor(np.stack(aux_features)),
        torch.tensor(np.stack(masks)),
    )


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
