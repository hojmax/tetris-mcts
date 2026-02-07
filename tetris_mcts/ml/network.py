"""
Neural Network Architecture for Tetris AlphaZero

Input Representation:
- Board state: 20 x 10 binary (1 = filled, 0 = empty)
- Current piece: 7 (one-hot)
- Hold piece: 8 (one-hot, 7 pieces + empty)
- Hold available: 1 (binary)
- Next queue: 5 x 7 = 35 (one-hot per slot)
- Move number: 1 (normalized: move_idx / max_moves)

Total input: 200 + 7 + 8 + 1 + 35 + 1 = 252 features

Output:
- Policy head: 734 outputs (softmax over actions)
- Value head: 1 output (predicted cumulative attack)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Sequence

from tetris_mcts.config import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    NUM_ACTIONS,
    NUM_PIECE_TYPES,
    QUEUE_SIZE,
)

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
        conv_filters: Sequence[int],
        fc_hidden: int,
        conv_kernel_size: int,
        conv_padding: int,
    ):
        super().__init__()

        self.num_actions = NUM_ACTIONS
        if len(conv_filters) != 2:
            raise ValueError(
                f"Expected exactly 2 convolutional filters, got {len(conv_filters)}"
            )
        conv0 = conv_filters[0]
        conv1 = conv_filters[1]

        # Convolutional layers for board
        self.conv1 = nn.Conv2d(
            1, conv0, kernel_size=conv_kernel_size, padding=conv_padding
        )
        self.bn1 = nn.BatchNorm2d(conv0)
        self.conv2 = nn.Conv2d(
            conv0,
            conv1,
            kernel_size=conv_kernel_size,
            padding=conv_padding,
        )
        self.bn2 = nn.BatchNorm2d(conv1)

        # Flattened conv output size: 20 * 10 * conv1
        conv_flat_size = BOARD_HEIGHT * BOARD_WIDTH * conv1

        # Fully connected layer
        self.fc1 = nn.Linear(conv_flat_size + AUX_FEATURES, fc_hidden)
        self.ln1 = nn.LayerNorm(fc_hidden)

        # Policy head
        self.policy_head = nn.Linear(fc_hidden, NUM_ACTIONS)

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
