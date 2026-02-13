"""
Neural network architecture for Tetris AlphaZero.

Input representation:
- Board state: 20 x 10 binary (1 = filled, 0 = empty)
- Current piece: 7 (one-hot)
- Hold piece: 8 (one-hot, 7 pieces + empty)
- Hold available: 1 (binary)
- Next queue: 5 x 7 = 35 (one-hot per slot)
- Placement count: 1 (normalized: placements_so_far / max_placements)
- Combo: 1 (normalized, capped)
- Back-to-back: 1 (binary)
- Next hidden piece distribution: 7 (7-bag probabilities)

Total input: 200 board + 61 aux = 261 features.

Output:
- Policy head: 735 outputs (734 placements + hold)
- Value head: 1 output (scalar target)
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
COMBO_FEATURES = 1
BACK_TO_BACK_FEATURES = 1
HIDDEN_PIECE_DISTRIBUTION_FEATURES = NUM_PIECE_TYPES  # 7

AUX_FEATURES = (
    CURRENT_PIECE_FEATURES
    + HOLD_PIECE_FEATURES
    + HOLD_AVAILABLE_FEATURES
    + QUEUE_FEATURES
    + MOVE_NUMBER_FEATURES
    + COMBO_FEATURES
    + BACK_TO_BACK_FEATURES
    + HIDDEN_PIECE_DISTRIBUTION_FEATURES
)  # 61


class TetrisNet(nn.Module):
    """Gated-fusion model with cached board embedding support."""

    def __init__(
        self,
        conv_filters: Sequence[int],
        fc_hidden: int,
        conv_kernel_size: int,
        conv_padding: int,
        aux_hidden: int = 24,
        num_fusion_blocks: int = 0,
    ):
        super().__init__()

        self.num_actions = NUM_ACTIONS
        if len(conv_filters) != 2:
            raise ValueError(
                f"Expected exactly 2 convolutional filters, got {len(conv_filters)}"
            )
        conv0 = conv_filters[0]
        conv1 = conv_filters[1]

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

        if num_fusion_blocks < 0:
            raise ValueError("num_fusion_blocks must be >= 0")

        conv_flat_size = BOARD_HEIGHT * BOARD_WIDTH * conv1
        fusion_hidden = fc_hidden

        # Board-only projection cached by Rust inference.
        self.board_proj = nn.Linear(conv_flat_size, fusion_hidden)

        # Aux-conditioned modulation.
        self.aux_fc = nn.Linear(AUX_FEATURES, aux_hidden)
        self.aux_ln = nn.LayerNorm(aux_hidden)
        self.gate_fc = nn.Linear(aux_hidden, fusion_hidden)
        self.aux_proj = nn.Linear(aux_hidden, fusion_hidden)

        # Post-fusion processing.
        self.fusion_ln = nn.LayerNorm(fusion_hidden)
        self.fusion_blocks = nn.ModuleList(
            [ResidualFusionBlock(fusion_hidden) for _ in range(num_fusion_blocks)]
        )

        self.policy_head = nn.Linear(fusion_hidden, NUM_ACTIONS)
        self.value_head = nn.Linear(fusion_hidden, 1)

    def forward_from_board_embedding(
        self, board_h: torch.Tensor, aux_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        aux_h = F.relu(self.aux_ln(self.aux_fc(aux_features)))
        gate = torch.sigmoid(self.gate_fc(aux_h))

        fused = board_h * (1.0 + gate) + self.aux_proj(aux_h)
        fused = F.relu(self.fusion_ln(fused))
        for block in self.fusion_blocks:
            fused = block(fused)

        policy_logits = self.policy_head(fused)
        value = self.value_head(fused)
        return policy_logits, value

    def forward(
        self,
        board: torch.Tensor,
        aux_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            board: Shape (batch, 1, 20, 10) - binary board state
            aux_features: Shape (batch, 61) - auxiliary features

        Returns:
            policy_logits: Shape (batch, 735) - raw logits (caller should apply
                action mask before softmax to mask invalid actions)
            value: Shape (batch, 1) - predicted scalar value target
        """
        x = F.relu(self.bn1(self.conv1(board)))
        x = F.relu(self.bn2(self.conv2(x)))
        board_h = self.board_proj(x.view(x.size(0), -1))
        return self.forward_from_board_embedding(board_h, aux_features)


class ResidualFusionBlock(nn.Module):
    """Residual MLP block used after board/aux fusion."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.ln1(x))
        h = self.fc1(h)
        h = F.relu(self.ln2(h))
        h = self.fc2(h)
        return x + h


class ConvBackbone(nn.Module):
    """Extracts conv layers from TetrisNet: board -> flattened conv features."""

    def __init__(self, parent: TetrisNet):
        super().__init__()
        self.conv1 = parent.conv1
        self.bn1 = parent.bn1
        self.conv2 = parent.conv2
        self.bn2 = parent.bn2

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(board)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x.view(x.size(0), -1)


class HeadsModel(nn.Module):
    """Extracts aux fusion + heads from TetrisNet: board_h + aux -> policy + value."""

    def __init__(self, parent: TetrisNet):
        super().__init__()
        self.aux_fc = parent.aux_fc
        self.aux_ln = parent.aux_ln
        self.gate_fc = parent.gate_fc
        self.aux_proj = parent.aux_proj
        self.fusion_ln = parent.fusion_ln
        self.fusion_blocks = parent.fusion_blocks
        self.policy_head = parent.policy_head
        self.value_head = parent.value_head

    def forward(
        self, board_h: torch.Tensor, aux_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        aux_h = F.relu(self.aux_ln(self.aux_fc(aux_features)))
        gate = torch.sigmoid(self.gate_fc(aux_h))
        fused = board_h * (1.0 + gate) + self.aux_proj(aux_h)
        fused = F.relu(self.fusion_ln(fused))
        for block in self.fusion_blocks:
            fused = block(fused)
        return self.policy_head(fused), self.value_head(fused)
