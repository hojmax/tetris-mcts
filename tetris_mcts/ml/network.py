"""
Neural network architecture for Tetris AlphaZero.

Input representation:
- Board state: 20 x 10 binary (1 = filled, 0 = empty)

Piece/game auxiliary features (61, uncached heads path):
- Current piece: 7 (one-hot)
- Hold piece: 8 (one-hot, 7 pieces + empty)
- Hold available: 1 (binary)
- Next queue: 5 x 7 = 35 (one-hot per slot)
- Placement count: 1 (normalized: placements_so_far / max_placements)
- Combo: 1 (normalized, capped)
- Back-to-back: 1 (binary)
- Next hidden piece distribution: 7 (7-bag probabilities)

Board stats features (36, folded into cached board embedding):
- Column heights: 10 (normalized)
- Max column height: 1 (normalized)
- Min column height: 1 (normalized)
- Row fill counts: 20 (normalized)
- Total blocks: 1 (normalized)
- Bumpiness: 1 (normalized)
- Holes: 1 (normalized)
- Overhang fields: 1 (normalized)

Total input: 200 board + 97 aux = 297 features.
Training data packs all 97 aux features together; the model splits internally.
Rust inference encodes board stats separately for the cached board embedding path.

Output:
- Policy head: 735 outputs (734 placements + hold)
- Value head: 1 output (scalar target)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
COLUMN_HEIGHT_FEATURES = BOARD_WIDTH  # 10
MAX_COLUMN_HEIGHT_FEATURES = 1
MIN_COLUMN_HEIGHT_FEATURES = 1
ROW_FILL_COUNT_FEATURES = BOARD_HEIGHT  # 20
TOTAL_BLOCKS_FEATURES = 1
BUMPINESS_FEATURES = 1
HOLES_FEATURES = 1
OVERHANG_FIELDS_FEATURES = 1

PIECE_AUX_FEATURES = (
    CURRENT_PIECE_FEATURES
    + HOLD_PIECE_FEATURES
    + HOLD_AVAILABLE_FEATURES
    + QUEUE_FEATURES
    + MOVE_NUMBER_FEATURES
    + COMBO_FEATURES
    + BACK_TO_BACK_FEATURES
    + HIDDEN_PIECE_DISTRIBUTION_FEATURES
)  # 61

BOARD_STATS_FEATURES = (
    COLUMN_HEIGHT_FEATURES
    + MAX_COLUMN_HEIGHT_FEATURES
    + MIN_COLUMN_HEIGHT_FEATURES
    + ROW_FILL_COUNT_FEATURES
    + TOTAL_BLOCKS_FEATURES
    + BUMPINESS_FEATURES
    + HOLES_FEATURES
    + OVERHANG_FIELDS_FEATURES
)  # 36

AUX_FEATURES = PIECE_AUX_FEATURES + BOARD_STATS_FEATURES  # 97

COMBO_NORMALIZATION_MAX = 4.0


def normalize_combo_for_feature(combo: float) -> float:
    return min(max(combo, 0.0), COMBO_NORMALIZATION_MAX) / COMBO_NORMALIZATION_MAX


def build_aux_features(
    current_piece: np.ndarray,
    hold_piece: np.ndarray,
    hold_available: float,
    next_queue: np.ndarray,
    placement_count: float,
    combo_feature: float,
    back_to_back: float,
    next_hidden_piece_probs: np.ndarray,
    column_heights: np.ndarray,
    max_column_height: float,
    min_column_height: float,
    row_fill_counts: np.ndarray,
    total_blocks: float,
    bumpiness: float,
    holes: float,
    overhang_fields: float,
) -> np.ndarray:
    hold_available_feature = np.array([hold_available], dtype=np.float32)
    placement_count_feature = np.array([placement_count], dtype=np.float32)
    combo_feature_array = np.array([combo_feature], dtype=np.float32)
    back_to_back_feature = np.array([back_to_back], dtype=np.float32)
    max_column_height_feature = np.array([max_column_height], dtype=np.float32)
    min_column_height_feature = np.array([min_column_height], dtype=np.float32)
    total_blocks_feature = np.array([total_blocks], dtype=np.float32)
    bumpiness_feature = np.array([bumpiness], dtype=np.float32)
    holes_feature = np.array([holes], dtype=np.float32)
    overhang_fields_feature = np.array([overhang_fields], dtype=np.float32)
    return np.concatenate(
        [
            current_piece.astype(np.float32),
            hold_piece.astype(np.float32),
            hold_available_feature,
            next_queue.astype(np.float32).reshape(-1),
            placement_count_feature,
            combo_feature_array,
            back_to_back_feature,
            next_hidden_piece_probs.astype(np.float32).reshape(-1),
            column_heights.astype(np.float32).reshape(-1),
            max_column_height_feature,
            min_column_height_feature,
            row_fill_counts.astype(np.float32).reshape(-1),
            total_blocks_feature,
            bumpiness_feature,
            holes_feature,
            overhang_fields_feature,
        ]
    )


class ResidualConvBlock(nn.Module):
    """Pre-activation residual block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv + skip."""

    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(x))
        h = self.conv1(h)
        h = F.relu(self.bn2(h))
        h = self.conv2(h)
        return x + h


class TetrisNet(nn.Module):
    """Gated-fusion model with cached board embedding support."""

    def __init__(
        self,
        trunk_channels: int,
        num_conv_residual_blocks: int,
        reduction_channels: int,
        fc_hidden: int,
        conv_kernel_size: int,
        conv_padding: int,
        aux_hidden: int = 64,
        num_fusion_blocks: int = 0,
    ):
        super().__init__()

        self.num_actions = NUM_ACTIONS

        # Conv backbone: initial -> res blocks -> stride-2 reduction
        self.conv_initial = nn.Conv2d(
            1, trunk_channels, kernel_size=conv_kernel_size, padding=conv_padding
        )
        self.bn_initial = nn.BatchNorm2d(trunk_channels)
        self.res_blocks = nn.ModuleList(
            [
                ResidualConvBlock(
                    trunk_channels, kernel_size=conv_kernel_size, padding=conv_padding
                )
                for _ in range(num_conv_residual_blocks)
            ]
        )
        self.conv_reduce = nn.Conv2d(
            trunk_channels,
            reduction_channels,
            kernel_size=conv_kernel_size,
            padding=conv_padding,
            stride=2,
        )
        self.bn_reduce = nn.BatchNorm2d(reduction_channels)

        if num_fusion_blocks < 0:
            raise ValueError("num_fusion_blocks must be >= 0")

        # Compute reduced spatial dims: stride-2 halves each dimension (ceil division)
        reduced_h = (BOARD_HEIGHT + 1) // 2  # 10
        reduced_w = (BOARD_WIDTH + 1) // 2  # 5
        conv_flat_size = reduction_channels * reduced_h * reduced_w
        fusion_hidden = fc_hidden

        # Board projection: conv features + board stats, cached by Rust inference.
        self.board_proj = nn.Linear(
            conv_flat_size + BOARD_STATS_FEATURES, fusion_hidden
        )

        # Aux-conditioned modulation (piece/game features only).
        self.aux_fc = nn.Linear(PIECE_AUX_FEATURES, aux_hidden)
        self.aux_ln = nn.LayerNorm(aux_hidden)
        self.gate_fc = nn.Linear(aux_hidden, fusion_hidden)
        self.aux_proj = nn.Linear(aux_hidden, fusion_hidden)

        # Post-fusion processing.
        self.fusion_ln = nn.LayerNorm(fusion_hidden)
        self.fusion_blocks = nn.ModuleList(
            [ResidualFusionBlock(fusion_hidden) for _ in range(num_fusion_blocks)]
        )

        # 2-layer MLP heads
        policy_hidden = fusion_hidden * 2
        value_hidden = fusion_hidden // 2
        self.policy_fc = nn.Linear(fusion_hidden, policy_hidden)
        self.policy_head = nn.Linear(policy_hidden, NUM_ACTIONS)
        self.value_fc = nn.Linear(fusion_hidden, value_hidden)
        self.value_head = nn.Linear(value_hidden, 1)

    def forward_from_board_embedding(
        self, board_h: torch.Tensor, piece_aux: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        aux_h = F.relu(self.aux_ln(self.aux_fc(piece_aux)))
        gate = torch.sigmoid(self.gate_fc(aux_h))

        fused = board_h * (1.0 + gate) + self.aux_proj(aux_h)
        fused = F.relu(self.fusion_ln(fused))
        for block in self.fusion_blocks:
            fused = block(fused)

        policy_logits = self.policy_head(F.relu(self.policy_fc(fused)))
        value = self.value_head(F.relu(self.value_fc(fused)))
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
            aux_features: Shape (batch, 97) - auxiliary features (61 piece/game + 36 board stats)

        Returns:
            policy_logits: Shape (batch, 735) - raw logits (caller should apply
                action mask before softmax to mask invalid actions)
            value: Shape (batch, 1) - predicted scalar value target
        """
        piece_aux = aux_features[:, :PIECE_AUX_FEATURES]
        board_stats = aux_features[:, PIECE_AUX_FEATURES:]
        x = F.relu(self.bn_initial(self.conv_initial(board)))
        for block in self.res_blocks:
            x = block(x)
        x = F.relu(self.bn_reduce(self.conv_reduce(x)))
        board_h = self.board_proj(
            torch.cat([x.view(x.size(0), -1), board_stats], dim=1)
        )
        return self.forward_from_board_embedding(board_h, piece_aux)


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
        self.conv_initial = parent.conv_initial
        self.bn_initial = parent.bn_initial
        self.res_blocks = parent.res_blocks
        self.conv_reduce = parent.conv_reduce
        self.bn_reduce = parent.bn_reduce

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn_initial(self.conv_initial(board)))
        for block in self.res_blocks:
            x = block(x)
        x = F.relu(self.bn_reduce(self.conv_reduce(x)))
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
        self.policy_fc = parent.policy_fc
        self.policy_head = parent.policy_head
        self.value_fc = parent.value_fc
        self.value_head = parent.value_head

    def forward(
        self, board_h: torch.Tensor, piece_aux: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        aux_h = F.relu(self.aux_ln(self.aux_fc(piece_aux)))
        gate = torch.sigmoid(self.gate_fc(aux_h))
        fused = board_h * (1.0 + gate) + self.aux_proj(aux_h)
        fused = F.relu(self.fusion_ln(fused))
        for block in self.fusion_blocks:
            fused = block(fused)
        policy_logits = self.policy_head(F.relu(self.policy_fc(fused)))
        value = self.value_head(F.relu(self.value_fc(fused)))
        return policy_logits, value
