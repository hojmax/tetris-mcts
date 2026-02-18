from __future__ import annotations

from dataclasses import dataclass

from tetris.ml.network import (
    AUX_FEATURES,
    BACK_TO_BACK_FEATURES,
    BUMPINESS_FEATURES,
    COLUMN_HEIGHT_FEATURES,
    COMBO_FEATURES,
    CURRENT_PIECE_FEATURES,
    HIDDEN_PIECE_DISTRIBUTION_FEATURES,
    HOLD_AVAILABLE_FEATURES,
    HOLD_PIECE_FEATURES,
    HOLES_FEATURES,
    MAX_COLUMN_HEIGHT_FEATURES,
    MIN_COLUMN_HEIGHT_FEATURES,
    MOVE_NUMBER_FEATURES,
    OVERHANG_FIELDS_FEATURES,
    QUEUE_FEATURES,
    ROW_FILL_COUNT_FEATURES,
    TOTAL_BLOCKS_FEATURES,
)


@dataclass(frozen=True)
class AuxFeatureLayout:
    column_heights: slice
    max_column_height: int
    min_column_height: int
    row_fill_counts: slice
    total_blocks: int
    bumpiness: int
    holes: int
    overhang_fields: int


def build_aux_feature_layout() -> AuxFeatureLayout:
    aux_idx = 0
    aux_idx += CURRENT_PIECE_FEATURES
    aux_idx += HOLD_PIECE_FEATURES
    aux_idx += HOLD_AVAILABLE_FEATURES
    aux_idx += QUEUE_FEATURES
    aux_idx += MOVE_NUMBER_FEATURES
    aux_idx += COMBO_FEATURES
    aux_idx += BACK_TO_BACK_FEATURES
    aux_idx += HIDDEN_PIECE_DISTRIBUTION_FEATURES

    column_heights = slice(aux_idx, aux_idx + COLUMN_HEIGHT_FEATURES)
    aux_idx += COLUMN_HEIGHT_FEATURES
    max_column_height = aux_idx
    aux_idx += MAX_COLUMN_HEIGHT_FEATURES
    min_column_height = aux_idx
    aux_idx += MIN_COLUMN_HEIGHT_FEATURES
    row_fill_counts = slice(aux_idx, aux_idx + ROW_FILL_COUNT_FEATURES)
    aux_idx += ROW_FILL_COUNT_FEATURES
    total_blocks = aux_idx
    aux_idx += TOTAL_BLOCKS_FEATURES
    bumpiness = aux_idx
    aux_idx += BUMPINESS_FEATURES
    holes = aux_idx
    aux_idx += HOLES_FEATURES
    overhang_fields = aux_idx
    aux_idx += OVERHANG_FIELDS_FEATURES

    if aux_idx != AUX_FEATURES:
        raise ValueError(
            f"Aux feature layout mismatch: computed {aux_idx}, expected {AUX_FEATURES}"
        )

    return AuxFeatureLayout(
        column_heights=column_heights,
        max_column_height=max_column_height,
        min_column_height=min_column_height,
        row_fill_counts=row_fill_counts,
        total_blocks=total_blocks,
        bumpiness=bumpiness,
        holes=holes,
        overhang_fields=overhang_fields,
    )


AUX_FEATURE_LAYOUT = build_aux_feature_layout()
