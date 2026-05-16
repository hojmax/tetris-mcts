from __future__ import annotations

from dataclasses import dataclass

from tetris_bot.ml.network import (
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
    MOVE_NUMBER_FEATURES,
    OVERHANG_FIELDS_FEATURES,
    QUEUE_FEATURES,
    ROW_FILL_COUNT_FEATURES,
    TOTAL_BLOCKS_FEATURES,
)


@dataclass(frozen=True)
class AuxFeatureLayout:
    current_piece: slice
    hold_piece: slice
    hold_available: int
    next_queue: slice
    placement_count: int
    combo: int
    back_to_back: int
    next_hidden_piece_probs: slice
    column_heights: slice
    max_column_height: int
    row_fill_counts: slice
    total_blocks: int
    bumpiness: int
    holes: int
    overhang_fields: int


def build_aux_feature_layout() -> AuxFeatureLayout:
    aux_idx = 0
    current_piece = slice(aux_idx, aux_idx + CURRENT_PIECE_FEATURES)
    aux_idx += CURRENT_PIECE_FEATURES
    hold_piece = slice(aux_idx, aux_idx + HOLD_PIECE_FEATURES)
    aux_idx += HOLD_PIECE_FEATURES
    hold_available = aux_idx
    aux_idx += HOLD_AVAILABLE_FEATURES
    next_queue = slice(aux_idx, aux_idx + QUEUE_FEATURES)
    aux_idx += QUEUE_FEATURES
    placement_count = aux_idx
    aux_idx += MOVE_NUMBER_FEATURES
    combo = aux_idx
    aux_idx += COMBO_FEATURES
    back_to_back = aux_idx
    aux_idx += BACK_TO_BACK_FEATURES
    next_hidden_piece_probs = slice(
        aux_idx, aux_idx + HIDDEN_PIECE_DISTRIBUTION_FEATURES
    )
    aux_idx += HIDDEN_PIECE_DISTRIBUTION_FEATURES

    column_heights = slice(aux_idx, aux_idx + COLUMN_HEIGHT_FEATURES)
    aux_idx += COLUMN_HEIGHT_FEATURES
    max_column_height = aux_idx
    aux_idx += MAX_COLUMN_HEIGHT_FEATURES
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
        current_piece=current_piece,
        hold_piece=hold_piece,
        hold_available=hold_available,
        next_queue=next_queue,
        placement_count=placement_count,
        combo=combo,
        back_to_back=back_to_back,
        next_hidden_piece_probs=next_hidden_piece_probs,
        column_heights=column_heights,
        max_column_height=max_column_height,
        row_fill_counts=row_fill_counts,
        total_blocks=total_blocks,
        bumpiness=bumpiness,
        holes=holes,
        overhang_fields=overhang_fields,
    )


AUX_FEATURE_LAYOUT = build_aux_feature_layout()
