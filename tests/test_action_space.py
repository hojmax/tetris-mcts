from __future__ import annotations

import numpy as np
import tetris_core.tetris_core as tetris_core

from tetris_bot.action_space import (
    ACTION_TO_CANONICAL_CELL,
    ACTIVE_CANONICAL_ROTATION_COUNTS,
    HOLD_ACTION_INDEX,
    LEGACY_NUM_ACTIONS,
    LEGACY_TO_CANONICAL_ACTION_BY_PIECE,
    NUM_ACTIONS,
    NUM_PLACEMENT_ACTIONS,
    PIECE_VALID_ACTION_COUNTS,
    VALID_CANONICAL_ACTION_MASK_BY_PIECE,
    adapt_legacy_action_masks,
    adapt_legacy_policy_targets,
)
from tetris_bot.scripts.inspection.policy_grid_visualizer import FLAT_ACTION_CELLS


def test_canonical_action_space_counts() -> None:
    assert NUM_PLACEMENT_ACTIONS == 671
    assert HOLD_ACTION_INDEX == 671
    assert NUM_ACTIONS == 672
    assert ACTIVE_CANONICAL_ROTATION_COUNTS == (178, 179, 152, 162)
    assert PIECE_VALID_ACTION_COUNTS == (310, 171, 628, 314, 314, 628, 628)


def test_legacy_adapter_collapses_redundant_i_rotations() -> None:
    piece_type = 0  # I
    legacy_mapping = LEGACY_TO_CANONICAL_ACTION_BY_PIECE[piece_type]
    legacy_targets = legacy_mapping[legacy_mapping >= 0]
    duplicate_counts = np.bincount(legacy_targets, minlength=NUM_PLACEMENT_ACTIONS)
    target_action = int(np.flatnonzero(duplicate_counts == 2)[0])
    source_columns = np.flatnonzero(legacy_mapping == target_action)
    assert source_columns.shape == (2,)

    current_pieces = np.zeros((1, 7), dtype=np.float32)
    current_pieces[0, piece_type] = 1.0

    legacy_policy_targets = np.zeros((1, LEGACY_NUM_ACTIONS), dtype=np.float32)
    legacy_policy_targets[0, source_columns[0]] = 0.25
    legacy_policy_targets[0, source_columns[1]] = 0.75
    legacy_policy_targets[0, -1] = 0.5

    adapted_policy_targets = adapt_legacy_policy_targets(
        current_pieces=current_pieces,
        legacy_policy_targets=legacy_policy_targets,
    )
    assert adapted_policy_targets.shape == (1, NUM_ACTIONS)
    assert adapted_policy_targets[0, target_action] == 1.0
    assert adapted_policy_targets[0, HOLD_ACTION_INDEX] == 0.5

    legacy_action_masks = np.zeros((1, LEGACY_NUM_ACTIONS), dtype=np.bool_)
    legacy_action_masks[0, source_columns[1]] = True
    legacy_action_masks[0, -1] = True

    adapted_action_masks = adapt_legacy_action_masks(
        current_pieces=current_pieces,
        legacy_action_masks=legacy_action_masks,
    )
    assert adapted_action_masks.shape == (1, NUM_ACTIONS)
    assert adapted_action_masks[0, target_action]
    assert adapted_action_masks[0, HOLD_ACTION_INDEX]


def test_visualizer_flat_layout_matches_shared_action_space() -> None:
    grouped_cells = [[] for _ in range(4)]
    for rotation, grid_x, grid_y in ACTION_TO_CANONICAL_CELL:
        grouped_cells[rotation].append((grid_x, grid_y))

    assert grouped_cells == FLAT_ACTION_CELLS


def test_piece_valid_masks_only_cover_canonical_rotations() -> None:
    per_piece_rotation_counts: list[list[int]] = []
    for piece_type in range(VALID_CANONICAL_ACTION_MASK_BY_PIECE.shape[0]):
        counts = [0, 0, 0, 0]
        for action_index, (rotation, _, _) in enumerate(ACTION_TO_CANONICAL_CELL):
            if VALID_CANONICAL_ACTION_MASK_BY_PIECE[piece_type, action_index]:
                counts[rotation] += 1
        per_piece_rotation_counts.append(counts)

    assert per_piece_rotation_counts == [
        [140, 170, 0, 0],  # I uses only two canonical rotations
        [171, 0, 0, 0],  # O uses only rotation 0
        [152, 162, 152, 162],  # T uses all four
        [152, 162, 0, 0],  # S uses only two canonical rotations
        [152, 162, 0, 0],  # Z uses only two canonical rotations
        [152, 162, 152, 162],  # J uses all four
        [152, 162, 152, 162],  # L uses all four
    ]


def test_empty_board_runtime_action_mask_counts_by_piece() -> None:
    expected_rotation_counts = [
        [7, 10, 0, 0],  # I
        [9, 0, 0, 0],  # O
        [8, 9, 8, 9],  # T
        [8, 9, 0, 0],  # S
        [8, 9, 0, 0],  # Z
        [8, 9, 8, 9],  # J
        [8, 9, 8, 9],  # L
    ]

    for piece_type, expected_counts in enumerate(expected_rotation_counts):
        env = tetris_core.TetrisEnv()
        env.set_current_piece_type(piece_type)
        action_mask = np.asarray(tetris_core.debug_get_action_mask(env), dtype=np.bool_)

        assert action_mask.shape == (NUM_ACTIONS,)
        assert action_mask[HOLD_ACTION_INDEX]

        actual_counts = [0, 0, 0, 0]
        for action_index, (rotation, _, _) in enumerate(ACTION_TO_CANONICAL_CELL):
            if action_mask[action_index]:
                actual_counts[rotation] += 1

        assert actual_counts == expected_counts
        assert int(action_mask[:NUM_PLACEMENT_ACTIONS].sum()) == sum(expected_counts)
        assert int(action_mask.sum()) == sum(expected_counts) + 1
