from __future__ import annotations

import numpy as np

from tetris_bot.constants import BOARD_HEIGHT, BOARD_WIDTH, NUM_PIECE_TYPES

ROTATION_LABELS = ("0", "R", "2", "L")
PIECES_TWO_ROTATIONS = frozenset({0, 3, 4})  # I, S, Z
PIECES_ONE_ROTATIONS = frozenset({1})  # O

X_MIN = -3
X_MAX_EXCLUSIVE = 10
Y_MIN = -3
Y_MAX_EXCLUSIVE = 20
X_RANGE = X_MAX_EXCLUSIVE - X_MIN
Y_RANGE = Y_MAX_EXCLUSIVE - Y_MIN
FULL_GRID_PLACEMENT_SLOTS = len(ROTATION_LABELS) * BOARD_HEIGHT * BOARD_WIDTH

LEGACY_NUM_PLACEMENT_ACTIONS = 734
LEGACY_HOLD_ACTION_INDEX = LEGACY_NUM_PLACEMENT_ACTIONS
LEGACY_NUM_ACTIONS = LEGACY_NUM_PLACEMENT_ACTIONS + 1

NUM_PLACEMENT_ACTIONS = 671
HOLD_ACTION_INDEX = NUM_PLACEMENT_ACTIONS
NUM_ACTIONS = NUM_PLACEMENT_ACTIONS + 1

TETROMINO_CELLS: tuple[tuple[tuple[tuple[int, int], ...], ...], ...] = (
    (
        ((0, 1), (1, 1), (2, 1), (3, 1)),
        ((2, 0), (2, 1), (2, 2), (2, 3)),
        ((0, 2), (1, 2), (2, 2), (3, 2)),
        ((1, 0), (1, 1), (1, 2), (1, 3)),
    ),
    (
        ((1, 1), (2, 1), (1, 2), (2, 2)),
        ((1, 1), (2, 1), (1, 2), (2, 2)),
        ((1, 1), (2, 1), (1, 2), (2, 2)),
        ((1, 1), (2, 1), (1, 2), (2, 2)),
    ),
    (
        ((1, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (2, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (1, 2)),
        ((1, 0), (0, 1), (1, 1), (1, 2)),
    ),
    (
        ((1, 0), (2, 0), (0, 1), (1, 1)),
        ((1, 0), (1, 1), (2, 1), (2, 2)),
        ((1, 1), (2, 1), (0, 2), (1, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
    ),
    (
        ((0, 0), (1, 0), (1, 1), (2, 1)),
        ((2, 0), (1, 1), (2, 1), (1, 2)),
        ((0, 1), (1, 1), (1, 2), (2, 2)),
        ((1, 0), (0, 1), (1, 1), (0, 2)),
    ),
    (
        ((0, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (2, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (2, 2)),
        ((1, 0), (1, 1), (0, 2), (1, 2)),
    ),
    (
        ((2, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (1, 2), (2, 2)),
        ((0, 1), (1, 1), (2, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
    ),
)


def is_redundant_rotation(piece_type: int, rotation: int) -> bool:
    if piece_type in PIECES_ONE_ROTATIONS:
        return rotation >= 1
    if piece_type in PIECES_TWO_ROTATIONS:
        return rotation >= 2
    return False


def canonical_rotation(piece_type: int, rotation: int) -> int:
    if piece_type in PIECES_ONE_ROTATIONS:
        return 0
    if piece_type in PIECES_TWO_ROTATIONS:
        return rotation % 2
    return rotation


def piece_min_offsets(piece_type: int, rotation: int) -> tuple[int, int]:
    cells = TETROMINO_CELLS[piece_type][rotation]
    min_dx = min(dx for dx, _ in cells)
    min_dy = min(dy for _, dy in cells)
    return min_dx, min_dy


def is_valid_position_empty_board(
    piece_type: int,
    rotation: int,
    x: int,
    y: int,
) -> bool:
    for dx, dy in TETROMINO_CELLS[piece_type][rotation]:
        board_x = x + dx
        board_y = y + dy
        if (
            board_x < 0
            or board_x >= BOARD_WIDTH
            or board_y < 0
            or board_y >= BOARD_HEIGHT
        ):
            return False
    return True


def placement_grid_flat_index(rotation: int, grid_x: int, grid_y: int) -> int:
    return rotation * BOARD_HEIGHT * BOARD_WIDTH + grid_y * BOARD_WIDTH + grid_x


def placement_to_canonical_cell(
    piece_type: int,
    x: int,
    y: int,
    rotation: int,
) -> tuple[int, int, int] | None:
    if not is_valid_position_empty_board(piece_type, rotation, x, y):
        return None
    min_dx, min_dy = piece_min_offsets(piece_type, rotation)
    grid_x = x + min_dx
    grid_y = y + min_dy
    if not (0 <= grid_x < BOARD_WIDTH and 0 <= grid_y < BOARD_HEIGHT):
        return None
    return canonical_rotation(piece_type, rotation), grid_x, grid_y


def is_valid_canonical_cell_for_piece(
    piece_type: int,
    rotation: int,
    grid_x: int,
    grid_y: int,
) -> bool:
    if is_redundant_rotation(piece_type, rotation):
        return False
    min_dx, min_dy = piece_min_offsets(piece_type, rotation)
    return is_valid_position_empty_board(
        piece_type,
        rotation,
        grid_x - min_dx,
        grid_y - min_dy,
    )


def _build_legacy_action_positions() -> tuple[tuple[int, int, int], ...]:
    valid_positions: list[tuple[int, int, int]] = []
    for y in range(Y_MIN, Y_MAX_EXCLUSIVE):
        for x in range(X_MIN, X_MAX_EXCLUSIVE):
            for rotation in range(len(ROTATION_LABELS)):
                if any(
                    is_valid_position_empty_board(piece_type, rotation, x, y)
                    for piece_type in range(NUM_PIECE_TYPES)
                ):
                    valid_positions.append((x, y, rotation))
    valid_positions.sort(key=lambda position: (position[2], position[1], position[0]))
    if len(valid_positions) != LEGACY_NUM_PLACEMENT_ACTIONS:
        raise ValueError(
            "Legacy action-space position build drifted: "
            f"expected {LEGACY_NUM_PLACEMENT_ACTIONS}, got {len(valid_positions)}"
        )
    return tuple(valid_positions)


def _build_canonical_cells() -> tuple[
    tuple[tuple[int, int, int], ...],
    np.ndarray,
    np.ndarray,
    tuple[int, int, int, int],
]:
    action_to_cell: list[tuple[int, int, int]] = []
    action_to_flat: list[int] = []
    cell_to_action = np.full(FULL_GRID_PLACEMENT_SLOTS, -1, dtype=np.int32)
    rotation_counts: list[int] = []

    for rotation in range(len(ROTATION_LABELS)):
        rotation_count = 0
        for grid_y in range(BOARD_HEIGHT):
            for grid_x in range(BOARD_WIDTH):
                active = any(
                    is_valid_canonical_cell_for_piece(
                        piece_type,
                        rotation,
                        grid_x,
                        grid_y,
                    )
                    for piece_type in range(NUM_PIECE_TYPES)
                )
                if not active:
                    continue
                flat_index = placement_grid_flat_index(rotation, grid_x, grid_y)
                cell_to_action[flat_index] = len(action_to_cell)
                action_to_cell.append((rotation, grid_x, grid_y))
                action_to_flat.append(flat_index)
                rotation_count += 1
        rotation_counts.append(rotation_count)

    if len(action_to_cell) != NUM_PLACEMENT_ACTIONS:
        raise ValueError(
            "Canonical action-space build drifted: "
            f"expected {NUM_PLACEMENT_ACTIONS}, got {len(action_to_cell)}"
        )
    if len(rotation_counts) != len(ROTATION_LABELS):
        raise ValueError(
            "Canonical action-space rotation counts drifted: "
            f"expected {len(ROTATION_LABELS)}, got {len(rotation_counts)}"
        )

    return (
        tuple(action_to_cell),
        np.asarray(action_to_flat, dtype=np.int64),
        cell_to_action,
        (
            rotation_counts[0],
            rotation_counts[1],
            rotation_counts[2],
            rotation_counts[3],
        ),
    )


def _build_piece_mappings() -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[int, ...],
    tuple[tuple[np.ndarray, np.ndarray], ...],
    tuple[tuple[tuple[int, np.ndarray], ...], ...],
]:
    legacy_to_canonical = np.full(
        (NUM_PIECE_TYPES, LEGACY_NUM_PLACEMENT_ACTIONS),
        -1,
        dtype=np.int32,
    )
    valid_mask_by_piece = np.zeros(
        (NUM_PIECE_TYPES, NUM_PLACEMENT_ACTIONS),
        dtype=np.bool_,
    )

    for piece_type in range(NUM_PIECE_TYPES):
        for legacy_action_idx, (x, y, rotation) in enumerate(LEGACY_ACTION_POSITIONS):
            canonical_cell = placement_to_canonical_cell(piece_type, x, y, rotation)
            if canonical_cell is None:
                continue
            canonical_rotation_idx, grid_x, grid_y = canonical_cell
            flat_index = placement_grid_flat_index(
                canonical_rotation_idx, grid_x, grid_y
            )
            canonical_action_idx = int(CELL_TO_ACTION_INDEX[flat_index])
            if canonical_action_idx < 0:
                raise ValueError(
                    "Canonical action-space build omitted a valid normalized cell: "
                    f"piece={piece_type}, rotation={rotation}, x={x}, y={y}, "
                    f"canonical_rotation={canonical_rotation_idx}, "
                    f"grid_x={grid_x}, grid_y={grid_y}"
                )
            legacy_to_canonical[piece_type, legacy_action_idx] = canonical_action_idx
            valid_mask_by_piece[piece_type, canonical_action_idx] = True

    valid_action_counts = tuple(
        int(valid_mask_by_piece[piece_type].sum())
        for piece_type in range(NUM_PIECE_TYPES)
    )
    direct_maps: list[tuple[np.ndarray, np.ndarray]] = []
    merged_maps: list[tuple[tuple[int, np.ndarray], ...]] = []

    for piece_type in range(NUM_PIECE_TYPES):
        source_columns_by_target: list[list[int]] = [
            [] for _ in range(NUM_PLACEMENT_ACTIONS)
        ]
        for legacy_action_idx, canonical_action_idx in enumerate(
            legacy_to_canonical[piece_type]
        ):
            if canonical_action_idx >= 0:
                source_columns_by_target[int(canonical_action_idx)].append(
                    legacy_action_idx
                )

        direct_targets: list[int] = []
        direct_sources: list[int] = []
        merged_groups: list[tuple[int, np.ndarray]] = []
        for target_action, source_columns in enumerate(source_columns_by_target):
            if not source_columns:
                continue
            if len(source_columns) == 1:
                direct_targets.append(target_action)
                direct_sources.append(source_columns[0])
                continue
            merged_groups.append(
                (
                    target_action,
                    np.asarray(source_columns, dtype=np.int64),
                )
            )

        direct_maps.append(
            (
                np.asarray(direct_targets, dtype=np.int64),
                np.asarray(direct_sources, dtype=np.int64),
            )
        )
        merged_maps.append(tuple(merged_groups))

    return (
        legacy_to_canonical,
        valid_mask_by_piece,
        valid_action_counts,
        tuple(direct_maps),
        tuple(merged_maps),
    )


LEGACY_ACTION_POSITIONS = _build_legacy_action_positions()
(
    ACTION_TO_CANONICAL_CELL,
    ACTION_TO_CANONICAL_FLAT_INDEX,
    CELL_TO_ACTION_INDEX,
    ACTIVE_CANONICAL_ROTATION_COUNTS,
) = _build_canonical_cells()
(
    LEGACY_TO_CANONICAL_ACTION_BY_PIECE,
    VALID_CANONICAL_ACTION_MASK_BY_PIECE,
    PIECE_VALID_ACTION_COUNTS,
    LEGACY_DIRECT_MAPS_BY_PIECE,
    LEGACY_MERGED_MAPS_BY_PIECE,
) = _build_piece_mappings()


def current_piece_indices_from_one_hot(current_pieces: np.ndarray) -> np.ndarray:
    if current_pieces.ndim != 2 or current_pieces.shape[1] != NUM_PIECE_TYPES:
        raise ValueError(
            f"current_pieces must have shape (N, {NUM_PIECE_TYPES}), "
            f"got {current_pieces.shape}"
        )
    return current_pieces.argmax(axis=1).astype(np.int64, copy=False)


def adapt_legacy_policy_targets(
    current_pieces: np.ndarray,
    legacy_policy_targets: np.ndarray,
) -> np.ndarray:
    if (
        legacy_policy_targets.ndim != 2
        or legacy_policy_targets.shape[1] != LEGACY_NUM_ACTIONS
    ):
        raise ValueError(
            f"legacy_policy_targets must have shape (N, {LEGACY_NUM_ACTIONS}), "
            f"got {legacy_policy_targets.shape}"
        )

    current_piece_indices = current_piece_indices_from_one_hot(current_pieces)
    adapted = np.zeros(
        (legacy_policy_targets.shape[0], NUM_ACTIONS),
        dtype=np.float32,
    )
    adapted[:, HOLD_ACTION_INDEX] = legacy_policy_targets[:, LEGACY_HOLD_ACTION_INDEX]
    legacy_placement_targets = legacy_policy_targets[:, :LEGACY_NUM_PLACEMENT_ACTIONS]

    for piece_type in range(NUM_PIECE_TYPES):
        row_mask = current_piece_indices == piece_type
        if not np.any(row_mask):
            continue
        piece_targets = legacy_placement_targets[row_mask]
        adapted_piece_targets = adapted[row_mask, :NUM_PLACEMENT_ACTIONS]
        direct_targets, direct_sources = LEGACY_DIRECT_MAPS_BY_PIECE[piece_type]
        if direct_targets.size > 0:
            adapted_piece_targets[:, direct_targets] = piece_targets[:, direct_sources]
        for target_action, source_columns in LEGACY_MERGED_MAPS_BY_PIECE[piece_type]:
            adapted_piece_targets[:, target_action] = piece_targets[
                :, source_columns
            ].sum(axis=1)
        adapted[row_mask, :NUM_PLACEMENT_ACTIONS] = adapted_piece_targets

    return adapted


def adapt_legacy_action_masks(
    current_pieces: np.ndarray,
    legacy_action_masks: np.ndarray,
) -> np.ndarray:
    if (
        legacy_action_masks.ndim != 2
        or legacy_action_masks.shape[1] != LEGACY_NUM_ACTIONS
    ):
        raise ValueError(
            f"legacy_action_masks must have shape (N, {LEGACY_NUM_ACTIONS}), "
            f"got {legacy_action_masks.shape}"
        )

    current_piece_indices = current_piece_indices_from_one_hot(current_pieces)
    adapted = np.zeros(
        (legacy_action_masks.shape[0], NUM_ACTIONS),
        dtype=np.bool_,
    )
    adapted[:, HOLD_ACTION_INDEX] = legacy_action_masks[
        :, LEGACY_HOLD_ACTION_INDEX
    ].astype(
        np.bool_,
        copy=False,
    )
    legacy_placement_masks = legacy_action_masks[
        :, :LEGACY_NUM_PLACEMENT_ACTIONS
    ].astype(
        np.bool_,
        copy=False,
    )

    for piece_type in range(NUM_PIECE_TYPES):
        row_mask = current_piece_indices == piece_type
        if not np.any(row_mask):
            continue
        piece_masks = legacy_placement_masks[row_mask]
        adapted_piece_masks = adapted[row_mask, :NUM_PLACEMENT_ACTIONS]
        direct_targets, direct_sources = LEGACY_DIRECT_MAPS_BY_PIECE[piece_type]
        if direct_targets.size > 0:
            adapted_piece_masks[:, direct_targets] = piece_masks[:, direct_sources]
        for target_action, source_columns in LEGACY_MERGED_MAPS_BY_PIECE[piece_type]:
            adapted_piece_masks[:, target_action] = piece_masks[:, source_columns].any(
                axis=1
            )
        adapted[row_mask, :NUM_PLACEMENT_ACTIONS] = adapted_piece_masks

    return adapted
