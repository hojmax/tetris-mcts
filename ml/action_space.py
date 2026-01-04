"""
Action Space Mapping for Tetris AlphaZero

Maps between action indices (0-733) and (x, y, rotation) placement tuples.
The action space covers all positions that are valid for at least one piece
on an empty board.
"""

import numpy as np
from typing import Optional

# Tetromino shapes: [piece_type][rotation][row][col]
# Same as in Rust: I, O, T, S, Z, J, L
TETROMINOS = [
    # I piece - index 0
    [
        [[0,0,0,0], [1,1,1,1], [0,0,0,0], [0,0,0,0]],  # rot 0
        [[0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0]],  # rot 1
        [[0,0,0,0], [0,0,0,0], [1,1,1,1], [0,0,0,0]],  # rot 2
        [[0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0]],  # rot 3
    ],
    # O piece - index 1
    [
        [[0,0,0,0], [0,1,1,0], [0,1,1,0], [0,0,0,0]],
        [[0,0,0,0], [0,1,1,0], [0,1,1,0], [0,0,0,0]],
        [[0,0,0,0], [0,1,1,0], [0,1,1,0], [0,0,0,0]],
        [[0,0,0,0], [0,1,1,0], [0,1,1,0], [0,0,0,0]],
    ],
    # T piece - index 2
    [
        [[0,1,0,0], [1,1,1,0], [0,0,0,0], [0,0,0,0]],
        [[0,1,0,0], [0,1,1,0], [0,1,0,0], [0,0,0,0]],
        [[0,0,0,0], [1,1,1,0], [0,1,0,0], [0,0,0,0]],
        [[0,1,0,0], [1,1,0,0], [0,1,0,0], [0,0,0,0]],
    ],
    # S piece - index 3
    [
        [[0,1,1,0], [1,1,0,0], [0,0,0,0], [0,0,0,0]],
        [[0,1,0,0], [0,1,1,0], [0,0,1,0], [0,0,0,0]],
        [[0,0,0,0], [0,1,1,0], [1,1,0,0], [0,0,0,0]],
        [[1,0,0,0], [1,1,0,0], [0,1,0,0], [0,0,0,0]],
    ],
    # Z piece - index 4
    [
        [[1,1,0,0], [0,1,1,0], [0,0,0,0], [0,0,0,0]],
        [[0,0,1,0], [0,1,1,0], [0,1,0,0], [0,0,0,0]],
        [[0,0,0,0], [1,1,0,0], [0,1,1,0], [0,0,0,0]],
        [[0,1,0,0], [1,1,0,0], [1,0,0,0], [0,0,0,0]],
    ],
    # J piece - index 5
    [
        [[1,0,0,0], [1,1,1,0], [0,0,0,0], [0,0,0,0]],
        [[0,1,1,0], [0,1,0,0], [0,1,0,0], [0,0,0,0]],
        [[0,0,0,0], [1,1,1,0], [0,0,1,0], [0,0,0,0]],
        [[0,1,0,0], [0,1,0,0], [1,1,0,0], [0,0,0,0]],
    ],
    # L piece - index 6
    [
        [[0,0,1,0], [1,1,1,0], [0,0,0,0], [0,0,0,0]],
        [[0,1,0,0], [0,1,0,0], [0,1,1,0], [0,0,0,0]],
        [[0,0,0,0], [1,1,1,0], [1,0,0,0], [0,0,0,0]],
        [[1,1,0,0], [0,1,0,0], [0,1,0,0], [0,0,0,0]],
    ],
]

PIECE_NAMES = ["I", "O", "T", "S", "Z", "J", "L"]
NUM_PIECE_TYPES = 7
BOARD_WIDTH = 10
BOARD_HEIGHT = 20


def get_cells(piece_type: int, rotation: int, x: int, y: int) -> list[tuple[int, int]]:
    """Get the cells occupied by a piece at position (x, y) with given rotation."""
    shape = TETROMINOS[piece_type][rotation]
    cells = []
    for dy in range(4):
        for dx in range(4):
            if shape[dy][dx] == 1:
                cells.append((x + dx, y + dy))
    return cells


def is_valid_position(piece_type: int, rotation: int, x: int, y: int) -> bool:
    """Check if a piece fits on an empty board at this position."""
    cells = get_cells(piece_type, rotation, x, y)
    for cx, cy in cells:
        if cx < 0 or cx >= BOARD_WIDTH:
            return False
        if cy < 0 or cy >= BOARD_HEIGHT:
            return False
    return True


def _build_action_mappings() -> tuple[list[tuple[int, int, int]], dict[tuple[int, int, int], int]]:
    """Build the action index mappings.

    Returns:
        ACTION_TO_PLACEMENT: List where index i gives (x, y, rotation) tuple
        PLACEMENT_TO_ACTION: Dict mapping (x, y, rotation) -> action index
    """
    # Search beyond 0-9, 0-19 because pieces have offsets within their 4x4 grid
    X_MIN, X_MAX = -3, BOARD_WIDTH
    Y_MIN, Y_MAX = -3, BOARD_HEIGHT

    # Track all positions valid for ANY piece
    valid_positions = set()

    for y in range(Y_MIN, Y_MAX):
        for x in range(X_MIN, X_MAX):
            for rot in range(4):
                for piece_type in range(NUM_PIECE_TYPES):
                    if is_valid_position(piece_type, rot, x, y):
                        valid_positions.add((x, y, rot))
                        break  # Position is valid if any piece fits

    # Sort for deterministic ordering: by rotation, then y, then x
    sorted_positions = sorted(valid_positions, key=lambda p: (p[2], p[1], p[0]))

    action_to_placement = sorted_positions
    placement_to_action = {pos: idx for idx, pos in enumerate(sorted_positions)}

    return action_to_placement, placement_to_action


# Build the mappings at module load time
ACTION_TO_PLACEMENT, PLACEMENT_TO_ACTION = _build_action_mappings()
NUM_ACTIONS = len(ACTION_TO_PLACEMENT)


def action_to_placement(action_idx: int) -> tuple[int, int, int]:
    """Convert action index to (x, y, rotation) placement tuple."""
    if action_idx < 0 or action_idx >= NUM_ACTIONS:
        raise ValueError(f"Invalid action index: {action_idx}. Must be in [0, {NUM_ACTIONS})")
    return ACTION_TO_PLACEMENT[action_idx]


def placement_to_action(x: int, y: int, rotation: int) -> Optional[int]:
    """Convert (x, y, rotation) to action index, or None if invalid."""
    return PLACEMENT_TO_ACTION.get((x, y, rotation))


def get_action_mask(board: np.ndarray, piece_type: int) -> np.ndarray:
    """
    Generate a binary mask of valid actions for the given piece and board state.

    Args:
        board: 2D numpy array of shape (20, 10), 1 = filled, 0 = empty
        piece_type: Integer 0-6 representing piece type

    Returns:
        1D numpy array of shape (NUM_ACTIONS,), 1 = valid, 0 = invalid
    """
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)

    for action_idx, (x, y, rot) in enumerate(ACTION_TO_PLACEMENT):
        if _piece_can_be_placed(board, piece_type, x, y, rot):
            mask[action_idx] = 1.0

    return mask


def _piece_can_be_placed(board: np.ndarray, piece_type: int, x: int, y: int, rotation: int) -> bool:
    """Check if a piece can be placed at the given position on the board."""
    cells = get_cells(piece_type, rotation, x, y)

    for cx, cy in cells:
        # Check bounds
        if cx < 0 or cx >= BOARD_WIDTH:
            return False
        if cy < 0 or cy >= BOARD_HEIGHT:
            return False
        # Check collision with existing blocks
        if board[cy, cx] != 0:
            return False

    return True


def get_valid_actions(board: np.ndarray, piece_type: int) -> list[int]:
    """
    Get list of valid action indices for the given piece and board state.

    Args:
        board: 2D numpy array of shape (20, 10), 1 = filled, 0 = empty
        piece_type: Integer 0-6 representing piece type

    Returns:
        List of valid action indices
    """
    valid = []
    for action_idx, (x, y, rot) in enumerate(ACTION_TO_PLACEMENT):
        if _piece_can_be_placed(board, piece_type, x, y, rot):
            valid.append(action_idx)
    return valid


# Pre-compute per-piece valid actions on empty board for quick reference
_PIECE_VALID_ACTIONS_EMPTY = {}
for _pt in range(NUM_PIECE_TYPES):
    _empty_board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8)
    _PIECE_VALID_ACTIONS_EMPTY[_pt] = get_valid_actions(_empty_board, _pt)


def get_piece_valid_actions_empty(piece_type: int) -> list[int]:
    """Get valid action indices for a piece on an empty board (cached)."""
    return _PIECE_VALID_ACTIONS_EMPTY[piece_type]


if __name__ == "__main__":
    # Print statistics
    print(f"Total action space size: {NUM_ACTIONS}")
    print()

    # Count by rotation
    by_rot = [0, 0, 0, 0]
    for x, y, rot in ACTION_TO_PLACEMENT:
        by_rot[rot] += 1
    print(f"Actions by rotation: {by_rot}")
    print()

    # Show range of x, y values
    xs = [p[0] for p in ACTION_TO_PLACEMENT]
    ys = [p[1] for p in ACTION_TO_PLACEMENT]
    print(f"X range: [{min(xs)}, {max(xs)}]")
    print(f"Y range: [{min(ys)}, {max(ys)}]")
    print()

    # Valid actions per piece on empty board
    print("Valid actions per piece (empty board):")
    for piece_type in range(NUM_PIECE_TYPES):
        count = len(get_piece_valid_actions_empty(piece_type))
        print(f"  {PIECE_NAMES[piece_type]}: {count}")
