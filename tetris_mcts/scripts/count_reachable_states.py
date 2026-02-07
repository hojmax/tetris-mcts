"""
Count how many of the 800 possible (x, y, rotation) positions are valid
for at least one piece.

800 = 10 columns * 20 rows * 4 rotations
"""

from tetris_mcts.config import BOARD_HEIGHT, BOARD_WIDTH, NUM_PIECE_TYPES, PIECE_NAMES

# Tetromino shapes: [piece_type][rotation][row][col]
# Same as in Rust: I, O, T, S, Z, J, L
TETROMINOS = [
    # I piece - index 0
    [
        [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],  # rot 0
        [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],  # rot 1
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],  # rot 2
        [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],  # rot 3
    ],
    # O piece - index 1
    [
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
    ],
    # T piece - index 2
    [
        [[0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [1, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        [[0, 1, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
    ],
    # S piece - index 3
    [
        [[0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
        [[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
    ],
    # Z piece - index 4
    [
        [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
        [[0, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
    ],
    # J piece - index 5
    [
        [[1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
        [[0, 1, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
    ],
    # L piece - index 6
    [
        [[0, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
        [[1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
    ],
]

def get_cells(piece_type, rotation, x, y):
    """Get the cells occupied by a piece at position (x, y) with given rotation."""
    shape = TETROMINOS[piece_type][rotation]
    cells = []
    for dy in range(4):
        for dx in range(4):
            if shape[dy][dx] == 1:
                cells.append((x + dx, y + dy))
    return cells


def is_valid_position(piece_type, rotation, x, y):
    """Check if a piece fits on an empty board at this position."""
    cells = get_cells(piece_type, rotation, x, y)
    for cx, cy in cells:
        # Check bounds
        if cx < 0 or cx >= BOARD_WIDTH:
            return False
        if cy < 0 or cy >= BOARD_HEIGHT:
            return False
    return True


def main():
    # Need to search beyond 0-9, 0-19 because pieces have offsets within their 4x4 grid
    # Allow x from -3 to 9 and y from -3 to 19 to cover all possible placements
    X_MIN, X_MAX = -3, BOARD_WIDTH
    Y_MIN, Y_MAX = -3, BOARD_HEIGHT

    total_positions = (X_MAX - X_MIN) * (Y_MAX - Y_MIN) * 4
    print(
        f"Checking x=[{X_MIN},{X_MAX}) y=[{Y_MIN},{Y_MAX}) rot=[0,4) = {total_positions} positions...\n"
    )

    # Track which positions are valid for each piece
    valid_by_piece = [set() for _ in range(NUM_PIECE_TYPES)]

    # Track all positions valid for ANY piece
    valid_any = set()

    for y in range(Y_MIN, Y_MAX):
        for x in range(X_MIN, X_MAX):
            for rot in range(4):
                for piece_type in range(NUM_PIECE_TYPES):
                    if is_valid_position(piece_type, rot, x, y):
                        valid_by_piece[piece_type].add((x, y, rot))
                        valid_any.add((x, y, rot))

    # Print per-piece stats
    print("Valid positions per piece:")
    for piece_type in range(NUM_PIECE_TYPES):
        count = len(valid_by_piece[piece_type])
        by_rot = [0, 0, 0, 0]
        for _, _, rot in valid_by_piece[piece_type]:
            by_rot[rot] += 1
        print(
            f"  {PIECE_NAMES[piece_type]}: {count:3d} positions "
            f"(rot0={by_rot[0]:2d}, rot1={by_rot[1]:2d}, rot2={by_rot[2]:2d}, rot3={by_rot[3]:2d})"
        )

    # Count by rotation
    by_rot = [0, 0, 0, 0]
    for _, _, rot in valid_any:
        by_rot[rot] += 1

    print()
    print("=" * 60)
    print(
        f"Search space: {total_positions} positions (x={X_MIN}..{X_MAX - 1}, y={Y_MIN}..{Y_MAX - 1}, rot=0..3)"
    )
    print(f"Valid for at least one piece: {len(valid_any)}")
    print(
        f"By rotation: rot0={by_rot[0]}, rot1={by_rot[1]}, rot2={by_rot[2]}, rot3={by_rot[3]}"
    )
    print()
    print(
        "For reference: 10x20 board = 200 cells, each can have 4 rotations = 800 cell-rotations"
    )
    print(
        "But pieces occupy 4 cells each, so actual unique placements vary by piece shape"
    )


if __name__ == "__main__":
    main()
