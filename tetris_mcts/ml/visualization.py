"""
Board visualization for training monitoring.

Renders Tetris board states to PIL Images for logging to wandb.
"""

from PIL import Image, ImageDraw, ImageFont
from typing import Optional
import numpy as np

# Piece colors (RGB) - matching tetris_core
PIECE_COLORS = [
    (93, 173, 212),  # I - Cyan
    (219, 174, 63),  # O - Yellow
    (178, 74, 156),  # T - Magenta
    (114, 184, 65),  # S - Green
    (204, 65, 65),  # Z - Red
    (59, 84, 165),  # J - Blue
    (227, 127, 59),  # L - Orange
]

# Board rendering constants
CELL_SIZE = 20
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
PADDING = 10
INFO_HEIGHT_BASIC = 40
INFO_HEIGHT_EXTENDED = 70
PIECE_NAMES = ["I", "O", "T", "S", "Z", "J", "L"]


def render_board(
    board: np.ndarray,
    board_colors: Optional[list[list[Optional[int]]]] = None,
    current_piece_cells: Optional[list[tuple[int, int]]] = None,
    current_piece_type: Optional[int] = None,
    ghost_cells: Optional[list[tuple[int, int]]] = None,
    move_number: int = 0,
    attack: int = 0,
    info_text: Optional[str] = None,
    # Extended info (shown on second line)
    show_piece_info: bool = False,
    current_piece_name: Optional[str] = None,
    hold_piece_name: Optional[str] = None,
    queue_pieces: Optional[list[str]] = None,
) -> Image.Image:
    """
    Render a Tetris board state to a PIL Image.

    Args:
        board: 2D array (20x10) of cell occupancy (0=empty, 1=filled)
        board_colors: 2D array of piece type indices for coloring locked pieces
        current_piece_cells: List of (x, y) cells for current piece
        current_piece_type: Piece type index (0-6) for coloring current piece
        ghost_cells: List of (x, y) cells for ghost piece outline
        move_number: Current move number to display
        attack: Current attack value to display
        info_text: Optional additional info text

    Returns:
        PIL Image of the rendered board
    """
    # Calculate image dimensions
    img_width = BOARD_WIDTH * CELL_SIZE + 2 * PADDING
    img_height = BOARD_HEIGHT * CELL_SIZE + 2 * PADDING + INFO_HEIGHT

    # Create image with dark background
    img = Image.new("RGB", (img_width, img_height), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)

    board_x = PADDING
    board_y = INFO_HEIGHT

    # Draw grid lines
    grid_color = (40, 40, 40)
    for x in range(BOARD_WIDTH + 1):
        x_pos = board_x + x * CELL_SIZE
        draw.line(
            [(x_pos, board_y), (x_pos, board_y + BOARD_HEIGHT * CELL_SIZE)],
            fill=grid_color,
        )
    for y in range(BOARD_HEIGHT + 1):
        y_pos = board_y + y * CELL_SIZE
        draw.line(
            [(board_x, y_pos), (board_x + BOARD_WIDTH * CELL_SIZE, y_pos)],
            fill=grid_color,
        )

    # Draw locked pieces
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            if board[y][x] != 0:
                color = (80, 80, 80)  # Default gray
                if board_colors is not None:
                    color_idx = board_colors[y][x]
                    if color_idx is not None:
                        color = PIECE_COLORS[color_idx]

                px = board_x + x * CELL_SIZE + 1
                py = board_y + y * CELL_SIZE + 1
                draw.rectangle(
                    [px, py, px + CELL_SIZE - 2, py + CELL_SIZE - 2], fill=color
                )

    # Draw ghost piece (outline only)
    if ghost_cells and current_piece_type is not None:
        ghost_color = tuple(c // 2 for c in PIECE_COLORS[current_piece_type])
        for x, y in ghost_cells:
            if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
                px = board_x + x * CELL_SIZE + 1
                py = board_y + y * CELL_SIZE + 1
                draw.rectangle(
                    [px, py, px + CELL_SIZE - 2, py + CELL_SIZE - 2],
                    outline=ghost_color,
                    width=2,
                )

    # Draw current piece
    if current_piece_cells and current_piece_type is not None:
        color = PIECE_COLORS[current_piece_type]
        for x, y in current_piece_cells:
            if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
                px = board_x + x * CELL_SIZE + 1
                py = board_y + y * CELL_SIZE + 1
                draw.rectangle(
                    [px, py, px + CELL_SIZE - 2, py + CELL_SIZE - 2], fill=color
                )

    # Draw border
    draw.rectangle(
        [
            board_x,
            board_y,
            board_x + BOARD_WIDTH * CELL_SIZE,
            board_y + BOARD_HEIGHT * CELL_SIZE,
        ],
        outline=(80, 80, 80),
        width=2,
    )

    # Draw info text at top - try common monospace fonts
    font = None
    for font_path in [
        "/System/Library/Fonts/Menlo.ttc",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Linux
        "C:/Windows/Fonts/consola.ttf",  # Windows
    ]:
        try:
            font = ImageFont.truetype(font_path, 14)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    text = f"Move: {move_number}  Attack: {attack}"
    if info_text:
        text += f"  {info_text}"
    draw.text((PADDING, 10), text, fill=(200, 200, 200), font=font)

    return img


def render_trajectory(
    env,
    actions: list[int],
    max_frames: int = 50,
) -> list[Image.Image]:
    """
    Render a trajectory of moves as a list of images.

    Args:
        env: TetrisEnv instance (will be cloned to avoid mutation)
        actions: List of action indices to execute
        max_frames: Maximum number of frames to render

    Returns:
        List of PIL Images showing the game progression
    """
    # Clone the environment to avoid mutation
    env_copy = env.clone_state()

    frames = []
    total_attack = 0

    for i, action in enumerate(actions):
        if i >= max_frames:
            break

        if env_copy.game_over:
            break

        # Get current state for rendering
        board = np.array(env_copy.get_board())
        board_colors = env_copy.get_board_colors()

        piece = env_copy.get_current_piece()
        piece_cells = None
        piece_type = None
        ghost_cells = None

        if piece:
            piece_cells = piece.get_cells()
            piece_type = piece.piece_type
            ghost = env_copy.get_ghost_piece()
            if ghost:
                ghost_cells = ghost.get_cells()

        # Render frame
        frame = render_board(
            board=board,
            board_colors=board_colors,
            current_piece_cells=piece_cells,
            current_piece_type=piece_type,
            ghost_cells=ghost_cells,
            move_number=i,
            attack=total_attack,
            info_text=f"Action: {action}",
        )
        frames.append(frame)

        # Execute action
        attack = env_copy.execute_action_index(action)
        if attack is not None:
            total_attack += attack

    # Add final frame showing end state
    if not env_copy.game_over and len(frames) < max_frames:
        board = np.array(env_copy.get_board())
        board_colors = env_copy.get_board_colors()
        piece = env_copy.get_current_piece()
        piece_cells = piece.get_cells() if piece else None
        piece_type = piece.piece_type if piece else None
        ghost = env_copy.get_ghost_piece()
        ghost_cells = ghost.get_cells() if ghost else None

        frame = render_board(
            board=board,
            board_colors=board_colors,
            current_piece_cells=piece_cells,
            current_piece_type=piece_type,
            ghost_cells=ghost_cells,
            move_number=len(actions),
            attack=total_attack,
            info_text="Final",
        )
        frames.append(frame)

    return frames


def create_trajectory_gif(
    frames: list[Image.Image],
    output_path: str,
    duration: int = 200,  # ms per frame
) -> None:
    """Save trajectory frames as an animated GIF."""
    if not frames:
        return

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
