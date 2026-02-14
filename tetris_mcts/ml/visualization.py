"""
Board visualization for training monitoring.

Renders Tetris board states to PIL Images for logging to wandb.
"""

from PIL import Image, ImageDraw, ImageFont
from typing import Optional
import numpy as np

from tetris_mcts.config import BOARD_HEIGHT, BOARD_WIDTH, PIECE_COLORS

# Board rendering constants
CELL_SIZE = 20
PADDING = 10
INFO_HEIGHT_BASIC = 36
INFO_HEIGHT_EXTENDED = 106


def render_board(
    board: np.ndarray,
    board_piece_types: Optional[list[list[Optional[int]]]] = None,
    current_piece_cells: Optional[list[tuple[int, int]]] = None,
    current_piece_type: Optional[int] = None,
    ghost_cells: Optional[list[tuple[int, int]]] = None,
    move_number: int = 0,
    attack: int = 0,
    info_text: Optional[str] = None,
    can_hold: Optional[bool] = None,
    combo: Optional[int] = None,
    back_to_back: Optional[bool] = None,
    is_terminal: bool = False,
    vpred: Optional[float] = None,
    value_pred: Optional[float] = None,
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
        board_piece_types: 2D array of piece type indices for coloring locked pieces
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
    info_height = INFO_HEIGHT_EXTENDED if show_piece_info else INFO_HEIGHT_BASIC
    img_width = BOARD_WIDTH * CELL_SIZE + 2 * PADDING
    img_height = BOARD_HEIGHT * CELL_SIZE + PADDING + info_height

    # Create image with dark background
    img = Image.new("RGB", (img_width, img_height), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)

    board_x = PADDING
    board_y = info_height

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
                if board_piece_types is not None:
                    color_idx = board_piece_types[y][x]
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

    # Draw info text
    resolved_value_pred = value_pred if value_pred is not None else vpred
    if value_pred is not None and vpred is not None and value_pred != vpred:
        raise ValueError(
            f"Conflicting value prediction inputs: value_pred={value_pred}, vpred={vpred}"
        )

    if show_piece_info:
        # 4 lines: Move/Attack, Piece/Hold, Queue, Can hold/Vpred
        current = current_piece_name or "?"
        hold = hold_piece_name or "-"
        queue = " ".join(queue_pieces) if queue_pieces else ""
        draw.text(
            (PADDING, 10),
            f"Move: {move_number}  Attack: {attack}",
            fill=(200, 200, 200),
            font=font,
        )
        has_standard_status = (
            can_hold is not None and combo is not None and back_to_back is not None
        )
        if has_standard_status:
            second_line = (
                f"{'Terminal  ' if is_terminal else ''}Can hold: {'y' if can_hold else 'n'}\n"
                f"Combo: {combo}  B2B: {'y' if back_to_back else 'n'}"
            )
        else:
            second_line = info_text or ""
        if resolved_value_pred is not None:
            if second_line:
                second_line += f"  Vpred: {resolved_value_pred:.2f}"
            else:
                second_line = f"Vpred: {resolved_value_pred:.2f}"
        draw.text(
            (PADDING, 30),
            f"Piece: {current}  Hold: {hold}",
            fill=(200, 200, 200),
            font=font,
        )
        draw.text((PADDING, 50), f"Queue: {queue}", fill=(200, 200, 200), font=font)
        if second_line:
            draw.text((PADDING, 70), second_line, fill=(200, 200, 200), font=font)
    else:
        # Single line
        text = f"Move: {move_number}  Attack: {attack}"
        if resolved_value_pred is not None:
            text += f"  Vpred: {resolved_value_pred:.2f}"
        if info_text:
            text += f"  {info_text}"
        draw.text((PADDING, 10), text, fill=(200, 200, 200), font=font)

    return img


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
