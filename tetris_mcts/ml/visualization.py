"""
Board visualization for training monitoring.

Renders Tetris board states to PIL Images for logging to wandb.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tetris_mcts.config import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    PIECE_COLORS,
    PIECE_SPAWN_CELLS,
)

# Board cell size
CELL_SIZE = 20
PADDING = 10

# Sidebar layout (used when show_piece_info=True)
MINI_CELL = 12
LEFT_SIDEBAR = 76
RIGHT_SIDEBAR = 76
SIDEBAR_LABEL_Y = 12
SIDEBAR_PIECE_Y = 38
QUEUE_SLOT_SPACING = 44
STATS_Y = 80
STATS_LINE_H = 16

# Info bar heights
INFO_HEIGHT_SIMPLE = 36
INFO_HEIGHT_SIDEBAR = 28

# Colors
BG_COLOR = (20, 20, 20)
GRID_COLOR = (40, 40, 40)
BORDER_COLOR = (80, 80, 80)
TEXT_COLOR = (200, 200, 200)
LABEL_COLOR = (140, 140, 140)
GRAY = (80, 80, 80)

_font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if size in _font_cache:
        return _font_cache[size]
    font = None
    for font_path in [
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_path, size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def _draw_mini_piece(
    draw: ImageDraw.ImageDraw,
    piece_type: int,
    center_x: int,
    center_y: int,
):
    cells = PIECE_SPAWN_CELLS[piece_type]
    color = PIECE_COLORS[piece_type]

    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    pw = (max_x - min_x + 1) * MINI_CELL
    ph = (max_y - min_y + 1) * MINI_CELL
    sx = center_x - pw // 2
    sy = center_y - ph // 2

    for cx, cy in cells:
        px = sx + (cx - min_x) * MINI_CELL
        py = sy + (cy - min_y) * MINI_CELL
        draw.rectangle(
            [px, py, px + MINI_CELL - 2, py + MINI_CELL - 2], fill=color
        )


def _draw_board_area(
    draw: ImageDraw.ImageDraw,
    board_x: int,
    board_y: int,
    board: np.ndarray,
    board_piece_types: list[list[int | None]] | None,
    current_piece_cells: list[tuple[int, int]] | None,
    current_piece_type: int | None,
    ghost_cells: list[tuple[int, int]] | None,
):
    # Grid
    for x in range(BOARD_WIDTH + 1):
        xp = board_x + x * CELL_SIZE
        draw.line(
            [(xp, board_y), (xp, board_y + BOARD_HEIGHT * CELL_SIZE)],
            fill=GRID_COLOR,
        )
    for y in range(BOARD_HEIGHT + 1):
        yp = board_y + y * CELL_SIZE
        draw.line(
            [(board_x, yp), (board_x + BOARD_WIDTH * CELL_SIZE, yp)],
            fill=GRID_COLOR,
        )

    # Locked pieces
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            if board[y][x] != 0:
                color = GRAY
                if board_piece_types is not None:
                    idx = board_piece_types[y][x]
                    if idx is not None:
                        color = PIECE_COLORS[idx]
                px = board_x + x * CELL_SIZE + 1
                py = board_y + y * CELL_SIZE + 1
                draw.rectangle(
                    [px, py, px + CELL_SIZE - 2, py + CELL_SIZE - 2], fill=color
                )

    # Ghost piece
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

    # Current piece
    if current_piece_cells and current_piece_type is not None:
        color = PIECE_COLORS[current_piece_type]
        for x, y in current_piece_cells:
            if 0 <= y < BOARD_HEIGHT and 0 <= x < BOARD_WIDTH:
                px = board_x + x * CELL_SIZE + 1
                py = board_y + y * CELL_SIZE + 1
                draw.rectangle(
                    [px, py, px + CELL_SIZE - 2, py + CELL_SIZE - 2], fill=color
                )

    # Border
    draw.rectangle(
        [
            board_x,
            board_y,
            board_x + BOARD_WIDTH * CELL_SIZE,
            board_y + BOARD_HEIGHT * CELL_SIZE,
        ],
        outline=BORDER_COLOR,
        width=2,
    )


def render_board(
    board: np.ndarray,
    board_piece_types: list[list[int | None]] | None = None,
    current_piece_cells: list[tuple[int, int]] | None = None,
    current_piece_type: int | None = None,
    ghost_cells: list[tuple[int, int]] | None = None,
    move_number: int = 0,
    attack: int = 0,
    info_text: str | None = None,
    can_hold: bool | None = None,
    combo: int | None = None,
    back_to_back: bool | None = None,
    is_terminal: bool = False,
    value_pred: float | None = None,
    # Sidebar piece display
    show_piece_info: bool = False,
    hold_piece_type: int | None = None,
    queue_piece_types: list[int] | None = None,
) -> Image.Image:
    board_w_px = BOARD_WIDTH * CELL_SIZE
    board_h_px = BOARD_HEIGHT * CELL_SIZE

    if show_piece_info:
        info_h = INFO_HEIGHT_SIDEBAR
        img_w = LEFT_SIDEBAR + board_w_px + RIGHT_SIDEBAR
        img_h = info_h + board_h_px
        board_x = LEFT_SIDEBAR
        board_y = info_h
    else:
        info_h = INFO_HEIGHT_SIMPLE
        img_w = board_w_px + 2 * PADDING
        img_h = info_h + board_h_px + PADDING
        board_x = PADDING
        board_y = info_h

    img = Image.new("RGB", (img_w, img_h), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    _draw_board_area(
        draw,
        board_x,
        board_y,
        board,
        board_piece_types,
        current_piece_cells,
        current_piece_type,
        ghost_cells,
    )

    font = _get_font(14)
    label_font = _get_font(11)

    if show_piece_info:
        # --- Top info bar ---
        parts = [f"Move: {move_number}", f"ATK: {attack}"]
        if value_pred is not None:
            parts.append(f"Vpred: {value_pred:.2f}")
        if is_terminal:
            parts.append("TERMINAL")
        draw.text((LEFT_SIDEBAR, 6), "  ".join(parts), fill=TEXT_COLOR, font=font)

        # --- Left sidebar: HOLD ---
        left_cx = LEFT_SIDEBAR // 2
        bbox = label_font.getbbox("HOLD")
        tw = bbox[2] - bbox[0]
        draw.text(
            (left_cx - tw // 2, board_y + SIDEBAR_LABEL_Y),
            "HOLD",
            fill=LABEL_COLOR,
            font=label_font,
        )

        if hold_piece_type is not None:
            _draw_mini_piece(
                draw, hold_piece_type, left_cx, board_y + SIDEBAR_PIECE_Y
            )

        # Stats below hold piece
        sy = board_y + STATS_Y
        sx = left_cx - 28
        stat_lines = [f"ATK: {attack}"]
        if combo is not None:
            stat_lines.append(f"Combo: {combo}")
        if back_to_back is not None:
            stat_lines.append(f"B2B: {'Y' if back_to_back else 'N'}")
        if can_hold is not None:
            stat_lines.append(f"Hold: {'Y' if can_hold else 'N'}")
        for line in stat_lines:
            draw.text((sx, sy), line, fill=LABEL_COLOR, font=label_font)
            sy += STATS_LINE_H

        # --- Right sidebar: NEXT ---
        right_cx = LEFT_SIDEBAR + board_w_px + RIGHT_SIDEBAR // 2
        bbox = label_font.getbbox("NEXT")
        tw = bbox[2] - bbox[0]
        draw.text(
            (right_cx - tw // 2, board_y + SIDEBAR_LABEL_Y),
            "NEXT",
            fill=LABEL_COLOR,
            font=label_font,
        )

        if queue_piece_types:
            for i, pt in enumerate(queue_piece_types):
                cy = board_y + SIDEBAR_PIECE_Y + i * QUEUE_SLOT_SPACING
                _draw_mini_piece(draw, pt, right_cx, cy)
    else:
        # Simple info line (no sidebars)
        text = f"Move: {move_number}  Attack: {attack}"
        if value_pred is not None:
            text += f"  Vpred: {value_pred:.2f}"
        if info_text:
            text += f"  {info_text}"
        draw.text((PADDING, 10), text, fill=TEXT_COLOR, font=font)

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
