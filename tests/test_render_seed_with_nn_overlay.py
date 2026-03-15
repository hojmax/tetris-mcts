from __future__ import annotations

from pathlib import Path

import pytest

from tetris_core.tetris_core import TetrisEnv
from tetris_bot.constants import BOARD_HEIGHT, BOARD_WIDTH, NUM_ACTIONS
from tetris_bot.scripts.inspection.render_seed_with_nn_overlay import (
    HOLD_ACTION_INDEX,
    build_predicted_move_overlays,
    resolve_move_overlay_output_path,
)
from tetris_bot.visualization import (
    CELL_SIZE,
    INFO_HEIGHT_SIDEBAR,
    LEFT_SIDEBAR,
    PredictedMoveOverlay,
    RIGHT_SIDEBAR,
    _capture_frame,
    _get_font,
    _place_overlay_label_rects,
    _rects_intersect,
)


def test_build_predicted_move_overlays_maps_hold_and_placements() -> None:
    env = TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, 1)
    current_piece = env.get_current_piece()
    assert current_piece is not None

    placements = env.get_possible_placements()
    assert len(placements) >= 2
    low_action = min(placements, key=lambda placement: int(placement.action_index))
    high_action = max(placements, key=lambda placement: int(placement.action_index))

    valid_actions = [
        int(HOLD_ACTION_INDEX),
        int(low_action.action_index),
        int(high_action.action_index),
    ]
    action_priors = [0.401, 0.332, 0.267]

    overlays = build_predicted_move_overlays(env, valid_actions, action_priors)

    assert [overlay.rank for overlay in overlays] == [1, 2, 3]
    assert [overlay.is_hold for overlay in overlays] == [True, False, False]
    assert overlays[0].probability == pytest.approx(0.401)
    assert overlays[0].piece_type == int(current_piece.piece_type)
    assert overlays[0].cells == tuple(
        (int(x), int(y)) for x, y in current_piece.get_cells()
    )

    assert overlays[1].probability == pytest.approx(0.332)
    assert overlays[1].piece_type == int(low_action.piece.piece_type)
    assert overlays[1].cells == tuple(
        (int(x), int(y)) for x, y in low_action.piece.get_cells()
    )

    assert overlays[2].probability == pytest.approx(0.267)
    assert overlays[2].piece_type == int(high_action.piece.piece_type)
    assert overlays[2].cells == tuple(
        (int(x), int(y)) for x, y in high_action.piece.get_cells()
    )


def test_build_predicted_move_overlays_validates_inputs() -> None:
    env = TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, 2)

    with pytest.raises(ValueError, match="same length"):
        build_predicted_move_overlays(
            env,
            valid_actions=[0, NUM_ACTIONS - 1],
            action_priors=[1.0],
        )


def test_resolve_move_overlay_output_path_uses_sibling_suffix() -> None:
    output_path = Path("/tmp/seed12_nn_overlay.gif")

    resolved = resolve_move_overlay_output_path(output_path)

    assert resolved == Path("/tmp/seed12_nn_overlay_top3_moves.gif")


def test_hold_overlay_does_not_draw_extra_piece_on_current_piece() -> None:
    env = TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, 1)
    current_piece = env.get_current_piece()
    assert current_piece is not None

    hold_overlay = PredictedMoveOverlay(
        probability=0.4,
        piece_type=int(current_piece.piece_type),
        cells=tuple((int(x), int(y)) for x, y in current_piece.get_cells()),
        rank=1,
        is_hold=True,
    )

    plain_frame = _capture_frame(
        env,
        placement_number=0,
        attack=0,
        show_ghost_piece=False,
    )
    overlay_frame = _capture_frame(
        env,
        placement_number=0,
        attack=0,
        predicted_move_overlays=[hold_overlay],
        show_ghost_piece=False,
    )

    sample_x, sample_y = current_piece.get_cells()[0]
    pixel_x = LEFT_SIDEBAR + int(sample_x) * CELL_SIZE + CELL_SIZE // 2
    pixel_y = INFO_HEIGHT_SIDEBAR + int(sample_y) * CELL_SIZE + CELL_SIZE // 2
    assert overlay_frame.getpixel((pixel_x, pixel_y)) == plain_frame.getpixel(
        (pixel_x, pixel_y)
    )


def test_overlay_frame_can_hide_live_drop_ghost() -> None:
    env = TetrisEnv.with_seed(BOARD_WIDTH, BOARD_HEIGHT, 1)
    ghost = env.get_ghost_piece()
    assert ghost is not None

    with_ghost = _capture_frame(env, placement_number=0, attack=0, show_ghost_piece=True)
    without_ghost = _capture_frame(env, placement_number=0, attack=0, show_ghost_piece=False)

    ghost_x, ghost_y = ghost.get_cells()[0]
    pixel_x = LEFT_SIDEBAR + int(ghost_x) * CELL_SIZE + CELL_SIZE // 2
    pixel_y = INFO_HEIGHT_SIDEBAR + int(ghost_y) * CELL_SIZE + 2
    assert with_ghost.getpixel((pixel_x, pixel_y)) != without_ghost.getpixel(
        (pixel_x, pixel_y)
    )


def test_place_overlay_label_rects_avoids_direct_label_overlap() -> None:
    img_size = (
        LEFT_SIDEBAR + BOARD_WIDTH * CELL_SIZE + RIGHT_SIDEBAR,
        INFO_HEIGHT_SIDEBAR + BOARD_HEIGHT * CELL_SIZE,
    )
    overlapping_cells = ((3, 18), (4, 18), (5, 18), (6, 18))
    overlays = [
        PredictedMoveOverlay(0.41, 0, overlapping_cells, rank=1),
        PredictedMoveOverlay(0.33, 1, overlapping_cells, rank=2),
        PredictedMoveOverlay(0.26, 2, overlapping_cells, rank=3),
    ]

    placements = _place_overlay_label_rects(
        LEFT_SIDEBAR,
        INFO_HEIGHT_SIDEBAR,
        img_size,
        overlays,
        _get_font(12),
    )

    assert len(placements) == 3
    for idx, placement in enumerate(placements):
        for other in placements[idx + 1 :]:
            assert not _rects_intersect(placement.rect, other.rect)
