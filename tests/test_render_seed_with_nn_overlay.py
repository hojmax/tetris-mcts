from __future__ import annotations

import pytest

from tetris_core.tetris_core import TetrisEnv
from tetris_bot.constants import BOARD_HEIGHT, BOARD_WIDTH, NUM_ACTIONS
from tetris_bot.scripts.inspection.render_seed_with_nn_overlay import (
    HOLD_ACTION_INDEX,
    build_predicted_move_overlays,
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
