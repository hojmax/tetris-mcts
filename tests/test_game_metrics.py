from __future__ import annotations

import pytest

from tetris_bot.ml.game_metrics import average_completed_games


def test_average_completed_games_emits_window_game_number_metadata() -> None:
    completed_games = [
        (
            41,
            {
                "episode_length": 10,
                "total_attack": 5,
                "total_lines": 4,
                "holds": 2,
                "perfect_clears": 0,
            },
        ),
        (
            42,
            {
                "episode_length": 20,
                "total_attack": 7,
                "total_lines": 8,
                "holds": 4,
                "perfect_clears": 1,
            },
        ),
    ]

    metrics = average_completed_games(completed_games)

    assert metrics["game_number"] == 42.0
    assert metrics["game/number"] == 42.0
    assert metrics["game/window_size"] == 2.0
    assert metrics["game/window_first_number"] == 41.0
    assert metrics["game/window_last_number"] == 42.0
    assert metrics["game/perfect_clears"] == 0.5
    assert metrics["game/attack_per_move"] == pytest.approx(12.0 / 30.0)
