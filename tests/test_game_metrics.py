from __future__ import annotations

import pytest
import torch

from tetris_bot.ml.game_metrics import (
    average_completed_games,
    compute_batch_feature_metrics,
)
from tetris_bot.ml.aux_features import AUX_FEATURE_LAYOUT, AUX_FEATURES


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


def test_compute_batch_feature_metrics_accepts_boolean_masks() -> None:
    aux = torch.zeros(2, AUX_FEATURES)
    aux[:, AUX_FEATURE_LAYOUT.row_fill_counts] = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.6, 0.8]]
    )
    aux[:, AUX_FEATURE_LAYOUT.max_column_height] = torch.tensor([0.3, 0.7])
    aux[:, AUX_FEATURE_LAYOUT.total_blocks] = torch.tensor([0.25, 0.5])
    aux[:, AUX_FEATURE_LAYOUT.bumpiness] = torch.tensor([0.4, 0.9])
    aux[:, AUX_FEATURE_LAYOUT.holes] = torch.tensor([0.1, 0.2])
    value_targets = torch.tensor([0.5, -0.5])
    overhang_fields = torch.tensor([0.25, 0.75])
    masks = torch.tensor([[True, False, True], [True, True, False]])

    metrics = compute_batch_feature_metrics(
        aux=aux,
        value_targets=value_targets,
        overhang_fields=overhang_fields,
        masks=masks,
    )

    assert metrics["batch/value_target_mean"] == pytest.approx(0.0)
    assert metrics["batch/valid_actions_mean"] == pytest.approx(2.0)
    assert metrics["batch/board_fill_mean"] == pytest.approx(0.375)
    assert metrics["batch/max_height_mean"] == pytest.approx(0.5)
    assert metrics["batch/overhang_fields_mean"] == pytest.approx(0.5)
