import pytest

from tetris_bot.ml.config import (
    SelfPlayConfig,
    TrainingConfig,
    default_self_play_config,
    default_training_config,
)
from tetris_bot.ml.trainer import Trainer


def _make_config(self_play: SelfPlayConfig) -> TrainingConfig:
    config = default_training_config()
    config.self_play = self_play
    return config


def test_candidate_weight_uses_multiplier_excess_as_delta() -> None:
    config = _make_config(
        default_self_play_config().model_copy(
            update={
                "nn_value_weight_promotion_multiplier": 1.4,
                "nn_value_weight_promotion_max_delta": 0.10,
                "nn_value_weight_cap": 1.0,
            }
        )
    )

    candidate_weight = Trainer._compute_candidate_nn_value_weight(0.025, config)

    assert candidate_weight == pytest.approx(0.035)


def test_candidate_weight_respects_max_delta_and_cap() -> None:
    config = _make_config(
        default_self_play_config().model_copy(
            update={
                "nn_value_weight_promotion_multiplier": 2.0,
                "nn_value_weight_promotion_max_delta": 0.10,
                "nn_value_weight_cap": 0.55,
            }
        )
    )

    candidate_weight = Trainer._compute_candidate_nn_value_weight(0.50, config)

    assert candidate_weight == pytest.approx(0.55)


def test_promotion_eval_window_default_is_small() -> None:
    assert default_self_play_config().model_promotion_eval_games == 20
