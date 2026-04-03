import pytest

from tetris_bot.constants import DEFAULT_CONFIG_PATH
from tetris_bot.ml.config import SelfPlayConfig, TrainingConfig, load_training_config
from tetris_bot.ml.trainer import Trainer

_DEFAULT_CONFIG = load_training_config(DEFAULT_CONFIG_PATH)
_DEFAULT_SELF_PLAY = _DEFAULT_CONFIG.self_play


def _make_config(self_play: SelfPlayConfig) -> TrainingConfig:
    config = _DEFAULT_CONFIG.model_copy(deep=True)
    config.self_play = self_play
    return config


def test_candidate_weight_uses_multiplier_excess_as_delta() -> None:
    config = _make_config(
        _DEFAULT_SELF_PLAY.model_copy(
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
        _DEFAULT_SELF_PLAY.model_copy(
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
    assert _DEFAULT_SELF_PLAY.model_promotion_eval_games == 20
