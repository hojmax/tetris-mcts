import pytest

from tetris_mcts.config import SelfPlayConfig, TrainingConfig
from tetris_mcts.ml.trainer import Trainer


def test_candidate_weight_uses_multiplier_excess_as_delta() -> None:
    config = TrainingConfig(
        self_play=SelfPlayConfig(
            nn_value_weight_promotion_multiplier=1.4,
            nn_value_weight_promotion_max_delta=0.10,
            nn_value_weight_cap=1.0,
        )
    )

    candidate_weight = Trainer._compute_candidate_nn_value_weight(0.025, config)

    assert candidate_weight == pytest.approx(0.035)


def test_candidate_weight_respects_max_delta_and_cap() -> None:
    config = TrainingConfig(
        self_play=SelfPlayConfig(
            nn_value_weight_promotion_multiplier=2.0,
            nn_value_weight_promotion_max_delta=0.10,
            nn_value_weight_cap=0.55,
        )
    )

    candidate_weight = Trainer._compute_candidate_nn_value_weight(0.50, config)

    assert candidate_weight == pytest.approx(0.55)
