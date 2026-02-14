import pytest

from tetris_mcts.config import TrainingConfig
from tetris_mcts.ml.training import Trainer


def test_candidate_weight_uses_multiplier_excess_as_delta() -> None:
    config = TrainingConfig(
        nn_value_weight_promotion_multiplier=1.4,
        nn_value_weight_promotion_max_delta=0.10,
        nn_value_weight_cap=1.0,
    )

    candidate_weight = Trainer._compute_candidate_nn_value_weight(0.025, config)

    assert candidate_weight == pytest.approx(0.035)


def test_candidate_weight_respects_max_delta_and_cap() -> None:
    config = TrainingConfig(
        nn_value_weight_promotion_multiplier=2.0,
        nn_value_weight_promotion_max_delta=0.10,
        nn_value_weight_cap=0.55,
    )

    candidate_weight = Trainer._compute_candidate_nn_value_weight(0.50, config)

    assert candidate_weight == pytest.approx(0.55)


def test_config_rejects_promotion_multiplier_below_one() -> None:
    with pytest.raises(
        ValueError,
        match="nn_value_weight_promotion_multiplier must be finite and >= 1.0",
    ):
        TrainingConfig(nn_value_weight_promotion_multiplier=0.99)
