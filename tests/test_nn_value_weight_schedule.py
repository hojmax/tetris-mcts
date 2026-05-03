from __future__ import annotations

import pytest

from tetris_bot.ml.config import NNValueWeightScheduleConfig
from tetris_bot.ml.nn_value_weight_schedule import compute_nn_value_weight

_RAMP = dict(initial_weight=0.01, multiplier=1.4, max_delta=0.10, cap=1.0)


def _per_promotion() -> NNValueWeightScheduleConfig:
    return NNValueWeightScheduleConfig(strategy="per_promotion")


def _per_games(interval: int) -> NNValueWeightScheduleConfig:
    return NNValueWeightScheduleConfig(
        strategy="per_games_interval", games_interval=interval
    )


def test_per_promotion_matches_legacy_step_in_exponential_phase():
    weight = compute_nn_value_weight(
        _per_promotion(), current_weight=0.025, cumulative_games=0, **_RAMP
    )
    assert weight == pytest.approx(0.035)


def test_per_promotion_clamps_at_cap():
    schedule = _per_promotion()
    weight = compute_nn_value_weight(
        schedule,
        current_weight=0.50,
        cumulative_games=0,
        initial_weight=0.01,
        multiplier=2.0,
        max_delta=0.10,
        cap=0.55,
    )
    assert weight == pytest.approx(0.55)


def test_per_promotion_ignores_cumulative_games():
    a = compute_nn_value_weight(
        _per_promotion(), current_weight=0.1, cumulative_games=0, **_RAMP
    )
    b = compute_nn_value_weight(
        _per_promotion(), current_weight=0.1, cumulative_games=10**9, **_RAMP
    )
    assert a == b


def test_per_games_interval_zero_ticks_returns_initial():
    weight = compute_nn_value_weight(
        _per_games(500),
        current_weight=999.0,
        cumulative_games=499,
        **_RAMP,
    )
    assert weight == pytest.approx(0.01)


def test_per_games_interval_one_tick_matches_single_step():
    weight = compute_nn_value_weight(
        _per_games(500), current_weight=999.0, cumulative_games=500, **_RAMP
    )
    expected = 0.01 * 1.4
    assert weight == pytest.approx(expected)


def test_per_games_interval_progresses_through_exponential_phase():
    schedule = _per_games(500)
    expected = 0.01
    for tick in range(1, 6):
        delta = min(expected * 0.4, 0.10)
        expected = min(1.0, expected + delta)
        weight = compute_nn_value_weight(
            schedule,
            current_weight=0.0,
            cumulative_games=tick * 500,
            **_RAMP,
        )
        assert weight == pytest.approx(expected), tick


def test_per_games_interval_saturates_at_cap():
    schedule = _per_games(1)
    weight = compute_nn_value_weight(
        schedule, current_weight=0.0, cumulative_games=10**6, **_RAMP
    )
    assert weight == pytest.approx(1.0)


def test_per_games_interval_independent_of_current_weight():
    schedule = _per_games(500)
    a = compute_nn_value_weight(
        schedule, current_weight=0.0, cumulative_games=2500, **_RAMP
    )
    b = compute_nn_value_weight(
        schedule, current_weight=0.99, cumulative_games=2500, **_RAMP
    )
    assert a == pytest.approx(b)


def test_negative_current_weight_rejected():
    with pytest.raises(ValueError):
        compute_nn_value_weight(
            _per_promotion(),
            current_weight=-0.1,
            cumulative_games=0,
            **_RAMP,
        )


def test_unknown_strategy_rejected():
    schedule = NNValueWeightScheduleConfig.model_construct(
        strategy="bogus", games_interval=1
    )
    with pytest.raises(ValueError):
        compute_nn_value_weight(
            schedule, current_weight=0.1, cumulative_games=0, **_RAMP
        )


def test_invalid_games_interval_rejected():
    with pytest.raises(ValueError):
        NNValueWeightScheduleConfig(strategy="per_games_interval", games_interval=0)
