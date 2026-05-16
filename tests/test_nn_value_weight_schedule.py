from __future__ import annotations

import pytest

from tetris_bot.ml.config import NNValueWeightScheduleConfig
from tetris_bot.ml.nn_value_weight_schedule import compute_nn_value_weight


def _per_promotion(
    *,
    initial: float = 0.01,
    multiplier: float = 1.4,
    max_delta: float = 0.10,
    cap: float = 1.0,
) -> NNValueWeightScheduleConfig:
    return NNValueWeightScheduleConfig(
        strategy="per_promotion",
        initial=initial,
        multiplier=multiplier,
        max_delta=max_delta,
        cap=cap,
    )


def _per_games(
    interval: int,
    *,
    initial: float = 0.01,
    multiplier: float = 1.4,
    max_delta: float = 0.10,
    cap: float = 1.0,
) -> NNValueWeightScheduleConfig:
    return NNValueWeightScheduleConfig(
        strategy="per_games_interval",
        games_interval=interval,
        initial=initial,
        multiplier=multiplier,
        max_delta=max_delta,
        cap=cap,
    )


def test_per_promotion_matches_legacy_step_in_exponential_phase():
    weight = compute_nn_value_weight(
        _per_promotion(), current_weight=0.025, cumulative_games=0
    )
    assert weight == pytest.approx(0.035)


def test_per_promotion_clamps_at_cap():
    schedule = _per_promotion(multiplier=2.0, cap=0.55)
    weight = compute_nn_value_weight(schedule, current_weight=0.50, cumulative_games=0)
    assert weight == pytest.approx(0.55)


def test_per_promotion_ignores_cumulative_games():
    a = compute_nn_value_weight(
        _per_promotion(), current_weight=0.1, cumulative_games=0
    )
    b = compute_nn_value_weight(
        _per_promotion(), current_weight=0.1, cumulative_games=10**9
    )
    assert a == b


def test_per_games_interval_zero_ticks_returns_initial():
    weight = compute_nn_value_weight(
        _per_games(500), current_weight=999.0, cumulative_games=499
    )
    assert weight == pytest.approx(0.01)


def test_per_games_interval_one_tick_matches_single_step():
    weight = compute_nn_value_weight(
        _per_games(500), current_weight=999.0, cumulative_games=500
    )
    assert weight == pytest.approx(0.01 * 1.4)


def test_per_games_interval_progresses_through_exponential_phase():
    schedule = _per_games(500)
    expected = 0.01
    for tick in range(1, 6):
        delta = min(expected * 0.4, 0.10)
        expected = min(1.0, expected + delta)
        weight = compute_nn_value_weight(
            schedule, current_weight=0.0, cumulative_games=tick * 500
        )
        assert weight == pytest.approx(expected), tick


def test_per_games_interval_saturates_at_cap():
    schedule = _per_games(1)
    weight = compute_nn_value_weight(
        schedule, current_weight=0.0, cumulative_games=10**6
    )
    assert weight == pytest.approx(1.0)


def test_per_games_interval_independent_of_current_weight():
    schedule = _per_games(500)
    a = compute_nn_value_weight(schedule, current_weight=0.0, cumulative_games=2500)
    b = compute_nn_value_weight(schedule, current_weight=0.99, cumulative_games=2500)
    assert a == pytest.approx(b)


def test_negative_current_weight_rejected():
    with pytest.raises(ValueError):
        compute_nn_value_weight(
            _per_promotion(), current_weight=-0.1, cumulative_games=0
        )


def test_unknown_strategy_rejected():
    schedule = NNValueWeightScheduleConfig.model_construct(
        strategy="bogus",
        games_interval=1,
        initial=0.01,
        multiplier=1.4,
        max_delta=0.1,
        cap=1.0,
    )
    with pytest.raises(ValueError):
        compute_nn_value_weight(schedule, current_weight=0.1, cumulative_games=0)


def test_invalid_games_interval_rejected():
    with pytest.raises(ValueError):
        NNValueWeightScheduleConfig(strategy="per_games_interval", games_interval=0)


def test_invalid_cap_rejected():
    with pytest.raises(ValueError):
        NNValueWeightScheduleConfig(initial=0.5, cap=0.1)
