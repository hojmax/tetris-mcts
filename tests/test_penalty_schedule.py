from __future__ import annotations

import pytest

from tetris_bot.ml.config import PenaltyScheduleConfig
from tetris_bot.ml.penalty_schedule import compute_penalty_scale, scaled_penalties


def _gated() -> PenaltyScheduleConfig:
    return PenaltyScheduleConfig(strategy="gated")


def _ctl(hold: int, decay: int) -> PenaltyScheduleConfig:
    return PenaltyScheduleConfig(
        strategy="constant_then_linear", hold_games=hold, decay_games=decay
    )


def test_gated_returns_one_below_cap():
    scale = compute_penalty_scale(
        _gated(), cumulative_games=10, nn_value_weight=0.9, nn_value_weight_cap=1.0
    )
    assert scale == 1.0


def test_gated_returns_zero_at_cap():
    scale = compute_penalty_scale(
        _gated(), cumulative_games=10, nn_value_weight=1.0, nn_value_weight_cap=1.0
    )
    assert scale == 0.0


def test_gated_ignores_games_clock():
    scale = compute_penalty_scale(
        _gated(),
        cumulative_games=10**9,
        nn_value_weight=0.5,
        nn_value_weight_cap=1.0,
    )
    assert scale == 1.0


def test_ctl_constant_during_hold():
    schedule = _ctl(hold=40_000, decay=10_000)
    for games in (0, 1, 39_999):
        assert (
            compute_penalty_scale(
                schedule,
                cumulative_games=games,
                nn_value_weight=1.0,
                nn_value_weight_cap=1.0,
            )
            == 1.0
        )


def test_ctl_linear_decay_endpoints():
    schedule = _ctl(hold=40_000, decay=10_000)
    assert (
        compute_penalty_scale(
            schedule,
            cumulative_games=40_000,
            nn_value_weight=0.0,
            nn_value_weight_cap=1.0,
        )
        == 1.0
    )
    midpoint = compute_penalty_scale(
        schedule,
        cumulative_games=45_000,
        nn_value_weight=0.0,
        nn_value_weight_cap=1.0,
    )
    assert midpoint == pytest.approx(0.5)


def test_ctl_zero_after_decay_window():
    schedule = _ctl(hold=40_000, decay=10_000)
    for games in (50_000, 50_001, 10**9):
        assert (
            compute_penalty_scale(
                schedule,
                cumulative_games=games,
                nn_value_weight=0.0,
                nn_value_weight_cap=1.0,
            )
            == 0.0
        )


def test_ctl_independent_of_nn_value_weight_cap():
    schedule = _ctl(hold=0, decay=10)
    scale_below_cap = compute_penalty_scale(
        schedule,
        cumulative_games=5,
        nn_value_weight=0.1,
        nn_value_weight_cap=1.0,
    )
    scale_above_cap = compute_penalty_scale(
        schedule,
        cumulative_games=5,
        nn_value_weight=10.0,
        nn_value_weight_cap=1.0,
    )
    assert scale_below_cap == scale_above_cap == pytest.approx(0.5)


def test_scaled_penalties_applies_uniform_scale():
    schedule = PenaltyScheduleConfig(
        strategy="constant_then_linear",
        hold_games=0,
        decay_games=10,
        death_penalty=10.0,
        overhang_penalty_weight=4.0,
    )
    death, overhang = scaled_penalties(
        schedule,
        cumulative_games=5,
        nn_value_weight=0.0,
        nn_value_weight_cap=1.0,
    )
    assert death == pytest.approx(5.0)
    assert overhang == pytest.approx(2.0)


def test_unknown_strategy_raises():
    schedule = PenaltyScheduleConfig.model_construct(
        strategy="bogus", hold_games=0, decay_games=1
    )
    with pytest.raises(ValueError):
        compute_penalty_scale(
            schedule,
            cumulative_games=0,
            nn_value_weight=0.0,
            nn_value_weight_cap=1.0,
        )


def test_config_validation_rejects_bad_decay():
    with pytest.raises(ValueError):
        PenaltyScheduleConfig(strategy="constant_then_linear", decay_games=0)


def test_config_validation_rejects_bad_hold():
    with pytest.raises(ValueError):
        PenaltyScheduleConfig(strategy="constant_then_linear", hold_games=-1)
