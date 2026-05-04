"""Compute the next `nn_value_weight` according to the configured schedule.

The ramp formula (`current * multiplier`, capped per-step by `max_delta`,
clamped at `cap`) and its parameters live on
`NNValueWeightScheduleConfig`. Only the trigger differs by strategy.
"""

from __future__ import annotations

from tetris_bot.ml.config import NNValueWeightScheduleConfig


def _ramp_step(current_weight: float, schedule: NNValueWeightScheduleConfig) -> float:
    delta = min(current_weight * (schedule.multiplier - 1.0), schedule.max_delta)
    return min(schedule.cap, current_weight + delta)


def compute_nn_value_weight(
    schedule: NNValueWeightScheduleConfig,
    *,
    current_weight: float,
    cumulative_games: int,
) -> float:
    if current_weight < 0.0:
        raise ValueError(f"current_weight must be >= 0 (got {current_weight})")
    if schedule.strategy == "per_promotion":
        return _ramp_step(current_weight, schedule)
    if schedule.strategy == "per_games_interval":
        ticks = cumulative_games // schedule.games_interval
        weight = schedule.initial
        for _ in range(ticks):
            if weight >= schedule.cap:
                return schedule.cap
            weight = _ramp_step(weight, schedule)
        return weight
    raise ValueError(
        f"Unknown nn_value_weight schedule strategy: {schedule.strategy!r}"
    )
