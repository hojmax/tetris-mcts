"""Compute the next `nn_value_weight` according to the configured schedule.

The ramp formula (`current * multiplier`, capped per-step by `max_delta`,
clamped at `cap`) stays the same — only the trigger differs. See
`NNValueWeightScheduleConfig` for the strategies.
"""

from __future__ import annotations

from tetris_bot.ml.config import NNValueWeightScheduleConfig


def _ramp_step(
    current_weight: float,
    *,
    multiplier: float,
    max_delta: float,
    cap: float,
) -> float:
    promotion_delta = current_weight * (multiplier - 1.0)
    delta = min(promotion_delta, max_delta)
    return min(cap, current_weight + delta)


def compute_nn_value_weight(
    schedule: NNValueWeightScheduleConfig,
    *,
    current_weight: float,
    cumulative_games: int,
    initial_weight: float,
    multiplier: float,
    max_delta: float,
    cap: float,
) -> float:
    if current_weight < 0.0:
        raise ValueError(f"current_weight must be >= 0 (got {current_weight})")
    if schedule.strategy == "per_promotion":
        return _ramp_step(
            current_weight, multiplier=multiplier, max_delta=max_delta, cap=cap
        )
    if schedule.strategy == "per_games_interval":
        ticks = cumulative_games // schedule.games_interval
        weight = initial_weight
        for _ in range(ticks):
            if weight >= cap:
                return cap
            weight = _ramp_step(
                weight, multiplier=multiplier, max_delta=max_delta, cap=cap
            )
        return weight
    raise ValueError(
        f"Unknown nn_value_weight schedule strategy: {schedule.strategy!r}"
    )
