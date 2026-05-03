"""Compute the schedule-driven scale for `death_penalty` / `overhang_penalty_weight`.

Single scalar in [0, 1] applied uniformly to both penalties at every site that
mutates the incumbent search state (no-gating direct sync, gating candidate
registration, candidate eval baseline, resumed-run baseline).
"""

from __future__ import annotations

from tetris_bot.ml.config import PenaltyScheduleConfig


def compute_penalty_scale(
    schedule: PenaltyScheduleConfig,
    *,
    cumulative_games: int,
    nn_value_weight: float,
    nn_value_weight_cap: float,
) -> float:
    if schedule.strategy == "gated":
        return 1.0 if nn_value_weight < nn_value_weight_cap else 0.0
    if schedule.strategy == "constant_then_linear":
        if cumulative_games < schedule.hold_games:
            return 1.0
        decayed = cumulative_games - schedule.hold_games
        if decayed >= schedule.decay_games:
            return 0.0
        return 1.0 - decayed / schedule.decay_games
    raise ValueError(f"Unknown penalty schedule strategy: {schedule.strategy!r}")


def scaled_penalties(
    schedule: PenaltyScheduleConfig,
    *,
    death_penalty: float,
    overhang_penalty_weight: float,
    cumulative_games: int,
    nn_value_weight: float,
    nn_value_weight_cap: float,
) -> tuple[float, float]:
    scale = compute_penalty_scale(
        schedule,
        cumulative_games=cumulative_games,
        nn_value_weight=nn_value_weight,
        nn_value_weight_cap=nn_value_weight_cap,
    )
    return death_penalty * scale, overhang_penalty_weight * scale
