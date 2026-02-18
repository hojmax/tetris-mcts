from __future__ import annotations

import torch

from tetris_bot.ml.aux_features import AUX_FEATURE_LAYOUT


def compute_batch_feature_metrics(
    aux: torch.Tensor,
    value_targets: torch.Tensor,
    overhang_fields: torch.Tensor,
    masks: torch.Tensor,
) -> dict[str, float]:
    layout = AUX_FEATURE_LAYOUT
    row_fill_counts = aux[:, layout.row_fill_counts]
    max_column_heights = aux[:, layout.max_column_height]
    total_blocks = aux[:, layout.total_blocks]
    bumpiness = aux[:, layout.bumpiness]
    holes = aux[:, layout.holes]

    return {
        "batch/value_target_mean": value_targets.mean().item(),
        "batch/valid_actions_mean": masks.sum(dim=1).mean().item(),
        "batch/board_fill_mean": total_blocks.mean().item(),
        "batch/max_height_mean": max_column_heights.mean().item(),
        "batch/row_fill_mean": row_fill_counts.mean().item(),
        "batch/bumpiness_mean": bumpiness.mean().item(),
        "batch/holes_mean": holes.mean().item(),
        "batch/overhang_fields_mean": overhang_fields.mean().item(),
    }


def summarize_completed_games(
    completed_games: list[tuple[int, dict[str, float | int]]],
) -> dict[str, float]:
    if not completed_games:
        return {}

    attack_sum = 0.0
    line_sum = 0.0
    episode_length_sum = 0.0
    holds_sum = 0.0
    max_attack = float("-inf")
    max_lines = float("-inf")

    for _, game_stats in completed_games:
        total_attack = float(game_stats["total_attack"])
        total_lines = float(game_stats["total_lines"])
        episode_length = float(game_stats["episode_length"])
        holds = float(game_stats["holds"])
        if episode_length <= 0.0:
            raise ValueError(
                "Invalid episode_length while aggregating completed games: "
                f"{episode_length}"
            )
        attack_sum += total_attack
        line_sum += total_lines
        episode_length_sum += episode_length
        holds_sum += holds
        max_attack = max(max_attack, total_attack)
        max_lines = max(max_lines, total_lines)

    completed_count = float(len(completed_games))
    first_game_number = float(completed_games[0][0])
    last_game_number = float(completed_games[-1][0])
    return {
        "replay/completed_games_logged": completed_count,
        "replay/completed_games_first_number": first_game_number,
        "replay/completed_games_last_number": last_game_number,
        "replay/completed_games_avg_attack": attack_sum / completed_count,
        "replay/completed_games_avg_lines": line_sum / completed_count,
        "replay/completed_games_avg_moves": episode_length_sum / completed_count,
        "replay/completed_games_max_attack": max_attack,
        "replay/completed_games_max_lines": max_lines,
        "replay/completed_games_avg_attack_per_move": attack_sum / episode_length_sum,
        "replay/completed_games_avg_hold_rate": holds_sum / episode_length_sum,
    }
