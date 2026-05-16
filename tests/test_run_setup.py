from __future__ import annotations

from tetris_bot.constants import DEFAULT_CONFIG_PATH
from tetris_bot.ml.config import load_training_config
from tetris_bot.run_setup import configure_wandb


def test_configure_wandb_defines_time_based_game_metrics(
    monkeypatch,
) -> None:
    calls: list[tuple[str, str | None]] = []

    monkeypatch.setattr(
        "tetris_bot.run_setup.initialize_or_update_wandb",
        lambda config, device, resume_dir=None: None,
    )
    monkeypatch.setattr(
        "tetris_bot.run_setup.wandb.define_metric",
        lambda name, step_metric=None: calls.append((name, step_metric)),
    )

    configure_wandb(load_training_config(DEFAULT_CONFIG_PATH), device="cpu")

    assert ("game/*", "game_number") in calls
    assert ("game_time/*", "wall_time_hours") in calls
