from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from tetris_bot.constants import CHECKPOINT_DIRNAME, CONFIG_FILENAME, LATEST_CHECKPOINT_FILENAME


def load_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(f"Run config not found: {config_path}")

    data = json.loads(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Run config must be a JSON object: {config_path}")
    return data


def load_self_play_config(run_dir: Path) -> dict[str, Any]:
    data = load_run_config(run_dir)
    self_play = data.get("self_play", data)
    if not isinstance(self_play, dict):
        raise ValueError(f"self_play config must be a JSON object: {run_dir / CONFIG_FILENAME}")
    return dict(self_play)


def default_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / CHECKPOINT_DIRNAME / LATEST_CHECKPOINT_FILENAME


def load_checkpoint_state(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint state must be a dict: {checkpoint_path}")
    return state


def apply_checkpoint_search_overrides(
    self_play_config: dict[str, Any],
    checkpoint_state: dict[str, Any],
) -> dict[str, Any]:
    resolved = dict(self_play_config)

    incumbent_nn_value_weight = checkpoint_state.get("incumbent_nn_value_weight")
    if incumbent_nn_value_weight is not None:
        restored_nn_value_weight = float(incumbent_nn_value_weight)
        if (
            not math.isfinite(restored_nn_value_weight)
            or restored_nn_value_weight < 0.0
        ):
            raise ValueError(
                "Checkpoint incumbent_nn_value_weight must be finite and >= 0 "
                f"(got {restored_nn_value_weight})"
            )
        resolved["nn_value_weight"] = restored_nn_value_weight

    incumbent_death_penalty = checkpoint_state.get("incumbent_death_penalty")
    incumbent_overhang_penalty_weight = checkpoint_state.get(
        "incumbent_overhang_penalty_weight"
    )
    if (
        incumbent_death_penalty is not None
        and incumbent_overhang_penalty_weight is not None
    ):
        restored_death_penalty = float(incumbent_death_penalty)
        restored_overhang_penalty_weight = float(incumbent_overhang_penalty_weight)
        if (
            not math.isfinite(restored_death_penalty)
            or restored_death_penalty < 0.0
        ):
            raise ValueError(
                "Checkpoint incumbent_death_penalty must be finite and >= 0 "
                f"(got {restored_death_penalty})"
            )
        if (
            not math.isfinite(restored_overhang_penalty_weight)
            or restored_overhang_penalty_weight < 0.0
        ):
            raise ValueError(
                "Checkpoint incumbent_overhang_penalty_weight must be finite and >= 0 "
                f"(got {restored_overhang_penalty_weight})"
            )
        resolved["death_penalty"] = restored_death_penalty
        resolved["overhang_penalty_weight"] = restored_overhang_penalty_weight
    else:
        nn_value_weight = float(
            resolved.get("nn_value_weight", self_play_config.get("nn_value_weight", 0.0))
        )
        nn_value_weight_cap = float(
            resolved.get(
                "nn_value_weight_cap",
                self_play_config.get("nn_value_weight_cap", nn_value_weight),
            )
        )
        if nn_value_weight >= nn_value_weight_cap:
            resolved["death_penalty"] = 0.0
            resolved["overhang_penalty_weight"] = 0.0

    return resolved


def load_effective_self_play_config(
    run_dir: Path,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    self_play_config = load_self_play_config(run_dir)
    resolved_checkpoint_path = (
        checkpoint_path if checkpoint_path is not None else default_checkpoint_path(run_dir)
    )
    if not resolved_checkpoint_path.exists():
        return self_play_config

    checkpoint_state = load_checkpoint_state(resolved_checkpoint_path)
    return apply_checkpoint_search_overrides(self_play_config, checkpoint_state)
