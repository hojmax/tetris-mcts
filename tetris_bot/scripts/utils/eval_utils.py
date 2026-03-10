from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from tetris_bot.constants import (
    CHECKPOINT_DIRNAME,
    CONFIG_FILENAME,
    LATEST_ONNX_FILENAME,
)


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    data = json.loads(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Run config must be a JSON object: {config_path}")
    return data


def resolve_config_value(
    cli_override: int | float | None,
    run_config: dict,
    key: str,
    default: int | float,
) -> int | float:
    if cli_override is not None:
        return cli_override
    value = run_config.get(key, default)
    if value is None:
        raise ValueError(f"Config value for {key} cannot be None")
    return value


def resolve_model_path(model_path_override: Path | None, run_dir: Path) -> Path:
    model_path = (
        model_path_override
        if model_path_override is not None
        else run_dir / CHECKPOINT_DIRNAME / LATEST_ONNX_FILENAME
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return model_path


def compute_attack_stats(seeds: list[int], result: object) -> tuple[float, float]:
    """Return (attack_std, attack_sem) for an evaluation result."""
    attack_values = [int(attack) for attack, _ in result.game_results]  # type: ignore[attr-defined]
    attack_std = float(statistics.pstdev(attack_values)) if len(attack_values) > 1 else 0.0
    attack_sem = attack_std / math.sqrt(len(attack_values)) if attack_values else 0.0
    return attack_std, attack_sem
