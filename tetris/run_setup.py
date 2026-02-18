"""Run directory setup and config serialization utilities."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
import wandb

from tetris.ml.config import TrainingConfig
from tetris.constants import CHECKPOINT_DIRNAME, CONFIG_FILENAME, TRAINING_RUNS_DIR


def config_to_json(config: TrainingConfig) -> str:
    d = asdict(config)
    for key in ["run_dir", "checkpoint_dir", "data_dir"]:
        if d["run"][key] is not None:
            d["run"][key] = str(d["run"][key])
    return json.dumps(d, indent=2)


def save_config(config: TrainingConfig, path: Path) -> None:
    path.write_text(config_to_json(config))


def get_next_version(base_dir: Path) -> int:
    if not base_dir.exists():
        return 0
    existing = [
        int(d.name[1:])
        for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
    ]
    return max(existing, default=-1) + 1


def setup_run_directory(
    config: TrainingConfig,
    base_dir: Path = TRAINING_RUNS_DIR,
    run_dir: Path | None = None,
) -> TrainingConfig:
    if run_dir is None:
        version = get_next_version(base_dir)
        run_dir = base_dir / f"v{version}"

    checkpoint_dir = run_dir / CHECKPOINT_DIRNAME

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    config.run.run_dir = run_dir
    config.run.checkpoint_dir = checkpoint_dir
    config.run.data_dir = run_dir

    if config.run.run_name is None:
        config.run.run_name = run_dir.name

    save_config(config, run_dir / CONFIG_FILENAME)

    return config


def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def initialize_or_update_wandb(config: TrainingConfig, device: str) -> None:
    wandb_config = json.loads(config_to_json(config))
    wandb_config["device"] = device

    if wandb.run is None:
        wandb.init(
            project=config.run.project_name,
            name=config.run.run_name,
            config=wandb_config,
        )
        return

    wandb.config.update(wandb_config, allow_val_change=True)


def configure_wandb(config: TrainingConfig, device: str) -> None:
    initialize_or_update_wandb(config, device)
    wandb.define_metric("trainer_step")
    for ns in [
        "train/*",
        "batch/*",
        "eval/*",
        "timing/*",
        "replay/*",
        "throughput/*",
        "incumbent/*",
        "model_gate/*",
    ]:
        wandb.define_metric(ns, step_metric="trainer_step")
    wandb.define_metric("game_number")
    wandb.define_metric("game/*", step_metric="game_number")
