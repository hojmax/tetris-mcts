"""Run directory setup and config serialization utilities."""

from __future__ import annotations

import os
from pathlib import Path

import structlog
import torch
import wandb

from tetris_bot.constants import (
    CHECKPOINT_DIRNAME,
    CONFIG_FILENAME,
    RUNTIME_OVERRIDES_FILENAME,
    TRAINING_RUNS_DIR,
)
from tetris_bot.ml.config import (
    RuntimeOverrides,
    TrainingConfig,
    save_runtime_overrides,
    save_training_config,
)
from tetris_bot.run_naming import generate_run_id

logger = structlog.get_logger()


def _allocate_unique_run_dir(base_dir: Path) -> Path:
    """Pick a fresh `<adjective>-<animal>-<timestamp>` dir under base_dir.

    Collisions are essentially impossible (same minute + same word pair),
    but we retry to be safe.
    """
    for _ in range(8):
        candidate = base_dir / generate_run_id()
        if not candidate.exists():
            return candidate
    raise RuntimeError(
        f"Failed to allocate a unique run directory under {base_dir} after 8 tries"
    )


def setup_run_directory(
    config: TrainingConfig,
    base_dir: Path = TRAINING_RUNS_DIR,
    run_dir: Path | None = None,
) -> TrainingConfig:
    if run_dir is None:
        base_dir.mkdir(parents=True, exist_ok=True)
        run_dir = _allocate_unique_run_dir(base_dir)

    checkpoint_dir = run_dir / CHECKPOINT_DIRNAME

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    config.run.run_dir = run_dir
    config.run.checkpoint_dir = checkpoint_dir
    config.run.data_dir = run_dir

    if config.run.run_name is None:
        config.run.run_name = run_dir.name

    save_training_config(config, run_dir / CONFIG_FILENAME)
    runtime_overrides_path = run_dir / RUNTIME_OVERRIDES_FILENAME
    if not runtime_overrides_path.exists():
        save_runtime_overrides(RuntimeOverrides(), runtime_overrides_path)

    return config


def apply_optimized_runtime_overrides(config: TrainingConfig) -> None:
    """Override `self_play.num_workers` from `TETRIS_OPT_NUM_WORKERS` if set.

    Populated by `make optimize` (sourced via the optimize-cache env file in
    the Makefile). Lets generator/trainer entrypoints auto-pick up the
    machine-tuned worker count without requiring `--num_workers` on every run.
    """
    workers_env = os.getenv("TETRIS_OPT_NUM_WORKERS")
    if workers_env is None or workers_env.strip() == "":
        return

    try:
        optimized_workers = int(workers_env)
    except ValueError as error:
        raise ValueError(
            f"TETRIS_OPT_NUM_WORKERS must be an integer (got {workers_env!r})"
        ) from error

    if optimized_workers <= 0:
        raise ValueError(
            f"TETRIS_OPT_NUM_WORKERS must be > 0 (got {optimized_workers})"
        )

    previous_workers = config.self_play.num_workers
    config.self_play.num_workers = optimized_workers
    logger.info(
        "Applied optimized self-play worker override from environment",
        previous_num_workers=previous_workers,
        optimized_num_workers=optimized_workers,
    )


def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def initialize_or_update_wandb(
    config: TrainingConfig, device: str, resume_dir: Path | None = None
) -> None:
    wandb_config = config.model_dump(mode="json")
    wandb_config["device"] = device
    if resume_dir is not None:
        wandb_config["resume_dir"] = str(resume_dir)

    if wandb.run is None:
        wandb.init(
            project=config.run.project_name,
            name=config.run.run_name,
            config=wandb_config,
        )
        return

    wandb.config.update(wandb_config, allow_val_change=True)


def configure_wandb(
    config: TrainingConfig, device: str, resume_dir: Path | None = None
) -> None:
    initialize_or_update_wandb(config, device, resume_dir=resume_dir)
    wandb.define_metric("trainer_step")
    wandb.define_metric("wall_time_hours")
    for ns in [
        "train/*",
        "batch/*",
        "eval/*",
        "timing/*",
        "replay/*",
        "throughput/*",
        "incumbent/*",
        "model_gate/*",
        "runtime_override/*",
    ]:
        wandb.define_metric(ns, step_metric="trainer_step")
    wandb.define_metric("model_gate_time/*", step_metric="wall_time_hours")
    wandb.define_metric("game_number")
    wandb.define_metric("game/*", step_metric="game_number")
    wandb.define_metric("game_time/*", step_metric="wall_time_hours")
