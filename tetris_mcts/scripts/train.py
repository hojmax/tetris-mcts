"""Training script for Tetris AlphaZero."""

import structlog
import torch
from dataclasses import dataclass
from simple_parsing import parse

from pathlib import Path
from tetris_mcts.config import TrainingConfig, setup_run_directory
from tetris_mcts.ml.training import Trainer
import json

import wandb

logger = structlog.get_logger()


def get_best_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ScriptArgs:
    """Training script arguments."""

    # Training config (all hyperparameters)
    training: TrainingConfig

    # Runtime
    device: str = "auto"  # Device to use (auto/cpu/cuda/mps)
    resume_dir: Path | None = (  # Resume from existing run dir (e.g., training_runs/v0)
        None
    )
    no_wandb: bool = False  # Disable WandB logging


def main(args: ScriptArgs) -> None:
    config = args.training

    # Set up run directory
    if args.resume_dir:
        # Resume from existing directory
        if not args.resume_dir.exists():
            logger.error("Resume directory does not exist", path=str(args.resume_dir))
            return
        config = setup_run_directory(config, run_dir=args.resume_dir)
        logger.info("Resuming training run", run_dir=str(config.run_dir))
    else:
        # Create new versioned run
        config = setup_run_directory(config)
        logger.info("Created new training run", run_dir=str(config.run_dir))

    # Auto-detect device if set to "auto"
    device = get_best_device() if args.device == "auto" else args.device
    logger.info("Using device", device=device)

    trainer = Trainer(config, device=device)

    logger.info(
        "Model created",
        parameters=sum(p.numel() for p in trainer.model.parameters()),
    )

    if args.resume_dir:
        if trainer.load():
            logger.info("Resumed from checkpoint", step=trainer.step)
        else:
            logger.info("No checkpoint found, starting fresh")

    log_to_wandb = not args.no_wandb

    if log_to_wandb:
        # Use config's JSON serialization for wandb config
        wandb_config = json.loads(config.to_json())
        wandb_config["device"] = device
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=wandb_config,
        )

    logger.info("Starting training with Rust game generation")
    trainer.train(log_to_wandb=log_to_wandb)

    logger.info("Training complete")


if __name__ == "__main__":
    args = parse(ScriptArgs)
    main(args)
