"""Training script for Tetris AlphaZero."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
import structlog
import torch
import wandb
from simple_parsing import parse

from tetris_mcts.config import (
    CHECKPOINT_DIRNAME,
    LATEST_CHECKPOINT_FILENAME,
    TRAINING_DATA_FILENAME,
    TrainingConfig,
    setup_run_directory,
)
from tetris_mcts.ml.training import Trainer
from tetris_mcts.ml.weights import load_checkpoint

logger = structlog.get_logger()


def get_best_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def initialize_or_update_wandb(config: TrainingConfig, device: str) -> None:
    wandb_config = json.loads(config.to_json())
    wandb_config["device"] = device

    if wandb.run is None:
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=wandb_config,
        )
        return

    wandb.config.update(wandb_config, allow_val_change=True)


@dataclass
class ScriptArgs:
    """Training script arguments."""

    # Training config (all hyperparameters)
    training: TrainingConfig

    # Runtime
    device: str = "auto"  # Device to use (auto/cpu/cuda/mps)
    resume_dir: (  # Bootstrap a new run from existing run dir
        Path | None
    ) = None  # Path(__file__).parent.parent / "training_runs" / "v17"
    resume_restore_optimizer_scheduler: bool = True  # If True, restore optimizer and scheduler from checkpoint when using resume_dir; if False, restore optimizer only and rebuild scheduler from current config
    init_checkpoint: Path | None = None  # Initialize model weights from checkpoint
    no_wandb: bool = False  # Disable WandB logging


def main(args: ScriptArgs) -> None:
    config = args.training
    resume_checkpoint: Path | None = None

    if args.resume_dir and args.init_checkpoint:
        logger.error("Cannot use resume_dir and init_checkpoint together")
        return

    # Set up run directory
    if args.resume_dir:
        # Bootstrap a new run from an existing run directory.
        source_run_dir = args.resume_dir
        if not source_run_dir.exists():
            logger.error("Resume directory does not exist", path=str(source_run_dir))
            return
        if not source_run_dir.is_dir():
            logger.error(
                "Resume directory is not a directory", path=str(source_run_dir)
            )
            return

        source_checkpoint = (
            source_run_dir / CHECKPOINT_DIRNAME / LATEST_CHECKPOINT_FILENAME
        )
        if not source_checkpoint.exists():
            logger.error(
                "Resume checkpoint does not exist",
                path=str(source_checkpoint),
            )
            return

        config = setup_run_directory(config)
        logger.info(
            "Created resumed training run",
            source_run_dir=str(source_run_dir),
            run_dir=str(config.run_dir),
            checkpoint=str(source_checkpoint),
        )

        # Copy replay buffer snapshot so Rust generator can preload it.
        source_training_data = source_run_dir / TRAINING_DATA_FILENAME
        if source_training_data.exists():
            assert config.data_dir is not None
            destination_training_data = config.data_dir / TRAINING_DATA_FILENAME
            shutil.copy2(source_training_data, destination_training_data)
            logger.info(
                "Copied replay buffer snapshot",
                source=str(source_training_data),
                destination=str(destination_training_data),
            )
        else:
            logger.warning(
                "Resume directory has no replay buffer snapshot",
                expected_path=str(source_training_data),
            )

        resume_checkpoint = source_checkpoint
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

    if resume_checkpoint is not None:
        load_optimizer = trainer.optimizer
        load_scheduler = (
            trainer.scheduler if args.resume_restore_optimizer_scheduler else None
        )
        state = load_checkpoint(
            resume_checkpoint,
            model=trainer.model,
            optimizer=load_optimizer,
            scheduler=load_scheduler,
        )
        checkpoint_step = state.get("step")
        if checkpoint_step is None:
            raise ValueError(f"Checkpoint is missing step: {resume_checkpoint}")
        trainer.step = int(checkpoint_step)
        if (
            trainer.scheduler is not None
            and not args.resume_restore_optimizer_scheduler
        ):
            trainer.align_scheduler_to_step(trainer.step)
        logger.info(
            "Initialized new run from checkpoint",
            checkpoint=str(resume_checkpoint),
            step=trainer.step,
            learning_rate=trainer.optimizer.param_groups[0]["lr"],
            lr_schedule=config.lr_schedule,
            restored_optimizer_scheduler=args.resume_restore_optimizer_scheduler,
        )
    elif args.init_checkpoint:
        if not args.init_checkpoint.exists():
            logger.error(
                "Init checkpoint does not exist", path=str(args.init_checkpoint)
            )
            return
        state = load_checkpoint(
            args.init_checkpoint, model=trainer.model, optimizer=None
        )
        logger.info(
            "Initialized model from checkpoint",
            path=str(args.init_checkpoint),
            checkpoint_step=state.get("step"),
        )

    log_to_wandb = not args.no_wandb

    if log_to_wandb:
        initialize_or_update_wandb(config, device)
        wandb.define_metric("trainer_step")
        wandb.define_metric("train/*", step_metric="trainer_step")
        wandb.define_metric("batch/*", step_metric="trainer_step")
        wandb.define_metric("eval/*", step_metric="trainer_step")
        wandb.define_metric("timing/*", step_metric="trainer_step")
        # Use game_number as x-axis for per-game metrics
        wandb.define_metric("game_number")
        wandb.define_metric("game/*", step_metric="game_number")

    logger.info("Starting training with Rust game generation")
    trainer.train(log_to_wandb=log_to_wandb)

    logger.info("Training complete")


if __name__ == "__main__":
    args = parse(ScriptArgs)
    main(args)
