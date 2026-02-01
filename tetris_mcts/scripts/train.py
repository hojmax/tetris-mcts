"""Training script for Tetris AlphaZero."""

import structlog
import torch
from dataclasses import dataclass
from simple_parsing import parse

from tetris_mcts.config import TrainingConfig
from tetris_mcts.ml.training import Trainer

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
    resume: bool = False  # Resume from latest checkpoint
    no_wandb: bool = False  # Disable WandB logging


def main(args: ScriptArgs) -> None:
    config = args.training

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect device if set to "auto"
    device = get_best_device() if args.device == "auto" else args.device
    logger.info("Using device", device=device)

    trainer = Trainer(config, device=device)

    logger.info(
        "Model created",
        parameters=sum(p.numel() for p in trainer.model.parameters()),
    )

    if args.resume:
        if trainer.load():
            logger.info("Resumed from checkpoint", step=trainer.step)
        else:
            logger.info("No checkpoint found, starting fresh")

    log_to_wandb = not args.no_wandb

    if log_to_wandb:
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config={
                # Training
                "total_steps": config.total_steps,
                "model_sync_interval": config.model_sync_interval,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "lr_schedule": config.lr_schedule,
                "lr_decay_steps": config.lr_decay_steps,
                # Network
                "conv_filters": config.conv_filters,
                "fc_hidden": config.fc_hidden,
                # MCTS
                "num_simulations": config.num_simulations,
                "temperature": config.temperature,
                "dirichlet_alpha": config.dirichlet_alpha,
                "dirichlet_epsilon": config.dirichlet_epsilon,
                # Buffer
                "buffer_size": config.buffer_size,
                "min_buffer_size": config.min_buffer_size,
                "games_per_save": config.games_per_save,
                # Intervals
                "eval_interval": config.eval_interval,
                "checkpoint_interval": config.checkpoint_interval,
                "log_interval": config.log_interval,
                # Device
                "device": device,
            },
        )

    logger.info("Starting training with Rust game generation")
    trainer.train(log_to_wandb=log_to_wandb)

    logger.info("Training complete")


if __name__ == "__main__":
    args = parse(ScriptArgs)
    main(args)
