from pathlib import Path

import structlog
import torch
from pydantic.dataclasses import dataclass
from simple_parsing import parse

from tetris_mcts.ml.training import Trainer, TrainingConfig

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
    # Training settings
    total_steps: int = 100000  # Total training steps
    model_sync_interval: int = 1000  # Steps between model exports

    # MCTS settings
    simulations: int = 100  # MCTS simulations per move
    temperature: float = 1.0  # Temperature for action selection

    # Network settings
    batch_size: int = 256  # Training batch size
    lr: float = 0.001  # Learning rate

    # Buffer settings
    buffer_size: int = 100000  # Replay buffer size
    min_buffer: int = 1000  # Minimum buffer size before training
    games_per_save: int = 100  # Games between disk saves (0 to disable)

    # Logging/checkpoints (outputs/ is at project root, next to tetris_mcts/)
    checkpoint_dir: Path = (
        Path(__file__).parent.parent.parent / "outputs" / "checkpoints"
    )  # Directory for checkpoints
    data_dir: Path = (
        Path(__file__).parent.parent.parent / "outputs" / "data"
    )  # Directory for game data
    checkpoint_interval: int = 100  # Save checkpoint every N iterations
    eval_interval: int = 100  # Evaluate every N iterations
    log_interval: int = 100  # Log every N training steps

    # WandB
    project: str = "tetris-alphazero"  # WandB project name
    run_name: str | None = None  # WandB run name
    no_wandb: bool = False  # Disable WandB logging

    # Device
    device: str = "auto"  # Device to use (auto/cpu/cuda/mps)

    # Resume
    resume: bool = False  # Resume from latest checkpoint


def main(args: ScriptArgs) -> None:
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect device if set to "auto"
    device = get_best_device() if args.device == "auto" else args.device
    logger.info("Using device", device=device)

    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_simulations=args.simulations,
        temperature=args.temperature,
        buffer_size=args.buffer_size,
        min_buffer_size=args.min_buffer,
        games_per_save=args.games_per_save,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        checkpoint_dir=str(args.checkpoint_dir),
        data_dir=str(args.data_dir),
        project_name=args.project,
        run_name=args.run_name,
    )

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
            project=args.project,
            name=args.run_name,
            config={
                # Training
                "total_steps": args.total_steps,
                "model_sync_interval": args.model_sync_interval,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": config.weight_decay,
                # MCTS
                "simulations": args.simulations,
                "temperature": args.temperature,
                "dirichlet_alpha": config.dirichlet_alpha,
                "dirichlet_epsilon": config.dirichlet_epsilon,
                # Buffer
                "buffer_size": args.buffer_size,
                "min_buffer": args.min_buffer,
                "games_per_save": args.games_per_save,
                # Intervals
                "eval_interval": args.eval_interval,
                "checkpoint_interval": args.checkpoint_interval,
                "log_interval": args.log_interval,
                # Device
                "device": device,
            },
        )

    logger.info("Starting training with Rust game generation")
    trainer.train(
        num_steps=args.total_steps,
        model_sync_interval=args.model_sync_interval,
        log_to_wandb=log_to_wandb,
    )

    logger.info("Training complete")


if __name__ == "__main__":
    args = parse(ScriptArgs)
    main(args)
