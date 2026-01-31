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
    iterations: int = 100  # Number of training iterations
    games_per_iter: int = 50  # Games to play per iteration
    train_steps_per_iter: int = 500  # Training steps per iteration

    # Parallel training
    parallel: bool = False  # Use parallel Rust game generation
    total_steps: int = 100000  # Total steps for parallel training
    model_sync_interval: int = 1000  # Steps between model exports (parallel mode)

    # MCTS settings
    simulations: int = 100  # MCTS simulations per move
    temperature: float = 1.0  # Temperature for action selection

    # Network settings
    batch_size: int = 256  # Training batch size
    lr: float = 0.001  # Learning rate

    # Buffer settings
    buffer_size: int = 100000  # Replay buffer size
    min_buffer: int = 1000  # Minimum buffer size before training

    # Logging/checkpoints
    checkpoint_dir: Path = (
        Path(__file__).parent.parent / "checkpoints"
    )  # Directory for checkpoints
    data_dir: Path = Path(__file__).parent.parent / "data"  # Directory for game data
    checkpoint_interval: int = 10  # Save checkpoint every N iterations
    eval_interval: int = 10  # Evaluate every N iterations
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
        num_games_per_iteration=args.games_per_iter,
        buffer_size=args.buffer_size,
        min_buffer_size=args.min_buffer,
        num_iterations=args.iterations,
        training_steps_per_iter=args.train_steps_per_iter,
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
                "iterations": args.iterations,
                "games_per_iter": args.games_per_iter,
                "train_steps_per_iter": args.train_steps_per_iter,
                "simulations": args.simulations,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "buffer_size": args.buffer_size,
            },
        )

    if args.parallel:
        logger.info("Starting parallel training with Rust game generation")
        trainer.train_parallel(
            num_steps=args.total_steps,
            model_sync_interval=args.model_sync_interval,
        )
    else:
        logger.info("Starting training")
        for i in range(config.num_iterations):
            logger.info(
                "Starting iteration",
                iteration=i + 1,
                total=config.num_iterations,
            )

            metrics = trainer.train_iteration(log_to_wandb=log_to_wandb)

            log_data = {"buffer_size": metrics["buffer_size"]}
            if "avg_loss" in metrics:
                log_data.update(
                    avg_loss=round(float(metrics["avg_loss"]), 4),
                    avg_policy_loss=round(float(metrics["avg_policy_loss"]), 4),
                    avg_value_loss=round(float(metrics["avg_value_loss"]), 4),
                )
            if "eval_avg_attack" in metrics:
                log_data.update(
                    eval_avg_attack=round(float(metrics["eval_avg_attack"]), 2),
                    eval_avg_moves=round(float(metrics["eval_avg_moves"]), 1),
                )
            logger.info("Iteration complete", **log_data)

        trainer.save()

    logger.info("Training complete")


if __name__ == "__main__":
    args = parse(ScriptArgs)
    main(args)
