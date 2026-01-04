from pathlib import Path

import structlog
from pydantic.dataclasses import dataclass
from simple_parsing import parse

from tetris_mcts.ml.training import Trainer, TrainingConfig

logger = structlog.get_logger()


@dataclass
class ScriptArgs:
    # Training settings
    iterations: int = 100  # Number of training iterations
    games_per_iter: int = 50  # Games to play per iteration
    train_steps_per_iter: int = 500  # Training steps per iteration

    # MCTS settings
    simulations: int = 100  # MCTS simulations per move
    temperature: float = 1.0  # Temperature for action selection
    no_mcts: bool = False  # Use random policy instead of MCTS (for debugging)

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
    checkpoint_interval: int = 10  # Save checkpoint every N iterations
    eval_interval: int = 10  # Evaluate every N iterations
    log_interval: int = 100  # Log every N training steps

    # WandB
    project: str = "tetris-alphazero"  # WandB project name
    run_name: str | None = None  # WandB run name
    no_wandb: bool = False  # Disable WandB logging

    # Device
    device: str = "cpu"  # Device to use (cpu/cuda/mps)

    # Resume
    resume: bool = False  # Resume from latest checkpoint


def main(args: ScriptArgs) -> None:
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
        project_name=args.project,
        run_name=args.run_name,
    )

    trainer = Trainer(config, device=args.device)

    logger.info(
        "Model created",
        parameters=sum(p.numel() for p in trainer.model.parameters()),
    )

    if args.resume:
        if trainer.load():
            logger.info("Resumed from checkpoint", step=trainer.step)
        else:
            logger.info("No checkpoint found, starting fresh")

    use_mcts = not args.no_mcts
    log_to_wandb = not args.no_wandb

    logger.info("Starting training")

    for i in range(config.num_iterations):
        logger.info(
            "Starting iteration",
            iteration=i + 1,
            total=config.num_iterations,
        )

        metrics = trainer.train_iteration(log_to_wandb=log_to_wandb, use_mcts=use_mcts)

        log_data = {"buffer_size": metrics["buffer_size"]}
        if "avg_loss" in metrics:
            log_data.update(
                avg_loss=round(metrics["avg_loss"], 4),
                avg_policy_loss=round(metrics["avg_policy_loss"], 4),
                avg_value_loss=round(metrics["avg_value_loss"], 4),
            )
        logger.info("Iteration complete", **log_data)

    trainer.save()
    logger.info("Training complete")


if __name__ == "__main__":
    args = parse(ScriptArgs)
    main(args)
