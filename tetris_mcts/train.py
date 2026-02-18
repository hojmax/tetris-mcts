"""Training script for Tetris AlphaZero."""

import shutil
from dataclasses import dataclass
from pathlib import Path

import structlog
import torch
from simple_parsing import parse

from tetris_mcts.constants import (
    CHECKPOINT_DIRNAME,
    INCUMBENT_ONNX_FILENAME,
    LATEST_CHECKPOINT_FILENAME,
    TRAINING_DATA_FILENAME,
)
from tetris_mcts.config import TrainingConfig
from tetris_mcts.run_setup import configure_wandb, get_best_device, setup_run_directory
from tetris_mcts.ml.training import Trainer, copy_model_artifact_bundle
from tetris_mcts.ml.weights import load_checkpoint

logger = structlog.get_logger()


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


def setup_run(
    args: ScriptArgs, config: TrainingConfig
) -> tuple[TrainingConfig, Path | None, Path | None]:
    """Set up training run directory. Returns (config, resume_checkpoint, resume_incumbent_model_path)."""
    if not args.resume_dir:
        config = setup_run_directory(config)
        logger.info("Created new training run", run_dir=str(config.run_dir))
        return config, None, None

    source_run_dir = args.resume_dir
    if not source_run_dir.exists():
        raise FileNotFoundError(f"Resume directory does not exist: {source_run_dir}")
    if not source_run_dir.is_dir():
        raise NotADirectoryError(
            f"Resume directory is not a directory: {source_run_dir}"
        )

    source_checkpoint = source_run_dir / CHECKPOINT_DIRNAME / LATEST_CHECKPOINT_FILENAME
    if not source_checkpoint.exists():
        raise FileNotFoundError(
            f"Resume checkpoint does not exist: {source_checkpoint}"
        )

    config = setup_run_directory(config)
    logger.info(
        "Created resumed training run",
        source_run_dir=str(source_run_dir),
        run_dir=str(config.run_dir),
        checkpoint=str(source_checkpoint),
    )

    source_training_data = source_run_dir / TRAINING_DATA_FILENAME
    if source_training_data.exists():
        assert config.data_dir is not None
        destination = config.data_dir / TRAINING_DATA_FILENAME
        shutil.copy2(source_training_data, destination)
        logger.info(
            "Copied replay buffer snapshot",
            source=str(source_training_data),
            destination=str(destination),
        )
    else:
        logger.warning(
            "Resume directory has no replay buffer snapshot",
            expected_path=str(source_training_data),
        )

    source_incumbent = source_run_dir / CHECKPOINT_DIRNAME / INCUMBENT_ONNX_FILENAME
    resume_incumbent_model_path: Path | None = None
    if source_incumbent.exists():
        assert config.checkpoint_dir is not None
        destination_incumbent = config.checkpoint_dir / INCUMBENT_ONNX_FILENAME
        copy_model_artifact_bundle(source_incumbent, destination_incumbent)
        resume_incumbent_model_path = destination_incumbent
        logger.info(
            "Copied incumbent model artifact bundle for resume",
            source=str(source_incumbent),
            destination=str(destination_incumbent),
        )
    else:
        logger.warning(
            "Resume directory has no incumbent model artifact bundle",
            expected_path=str(source_incumbent),
        )

    return config, source_checkpoint, resume_incumbent_model_path


def restore_trainer_from_checkpoint(
    trainer: Trainer,
    args: ScriptArgs,
    config: TrainingConfig,
    checkpoint: Path,
    incumbent_model_path: Path | None,
) -> None:
    load_scheduler = (
        trainer.scheduler if args.resume_restore_optimizer_scheduler else None
    )
    state = load_checkpoint(
        checkpoint,
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=load_scheduler,
    )

    checkpoint_step = state.get("step")
    if checkpoint_step is None:
        raise ValueError(f"Checkpoint is missing step: {checkpoint}")
    trainer.step = int(checkpoint_step)

    if trainer.scheduler is not None and not args.resume_restore_optimizer_scheduler:
        trainer.align_scheduler_to_step(trainer.step)

    incumbent_uses_network = state.get("incumbent_uses_network")
    if incumbent_uses_network is None:
        config.bootstrap_without_network = False
        start_with_network = True
        logger.warning(
            "Checkpoint missing incumbent network state; starting self-play with network",
            checkpoint=str(checkpoint),
            start_with_network=True,
        )
    else:
        start_with_network = bool(incumbent_uses_network)
        config.bootstrap_without_network = not start_with_network
        logger.info(
            "Restored self-play startup mode from checkpoint",
            checkpoint=str(checkpoint),
            start_with_network=start_with_network,
            incumbent_uses_network=start_with_network,
        )

    incumbent_nn_value_weight = state.get("incumbent_nn_value_weight")
    if incumbent_nn_value_weight is None:
        logger.warning(
            "Checkpoint missing incumbent nn_value_weight; using config value",
            checkpoint=str(checkpoint),
            nn_value_weight=config.nn_value_weight,
        )
    else:
        restored_nn_value_weight = float(incumbent_nn_value_weight)
        if restored_nn_value_weight < 0.0:
            raise ValueError(
                f"Checkpoint incumbent_nn_value_weight must be >= 0 (got {restored_nn_value_weight})"
            )
        config.nn_value_weight = restored_nn_value_weight
        logger.info(
            "Restored incumbent nn_value_weight from checkpoint",
            checkpoint=str(checkpoint),
            nn_value_weight=restored_nn_value_weight,
        )

    incumbent_eval_avg_attack = state.get("incumbent_eval_avg_attack")
    if incumbent_eval_avg_attack is None:
        restored_incumbent_eval_avg_attack = 0.0
        logger.warning(
            "Checkpoint missing incumbent_eval_avg_attack; using zero",
            checkpoint=str(checkpoint),
        )
    else:
        restored_incumbent_eval_avg_attack = float(incumbent_eval_avg_attack)
        if restored_incumbent_eval_avg_attack < 0.0:
            raise ValueError(
                f"Checkpoint incumbent_eval_avg_attack must be >= 0 (got {restored_incumbent_eval_avg_attack})"
            )
    trainer.initial_incumbent_eval_avg_attack = restored_incumbent_eval_avg_attack
    logger.info(
        "Restored incumbent eval avg attack from checkpoint",
        checkpoint=str(checkpoint),
        incumbent_eval_avg_attack=restored_incumbent_eval_avg_attack,
    )

    if start_with_network and incumbent_model_path is not None:
        trainer.initial_incumbent_model_path = incumbent_model_path
        logger.info(
            "Configured resumed incumbent model artifact for generator startup",
            path=str(incumbent_model_path),
        )
    elif start_with_network:
        logger.warning(
            "Resuming with network but no incumbent model artifact bundle; falling back to trainer checkpoint model",
            expected_path=(
                str(config.checkpoint_dir / INCUMBENT_ONNX_FILENAME)
                if config.checkpoint_dir is not None
                else None
            ),
        )

    logger.info(
        "Initialized new run from checkpoint",
        checkpoint=str(checkpoint),
        step=trainer.step,
        learning_rate=trainer.optimizer.param_groups[0]["lr"],
        lr_schedule=config.lr_schedule,
        restored_optimizer_scheduler=args.resume_restore_optimizer_scheduler,
    )


def main(args: ScriptArgs) -> None:
    if args.resume_dir and args.init_checkpoint:
        raise ValueError("Cannot use resume_dir and init_checkpoint together")

    config, resume_checkpoint, resume_incumbent_model_path = setup_run(
        args, args.training
    )

    device = get_best_device() if args.device == "auto" else args.device
    logger.info("Using device", device=device)

    trainer = Trainer(config, device=device)
    logger.info(
        "Model created",
        parameters=sum(p.numel() for p in trainer.model.parameters()),
    )

    if resume_checkpoint is not None:
        restore_trainer_from_checkpoint(
            trainer, args, config, resume_checkpoint, resume_incumbent_model_path
        )
    elif args.init_checkpoint:
        if not args.init_checkpoint.exists():
            raise FileNotFoundError(
                f"Init checkpoint does not exist: {args.init_checkpoint}"
            )
        state = load_checkpoint(
            args.init_checkpoint, model=trainer.model, optimizer=None
        )
        logger.info(
            "Initialized model from checkpoint",
            path=str(args.init_checkpoint),
            checkpoint_step=state.get("step"),
        )

    if not args.no_wandb:
        configure_wandb(config, device)

    torch.set_float32_matmul_precision("high")
    logger.info("Starting training with Rust game generation")
    trainer.train(log_to_wandb=not args.no_wandb)

    logger.info("Training complete")


if __name__ == "__main__":
    args = parse(ScriptArgs)
    main(args)
