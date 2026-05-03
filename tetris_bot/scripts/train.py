"""Training script for Tetris AlphaZero."""

import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import structlog
import torch
from dotenv import load_dotenv
from simple_parsing import parse

# Loaded as early as possible so subsequent imports (e.g., r2_sync) can read
# R2_ENDPOINT_URL / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY without the user
# having to `source .env` manually before each run.
load_dotenv()

from tetris_bot.constants import (  # noqa: E402
    CHECKPOINT_DIRNAME,
    DEFAULT_CONFIG_PATH,
    INCUMBENT_ONNX_FILENAME,
    LATEST_CHECKPOINT_FILENAME,
    RUNTIME_OVERRIDES_FILENAME,
    TRAINING_DATA_FILENAME,
)
from tetris_bot.ml.config import (  # noqa: E402
    ResolvedRuntimeOptimizerOverrides,
    ResolvedRuntimeOverrides,
    ResolvedRuntimeRunOverrides,
    TrainingConfig,
    load_training_config,
)
from tetris_bot.run_setup import (  # noqa: E402
    configure_wandb,
    get_best_device,
    setup_run_directory,
)
from tetris_bot.ml.trainer import Trainer  # noqa: E402
from tetris_bot.ml.artifacts import copy_model_artifact_bundle  # noqa: E402
from tetris_bot.ml.wandb_resume import (  # noqa: E402
    WandbResumeSource,
    prepare_wandb_resume_source,
)
from tetris_bot.ml.weights import load_checkpoint  # noqa: E402

logger = structlog.get_logger()


@dataclass
class ScriptArgs:
    """Training script arguments."""

    config: Path = DEFAULT_CONFIG_PATH

    # Runtime
    device: str = "auto"  # Device to use (auto/cpu/cuda/mps)
    resume_dir: (  # Bootstrap a new run from existing run dir
        Path | None
    ) = None
    resume_restore_optimizer_scheduler: bool = True  # If True, restore optimizer and scheduler from checkpoint when using resume_dir; if False, restore optimizer only and rebuild scheduler from current config
    resume_latest_as_incumbent: bool = False  # If True, resumed runs start self-play from the restored latest checkpoint model instead of the source run's saved incumbent bundle, and recompute the incumbent gate baseline on fixed seeds
    resume_wandb: (  # Resume from WandB run/artifact reference (entity/project/run_id or entity/project/artifact:alias)
        str | None
    ) = None
    init_checkpoint: Path | None = None  # Initialize model weights from checkpoint
    no_wandb: bool = False  # Disable WandB logging


def setup_run(
    args: ScriptArgs, config: TrainingConfig, resume_dir: Path | None = None
) -> tuple[TrainingConfig, Path | None, Path | None]:
    """Set up training run directory. Returns (config, resume_checkpoint, resume_incumbent_model_path)."""
    source_run_dir = resume_dir if resume_dir is not None else args.resume_dir
    if source_run_dir is None:
        config = setup_run_directory(config)
        logger.info("Created new training run", run_dir=str(config.run.run_dir))
        return config, None, None

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
        run_dir=str(config.run.run_dir),
        checkpoint=str(source_checkpoint),
    )

    source_training_data = source_run_dir / TRAINING_DATA_FILENAME
    if source_training_data.exists():
        assert config.run.data_dir is not None
        destination = config.run.data_dir / TRAINING_DATA_FILENAME
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

    source_runtime_overrides = source_run_dir / RUNTIME_OVERRIDES_FILENAME
    if source_runtime_overrides.exists():
        assert config.run.run_dir is not None
        destination_runtime_overrides = config.run.run_dir / RUNTIME_OVERRIDES_FILENAME
        shutil.copy2(source_runtime_overrides, destination_runtime_overrides)
        logger.info(
            "Copied runtime overrides file",
            source=str(source_runtime_overrides),
            destination=str(destination_runtime_overrides),
        )

    resume_incumbent_model_path: Path | None = None
    if args.resume_latest_as_incumbent:
        logger.info(
            "Configured resumed run to start from checkpoint latest model as incumbent",
            checkpoint=str(source_checkpoint),
        )
    else:
        source_incumbent = source_run_dir / CHECKPOINT_DIRNAME / INCUMBENT_ONNX_FILENAME
        if source_incumbent.exists():
            assert config.run.checkpoint_dir is not None
            destination_incumbent = config.run.checkpoint_dir / INCUMBENT_ONNX_FILENAME
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
        ema_model=trainer.ema_model,
        optimizer=trainer.optimizer,
        scheduler=load_scheduler,
    )

    checkpoint_step = state.get("step")
    if checkpoint_step is None:
        raise ValueError(f"Checkpoint is missing step: {checkpoint}")
    trainer.step = int(checkpoint_step)

    if trainer.scheduler is not None and not args.resume_restore_optimizer_scheduler:
        trainer.align_scheduler_to_step(trainer.step)

    runtime_override_lr_multiplier = state.get("runtime_override_lr_multiplier")
    runtime_override_grad_clip_norm = state.get("runtime_override_grad_clip_norm")
    runtime_override_weight_decay = state.get("runtime_override_weight_decay")
    runtime_override_mirror_augmentation_probability = state.get(
        "runtime_override_mirror_augmentation_probability"
    )
    runtime_override_log_interval_seconds = state.get(
        "runtime_override_log_interval_seconds"
    )
    runtime_override_checkpoint_interval_seconds = state.get(
        "runtime_override_checkpoint_interval_seconds"
    )
    if (
        runtime_override_lr_multiplier is not None
        and runtime_override_grad_clip_norm is not None
        and runtime_override_weight_decay is not None
        and runtime_override_mirror_augmentation_probability is not None
        and runtime_override_log_interval_seconds is not None
        and runtime_override_checkpoint_interval_seconds is not None
    ):
        trainer.restore_runtime_override_state(
            ResolvedRuntimeOverrides(
                optimizer=ResolvedRuntimeOptimizerOverrides(
                    lr_multiplier=float(runtime_override_lr_multiplier),
                    grad_clip_norm=float(runtime_override_grad_clip_norm),
                    weight_decay=float(runtime_override_weight_decay),
                    mirror_augmentation_probability=float(
                        runtime_override_mirror_augmentation_probability
                    ),
                ),
                run=ResolvedRuntimeRunOverrides(
                    log_interval_seconds=float(runtime_override_log_interval_seconds),
                    checkpoint_interval_seconds=float(
                        runtime_override_checkpoint_interval_seconds
                    ),
                ),
            ),
            lrs_already_scaled=args.resume_restore_optimizer_scheduler,
        )
        logger.info(
            "Restored runtime override state from checkpoint",
            checkpoint=str(checkpoint),
            lr_multiplier=runtime_override_lr_multiplier,
            grad_clip_norm=runtime_override_grad_clip_norm,
            weight_decay=runtime_override_weight_decay,
            mirror_augmentation_probability=(
                runtime_override_mirror_augmentation_probability
            ),
            log_interval_seconds=runtime_override_log_interval_seconds,
            checkpoint_interval_seconds=(runtime_override_checkpoint_interval_seconds),
        )

    incumbent_uses_network = state.get("incumbent_uses_network")
    if incumbent_uses_network is None:
        config.self_play.bootstrap_without_network = False
        start_with_network = True
        logger.warning(
            "Checkpoint missing incumbent network state; starting self-play with network",
            checkpoint=str(checkpoint),
            start_with_network=True,
        )
    else:
        start_with_network = bool(incumbent_uses_network)
        config.self_play.bootstrap_without_network = not start_with_network
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
            nn_value_weight=config.self_play.nn_value_weight,
        )
    else:
        restored_nn_value_weight = float(incumbent_nn_value_weight)
        if (
            not math.isfinite(restored_nn_value_weight)
            or restored_nn_value_weight < 0.0
        ):
            raise ValueError(
                f"Checkpoint incumbent_nn_value_weight must be >= 0 (got {restored_nn_value_weight})"
            )
        config.self_play.nn_value_weight = restored_nn_value_weight
        logger.info(
            "Restored incumbent nn_value_weight from checkpoint",
            checkpoint=str(checkpoint),
            nn_value_weight=restored_nn_value_weight,
        )

    incumbent_death_penalty = state.get("incumbent_death_penalty")
    incumbent_overhang_penalty_weight = state.get("incumbent_overhang_penalty_weight")
    if (
        incumbent_death_penalty is not None
        and incumbent_overhang_penalty_weight is not None
    ):
        restored_death_penalty = float(incumbent_death_penalty)
        if not math.isfinite(restored_death_penalty) or restored_death_penalty < 0.0:
            raise ValueError(
                "Checkpoint incumbent_death_penalty must be finite and >= 0 "
                f"(got {restored_death_penalty})"
            )

        restored_overhang_penalty_weight = float(incumbent_overhang_penalty_weight)
        if (
            not math.isfinite(restored_overhang_penalty_weight)
            or restored_overhang_penalty_weight < 0.0
        ):
            raise ValueError(
                "Checkpoint incumbent_overhang_penalty_weight must be finite and >= 0 "
                f"(got {restored_overhang_penalty_weight})"
            )

        config.self_play.death_penalty = restored_death_penalty
        config.self_play.overhang_penalty_weight = restored_overhang_penalty_weight
        logger.info(
            "Restored incumbent search penalties from checkpoint",
            checkpoint=str(checkpoint),
            death_penalty=restored_death_penalty,
            overhang_penalty_weight=restored_overhang_penalty_weight,
        )
    else:
        penalties_should_be_disabled = (
            config.self_play.nn_value_weight >= config.self_play.nn_value_weight_cap
        )
        if penalties_should_be_disabled:
            config.self_play.death_penalty = 0.0
            config.self_play.overhang_penalty_weight = 0.0
            logger.warning(
                "Checkpoint missing incumbent penalty state at nn_value_weight cap; inferring zero penalties",
                checkpoint=str(checkpoint),
                nn_value_weight=config.self_play.nn_value_weight,
                nn_value_weight_cap=config.self_play.nn_value_weight_cap,
            )
        else:
            logger.warning(
                "Checkpoint missing incumbent penalty state; using config values",
                checkpoint=str(checkpoint),
                death_penalty=config.self_play.death_penalty,
                overhang_penalty_weight=config.self_play.overhang_penalty_weight,
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

    next_display_game_number = state.get("next_display_game_number")
    if next_display_game_number is not None:
        restored_next_display_game_number = int(next_display_game_number)
        if restored_next_display_game_number < 1:
            raise ValueError(
                "Checkpoint next_display_game_number must be >= 1 "
                f"(got {restored_next_display_game_number})"
            )
        trainer._next_display_game_number = restored_next_display_game_number
        logger.info(
            "Restored sequential W&B game-number counter from checkpoint",
            checkpoint=str(checkpoint),
            next_display_game_number=restored_next_display_game_number,
        )

    candidate_gate_current_interval = state.get(
        "candidate_gate_current_interval_seconds"
    )
    if candidate_gate_current_interval is not None:
        restored_candidate_gate_current_interval = float(
            candidate_gate_current_interval
        )
        if (
            not math.isfinite(restored_candidate_gate_current_interval)
            or restored_candidate_gate_current_interval <= 0.0
        ):
            raise ValueError(
                "Checkpoint candidate_gate_current_interval_seconds must be "
                f"finite and > 0 (got {restored_candidate_gate_current_interval})"
            )
        trainer.initial_candidate_gate_interval_seconds = (
            restored_candidate_gate_current_interval
        )

    candidate_gate_failed_promotion_streak = state.get(
        "candidate_gate_failed_promotion_streak"
    )
    if candidate_gate_failed_promotion_streak is not None:
        restored_candidate_gate_failed_promotion_streak = int(
            candidate_gate_failed_promotion_streak
        )
        if restored_candidate_gate_failed_promotion_streak < 0:
            raise ValueError(
                "Checkpoint candidate_gate_failed_promotion_streak must be >= 0 "
                f"(got {restored_candidate_gate_failed_promotion_streak})"
            )
        trainer.initial_candidate_gate_failed_promotion_streak = (
            restored_candidate_gate_failed_promotion_streak
        )

    candidate_gate_next_export_delay = state.get(
        "candidate_gate_next_export_delay_seconds"
    )
    if candidate_gate_next_export_delay is not None:
        restored_candidate_gate_next_export_delay = float(
            candidate_gate_next_export_delay
        )
        if (
            not math.isfinite(restored_candidate_gate_next_export_delay)
            or restored_candidate_gate_next_export_delay < 0.0
        ):
            raise ValueError(
                "Checkpoint candidate_gate_next_export_delay_seconds must be "
                f"finite and >= 0 (got {restored_candidate_gate_next_export_delay})"
            )
        trainer.initial_candidate_gate_next_export_delay_seconds = (
            restored_candidate_gate_next_export_delay
        )

    if (
        candidate_gate_current_interval is not None
        or candidate_gate_failed_promotion_streak is not None
        or candidate_gate_next_export_delay is not None
    ):
        logger.info(
            "Restored candidate gate schedule from checkpoint",
            checkpoint=str(checkpoint),
            current_interval_seconds=trainer.initial_candidate_gate_interval_seconds,
            failed_promotion_streak=(
                trainer.initial_candidate_gate_failed_promotion_streak
            ),
            next_export_delay_seconds=(
                trainer.initial_candidate_gate_next_export_delay_seconds
            ),
        )

    if start_with_network and args.resume_latest_as_incumbent:
        trainer.initial_incumbent_model_path = None
        trainer.initial_incumbent_eval_avg_attack = 0.0
        trainer.recompute_initial_incumbent_eval_avg_attack = (
            config.self_play.use_candidate_gating
        )
        logger.info(
            "Configured resumed run to use checkpoint latest model as starting incumbent",
            checkpoint=str(checkpoint),
            incumbent_step=trainer.step,
            recompute_incumbent_eval_avg_attack=(
                trainer.recompute_initial_incumbent_eval_avg_attack
            ),
        )
    elif start_with_network and incumbent_model_path is not None:
        trainer.initial_incumbent_model_path = incumbent_model_path
        logger.info(
            "Configured resumed incumbent model artifact for generator startup",
            path=str(incumbent_model_path),
        )
    elif args.resume_latest_as_incumbent:
        logger.warning(
            "Ignoring resume_latest_as_incumbent because checkpoint starts without a network incumbent",
            checkpoint=str(checkpoint),
        )
    elif start_with_network:
        logger.warning(
            "Resuming with network but no incumbent model artifact bundle; falling back to trainer checkpoint model",
            expected_path=(
                str(config.run.checkpoint_dir / INCUMBENT_ONNX_FILENAME)
                if config.run.checkpoint_dir is not None
                else None
            ),
        )

    logger.info(
        "Initialized new run from checkpoint",
        checkpoint=str(checkpoint),
        step=trainer.step,
        learning_rate=trainer.optimizer.param_groups[0]["lr"],
        lr_schedule=config.optimizer.lr_schedule,
        restored_optimizer_scheduler=args.resume_restore_optimizer_scheduler,
    )


def apply_optimized_runtime_overrides(config: TrainingConfig) -> None:
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


def main(args: ScriptArgs, *, config: TrainingConfig | None = None) -> None:
    resume_source_count = int(args.resume_dir is not None) + int(
        args.resume_wandb is not None
    )
    if resume_source_count > 1:
        raise ValueError("Cannot use resume_dir and resume_wandb together")
    if resume_source_count > 0 and args.init_checkpoint:
        raise ValueError(
            "Cannot use a resume source (resume_dir or resume_wandb) "
            "together with init_checkpoint"
        )

    effective_resume_dir = args.resume_dir
    wandb_resume_source: WandbResumeSource | None = None
    if args.resume_wandb is not None:
        wandb_resume_source = prepare_wandb_resume_source(args.resume_wandb)
        effective_resume_dir = wandb_resume_source.resume_dir

    if effective_resume_dir is not None:
        logger.info("Using resume source", source_dir=str(effective_resume_dir))

    if config is None:
        config_path = args.config.resolve()
        config = load_training_config(config_path)
        logger.info("Loaded training config", config_path=str(config_path))

    try:
        config, resume_checkpoint, resume_incumbent_model_path = setup_run(
            args, config, resume_dir=effective_resume_dir
        )
        apply_optimized_runtime_overrides(config)

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
                args.init_checkpoint,
                model=trainer.model,
                ema_model=trainer.ema_model,
                optimizer=None,
            )
            logger.info(
                "Initialized model from checkpoint",
                path=str(args.init_checkpoint),
                checkpoint_step=state.get("step"),
            )

        if not args.no_wandb:
            configure_wandb(config, device, resume_dir=effective_resume_dir)

        torch.set_float32_matmul_precision("high")
        logger.info("Starting training with Rust game generation")
        trainer.train(log_to_wandb=not args.no_wandb)

        logger.info("Training complete")
    finally:
        if wandb_resume_source is not None:
            shutil.rmtree(wandb_resume_source.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    args = parse(ScriptArgs)
    main(args)
