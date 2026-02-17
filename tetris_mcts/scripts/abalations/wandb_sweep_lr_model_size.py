from dataclasses import dataclass, field
from pathlib import Path

import structlog
import wandb
from simple_parsing import parse

from tetris_mcts.config import TrainingConfig
from tetris_mcts.train import ScriptArgs as TrainScriptArgs
from tetris_mcts.train import main as train_main

logger = structlog.get_logger()

MODEL_SIZE_PRESETS = {
    "baseline": {"trunk_channels": 16, "num_conv_residual_blocks": 1, "reduction_channels": 32, "fusion_hidden": 128},
    "large": {"trunk_channels": 32, "num_conv_residual_blocks": 2, "reduction_channels": 64, "fusion_hidden": 256},
}


@dataclass
class SweepArgs:
    sweep_id: str | None = None  # Use an existing sweep ID instead of creating one
    create_only: bool = False  # Create sweep and exit without starting agents
    count: int = 12  # Number of runs to execute (ignored when create_only=True)
    method: str = "bayes"  # Sweep method: bayes, random, grid
    metric_name: str = "eval/avg_attack"  # Metric name to optimize
    metric_goal: str = "maximize"  # Metric goal: maximize or minimize
    project_name: str = "tetris-alphazero"
    entity: str | None = None

    # Search space
    learning_rate_min: float = 1e-4
    learning_rate_max: float = 2e-3
    model_sizes: list[str] = field(default_factory=lambda: ["baseline", "large"])

    # Base training settings for each trial
    total_steps: int = 200_000
    batch_size: int = 1024
    lr_schedule: str = "linear"
    lr_decay_steps: int = 200_000
    lr_min_factor: float = 0.2
    run_name_prefix: str = "sweep-lr-modelsize"
    device: str = "auto"

    # Optional initialization controls
    resume_dir: Path | None = None
    resume_restore_optimizer_scheduler: bool = True
    init_checkpoint: Path | None = None

    # Optional sweep metadata
    sweep_name: str = "offline-lr-model-size"


def build_sweep_config(args: SweepArgs) -> dict:
    return {
        "name": args.sweep_name,
        "method": args.method,
        "metric": {
            "name": args.metric_name,
            "goal": args.metric_goal,
        },
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": args.learning_rate_min,
                "max": args.learning_rate_max,
            },
            "model_size": {
                "values": args.model_sizes,
            },
        },
    }


def validate_args(args: SweepArgs) -> None:
    if args.metric_goal not in {"maximize", "minimize"}:
        raise ValueError(
            f"metric_goal must be maximize or minimize (got {args.metric_goal})"
        )
    if args.method not in {"bayes", "random", "grid"}:
        raise ValueError(f"Unsupported sweep method: {args.method}")
    if args.learning_rate_min <= 0.0:
        raise ValueError(
            f"learning_rate_min must be > 0 (got {args.learning_rate_min})"
        )
    if args.learning_rate_max <= args.learning_rate_min:
        raise ValueError(
            "learning_rate_max must be > learning_rate_min "
            f"(got {args.learning_rate_max} <= {args.learning_rate_min})"
        )
    if args.count <= 0:
        raise ValueError(f"count must be > 0 (got {args.count})")
    if args.total_steps <= 0:
        raise ValueError(f"total_steps must be > 0 (got {args.total_steps})")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0 (got {args.batch_size})")
    if not args.model_sizes:
        raise ValueError("model_sizes cannot be empty")
    unknown_model_sizes = [
        model_size
        for model_size in args.model_sizes
        if model_size not in MODEL_SIZE_PRESETS
    ]
    if unknown_model_sizes:
        raise ValueError(f"Unknown model_sizes: {unknown_model_sizes}")
    if args.resume_dir and args.init_checkpoint:
        raise ValueError("Cannot set both resume_dir and init_checkpoint")


def build_training_config(
    args: SweepArgs, run_name: str, sampled: dict
) -> TrainingConfig:
    model_size = str(sampled["model_size"])
    learning_rate = float(sampled["learning_rate"])
    model_preset = MODEL_SIZE_PRESETS[model_size]

    config = TrainingConfig(
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        learning_rate=learning_rate,
        lr_schedule=args.lr_schedule,
        lr_decay_steps=args.lr_decay_steps,
        lr_min_factor=args.lr_min_factor,
        project_name=args.project_name,
        run_name=f"{args.run_name_prefix}-{run_name}",
    )
    config.trunk_channels = int(model_preset["trunk_channels"])
    config.num_conv_residual_blocks = int(model_preset["num_conv_residual_blocks"])
    config.reduction_channels = int(model_preset["reduction_channels"])
    config.fc_hidden = int(model_preset["fusion_hidden"])
    return config


def run_single_trial(args: SweepArgs) -> None:
    with wandb.init(project=args.project_name, entity=args.entity) as run:
        if run is None:
            raise RuntimeError("wandb.init returned None")

        sampled = dict(wandb.config)
        if "learning_rate" not in sampled:
            raise ValueError("Sweep config missing learning_rate")
        if "model_size" not in sampled:
            raise ValueError("Sweep config missing model_size")

        model_size = str(sampled["model_size"])
        learning_rate = float(sampled["learning_rate"])
        model_preset = MODEL_SIZE_PRESETS[model_size]
        run_name = run.name if run.name is not None else run.id

        training_config = build_training_config(args, run_name, sampled)
        wandb.config.update(
            {
                "sweep_model_size": model_size,
                "sweep_trunk_channels": int(model_preset["trunk_channels"]),
                "sweep_num_conv_residual_blocks": int(model_preset["num_conv_residual_blocks"]),
                "sweep_reduction_channels": int(model_preset["reduction_channels"]),
                "sweep_fusion_hidden": int(model_preset["fusion_hidden"]),
                "sweep_learning_rate": learning_rate,
            },
            allow_val_change=True,
        )

        logger.info(
            "Starting sweep trial",
            wandb_run_name=run.name,
            learning_rate=learning_rate,
            model_size=model_size,
            trunk_channels=training_config.trunk_channels,
            reduction_channels=training_config.reduction_channels,
            fusion_hidden=int(model_preset["fusion_hidden"]),
            total_steps=training_config.total_steps,
            batch_size=training_config.batch_size,
        )

        train_args = TrainScriptArgs(
            training=training_config,
            device=args.device,
            resume_dir=args.resume_dir,
            resume_restore_optimizer_scheduler=args.resume_restore_optimizer_scheduler,
            init_checkpoint=args.init_checkpoint,
            no_wandb=False,
        )
        train_main(train_args)


def create_sweep(args: SweepArgs) -> str:
    sweep_config = build_sweep_config(args)
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=args.project_name,
        entity=args.entity,
    )
    logger.info(
        "Created sweep",
        sweep_id=sweep_id,
        project=args.project_name,
        entity=args.entity,
        sweep_name=args.sweep_name,
    )
    return sweep_id


def run_agent(args: SweepArgs, sweep_id: str) -> None:
    logger.info(
        "Starting sweep agent",
        sweep_id=sweep_id,
        count=args.count,
        project=args.project_name,
        entity=args.entity,
    )
    wandb.agent(
        sweep_id=sweep_id,
        function=lambda: run_single_trial(args),
        count=args.count,
        project=args.project_name,
        entity=args.entity,
    )


def main(args: SweepArgs) -> None:
    validate_args(args)
    sweep_id = args.sweep_id if args.sweep_id is not None else create_sweep(args)
    if args.create_only:
        logger.info("Sweep created (create_only=True)", sweep_id=sweep_id)
        return
    run_agent(args, sweep_id)


if __name__ == "__main__":
    main(parse(SweepArgs))
