from __future__ import annotations

import copy
import json
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import structlog
import torch
import wandb
from simple_parsing import parse

from tetris_core.tetris_core import MCTSConfig, evaluate_model
from tetris_bot.constants import (
    CONFIG_FILENAME,
    INCUMBENT_ONNX_FILENAME,
    LATEST_METADATA_FILENAME,
    LATEST_ONNX_FILENAME,
    PARALLEL_ONNX_FILENAME,
    TRAINING_DATA_FILENAME,
)
from tetris_bot.ml.artifacts import (
    assert_rust_inference_artifacts,
    copy_model_artifact_bundle,
)
from tetris_bot.ml.config import TrainingConfig, load_training_config_json
from tetris_bot.ml.loss import RunningLossBalancer, compute_loss
from tetris_bot.ml.network import TetrisNet
from tetris_bot.ml.trainer import Trainer
from tetris_bot.ml.weights import export_metadata, save_checkpoint
from tetris_bot.run_setup import config_to_json, setup_run_directory
from tetris_bot.scripts.ablations.compare_offline_architectures import (
    OfflineDataSource,
    OfflineDatasetSetup,
    build_tensor_dataset,
    build_torch_batch,
    ensure_required_keys,
    get_preload_mode,
    pick_device,
    select_subset,
    tensor_dataset_bytes,
    validate_shapes,
)

logger = structlog.get_logger()


@dataclass
class ScriptArgs:
    source_run_dir: Path
    output_run_dir: Path | None = None
    device: str = "auto"
    seed: int = 123
    epochs: float = 20.0
    max_examples: int = 0
    train_fraction: float = 0.9
    batch_size: int | None = None
    learning_rate: float | None = None
    weight_decay: float | None = None
    grad_clip_norm: float | None = None
    eval_interval: int = 0
    eval_examples: int = 32_768
    eval_batch_size: int = 2_048
    preload_to_gpu: bool = True
    preload_to_ram: bool = False
    num_eval_games: int = 20
    eval_seed_start: int = 0
    eval_num_workers: int = 0
    eval_num_simulations: int | None = None
    eval_max_placements: int | None = None
    mcts_seed: int | None = 0
    wandb_project: str = "tetris-mcts-offline"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(default_factory=lambda: ["offline", "warm-start"])


def validate_args(args: ScriptArgs) -> None:
    if args.epochs <= 0:
        raise ValueError(f"epochs must be > 0 (got {args.epochs})")
    if args.max_examples < 0:
        raise ValueError(f"max_examples must be >= 0 (got {args.max_examples})")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError(
            f"train_fraction must be in (0, 1) (got {args.train_fraction})"
        )
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0 (got {args.batch_size})")
    if args.learning_rate is not None and args.learning_rate <= 0.0:
        raise ValueError(
            f"learning_rate must be > 0 when set (got {args.learning_rate})"
        )
    if args.weight_decay is not None and args.weight_decay < 0.0:
        raise ValueError(
            f"weight_decay must be >= 0 when set (got {args.weight_decay})"
        )
    if args.grad_clip_norm is not None and args.grad_clip_norm <= 0.0:
        raise ValueError(
            f"grad_clip_norm must be > 0 when set (got {args.grad_clip_norm})"
        )
    if args.eval_interval < 0:
        raise ValueError(f"eval_interval must be >= 0 (got {args.eval_interval})")
    if args.eval_examples <= 0:
        raise ValueError(f"eval_examples must be > 0 (got {args.eval_examples})")
    if args.eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size must be > 0 (got {args.eval_batch_size})")
    if args.num_eval_games <= 0:
        raise ValueError(f"num_eval_games must be > 0 (got {args.num_eval_games})")
    if args.eval_num_workers < 0:
        raise ValueError(f"eval_num_workers must be >= 0 (got {args.eval_num_workers})")
    if args.eval_num_simulations is not None and args.eval_num_simulations <= 0:
        raise ValueError(
            "eval_num_simulations must be > 0 when set "
            f"(got {args.eval_num_simulations})"
        )
    if args.eval_max_placements is not None and args.eval_max_placements <= 0:
        raise ValueError(
            f"eval_max_placements must be > 0 when set (got {args.eval_max_placements})"
        )

    source_run_dir = args.source_run_dir.resolve()
    if not source_run_dir.exists():
        raise FileNotFoundError(
            f"Source run directory does not exist: {source_run_dir}"
        )
    if not source_run_dir.is_dir():
        raise NotADirectoryError(
            f"Source run directory is not a directory: {source_run_dir}"
        )

    config_path = source_run_dir / CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(f"Source config not found: {config_path}")

    training_data_path = source_run_dir / TRAINING_DATA_FILENAME
    if not training_data_path.exists():
        raise FileNotFoundError(f"Source training data not found: {training_data_path}")

    if args.output_run_dir is not None:
        output_run_dir = args.output_run_dir.resolve()
        if output_run_dir == source_run_dir:
            raise ValueError("output_run_dir must differ from source_run_dir")
        if output_run_dir.exists():
            raise FileExistsError(
                f"Output run directory already exists: {output_run_dir}"
            )


def clone_model_state_dict(model: TetrisNet) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def compute_training_steps(
    train_examples: int,
    *,
    batch_size: int,
    epochs: float,
) -> int:
    if train_examples <= 0:
        raise ValueError(f"train_examples must be > 0 (got {train_examples})")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0 (got {batch_size})")
    if epochs <= 0:
        raise ValueError(f"epochs must be > 0 (got {epochs})")
    return max(1, math.ceil((train_examples * epochs) / batch_size))


def setup_offline_dataset(
    npz: np.lib.npyio.NpzFile,
    *,
    seed: int,
    max_examples: int,
    train_fraction: float,
    eval_examples: int,
    preload_mode: str,
    device: torch.device,
) -> OfflineDatasetSetup:
    ensure_required_keys(npz)
    total_examples = validate_shapes(npz)
    selected_global_indices = np.arange(total_examples, dtype=np.int64)

    split_rng = np.random.default_rng(seed)
    split_rng.shuffle(selected_global_indices)
    if max_examples > 0:
        selected_global_indices = selected_global_indices[:max_examples]

    num_selected = len(selected_global_indices)
    split_point = int(num_selected * train_fraction)
    if split_point <= 0 or split_point >= num_selected:
        raise ValueError(
            "Invalid train/val split; adjust max_examples or train_fraction"
        )

    train_local_indices = np.arange(split_point, dtype=np.int64)
    val_local_indices = np.arange(split_point, num_selected, dtype=np.int64)

    preload_start = time.perf_counter()
    tensor_data = None
    if preload_mode != "none":
        tensor_data = build_tensor_dataset(
            data=npz,
            selected_global_indices=selected_global_indices,
            mode=preload_mode,
            train_device=device,
        )
    preload_sec = time.perf_counter() - preload_start

    source = OfflineDataSource(
        npz=npz,
        selected_global_indices=selected_global_indices,
        tensor_data=tensor_data,
    )
    train_eval_local_indices = select_subset(
        train_local_indices,
        max_examples=eval_examples,
        seed=seed + 1,
    )
    val_eval_local_indices = select_subset(
        val_local_indices,
        max_examples=eval_examples,
        seed=seed + 2,
    )

    logger.info(
        "Warm-start dataset split",
        total_examples=total_examples,
        used_examples=num_selected,
        train_examples=len(train_local_indices),
        val_examples=len(val_local_indices),
        train_eval_examples=len(train_eval_local_indices),
        val_eval_examples=len(val_eval_local_indices),
        preload_mode=preload_mode,
        preload_seconds=preload_sec,
        preload_bytes=(
            tensor_dataset_bytes(tensor_data) if tensor_data is not None else 0
        ),
    )

    return OfflineDatasetSetup(
        source=source,
        train_local_indices=train_local_indices,
        val_local_indices=val_local_indices,
        train_eval_local_indices=train_eval_local_indices,
        val_eval_local_indices=val_eval_local_indices,
        total_examples=total_examples,
        num_selected=num_selected,
        preload_sec=preload_sec,
    )


def evaluate_offline_losses(
    model: TetrisNet,
    *,
    source: OfflineDataSource,
    local_indices: np.ndarray,
    device: torch.device,
    eval_batch_size: int,
    value_loss_weight: float,
    use_huber_value_loss: bool,
) -> dict[str, float]:
    total_sum = 0.0
    policy_sum = 0.0
    value_sum = 0.0
    sample_count = 0

    model.eval()
    with torch.no_grad():
        for start in range(0, len(local_indices), eval_batch_size):
            batch_indices = local_indices[start : start + eval_batch_size]
            boards, aux, policy_targets, value_targets, action_masks = (
                build_torch_batch(source, batch_indices, device)
            )
            total_loss, policy_loss, value_loss = compute_loss(
                model=model,
                boards=boards,
                aux_features=aux,
                policy_targets=policy_targets,
                value_targets=value_targets,
                action_masks=action_masks,
                value_loss_weight=value_loss_weight,
                use_huber_value_loss=use_huber_value_loss,
            )
            batch_size = len(batch_indices)
            total_sum += total_loss.item() * batch_size
            policy_sum += policy_loss.item() * batch_size
            value_sum += value_loss.item() * batch_size
            sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Cannot evaluate offline losses on an empty split")

    return {
        "total_loss": total_sum / sample_count,
        "policy_loss": policy_sum / sample_count,
        "value_loss": value_sum / sample_count,
    }


def build_output_config(
    source_config: TrainingConfig,
    *,
    source_run_dir: Path,
    output_run_dir: Path | None,
) -> TrainingConfig:
    config = copy.deepcopy(source_config)
    config.self_play.nn_value_weight = config.self_play.nn_value_weight_cap
    config.self_play.death_penalty = 0.0
    config.self_play.overhang_penalty_weight = 0.0
    config.self_play.bootstrap_without_network = False
    config.run.run_name = None
    config.run.run_dir = None
    config.run.checkpoint_dir = None
    config.run.data_dir = None

    return setup_run_directory(
        config,
        base_dir=source_run_dir.parent,
        run_dir=output_run_dir,
    )


def train_warm_start_model(
    model: TetrisNet,
    *,
    dataset_setup: OfflineDatasetSetup,
    device: torch.device,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float,
    num_steps: int,
    eval_interval: int,
    eval_batch_size: int,
    use_huber_value_loss: bool,
    value_loss_window: int,
    seed: int,
) -> tuple[
    dict[str, torch.Tensor], dict[str, float | int], list[dict[str, float | int]]
]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_balancer = RunningLossBalancer(value_loss_window)
    current_value_loss_weight = 1.0
    history: list[dict[str, float | int]] = []
    best_record: dict[str, float | int] | None = None
    best_state_dict = clone_model_state_dict(model)
    rng = np.random.default_rng(seed)
    log_interval = max(1, num_steps // 20)
    start_time = time.perf_counter()

    def record_eval(step: int, epochs_seen: float) -> None:
        nonlocal best_record, best_state_dict
        train_metrics = evaluate_offline_losses(
            model,
            source=dataset_setup.source,
            local_indices=dataset_setup.train_eval_local_indices,
            device=device,
            eval_batch_size=eval_batch_size,
            value_loss_weight=current_value_loss_weight,
            use_huber_value_loss=use_huber_value_loss,
        )
        val_metrics = evaluate_offline_losses(
            model,
            source=dataset_setup.source,
            local_indices=dataset_setup.val_eval_local_indices,
            device=device,
            eval_batch_size=eval_batch_size,
            value_loss_weight=current_value_loss_weight,
            use_huber_value_loss=use_huber_value_loss,
        )
        record = {
            "step": step,
            "epochs_seen": epochs_seen,
            "value_loss_weight": current_value_loss_weight,
            "train_total_loss": train_metrics["total_loss"],
            "train_policy_loss": train_metrics["policy_loss"],
            "train_value_loss": train_metrics["value_loss"],
            "val_total_loss": val_metrics["total_loss"],
            "val_policy_loss": val_metrics["policy_loss"],
            "val_value_loss": val_metrics["value_loss"],
        }
        history.append(record)
        logger.info(
            "Warm-start offline eval",
            step=step,
            epochs_seen=epochs_seen,
            train_total_loss=record["train_total_loss"],
            val_total_loss=record["val_total_loss"],
            train_policy_loss=record["train_policy_loss"],
            val_policy_loss=record["val_policy_loss"],
            train_value_loss=record["train_value_loss"],
            val_value_loss=record["val_value_loss"],
            value_loss_weight=current_value_loss_weight,
        )
        log_wandb(
            {
                "offline_step": step,
                "warm_start/eval_epochs_seen": epochs_seen,
                "warm_start/value_loss_weight": current_value_loss_weight,
                "warm_start/eval_train_total_loss": record["train_total_loss"],
                "warm_start/eval_train_policy_loss": record["train_policy_loss"],
                "warm_start/eval_train_value_loss": record["train_value_loss"],
                "warm_start/eval_val_total_loss": record["val_total_loss"],
                "warm_start/eval_val_policy_loss": record["val_policy_loss"],
                "warm_start/eval_val_value_loss": record["val_value_loss"],
            }
        )
        if best_record is None or float(record["val_total_loss"]) < float(
            best_record["val_total_loss"]
        ):
            best_record = dict(record)
            best_state_dict = clone_model_state_dict(model)

    record_eval(step=0, epochs_seen=0.0)

    for step in range(1, num_steps + 1):
        positions = rng.integers(
            0, len(dataset_setup.train_local_indices), size=batch_size
        )
        batch_indices = dataset_setup.train_local_indices[positions]
        boards, aux, policy_targets, value_targets, action_masks = build_torch_batch(
            dataset_setup.source,
            batch_indices,
            device,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, policy_loss, value_loss = compute_loss(
            model=model,
            boards=boards,
            aux_features=aux,
            policy_targets=policy_targets,
            value_targets=value_targets,
            action_masks=action_masks,
            value_loss_weight=current_value_loss_weight,
            use_huber_value_loss=use_huber_value_loss,
        )
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        policy_loss_scalar = policy_loss.item()
        value_loss_scalar = value_loss.item()
        loss_balancer.append(policy_loss_scalar, value_loss_scalar)
        if loss_balancer.has_history():
            current_value_loss_weight = loss_balancer.value_loss_weight()

        if step % log_interval == 0 or step == num_steps:
            elapsed_sec = time.perf_counter() - start_time
            epochs_seen = (step * batch_size) / len(dataset_setup.train_local_indices)
            logger.info(
                "Warm-start offline train",
                step=step,
                num_steps=num_steps,
                epochs_seen=epochs_seen,
                total_loss=total_loss.item(),
                policy_loss=policy_loss_scalar,
                value_loss=value_loss_scalar,
                value_loss_weight=current_value_loss_weight,
                grad_norm=float(grad_norm.item()),
                learning_rate=optimizer.param_groups[0]["lr"],
                elapsed_sec=elapsed_sec,
            )
            log_wandb(
                {
                    "offline_step": step,
                    "warm_start/train_epochs_seen": epochs_seen,
                    "warm_start/train_batch_total_loss": total_loss.item(),
                    "warm_start/train_batch_policy_loss": policy_loss_scalar,
                    "warm_start/train_batch_value_loss": value_loss_scalar,
                    "warm_start/value_loss_weight": current_value_loss_weight,
                    "warm_start/grad_norm": float(grad_norm.item()),
                    "warm_start/learning_rate": optimizer.param_groups[0]["lr"],
                    "warm_start/elapsed_sec": elapsed_sec,
                    "warm_start/progress_fraction": step / num_steps,
                }
            )

        if step % eval_interval == 0 or step == num_steps:
            epochs_seen = (step * batch_size) / len(dataset_setup.train_local_indices)
            record_eval(step=step, epochs_seen=epochs_seen)

    if best_record is None:
        raise RuntimeError("Warm-start offline training never produced an evaluation")

    return best_state_dict, best_record, history


def save_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def log_wandb(payload: dict[str, object]) -> None:
    if wandb.run is not None:
        wandb.log(payload)


def main(args: ScriptArgs) -> None:
    validate_args(args)

    source_run_dir = args.source_run_dir.resolve()
    output_run_dir = (
        args.output_run_dir.resolve() if args.output_run_dir is not None else None
    )
    source_config_path = source_run_dir / CONFIG_FILENAME
    source_training_data_path = source_run_dir / TRAINING_DATA_FILENAME
    source_config = load_training_config_json(source_config_path)
    output_config = build_output_config(
        source_config,
        source_run_dir=source_run_dir,
        output_run_dir=output_run_dir,
    )

    if output_config.run.run_dir is None or output_config.run.checkpoint_dir is None:
        raise RuntimeError("Output run directory was not set by setup_run_directory")
    if output_config.run.data_dir is None:
        raise RuntimeError("Output data directory was not set by setup_run_directory")

    device_str = pick_device(args.device)
    device = torch.device(device_str)
    preload_mode = get_preload_mode(args)
    if preload_mode == "gpu" and device.type == "cpu":
        raise ValueError("preload_to_gpu requires a non-CPU device")

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else output_config.optimizer.batch_size
    )
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else output_config.optimizer.learning_rate
    )
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else output_config.optimizer.weight_decay
    )
    grad_clip_norm = (
        args.grad_clip_norm
        if args.grad_clip_norm is not None
        else output_config.optimizer.grad_clip_norm
    )
    resolved_wandb_run_name = (
        args.wandb_run_name
        if args.wandb_run_name is not None
        else f"warm-start-{source_run_dir.name}-to-{output_config.run.run_dir.name}"
    )
    wandb_config = {
        "source_run_dir": str(source_run_dir),
        "source_config_path": str(source_config_path),
        "source_training_data_path": str(source_training_data_path),
        "output_run_dir": str(output_config.run.run_dir),
        "device": device_str,
        "seed": args.seed,
        "epochs": args.epochs,
        "max_examples": args.max_examples,
        "train_fraction": args.train_fraction,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "grad_clip_norm": grad_clip_norm,
        "eval_interval": args.eval_interval,
        "eval_examples": args.eval_examples,
        "eval_batch_size": args.eval_batch_size,
        "preload_to_gpu": args.preload_to_gpu,
        "preload_to_ram": args.preload_to_ram,
        "preload_mode": preload_mode,
        "num_eval_games": args.num_eval_games,
        "eval_seed_start": args.eval_seed_start,
        "eval_num_workers": args.eval_num_workers,
        "eval_num_simulations": args.eval_num_simulations,
        "eval_max_placements": args.eval_max_placements,
        "mcts_seed": args.mcts_seed,
        "wandb_tags": args.wandb_tags,
        "output_config": json.loads(config_to_json(output_config)),
    }
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=resolved_wandb_run_name,
        tags=args.wandb_tags,
        config=wandb_config,
    )
    wandb.define_metric("offline_step")
    wandb.define_metric("warm_start/*", step_metric="offline_step")
    wandb.define_metric("dataset/*", step_metric="offline_step")
    wandb.define_metric("final_eval/*", step_metric="offline_step")

    logger.info(
        "Starting warm start",
        source_run_dir=str(source_run_dir),
        output_run_dir=str(output_config.run.run_dir),
        device=device_str,
        epochs=args.epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        preload_mode=preload_mode,
        wandb_project=args.wandb_project,
        wandb_run_name=resolved_wandb_run_name,
    )

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    model = TetrisNet(**output_config.network.to_model_kwargs()).to(device)

    try:
        npz = np.load(source_training_data_path, mmap_mode="r")
        try:
            dataset_setup = setup_offline_dataset(
                npz,
                seed=args.seed,
                max_examples=args.max_examples,
                train_fraction=args.train_fraction,
                eval_examples=args.eval_examples,
                preload_mode=preload_mode,
                device=device,
            )
            log_wandb(
                {
                    "offline_step": 0,
                    "dataset/total_examples": dataset_setup.total_examples,
                    "dataset/used_examples": dataset_setup.num_selected,
                    "dataset/train_examples": len(dataset_setup.train_local_indices),
                    "dataset/val_examples": len(dataset_setup.val_local_indices),
                    "dataset/train_eval_examples": len(
                        dataset_setup.train_eval_local_indices
                    ),
                    "dataset/val_eval_examples": len(
                        dataset_setup.val_eval_local_indices
                    ),
                    "dataset/preload_seconds": dataset_setup.preload_sec,
                    "dataset/preload_mode": preload_mode,
                }
            )
            num_steps = compute_training_steps(
                len(dataset_setup.train_local_indices),
                batch_size=batch_size,
                epochs=args.epochs,
            )
            eval_interval = (
                args.eval_interval
                if args.eval_interval > 0
                else max(1, num_steps // 10)
            )

            best_state_dict, best_record, offline_history = train_warm_start_model(
                model,
                dataset_setup=dataset_setup,
                device=device,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                grad_clip_norm=grad_clip_norm,
                num_steps=num_steps,
                eval_interval=eval_interval,
                eval_batch_size=args.eval_batch_size,
                use_huber_value_loss=output_config.optimizer.use_huber_value_loss,
                value_loss_window=output_config.optimizer.value_loss_weight_window,
                seed=args.seed,
            )
        finally:
            npz.close()

        model.load_state_dict(best_state_dict)
        model.eval()

        output_training_data_path = output_config.run.data_dir / TRAINING_DATA_FILENAME
        shutil.copy2(source_training_data_path, output_training_data_path)

        trainer = Trainer(output_config, model=model, device=device_str)
        trainer.step = 0
        initial_checkpoint_state = {
            "incumbent_uses_network": True,
            "incumbent_model_step": 0,
            "incumbent_nn_value_weight": output_config.self_play.nn_value_weight,
            "incumbent_death_penalty": output_config.self_play.death_penalty,
            "incumbent_overhang_penalty_weight": (
                output_config.self_play.overhang_penalty_weight
            ),
            "incumbent_eval_avg_attack": 0.0,
            "incumbent_model_source_path": str(
                output_config.run.checkpoint_dir / INCUMBENT_ONNX_FILENAME
            ),
            "incumbent_model_artifact": INCUMBENT_ONNX_FILENAME,
        }
        saved_paths = trainer.save(extra_checkpoint_state=initial_checkpoint_state)

        latest_onnx_path = output_config.run.checkpoint_dir / LATEST_ONNX_FILENAME
        incumbent_onnx_path = output_config.run.checkpoint_dir / INCUMBENT_ONNX_FILENAME
        parallel_onnx_path = output_config.run.checkpoint_dir / PARALLEL_ONNX_FILENAME
        copy_model_artifact_bundle(latest_onnx_path, incumbent_onnx_path)
        copy_model_artifact_bundle(latest_onnx_path, parallel_onnx_path)
        assert_rust_inference_artifacts(incumbent_onnx_path)
        assert_rust_inference_artifacts(parallel_onnx_path)

        eval_config = copy.deepcopy(output_config.self_play)
        eval_num_workers = max(
            2,
            args.eval_num_workers
            if args.eval_num_workers > 0
            else eval_config.num_workers,
        )
        eval_num_simulations = (
            args.eval_num_simulations
            if args.eval_num_simulations is not None
            else eval_config.num_simulations
        )
        eval_max_placements = (
            args.eval_max_placements
            if args.eval_max_placements is not None
            else eval_config.max_placements
        )
        resolved_mcts_seed = (
            args.mcts_seed if args.mcts_seed is not None else eval_config.mcts_seed
        )
        mcts_config = MCTSConfig()
        mcts_config.num_simulations = eval_num_simulations
        mcts_config.c_puct = eval_config.c_puct
        mcts_config.temperature = eval_config.temperature
        mcts_config.dirichlet_alpha = eval_config.dirichlet_alpha
        mcts_config.dirichlet_epsilon = eval_config.dirichlet_epsilon
        mcts_config.visit_sampling_epsilon = 0.0
        mcts_config.max_placements = eval_max_placements
        mcts_config.q_scale = (
            eval_config.q_scale if eval_config.use_tanh_q_normalization else None
        )
        mcts_config.reuse_tree = eval_config.reuse_tree
        mcts_config.use_parent_value_for_unvisited_q = (
            eval_config.use_parent_value_for_unvisited_q
        )
        mcts_config.nn_value_weight = output_config.self_play.nn_value_weight
        mcts_config.death_penalty = output_config.self_play.death_penalty
        mcts_config.overhang_penalty_weight = (
            output_config.self_play.overhang_penalty_weight
        )
        mcts_config.seed = resolved_mcts_seed

        eval_seeds = list(
            range(args.eval_seed_start, args.eval_seed_start + args.num_eval_games)
        )
        eval_result = evaluate_model(
            model_path=str(incumbent_onnx_path),
            seeds=eval_seeds,
            config=mcts_config,
            max_placements=eval_max_placements,
            num_workers=eval_num_workers,
            add_noise=False,
        )
        eval_metrics = {
            "num_games": int(eval_result.num_games),
            "avg_attack": float(eval_result.avg_attack),
            "avg_lines": float(eval_result.avg_lines),
            "avg_moves": float(eval_result.avg_moves),
            "max_attack": int(eval_result.max_attack),
            "attack_per_piece": float(eval_result.attack_per_piece),
            "avg_tree_nodes": float(eval_result.avg_tree_nodes),
            "avg_trajectory_predicted_total_attack_variance": float(
                eval_result.avg_trajectory_predicted_total_attack_variance
            ),
            "avg_trajectory_predicted_total_attack_std": float(
                eval_result.avg_trajectory_predicted_total_attack_std
            ),
            "avg_trajectory_predicted_total_attack_rmse": float(
                eval_result.avg_trajectory_predicted_total_attack_rmse
            ),
            "game_results": [
                {"attack": int(attack), "moves": int(moves)}
                for attack, moves in eval_result.game_results
            ],
        }
        log_wandb(
            {
                "offline_step": num_steps,
                "final_eval/num_games": eval_metrics["num_games"],
                "final_eval/avg_attack": eval_metrics["avg_attack"],
                "final_eval/avg_lines": eval_metrics["avg_lines"],
                "final_eval/avg_moves": eval_metrics["avg_moves"],
                "final_eval/max_attack": eval_metrics["max_attack"],
                "final_eval/attack_per_piece": eval_metrics["attack_per_piece"],
                "final_eval/avg_tree_nodes": eval_metrics["avg_tree_nodes"],
                "final_eval/avg_trajectory_predicted_total_attack_variance": (
                    eval_metrics["avg_trajectory_predicted_total_attack_variance"]
                ),
                "final_eval/avg_trajectory_predicted_total_attack_std": (
                    eval_metrics["avg_trajectory_predicted_total_attack_std"]
                ),
                "final_eval/avg_trajectory_predicted_total_attack_rmse": (
                    eval_metrics["avg_trajectory_predicted_total_attack_rmse"]
                ),
                "final_eval/seed_start": args.eval_seed_start,
                "final_eval/mcts_seed": resolved_mcts_seed,
                "final_eval/num_workers": eval_num_workers,
                "final_eval/num_simulations": eval_num_simulations,
                "final_eval/max_placements": eval_max_placements,
            }
        )

        checkpoint_path = saved_paths["checkpoint"]
        save_checkpoint(
            trainer.model,
            trainer.optimizer,
            trainer.scheduler,
            step=0,
            filepath=checkpoint_path,
            incumbent_uses_network=True,
            incumbent_model_step=0,
            incumbent_nn_value_weight=output_config.self_play.nn_value_weight,
            incumbent_death_penalty=output_config.self_play.death_penalty,
            incumbent_overhang_penalty_weight=(
                output_config.self_play.overhang_penalty_weight
            ),
            incumbent_eval_avg_attack=eval_metrics["avg_attack"],
            incumbent_model_source_path=str(incumbent_onnx_path),
            incumbent_model_artifact=INCUMBENT_ONNX_FILENAME,
        )
        export_metadata(
            output_config.run.checkpoint_dir / LATEST_METADATA_FILENAME,
            step=0,
            eval_metrics={
                "warm_start_eval": eval_metrics,
                "offline_best": best_record,
            },
            config=json.loads(config_to_json(output_config)),
        )

        summary = {
            "source_run_dir": str(source_run_dir),
            "output_run_dir": str(output_config.run.run_dir),
            "source_training_data_path": str(source_training_data_path),
            "copied_training_data_path": str(output_training_data_path),
            "device": device_str,
            "epochs": args.epochs,
            "num_steps": num_steps,
            "seed": args.seed,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "grad_clip_norm": grad_clip_norm,
            "train_fraction": args.train_fraction,
            "max_examples": args.max_examples,
            "preload_mode": preload_mode,
            "offline_dataset": {
                "total_examples": dataset_setup.total_examples,
                "used_examples": dataset_setup.num_selected,
                "train_examples": len(dataset_setup.train_local_indices),
                "val_examples": len(dataset_setup.val_local_indices),
                "train_eval_examples": len(dataset_setup.train_eval_local_indices),
                "val_eval_examples": len(dataset_setup.val_eval_local_indices),
                "preload_seconds": dataset_setup.preload_sec,
            },
            "offline_best": best_record,
            "offline_history": offline_history,
            "final_eval": {
                **eval_metrics,
                "seed_start": args.eval_seed_start,
                "mcts_seed": resolved_mcts_seed,
                "num_workers": eval_num_workers,
                "num_simulations": eval_num_simulations,
                "max_placements": eval_max_placements,
            },
            "artifacts": {
                "latest_checkpoint": str(checkpoint_path),
                "latest_onnx": str(latest_onnx_path),
                "incumbent_onnx": str(incumbent_onnx_path),
                "parallel_onnx": str(parallel_onnx_path),
            },
        }
        save_summary(
            output_config.run.run_dir / "analysis" / "warm_start_summary.json",
            summary,
        )
        log_wandb(
            {
                "offline_step": num_steps,
                "warm_start/best_step": best_record["step"],
                "warm_start/best_epochs_seen": best_record["epochs_seen"],
                "warm_start/best_train_total_loss": best_record["train_total_loss"],
                "warm_start/best_val_total_loss": best_record["val_total_loss"],
            }
        )

        logger.info(
            "Warm start complete",
            output_run_dir=str(output_config.run.run_dir),
            incumbent_onnx=str(incumbent_onnx_path),
            avg_attack=eval_metrics["avg_attack"],
            num_eval_games=eval_metrics["num_games"],
        )
    finally:
        wandb.finish()


if __name__ == "__main__":
    main(parse(ScriptArgs))
