from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import structlog
import torch
import wandb
from simple_parsing import parse

from tetris_mcts.config import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    NUM_ACTIONS,
    PROJECT_ROOT,
)
from tetris_mcts.ml.network import AUX_FEATURES, TetrisNet
from compare_offline_architectures import (
    FlopBreakdown,
    OfflineDataSource,
    OfflineTensorDataset,
    build_tensor_dataset,
    count_parameters,
    ensure_required_keys,
    get_preload_mode,
    pick_device,
    select_subset,
    tensor_dataset_bytes,
    train_offline_model,
    validate_shapes,
)

logger = structlog.get_logger()


@dataclass
class ScriptArgs:
    data_path: Path = (
        PROJECT_ROOT / "training_runs" / "v17" / "training_data.npz"
    )  # Path to offline replay buffer NPZ
    device: str = "auto"  # auto/cpu/cuda/mps
    seed: int = 123
    max_examples: int = 0  # 0 = use all examples in NPZ
    train_fraction: float = 0.9
    steps: int = 20000
    batch_size: int = 1024
    eval_interval: int = 100
    eval_examples: int = 32_768  # Max examples to use per train/val eval pass
    eval_batch_size: int = 2048
    log_train_metrics_every: int = 1  # Batch metric logging cadence
    preload_to_gpu: bool = True  # Preload selected dataset tensors to GPU
    preload_to_ram: bool = False  # Preload selected dataset tensors to CPU RAM
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    value_loss_weight: float = 1.0

    conv_filters: list[int] = field(default_factory=lambda: [4, 8])
    fc_hidden: int = 128
    aux_hidden: int = 24
    num_fusion_blocks: int = 0
    conv_kernel_size: int = 3
    conv_padding: int = 1

    board_trunk_multiplier: int = 2  # Multiply conv filter counts
    post_fusion_multiplier: int = 2  # Multiply fc_hidden size
    cache_hit_rate_for_effective_flops: float = 0.96

    wandb_project: str = "tetris-mcts-offline"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(
        default_factory=lambda: ["offline", "network-scaling"]
    )


@dataclass(frozen=True)
class ScalingVariant:
    name: str
    wandb_prefix: str
    conv_filters: tuple[int, int]
    fc_hidden: int


def validate_args(args: ScriptArgs) -> None:
    if args.steps <= 0:
        raise ValueError("steps must be > 0")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if args.eval_interval <= 0:
        raise ValueError("eval_interval must be > 0")
    if args.log_train_metrics_every <= 0:
        raise ValueError("log_train_metrics_every must be > 0")
    if args.eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be > 0")
    if args.eval_examples <= 0:
        raise ValueError("eval_examples must be > 0")
    if args.max_examples < 0:
        raise ValueError("max_examples must be >= 0")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if args.grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be > 0")
    if len(args.conv_filters) != 2:
        raise ValueError("conv_filters must contain exactly two values")
    if any(f <= 0 for f in args.conv_filters):
        raise ValueError("conv_filters values must be > 0")
    if args.fc_hidden <= 0:
        raise ValueError("fc_hidden must be > 0")
    if args.aux_hidden <= 0:
        raise ValueError("aux_hidden must be > 0")
    if args.num_fusion_blocks < 0:
        raise ValueError("num_fusion_blocks must be >= 0")
    if args.conv_kernel_size <= 0:
        raise ValueError("conv_kernel_size must be > 0")
    if args.board_trunk_multiplier <= 0:
        raise ValueError("board_trunk_multiplier must be > 0")
    if args.post_fusion_multiplier <= 0:
        raise ValueError("post_fusion_multiplier must be > 0")
    if not 0.0 <= args.cache_hit_rate_for_effective_flops <= 1.0:
        raise ValueError("cache_hit_rate_for_effective_flops must be in [0, 1]")


def build_variants(args: ScriptArgs) -> list[ScalingVariant]:
    base_conv = tuple(args.conv_filters)
    trunk_scaled = tuple(f * args.board_trunk_multiplier for f in base_conv)
    post_fusion_scaled_hidden = args.fc_hidden * args.post_fusion_multiplier

    variants = [
        ScalingVariant(
            name="default",
            wandb_prefix="default",
            conv_filters=(int(base_conv[0]), int(base_conv[1])),
            fc_hidden=args.fc_hidden,
        ),
        ScalingVariant(
            name="double_board_trunk",
            wandb_prefix="double_board_trunk",
            conv_filters=(int(trunk_scaled[0]), int(trunk_scaled[1])),
            fc_hidden=args.fc_hidden,
        ),
        ScalingVariant(
            name="double_post_fusion",
            wandb_prefix="double_post_fusion",
            conv_filters=(int(base_conv[0]), int(base_conv[1])),
            fc_hidden=post_fusion_scaled_hidden,
        ),
    ]

    seen: set[tuple[tuple[int, int], int]] = set()
    for variant in variants:
        key = (variant.conv_filters, variant.fc_hidden)
        if key in seen:
            raise ValueError(
                "Variant definitions are not unique; adjust multipliers "
                "so default, double_board_trunk, and double_post_fusion differ"
            )
        seen.add(key)

    return variants


def conv_out_size(size: int, kernel_size: int, padding: int) -> int:
    return size + 2 * padding - kernel_size + 1


def conv_output_hw(kernel_size: int, padding: int) -> tuple[int, int, int, int]:
    h1 = conv_out_size(BOARD_HEIGHT, kernel_size, padding)
    w1 = conv_out_size(BOARD_WIDTH, kernel_size, padding)
    h2 = conv_out_size(h1, kernel_size, padding)
    w2 = conv_out_size(w1, kernel_size, padding)
    if h1 <= 0 or w1 <= 0 or h2 <= 0 or w2 <= 0:
        raise ValueError(
            "Invalid conv output shape; adjust conv_kernel_size/conv_padding "
            f"(got h2={h2}, w2={w2})"
        )
    return h1, w1, h2, w2


def variant_flop_breakdown(
    variant: ScalingVariant,
    args: ScriptArgs,
) -> FlopBreakdown:
    conv0, conv1 = variant.conv_filters
    k = args.conv_kernel_size
    h1, w1, h2, w2 = conv_output_hw(k, args.conv_padding)
    conv_flat = h2 * w2 * conv1
    fusion_hidden = variant.fc_hidden

    miss_only = 0
    miss_only += h1 * w1 * conv0 * (2 * 1 * k * k + 1)
    miss_only += 2 * h1 * w1 * conv0
    miss_only += h2 * w2 * conv1 * (2 * conv0 * k * k + 1)
    miss_only += 2 * h2 * w2 * conv1
    miss_only += h1 * w1 * conv0 + h2 * w2 * conv1
    miss_only += fusion_hidden * (2 * conv_flat + 1)

    hit_path = 0
    hit_path += args.aux_hidden * (2 * AUX_FEATURES + 1)
    hit_path += 5 * args.aux_hidden
    hit_path += args.aux_hidden
    hit_path += fusion_hidden * (2 * args.aux_hidden + 1)
    hit_path += fusion_hidden * (2 * args.aux_hidden + 1)
    hit_path += 3 * fusion_hidden
    hit_path += 5 * fusion_hidden
    hit_path += fusion_hidden

    for _ in range(args.num_fusion_blocks):
        hit_path += 5 * fusion_hidden
        hit_path += fusion_hidden
        hit_path += fusion_hidden * (2 * fusion_hidden + 1)
        hit_path += 5 * fusion_hidden
        hit_path += fusion_hidden
        hit_path += fusion_hidden * (2 * fusion_hidden + 1)
        hit_path += fusion_hidden

    hit_path += NUM_ACTIONS * (2 * fusion_hidden + 1)
    hit_path += 2 * fusion_hidden + 1

    full = miss_only + hit_path
    effective = hit_path + (1.0 - args.cache_hit_rate_for_effective_flops) * miss_only
    return FlopBreakdown(
        miss_only=miss_only,
        hit_path=hit_path,
        full=full,
        effective=effective,
    )


def normalize_args_for_wandb(
    args: ScriptArgs,
    variants: list[ScalingVariant],
) -> dict:
    normalized = asdict(args)
    normalized["data_path"] = str(args.data_path)
    normalized["variants"] = [
        {
            "name": variant.name,
            "wandb_prefix": variant.wandb_prefix,
            "conv_filters": list(variant.conv_filters),
            "fc_hidden": variant.fc_hidden,
        }
        for variant in variants
    ]
    return normalized


def main(args: ScriptArgs) -> None:
    validate_args(args)
    if not args.data_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {args.data_path}")
    if args.data_path.suffix != ".npz":
        raise ValueError(f"Expected .npz file, got: {args.data_path}")

    variants = build_variants(args)
    device_str = pick_device(args.device)
    device = torch.device(device_str)
    preload_mode = get_preload_mode(args)
    if preload_mode == "gpu" and device.type == "cpu":
        raise ValueError("preload_to_gpu requires a non-CPU device")
    logger.info("Using device", device=device_str, variants=[v.name for v in variants])

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=args.wandb_tags,
        config=normalize_args_for_wandb(args, variants),
    )
    run = wandb.run
    if run is None:
        raise RuntimeError("wandb.init did not create a run")

    wandb.define_metric("offline_step")
    wandb.define_metric("variants/*", step_metric="offline_step")

    npz = np.load(args.data_path, mmap_mode="r")
    try:
        ensure_required_keys(npz)
        total_examples = validate_shapes(npz)
        selected_global_indices = np.arange(total_examples, dtype=np.int64)

        split_rng = np.random.default_rng(args.seed)
        split_rng.shuffle(selected_global_indices)
        if args.max_examples > 0:
            selected_global_indices = selected_global_indices[: args.max_examples]

        num_selected = len(selected_global_indices)
        split_point = int(num_selected * args.train_fraction)
        if split_point <= 0 or split_point >= num_selected:
            raise ValueError(
                "Invalid train/val split; adjust max_examples or train_fraction"
            )

        train_local_indices = np.arange(split_point, dtype=np.int64)
        val_local_indices = np.arange(split_point, num_selected, dtype=np.int64)

        preload_start = time.perf_counter()
        tensor_data: OfflineTensorDataset | None = None
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
            max_examples=args.eval_examples,
            seed=args.seed + 1,
        )
        val_eval_local_indices = select_subset(
            val_local_indices,
            max_examples=args.eval_examples,
            seed=args.seed + 2,
        )

        logger.info(
            "Dataset split",
            total_examples=total_examples,
            used_examples=num_selected,
            train_examples=len(train_local_indices),
            val_examples=len(val_local_indices),
            train_eval_examples=len(train_eval_local_indices),
            val_eval_examples=len(val_eval_local_indices),
            preload_mode=preload_mode,
            preload_seconds=preload_sec,
        )

        dataset_log: dict[str, float | int | str] = {
            "dataset/total_examples": total_examples,
            "dataset/used_examples": num_selected,
            "dataset/train_examples": len(train_local_indices),
            "dataset/val_examples": len(val_local_indices),
            "dataset/train_eval_examples": len(train_eval_local_indices),
            "dataset/val_eval_examples": len(val_eval_local_indices),
            "dataset/preload_mode": preload_mode,
            "dataset/preload_seconds": preload_sec,
        }
        if tensor_data is not None:
            dataset_log["dataset/preload_bytes"] = tensor_dataset_bytes(tensor_data)
        wandb.log(dataset_log)

        results: list[dict] = []
        for index, variant in enumerate(variants):
            torch.manual_seed(args.seed)
            model = TetrisNet(
                conv_filters=list(variant.conv_filters),
                fc_hidden=variant.fc_hidden,
                conv_kernel_size=args.conv_kernel_size,
                conv_padding=args.conv_padding,
                aux_hidden=args.aux_hidden,
                num_fusion_blocks=args.num_fusion_blocks,
            ).to(device)

            num_parameters = count_parameters(model)
            flops = variant_flop_breakdown(variant, args)
            wandb.log(
                {
                    f"variants/{variant.wandb_prefix}/num_parameters": num_parameters,
                    f"variants/{variant.wandb_prefix}/conv_filter_0": variant.conv_filters[
                        0
                    ],
                    f"variants/{variant.wandb_prefix}/conv_filter_1": variant.conv_filters[
                        1
                    ],
                    f"variants/{variant.wandb_prefix}/fc_hidden": variant.fc_hidden,
                    f"variants/{variant.wandb_prefix}/aux_hidden": args.aux_hidden,
                    f"variants/{variant.wandb_prefix}/num_fusion_blocks": (
                        args.num_fusion_blocks
                    ),
                    f"variants/{variant.wandb_prefix}/miss_only_flops": flops.miss_only,
                    f"variants/{variant.wandb_prefix}/hit_path_flops": flops.hit_path,
                    f"variants/{variant.wandb_prefix}/full_flops": flops.full,
                    f"variants/{variant.wandb_prefix}/effective_flops": flops.effective,
                    "arch/cache_hit_rate_for_effective_flops": (
                        args.cache_hit_rate_for_effective_flops
                    ),
                }
            )
            logger.info(
                "Training variant",
                variant=variant.name,
                variant_index=index,
                num_variants=len(variants),
                conv_filters=variant.conv_filters,
                fc_hidden=variant.fc_hidden,
                params=num_parameters,
                effective_flops=flops.effective,
            )

            result = train_offline_model(
                model_name=variant.name,
                wandb_prefix=f"variants/{variant.wandb_prefix}",
                model=model,
                source=source,
                train_local_indices=train_local_indices,
                train_eval_local_indices=train_eval_local_indices,
                val_eval_local_indices=val_eval_local_indices,
                args=args,
                device=device,
                schedule_seed=args.seed + 12345,
            )
            result["variant"] = variant
            result["flops"] = flops
            results.append(result)

        winner = min(results, key=lambda x: x["final"]["val_total_loss"])
        default = next(
            result for result in results if result["variant"].wandb_prefix == "default"
        )

        comparison_log: dict[str, float | int | str] = {
            "comparison/winner": winner["variant"].name,
            "comparison/winner_is_default": (
                1 if winner["variant"].wandb_prefix == "default" else 0
            ),
        }

        for result in results:
            prefix = result["variant"].wandb_prefix
            final = result["final"]
            comparison_log[f"comparison/final_val_total_loss/{prefix}"] = final[
                "val_total_loss"
            ]
            comparison_log[f"comparison/final_val_policy_loss/{prefix}"] = final[
                "val_policy_loss"
            ]
            comparison_log[f"comparison/final_val_value_loss/{prefix}"] = final[
                "val_value_loss"
            ]
            comparison_log[f"comparison/final_train_total_loss/{prefix}"] = final[
                "train_total_loss"
            ]
            comparison_log[f"comparison/final_elapsed_sec/{prefix}"] = result[
                "elapsed_sec"
            ]
            comparison_log[f"comparison/final_train_batches_per_sec/{prefix}"] = final[
                "train_batches_per_sec"
            ]
            comparison_log[f"comparison/final_train_examples_per_sec/{prefix}"] = (
                final["train_batches_per_sec"] * args.batch_size
            )
            comparison_log[f"comparison/num_parameters/{prefix}"] = result[
                "num_parameters"
            ]
            comparison_log[f"comparison/effective_flops/{prefix}"] = result[
                "flops"
            ].effective

        default_final = default["final"]
        default_batches_per_sec = default_final["train_batches_per_sec"]
        default_val_total_loss = default_final["val_total_loss"]

        for result in results:
            prefix = result["variant"].wandb_prefix
            if prefix == "default":
                continue
            final = result["final"]
            comparison_log[f"comparison/val_total_loss_delta_vs_default/{prefix}"] = (
                final["val_total_loss"] - default_val_total_loss
            )
            comparison_log[f"comparison/train_speedup_vs_default/{prefix}"] = (
                final["train_batches_per_sec"] / default_batches_per_sec
                if default_batches_per_sec > 0
                else 0.0
            )

        wandb.log(comparison_log)

        run.summary["winner_by_final_val_total_loss"] = winner["variant"].name
        run.summary["num_variants"] = len(results)
        run.summary["cache_hit_rate_for_effective_flops"] = (
            args.cache_hit_rate_for_effective_flops
        )

        for result in results:
            prefix = result["variant"].wandb_prefix
            run.summary[f"{prefix}_final_val_total_loss"] = result["final"][
                "val_total_loss"
            ]
            run.summary[f"{prefix}_final_train_batches_per_sec"] = result["final"][
                "train_batches_per_sec"
            ]
            run.summary[f"{prefix}_num_parameters"] = result["num_parameters"]
            run.summary[f"{prefix}_effective_flops"] = result["flops"].effective

        logger.info(
            "Offline network scaling comparison complete",
            winner=winner["variant"].name,
            default_final_val_total_loss=default["final"]["val_total_loss"],
            winner_final_val_total_loss=winner["final"]["val_total_loss"],
        )
    finally:
        npz.close()
        wandb.finish()


if __name__ == "__main__":
    main(parse(ScriptArgs))
