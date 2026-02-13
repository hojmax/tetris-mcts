from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from simple_parsing import parse

from tetris_mcts.config import BOARD_HEIGHT, BOARD_WIDTH, NUM_ACTIONS, PROJECT_ROOT
from tetris_mcts.ml.loss import compute_loss
from tetris_mcts.ml.network import AUX_FEATURES, TetrisNet

logger = structlog.get_logger()

REQUIRED_NPZ_KEYS = (
    "boards",
    "current_pieces",
    "hold_pieces",
    "hold_available",
    "next_queue",
    "placement_counts",
    "policy_targets",
    "value_targets",
    "action_masks",
)


@dataclass
class ScriptArgs:
    data_path: Path = (
        PROJECT_ROOT / "training_runs" / "v17" / "training_data.npz"
    )  # Path to offline replay buffer NPZ
    device: str = "auto"  # auto/cpu/cuda/mps
    seed: int = 123
    max_examples: int = 200_000  # 0 = use all examples in NPZ
    train_fraction: float = 0.9
    steps: int = 1500
    batch_size: int = 1024
    eval_interval: int = 100
    eval_examples: int = 32_768  # Max examples to use per train/val eval pass
    eval_batch_size: int = 2048
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    value_loss_weight: float = 1.0

    conv_filters: list[int] = field(default_factory=lambda: [4, 8])
    fc_hidden: int = 128
    conv_kernel_size: int = 3
    conv_padding: int = 1

    match_aux_hidden_min: int = 32
    match_aux_hidden_max: int = 256
    match_aux_hidden_step: int = 8
    match_fusion_hidden_min: int = 32
    match_fusion_hidden_max: int = 384
    match_fusion_hidden_step: int = 8
    match_num_fusion_blocks_options: list[int] = field(
        default_factory=lambda: [1, 2, 3]
    )
    match_param_tolerance: float = 0.01  # Relative tolerance
    match_flop_tolerance: float = (  # Relative tolerance on cache-weighted FLOPs
        0.01
    )
    match_flop_weight: float = 1.0  # Score weight for cache-weighted FLOP error
    cache_hit_rate_for_matching: float = (
        0.96
    )  # Expected board-cache hit rate for effective FLOP matching

    wandb_project: str = "tetris-mcts-offline"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(
        default_factory=lambda: ["offline", "architecture-compare"]
    )


@dataclass
class MatchedGatedConfig:
    aux_hidden: int
    fusion_hidden: int
    num_fusion_blocks: int
    params: int
    miss_only_flops: int
    hit_path_flops: int
    full_flops: int
    effective_flops: float
    param_rel_error: float
    flop_rel_error: float
    score: float


@dataclass
class FlopBreakdown:
    miss_only: int
    hit_path: int
    full: int
    effective: float


class ResidualFusionBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.ln1(x))
        h = self.fc1(h)
        h = F.relu(self.ln2(h))
        h = self.fc2(h)
        return x + h


class GatedFusionTetrisNet(nn.Module):
    def __init__(
        self,
        conv_filters: list[int],
        conv_kernel_size: int,
        conv_padding: int,
        aux_hidden: int,
        fusion_hidden: int,
        num_fusion_blocks: int,
    ):
        super().__init__()
        if len(conv_filters) != 2:
            raise ValueError(
                f"Expected exactly 2 convolutional filters, got {len(conv_filters)}"
            )
        if num_fusion_blocks <= 0:
            raise ValueError("num_fusion_blocks must be > 0")

        conv0 = conv_filters[0]
        conv1 = conv_filters[1]
        conv_flat_size = BOARD_HEIGHT * BOARD_WIDTH * conv1

        self.conv1 = nn.Conv2d(
            1, conv0, kernel_size=conv_kernel_size, padding=conv_padding
        )
        self.bn1 = nn.BatchNorm2d(conv0)
        self.conv2 = nn.Conv2d(
            conv0,
            conv1,
            kernel_size=conv_kernel_size,
            padding=conv_padding,
        )
        self.bn2 = nn.BatchNorm2d(conv1)

        self.board_proj = nn.Linear(conv_flat_size, fusion_hidden)
        self.aux_fc = nn.Linear(AUX_FEATURES, aux_hidden)
        self.aux_ln = nn.LayerNorm(aux_hidden)
        self.gate_fc = nn.Linear(aux_hidden, fusion_hidden)
        self.aux_proj = nn.Linear(aux_hidden, fusion_hidden)
        self.fusion_ln = nn.LayerNorm(fusion_hidden)
        self.fusion_blocks = nn.ModuleList(
            [ResidualFusionBlock(fusion_hidden) for _ in range(num_fusion_blocks)]
        )
        self.policy_head = nn.Linear(fusion_hidden, NUM_ACTIONS)
        self.value_head = nn.Linear(fusion_hidden, 1)

    def forward(
        self,
        board: torch.Tensor,
        aux_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn1(self.conv1(board)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)

        board_h = self.board_proj(x)
        aux_h = F.relu(self.aux_ln(self.aux_fc(aux_features)))
        gate = torch.sigmoid(self.gate_fc(aux_h))

        fused = board_h * (1.0 + gate) + self.aux_proj(aux_h)
        fused = F.relu(self.fusion_ln(fused))
        for block in self.fusion_blocks:
            fused = block(fused)

        policy_logits = self.policy_head(fused)
        value = self.value_head(fused)
        return policy_logits, value


def pick_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def baseline_flop_breakdown(args: ScriptArgs, cache_hit_rate: float) -> FlopBreakdown:
    conv0, conv1 = args.conv_filters
    k = args.conv_kernel_size
    h1, w1, h2, w2 = conv_output_hw(k, args.conv_padding)
    conv_flat = h2 * w2 * conv1

    miss_only = 0
    miss_only += h1 * w1 * conv0 * (2 * 1 * k * k + 1)
    miss_only += 2 * h1 * w1 * conv0
    miss_only += h2 * w2 * conv1 * (2 * conv0 * k * k + 1)
    miss_only += 2 * h2 * w2 * conv1
    miss_only += h1 * w1 * conv0 + h2 * w2 * conv1
    # Cached board embedding in Rust includes FC(board) + bias
    miss_only += args.fc_hidden * (2 * conv_flat + 1)

    hit_path = 0
    # Aux contribution and merge with cached board embedding
    hit_path += args.fc_hidden * (2 * AUX_FEATURES + 1)
    hit_path += args.fc_hidden
    # LayerNorm + ReLU and heads
    hit_path += 5 * args.fc_hidden
    hit_path += args.fc_hidden
    hit_path += NUM_ACTIONS * (2 * args.fc_hidden + 1)
    hit_path += 2 * args.fc_hidden + 1

    full = miss_only + hit_path
    effective = hit_path + (1.0 - cache_hit_rate) * miss_only
    return FlopBreakdown(
        miss_only=miss_only,
        hit_path=hit_path,
        full=full,
        effective=effective,
    )


def gated_flop_breakdown(
    args: ScriptArgs,
    aux_hidden: int,
    fusion_hidden: int,
    num_fusion_blocks: int,
    cache_hit_rate: float,
) -> FlopBreakdown:
    conv0, conv1 = args.conv_filters
    k = args.conv_kernel_size
    h1, w1, h2, w2 = conv_output_hw(k, args.conv_padding)
    conv_flat = h2 * w2 * conv1

    miss_only = 0
    miss_only += h1 * w1 * conv0 * (2 * 1 * k * k + 1)
    miss_only += 2 * h1 * w1 * conv0
    miss_only += h2 * w2 * conv1 * (2 * conv0 * k * k + 1)
    miss_only += 2 * h2 * w2 * conv1
    miss_only += h1 * w1 * conv0 + h2 * w2 * conv1
    # Cached board path includes board projection.
    miss_only += fusion_hidden * (2 * conv_flat + 1)

    hit_path = 0
    hit_path += aux_hidden * (2 * AUX_FEATURES + 1)
    hit_path += 5 * aux_hidden
    hit_path += aux_hidden
    hit_path += fusion_hidden * (2 * aux_hidden + 1)
    hit_path += fusion_hidden * (2 * aux_hidden + 1)
    # board_h * (1 + gate) + aux_proj
    hit_path += 3 * fusion_hidden
    hit_path += 5 * fusion_hidden
    hit_path += fusion_hidden

    for _ in range(num_fusion_blocks):
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
    effective = hit_path + (1.0 - cache_hit_rate) * miss_only
    return FlopBreakdown(
        miss_only=miss_only,
        hit_path=hit_path,
        full=full,
        effective=effective,
    )


def gated_parameter_count(
    args: ScriptArgs,
    aux_hidden: int,
    fusion_hidden: int,
    num_fusion_blocks: int,
) -> int:
    conv0, conv1 = args.conv_filters
    k = args.conv_kernel_size
    _, _, h2, w2 = conv_output_hw(k, args.conv_padding)
    conv_flat = h2 * w2 * conv1

    params = 0
    params += conv0 * k * k + conv0
    params += 2 * conv0
    params += conv1 * conv0 * k * k + conv1
    params += 2 * conv1

    params += conv_flat * fusion_hidden + fusion_hidden
    params += AUX_FEATURES * aux_hidden + aux_hidden
    params += 2 * aux_hidden
    params += aux_hidden * fusion_hidden + fusion_hidden
    params += aux_hidden * fusion_hidden + fusion_hidden
    params += 2 * fusion_hidden

    for _ in range(num_fusion_blocks):
        params += 2 * fusion_hidden
        params += fusion_hidden * fusion_hidden + fusion_hidden
        params += 2 * fusion_hidden
        params += fusion_hidden * fusion_hidden + fusion_hidden

    params += fusion_hidden * NUM_ACTIONS + NUM_ACTIONS
    params += fusion_hidden + 1
    return params


def find_matched_gated_config(
    args: ScriptArgs,
    target_params: int,
    target_effective_flops: float,
) -> MatchedGatedConfig:
    best: MatchedGatedConfig | None = None
    for num_blocks in args.match_num_fusion_blocks_options:
        for aux_hidden in range(
            args.match_aux_hidden_min,
            args.match_aux_hidden_max + 1,
            args.match_aux_hidden_step,
        ):
            for fusion_hidden in range(
                args.match_fusion_hidden_min,
                args.match_fusion_hidden_max + 1,
                args.match_fusion_hidden_step,
            ):
                params = gated_parameter_count(
                    args, aux_hidden, fusion_hidden, num_blocks
                )
                flops = gated_flop_breakdown(
                    args=args,
                    aux_hidden=aux_hidden,
                    fusion_hidden=fusion_hidden,
                    num_fusion_blocks=num_blocks,
                    cache_hit_rate=args.cache_hit_rate_for_matching,
                )
                param_rel_error = abs(params - target_params) / target_params
                flop_rel_error = (
                    abs(flops.effective - target_effective_flops)
                    / target_effective_flops
                )
                score = param_rel_error + args.match_flop_weight * flop_rel_error
                candidate = MatchedGatedConfig(
                    aux_hidden=aux_hidden,
                    fusion_hidden=fusion_hidden,
                    num_fusion_blocks=num_blocks,
                    params=params,
                    miss_only_flops=flops.miss_only,
                    hit_path_flops=flops.hit_path,
                    full_flops=flops.full,
                    effective_flops=flops.effective,
                    param_rel_error=param_rel_error,
                    flop_rel_error=flop_rel_error,
                    score=score,
                )
                if best is None or candidate.score < best.score:
                    best = candidate

    if best is None:
        raise ValueError("No gated architecture candidates were evaluated")

    if best.param_rel_error > args.match_param_tolerance:
        raise ValueError(
            "Unable to match parameter count within tolerance. "
            f"Best relative error={best.param_rel_error:.6f}, "
            f"tolerance={args.match_param_tolerance:.6f}. "
            "Expand match search ranges."
        )
    if best.flop_rel_error > args.match_flop_tolerance:
        raise ValueError(
            "Unable to match cache-weighted FLOPs within tolerance. "
            f"Best relative error={best.flop_rel_error:.6f}, "
            f"tolerance={args.match_flop_tolerance:.6f}. "
            "Expand match search ranges."
        )
    return best


def validate_args(args: ScriptArgs) -> None:
    if args.steps <= 0:
        raise ValueError("steps must be > 0")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if args.eval_interval <= 0:
        raise ValueError("eval_interval must be > 0")
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
    if args.conv_kernel_size <= 0:
        raise ValueError("conv_kernel_size must be > 0")
    if args.match_aux_hidden_step <= 0:
        raise ValueError("match_aux_hidden_step must be > 0")
    if args.match_fusion_hidden_step <= 0:
        raise ValueError("match_fusion_hidden_step must be > 0")
    if args.match_aux_hidden_min <= 0 or args.match_aux_hidden_max <= 0:
        raise ValueError("match_aux_hidden bounds must be > 0")
    if args.match_fusion_hidden_min <= 0 or args.match_fusion_hidden_max <= 0:
        raise ValueError("match_fusion_hidden bounds must be > 0")
    if args.match_aux_hidden_min > args.match_aux_hidden_max:
        raise ValueError("match_aux_hidden_min must be <= match_aux_hidden_max")
    if args.match_fusion_hidden_min > args.match_fusion_hidden_max:
        raise ValueError("match_fusion_hidden_min must be <= match_fusion_hidden_max")
    if any(blocks <= 0 for blocks in args.match_num_fusion_blocks_options):
        raise ValueError("match_num_fusion_blocks_options must contain only positive values")
    if args.match_param_tolerance <= 0:
        raise ValueError("match_param_tolerance must be > 0")
    if args.match_flop_tolerance <= 0:
        raise ValueError("match_flop_tolerance must be > 0")
    if args.match_flop_weight <= 0:
        raise ValueError("match_flop_weight must be > 0")
    if not 0.0 <= args.cache_hit_rate_for_matching <= 1.0:
        raise ValueError("cache_hit_rate_for_matching must be in [0, 1]")


def ensure_required_keys(data: np.lib.npyio.NpzFile) -> None:
    missing = [key for key in REQUIRED_NPZ_KEYS if key not in data]
    if missing:
        raise KeyError(f"NPZ is missing required keys: {missing}")


def validate_shapes(data: np.lib.npyio.NpzFile) -> int:
    n = int(data["boards"].shape[0])
    if data["boards"].shape[1:] != (BOARD_HEIGHT, BOARD_WIDTH):
        raise ValueError(
            f"boards must have shape (N, {BOARD_HEIGHT}, {BOARD_WIDTH}), "
            f"got {data['boards'].shape}"
        )
    if data["current_pieces"].shape[1] != 7:
        raise ValueError("current_pieces must have shape (N, 7)")
    if data["hold_pieces"].shape[1] != 8:
        raise ValueError("hold_pieces must have shape (N, 8)")
    if data["next_queue"].shape[1:] != (5, 7):
        raise ValueError("next_queue must have shape (N, 5, 7)")
    if data["policy_targets"].shape[1] != NUM_ACTIONS:
        raise ValueError(f"policy_targets must have shape (N, {NUM_ACTIONS})")
    if data["action_masks"].shape[1] != NUM_ACTIONS:
        raise ValueError(f"action_masks must have shape (N, {NUM_ACTIONS})")

    per_example_keys = (
        "current_pieces",
        "hold_pieces",
        "hold_available",
        "next_queue",
        "placement_counts",
        "policy_targets",
        "value_targets",
        "action_masks",
    )
    for key in per_example_keys:
        if int(data[key].shape[0]) != n:
            raise ValueError(
                f"{key} has inconsistent first dimension: {data[key].shape[0]} vs {n}"
            )
    return n


def build_aux_batch(data: np.lib.npyio.NpzFile, indices: np.ndarray) -> np.ndarray:
    current_pieces = data["current_pieces"][indices].astype(np.float32, copy=False)
    hold_pieces = data["hold_pieces"][indices].astype(np.float32, copy=False)
    hold_available = data["hold_available"][indices].astype(np.float32).reshape(-1, 1)
    next_queue = data["next_queue"][indices].astype(np.float32).reshape(len(indices), -1)
    placement_counts = data["placement_counts"][indices].astype(np.float32).reshape(-1, 1)
    return np.concatenate(
        [current_pieces, hold_pieces, hold_available, next_queue, placement_counts],
        axis=1,
    )


def build_torch_batch(
    data: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    boards_np = data["boards"][indices].astype(np.float32)
    aux_np = build_aux_batch(data, indices)
    policy_targets_np = data["policy_targets"][indices].astype(np.float32, copy=False)
    value_targets_np = data["value_targets"][indices].astype(np.float32, copy=False)
    action_masks_np = data["action_masks"][indices].astype(np.float32, copy=False)

    boards = torch.from_numpy(boards_np).unsqueeze(1).to(device)
    aux = torch.from_numpy(aux_np).to(device)
    policy_targets = torch.from_numpy(policy_targets_np).to(device)
    value_targets = torch.from_numpy(value_targets_np).to(device)
    action_masks = torch.from_numpy(action_masks_np).to(device)
    return boards, aux, policy_targets, value_targets, action_masks


def select_subset(indices: np.ndarray, max_examples: int, seed: int) -> np.ndarray:
    if max_examples <= 0 or max_examples >= len(indices):
        return indices
    rng = np.random.default_rng(seed)
    return rng.choice(indices, size=max_examples, replace=False)


def evaluate_losses(
    model: nn.Module,
    data: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    device: torch.device,
    eval_batch_size: int,
    value_loss_weight: float,
) -> dict[str, float]:
    total_sum = 0.0
    policy_sum = 0.0
    value_sum = 0.0
    sample_count = 0

    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), eval_batch_size):
            batch_indices = indices[start : start + eval_batch_size]
            boards, aux, policy_targets, value_targets, action_masks = build_torch_batch(
                data, batch_indices, device
            )
            total_loss, policy_loss, value_loss = compute_loss(
                model=model,
                boards=boards,
                aux_features=aux,
                policy_targets=policy_targets,
                value_targets=value_targets,
                action_masks=action_masks,
                value_loss_weight=value_loss_weight,
            )

            n = len(batch_indices)
            total_sum += total_loss.item() * n
            policy_sum += policy_loss.item() * n
            value_sum += value_loss.item() * n
            sample_count += n

    if sample_count == 0:
        raise ValueError("Cannot evaluate with empty indices")

    return {
        "total_loss": total_sum / sample_count,
        "policy_loss": policy_sum / sample_count,
        "value_loss": value_sum / sample_count,
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_offline_model(
    model_name: str,
    wandb_prefix: str,
    model: nn.Module,
    data: np.lib.npyio.NpzFile,
    train_indices: np.ndarray,
    train_eval_indices: np.ndarray,
    val_eval_indices: np.ndarray,
    args: ScriptArgs,
    device: torch.device,
    schedule_seed: int,
) -> dict:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, float | int]] = []
    rng = np.random.default_rng(schedule_seed)
    start_time = time.perf_counter()

    def record_eval(step: int) -> None:
        train_metrics = evaluate_losses(
            model=model,
            data=data,
            indices=train_eval_indices,
            device=device,
            eval_batch_size=args.eval_batch_size,
            value_loss_weight=args.value_loss_weight,
        )
        val_metrics = evaluate_losses(
            model=model,
            data=data,
            indices=val_eval_indices,
            device=device,
            eval_batch_size=args.eval_batch_size,
            value_loss_weight=args.value_loss_weight,
        )
        elapsed_sec = time.perf_counter() - start_time
        row = {
            "step": step,
            "elapsed_sec": elapsed_sec,
            "train_total_loss": train_metrics["total_loss"],
            "train_policy_loss": train_metrics["policy_loss"],
            "train_value_loss": train_metrics["value_loss"],
            "val_total_loss": val_metrics["total_loss"],
            "val_policy_loss": val_metrics["policy_loss"],
            "val_value_loss": val_metrics["value_loss"],
        }
        history.append(row)
        wandb.log(
            {
                "offline_step": step,
                f"{wandb_prefix}/eval_train_total_loss": row["train_total_loss"],
                f"{wandb_prefix}/eval_train_policy_loss": row["train_policy_loss"],
                f"{wandb_prefix}/eval_train_value_loss": row["train_value_loss"],
                f"{wandb_prefix}/eval_val_total_loss": row["val_total_loss"],
                f"{wandb_prefix}/eval_val_policy_loss": row["val_policy_loss"],
                f"{wandb_prefix}/eval_val_value_loss": row["val_value_loss"],
                f"{wandb_prefix}/elapsed_sec": elapsed_sec,
            }
        )
        logger.info(
            "Offline eval",
            model=model_name,
            step=step,
            train_total_loss=row["train_total_loss"],
            val_total_loss=row["val_total_loss"],
            train_policy_loss=row["train_policy_loss"],
            val_policy_loss=row["val_policy_loss"],
            train_value_loss=row["train_value_loss"],
            val_value_loss=row["val_value_loss"],
        )

    record_eval(step=0)

    for step in range(1, args.steps + 1):
        positions = rng.integers(0, len(train_indices), size=args.batch_size)
        batch_indices = train_indices[positions]
        boards, aux, policy_targets, value_targets, action_masks = build_torch_batch(
            data, batch_indices, device
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
            value_loss_weight=args.value_loss_weight,
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        wandb.log(
            {
                "offline_step": step,
                f"{wandb_prefix}/train_batch_total_loss": total_loss.item(),
                f"{wandb_prefix}/train_batch_policy_loss": policy_loss.item(),
                f"{wandb_prefix}/train_batch_value_loss": value_loss.item(),
            }
        )

        if step % args.eval_interval == 0 or step == args.steps:
            record_eval(step=step)

    elapsed_sec = time.perf_counter() - start_time
    final_metrics = history[-1]
    return {
        "model_name": model_name,
        "num_parameters": count_parameters(model),
        "elapsed_sec": elapsed_sec,
        "final": final_metrics,
        "history": history,
    }


def normalize_args_for_wandb(args: ScriptArgs) -> dict:
    normalized = asdict(args)
    normalized["data_path"] = str(args.data_path)
    return normalized


def main(args: ScriptArgs) -> None:
    validate_args(args)
    if not args.data_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {args.data_path}")
    if args.data_path.suffix != ".npz":
        raise ValueError(f"Expected .npz file, got: {args.data_path}")

    device_str = pick_device(args.device)
    device = torch.device(device_str)
    logger.info("Using device", device=device_str)

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=args.wandb_tags,
        config=normalize_args_for_wandb(args),
    )
    wandb.define_metric("offline_step")
    wandb.define_metric("baseline/*", step_metric="offline_step")
    wandb.define_metric("gated/*", step_metric="offline_step")

    npz = np.load(args.data_path, mmap_mode="r")
    try:
        ensure_required_keys(npz)
        total_examples = validate_shapes(npz)
        all_indices = np.arange(total_examples, dtype=np.int64)

        split_rng = np.random.default_rng(args.seed)
        split_rng.shuffle(all_indices)
        if args.max_examples > 0:
            all_indices = all_indices[: args.max_examples]

        split_point = int(len(all_indices) * args.train_fraction)
        if split_point <= 0 or split_point >= len(all_indices):
            raise ValueError(
                "Invalid train/val split; adjust max_examples or train_fraction"
            )
        train_indices = all_indices[:split_point]
        val_indices = all_indices[split_point:]

        train_eval_indices = select_subset(
            train_indices,
            max_examples=args.eval_examples,
            seed=args.seed + 1,
        )
        val_eval_indices = select_subset(
            val_indices,
            max_examples=args.eval_examples,
            seed=args.seed + 2,
        )

        logger.info(
            "Dataset split",
            total_examples=total_examples,
            used_examples=len(all_indices),
            train_examples=len(train_indices),
            val_examples=len(val_indices),
            train_eval_examples=len(train_eval_indices),
            val_eval_examples=len(val_eval_indices),
        )
        wandb.log(
            {
                "dataset/total_examples": total_examples,
                "dataset/used_examples": len(all_indices),
                "dataset/train_examples": len(train_indices),
                "dataset/val_examples": len(val_indices),
                "dataset/train_eval_examples": len(train_eval_indices),
                "dataset/val_eval_examples": len(val_eval_indices),
            }
        )

        torch.manual_seed(args.seed)
        baseline_model = TetrisNet(
            conv_filters=args.conv_filters,
            fc_hidden=args.fc_hidden,
            conv_kernel_size=args.conv_kernel_size,
            conv_padding=args.conv_padding,
        ).to(device)
        baseline_params = count_parameters(baseline_model)
        baseline_flops = baseline_flop_breakdown(
            args=args,
            cache_hit_rate=args.cache_hit_rate_for_matching,
        )

        matched = find_matched_gated_config(
            args=args,
            target_params=baseline_params,
            target_effective_flops=baseline_flops.effective,
        )

        torch.manual_seed(args.seed)
        gated_model = GatedFusionTetrisNet(
            conv_filters=args.conv_filters,
            conv_kernel_size=args.conv_kernel_size,
            conv_padding=args.conv_padding,
            aux_hidden=matched.aux_hidden,
            fusion_hidden=matched.fusion_hidden,
            num_fusion_blocks=matched.num_fusion_blocks,
        ).to(device)
        gated_params_actual = count_parameters(gated_model)

        if gated_params_actual != matched.params:
            raise ValueError(
                "Analytical gated parameter count does not match instantiated model: "
                f"formula={matched.params}, actual={gated_params_actual}"
            )

        logger.info(
            "Matched architectures",
            baseline_params=baseline_params,
            baseline_miss_only_flops=baseline_flops.miss_only,
            baseline_hit_path_flops=baseline_flops.hit_path,
            baseline_full_flops=baseline_flops.full,
            baseline_effective_flops=baseline_flops.effective,
            gated_params=matched.params,
            gated_miss_only_flops=matched.miss_only_flops,
            gated_hit_path_flops=matched.hit_path_flops,
            gated_full_flops=matched.full_flops,
            gated_effective_flops=matched.effective_flops,
            gated_aux_hidden=matched.aux_hidden,
            gated_fusion_hidden=matched.fusion_hidden,
            gated_num_fusion_blocks=matched.num_fusion_blocks,
            cache_hit_rate_for_matching=args.cache_hit_rate_for_matching,
            param_rel_error=matched.param_rel_error,
            cache_weighted_flop_rel_error=matched.flop_rel_error,
        )
        wandb.log(
            {
                "arch/baseline_params": baseline_params,
                "arch/baseline_miss_only_flops": baseline_flops.miss_only,
                "arch/baseline_hit_path_flops": baseline_flops.hit_path,
                "arch/baseline_full_flops": baseline_flops.full,
                "arch/baseline_effective_flops": baseline_flops.effective,
                "arch/gated_params": matched.params,
                "arch/gated_miss_only_flops": matched.miss_only_flops,
                "arch/gated_hit_path_flops": matched.hit_path_flops,
                "arch/gated_full_flops": matched.full_flops,
                "arch/gated_effective_flops": matched.effective_flops,
                "arch/gated_aux_hidden": matched.aux_hidden,
                "arch/gated_fusion_hidden": matched.fusion_hidden,
                "arch/gated_num_fusion_blocks": matched.num_fusion_blocks,
                "arch/cache_hit_rate_for_matching": args.cache_hit_rate_for_matching,
                "arch/param_rel_error": matched.param_rel_error,
                "arch/cache_weighted_flop_rel_error": matched.flop_rel_error,
            }
        )

        baseline_result = train_offline_model(
            model_name="baseline_concat_fc",
            wandb_prefix="baseline",
            model=baseline_model,
            data=npz,
            train_indices=train_indices,
            train_eval_indices=train_eval_indices,
            val_eval_indices=val_eval_indices,
            args=args,
            device=device,
            schedule_seed=args.seed + 12345,
        )

        gated_result = train_offline_model(
            model_name="gated_fusion",
            wandb_prefix="gated",
            model=gated_model,
            data=npz,
            train_indices=train_indices,
            train_eval_indices=train_eval_indices,
            val_eval_indices=val_eval_indices,
            args=args,
            device=device,
            schedule_seed=args.seed + 12345,
        )

        winner = min(
            [baseline_result, gated_result],
            key=lambda x: x["final"]["val_total_loss"],
        )["model_name"]

        wandb.log(
            {
                "comparison/baseline_final_val_total_loss": baseline_result["final"][
                    "val_total_loss"
                ],
                "comparison/gated_final_val_total_loss": gated_result["final"][
                    "val_total_loss"
                ],
                "comparison/baseline_final_val_policy_loss": baseline_result["final"][
                    "val_policy_loss"
                ],
                "comparison/gated_final_val_policy_loss": gated_result["final"][
                    "val_policy_loss"
                ],
                "comparison/baseline_final_val_value_loss": baseline_result["final"][
                    "val_value_loss"
                ],
                "comparison/gated_final_val_value_loss": gated_result["final"][
                    "val_value_loss"
                ],
                "comparison/winner_is_gated": 1 if winner == "gated_fusion" else 0,
            }
        )

        wandb.run.summary["winner_by_final_val_total_loss"] = winner
        wandb.run.summary["baseline_final_val_total_loss"] = baseline_result["final"][
            "val_total_loss"
        ]
        wandb.run.summary["gated_final_val_total_loss"] = gated_result["final"][
            "val_total_loss"
        ]
        wandb.run.summary["matched_gated_aux_hidden"] = matched.aux_hidden
        wandb.run.summary["matched_gated_fusion_hidden"] = matched.fusion_hidden
        wandb.run.summary["matched_gated_num_fusion_blocks"] = (
            matched.num_fusion_blocks
        )
        wandb.run.summary["matched_param_rel_error"] = matched.param_rel_error
        wandb.run.summary["matched_cache_weighted_flop_rel_error"] = (
            matched.flop_rel_error
        )
        wandb.run.summary["cache_hit_rate_for_matching"] = (
            args.cache_hit_rate_for_matching
        )
        wandb.run.summary["baseline_effective_flops"] = baseline_flops.effective
        wandb.run.summary["gated_effective_flops"] = matched.effective_flops

        logger.info(
            "Offline architecture comparison complete",
            winner=winner,
            baseline_final_val_loss=baseline_result["final"]["val_total_loss"],
            gated_final_val_loss=gated_result["final"]["val_total_loss"],
        )
    finally:
        npz.close()
        wandb.finish()


if __name__ == "__main__":
    main(parse(ScriptArgs))
