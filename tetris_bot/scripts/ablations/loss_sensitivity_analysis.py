"""Loss sensitivity analysis via runtime output noise.

Sweeps Gaussian noise over either:
- policy logits, before invalid-action masking and softmax
- value predictions, after the value head output

For each noise setting, the script:
1. Measures held-out policy/value loss on replay-buffer examples
2. Runs fixed-seed Rust evaluation with the same output-noise parameters
3. Fits sigmoid curves of held-out loss vs average attack
4. Writes JSON summaries plus one plot for policy and one for value

Usage:
    python tetris_bot/scripts/ablations/loss_sensitivity_analysis.py training_runs/v32
"""

from __future__ import annotations

from dataclasses import asdict, field
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
import numpy as np
from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from simple_parsing import field as sp_field
from simple_parsing import parse
import structlog
import torch
import torch.nn.functional as F

from tetris_bot.constants import (
    BENCHMARKS_DIR,
    CHECKPOINT_DIRNAME,
    CONFIG_FILENAME,
    INCUMBENT_ONNX_FILENAME,
    LATEST_CHECKPOINT_FILENAME,
    TRAINING_DATA_FILENAME,
)
from tetris_bot.ml.config import TrainingConfig, load_training_config_json
from tetris_bot.ml.loss import apply_action_mask
from tetris_bot.ml.network import TetrisNet
from tetris_bot.ml.weights import (
    export_onnx,
    export_split_models,
    load_checkpoint,
    split_model_paths,
)
from tetris_bot.scripts.ablations.compare_offline_architectures import (
    OfflineDataSource,
    build_tensor_dataset,
    build_torch_batch,
    ensure_required_keys,
    select_subset,
    validate_shapes,
)
from tetris_bot.scripts.utils.eval_utils import compute_attack_stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tetris_core.tetris_core import MCTSConfig, evaluate_model
except ImportError:
    evaluate_model = None  # type: ignore[assignment]
    MCTSConfig = None  # type: ignore[assignment,misc]

logger = structlog.get_logger(__name__)


class SweepPoint(BaseModel):
    noise_type: str
    noise_std: float
    repeat_index: int
    noise_seed: int | None
    policy_loss: float
    value_loss: float
    avg_attack: float
    avg_lines: float
    avg_moves: float
    max_attack: int
    total_attack: int
    num_games: int
    attack_std: float
    attack_sem: float
    elapsed_sec: float


class AggregatedPoint(BaseModel):
    noise_std: float
    num_repeats: int
    loss_mean: float
    loss_std: float
    loss_sem: float
    policy_loss_mean: float
    value_loss_mean: float
    avg_attack_mean: float
    avg_attack_std: float
    avg_attack_sem: float


class SigmoidFit(BaseModel):
    L: float
    k: float
    x0: float
    b: float


class SweepAnalysis(BaseModel):
    noise_type: str
    loss_metric: str
    baseline_loss: float
    baseline_avg_attack: float
    aggregated: list[AggregatedPoint]
    sigmoid_fit: SigmoidFit | None
    derivative_at_baseline: float | None
    plot_path: str | None


class RecommendedWeighting(BaseModel):
    value_loss_weight: float
    policy_derivative: float
    value_derivative: float
    interpretation: str


class AnalysisSummary(BaseModel):
    run_dir: str
    generated_at: str
    args: dict[str, object]
    analysis_model_path: str
    analysis_model_source: str
    baseline: SweepPoint
    policy: SweepAnalysis
    value: SweepAnalysis
    recommended_weighting: RecommendedWeighting | None


@dataclass
class ScriptArgs:
    run_dir: Path = sp_field(
        positional=True
    )  # Training run directory with checkpoints/, config.json, training_data.npz
    policy_noise_stds: list[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    )
    value_noise_stds: list[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    )
    num_noise_repeats: int = 3  # Random draws per non-zero noise level
    max_examples: int = 0  # 0 uses the full replay buffer before the train/eval split
    train_fraction: float = (
        0.9  # Fraction of shuffled examples reserved for the train split
    )
    eval_examples: int = (
        32_768  # Max held-out examples used for each loss evaluation pass
    )
    eval_batch_size: int = 2048
    preload_to_gpu: bool = True  # Preload selected replay data to GPU if available
    preload_to_ram: bool = False  # Preload selected replay data to CPU RAM
    num_eval_games: int = 20
    eval_num_simulations: int = 2000
    eval_max_placements: int = 50
    eval_num_workers: int = 7
    eval_seed_start: int = 0
    eval_mcts_seed: int | None = 0  # Fixed MCTS seed so only prediction noise varies
    device: str = "auto"
    seed: int = 42
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        self.policy_noise_stds = normalize_noise_stds(
            self.policy_noise_stds,
            label="policy_noise_stds",
        )
        self.value_noise_stds = normalize_noise_stds(
            self.value_noise_stds,
            label="value_noise_stds",
        )
        if self.num_noise_repeats <= 0:
            raise ValueError(
                f"num_noise_repeats must be > 0, got {self.num_noise_repeats}"
            )
        if self.max_examples < 0:
            raise ValueError(f"max_examples must be >= 0, got {self.max_examples}")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError(
                f"train_fraction must be in (0, 1), got {self.train_fraction}"
            )
        if self.eval_examples <= 0:
            raise ValueError(f"eval_examples must be > 0, got {self.eval_examples}")
        if self.eval_batch_size <= 0:
            raise ValueError(f"eval_batch_size must be > 0, got {self.eval_batch_size}")
        if self.num_eval_games <= 0:
            raise ValueError(f"num_eval_games must be > 0, got {self.num_eval_games}")
        if self.eval_num_simulations <= 0:
            raise ValueError(
                f"eval_num_simulations must be > 0, got {self.eval_num_simulations}"
            )
        if self.eval_max_placements <= 0:
            raise ValueError(
                f"eval_max_placements must be > 0, got {self.eval_max_placements}"
            )
        if self.eval_num_workers <= 1:
            raise ValueError(
                f"eval_num_workers must be > 1 for normal parallel evaluation, got {self.eval_num_workers}"
            )
        if self.eval_num_workers > self.num_eval_games:
            raise ValueError(
                "eval_num_workers must be <= num_eval_games so each worker has work "
                f"(got workers={self.eval_num_workers}, games={self.num_eval_games})"
            )


def normalize_noise_stds(values: list[float], label: str) -> list[float]:
    if not values:
        raise ValueError(f"{label} must not be empty")
    normalized: set[float] = {0.0}
    for value in values:
        if not np.isfinite(value):
            raise ValueError(f"{label} must contain finite values, got {value}")
        if value < 0:
            raise ValueError(f"{label} must contain only values >= 0, got {value}")
        normalized.add(float(value))
    return sorted(normalized)


def pick_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_preload_mode(args: ScriptArgs, device: torch.device) -> str:
    if args.preload_to_gpu and device.type != "cpu":
        return "gpu"
    if args.preload_to_ram:
        return "ram"
    return "none"


class OnnxModel:
    """Wraps an ONNX model via onnxruntime to match TetrisNet's call signature."""

    def __init__(self, onnx_path: Path, device: torch.device) -> None:
        import onnxruntime as ort  # pyright: ignore[reportMissingImports]

        providers: list[str] = []
        if device.type == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.device = device
        self.onnx_path = onnx_path

    def eval(self) -> None:
        pass

    def __call__(
        self, board: torch.Tensor, aux_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        board_np = board.cpu().numpy()
        aux_np = aux_features.cpu().numpy()
        policy_logits_np, value_np = self.session.run(
            ["policy_logits", "value"],
            {"board": board_np, "aux_features": aux_np},
        )
        policy_logits = torch.from_numpy(policy_logits_np).to(self.device)
        value = torch.from_numpy(value_np).to(self.device)
        return policy_logits, value


AnalysisModel = TetrisNet | OnnxModel


def load_analysis_model(
    run_dir: Path,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[AnalysisModel, str]:
    checkpoint_path = run_dir / CHECKPOINT_DIRNAME / LATEST_CHECKPOINT_FILENAME
    if checkpoint_path.exists():
        raw_model = TetrisNet(**config.network.to_model_kwargs()).to(device)
        ema_model = TetrisNet(**config.network.to_model_kwargs()).to(device)
        state = load_checkpoint(checkpoint_path, model=raw_model, ema_model=ema_model)
        model_source = "ema" if state.get("ema_state_dict") is not None else "raw"
        analysis_model: AnalysisModel = (
            ema_model if model_source == "ema" else raw_model
        )
        analysis_model.eval()
        return analysis_model, model_source

    incumbent_path = run_dir / CHECKPOINT_DIRNAME / INCUMBENT_ONNX_FILENAME
    if incumbent_path.exists():
        logger.info(
            "No latest.pt found; loading incumbent ONNX model",
            incumbent_path=str(incumbent_path),
        )
        return OnnxModel(incumbent_path, device), "incumbent_onnx"

    raise FileNotFoundError(
        f"Neither {checkpoint_path} nor {incumbent_path} found in run dir"
    )


def export_analysis_model(
    model: AnalysisModel, output_dir: Path, run_dir: Path
) -> Path:
    model_path = output_dir / "analysis_model.onnx"
    if isinstance(model, OnnxModel):
        import shutil

        src_onnx = model.onnx_path
        shutil.copy2(src_onnx, model_path)
        src_splits = split_model_paths(src_onnx)
        dst_splits = split_model_paths(model_path)
        for src_split, dst_split in zip(src_splits, dst_splits):
            if src_split.exists():
                shutil.copy2(src_split, dst_split)
        return model_path

    if not export_onnx(model, model_path):
        raise RuntimeError("ONNX export failed")
    if not export_split_models(model, model_path):
        raise RuntimeError("Split-model export failed")
    return model_path


def sample_noise_tensor(
    *,
    shape: torch.Size,
    mean: float,
    std: float,
    generator: torch.Generator | None,
    device: torch.device,
) -> torch.Tensor | None:
    if std == 0.0 and mean == 0.0:
        return None
    if std == 0.0:
        return torch.full(shape, mean, dtype=torch.float32, device=device)
    cpu_noise = torch.randn(shape, generator=generator, dtype=torch.float32)
    noise = cpu_noise.mul(std).add(mean)
    return noise.to(device=device)


def evaluate_losses_with_output_noise(
    model: AnalysisModel,
    source: OfflineDataSource,
    eval_indices: np.ndarray,
    device: torch.device,
    eval_batch_size: int,
    *,
    noise_type: str | None,
    noise_std: float,
    noise_seed: int | None,
) -> tuple[float, float]:
    policy_sum = 0.0
    value_sum = 0.0
    count = 0
    noise_generator = None
    if noise_seed is not None:
        noise_generator = torch.Generator(device="cpu")
        noise_generator.manual_seed(noise_seed)

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(eval_indices), eval_batch_size):
            batch_idx = eval_indices[start : start + eval_batch_size]
            boards, aux, policy_targets, value_targets, action_masks = (
                build_torch_batch(
                    source,
                    batch_idx,
                    device,
                )
            )
            policy_logits, value_pred = model(boards.float(), aux)
            if noise_type == "policy":
                policy_noise = sample_noise_tensor(
                    shape=policy_logits.shape,
                    mean=0.0,
                    std=noise_std,
                    generator=noise_generator,
                    device=policy_logits.device,
                )
                if policy_noise is not None:
                    policy_logits = policy_logits + policy_noise
            elif noise_type == "value":
                value_noise = sample_noise_tensor(
                    shape=value_pred.shape,
                    mean=0.0,
                    std=noise_std,
                    generator=noise_generator,
                    device=value_pred.device,
                )
                if value_noise is not None:
                    value_pred = value_pred + value_noise

            masked_logits = apply_action_mask(policy_logits, action_masks)
            log_policy = F.log_softmax(masked_logits, dim=-1)
            log_policy = torch.where(
                torch.isinf(log_policy),
                torch.zeros_like(log_policy),
                log_policy,
            )
            policy_loss = -torch.sum(policy_targets * log_policy, dim=1).mean()
            value_loss = F.mse_loss(value_pred.squeeze(-1), value_targets)

            batch_size = len(batch_idx)
            policy_sum += policy_loss.item() * batch_size
            value_sum += value_loss.item() * batch_size
            count += batch_size

    if count == 0:
        raise ValueError("Held-out evaluation set is empty")
    return policy_sum / count, value_sum / count


def run_games(
    model_path: Path,
    args: ScriptArgs,
    *,
    noise_type: str | None,
    noise_std: float,
) -> dict[str, float | int]:
    if evaluate_model is None or MCTSConfig is None:
        raise ImportError("tetris_core not available; rebuild with make build-dev")

    config = MCTSConfig()
    config.num_simulations = args.eval_num_simulations
    config.max_placements = args.eval_max_placements
    config.visit_sampling_epsilon = 0.0
    config.nn_value_weight = 1.0
    config.death_penalty = 0.0
    config.overhang_penalty_weight = 0.0
    config.reuse_tree = True
    config.seed = args.eval_mcts_seed
    config.prediction_noise_seed = None

    if noise_type == "policy":
        config.policy_noise_mean = 0.0
        config.policy_noise_std = noise_std
    elif noise_type == "value":
        config.value_noise_mean = 0.0
        config.value_noise_std = noise_std

    seeds = list(
        range(args.eval_seed_start, args.eval_seed_start + args.num_eval_games)
    )
    result = evaluate_model(
        model_path=str(model_path),
        seeds=seeds,
        config=config,
        max_placements=args.eval_max_placements,
        num_workers=args.eval_num_workers,
        add_noise=False,
    )
    attack_std, attack_sem = compute_attack_stats(seeds, result)
    game_results = [(int(attack), int(moves)) for attack, moves in result.game_results]
    return {
        "avg_attack": float(result.avg_attack),
        "avg_lines": float(result.avg_lines),
        "avg_moves": float(result.avg_moves),
        "max_attack": int(result.max_attack),
        "total_attack": int(sum(attack for attack, _ in game_results)),
        "num_games": int(result.num_games),
        "attack_std": attack_std,
        "attack_sem": attack_sem,
    }


def sigmoid(x: np.ndarray, L: float, k: float, x0: float, b: float) -> np.ndarray:
    return L / (1.0 + np.exp(-k * (x - x0))) + b


def sigmoid_derivative(x: float, L: float, k: float, x0: float) -> float:
    exponent = np.exp(-k * (x - x0))
    return float(L * k * exponent / (1.0 + exponent) ** 2)


def fit_sigmoid(losses: np.ndarray, attacks: np.ndarray) -> SigmoidFit | None:
    try:
        from scipy.optimize import curve_fit  # pyright: ignore[reportMissingImports]
    except ImportError:
        logger.warning("scipy not available; skipping sigmoid fit")
        return None

    if len(losses) < 4:
        logger.warning(
            "Not enough aggregated points for sigmoid fit", num_points=len(losses)
        )
        return None

    sort_order = np.argsort(losses)
    losses = losses[sort_order]
    attacks = attacks[sort_order]
    attack_min = float(attacks.min())
    attack_max = float(attacks.max())
    attack_range = attack_max - attack_min
    initial_L = attack_range if attack_range > 0 else 1.0
    initial_x0 = float(np.median(losses))
    initial_b = attack_min

    try:
        params, _ = curve_fit(
            sigmoid,
            losses,
            attacks,
            p0=[initial_L, -1.0, initial_x0, initial_b],
            maxfev=20_000,
        )
    except Exception as exc:  # curve_fit throws heterogeneous exception types
        logger.warning("Sigmoid fit failed", error=str(exc))
        return None

    return SigmoidFit(
        L=float(params[0]), k=float(params[1]), x0=float(params[2]), b=float(params[3])
    )


def noise_seed_for_point(
    *,
    base_seed: int,
    noise_type: str,
    noise_index: int,
    repeat_index: int,
    noise_std: float,
) -> int | None:
    """Seed for Python-side held-out loss noise (deterministic per point).

    Rust eval uses thread_rng (prediction_noise_seed=None) so each MCTS
    inference call gets independent noise draws.
    """
    if noise_std == 0.0:
        return None
    type_offset = 0 if noise_type == "policy" else 1_000_000
    return base_seed + type_offset + (noise_index * 10_000) + repeat_index


def mean_std_sem(values: list[float]) -> tuple[float, float, float]:
    if not values:
        raise ValueError("Cannot summarize an empty list")
    if len(values) == 1:
        return float(values[0]), 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    std = float(np.std(array, ddof=0))
    sem = float(std / np.sqrt(len(array)))
    return float(array.mean()), std, sem


def aggregate_points(
    points: list[SweepPoint],
    *,
    noise_type: str,
    baseline_loss: float,
    baseline_avg_attack: float,
) -> SweepAnalysis:
    loss_metric = "policy_loss" if noise_type == "policy" else "value_loss"
    rows_by_std: dict[float, list[SweepPoint]] = {}
    for point in points:
        if point.noise_type == noise_type:
            rows_by_std.setdefault(point.noise_std, []).append(point)

    aggregated_rows: list[AggregatedPoint] = []
    for noise_std in sorted(rows_by_std):
        row_points = rows_by_std[noise_std]
        losses = [
            point.policy_loss if loss_metric == "policy_loss" else point.value_loss
            for point in row_points
        ]
        attacks = [point.avg_attack for point in row_points]
        loss_mean, loss_std, loss_sem = mean_std_sem(losses)
        attack_mean, attack_std, attack_sem = mean_std_sem(attacks)
        aggregated_rows.append(
            AggregatedPoint(
                noise_std=noise_std,
                num_repeats=len(row_points),
                loss_mean=loss_mean,
                loss_std=loss_std,
                loss_sem=loss_sem,
                policy_loss_mean=float(
                    np.mean([point.policy_loss for point in row_points])
                ),
                value_loss_mean=float(
                    np.mean([point.value_loss for point in row_points])
                ),
                avg_attack_mean=attack_mean,
                avg_attack_std=attack_std,
                avg_attack_sem=attack_sem,
            )
        )

    loss_array = np.asarray(
        [row.loss_mean for row in aggregated_rows], dtype=np.float64
    )
    attack_array = np.asarray(
        [row.avg_attack_mean for row in aggregated_rows], dtype=np.float64
    )
    sigmoid_fit = fit_sigmoid(loss_array, attack_array)
    derivative = None
    if sigmoid_fit is not None:
        derivative = sigmoid_derivative(
            baseline_loss,
            sigmoid_fit.L,
            sigmoid_fit.k,
            sigmoid_fit.x0,
        )

    return SweepAnalysis(
        noise_type=noise_type,
        loss_metric=loss_metric,
        baseline_loss=baseline_loss,
        baseline_avg_attack=baseline_avg_attack,
        aggregated=aggregated_rows,
        sigmoid_fit=sigmoid_fit,
        derivative_at_baseline=derivative,
        plot_path=None,
    )


def plot_analysis(
    analysis: SweepAnalysis,
    output_path: Path,
    *,
    run_dir: Path,
    eval_examples: int,
    num_games: int,
    num_simulations: int,
) -> None:
    if not analysis.aggregated:
        raise ValueError(
            f"No aggregated points available for {analysis.noise_type} plot"
        )

    x_values = np.asarray(
        [row.loss_mean for row in analysis.aggregated], dtype=np.float64
    )
    y_values = np.asarray(
        [row.avg_attack_mean for row in analysis.aggregated], dtype=np.float64
    )
    x_err = np.asarray([row.loss_sem for row in analysis.aggregated], dtype=np.float64)
    y_err = np.asarray(
        [row.avg_attack_sem for row in analysis.aggregated], dtype=np.float64
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.errorbar(
        x_values,
        y_values,
        xerr=x_err,
        yerr=y_err,
        fmt="o",
        color="#1f77b4",
        ecolor="#4c78a8",
        capsize=4,
        linewidth=1.2,
    )

    for row in analysis.aggregated:
        ax.annotate(
            f"{row.avg_attack_mean:.2f}\nσ={row.noise_std:g}, n={row.num_repeats}",
            (row.loss_mean, row.avg_attack_mean),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7,
        )

    if analysis.sigmoid_fit is not None:
        fit_x = np.linspace(float(x_values.min()), float(x_values.max()), 300)
        fit_y = sigmoid(
            fit_x,
            analysis.sigmoid_fit.L,
            analysis.sigmoid_fit.k,
            analysis.sigmoid_fit.x0,
            analysis.sigmoid_fit.b,
        )
        ax.plot(fit_x, fit_y, color="#d62728", linewidth=1.5, label="Sigmoid fit")
        ax.legend(loc="best")

    x_padding = max(0.02, 0.05 * max(1e-6, float(x_values.max() - x_values.min())))
    y_min = min(float(y_values.min()), analysis.baseline_avg_attack)
    y_max = max(float(y_values.max()), analysis.baseline_avg_attack)
    y_padding = max(0.2, 0.1 * max(1e-6, y_max - y_min))

    ax.scatter(
        [analysis.baseline_loss],
        [analysis.baseline_avg_attack],
        color="#2ca02c",
        marker="D",
        zorder=4,
    )
    ax.annotate(
        f"{analysis.baseline_avg_attack:.2f}\nbaseline",
        (analysis.baseline_loss, analysis.baseline_avg_attack),
        xytext=(6, -20),
        textcoords="offset points",
        fontsize=7,
    )
    ax.set_xlim(float(x_values.min()) - x_padding, float(x_values.max()) + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    ax.grid(axis="y", alpha=0.3)

    x_label = (
        "Held-out policy cross-entropy"
        if analysis.loss_metric == "policy_loss"
        else "Held-out value MSE"
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Average attack over fixed-seed eval")

    title = (
        "Policy Loss vs Average Attack"
        if analysis.noise_type == "policy"
        else "Value Loss vs Average Attack"
    )
    subtitle = (
        f"Run: {run_dir.name} | Held-out examples: {eval_examples:,} | "
        f"Eval games: {num_games} | Sims: {num_simulations}"
    )
    fig.suptitle(title, fontsize=12)
    ax.set_title(subtitle, fontsize=8, color="gray", pad=4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_point(
    *,
    noise_type: str,
    noise_std: float,
    repeat_index: int,
    noise_seed: int | None,
    policy_loss: float,
    value_loss: float,
    game_metrics: dict[str, float | int],
    elapsed_sec: float,
) -> SweepPoint:
    return SweepPoint(
        noise_type=noise_type,
        noise_std=noise_std,
        repeat_index=repeat_index,
        noise_seed=noise_seed,
        policy_loss=policy_loss,
        value_loss=value_loss,
        avg_attack=float(game_metrics["avg_attack"]),
        avg_lines=float(game_metrics["avg_lines"]),
        avg_moves=float(game_metrics["avg_moves"]),
        max_attack=int(game_metrics["max_attack"]),
        total_attack=int(game_metrics["total_attack"]),
        num_games=int(game_metrics["num_games"]),
        attack_std=float(game_metrics["attack_std"]),
        attack_sem=float(game_metrics["attack_sem"]),
        elapsed_sec=elapsed_sec,
    )


def run_sweep(
    *,
    model: AnalysisModel,
    model_path: Path,
    source: OfflineDataSource,
    eval_indices: np.ndarray,
    device: torch.device,
    args: ScriptArgs,
    baseline_point: SweepPoint,
) -> list[SweepPoint]:
    all_points: list[SweepPoint] = [
        baseline_point.model_copy(update={"noise_type": "policy"}),
        baseline_point.model_copy(update={"noise_type": "value"}),
    ]
    sweep_specs = [
        ("policy", args.policy_noise_stds),
        ("value", args.value_noise_stds),
    ]

    for noise_type, noise_stds in sweep_specs:
        for noise_index, noise_std in enumerate(noise_stds):
            if noise_std == 0.0:
                continue
            for repeat_index in range(args.num_noise_repeats):
                noise_seed = noise_seed_for_point(
                    base_seed=args.seed,
                    noise_type=noise_type,
                    noise_index=noise_index,
                    repeat_index=repeat_index,
                    noise_std=noise_std,
                )
                logger.info(
                    "Running noise point",
                    noise_type=noise_type,
                    noise_std=noise_std,
                    repeat_index=repeat_index,
                    repeats=args.num_noise_repeats,
                    noise_seed=noise_seed,
                )
                start = datetime.now(timezone.utc)
                policy_loss, value_loss = evaluate_losses_with_output_noise(
                    model,
                    source,
                    eval_indices,
                    device,
                    args.eval_batch_size,
                    noise_type=noise_type,
                    noise_std=noise_std,
                    noise_seed=noise_seed,
                )
                game_metrics = run_games(
                    model_path,
                    args,
                    noise_type=noise_type,
                    noise_std=noise_std,
                )
                elapsed_sec = (datetime.now(timezone.utc) - start).total_seconds()
                point = make_point(
                    noise_type=noise_type,
                    noise_std=noise_std,
                    repeat_index=repeat_index,
                    noise_seed=noise_seed,
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    game_metrics=game_metrics,
                    elapsed_sec=elapsed_sec,
                )
                all_points.append(point)
                logger.info(
                    "Noise point complete",
                    noise_type=noise_type,
                    noise_std=noise_std,
                    repeat_index=repeat_index,
                    policy_loss=f"{policy_loss:.4f}",
                    value_loss=f"{value_loss:.4f}",
                    avg_attack=f"{point.avg_attack:.2f}",
                    elapsed_sec=f"{elapsed_sec:.1f}",
                )

    return all_points


def recommended_weighting(
    policy: SweepAnalysis,
    value: SweepAnalysis,
) -> RecommendedWeighting | None:
    if policy.derivative_at_baseline is None or value.derivative_at_baseline is None:
        return None
    if policy.derivative_at_baseline == 0.0:
        return None

    ratio = abs(value.derivative_at_baseline) / abs(policy.derivative_at_baseline)
    return RecommendedWeighting(
        value_loss_weight=ratio,
        policy_derivative=policy.derivative_at_baseline,
        value_derivative=value.derivative_at_baseline,
        interpretation=(
            f"Value loss is {ratio:.2f}x as impactful as policy loss per unit held-out loss "
            "change at the current operating point. A matching training objective would scale "
            f"value loss by about {ratio:.3f} relative to policy loss."
        ),
    )


def main(args: ScriptArgs) -> None:
    run_dir = args.run_dir.resolve()
    config_path = run_dir / CONFIG_FILENAME
    data_path = run_dir / TRAINING_DATA_FILENAME
    for path, label in [
        (config_path, "config.json"),
        (data_path, "training_data.npz"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = BENCHMARKS_DIR / "loss_sensitivity" / f"{run_dir.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_training_config_json(config_path)
    device = torch.device(pick_device(args.device))
    logger.info("Loading analysis model", run_dir=str(run_dir), device=str(device))
    model, model_source = load_analysis_model(run_dir, config, device)
    model_path = export_analysis_model(model, output_dir, run_dir)
    logger.info(
        "Exported analysis model",
        model_path=str(model_path),
        model_source=model_source,
    )

    preload_mode = get_preload_mode(args, device)
    npz = np.load(data_path, mmap_mode="r")
    try:
        ensure_required_keys(npz)
        total_examples = validate_shapes(npz)
        selected = np.arange(total_examples, dtype=np.int64)
        rng = np.random.default_rng(args.seed)
        rng.shuffle(selected)
        if args.max_examples > 0:
            selected = selected[: args.max_examples]

        split_point = int(len(selected) * args.train_fraction)
        eval_pool_indices = np.arange(split_point, len(selected), dtype=np.int64)
        eval_indices = select_subset(
            eval_pool_indices,
            max_examples=args.eval_examples,
            seed=args.seed + 17,
        )

        tensor_data = None
        if preload_mode != "none":
            tensor_data = build_tensor_dataset(
                data=npz,
                selected_global_indices=selected,
                mode=preload_mode,
                train_device=device,
            )
        source = OfflineDataSource(
            npz=npz,
            selected_global_indices=selected,
            tensor_data=tensor_data,
        )
        logger.info(
            "Prepared held-out dataset",
            total_examples=total_examples,
            selected_examples=len(selected),
            eval_examples=len(eval_indices),
            preload_mode=preload_mode,
        )

        baseline_policy_loss, baseline_value_loss = evaluate_losses_with_output_noise(
            model,
            source,
            eval_indices,
            device,
            args.eval_batch_size,
            noise_type=None,
            noise_std=0.0,
            noise_seed=None,
        )
        baseline_game_metrics = run_games(
            model_path,
            args,
            noise_type=None,
            noise_std=0.0,
        )
        baseline_point = make_point(
            noise_type="baseline",
            noise_std=0.0,
            repeat_index=0,
            noise_seed=None,
            policy_loss=baseline_policy_loss,
            value_loss=baseline_value_loss,
            game_metrics=baseline_game_metrics,
            elapsed_sec=0.0,
        )
        logger.info(
            "Baseline measured",
            policy_loss=f"{baseline_policy_loss:.4f}",
            value_loss=f"{baseline_value_loss:.4f}",
            avg_attack=f"{baseline_point.avg_attack:.2f}",
        )

        points = run_sweep(
            model=model,
            model_path=model_path,
            source=source,
            eval_indices=eval_indices,
            device=device,
            args=args,
            baseline_point=baseline_point,
        )

        policy_analysis = aggregate_points(
            points,
            noise_type="policy",
            baseline_loss=baseline_policy_loss,
            baseline_avg_attack=baseline_point.avg_attack,
        )
        value_analysis = aggregate_points(
            points,
            noise_type="value",
            baseline_loss=baseline_value_loss,
            baseline_avg_attack=baseline_point.avg_attack,
        )

        policy_plot = output_dir / "policy_loss_vs_attack.png"
        value_plot = output_dir / "value_loss_vs_attack.png"
        plot_analysis(
            policy_analysis,
            policy_plot,
            run_dir=run_dir,
            eval_examples=len(eval_indices),
            num_games=args.num_eval_games,
            num_simulations=args.eval_num_simulations,
        )
        plot_analysis(
            value_analysis,
            value_plot,
            run_dir=run_dir,
            eval_examples=len(eval_indices),
            num_games=args.num_eval_games,
            num_simulations=args.eval_num_simulations,
        )
        policy_analysis = policy_analysis.model_copy(
            update={"plot_path": str(policy_plot.relative_to(output_dir))}
        )
        value_analysis = value_analysis.model_copy(
            update={"plot_path": str(value_plot.relative_to(output_dir))}
        )

        summary = AnalysisSummary(
            run_dir=str(run_dir),
            generated_at=datetime.now(timezone.utc).isoformat(),
            args={
                **asdict(args),
                "run_dir": str(run_dir),
                "output_dir": str(output_dir),
            },
            analysis_model_path=str(model_path),
            analysis_model_source=model_source,
            baseline=baseline_point,
            policy=policy_analysis,
            value=value_analysis,
            recommended_weighting=recommended_weighting(
                policy_analysis, value_analysis
            ),
        )

        raw_points_path = output_dir / "sensitivity_points.json"
        raw_points_path.write_text(
            json.dumps([point.model_dump(mode="json") for point in points], indent=2)
            + "\n"
        )
        summary_path = output_dir / "sensitivity_summary.json"
        summary_path.write_text(
            json.dumps(summary.model_dump(mode="json"), indent=2) + "\n"
        )

        logger.info(
            "Loss sensitivity analysis complete",
            output_dir=str(output_dir),
            policy_plot=str(policy_plot),
            value_plot=str(value_plot),
        )
        if summary.recommended_weighting is not None:
            logger.info(
                "Recommended loss weighting",
                value_loss_weight=f"{summary.recommended_weighting.value_loss_weight:.3f}",
                policy_derivative=f"{summary.recommended_weighting.policy_derivative:.4f}",
                value_derivative=f"{summary.recommended_weighting.value_derivative:.4f}",
            )
        else:
            logger.warning(
                "Could not compute a recommended weighting from the fitted curves"
            )
    finally:
        npz.close()


if __name__ == "__main__":
    main(parse(ScriptArgs))
