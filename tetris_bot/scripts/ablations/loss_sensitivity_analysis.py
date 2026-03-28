"""Loss sensitivity analysis: marginal utility of policy vs value loss.

Takes a trained model and measures how game performance (avg attack) degrades
when policy or value predictions are perturbed with increasing noise.

For each noise level:
  1. Clone model, perturb policy-head weights → export → run games → record
  2. Clone model, perturb value-head weights  → export → run games → record
  3. Measure held-out policy/value loss for both perturbed variants

Fits sigmoid curves  attack = L / (1 + exp(-k*(loss - x0))) + b  to each
noise-type series, then computes d(attack)/d(loss) at the clean operating
point.  The derivative ratio gives the recommended relative weighting of
value loss vs policy loss in the training objective.

Usage:
    python -m tetris_bot.scripts.ablations.loss_sensitivity_analysis \
        --run_dir training_runs/v32
"""

from __future__ import annotations

import copy
import json
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import structlog
import torch
from simple_parsing import parse

from tetris_bot.constants import (
    BENCHMARKS_DIR,
    CHECKPOINT_DIRNAME,
    CONFIG_FILENAME,
    LATEST_CHECKPOINT_FILENAME,
    LATEST_ONNX_FILENAME,
    TRAINING_DATA_FILENAME,
)
from tetris_bot.ml.config import TrainingConfig, load_training_config_json
from tetris_bot.ml.loss import compute_loss
from tetris_bot.ml.network import TetrisNet
from tetris_bot.ml.weights import export_onnx, export_split_models, load_checkpoint
from tetris_bot.scripts.ablations.compare_offline_architectures import (
    OfflineDataSource,
    build_tensor_dataset,
    build_torch_batch,
    ensure_required_keys,
    select_subset,
    validate_shapes,
)

try:
    from tetris_core.tetris_core import MCTSConfig, evaluate_model
except ImportError:
    evaluate_model = None  # type: ignore[assignment]
    MCTSConfig = None  # type: ignore[assignment,misc]

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


@dataclass
class ScriptArgs:
    run_dir: Path  # Training run directory (has checkpoints/, config.json, training_data.npz)

    # Noise sweep parameters
    noise_fractions: list[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    )
    num_noise_repeats: int = 3  # Random noise samples per noise level

    # Offline loss evaluation
    max_examples: int = 0
    train_fraction: float = 0.9
    eval_examples: int = 32_768
    eval_batch_size: int = 2048
    preload_to_gpu: bool = True
    preload_to_ram: bool = False

    # Game evaluation
    num_eval_games: int = 20
    eval_num_simulations: int = 2000
    eval_max_placements: int = 50
    eval_num_workers: int = 7
    eval_seed_start: int = 0
    eval_mcts_seed: int | None = 0

    # General
    device: str = "auto"
    seed: int = 42
    output_dir: Path | None = (
        None  # Default: benchmarks/loss_sensitivity/<run_name>_<ts>
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pick_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_from_run(
    run_dir: Path, config: TrainingConfig, device: torch.device
) -> TetrisNet:
    checkpoint_path = run_dir / CHECKPOINT_DIRNAME / LATEST_CHECKPOINT_FILENAME
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = TetrisNet(**config.network.to_model_kwargs()).to(device)
    load_checkpoint(checkpoint_path, model=model, ema_model=None)
    model.eval()
    return model


def evaluate_val_losses(
    model: TetrisNet,
    source: OfflineDataSource,
    val_indices: np.ndarray,
    device: torch.device,
    eval_batch_size: int,
) -> dict[str, float]:
    """Compute policy and value loss on held-out validation data."""
    policy_sum = 0.0
    value_sum = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(val_indices), eval_batch_size):
            batch_idx = val_indices[start : start + eval_batch_size]
            boards, aux, policy_targets, value_targets, action_masks = (
                build_torch_batch(source, batch_idx, device)
            )
            _, policy_loss, value_loss = compute_loss(
                model=model,
                boards=boards,
                aux_features=aux,
                policy_targets=policy_targets,
                value_targets=value_targets,
                action_masks=action_masks,
                value_loss_weight=1.0,
            )
            n = len(batch_idx)
            policy_sum += policy_loss.item() * n
            value_sum += value_loss.item() * n
            count += n
    return {
        "policy_loss": policy_sum / count,
        "value_loss": value_sum / count,
    }


def weight_rms(module: torch.nn.Module) -> float:
    """Root-mean-square of all parameters in a module."""
    total = 0.0
    count = 0
    for p in module.parameters():
        total += (p.data**2).sum().item()
        count += p.numel()
    return (total / max(count, 1)) ** 0.5


def perturb_module_weights(
    module: torch.nn.Module,
    noise_fraction: float,
    rng: torch.Generator,
) -> None:
    """Add Gaussian noise to all parameters: param += N(0, noise_fraction * rms)."""
    rms = weight_rms(module)
    sigma = noise_fraction * rms
    if sigma <= 0:
        return
    for p in module.parameters():
        p.data.add_(torch.randn_like(p.data, generator=rng) * sigma)


def clone_and_perturb(
    model: TetrisNet,
    noise_type: str,
    noise_fraction: float,
    noise_seed: int,
) -> TetrisNet:
    """Clone a model and perturb either policy or value head weights."""
    perturbed = copy.deepcopy(model)
    if noise_fraction == 0.0:
        return perturbed
    rng = torch.Generator()
    rng.manual_seed(noise_seed)
    if noise_type == "policy":
        perturb_module_weights(perturbed.policy_fc, noise_fraction, rng)
        perturb_module_weights(perturbed.policy_head, noise_fraction, rng)
    elif noise_type == "value":
        perturb_module_weights(perturbed.value_fc, noise_fraction, rng)
        perturb_module_weights(perturbed.value_head, noise_fraction, rng)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
    return perturbed


def export_to_tempdir(model: TetrisNet, tmpdir: Path) -> Path:
    """Export split models to a temporary directory, return the base onnx path."""
    onnx_path = tmpdir / LATEST_ONNX_FILENAME
    if not export_onnx(model, onnx_path):
        raise RuntimeError("ONNX export failed")
    if not export_split_models(model, onnx_path):
        raise RuntimeError("Split-model export failed")
    return onnx_path


def run_games(
    model_path: Path,
    args: ScriptArgs,
) -> dict[str, float]:
    """Run deterministic fixed-seed games and return attack metrics."""
    if evaluate_model is None:
        raise ImportError("tetris_core not available; rebuild with make build-dev")
    config = MCTSConfig()
    config.num_simulations = args.eval_num_simulations
    config.max_placements = args.eval_max_placements
    config.visit_sampling_epsilon = 0.0
    config.nn_value_weight = 1.0
    config.death_penalty = 0.0
    config.overhang_penalty_weight = 0.0
    config.reuse_tree = True
    if args.eval_mcts_seed is not None:
        config.seed = args.eval_mcts_seed

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
    game_results = [(int(a), int(m)) for a, m in result.game_results]
    return {
        "avg_attack": float(result.avg_attack),
        "avg_lines": float(result.avg_lines),
        "avg_moves": float(result.avg_moves),
        "max_attack": int(result.max_attack),
        "total_attack": sum(a for a, _ in game_results),
        "num_games": int(result.num_games),
    }


# ---------------------------------------------------------------------------
# Sigmoid fitting
# ---------------------------------------------------------------------------


def sigmoid(x: np.ndarray, L: float, k: float, x0: float, b: float) -> np.ndarray:
    return L / (1.0 + np.exp(-k * (x - x0))) + b


def sigmoid_derivative(x: float, L: float, k: float, x0: float) -> float:
    """d/dx of sigmoid(x, L, k, x0, b)."""
    e = np.exp(-k * (x - x0))
    return L * k * e / (1.0 + e) ** 2


def fit_sigmoid(losses: np.ndarray, attacks: np.ndarray) -> dict[str, float] | None:
    """Fit sigmoid to (loss, attack) data. Returns params or None on failure."""
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        logger.warning("scipy not available; skipping sigmoid fit")
        return None

    if len(losses) < 4:
        logger.warning("Not enough data points for sigmoid fit", n=len(losses))
        return None

    # Initial guess: L = range of attack, x0 = median loss, k = -1 (decreasing)
    a_min, a_max = float(attacks.min()), float(attacks.max())
    L0 = a_max - a_min if a_max > a_min else 1.0
    x0_init = float(np.median(losses))
    b0 = a_min

    try:
        popt, _ = curve_fit(
            sigmoid,
            losses,
            attacks,
            p0=[L0, -1.0, x0_init, b0],
            maxfev=10000,
        )
        L, k, x0, b = popt
        return {"L": float(L), "k": float(k), "x0": float(x0), "b": float(b)}
    except RuntimeError:
        logger.warning("Sigmoid curve_fit did not converge")
        return None


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


@dataclass
class SensitivityPoint:
    noise_type: str  # "policy" or "value"
    noise_fraction: float
    noise_seed: int
    policy_loss: float
    value_loss: float
    avg_attack: float
    avg_lines: float
    avg_moves: float
    max_attack: int
    total_attack: int
    num_games: int
    elapsed_sec: float


def run_sensitivity_sweep(
    model: TetrisNet,
    source: OfflineDataSource,
    val_indices: np.ndarray,
    device: torch.device,
    args: ScriptArgs,
) -> list[SensitivityPoint]:
    results: list[SensitivityPoint] = []
    noise_types = ["policy", "value"]
    total_runs = len(noise_types) * len(args.noise_fractions) * args.num_noise_repeats
    run_index = 0

    for noise_type in noise_types:
        for noise_fraction in sorted(args.noise_fractions):
            repeats = 1 if noise_fraction == 0.0 else args.num_noise_repeats
            for repeat in range(repeats):
                run_index += 1
                noise_seed = args.seed + repeat * 1000 + int(noise_fraction * 10000)

                logger.info(
                    "Running sensitivity point",
                    run=f"{run_index}/{total_runs}",
                    noise_type=noise_type,
                    noise_fraction=noise_fraction,
                    repeat=repeat,
                )

                start = time.perf_counter()

                # Perturb model
                perturbed = clone_and_perturb(
                    model, noise_type, noise_fraction, noise_seed
                )
                perturbed.to(device)

                # Measure held-out losses
                losses = evaluate_val_losses(
                    perturbed, source, val_indices, device, args.eval_batch_size
                )

                # Export and run games
                perturbed.cpu()
                with tempfile.TemporaryDirectory(prefix="sensitivity_") as tmpdir:
                    onnx_path = export_to_tempdir(perturbed, Path(tmpdir))
                    game_metrics = run_games(onnx_path, args)

                elapsed = time.perf_counter() - start

                point = SensitivityPoint(
                    noise_type=noise_type,
                    noise_fraction=noise_fraction,
                    noise_seed=noise_seed,
                    policy_loss=losses["policy_loss"],
                    value_loss=losses["value_loss"],
                    avg_attack=game_metrics["avg_attack"],
                    avg_lines=game_metrics["avg_lines"],
                    avg_moves=game_metrics["avg_moves"],
                    max_attack=game_metrics["max_attack"],
                    total_attack=game_metrics["total_attack"],
                    num_games=game_metrics["num_games"],
                    elapsed_sec=elapsed,
                )
                results.append(point)

                logger.info(
                    "Sensitivity point done",
                    noise_type=noise_type,
                    noise_fraction=noise_fraction,
                    policy_loss=f"{losses['policy_loss']:.4f}",
                    value_loss=f"{losses['value_loss']:.4f}",
                    avg_attack=f"{game_metrics['avg_attack']:.2f}",
                    elapsed_sec=f"{elapsed:.1f}",
                )

    return results


def aggregate_and_fit(
    points: list[SensitivityPoint],
    baseline_policy_loss: float,
    baseline_value_loss: float,
) -> dict:
    """Aggregate repeated noise samples, fit sigmoids, compute derivatives."""
    analysis: dict = {}

    for noise_type in ["policy", "value"]:
        type_points = [p for p in points if p.noise_type == noise_type]
        loss_key = "policy_loss" if noise_type == "policy" else "value_loss"

        # Average over repeats at each noise_fraction
        by_fraction: dict[float, list[SensitivityPoint]] = {}
        for p in type_points:
            by_fraction.setdefault(p.noise_fraction, []).append(p)

        aggregated_rows = []
        for frac in sorted(by_fraction):
            group = by_fraction[frac]
            row = {
                "noise_fraction": frac,
                "loss_mean": float(np.mean([getattr(p, loss_key) for p in group])),
                "loss_std": float(np.std([getattr(p, loss_key) for p in group])),
                "avg_attack_mean": float(np.mean([p.avg_attack for p in group])),
                "avg_attack_std": float(np.std([p.avg_attack for p in group])),
                "policy_loss_mean": float(np.mean([p.policy_loss for p in group])),
                "value_loss_mean": float(np.mean([p.value_loss for p in group])),
                "n_repeats": len(group),
            }
            aggregated_rows.append(row)

        losses_arr = np.array([r["loss_mean"] for r in aggregated_rows])
        attacks_arr = np.array([r["avg_attack_mean"] for r in aggregated_rows])

        fit_params = fit_sigmoid(losses_arr, attacks_arr)
        baseline_loss = (
            baseline_policy_loss if noise_type == "policy" else baseline_value_loss
        )

        deriv_at_baseline = None
        if fit_params is not None:
            deriv_at_baseline = sigmoid_derivative(
                baseline_loss, fit_params["L"], fit_params["k"], fit_params["x0"]
            )

        analysis[noise_type] = {
            "aggregated": aggregated_rows,
            "sigmoid_fit": fit_params,
            "baseline_loss": baseline_loss,
            "derivative_at_baseline": deriv_at_baseline,
        }

    # Compute recommended weighting
    policy_deriv = analysis["policy"].get("derivative_at_baseline")
    value_deriv = analysis["value"].get("derivative_at_baseline")

    recommended_weighting = None
    if policy_deriv is not None and value_deriv is not None and policy_deriv != 0:
        # ratio = |d(attack)/d(value_loss)| / |d(attack)/d(policy_loss)|
        # If ratio > 1, value loss improvement is more valuable per unit loss change.
        ratio = abs(value_deriv) / abs(policy_deriv)
        recommended_weighting = {
            "value_loss_weight": ratio,
            "policy_derivative": policy_deriv,
            "value_derivative": value_deriv,
            "interpretation": (
                f"Value loss is {ratio:.2f}x as impactful as policy loss "
                f"per unit loss change at the current operating point. "
                f"Recommended: value_loss_weight = {ratio:.3f}"
            ),
        }

    analysis["recommended_weighting"] = recommended_weighting
    return analysis


def main(args: ScriptArgs) -> None:
    # Validate paths
    run_dir = args.run_dir.resolve()
    config_path = run_dir / CONFIG_FILENAME
    data_path = run_dir / TRAINING_DATA_FILENAME
    checkpoint_path = run_dir / CHECKPOINT_DIRNAME / LATEST_CHECKPOINT_FILENAME

    for path, desc in [
        (config_path, "config.json"),
        (data_path, "training_data.npz"),
        (checkpoint_path, "checkpoint"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{desc} not found: {path}")

    # Output directory
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = BENCHMARKS_DIR / "loss_sensitivity" / f"{run_dir.name}_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config and model
    config = load_training_config_json(config_path)
    device_str = pick_device(args.device)
    device = torch.device(device_str)
    logger.info("Loading model", run_dir=str(run_dir), device=device_str)

    model = load_model_from_run(run_dir, config, device)

    # Load validation data (no wandb dependency)
    preload_mode = "none"
    if args.preload_to_gpu and device.type != "cpu":
        preload_mode = "gpu"
    elif args.preload_to_ram:
        preload_mode = "ram"

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
        val_local_indices = np.arange(split_point, len(selected), dtype=np.int64)
        val_eval_indices = select_subset(
            val_local_indices, max_examples=args.eval_examples, seed=args.seed + 2
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
            "Dataset loaded",
            total=total_examples,
            selected=len(selected),
            val_eval=len(val_eval_indices),
        )

        # Baseline (clean model) losses
        baseline_losses = evaluate_val_losses(
            model, source, val_eval_indices, device, args.eval_batch_size
        )
        logger.info(
            "Baseline losses",
            policy_loss=f"{baseline_losses['policy_loss']:.4f}",
            value_loss=f"{baseline_losses['value_loss']:.4f}",
        )

        # Run sensitivity sweep
        points = run_sensitivity_sweep(
            model=model,
            source=source,
            val_indices=val_eval_indices,
            device=device,
            args=args,
        )

        # Aggregate and fit
        analysis = aggregate_and_fit(
            points,
            baseline_policy_loss=baseline_losses["policy_loss"],
            baseline_value_loss=baseline_losses["value_loss"],
        )

        # Write outputs
        raw_path = output_dir / "sensitivity_points.json"
        raw_path.write_text(json.dumps([asdict(p) for p in points], indent=2) + "\n")

        summary = {
            "run_dir": str(run_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "args": {
                **asdict(args),
                "run_dir": str(run_dir),
                "output_dir": str(output_dir),
            },
            "baseline": baseline_losses,
            "analysis": analysis,
        }
        summary_path = output_dir / "sensitivity_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

        # Print summary
        logger.info(
            "Loss sensitivity analysis complete",
            output_dir=str(output_dir),
            baseline_policy_loss=f"{baseline_losses['policy_loss']:.4f}",
            baseline_value_loss=f"{baseline_losses['value_loss']:.4f}",
        )
        if analysis.get("recommended_weighting") is not None:
            rec = analysis["recommended_weighting"]
            logger.info(
                "Recommended loss weighting",
                value_loss_weight=f"{rec['value_loss_weight']:.3f}",
                policy_derivative=f"{rec['policy_derivative']:.4f}",
                value_derivative=f"{rec['value_derivative']:.4f}",
            )
        else:
            logger.warning("Could not compute recommended weighting (fit failed)")

    finally:
        npz.close()


if __name__ == "__main__":
    main(parse(ScriptArgs))
