"""Training configuration for Tetris AlphaZero."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

# Project root (tetris-mcts/)
PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_RUNS_DIR = PROJECT_ROOT / "training_runs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"

# Core game constants
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
NUM_PIECE_TYPES = 7
QUEUE_SIZE = 5
NUM_ACTIONS = 735

# Artifact names
CHECKPOINT_DIRNAME = "checkpoints"
MODEL_CANDIDATES_DIRNAME = "model_candidates"
CONFIG_FILENAME = "config.json"
PARALLEL_ONNX_FILENAME = "parallel.onnx"
INCUMBENT_ONNX_FILENAME = "incumbent.onnx"
TRAINING_DATA_FILENAME = "training_data.npz"
LATEST_ONNX_FILENAME = "latest.onnx"
LATEST_METADATA_FILENAME = "latest_metadata.json"
LATEST_CHECKPOINT_FILENAME = "latest.pt"
CHECKPOINT_FILENAME_PREFIX = "checkpoint"

# Visualization / logging defaults
DEFAULT_GIF_FRAME_DURATION_MS = 300

# Piece metadata
PIECE_NAMES = ["I", "O", "T", "S", "Z", "J", "L"]
PIECE_COLORS = [
    (93, 173, 212),  # I - Cyan
    (219, 174, 63),  # O - Yellow
    (178, 74, 156),  # T - Magenta
    (114, 184, 65),  # S - Green
    (204, 65, 65),  # Z - Red
    (59, 84, 165),  # J - Blue
    (227, 127, 59),  # L - Orange
]

# Spawn-rotation (state 0) cell offsets — mirrors Rust TETROMINO_CELLS[][0]
PIECE_SPAWN_CELLS: list[list[tuple[int, int]]] = [
    [(0, 1), (1, 1), (2, 1), (3, 1)],  # I
    [(1, 1), (2, 1), (1, 2), (2, 2)],  # O
    [(1, 0), (0, 1), (1, 1), (2, 1)],  # T
    [(1, 0), (2, 0), (0, 1), (1, 1)],  # S
    [(0, 0), (1, 0), (1, 1), (2, 1)],  # Z
    [(0, 0), (0, 1), (1, 1), (2, 1)],  # J
    [(2, 0), (0, 1), (1, 1), (2, 1)],  # L
]


@dataclass
class TrainingConfig:
    """Training hyperparameters - all configurable via CLI."""

    # Network architecture
    trunk_channels: int = 16
    num_conv_residual_blocks: int = 1
    reduction_channels: int = 32
    fc_hidden: int = 128
    conv_kernel_size: int = 3
    conv_padding: int = 1

    # Training
    total_steps: int = 100_000_000_000
    batch_size: int = 1024
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    lr_schedule: str = "linear"  # 'linear', 'cosine', 'step', 'none'
    lr_decay_steps: int = 200_000
    lr_min_factor: float = 0.2  # Final LR as fraction of initial (for linear/cosine)
    lr_step_gamma: float = 0.1  # LR decay factor (for step scheduler)
    lr_step_divisor: int = 3  # Decay every (lr_decay_steps // divisor) steps
    value_loss_weight_window: int = (  # Rolling window size for dynamic value-loss weighting
        2000
    )
    train_step_metrics_interval: int = (  # Collect full train-step scalar metrics every N updates (1 = every step)
        16
    )
    compute_extra_train_metrics_on_log: bool = (  # If True, run an extra forward pass at log ticks for diagnostics (policy entropy, accuracy)
        True
    )
    log_individual_games_to_wandb: bool = (  # If True, log one WandB row per completed game instead of aggregated replay summaries
        True
    )
    use_huber_value_loss: bool = (  # If True, use Huber loss for value head; if False, use MSE
        True
    )
    use_torch_compile: bool = (  # If True, use torch.compile for model forward/backward optimization
        True
    )

    # MCTS / Self-play
    num_simulations: int = 1000  # MCTS simulations per move
    c_puct: float = 1.5  # PUCT exploration constant
    temperature: float = (  # Sharpening / Smoothening of MCTS visit-count policy target
        0.8
    )
    dirichlet_alpha: float = 0.02
    dirichlet_epsilon: float = 0.25
    add_noise: bool = (  # Whether to add Dirichlet noise at the MCTS root during self-play
        True
    )
    nn_value_weight: float = (  # Scale factor for NN value output in MCTS (0.0 ignores value head)
        0.01
    )
    nn_value_weight_promotion_multiplier: float = (  # Multiplicative growth target per accepted promotion (e.g. 1.4 means +40%)
        1.4
    )
    nn_value_weight_promotion_max_delta: float = (  # Hard cap on per-promotion absolute increase in nn_value_weight
        0.10
    )
    nn_value_weight_cap: float = (  # Maximum allowed nn_value_weight during promotion ramp
        1.0
    )
    use_tanh_q_normalization: bool = (  # If True, use tanh(Q/q_scale) squashing; if False, use sibling min-max Q normalization
        True
    )
    q_scale: float = (  # Scale for tanh Q squashing in PUCT (only used when use_tanh_q_normalization=True)
        8.0
    )
    visit_sampling_epsilon: float = (  # Fraction of self-play moves sampled from visit-policy instead of argmax
        0
    )
    num_workers: int = 7  # Parallel game generation threads
    max_placements: int = (  # Maximum placements (holds excluded) for placement-count normalization
        50
    )
    death_penalty: float = 5.0  # Search-time terminal penalty when game ends in death
    overhang_penalty_weight: float = (  # Search-time weight for normalized overhang penalty
        5.0
    )
    model_promotion_eval_games: int = (  # Candidate games to average before promoting a new self-play model
        50
    )
    bootstrap_without_network: bool = True  # If True, self-play starts with uniform-prior/zero-value MCTS until first promotion
    bootstrap_num_simulations: int = (  # Simulations per move before first promoted NN model
        4000
    )

    # Replay buffer
    buffer_size: int = 2_000_000  # Maximum buffer size. FIFO eviction.
    min_buffer_size: int = 100  # Minimum buffer size before training starts
    prefetch_batches: int = (  # Number of train batches sampled/staged per generator.sample_batch call
        1
    )
    staged_batch_cache_batches: int = (  # Target number of train batches kept staged in memory/device queue
        1
    )
    mirror_replay_on_accelerator: bool = (  # If True, mirror full replay buffer on CUDA/MPS and train from device-local samples
        True
    )
    replay_mirror_refresh_seconds: float = (  # Seconds between full replay mirror refreshes while training from device-local replay
        10.0
    )
    replay_mirror_delta_chunk_examples: int = (  # Max replay examples pulled per incremental mirror delta call
        65_536
    )
    pin_memory_batches: bool = (  # If True, pin host tensors before CUDA transfer
        True
    )

    # Intervals (seconds)
    model_sync_interval_seconds: float = 300  # Seconds between ONNX exports
    checkpoint_interval_seconds: float = 10800  # Seconds between checkpoints
    log_interval_seconds: float = 10  # Seconds between logging
    save_interval_seconds: float = (  # Seconds between replay snapshot saves (0 to disable)
        3600
    )

    # Paths (set automatically by setup_run_directory)
    run_dir: Path | None = None  # e.g., training_runs/v0
    checkpoint_dir: Path | None = None  # e.g., training_runs/v0/checkpoints
    data_dir: Path | None = None  # e.g., training_runs/v0 (for training_data.npz)

    # WandB
    project_name: str = "tetris-alphazero"
    run_name: str | None = None

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be > 0")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be > 0")
        if self.max_placements <= 0:
            raise ValueError("max_placements must be > 0")
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")
        if self.min_buffer_size <= 0:
            raise ValueError("min_buffer_size must be > 0")
        if self.min_buffer_size > self.buffer_size:
            raise ValueError(
                "min_buffer_size must be <= buffer_size "
                f"(got {self.min_buffer_size} > {self.buffer_size})"
            )
        if self.prefetch_batches <= 0:
            raise ValueError("prefetch_batches must be > 0")
        if self.staged_batch_cache_batches <= 0:
            raise ValueError("staged_batch_cache_batches must be > 0")
        if (
            not math.isfinite(self.replay_mirror_refresh_seconds)
            or self.replay_mirror_refresh_seconds <= 0
        ):
            raise ValueError("replay_mirror_refresh_seconds must be finite and > 0")
        if self.replay_mirror_delta_chunk_examples <= 0:
            raise ValueError("replay_mirror_delta_chunk_examples must be > 0")
        if (
            not math.isfinite(self.save_interval_seconds)
            or self.save_interval_seconds < 0
        ):
            raise ValueError("save_interval_seconds must be finite and >= 0")
        if not 0.0 <= self.visit_sampling_epsilon <= 1.0:
            raise ValueError(
                "visit_sampling_epsilon must be in [0, 1] "
                f"(got {self.visit_sampling_epsilon})"
            )
        if self.nn_value_weight < 0.0:
            raise ValueError(
                f"nn_value_weight must be >= 0 (got {self.nn_value_weight})"
            )
        if self.use_tanh_q_normalization and (
            not math.isfinite(self.q_scale) or self.q_scale <= 0.0
        ):
            raise ValueError(f"q_scale must be finite and > 0 (got {self.q_scale})")
        if (
            not math.isfinite(self.nn_value_weight_promotion_multiplier)
            or self.nn_value_weight_promotion_multiplier < 1.0
        ):
            raise ValueError(
                "nn_value_weight_promotion_multiplier must be finite and >= 1.0 "
                f"(got {self.nn_value_weight_promotion_multiplier})"
            )
        if (
            not math.isfinite(self.nn_value_weight_promotion_max_delta)
            or self.nn_value_weight_promotion_max_delta < 0.0
        ):
            raise ValueError(
                "nn_value_weight_promotion_max_delta must be finite and >= 0 "
                f"(got {self.nn_value_weight_promotion_max_delta})"
            )
        if (
            not math.isfinite(self.nn_value_weight_cap)
            or self.nn_value_weight_cap < 0.0
        ):
            raise ValueError(
                "nn_value_weight_cap must be finite and >= 0 "
                f"(got {self.nn_value_weight_cap})"
            )
        if self.nn_value_weight > self.nn_value_weight_cap:
            raise ValueError(
                "nn_value_weight must be <= nn_value_weight_cap "
                f"(got {self.nn_value_weight} > {self.nn_value_weight_cap})"
            )
        if (
            not math.isfinite(self.model_sync_interval_seconds)
            or self.model_sync_interval_seconds <= 0
        ):
            raise ValueError("model_sync_interval_seconds must be finite and > 0")
        if (
            not math.isfinite(self.checkpoint_interval_seconds)
            or self.checkpoint_interval_seconds <= 0
        ):
            raise ValueError("checkpoint_interval_seconds must be finite and > 0")
        if (
            not math.isfinite(self.log_interval_seconds)
            or self.log_interval_seconds <= 0
        ):
            raise ValueError("log_interval_seconds must be finite and > 0")
        if self.model_promotion_eval_games <= 0:
            raise ValueError("model_promotion_eval_games must be > 0")
        if self.bootstrap_num_simulations <= 0:
            raise ValueError("bootstrap_num_simulations must be > 0")
        if self.lr_schedule not in {"linear", "cosine", "step", "none"}:
            raise ValueError(
                "lr_schedule must be one of linear, cosine, step, none "
                f"(got {self.lr_schedule})"
            )
        if self.lr_decay_steps <= 0:
            raise ValueError("lr_decay_steps must be > 0")
        if self.lr_step_divisor <= 0:
            raise ValueError("lr_step_divisor must be > 0")
        if self.value_loss_weight_window <= 0:
            raise ValueError("value_loss_weight_window must be > 0")
        if self.train_step_metrics_interval <= 0:
            raise ValueError("train_step_metrics_interval must be > 0")
        if self.lr_schedule == "step":
            step_size = self.lr_decay_steps // self.lr_step_divisor
            if step_size <= 0:
                raise ValueError(
                    "StepLR step_size must be > 0; adjust lr_decay_steps or lr_step_divisor "
                    f"(got {self.lr_decay_steps} // {self.lr_step_divisor} = {step_size})"
                )

    def to_json(self) -> str:
        """Serialize config to JSON string."""
        d = asdict(self)
        # Convert Path objects to strings
        for key in ["run_dir", "checkpoint_dir", "data_dir"]:
            if d[key] is not None:
                d[key] = str(d[key])
        return json.dumps(d, indent=2)

    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        path.write_text(self.to_json())


def get_next_version(base_dir: Path) -> int:
    """Find the next available version number in base_dir."""
    if not base_dir.exists():
        return 0
    existing = [
        int(d.name[1:])
        for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
    ]
    return max(existing, default=-1) + 1


def setup_run_directory(
    config: TrainingConfig,
    base_dir: Path = TRAINING_RUNS_DIR,
    run_dir: Path | None = None,
) -> TrainingConfig:
    """
    Set up run directory and update config paths.

    For new runs, creates:
        base_dir/vN/
        base_dir/vN/checkpoints/
        base_dir/vN/config.json

    For resumed runs (run_dir provided), uses existing directory.

    Args:
        config: Training config to update
        base_dir: Base directory for all training runs (used for new runs)
        run_dir: Existing run directory to resume (skips version auto-increment)

    Returns:
        Updated config with paths set
    """
    if run_dir is None:
        # Create new versioned run
        version = get_next_version(base_dir)
        run_dir = base_dir / f"v{version}"

    checkpoint_dir = run_dir / CHECKPOINT_DIRNAME

    # Create directories
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    # Update config paths
    config.run_dir = run_dir
    config.checkpoint_dir = checkpoint_dir
    config.data_dir = run_dir  # training_data.npz goes in run_dir

    # Auto-set run name if not specified (use directory name)
    if config.run_name is None:
        config.run_name = run_dir.name

    # Save config to run directory
    config.save(run_dir / CONFIG_FILENAME)

    return config
