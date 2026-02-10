"""Training configuration for Tetris AlphaZero."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

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
EVAL_ONNX_FILENAME = "eval.onnx"
EVAL_REPLAYS_FILENAME = "eval_replays.jsonl"
TRAINING_DATA_FILENAME = "training_data.npz"
LATEST_ONNX_FILENAME = "latest.onnx"
LATEST_METADATA_FILENAME = "latest_metadata.json"
LATEST_CHECKPOINT_FILENAME = "latest.pt"
CHECKPOINT_FILENAME_PREFIX = "checkpoint"

# Visualization / logging defaults
DEFAULT_EVAL_TRAJECTORY_MAX_FRAMES = 30
DEFAULT_GIF_FRAME_DURATION_MS = 300
DEFAULT_GIF_FPS = 3

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


@dataclass
class TrainingConfig:
    """Training hyperparameters - all configurable via CLI."""

    # Network architecture
    conv_filters: list[int] = field(default_factory=lambda: [4, 8])
    fc_hidden: int = 128
    conv_kernel_size: int = 3
    conv_padding: int = 1

    # Training
    total_steps: int = 100_000_000_000
    batch_size: int = 1024
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    lr_schedule: str = "cosine"  # 'cosine', 'step', 'none'
    lr_decay_steps: int = 100_000
    lr_min_factor: float = 0.5  # Minimum LR as fraction of initial (for cosine)
    lr_step_gamma: float = 0.1  # LR decay factor (for step scheduler)
    lr_step_divisor: int = 3  # Decay every (lr_decay_steps // divisor) steps
    value_loss_weight: float = 30.0  # Scale factor for value loss in total loss
    value_loss_weight_window: int = (  # Rolling window size for dynamic value-loss weighting
        2000
    )

    # MCTS / Self-play
    num_simulations: int = 1000
    c_puct: float = 1.5  # PUCT exploration constant
    temperature: float = 1.5
    dirichlet_alpha: float = 0.01
    dirichlet_epsilon: float = 0.25
    num_workers: int = 7  # Parallel game generation threads
    max_moves: int = 100  # Maximum moves for move number normalization
    death_penalty: float = 5.0  # Penalty subtracted from value when game ends in death
    overhang_penalty_weight: float = (  # Weight for normalized overhang penalty in value targets
        5.0
    )
    model_promotion_eval_games: int = (
        30  # Candidate games to average before promoting a new self-play model
    )
    model_promotion_eval_add_noise: bool = (
        False  # Whether evaluator worker adds Dirichlet noise while gating candidates
    )
    bootstrap_without_network: bool = True  # If True, self-play starts with uniform-prior/zero-value MCTS until first promotion
    bootstrap_num_simulations: int = (
        4000  # Simulations per move before first promoted NN model
    )

    # Replay buffer
    buffer_size: int = 500_000
    min_buffer_size: int = 100
    games_per_save: int = 2000  # Games between disk saves (0 to disable)

    # Intervals
    model_sync_interval: int = 2000  # Steps between ONNX exports
    checkpoint_interval: int = 80000  # Steps between checkpoints
    eval_interval: int = 30000  # Steps between evaluations
    log_interval: int = 100  # Steps between logging

    # Evaluation
    eval_seeds: list[int] = field(default_factory=lambda: list(range(1)))
    eval_mcts_seed: int = 12345  # Fixed MCTS RNG seed for deterministic evaluation

    # Paths (set automatically by setup_run_directory)
    run_dir: Optional[Path] = None  # e.g., training_runs/v0
    checkpoint_dir: Optional[Path] = None  # e.g., training_runs/v0/checkpoints
    data_dir: Optional[Path] = None  # e.g., training_runs/v0 (for training_data.npz)

    # WandB
    project_name: str = "tetris-alphazero"
    run_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be > 0")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be > 0")
        if self.max_moves <= 0:
            raise ValueError("max_moves must be > 0")
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")
        if self.min_buffer_size <= 0:
            raise ValueError("min_buffer_size must be > 0")
        if self.min_buffer_size > self.buffer_size:
            raise ValueError(
                "min_buffer_size must be <= buffer_size "
                f"(got {self.min_buffer_size} > {self.buffer_size})"
            )
        if self.games_per_save < 0:
            raise ValueError("games_per_save must be >= 0")
        if self.model_sync_interval <= 0:
            raise ValueError("model_sync_interval must be > 0")
        if self.model_promotion_eval_games <= 0:
            raise ValueError("model_promotion_eval_games must be > 0")
        if self.bootstrap_num_simulations <= 0:
            raise ValueError("bootstrap_num_simulations must be > 0")
        if self.lr_schedule not in {"cosine", "step", "none"}:
            raise ValueError(
                f"lr_schedule must be one of cosine, step, none (got {self.lr_schedule})"
            )
        if self.lr_decay_steps <= 0:
            raise ValueError("lr_decay_steps must be > 0")
        if self.lr_step_divisor <= 0:
            raise ValueError("lr_step_divisor must be > 0")
        if self.value_loss_weight <= 0:
            raise ValueError("value_loss_weight must be > 0")
        if self.value_loss_weight_window <= 0:
            raise ValueError("value_loss_weight_window must be > 0")
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
    run_dir: Optional[Path] = None,
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
