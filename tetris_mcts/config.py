"""Training configuration for Tetris AlphaZero."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Project root (tetris-mcts/)
PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_RUNS_DIR = PROJECT_ROOT / "training_runs"


@dataclass
class TrainingConfig:
    """Training hyperparameters - all configurable via CLI."""

    # Training
    total_steps: int = 100_000
    model_sync_interval: int = 1000  # Steps between ONNX exports

    # Network architecture
    conv_filters: list[int] = field(default_factory=lambda: [4, 8])
    fc_hidden: int = 128

    # Optimizer
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    lr_schedule: str = "cosine"  # 'cosine', 'step', 'none'
    lr_decay_steps: int = 100000

    # MCTS / Self-play
    num_simulations: int = 400
    temperature: float = 1.0
    dirichlet_alpha: float = 0.15
    dirichlet_epsilon: float = 0.25

    # Replay buffer
    buffer_size: int = 100_000
    min_buffer_size: int = 100
    games_per_save: int = 2000  # Games between disk saves (0 to disable)

    # Intervals
    checkpoint_interval: int = 1000  # Steps between checkpoints
    eval_interval: int = 20000  # Steps between evaluations
    log_interval: int = 100  # Steps between logging

    # Evaluation
    eval_seeds: list[int] = field(default_factory=lambda: list(range(20)))

    # Paths (set automatically by setup_run_directory)
    run_dir: Optional[Path] = None  # e.g., training_runs/v0
    checkpoint_dir: Optional[Path] = None  # e.g., training_runs/v0/checkpoints
    data_dir: Optional[Path] = None  # e.g., training_runs/v0 (for training_data.npz)

    # WandB
    project_name: str = "tetris-alphazero"
    run_name: Optional[str] = None

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

    checkpoint_dir = run_dir / "checkpoints"

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
    config.save(run_dir / "config.json")

    return config
