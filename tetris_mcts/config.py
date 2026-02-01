"""Training configuration for Tetris AlphaZero."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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
    num_simulations: int = 100
    temperature: float = 1.0
    dirichlet_alpha: float = 0.15
    dirichlet_epsilon: float = 0.25

    # Replay buffer
    buffer_size: int = 100_000
    min_buffer_size: int = 100
    games_per_save: int = 100  # Games between disk saves (0 to disable)

    # Intervals
    checkpoint_interval: int = 1000  # Steps between checkpoints
    eval_interval: int = 1000  # Steps between evaluations
    log_interval: int = 100  # Steps between logging

    # Evaluation
    eval_seeds: list[int] = field(default_factory=lambda: list(range(20)))

    # Paths
    checkpoint_dir: Path = Path("outputs/checkpoints")
    data_dir: Path = Path("outputs/data")

    # WandB
    project_name: str = "tetris-alphazero"
    run_name: Optional[str] = None
