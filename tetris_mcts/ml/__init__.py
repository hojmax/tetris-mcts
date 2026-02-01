"""
Machine Learning Module for Tetris AlphaZero

Exports:
- TetrisNet: Neural network for policy and value prediction
- Trainer, TrainingConfig: Training loop and configuration
- Evaluator: Model evaluation on fixed seeds
- SelfPlayGenerator: Self-play data generation
- ReplayBuffer, TrainingExample: Data structures
- compute_loss, compute_metrics: Loss functions
- train_from_data: Train from pre-generated data
"""

from tetris_mcts.ml.network import TetrisNet
from tetris_mcts.ml.training import Trainer, TrainingConfig
from tetris_mcts.ml.evaluation import Evaluator
from tetris_mcts.ml.selfplay import SelfPlayGenerator
from tetris_mcts.ml.data import ReplayBuffer, TrainingExample
from tetris_mcts.ml.loss import compute_loss, compute_metrics
from tetris_mcts.ml.dataset_training import train_from_data

__all__ = [
    "TetrisNet",
    "Trainer",
    "TrainingConfig",
    "Evaluator",
    "SelfPlayGenerator",
    "ReplayBuffer",
    "TrainingExample",
    "compute_loss",
    "compute_metrics",
    "train_from_data",
]
