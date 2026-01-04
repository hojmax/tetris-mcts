"""
Tetris AlphaZero ML Package

This package contains the neural network, training loop, and self-play
infrastructure for AlphaZero-style training on Tetris.
"""

from .action_space import (
    ACTION_TO_PLACEMENT,
    PLACEMENT_TO_ACTION,
    NUM_ACTIONS,
    get_action_mask,
    placement_to_action,
    action_to_placement,
)

from .network import TetrisNet, encode_state, encode_batch

from .data import (
    TrainingExample,
    save_training_data,
    load_training_data,
    TetrisDataset,
    ReplayBuffer,
)

from .weights import (
    save_checkpoint,
    load_checkpoint,
    export_binary,
    load_binary,
    WeightManager,
)

from .selfplay import (
    play_game_random,
    generate_random_games,
    evaluate_policy,
    GameHistory,
    EvalMetrics,
)

from .training import (
    Trainer,
    TrainingConfig,
    compute_loss,
    train_from_data,
)

__all__ = [
    # Action space
    "ACTION_TO_PLACEMENT",
    "PLACEMENT_TO_ACTION",
    "NUM_ACTIONS",
    "get_action_mask",
    "placement_to_action",
    "action_to_placement",
    # Network
    "TetrisNet",
    "encode_state",
    "encode_batch",
    # Data
    "TrainingExample",
    "save_training_data",
    "load_training_data",
    "TetrisDataset",
    "ReplayBuffer",
    # Weights
    "save_checkpoint",
    "load_checkpoint",
    "export_binary",
    "load_binary",
    "WeightManager",
    # Self-play
    "play_game_random",
    "generate_random_games",
    "evaluate_policy",
    "GameHistory",
    "EvalMetrics",
    # Training
    "Trainer",
    "TrainingConfig",
    "compute_loss",
    "train_from_data",
]
