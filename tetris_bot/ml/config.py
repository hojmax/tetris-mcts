"""Training configuration for Tetris AlphaZero."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class NetworkConfig(ConfigModel):
    """Neural network architecture hyperparameters."""

    trunk_channels: int
    num_conv_residual_blocks: int
    reduction_channels: int
    board_stats_hidden: int
    board_proj_hidden: int
    fc_hidden: int
    aux_hidden: int
    num_aux_hidden_layers: int
    fusion_hidden: int
    num_fusion_blocks: int
    conv_kernel_size: int
    conv_padding: int


class OptimizerConfig(ConfigModel):
    """Training loop, optimizer, loss, and logging hyperparameters."""

    total_steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    lr_schedule: str
    lr_decay_steps: int
    lr_min_factor: float
    lr_step_gamma: float
    lr_step_divisor: int
    value_loss_weight_window: int
    policy_loss_scale: float
    ema_decay: float
    use_torch_compile: bool
    train_step_metrics_interval: int
    compute_extra_train_metrics_on_log: bool
    mirror_augmentation_probability: float
    log_individual_games_to_wandb: bool


class SelfPlayConfig(ConfigModel):
    """MCTS and self-play generation hyperparameters."""

    num_simulations: int
    c_puct: float
    temperature: float
    dirichlet_alpha: float
    dirichlet_epsilon: float
    add_noise: bool
    nn_value_weight: float
    nn_value_weight_promotion_multiplier: float
    nn_value_weight_promotion_max_delta: float
    nn_value_weight_cap: float
    use_parent_value_for_unvisited_q: bool
    visit_sampling_epsilon: float
    mcts_seed: int | None
    reuse_tree: bool
    num_workers: int
    max_placements: int
    death_penalty: float
    overhang_penalty_weight: float
    use_candidate_gating: bool
    model_promotion_eval_games: int
    bootstrap_without_network: bool
    bootstrap_num_simulations: int
    save_eval_trees: bool


class ReplayConfig(ConfigModel):
    """Replay buffer and batch sampling hyperparameters."""

    buffer_size: int
    min_buffer_size: int
    prefetch_batches: int
    staged_batch_cache_batches: int
    mirror_replay_on_accelerator: bool
    replay_mirror_refresh_seconds: float
    replay_mirror_delta_chunk_examples: int
    pin_memory_batches: bool


class RunConfig(ConfigModel):
    """Run management: WandB identity, timing intervals, and auto-populated paths."""

    project_name: str
    run_name: str | None
    model_sync_interval_seconds: float
    model_sync_failure_backoff_seconds: float
    model_sync_max_interval_seconds: float
    checkpoint_interval_seconds: float
    log_interval_seconds: float
    save_interval_seconds: float
    run_dir: Path | None
    checkpoint_dir: Path | None
    data_dir: Path | None


class TrainingConfig(ConfigModel):
    """Training hyperparameters."""

    network: NetworkConfig
    optimizer: OptimizerConfig
    self_play: SelfPlayConfig
    replay: ReplayConfig
    run: RunConfig


def save_training_config(config: TrainingConfig, path: Path) -> None:
    path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))


def load_training_config(path: Path) -> TrainingConfig:
    return TrainingConfig.model_validate(yaml.safe_load(path.read_text()))
