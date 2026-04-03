"""Training configuration for Tetris AlphaZero."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, TypedDict, cast

import yaml
from pydantic import BaseModel, ConfigDict

from tetris_bot.constants import DEFAULT_CONFIG_PATH


class ModelKwargs(TypedDict):
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

    def to_model_kwargs(self) -> ModelKwargs:
        return {
            "trunk_channels": self.trunk_channels,
            "num_conv_residual_blocks": self.num_conv_residual_blocks,
            "reduction_channels": self.reduction_channels,
            "board_stats_hidden": self.board_stats_hidden,
            "board_proj_hidden": self.board_proj_hidden,
            "fc_hidden": self.fc_hidden,
            "aux_hidden": self.aux_hidden,
            "num_aux_hidden_layers": self.num_aux_hidden_layers,
            "fusion_hidden": self.fusion_hidden,
            "num_fusion_blocks": self.num_fusion_blocks,
            "conv_kernel_size": self.conv_kernel_size,
            "conv_padding": self.conv_padding,
        }


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
    use_tanh_q_normalization: bool
    q_scale: float
    use_parent_value_for_unvisited_q: bool
    bootstrap_use_min_max_q_normalization: bool
    visit_sampling_epsilon: float
    mcts_seed: int | None
    reuse_tree: bool
    num_workers: int
    max_placements: int
    death_penalty: float
    overhang_penalty_weight: float
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


def _coerce_mapping(
    raw_value: object,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(raw_value, Mapping):
        raise TypeError(
            f"{field_name} must be a mapping (got {type(raw_value).__name__})"
        )
    return raw_value


def training_config_from_dict(data: Mapping[str, Any]) -> TrainingConfig:
    return TrainingConfig.model_validate(_coerce_mapping(data, field_name="config"))


def training_config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    return cast(dict[str, Any], config.model_dump(mode="json"))


def training_config_to_yaml(config: TrainingConfig) -> str:
    return yaml.safe_dump(
        training_config_to_dict(config),
        sort_keys=False,
    )


def save_training_config(config: TrainingConfig, path: Path) -> None:
    path.write_text(training_config_to_yaml(config))


def load_training_config(path: Path) -> TrainingConfig:
    raw_data = yaml.safe_load(path.read_text())
    return training_config_from_dict(_coerce_mapping(raw_data, field_name=str(path)))


def default_training_config(config_path: Path = DEFAULT_CONFIG_PATH) -> TrainingConfig:
    return load_training_config(config_path)


def default_network_config(config_path: Path = DEFAULT_CONFIG_PATH) -> NetworkConfig:
    return default_training_config(config_path).network


def default_optimizer_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> OptimizerConfig:
    return default_training_config(config_path).optimizer


def default_self_play_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> SelfPlayConfig:
    return default_training_config(config_path).self_play


def default_replay_config(config_path: Path = DEFAULT_CONFIG_PATH) -> ReplayConfig:
    return default_training_config(config_path).replay


def default_run_config(config_path: Path = DEFAULT_CONFIG_PATH) -> RunConfig:
    return default_training_config(config_path).run
