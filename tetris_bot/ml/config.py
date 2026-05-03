"""Training configuration for Tetris AlphaZero."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


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


class R2SyncConfig(ConfigModel):
    """Cloudflare R2 (S3-compatible) sync configuration for multi-machine training.

    Account-specific values (`R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`,
    `R2_SECRET_ACCESS_KEY`) are sourced from environment variables — typically
    via a `.env` file loaded with `python-dotenv` at process startup. The
    config object only carries non-secret, run-shared fields like role,
    bucket name, and sync intervals.

    Roles:
    - "trainer": pushes ONNX bundles, ingests replay chunks
    - "generator": pushes replay chunks, pulls ONNX bundles
    - "both": runs trainer locally and also ingests remote chunks (default for
      Vast.ai trainer that still wants its own local generation alongside remote)
    - "off": R2 sync disabled (default; existing single-machine workflow)
    """

    role: str = "off"
    bucket: str | None = None
    prefix: str = "tetris-mcts"
    sync_run_id: str | None = None
    chunk_max_examples: int = 4096
    chunk_upload_interval_seconds: float = 15.0
    chunk_download_poll_interval_seconds: float = 10.0
    model_pointer_poll_interval_seconds: float = 20.0
    request_timeout_seconds: float = 30.0


class TrainingConfig(ConfigModel):
    """Training hyperparameters."""

    network: NetworkConfig
    optimizer: OptimizerConfig
    self_play: SelfPlayConfig
    replay: ReplayConfig
    run: RunConfig
    r2_sync: R2SyncConfig = Field(default_factory=R2SyncConfig)


class RuntimeOptimizerOverrides(ConfigModel):
    """Live-tunable optimizer overrides loaded from runtime_overrides.yaml."""

    lr_multiplier: float | None = 1.0
    grad_clip_norm: float | None = None
    weight_decay: float | None = None
    mirror_augmentation_probability: float | None = None


class RuntimeRunOverrides(ConfigModel):
    """Live-tunable run interval overrides loaded from runtime_overrides.yaml."""

    log_interval_seconds: float | None = None
    checkpoint_interval_seconds: float | None = None


class RuntimeSelfPlayOverrides(ConfigModel):
    """One-shot self-play triggers loaded from runtime_overrides.yaml.

    Unlike the optimizer/run override fields, these are consumed once and
    reset back to their default in the file by the trainer.
    """

    force_promote_next_candidate: bool = False


class RuntimeOverrides(ConfigModel):
    """Whitelisted runtime overrides that can change during training."""

    optimizer: RuntimeOptimizerOverrides = Field(
        default_factory=RuntimeOptimizerOverrides
    )
    run: RuntimeRunOverrides = Field(default_factory=RuntimeRunOverrides)
    self_play: RuntimeSelfPlayOverrides = Field(
        default_factory=RuntimeSelfPlayOverrides
    )


class ResolvedRuntimeOptimizerOverrides(ConfigModel):
    lr_multiplier: float
    grad_clip_norm: float
    weight_decay: float
    mirror_augmentation_probability: float


class ResolvedRuntimeRunOverrides(ConfigModel):
    log_interval_seconds: float
    checkpoint_interval_seconds: float


class ResolvedRuntimeOverrides(ConfigModel):
    optimizer: ResolvedRuntimeOptimizerOverrides
    run: ResolvedRuntimeRunOverrides


def save_training_config(config: TrainingConfig, path: Path) -> None:
    path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))


def load_training_config(path: Path) -> TrainingConfig:
    return TrainingConfig.model_validate(yaml.safe_load(path.read_text()))


def save_runtime_overrides(overrides: RuntimeOverrides, path: Path) -> None:
    path.write_text(yaml.safe_dump(overrides.model_dump(mode="json"), sort_keys=False))


def load_runtime_overrides(path: Path) -> RuntimeOverrides:
    raw_data = yaml.safe_load(path.read_text())
    if raw_data is None:
        return RuntimeOverrides()
    return RuntimeOverrides.model_validate(raw_data)
