"""Training configuration for Tetris AlphaZero."""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Mapping, TypeVar
from typing import TypedDict


class ModelKwargs(TypedDict):
    architecture: str
    trunk_channels: int
    num_conv_residual_blocks: int
    reduction_channels: int
    fc_hidden: int
    aux_hidden: int
    num_fusion_blocks: int
    conv_kernel_size: int
    conv_padding: int


@dataclass
class NetworkConfig:
    """Neural network architecture hyperparameters."""

    architecture: str = "gated_fusion"  # 'gated_fusion' (default) or 'simple_aux_mlp'
    trunk_channels: int = 16
    num_conv_residual_blocks: int = 3
    reduction_channels: int = 32
    fc_hidden: int = 128
    aux_hidden: int = 64
    num_fusion_blocks: int = 1
    conv_kernel_size: int = 3
    conv_padding: int = 1

    def to_model_kwargs(self) -> ModelKwargs:
        return {
            "architecture": self.architecture,
            "trunk_channels": self.trunk_channels,
            "num_conv_residual_blocks": self.num_conv_residual_blocks,
            "reduction_channels": self.reduction_channels,
            "fc_hidden": self.fc_hidden,
            "aux_hidden": self.aux_hidden,
            "num_fusion_blocks": self.num_fusion_blocks,
            "conv_kernel_size": self.conv_kernel_size,
            "conv_padding": self.conv_padding,
        }


@dataclass
class OptimizerConfig:
    """Training loop, optimizer, loss, and logging hyperparameters."""

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
    use_torch_compile: bool = (  # If True, use torch.compile for model forward/backward optimization
        True
    )
    train_step_metrics_interval: int = (  # Collect full train-step scalar metrics every N updates (1 = every step)
        16
    )
    compute_extra_train_metrics_on_log: bool = (  # If True, run an extra forward pass at log ticks for diagnostics (policy entropy, accuracy)
        True
    )
    log_individual_games_to_wandb: bool = (  # If True, log one WandB row per completed game instead of aggregated replay summaries
        False
    )


@dataclass
class SelfPlayConfig:
    """MCTS and self-play generation hyperparameters."""

    num_simulations: int = 2000  # MCTS simulations per move
    c_puct: float = 1.5  # PUCT exploration constant
    temperature: float = (  # Sharpening / Smoothening of MCTS visit-count policy target
        1.0
    )
    dirichlet_alpha: float = 0.02
    dirichlet_epsilon: float = 0.25
    add_noise: bool = (  # Whether to add Dirichlet noise at the MCTS root during self-play
        True
    )
    nn_value_weight: float = (  # Scale factor for NN value output in MCTS (0.0 ignores value head)
        1
    )
    nn_value_weight_promotion_multiplier: float = (  # Multiplicative growth target per accepted promotion (e.g. 1.4 means plus-40pct)
        1.4
    )
    nn_value_weight_promotion_max_delta: float = (  # Hard cap on per-promotion absolute increase in nn_value_weight
        0.10
    )
    nn_value_weight_cap: float = (  # Maximum allowed nn_value_weight during promotion ramp
        1.0
    )
    use_tanh_q_normalization: bool = (  # If True, use tanh(Q/q_scale) squashing; if False, use global min-max Q normalization
        False
    )
    q_scale: float = (  # Scale for tanh Q squashing in PUCT (only used when use_tanh_q_normalization=True)
        8.0
    )
    use_parent_value_for_unvisited_q: bool = (  # If True, unvisited action children start from the parent node's initial total-value estimate instead of zero Q
        True
    )
    bootstrap_use_min_max_q_normalization: bool = (  # If True, force no-network bootstrap rollouts to use global min-max Q normalization
        True
    )
    visit_sampling_epsilon: float = (  # Fraction of self-play moves sampled from visit-policy instead of argmax
        0.1
    )
    mcts_seed: (  # Base MCTS seed (combined with env seed + placement); set None for non-deterministic search RNG
        int | None
    ) = 0
    reuse_tree: bool = (  # Reuse MCTS subtree from previous move instead of building fresh tree
        True
    )
    num_workers: int = 7  # Parallel game generation threads
    max_placements: int = (  # Maximum placements (holds excluded) for placement-count normalization
        50
    )
    death_penalty: float = 0.0  # Search-time terminal penalty when game ends in death
    overhang_penalty_weight: float = (  # Search-time weight for normalized overhang penalty
        0.0
    )
    model_promotion_eval_games: int = (  # Candidate games to average before promoting a new self-play model
        20
    )
    bootstrap_without_network: bool = True  # If True, self-play starts with uniform-prior/zero-value MCTS until first promotion
    bootstrap_num_simulations: int = (  # Simulations per move before first promoted NN model
        4000
    )


@dataclass
class ReplayConfig:
    """Replay buffer and batch sampling hyperparameters."""

    buffer_size: int = 4_000_000  # Maximum buffer size. FIFO eviction.
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


@dataclass
class RunConfig:
    """Run management: WandB identity, timing intervals, and auto-populated paths."""

    # WandB
    project_name: str = "tetris-alphazero"
    run_name: str | None = None

    # Intervals (seconds)
    model_sync_interval_seconds: float = 120  # Seconds between ONNX exports
    checkpoint_interval_seconds: float = 10800  # Seconds between checkpoints
    log_interval_seconds: float = 10  # Seconds between logging
    save_interval_seconds: float = (
        0  # Seconds between replay snapshot saves (0 to disable)
    )

    # Paths (set automatically by setup_run_directory)
    run_dir: Path | None = None  # e.g., training_runs/v0
    checkpoint_dir: Path | None = None  # e.g., training_runs/v0/checkpoints
    data_dir: Path | None = None  # e.g., training_runs/v0 (for training_data.npz)


@dataclass
class TrainingConfig:
    """Training hyperparameters - all configurable via CLI."""

    network: NetworkConfig
    optimizer: OptimizerConfig
    self_play: SelfPlayConfig
    replay: ReplayConfig
    run: RunConfig


_ConfigSectionT = TypeVar(
    "_ConfigSectionT",
    NetworkConfig,
    OptimizerConfig,
    SelfPlayConfig,
    ReplayConfig,
    RunConfig,
)


def _coerce_mapping(
    raw_value: object,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(raw_value, Mapping):
        raise TypeError(
            f"{field_name} must be a mapping in saved training config "
            f"(got {type(raw_value).__name__})"
        )
    return raw_value


def _load_config_section(
    section_type: type[_ConfigSectionT],
    data: Mapping[str, Any],
    *,
    section_name: str,
    path_fields: frozenset[str] = frozenset(),
) -> _ConfigSectionT:
    known_fields = {field.name for field in dataclass_fields(section_type)}
    raw_section = data.get(section_name)
    if raw_section is None:
        section_data = {
            key: value for key, value in data.items() if key in known_fields
        }
    else:
        section_data = dict(_coerce_mapping(raw_section, field_name=section_name))

    kwargs = {key: value for key, value in section_data.items() if key in known_fields}
    for path_field in path_fields:
        if kwargs.get(path_field) is not None:
            kwargs[path_field] = Path(str(kwargs[path_field]))
    return section_type(**kwargs)


def training_config_from_dict(data: Mapping[str, Any]) -> TrainingConfig:
    if not isinstance(data, Mapping):
        raise TypeError(
            "Saved training config must be a mapping at the top level "
            f"(got {type(data).__name__})"
        )

    return TrainingConfig(
        network=_load_config_section(NetworkConfig, data, section_name="network"),
        optimizer=_load_config_section(
            OptimizerConfig,
            data,
            section_name="optimizer",
        ),
        self_play=_load_config_section(
            SelfPlayConfig,
            data,
            section_name="self_play",
        ),
        replay=_load_config_section(ReplayConfig, data, section_name="replay"),
        run=_load_config_section(
            RunConfig,
            data,
            section_name="run",
            path_fields=frozenset({"run_dir", "checkpoint_dir", "data_dir"}),
        ),
    )


def load_training_config_json(path: Path) -> TrainingConfig:
    raw_data = json.loads(path.read_text())
    return training_config_from_dict(_coerce_mapping(raw_data, field_name="config"))
