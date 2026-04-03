from pathlib import Path

from tetris_bot.constants import DEFAULT_CONFIG_PATH
from tetris_bot.ml.config import NetworkConfig, TrainingConfig, load_training_config
from tetris_bot.ml.trainer import Trainer

_DEFAULT_CONFIG = load_training_config(DEFAULT_CONFIG_PATH)


def _make_config(tmp_path: Path, network: NetworkConfig) -> TrainingConfig:
    config = _DEFAULT_CONFIG.model_copy(deep=True)
    checkpoint_dir = tmp_path / "checkpoints"
    data_dir = tmp_path / "data"
    config.network = network
    config.run = config.run.model_copy(
        update={
            "run_dir": tmp_path,
            "checkpoint_dir": checkpoint_dir,
            "data_dir": data_dir,
        }
    )
    return config


def test_network_config_to_model_kwargs_includes_full_model_surface() -> None:
    network = _DEFAULT_CONFIG.network.model_copy(
        update={
            "trunk_channels": 9,
            "num_conv_residual_blocks": 3,
            "reduction_channels": 18,
            "board_stats_hidden": 24,
            "board_proj_hidden": 192,
            "fc_hidden": 144,
            "aux_hidden": 48,
            "fusion_hidden": 160,
            "num_fusion_blocks": 2,
            "conv_kernel_size": 5,
            "conv_padding": 2,
        }
    )

    assert network.to_model_kwargs() == {
        "trunk_channels": 9,
        "num_conv_residual_blocks": 3,
        "reduction_channels": 18,
        "board_stats_hidden": 24,
        "board_proj_hidden": 192,
        "fc_hidden": 144,
        "aux_hidden": 48,
        "num_aux_hidden_layers": 1,
        "fusion_hidden": 160,
        "num_fusion_blocks": 2,
        "conv_kernel_size": 5,
        "conv_padding": 2,
    }


def test_network_config_defaults_reflect_current_cached_trunk_size() -> None:
    network = _DEFAULT_CONFIG.network

    assert network.trunk_channels == 32
    assert network.num_conv_residual_blocks == 5
    assert network.reduction_channels == 32
    assert network.board_stats_hidden == 32
    assert network.board_proj_hidden == 512
    assert network.fc_hidden == 256
    assert network.aux_hidden == 128
    assert network.num_aux_hidden_layers == 1
    assert network.fusion_hidden == 256
    assert network.num_fusion_blocks == 1


def test_training_config_defaults_match_repo_baseline() -> None:
    assert _DEFAULT_CONFIG.model_dump(mode="json") == {
        "network": {
            "trunk_channels": 32,
            "num_conv_residual_blocks": 5,
            "reduction_channels": 32,
            "board_stats_hidden": 32,
            "board_proj_hidden": 512,
            "fc_hidden": 256,
            "aux_hidden": 128,
            "num_aux_hidden_layers": 1,
            "fusion_hidden": 256,
            "num_fusion_blocks": 1,
            "conv_kernel_size": 3,
            "conv_padding": 1,
        },
        "optimizer": {
            "total_steps": 100000000000,
            "batch_size": 2048,
            "learning_rate": 0.0005,
            "weight_decay": 0.00005,
            "grad_clip_norm": 10.0,
            "lr_schedule": "linear",
            "lr_decay_steps": 200000,
            "lr_min_factor": 0.2,
            "lr_step_gamma": 0.1,
            "lr_step_divisor": 3,
            "value_loss_weight_window": 2000,
            "policy_loss_scale": 10.0,
            "ema_decay": 0.999,
            "use_torch_compile": True,
            "train_step_metrics_interval": 16,
            "compute_extra_train_metrics_on_log": True,
            "mirror_augmentation_probability": 0.5,
            "log_individual_games_to_wandb": False,
        },
        "self_play": {
            "num_simulations": 2000,
            "c_puct": 1.5,
            "temperature": 1.0,
            "dirichlet_alpha": 0.02,
            "dirichlet_epsilon": 0.25,
            "add_noise": True,
            "nn_value_weight": 1.0,
            "nn_value_weight_promotion_multiplier": 1.4,
            "nn_value_weight_promotion_max_delta": 0.1,
            "nn_value_weight_cap": 1.0,
            "use_parent_value_for_unvisited_q": True,
            "visit_sampling_epsilon": 0.1,
            "mcts_seed": 0,
            "reuse_tree": True,
            "num_workers": 7,
            "max_placements": 50,
            "death_penalty": 0.0,
            "overhang_penalty_weight": 0.0,
            "use_candidate_gating": True,
            "model_promotion_eval_games": 20,
            "bootstrap_without_network": True,
            "bootstrap_num_simulations": 4000,
            "save_eval_trees": False,
        },
        "replay": {
            "buffer_size": 7000000,
            "min_buffer_size": 100,
            "prefetch_batches": 1,
            "staged_batch_cache_batches": 1,
            "mirror_replay_on_accelerator": True,
            "replay_mirror_refresh_seconds": 10.0,
            "replay_mirror_delta_chunk_examples": 65536,
            "pin_memory_batches": True,
        },
        "run": {
            "project_name": "tetris-alphazero",
            "run_name": None,
            "model_sync_interval_seconds": 120.0,
            "model_sync_failure_backoff_seconds": 120.0,
            "model_sync_max_interval_seconds": 0.0,
            "checkpoint_interval_seconds": 10800.0,
            "log_interval_seconds": 10.0,
            "save_interval_seconds": 10800.0,
            "run_dir": None,
            "checkpoint_dir": None,
            "data_dir": None,
        },
    }


def test_trainer_builds_model_from_full_network_config(tmp_path: Path) -> None:
    network = _DEFAULT_CONFIG.network.model_copy(
        update={
            "trunk_channels": 6,
            "num_conv_residual_blocks": 3,
            "reduction_channels": 12,
            "board_stats_hidden": 20,
            "board_proj_hidden": 88,
            "fc_hidden": 96,
            "aux_hidden": 40,
            "fusion_hidden": 104,
            "num_fusion_blocks": 2,
        }
    )
    trainer = Trainer(_make_config(tmp_path, network), device="cpu")

    assert trainer.model.conv_initial.out_channels == 6
    assert len(trainer.model.res_blocks) == 3
    assert trainer.model.conv_reduce.out_channels == 12
    assert trainer.model.board_stats_fc.out_features == 20
    assert trainer.model.board_proj_fc1.out_features == 88
    assert trainer.model.aux_mlp[-2].out_features == 40
    assert trainer.model.fusion_fc.out_features == 104
    assert len(trainer.model.fusion_blocks) == 2
    assert trainer.model.policy_fc.in_features == 104
