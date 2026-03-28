from pathlib import Path

from tetris_bot.ml.config import (
    NetworkConfig,
    OptimizerConfig,
    ReplayConfig,
    RunConfig,
    SelfPlayConfig,
    TrainingConfig,
)
from tetris_bot.ml.trainer import Trainer


def _make_config(tmp_path: Path, network: NetworkConfig) -> TrainingConfig:
    checkpoint_dir = tmp_path / "checkpoints"
    data_dir = tmp_path / "data"
    return TrainingConfig(
        network=network,
        optimizer=OptimizerConfig(),
        self_play=SelfPlayConfig(),
        replay=ReplayConfig(),
        run=RunConfig(
            run_dir=tmp_path,
            checkpoint_dir=checkpoint_dir,
            data_dir=data_dir,
        ),
    )


def test_network_config_to_model_kwargs_includes_full_gated_fusion_surface() -> None:
    network = NetworkConfig(
        architecture="gated_fusion",
        trunk_channels=9,
        num_conv_residual_blocks=3,
        reduction_channels=18,
        board_stats_hidden=24,
        board_proj_hidden=192,
        fc_hidden=144,
        aux_hidden=48,
        fusion_hidden=160,
        num_fusion_blocks=2,
        conv_kernel_size=5,
        conv_padding=2,
    )

    assert network.to_model_kwargs() == {
        "architecture": "gated_fusion",
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


def test_network_config_defaults_reflect_current_cached_trunk_size() -> None:
    network = NetworkConfig()

    assert network.trunk_channels == 16
    assert network.num_conv_residual_blocks == 3
    assert network.reduction_channels == 32
    assert network.board_stats_hidden == 32
    assert network.board_proj_hidden == 256
    assert network.fc_hidden == 256
    assert network.aux_hidden == 64
    assert network.fusion_hidden == 256
    assert network.num_fusion_blocks == 1


def test_trainer_builds_model_from_full_network_config(tmp_path: Path) -> None:
    network = NetworkConfig(
        trunk_channels=6,
        num_conv_residual_blocks=3,
        reduction_channels=12,
        board_stats_hidden=20,
        board_proj_hidden=88,
        fc_hidden=96,
        aux_hidden=40,
        fusion_hidden=104,
        num_fusion_blocks=2,
    )
    trainer = Trainer(_make_config(tmp_path, network), device="cpu")

    assert trainer.model.conv_initial.out_channels == 6
    assert len(trainer.model.res_blocks) == 3
    assert trainer.model.conv_reduce.out_channels == 12
    assert trainer.model.board_stats_fc.out_features == 20
    assert trainer.model.board_proj_fc1.out_features == 88
    assert trainer.model.aux_fc.out_features == 40
    assert trainer.model.fusion_fc.out_features == 104
    assert len(trainer.model.fusion_blocks) == 2
    assert trainer.model.policy_fc.in_features == 104
