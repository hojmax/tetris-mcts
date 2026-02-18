import dataclasses
import json
from pathlib import Path

import numpy as np
import structlog
import torch

from tetris_mcts.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
)
from tetris_mcts.config import TrainingConfig
from tetris_mcts.ml.network import TetrisNet, build_aux_features
from tetris_mcts.ml.weights import load_checkpoint

logger = structlog.get_logger()


def load_training_config(config_path: Path) -> TrainingConfig:
    config_data = json.loads(config_path.read_text())
    known_fields = {f.name for f in dataclasses.fields(TrainingConfig)}
    config_data = {k: v for k, v in config_data.items() if k in known_fields}
    return TrainingConfig(**config_data)


def try_load_value_predictor(
    checkpoint_path: Path, config_path: Path
) -> "ValuePredictor | None":
    if not checkpoint_path.exists() or not config_path.exists():
        logger.warning(
            "Model predictions disabled: checkpoint/config not found",
            checkpoint_path=str(checkpoint_path),
            config_path=str(config_path),
        )
        return None
    try:
        return ValuePredictor(checkpoint_path, config_path)
    except RuntimeError as e:
        logger.warning(
            "Model predictions disabled: incompatible checkpoint", error=str(e)
        )
        return None


class ValuePredictor:
    def __init__(self, checkpoint_path: Path, config_path: Path):
        self.model = self._load_model(checkpoint_path, config_path)
        self.value_cache: dict[int, float] = {}

    def _load_model(self, checkpoint_path: Path, config_path: Path) -> TetrisNet:
        config = load_training_config(config_path)
        model = TetrisNet(
            trunk_channels=config.trunk_channels,
            num_conv_residual_blocks=config.num_conv_residual_blocks,
            reduction_channels=config.reduction_channels,
            fc_hidden=config.fc_hidden,
            conv_kernel_size=config.conv_kernel_size,
            conv_padding=config.conv_padding,
        )
        load_checkpoint(checkpoint_path, model=model)
        model.eval()
        return model

    def predict_value(
        self,
        index: int,
        board: np.ndarray,
        current_piece: np.ndarray,
        hold_piece: np.ndarray,
        hold_available: float,
        next_queue: np.ndarray,
        placement_count: float,
        combo_feature: float,
        back_to_back: float,
        next_hidden_piece_probs: np.ndarray,
        column_heights: np.ndarray,
        max_column_height: float,
        min_column_height: float,
        row_fill_counts: np.ndarray,
        total_blocks: float,
        bumpiness: float,
        holes: float,
        overhang_fields: float,
    ) -> float:
        if index in self.value_cache:
            return self.value_cache[index]

        board_tensor = torch.from_numpy(board.astype(np.float32)).reshape(
            1, 1, BOARD_HEIGHT, BOARD_WIDTH
        )
        aux = build_aux_features(
            current_piece=current_piece,
            hold_piece=hold_piece,
            hold_available=hold_available,
            next_queue=next_queue,
            placement_count=placement_count,
            combo_feature=combo_feature,
            back_to_back=back_to_back,
            next_hidden_piece_probs=next_hidden_piece_probs,
            column_heights=column_heights,
            max_column_height=max_column_height,
            min_column_height=min_column_height,
            row_fill_counts=row_fill_counts,
            total_blocks=total_blocks,
            bumpiness=bumpiness,
            holes=holes,
            overhang_fields=overhang_fields,
        )
        aux_tensor = torch.from_numpy(aux).reshape(1, -1)

        with torch.inference_mode():
            _, value = self.model(board_tensor, aux_tensor)
        value_pred = float(value.item())
        self.value_cache[index] = value_pred
        return value_pred
