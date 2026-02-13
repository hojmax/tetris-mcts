import json
from pathlib import Path

import numpy as np
import torch

from tetris_mcts.config import BOARD_HEIGHT, BOARD_WIDTH, TrainingConfig
from tetris_mcts.ml.network import TetrisNet
from tetris_mcts.ml.weights import load_checkpoint

COMBO_NORMALIZATION_MAX = 12.0


def load_training_config(config_path: Path) -> TrainingConfig:
    config_data = json.loads(config_path.read_text())
    return TrainingConfig(**config_data)


class ValuePredictor:
    def __init__(self, checkpoint_path: Path, config_path: Path):
        self.model = self._load_model(checkpoint_path, config_path)
        self.value_cache: dict[int, float] = {}

    def _load_model(self, checkpoint_path: Path, config_path: Path) -> TetrisNet:
        config = load_training_config(config_path)
        model = TetrisNet(
            conv_filters=config.conv_filters,
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
        combo: float,
        back_to_back: float,
        next_hidden_piece_probs: np.ndarray,
    ) -> float:
        if index in self.value_cache:
            return self.value_cache[index]

        board_tensor = torch.from_numpy(board.astype(np.float32)).reshape(
            1, 1, BOARD_HEIGHT, BOARD_WIDTH
        )
        hold_available_feature = np.array([hold_available], dtype=np.float32)
        placement_count_feature = np.array([placement_count], dtype=np.float32)
        combo_feature = np.array(
            [min(combo, COMBO_NORMALIZATION_MAX) / COMBO_NORMALIZATION_MAX],
            dtype=np.float32,
        )
        back_to_back_feature = np.array([back_to_back], dtype=np.float32)
        aux = np.concatenate(
            [
                current_piece.astype(np.float32),
                hold_piece.astype(np.float32),
                hold_available_feature,
                next_queue.astype(np.float32).reshape(-1),
                placement_count_feature,
                combo_feature,
                back_to_back_feature,
                next_hidden_piece_probs.astype(np.float32).reshape(-1),
            ]
        )
        aux_tensor = torch.from_numpy(aux).reshape(1, -1)

        with torch.inference_mode():
            _, value = self.model(board_tensor, aux_tensor)
        value_pred = float(value.item())
        self.value_cache[index] = value_pred
        return value_pred
