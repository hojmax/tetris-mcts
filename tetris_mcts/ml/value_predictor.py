import json
from pathlib import Path

import numpy as np
import torch

from tetris_mcts.config import BOARD_HEIGHT, BOARD_WIDTH, TrainingConfig
from tetris_mcts.ml.network import TetrisNet, build_aux_features
from tetris_mcts.ml.weights import load_checkpoint


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
        combo_feature: float,
        back_to_back: float,
        next_hidden_piece_probs: np.ndarray,
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
        )
        aux_tensor = torch.from_numpy(aux).reshape(1, -1)

        with torch.inference_mode():
            _, value = self.model(board_tensor, aux_tensor)
        value_pred = float(value.item())
        self.value_cache[index] = value_pred
        return value_pred
