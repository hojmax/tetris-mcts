"""
Dataset Training

Train from pre-generated data files (not self-play).
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
import numpy as np

from tetris_mcts.ml.network import TetrisNet
from tetris_mcts.ml.data import TetrisDataset
from tetris_mcts.ml.loss import compute_loss


def train_from_data(
    data_path: str | Path,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    conv_filters: Optional[list[int]] = None,
    fc_hidden: int = 128,
    num_epochs: int = 10,
    device: str = "cpu",
) -> TetrisNet:
    """Train from pre-generated data file.

    Args:
        data_path: Path to NPZ data file
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        conv_filters: Convolutional filter sizes
        fc_hidden: Hidden layer size for fully connected layers
        num_epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')

    Returns:
        Trained TetrisNet model
    """
    if conv_filters is None:
        conv_filters = [4, 8]

    # Load data
    dataset = TetrisDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Create model
    model = TetrisNet(
        conv_filters=conv_filters,
        fc_hidden=fc_hidden,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Training loop
    print(f"Training on {len(dataset)} examples for {num_epochs} epochs")

    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_policy_losses = []
        epoch_value_losses = []

        for batch in dataloader:
            boards, aux, policy_targets, value_targets, masks = batch
            boards = boards.to(device)
            aux = aux.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            total_loss, policy_loss, value_loss = compute_loss(
                model, boards, aux, policy_targets, value_targets, masks
            )
            total_loss.backward()
            optimizer.step()

            epoch_losses.append(total_loss.item())
            epoch_policy_losses.append(policy_loss.item())
            epoch_value_losses.append(value_loss.item())

        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"loss={np.mean(epoch_losses):.4f}, "
            f"policy={np.mean(epoch_policy_losses):.4f}, "
            f"value={np.mean(epoch_value_losses):.4f}"
        )

    return model
