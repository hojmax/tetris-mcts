"""
Loss and Metrics Computation

Implements:
- Action masking for invalid actions
- Policy cross-entropy loss
- Value MSE loss
- Training metrics (entropy, accuracy)
"""

from collections import deque

import torch
import torch.nn.functional as F

from tetris_mcts.ml.network import TetrisNet


class RunningLossBalancer:
    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError(
                f"window_size must be > 0 for RunningLossBalancer (got {window_size})"
            )
        self._policy_losses: deque[float] = deque(maxlen=window_size)
        self._value_losses: deque[float] = deque(maxlen=window_size)

    def has_history(self) -> bool:
        return len(self._policy_losses) > 0

    def append(self, policy_loss: float, value_loss: float) -> None:
        if policy_loss < 0:
            raise ValueError(f"policy_loss must be >= 0 (got {policy_loss})")
        if value_loss <= 0:
            raise ValueError(f"value_loss must be > 0 (got {value_loss})")
        self._policy_losses.append(policy_loss)
        self._value_losses.append(value_loss)

    def averages(self) -> tuple[float, float]:
        if not self.has_history():
            raise ValueError("Cannot compute running averages without history")
        policy_avg = sum(self._policy_losses) / len(self._policy_losses)
        value_avg = sum(self._value_losses) / len(self._value_losses)
        if value_avg <= 0:
            raise ValueError(f"Running average value_loss must be > 0 (got {value_avg})")
        return policy_avg, value_avg

    def value_loss_weight(self) -> float:
        policy_avg, value_avg = self.averages()
        return policy_avg / value_avg


def apply_action_mask(logits: torch.Tensor, action_masks: torch.Tensor) -> torch.Tensor:
    """
    Apply action masks to logits.

    Args:
        logits: Shape (batch, num_actions) - raw policy logits
        action_masks: Shape (batch, num_actions) - 1 for valid actions, 0 for invalid

    Returns:
        masked_logits: Shape (batch, num_actions) - logits with invalid actions set to -inf

    Raises:
        ValueError: If any sample has no valid actions (indicates terminal state in training data)
    """
    valid_counts = action_masks.sum(dim=1)
    if (valid_counts == 0).any():
        invalid_indices = (valid_counts == 0).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"Samples at indices {invalid_indices} have no valid actions. "
            "Terminal states should not be in training data."
        )

    return logits.masked_fill(action_masks == 0, float("-inf"))


def compute_loss(
    model: TetrisNet,
    boards: torch.Tensor,
    aux_features: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    action_masks: torch.Tensor,
    value_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute combined policy and value loss.

    Args:
        model: TetrisNet model
        boards: (batch, 1, 20, 10)
        aux_features: (batch, 52)
        policy_targets: (batch, 735) - MCTS policy targets
        value_targets: (batch,) - discounted attack targets
        action_masks: (batch, 735) - valid action masks
        value_loss_weight: Scale factor applied to value loss in total loss

    Returns:
        total_loss, policy_loss, value_loss
    """
    # Forward pass
    policy_logits, value_pred = model(boards, aux_features)

    # Apply action mask and compute log softmax
    masked_logits = apply_action_mask(policy_logits, action_masks)
    log_policy = F.log_softmax(masked_logits, dim=-1)

    # Replace -inf with 0 for numerical stability (won't contribute to loss anyway
    # since policy_targets should be 0 for invalid actions)
    log_policy = torch.where(
        torch.isinf(log_policy), torch.zeros_like(log_policy), log_policy
    )

    # Policy loss: cross-entropy with MCTS policy
    # -sum(target * log(pred))
    policy_loss = -torch.sum(policy_targets * log_policy, dim=1).mean()

    # Value loss: MSE
    value_loss = F.mse_loss(value_pred.squeeze(-1), value_targets)

    # Total loss with configurable value-loss scaling.
    total_loss = policy_loss + (value_loss_weight * value_loss)

    return total_loss, policy_loss, value_loss


def compute_metrics(
    model: TetrisNet,
    boards: torch.Tensor,
    aux_features: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    action_masks: torch.Tensor,
) -> dict:
    """Compute additional metrics for logging."""
    with torch.no_grad():
        policy_logits, value_pred = model(boards, aux_features)

        # Apply action mask and compute softmax
        masked_logits = apply_action_mask(policy_logits, action_masks)
        policy_probs = F.softmax(masked_logits, dim=-1)

        # Policy entropy (only over valid actions to avoid 0 * -inf = NaN)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        # Replace -inf with 0 so that 0 * 0 = 0 instead of 0 * -inf = NaN
        log_probs_safe = torch.where(
            action_masks == 1, log_probs, torch.zeros_like(log_probs)
        )
        entropy = -torch.sum(policy_probs * log_probs_safe, dim=-1).mean()

        # Value prediction error
        value_error = torch.abs(value_pred.squeeze(-1) - value_targets).mean()

        # Top-1 accuracy (if target is argmax of MCTS policy)
        pred_actions = policy_probs.argmax(dim=-1)
        target_actions = policy_targets.argmax(dim=-1)
        top1_acc = (pred_actions == target_actions).float().mean()

    return {
        "policy_entropy": entropy.item(),
        "value_error": value_error.item(),
        "top1_accuracy": top1_acc.item(),
    }
