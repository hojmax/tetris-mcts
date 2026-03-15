from __future__ import annotations

import pytest
import torch

from tetris_bot.ml.loss import apply_action_mask


def test_apply_action_mask_accepts_boolean_masks() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    action_masks = torch.tensor([[True, False, True], [False, True, False]])

    masked_logits = apply_action_mask(logits, action_masks)

    assert masked_logits[0, 0].item() == pytest.approx(1.0)
    assert torch.isneginf(masked_logits[0, 1])
    assert masked_logits[1, 1].item() == pytest.approx(5.0)
    assert torch.isneginf(masked_logits[1, 2])


def test_apply_action_mask_rejects_empty_boolean_rows() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    action_masks = torch.tensor([[False, False, False]])

    with pytest.raises(ValueError, match="have no valid actions"):
        apply_action_mask(logits, action_masks)
