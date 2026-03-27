from __future__ import annotations

import copy

import torch
from torch import nn


class ExponentialMovingAverage:
    """Track an eval-only EMA shadow copy for export and evaluation."""

    def __init__(self, model: nn.Module, decay: float):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1) (got {decay})")
        self.decay = decay
        self._ema_model = copy.deepcopy(model).eval()
        self._ema_model.requires_grad_(False)
        self._named_parameters = dict(self._ema_model.named_parameters())
        self._named_buffers = dict(self._ema_model.named_buffers())

    @property
    def model(self) -> nn.Module:
        return self._ema_model

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            self._named_parameters[name].lerp_(
                parameter.detach(),
                1.0 - self.decay,
            )
        for name, buffer in model.named_buffers():
            self._named_buffers[name].copy_(buffer.detach())
