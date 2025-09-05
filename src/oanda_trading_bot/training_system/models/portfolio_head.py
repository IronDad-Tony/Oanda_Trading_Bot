import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DifferentiableMeanVarianceHead(nn.Module):
    """
    Simple differentiable portfolio head.
    Converts per-asset alpha to weights with a risk-aware adjustment.
    Uses a softmax budget with risk_aversion on (approx) volatility.
    """
    def __init__(self, risk_aversion: float = 1.0, scale: float = 10.0):
        super().__init__()
        self.risk_aversion = float(risk_aversion)
        self.scale = float(scale)

    def forward(
        self,
        alpha: torch.Tensor,           # [B, N]
        vol_est: Optional[torch.Tensor] = None,  # [B, N] optional
    ) -> torch.Tensor:
        if alpha.dim() != 2:
            raise ValueError("alpha must be [B, N]")
        x = alpha
        if vol_est is not None:
            if vol_est.shape != alpha.shape:
                raise ValueError("vol_est shape must match alpha")
            x = x - self.risk_aversion * vol_est
        logits = self.scale * x
        w = torch.softmax(logits, dim=-1)
        return w

