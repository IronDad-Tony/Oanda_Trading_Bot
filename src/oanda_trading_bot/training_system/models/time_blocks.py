import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TimePatchify(nn.Module):
    def __init__(self, in_dim: int, d_model: int, patch_size: int = 16, stride: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.proj = nn.Linear(in_dim * patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C] -> patches [B, P, d_model]
        """
        b, t, c = x.shape
        if t < self.patch_size:
            # pad at front
            pad = self.patch_size - t
            x = F.pad(x, (0, 0, pad, 0))
            t = x.size(1)
        unfold = x.unfold(dimension=1, size=self.patch_size, step=self.stride)  # [B, P, patch, C]
        b, p, ps, c = unfold.shape
        unfold = unfold.contiguous().view(b, p, ps * c)
        out = self.proj(unfold)  # [B,P,d]
        return out


class DilatedTCNBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 3, dilations: Tuple[int, ...] = (1, 2, 4), dropout: float = 0.1):
        super().__init__()
        layers = []
        for d in dilations:
            layers += [
                nn.Conv1d(d_model, d_model, kernel_size=kernel_size, dilation=d, padding=((kernel_size - 1) * d) // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(d_model, d_model, kernel_size=1),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """
        y = x.transpose(1, 2)  # [B,D,T]
        y = self.net(y)
        y = y.transpose(1, 2)
        return self.norm(x + y)

