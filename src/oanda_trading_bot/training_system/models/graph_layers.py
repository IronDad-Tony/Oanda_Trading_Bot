import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def cosine_adj(x: torch.Tensor, topk: int = 5) -> torch.Tensor:
    """
    Build row-normalized cosine similarity adjacency.
    Args:
      x: [B, N, D]
      topk: keep top-k neighbors (including self) per node
    Returns:
      A: [B, N, N] row-stochastic adjacency
    """
    b, n, d = x.shape
    x_norm = F.normalize(x, p=2, dim=-1)
    sim = torch.matmul(x_norm, x_norm.transpose(1, 2))  # [B,N,N]
    # Keep top-k per row
    if topk > 0 and topk < n:
        vals, idx = torch.topk(sim, k=topk, dim=-1)
        mask = torch.zeros_like(sim).scatter_(-1, idx, 1.0)
        sim = sim * mask + (-1e9) * (1 - mask)
    # row softmax -> stochastic
    A = torch.softmax(sim, dim=-1)
    return A


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT-like layer using additive attention on node features.
    """
    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.dk = d_model // heads
        assert d_model % heads == 0, "d_model must be divisible by heads"
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B,N,D], adj: [B,N,N] row-stochastic (optional). If provided, will weight attention logits.
        """
        b, n, d = x.shape
        q = self.q_proj(x).view(b, n, self.heads, self.dk).transpose(1, 2)  # [B,H,N,dk]
        k = self.k_proj(x).view(b, n, self.heads, self.dk).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.heads, self.dk).transpose(1, 2)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.dk ** 0.5)  # [B,H,N,N]
        if adj is not None:
            attn_logits = attn_logits + torch.log(adj.unsqueeze(1).clamp_min(1e-8))
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B,H,N,dk]
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        out = self.o_proj(out)
        return self.norm(x + self.dropout(out))


class GraphBlock(nn.Module):
    def __init__(self, d_model: int, layers: int = 2, heads: int = 4, dropout: float = 0.1, topk: int = 5):
        super().__init__()
        self.topk = topk
        self.layers = nn.ModuleList([GraphAttentionLayer(d_model, heads, dropout) for _ in range(layers)])

    def forward(self, x_sym: torch.Tensor) -> torch.Tensor:
        """
        x_sym: [B, N, D]
        """
        adj = cosine_adj(x_sym, topk=self.topk)
        h = x_sym
        for gat in self.layers:
            h = gat(h, adj)
        return h

