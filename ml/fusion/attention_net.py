"""PyTorch network for attention fusion (imported only when torch is installed)."""

from __future__ import annotations

import torch
from torch import nn


class ModalityAttentionNet(nn.Module):
    """Per-modality encoders combined by availability-masked, input-dependent attention."""

    def __init__(self, dims: list[int], hidden: int, dropout: float) -> None:
        super().__init__()
        self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)) for d in dims])
        self.score = nn.Linear(hidden, 1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, blocks: list[torch.Tensor], mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.stack([enc(b) for enc, b in zip(self.encoders, blocks, strict=True)], dim=1)  # (B, M, H)
        scores = self.score(torch.tanh(h)).squeeze(-1).masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=1)
        fused = (attn.unsqueeze(-1) * h).sum(1)
        return self.head(fused).squeeze(-1), attn
