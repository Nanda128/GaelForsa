from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ReconHead(nn.Module):
    def __init__(self, D: int, F: int) -> None:
        super().__init__()
        self.proj = nn.Linear(int(D), int(F))

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: (B,L,D)
        return self.proj(H)  # (B,L,F)


class ForecastHead(nn.Module):
    def __init__(self, D: int, K: int, F: int, summary: str = "last", hidden: Optional[int] = None) -> None:
        super().__init__()
        self.K = int(K)
        self.F = int(F)
        self.summary = str(summary)

        if hidden is None:
            self.mlp = nn.Linear(int(D), self.K * self.F)
        else:
            h = int(hidden)
            self.mlp = nn.Sequential(nn.Linear(int(D), h), nn.ReLU(), nn.Linear(h, self.K * self.F))

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: (B,L,D)
        if self.summary == "last":
            h = H[:, -1, :]
        else:
            raise ValueError(f"Unsupported summary: {self.summary}")

        out = self.mlp(h)  # (B, K*F)
        return out.view(out.shape[0], self.K, self.F)


class FaultHead(nn.Module):
    def __init__(
        self,
        D: int,
        C: int,
        pooling: str = "mean",
        multilabel: bool = False,
        hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.pooling = str(pooling)
        self.multilabel = bool(multilabel)

        if hidden is None:
            self.mlp = nn.Linear(int(D), int(C))
        else:
            h = int(hidden)
            self.mlp = nn.Sequential(nn.Linear(int(D), h), nn.ReLU(), nn.Linear(h, int(C)))

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: (B,L,D)
        if self.pooling == "mean":
            h = H.mean(dim=1)
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")
        return self.mlp(h)  # logits (B,C)
