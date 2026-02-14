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
        horizon_count: int = 0,
        horizon_classes: int = 0,
    ) -> None:
        super().__init__()
        self.pooling = str(pooling)
        self.multilabel = bool(multilabel)
        self.horizon_count = int(horizon_count)
        self.horizon_classes = int(horizon_classes)

        if hidden is None:
            self.mlp = nn.Linear(int(D), int(C))
            base_dim = int(D)
        else:
            h = int(hidden)
            self.mlp = nn.Sequential(nn.Linear(int(D), h), nn.ReLU(), nn.Linear(h, int(C)))
            base_dim = h

        if self.horizon_count > 0 and self.horizon_classes > 0:
            if hidden is None:
                self.horizon_mlp = nn.Linear(int(D), self.horizon_count * self.horizon_classes)
            else:
                h = int(hidden)
                self.horizon_mlp = nn.Sequential(
                    nn.Linear(int(D), h), nn.ReLU(), nn.Linear(h, self.horizon_count * self.horizon_classes)
                )
        else:
            self.horizon_mlp = None

    def _pool(self, H: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return H.mean(dim=1)
        raise ValueError(f"Unsupported pooling: {self.pooling}")

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # H: (B,L,D)
        h = self._pool(H)
        logits = self.mlp(h)  # (B,C)
        horizon_logits = None
        if self.horizon_mlp is not None:
            out = self.horizon_mlp(h)
            horizon_logits = out.view(out.shape[0], self.horizon_count, self.horizon_classes)
        return logits, horizon_logits
