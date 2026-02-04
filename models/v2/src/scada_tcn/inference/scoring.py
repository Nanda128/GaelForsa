from __future__ import annotations

from typing import Tuple

import torch


def compute_E_rec(X: torch.Tensor, X_hat: torch.Tensor, M_score: torch.Tensor) -> torch.Tensor:
    return ((1.0 - M_score) * (X - X_hat)).abs()


def aggregate_score(E_rec: torch.Tensor, cfg_infer, M_score: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    name = str(cfg_infer.aggregation.get("name", "masked_mean"))

    if name == "masked_mean":
        if M_score is None:
            # fallback: mean over all
            return E_rec.mean(dim=(1, 2))
        count = (1.0 - M_score).sum(dim=(1, 2)).clamp_min(eps)
        s = E_rec.sum(dim=(1, 2)) / count
        return s

    raise ValueError(f"Unknown aggregation: {name}")


def per_feature_contribution(E_rec: torch.Tensor, M_score: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # contrib[b,f] = mean over masked timesteps for feature f
    mask = (1.0 - M_score)
    num = (E_rec * mask).sum(dim=1)  # (B,F)
    den = mask.sum(dim=1).clamp_min(eps)  # (B,F)
    return num / den


def top_contributors(contrib: torch.Tensor, topn: int) -> Tuple[torch.Tensor, torch.Tensor]:
    vals, idx = torch.topk(contrib, k=int(topn), dim=-1, largest=True, sorted=True)
    return idx, vals
