from __future__ import annotations

from typing import Optional

import torch


def sample_mask(M_miss: torch.Tensor, p_mask: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    M_miss: (B,L,F) 1 present, 0 missing
    returns M_mask: (B,L,F) 1 keep, 0 hide
    Constraint: missing values cannot be masked (forced keep=1).
    """
    if not (0.0 <= p_mask <= 1.0):
        raise ValueError(f"p_mask must be in [0,1], got {p_mask}")

    keep_prob = 1.0 - float(p_mask)
    rand = torch.rand(M_miss.shape, device=M_miss.device, generator=generator)
    M_mask = (rand < keep_prob).to(torch.float32)

    # force keep for missing positions
    M_mask = torch.where(M_miss > 0.5, M_mask, torch.ones_like(M_mask))
    return M_mask


def apply_mask(X: torch.Tensor, M_mask: torch.Tensor, fill_value: float) -> torch.Tensor:
    return M_mask * X + (1.0 - M_mask) * float(fill_value)


def build_model_input(X_corrupt: torch.Tensor, M_miss: torch.Tensor, M_mask: torch.Tensor, Flags: torch.Tensor) -> torch.Tensor:
    # X_in = [X_corrupt, M_miss, M_mask, Flags]
    return torch.cat([X_corrupt, M_miss, M_mask, Flags], dim=-1).to(torch.float32)


def mask_stats(M_mask: torch.Tensor, M_miss: torch.Tensor) -> dict[str, float]:
    present = (M_miss > 0.5).float()
    masked = ((M_mask < 0.5) & (M_miss > 0.5)).float()

    denom = float(present.sum().item() + 1e-6)
    frac_masked = float(masked.sum().item() / denom)
    return {
        "present_count": float(present.sum().item()),
        "masked_count": float(masked.sum().item()),
        "frac_masked_among_present": frac_masked,
    }
