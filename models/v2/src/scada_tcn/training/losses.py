from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..registry import FlagRegistry


def huber_elementwise(err: torch.Tensor, delta: float) -> torch.Tensor:
    d = float(delta)
    abs_e = err.abs()
    quad = torch.minimum(abs_e, torch.tensor(d, device=err.device, dtype=err.dtype))
    lin = abs_e - quad
    return 0.5 * quad * quad + d * lin


def huber_loss(y_true: torch.Tensor, y_pred: torch.Tensor, delta: float, reduction: str = "mean") -> torch.Tensor:
    err = y_true - y_pred
    el = huber_elementwise(err, delta=delta)
    if reduction == "mean":
        return el.mean()
    if reduction == "sum":
        return el.sum()
    raise ValueError(f"Unknown reduction: {reduction}")


def masked_huber_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask: torch.Tensor,
    delta: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    err = y_true - y_pred
    el = huber_elementwise(err, delta=delta)
    num = (el * mask).sum()
    den = mask.sum().clamp_min(float(eps))
    return num / den


def forecast_loss(Y_true: torch.Tensor, Y_hat: torch.Tensor, delta: float) -> torch.Tensor:
    return huber_loss(Y_true, Y_hat, delta=delta, reduction="mean")


def fault_loss(logits: torch.Tensor, y: torch.Tensor, multilabel: bool) -> torch.Tensor:
    if multilabel:
        # y expected float32 (B,C)
        return F.binary_cross_entropy_with_logits(logits, y.float(), reduction="mean")
    # y expected int64 (B,)
    return F.cross_entropy(logits, y.long(), reduction="mean")


def apply_regime_weight(
    loss: torch.Tensor,
    Flags: torch.Tensor,
    enabled: bool,
    weight_map: dict[str, float],
    flag_reg: Optional[FlagRegistry] = None,
) -> torch.Tensor:
    """
    Minimal regime weighting:
      - compute per-sample weight based on active flags at last timestep
      - conservative: use MIN weight among active flags (so any active regime downweights)
    """
    if not enabled:
        return loss

    if Flags.ndim != 3:
        return loss

    B = Flags.shape[0]
    last = Flags[:, -1, :]  # (B,R)
    w = torch.ones((B,), device=Flags.device, dtype=Flags.dtype)

    if flag_reg is None:
        # no registry: do nothing
        return loss

    name_to_idx = {n: i for i, n in enumerate(flag_reg.flag_names)}

    for fname, ww in weight_map.items():
        if fname not in name_to_idx:
            continue
        idx = name_to_idx[fname]
        active = last[:, idx] > 0.5
        w = torch.where(active, torch.minimum(w, torch.tensor(float(ww), device=w.device, dtype=w.dtype)), w)

    return loss * w.mean()
