# src/scada_tcn/utils/shapes.py
from __future__ import annotations

from typing import Optional, Sequence

import torch

from ..contracts import InferBatch, ModelOutputs, TrainBatch


def assert_shape(t: torch.Tensor, shape: Sequence[Optional[int]], name: str) -> None:
    if t.ndim != len(shape):
        raise ValueError(f"{name}.ndim expected {len(shape)} got {t.ndim} with shape={tuple(t.shape)}")
    for i, exp in enumerate(shape):
        if exp is None:
            continue
        if int(t.shape[i]) != int(exp):
            raise ValueError(f"{name}.shape[{i}] expected {exp} got {t.shape[i]} full={tuple(t.shape)}")


def assert_finite(t: torch.Tensor, name: str) -> None:
    if not torch.isfinite(t).all().item():
        bad = (~torch.isfinite(t)).sum().item()
        raise ValueError(f"{name} has non-finite values: count={bad}")


def assert_binary_mask(t: torch.Tensor, name: str, tol: float = 1e-6) -> None:
    if t.numel() == 0:
        return
    mn = float(t.min().item())
    mx = float(t.max().item())
    if mn < -tol or mx > 1.0 + tol:
        raise ValueError(f"{name} outside [0,1]: min={mn}, max={mx}")
    # Loose check: values close to 0/1
    frac_mid = float(((t > tol) & (t < 1.0 - tol)).float().mean().item())
    if frac_mid > 1e-3:
        raise ValueError(f"{name} not binary-ish: fraction strictly between 0 and 1 = {frac_mid:.6f}")


def assert_contract_train_batch(batch: TrainBatch, F: int, R: int, K: Optional[int]) -> None:
    B, L, F_ = batch.X.shape
    assert_shape(batch.X, (B, L, F), "batch.X")
    assert_shape(batch.M_miss, (B, L, F), "batch.M_miss")
    assert_shape(batch.Flags, (B, L, R), "batch.Flags")
    assert_binary_mask(batch.M_miss, "batch.M_miss")
    assert_finite(batch.X, "batch.X")
    assert_finite(batch.M_miss, "batch.M_miss")
    assert_finite(batch.Flags, "batch.Flags")
    if K is not None and K > 0:
        if batch.Y_true is None:
            raise ValueError("K>0 but batch.Y_true is None")
        assert_shape(batch.Y_true, (B, K, F), "batch.Y_true")


def assert_contract_infer_batch(batch: InferBatch, F: int, R: int) -> None:
    B, L, _ = batch.X.shape
    assert_shape(batch.X, (B, L, F), "batch.X")
    assert_shape(batch.M_miss, (B, L, F), "batch.M_miss")
    assert_shape(batch.Flags, (B, L, R), "batch.Flags")
    assert_binary_mask(batch.M_miss, "batch.M_miss")
    assert_finite(batch.X, "batch.X")
    assert_finite(batch.M_miss, "batch.M_miss")
    assert_finite(batch.Flags, "batch.Flags")


def assert_contract_model_outputs(
    out: ModelOutputs,
    B: int,
    L: int,
    F: int,
    K: int,
    C: int,
    enabled: dict[str, bool],
) -> None:
    if enabled.get("recon", False):
        if out.X_hat is None:
            raise ValueError("recon enabled but out.X_hat is None")
        assert_shape(out.X_hat, (B, L, F), "out.X_hat")
    if enabled.get("forecast", False):
        if out.Y_hat is None:
            raise ValueError("forecast enabled but out.Y_hat is None")
        assert_shape(out.Y_hat, (B, K, F), "out.Y_hat")
    if enabled.get("fault", False):
        if out.p_fault is None:
            raise ValueError("fault enabled but out.p_fault is None")
        assert_shape(out.p_fault, (B, C), "out.p_fault")
