# src/scada_tcn/contracts.py
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional, TypeVar, Union

import torch

T = TypeVar("T")


@dataclass
class BatchBase:
    X: torch.Tensor  # (B,L,F) float32
    M_miss: torch.Tensor  # (B,L,F) float32 0/1
    Flags: torch.Tensor  # (B,L,R) float32 0/1
    turbine_ids: Optional[list[str]] = None
    times: Optional[torch.Tensor] = None  # (B,L) or (B,)
    meta: Optional[dict[str, Any]] = None


@dataclass
class TrainBatch(BatchBase):
    Y_true: Optional[torch.Tensor] = None  # (B,K,F)
    y: Optional[torch.Tensor] = None  # (B,) int64 or (B,C) float32
    has_label: Optional[torch.Tensor] = None  # (B,) float/bool


@dataclass
class InferBatch(BatchBase):
    pass


@dataclass
class ModelOutputs:
    X_hat: Optional[torch.Tensor] = None  # (B,L,F)
    Y_hat: Optional[torch.Tensor] = None  # (B,K,F)
    p_fault: Optional[torch.Tensor] = None  # (B,C) logits


@dataclass
class TrainStepOutputs:
    loss: torch.Tensor
    losses: dict[str, float]
    metrics: dict[str, float]
    debug: dict[str, Any]


@dataclass
class InferStepOutputs:
    outputs: ModelOutputs
    s_now: torch.Tensor  # (B,)
    feat_contrib: torch.Tensor  # (B,F)
    top_idx: torch.Tensor  # (B,topn)
    debug: dict[str, Any]


def _to_device_tensor(t: Optional[torch.Tensor], device: Union[str, torch.device]) -> Optional[torch.Tensor]:
    return None if t is None else t.to(device)


def to_device(obj: T, device: Union[str, torch.device]) -> T:
    if isinstance(obj, (TrainBatch, InferBatch)):
        return replace(
            obj,
            X=obj.X.to(device),
            M_miss=obj.M_miss.to(device),
            Flags=obj.Flags.to(device),
            times=_to_device_tensor(obj.times, device),
            Y_true=_to_device_tensor(getattr(obj, "Y_true", None), device),
            y=_to_device_tensor(getattr(obj, "y", None), device),
            has_label=_to_device_tensor(getattr(obj, "has_label", None), device),
        )
    if isinstance(obj, ModelOutputs):
        return replace(
            obj,
            X_hat=_to_device_tensor(obj.X_hat, device),
            Y_hat=_to_device_tensor(obj.Y_hat, device),
            p_fault=_to_device_tensor(obj.p_fault, device),
        )
    if isinstance(obj, (TrainStepOutputs, InferStepOutputs)):
        raise TypeError("Move the batch and model outputs; step outputs should be detached/CPU-cast separately.")
    return obj


def _detach_cpu(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if t is None else t.detach().cpu()


def detach_to_cpu(obj: T) -> T:
    if isinstance(obj, (TrainBatch, InferBatch)):
        return replace(
            obj,
            X=_detach_cpu(obj.X),
            M_miss=_detach_cpu(obj.M_miss),
            Flags=_detach_cpu(obj.Flags),
            times=_detach_cpu(obj.times),
            Y_true=_detach_cpu(getattr(obj, "Y_true", None)),
            y=_detach_cpu(getattr(obj, "y", None)),
            has_label=_detach_cpu(getattr(obj, "has_label", None)),
        )
    if isinstance(obj, ModelOutputs):
        return replace(
            obj,
            X_hat=_detach_cpu(obj.X_hat),
            Y_hat=_detach_cpu(obj.Y_hat),
            p_fault=_detach_cpu(obj.p_fault),
        )
    return obj
