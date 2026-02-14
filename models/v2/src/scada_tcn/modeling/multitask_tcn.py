from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..contracts import ModelOutputs
from .heads import FaultHead, ForecastHead, ReconHead
from .tcn import TCNBackbone, resolve_dilations


def predict_proba_fault(logits: torch.Tensor, multilabel: bool) -> torch.Tensor:
    if multilabel:
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=-1)


class MultiTaskTCN(nn.Module):
    def __init__(
        self,
        F_in: int,
        F: int,
        R: int,
        K: int,
        C: int,
        cfg_model: Any,
    ) -> None:
        super().__init__()
        model_dict = cfg_model.__dict__ if hasattr(cfg_model, "__dict__") else dict(cfg_model)
        tcn_cfg = dict(model_dict.get("tcn", {}))
        heads_cfg = dict(model_dict.get("heads", {}))

        D = int(tcn_cfg.get("D", 64))
        kernel_size = int(tcn_cfg.get("kernel_size", 3))
        dropout = float(tcn_cfg.get("dropout", 0.0))
        norm = str(tcn_cfg.get("norm", "none"))
        dilations = resolve_dilations(tcn_cfg)

        self.backbone = TCNBackbone(
            F_in=int(F_in),
            D=int(D),
            kernel_size=int(kernel_size),
            dilations=dilations,
            dropout=float(dropout),
            norm=norm,
        )

        self.recon_enabled = bool(heads_cfg.get("recon", {}).get("enabled", True))
        self.forecast_enabled = bool(heads_cfg.get("forecast", {}).get("enabled", True))
        self.fault_enabled = bool(heads_cfg.get("fault", {}).get("enabled", False))
        self.multilabel = bool(heads_cfg.get("fault", {}).get("multilabel", False))

        hz_days = list(heads_cfg.get("fault", {}).get("horizon_days", []))
        include_normal = bool(heads_cfg.get("fault", {}).get("horizon_include_normal", False))
        self.horizon_count = len(hz_days)
        self.horizon_classes = int(C if include_normal else max(0, C - 1))

        self.recon_head = ReconHead(D, F) if self.recon_enabled else None

        if self.forecast_enabled:
            summary = str(heads_cfg.get("forecast", {}).get("summary", "last"))
            hidden = heads_cfg.get("forecast", {}).get("hidden", None)
            self.forecast_head = ForecastHead(D, K, F, summary=summary, hidden=hidden)
        else:
            self.forecast_head = None

        if self.fault_enabled:
            pooling = str(heads_cfg.get("fault", {}).get("pooling", "mean"))
            hidden = heads_cfg.get("fault", {}).get("hidden", None)
            self.fault_head = FaultHead(
                D,
                C,
                pooling=pooling,
                multilabel=self.multilabel,
                hidden=hidden,
                horizon_count=self.horizon_count,
                horizon_classes=self.horizon_classes,
            )
        else:
            self.fault_head = None

    def forward(
        self,
        X_in: torch.Tensor,
        return_recon: bool = True,
        return_forecast: bool = True,
        return_fault: bool = True,
    ) -> ModelOutputs:
        H = self.backbone(X_in)  # (B,L,D)

        X_hat = self.recon_head(H) if (return_recon and self.recon_enabled and self.recon_head is not None) else None
        Y_hat = (
            self.forecast_head(H)
            if (return_forecast and self.forecast_enabled and self.forecast_head is not None)
            else None
        )
        p_fault = None
        p_fault_horizons = None
        if return_fault and self.fault_enabled and self.fault_head is not None:
            p_fault, p_fault_horizons = self.fault_head(H)
        return ModelOutputs(X_hat=X_hat, Y_hat=Y_hat, p_fault=p_fault, p_fault_horizons=p_fault_horizons)
