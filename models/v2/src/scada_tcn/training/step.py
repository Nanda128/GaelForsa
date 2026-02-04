from __future__ import annotations

from typing import Optional

import torch

from ..contracts import TrainBatch, TrainStepOutputs
from ..registry import FeatureRegistry, FlagRegistry
from ..utils.shapes import assert_contract_train_batch
from ..modeling.multitask_tcn import MultiTaskTCN
from .losses import apply_regime_weight, fault_loss, forecast_loss, masked_huber_loss
from .masking import apply_mask, build_model_input, mask_stats, sample_mask


def train_step(
    model: MultiTaskTCN,
    batch: TrainBatch,
    cfg,
    feature_reg: FeatureRegistry,
    flag_reg: FlagRegistry,
    optimizer: torch.optim.Optimizer,
    generator: Optional[torch.Generator] = None,
) -> TrainStepOutputs:
    F = len(feature_reg.feature_names)
    R = len(flag_reg.flag_names)
    K = int(cfg.data.windowing.get("K", 0))
    assert_contract_train_batch(batch, F=F, R=R, K=K if K > 0 else None)

    X = batch.X
    M_miss = batch.M_miss
    Flags = batch.Flags

    # ---- Stream A: masked reconstruction ----
    M_mask_A = sample_mask(M_miss, p_mask=float(cfg.train.p_mask), generator=generator)
    X_corrupt_A = apply_mask(X, M_mask_A, float(cfg.features.fill_value_c))
    X_in_A = build_model_input(X_corrupt_A, M_miss, M_mask_A, Flags)

    out_A = model(X_in_A, return_recon=True, return_forecast=False, return_fault=False)
    if out_A.X_hat is None:
        raise ValueError("Recon path expected X_hat but got None")

    recon_target_mask = (1.0 - M_mask_A) * M_miss  # masked-and-present only
    delta = float(cfg.train.loss.get("huber_delta", 1.0))
    L_rec = masked_huber_loss(X, out_A.X_hat, recon_target_mask, delta=delta)

    # ---- Stream B: clean forecast + fault ----
    M_mask_B = torch.ones_like(M_miss)
    X_in_B = build_model_input(X, M_miss, M_mask_B, Flags)
    out_B = model(
        X_in_B,
        return_recon=False,
        return_forecast=bool(cfg.model.heads.get("forecast", {}).get("enabled", True)),
        return_fault=bool(cfg.model.heads.get("fault", {}).get("enabled", False)),
    )

    L_pred = torch.tensor(0.0, device=X.device)
    if bool(cfg.model.heads.get("forecast", {}).get("enabled", True)):
        if batch.Y_true is None or out_B.Y_hat is None:
            raise ValueError("Forecast enabled but Y_true or Y_hat missing")
        L_pred = forecast_loss(batch.Y_true, out_B.Y_hat, delta=delta)

    L_fault = torch.tensor(0.0, device=X.device)
    fault_enabled = bool(cfg.model.heads.get("fault", {}).get("enabled", False))
    if fault_enabled and out_B.p_fault is not None and batch.y is not None:
        multilabel = bool(cfg.model.heads.get("fault", {}).get("multilabel", False))

        if batch.has_label is not None:
            mask = batch.has_label > 0.5
            if mask.any():
                L_fault = fault_loss(out_B.p_fault[mask], batch.y[mask], multilabel=multilabel)
        else:
            L_fault = fault_loss(out_B.p_fault, batch.y, multilabel=multilabel)

    # ---- Optional regime weighting ----
    rw_cfg = cfg.train.loss.get("regime_weighting", {})
    rw_enabled = bool(rw_cfg.get("enabled", False))
    weight_map = dict(rw_cfg.get("weight_map", {}))

    L_rec = apply_regime_weight(L_rec, Flags, enabled=rw_enabled, weight_map=weight_map, flag_reg=flag_reg)
    L_pred = apply_regime_weight(L_pred, Flags, enabled=rw_enabled, weight_map=weight_map, flag_reg=flag_reg)
    L_fault = apply_regime_weight(L_fault, Flags, enabled=rw_enabled, weight_map=weight_map, flag_reg=flag_reg)

    lambdas = cfg.train.loss.get("lambdas", {})
    lam_r = float(lambdas.get("recon", 1.0))
    lam_p = float(lambdas.get("pred", 1.0))
    lam_f = float(lambdas.get("fault", 1.0))

    loss = lam_r * L_rec + lam_p * L_pred + lam_f * L_fault

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    debug = {}
    debug.update(mask_stats(M_mask_A, M_miss))

    return TrainStepOutputs(
        loss=loss,
        losses={
            "recon": float(L_rec.detach().cpu().item()),
            "pred": float(L_pred.detach().cpu().item()),
            "fault": float(L_fault.detach().cpu().item()),
            "total": float(loss.detach().cpu().item()),
        },
        metrics={},
        debug=debug,
    )
