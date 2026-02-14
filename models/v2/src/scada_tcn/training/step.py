from __future__ import annotations

from typing import Optional

import torch

from ..contracts import TrainBatch, TrainStepOutputs
from ..registry import FeatureRegistry, FlagRegistry
from ..utils.shapes import assert_contract_train_batch
from ..modeling.multitask_tcn import MultiTaskTCN
from .losses import apply_regime_weight, fault_loss, forecast_loss, masked_huber_loss
from .masking import apply_mask, build_model_input, mask_stats, sample_mask


def _align_horizon_targets(y_h: torch.Tensor, logits_h: torch.Tensor) -> torch.Tensor:
    """Align horizon targets to logits shape when include_normal config drifts."""
    if y_h.ndim != 3 or logits_h.ndim != 3:
        return y_h

    b_y, h_y, c_y = y_h.shape
    b_l, h_l, c_l = logits_h.shape

    b = min(b_y, b_l)
    h = min(h_y, h_l)
    y = y_h[:b, :h, :]

    if c_y == c_l:
        return y

    # Common mismatch: targets include "normal" but logits exclude it (or vice versa).
    if c_y == c_l + 1:
        return y[..., 1:]
    if c_l == c_y + 1:
        pad = torch.zeros((*y.shape[:2], 1), dtype=y.dtype, device=y.device)
        return torch.cat([pad, y], dim=-1)

    if c_y > c_l:
        return y[..., :c_l]

    pad = torch.zeros((*y.shape[:2], c_l - c_y), dtype=y.dtype, device=y.device)
    return torch.cat([y, pad], dim=-1)



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
    L_fault_horizons = torch.tensor(0.0, device=X.device)
    fault_enabled = bool(cfg.model.heads.get("fault", {}).get("enabled", False))
    if fault_enabled and out_B.p_fault is not None and batch.y is not None:
        multilabel = bool(cfg.model.heads.get("fault", {}).get("multilabel", False))

        if batch.has_label is not None:
            mask = batch.has_label > 0.5
            if mask.any():
                L_fault = fault_loss(out_B.p_fault[mask], batch.y[mask], multilabel=multilabel)
                if out_B.p_fault_horizons is not None and batch.y_horizons is not None:
                    y_h = _align_horizon_targets(batch.y_horizons[mask], out_B.p_fault_horizons[mask])
                    L_fault_horizons = fault_loss(out_B.p_fault_horizons[mask], y_h, multilabel=True)
        else:
            L_fault = fault_loss(out_B.p_fault, batch.y, multilabel=multilabel)
            if out_B.p_fault_horizons is not None and batch.y_horizons is not None:
                y_h = _align_horizon_targets(batch.y_horizons, out_B.p_fault_horizons)
                L_fault_horizons = fault_loss(out_B.p_fault_horizons, y_h, multilabel=True)

    # ---- Optional regime weighting ----
    rw_cfg = cfg.train.loss.get("regime_weighting", {})
    rw_enabled = bool(rw_cfg.get("enabled", False))
    weight_map = dict(rw_cfg.get("weight_map", {}))

    L_rec = apply_regime_weight(L_rec, Flags, enabled=rw_enabled, weight_map=weight_map, flag_reg=flag_reg)
    L_pred = apply_regime_weight(L_pred, Flags, enabled=rw_enabled, weight_map=weight_map, flag_reg=flag_reg)
    L_fault = apply_regime_weight(L_fault, Flags, enabled=rw_enabled, weight_map=weight_map, flag_reg=flag_reg)
    L_fault_horizons = apply_regime_weight(L_fault_horizons, Flags, enabled=rw_enabled, weight_map=weight_map, flag_reg=flag_reg)

    lambdas = cfg.train.loss.get("lambdas", {})
    lam_r = float(lambdas.get("recon", 1.0))
    lam_p = float(lambdas.get("pred", 1.0))
    lam_f = float(lambdas.get("fault", 1.0))
    lam_fh = float(lambdas.get("fault_horizons", 1.0))

    loss = lam_r * L_rec + lam_p * L_pred + lam_f * L_fault + lam_fh * L_fault_horizons

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    debug = {}
    debug.update(mask_stats(M_mask_A, M_miss))
    if batch.has_label is not None:
        debug["has_label_frac"] = float((batch.has_label > 0.5).float().mean().detach().cpu().item())
    if batch.y is not None:
        debug["y_shape"] = list(batch.y.shape)

    metrics = {}
    if fault_enabled and out_B.p_fault is not None and batch.y is not None:
        with torch.no_grad():
            if batch.has_label is not None:
                m = batch.has_label > 0.5
            else:
                m = torch.ones(batch.y.shape[0], dtype=torch.bool, device=batch.y.device)
            labeled_n = int(m.sum().detach().cpu().item())
            metrics["fault_labeled_count"] = float(labeled_n)
            if labeled_n > 0 and not bool(cfg.model.heads.get("fault", {}).get("multilabel", False)):
                logits_l = out_B.p_fault[m]
                y_l = batch.y[m].to(torch.long)
                pred_l = torch.argmax(logits_l, dim=1)
                metrics["fault_pred_pos_rate"] = float((pred_l > 0).float().mean().detach().cpu().item())
                metrics["fault_true_pos_rate"] = float((y_l > 0).float().mean().detach().cpu().item())

                pos_mask = y_l > 0
                metrics["fault_batch_has_positive"] = float(pos_mask.any().detach().cpu().item())
                neg_mask = y_l == 0
                has_pos = bool(pos_mask.any())
                has_neg = bool(neg_mask.any())

                # NOTE: single-class batches (all-normal or all-fault) can trivially
                # produce 1.0 accuracy values that are misleading during training.
                # Keep an explicit flag and only compute "overall"/"balanced" metrics
                # when both classes are present in the batch.
                metrics["fault_batch_has_both_classes"] = float(has_pos and has_neg)
                if has_pos and has_neg:
                    metrics["fault_acc_overall"] = float((pred_l == y_l).float().mean().detach().cpu().item())
                else:
                    metrics["fault_acc_overall"] = 0.0

                if has_neg:
                    metrics["fault_tnr_normal"] = float((pred_l[neg_mask] == 0).float().mean().detach().cpu().item())
                if has_pos:
                    metrics["fault_recall_non_normal"] = float((pred_l[pos_mask] == y_l[pos_mask]).float().mean().detach().cpu().item())
                    # "fault_acc" intentionally focuses on non-normal recall whenever a
                    # batch contains fault examples. This avoids misleading 1.0 readings
                    # on all-normal batches.
                    metrics["fault_acc"] = float((pred_l[pos_mask] == y_l[pos_mask]).float().mean().detach().cpu().item())

                tnr = metrics.get("fault_tnr_normal", 0.0)
                tpr = metrics.get("fault_recall_non_normal", 0.0)
                if has_pos and has_neg:
                    metrics["fault_balanced_acc"] = float(0.5 * (tnr + tpr))
                else:
                    metrics["fault_balanced_acc"] = 0.0

                if has_pos:
                    metrics["fault_acc_non_normal"] = float((pred_l[pos_mask] == y_l[pos_mask]).float().mean().detach().cpu().item())

    return TrainStepOutputs(
        loss=loss,
        losses={
            "recon": float(L_rec.detach().cpu().item()),
            "pred": float(L_pred.detach().cpu().item()),
            "fault": float(L_fault.detach().cpu().item()),
            "fault_horizons": float(L_fault_horizons.detach().cpu().item()),
            "total": float(loss.detach().cpu().item()),
        },
        metrics=metrics,
        debug=debug,
    )
