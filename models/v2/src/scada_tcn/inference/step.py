from __future__ import annotations

from typing import Optional

import torch

from ..contracts import InferBatch, InferStepOutputs, ModelOutputs
from ..registry import FeatureRegistry, FlagRegistry
from ..utils.shapes import assert_contract_infer_batch
from ..modeling.multitask_tcn import MultiTaskTCN
from ..training.masking import apply_mask, build_model_input, sample_mask
from .scoring import aggregate_score, compute_E_rec, per_feature_contribution, top_contributors


def infer_step(
    model: MultiTaskTCN,
    batch: InferBatch,
    cfg,
    feature_reg: FeatureRegistry,
    flag_reg: FlagRegistry,
    generator: Optional[torch.Generator] = None,
) -> InferStepOutputs:
    F = len(feature_reg.feature_names)
    R = len(flag_reg.flag_names)
    assert_contract_infer_batch(batch, F=F, R=R)

    X = batch.X
    M_miss = batch.M_miss
    Flags = batch.Flags

    # ---- clean output stream ----
    M_ones = torch.ones_like(M_miss)
    X_in_I = build_model_input(X, M_miss, M_ones, Flags)
    out_I = model(
        X_in_I,
        return_recon=False,
        return_forecast=bool(cfg.model.heads.get("forecast", {}).get("enabled", True)),
        return_fault=bool(cfg.model.heads.get("fault", {}).get("enabled", False)),
    )
    outputs = ModelOutputs(X_hat=None, Y_hat=out_I.Y_hat, p_fault=out_I.p_fault)

    # ---- scoring stream (mask-at-test) ----
    M_score = sample_mask(M_miss, p_mask=float(cfg.infer.p_score), generator=generator)
    X_corrupt_S = apply_mask(X, M_score, float(cfg.features.fill_value_c))
    X_in_S = build_model_input(X_corrupt_S, M_miss, M_score, Flags)

    out_S = model(X_in_S, return_recon=True, return_forecast=False, return_fault=False)
    if out_S.X_hat is None:
        raise ValueError("Recon scoring requires X_hat but got None")

    E_rec = compute_E_rec(X, out_S.X_hat, M_score)
    s_now = aggregate_score(E_rec, cfg.infer, M_score=M_score)
    contrib = per_feature_contribution(E_rec, M_score)
    top_idx, _top_vals = top_contributors(contrib, int(cfg.infer.topn_features))

    return InferStepOutputs(
        outputs=outputs,
        s_now=s_now,
        feat_contrib=contrib,
        top_idx=top_idx,
        debug={
            "masked_frac": float(((M_score < 0.5) & (M_miss > 0.5)).float().mean().item()),
        },
    )
