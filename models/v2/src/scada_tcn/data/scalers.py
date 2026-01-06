# src/scada_tcn/data/scalers.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..utils.io import load_json, save_json


@dataclass
class RobustScalerParams:
    median: dict[str, float]
    iqr: dict[str, float]
    eps: float
    clip_iqr: Optional[float]
    feature_names: list[str]
    fitted_on: dict[str, Any]


def fit_robust_scaler(
    df: pd.DataFrame,
    feature_cols: list[str],
    eps: float = 1e-6,
    clip_iqr: Optional[float] = None,
    fitted_on: Optional[dict[str, Any]] = None,
) -> RobustScalerParams:
    med: dict[str, float] = {}
    iqr: dict[str, float] = {}
    for c in feature_cols:
        s = df[c].dropna()
        if s.empty:
            med[c] = 0.0
            iqr[c] = 1.0
            continue
        q25 = float(s.quantile(0.25))
        q75 = float(s.quantile(0.75))
        med[c] = float(s.median())
        iqr[c] = float(max(q75 - q25, eps))
    return RobustScalerParams(
        median=med,
        iqr=iqr,
        eps=float(eps),
        clip_iqr=None if clip_iqr is None else float(clip_iqr),
        feature_names=list(feature_cols),
        fitted_on=fitted_on or {},
    )


def transform_robust(df: pd.DataFrame, params: RobustScalerParams) -> pd.DataFrame:
    missing = [c for c in params.feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Scaler columns missing from df: {missing}")

    out = df.copy()
    for c in params.feature_names:
        denom = max(float(params.iqr[c]), float(params.eps))
        out[c] = (out[c] - float(params.median[c])) / denom
        if params.clip_iqr is not None:
            out[c] = out[c].clip(lower=-params.clip_iqr, upper=params.clip_iqr)
    return out


def save_scaler(params: RobustScalerParams, path: str) -> None:
    payload = {"schema_version": 1, "params": asdict(params)}
    save_json(payload, path)


def load_scaler(path: str) -> RobustScalerParams:
    payload = load_json(path)
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported scaler schema_version: {payload.get('schema_version')}")
    return RobustScalerParams(**payload["params"])
