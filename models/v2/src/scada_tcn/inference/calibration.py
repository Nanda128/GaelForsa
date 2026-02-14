from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CalibrationResult:
    thresholds: dict[str, float]
    method: str
    percentile: float


def _key(turbine_id: str, regime_key: str, per_turbine: bool, per_regime: bool) -> str:
    t = turbine_id if per_turbine else "*"
    r = regime_key if per_regime else "*"
    return f"{t}::{r}"


def fit_percentile_thresholds(
    df: pd.DataFrame,
    *,
    score_col: str = "s_now",
    turbine_col: str = "turbine_id",
    regime_col: str = "regime_key",
    percentile: float = 0.995,
    per_turbine: bool = True,
    per_regime: bool = True,
) -> CalibrationResult:
    if not (0.0 < float(percentile) < 1.0):
        raise ValueError(f"percentile must be in (0,1), got {percentile}")

    if df.empty:
        return CalibrationResult(thresholds={"*::*": float("nan")}, method="percentile", percentile=percentile)

    data = df.copy()
    if turbine_col not in data.columns:
        data[turbine_col] = "unknown"
    if regime_col not in data.columns:
        data[regime_col] = "none"

    thresholds: dict[str, float] = {}
    group_cols = []
    if per_turbine:
        group_cols.append(turbine_col)
    if per_regime:
        group_cols.append(regime_col)

    if group_cols:
        grouped = data.groupby(group_cols, dropna=False)
        for keys, g in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            t = str(keys[0]) if per_turbine else "*"
            r = str(keys[-1]) if per_regime else "*"
            thresholds[_key(t, r, per_turbine, per_regime)] = float(np.quantile(g[score_col].to_numpy(), percentile))
    else:
        thresholds["*::*"] = float(np.quantile(data[score_col].to_numpy(), percentile))

    # global fallback
    thresholds["*::*"] = float(np.quantile(data[score_col].to_numpy(), percentile))
    return CalibrationResult(thresholds=thresholds, method="percentile", percentile=percentile)


def lookup_threshold(
    calib: CalibrationResult,
    *,
    turbine_id: Optional[str],
    regime_key: Optional[str],
    per_turbine: bool,
    per_regime: bool,
) -> float:
    t = turbine_id or "*"
    r = regime_key or "*"
    keys = [
        _key(t, r, per_turbine, per_regime),
        _key(t, "*", per_turbine, False),
        _key("*", r, False, per_regime),
        "*::*",
    ]
    for k in keys:
        if k in calib.thresholds:
            return float(calib.thresholds[k])
    return float("nan")
