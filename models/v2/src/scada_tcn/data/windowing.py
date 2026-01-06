# src/scada_tcn/data/windowing.py
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ..config_schema import DataConfig


def build_window_index(
    df_X: pd.DataFrame,
    df_M_miss: pd.DataFrame,
    df_Flags: pd.DataFrame,
    cfg: DataConfig,
    labels_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if not (df_X.index.equals(df_M_miss.index) and df_X.index.equals(df_Flags.index)):
        raise ValueError("df_X, df_M_miss, df_Flags must share identical MultiIndex")

    L = int(cfg.windowing["L"])
    K = int(cfg.windowing.get("K", 0))
    stride = int(cfg.windowing.get("stride", 1))

    rows: list[dict[str, Any]] = []
    for turbine_id, g in df_X.groupby(level=0, sort=False):
        # positions are within this turbine's timeline
        idx = g.index.get_level_values(1)
        T = len(g)
        max_start = T - (L + K) + 1
        if max_start <= 0:
            continue
        for start in range(0, max_start, stride):
            row = {
                "turbine_id": turbine_id,
                "start_pos": int(start),
                "end_pos": int(start + L),
                "target_start_pos": int(start + L),
                "target_end_pos": int(start + L + K),
                "start_time": idx[start],
                "has_future": bool(K > 0),
            }
            # label plumbing is intentionally minimal here; add later when labels are enabled.
            rows.append(row)

    df_index = pd.DataFrame(rows)
    if df_index.empty:
        raise ValueError("No windows created. Check L/K/stride and available data per turbine.")
    return df_index


def split_window_index(df_index: pd.DataFrame, cfg: DataConfig) -> dict[str, pd.DataFrame]:
    method = str(cfg.splits.get("method", "time"))
    if method == "time":
        tr0, tr1 = cfg.splits["train_range"]
        va0, va1 = cfg.splits["val_range"]
        te0, te1 = cfg.splits["test_range"]
        tr0 = pd.to_datetime(tr0); tr1 = pd.to_datetime(tr1)
        va0 = pd.to_datetime(va0); va1 = pd.to_datetime(va1)
        te0 = pd.to_datetime(te0); te1 = pd.to_datetime(te1)

        t = pd.to_datetime(df_index["start_time"])
        train = df_index[(t >= tr0) & (t <= tr1)].reset_index(drop=True)
        val = df_index[(t >= va0) & (t <= va1)].reset_index(drop=True)
        test = df_index[(t >= te0) & (t <= te1)].reset_index(drop=True)
        return {"train": train, "val": val, "test": test}

    if method == "turbine":
        # simplest deterministic split by sorted turbine id
        tids = sorted(df_index["turbine_id"].unique().tolist())
        n = len(tids)
        ntr = int(0.7 * n)
        nva = int(0.15 * n)
        train_set = set(tids[:ntr])
        val_set = set(tids[ntr : ntr + nva])
        test_set = set(tids[ntr + nva :])
        return {
            "train": df_index[df_index["turbine_id"].isin(train_set)].reset_index(drop=True),
            "val": df_index[df_index["turbine_id"].isin(val_set)].reset_index(drop=True),
            "test": df_index[df_index["turbine_id"].isin(test_set)].reset_index(drop=True),
        }

    raise ValueError(f"Unsupported split method: {method}")


def slice_window(arr, start: int, length: int):
    return arr[start : start + length]
