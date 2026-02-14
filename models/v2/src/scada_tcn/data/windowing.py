from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ..config_schema import DataConfig


def _resolve_ratios(splits_cfg: dict[str, Any]) -> tuple[float, float, float]:
    ratios = splits_cfg.get("ratios", [0.7, 0.15, 0.15])
    if len(ratios) != 3:
        raise ValueError(f"splits.ratios must have 3 values, got {ratios}")
    tr, va, te = [float(x) for x in ratios]
    if min(tr, va, te) < 0:
        raise ValueError(f"splits.ratios must be non-negative, got {ratios}")
    s = tr + va + te
    if s <= 0:
        raise ValueError(f"splits.ratios sum must be >0, got {ratios}")
    return tr / s, va / s, te / s


def _window_label(labels: np.ndarray, start: int, L: int, K: int, rule: str) -> int:
    input_slice = labels[start : start + L]
    target_start = start + L
    future_slice = labels[target_start : target_start + K] if K > 0 else labels[0:0]

    if rule == "last_timestep":
        seg = input_slice[-1:]
    elif rule == "window_any":
        seg = input_slice
    elif rule == "window_any_future_k":
        seg = future_slice
    elif rule == "window_any_full":
        seg = np.concatenate([input_slice, future_slice], axis=0)
    else:
        seg = input_slice[-1:]

    if seg.size == 0:
        return 0
    seg_i = np.asarray(seg).astype(np.int64).reshape(-1)
    pos = seg_i[seg_i > 0]
    if pos.size == 0:
        return 0

    vals, counts = np.unique(pos, return_counts=True)
    order = np.lexsort((vals, counts))
    return int(vals[order[-1]])


def build_window_index(
    df_X: pd.DataFrame,
    df_M_miss: pd.DataFrame,
    df_Flags: pd.DataFrame,
    cfg: DataConfig,
    labels_df: Optional[pd.Series] = None,
) -> pd.DataFrame:
    if not (df_X.index.equals(df_M_miss.index) and df_X.index.equals(df_Flags.index)):
        raise ValueError("df_X, df_M_miss, df_Flags must share identical MultiIndex")

    L = int(cfg.windowing["L"])
    K = int(cfg.windowing.get("K", 0))
    stride = int(cfg.windowing.get("stride", 1))

    has_labels = labels_df is not None
    rule = str(cfg.labels.get("label_rule", "last_timestep")).strip().lower() if has_labels else "last_timestep"

    rows: list[dict[str, Any]] = []
    for turbine_id, g in df_X.groupby(level=0, sort=False):
        idx = g.index.get_level_values(1)
        T = len(g)
        max_start = T - (L + K) + 1
        if max_start <= 0:
            continue

        labels_np = None
        if has_labels:
            try:
                labels_np = labels_df.xs(turbine_id, level=0).to_numpy(dtype=np.int64)
            except Exception:
                labels_np = None

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
            if labels_np is not None:
                y = _window_label(labels_np, start=start, L=L, K=K, rule=rule)
                row["window_label"] = int(y)
                row["window_has_positive"] = bool(y > 0)
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

    if method in ("time_auto", "time_per_turbine"):
        tr_r, va_r, _te_r = _resolve_ratios(cfg.splits)

        parts = {"train": [], "val": [], "test": []}
        has_window_labels = "window_has_positive" in df_index.columns

        for _, g in df_index.groupby("turbine_id", sort=False):
            gg = g.sort_values("start_time").reset_index(drop=True)
            n = len(gg)
            ntr = int(n * tr_r)
            nva = int(n * va_r)
            nte = max(0, n - ntr - nva)

            if n >= 3:
                ntr = max(1, ntr)
                nva = max(1, nva)
                nte = max(1, n - ntr - nva)
                while ntr + nva + nte > n and ntr > 1:
                    ntr -= 1

            train = gg.iloc[:ntr].copy()
            val = gg.iloc[ntr : ntr + nva].copy()
            test = gg.iloc[ntr + nva : ntr + nva + nte].copy()

            if has_window_labels and bool(gg["window_has_positive"].any()) and not bool(train["window_has_positive"].any()):
                donor_name = None
                donor = None
                if bool(val["window_has_positive"].any()):
                    donor_name, donor = "val", val
                elif bool(test["window_has_positive"].any()):
                    donor_name, donor = "test", test

                if donor is not None and donor_name is not None:
                    pos_idx = int(donor.index[donor["window_has_positive"]].min())
                    moved = donor.loc[[pos_idx]].copy()
                    if donor_name == "val":
                        val = donor.drop(index=pos_idx)
                    else:
                        test = donor.drop(index=pos_idx)
                    train = pd.concat([train, moved], axis=0, ignore_index=True).sort_values("start_time").reset_index(drop=True)

            parts["train"].append(train)
            parts["val"].append(val)
            parts["test"].append(test)

        return {
            k: pd.concat(v, axis=0, ignore_index=True) if v else pd.DataFrame(columns=df_index.columns)
            for k, v in parts.items()
        }

    if method == "turbine":
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
