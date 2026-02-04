from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from ..config_schema import DataConfig, FeatureConfig


def load_raw_tables(raw_dir: str, cfg: DataConfig) -> pd.DataFrame:
    if not os.path.isdir(raw_dir):
        raise ValueError(f"raw_dir not found: {raw_dir}")

    files: list[str] = []
    for fn in os.listdir(raw_dir):
        if fn.endswith(".parquet") or fn.endswith(".pq") or fn.endswith(".csv"):
            files.append(os.path.join(raw_dir, fn))
    if not files:
        raise ValueError(f"No .parquet/.csv files found in {raw_dir}")

    dfs: list[pd.DataFrame] = []
    for p in sorted(files):
        if p.endswith(".csv"):
            df = pd.read_csv(p)
        else:
            df = pd.read_parquet(p)
        dfs.append(df)

    df_raw = pd.concat(dfs, axis=0, ignore_index=True)

    tid = cfg.identifiers["turbine_id_col"]
    tcol = cfg.identifiers["time_col"]
    if tid not in df_raw.columns or tcol not in df_raw.columns:
        raise ValueError(f"Missing required columns: {tid}, {tcol}")

    df_raw[tcol] = pd.to_datetime(df_raw[tcol], errors="coerce")
    df_raw = df_raw.dropna(subset=[tcol])
    df_raw = df_raw.sort_values([tid, tcol]).reset_index(drop=True)
    return df_raw


def clamp_values(df: pd.DataFrame, clamp_map: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for col, (lo, hi) in clamp_map.items():
        if col in out.columns:
            out[col] = out[col].clip(lower=float(lo), upper=float(hi))
    return out


def winsorize_values(
    df: pd.DataFrame,
    winsor_map: dict[str, tuple[float, float]],
    group_key: Optional[str] = None,
) -> pd.DataFrame:
    out = df.copy()
    for col, (q_lo, q_hi) in winsor_map.items():
        if col not in out.columns:
            continue
        if group_key is None:
            lo = out[col].quantile(float(q_lo))
            hi = out[col].quantile(float(q_hi))
            out[col] = out[col].clip(lower=lo, upper=hi)
        else:

            def _cap(g: pd.DataFrame) -> pd.DataFrame:
                lo = g[col].quantile(float(q_lo))
                hi = g[col].quantile(float(q_hi))
                g[col] = g[col].clip(lower=lo, upper=hi)
                return g

            out = out.groupby(group_key, group_keys=False).apply(_cap)
    return out


def _dedupe(df: pd.DataFrame, tid: str, tcol: str, rule: str) -> pd.DataFrame:
    if rule == "last":
        return df.drop_duplicates(subset=[tid, tcol], keep="last")
    if rule in ("mean", "max", "min"):
        gb = df.groupby([tid, tcol], as_index=False)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        keep_cols = [c for c in df.columns if c not in num_cols]
        base = gb[keep_cols].last()
        if rule == "mean":
            agg = gb[num_cols].mean()
        elif rule == "max":
            agg = gb[num_cols].max()
        else:
            agg = gb[num_cols].min()
        out = pd.merge(base, agg, on=[tid, tcol], how="inner")
        return out
    raise ValueError(f"Unknown dedupe_rule: {rule}")


def run_qc_align(
    df_raw: pd.DataFrame,
    cfg: DataConfig,
    feature_cfg: FeatureConfig,
    extra_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_aligned: MultiIndex (turbine_id, timestamp), includes raw_channels + extra_cols
      df_miss_raw: same index, raw_channels only, uint8 1 present, 0 missing
    """
    tid = cfg.identifiers["turbine_id_col"]
    tcol = cfg.identifiers["time_col"]
    dt = int(cfg.sampling["dt_minutes"])
    raw_channels = list(feature_cfg.raw_channels)

    extra_cols = list(extra_cols or [])
    extra_cols = [c for c in extra_cols if c not in raw_channels]

    needed_cols = [tid, tcol] + raw_channels + extra_cols
    for c in needed_cols:
        if c not in df_raw.columns:
            # raw channels must exist; extras can be absent
            if c in raw_channels:
                raise ValueError(f"Missing raw column: {c}")

    keep_cols = [c for c in needed_cols if c in df_raw.columns]
    df = df_raw[keep_cols].copy()
    df = _dedupe(df, tid, tcol, str(cfg.qc.get("dedupe_rule", "last")))

    clamp_map = cfg.qc.get("clamp") or {}
    if clamp_map:
        clamp_map = {k: (float(v[0]), float(v[1])) for k, v in clamp_map.items()}
        df = clamp_values(df, clamp_map)

    winsor_map = cfg.qc.get("winsorize") or None
    if winsor_map:
        winsor_map = {k: (float(v[0]), float(v[1])) for k, v in winsor_map.items()}
        df = winsorize_values(df, winsor_map, group_key=None)

    aligned_frames: list[pd.DataFrame] = []
    miss_frames: list[pd.DataFrame] = []

    for turbine_id, g in df.groupby(tid, sort=False):
        g = g.sort_values(tcol)
        g = g.set_index(pd.DatetimeIndex(g[tcol]))
        g = g.drop(columns=[tcol])

        rule = f"{dt}min"
        resample_cols = [c for c in raw_channels + extra_cols if c in g.columns]

        # raw channels: mean is typical; extra cols (flags) want max over bin
        g_raw = g[[c for c in raw_channels if c in g.columns]].resample(rule).mean()
        g_extra = None
        if extra_cols:
            present_extra = [c for c in extra_cols if c in g.columns]
            if present_extra:
                g_extra = g[present_extra].resample(rule).max()

        if g_extra is not None:
            g_res = pd.concat([g_raw, g_extra], axis=1)
            g_res = g_res[resample_cols]
        else:
            g_res = g_raw

        miss = (~g_res[raw_channels].isna()).astype("uint8")
        g_res.insert(0, tid, turbine_id)
        g_res.index.name = tcol

        aligned_frames.append(g_res)
        miss_frames.append(miss.assign(**{tid: turbine_id}).set_index([tid], append=True).swaplevel(0, 1))

    df_aligned = pd.concat(aligned_frames, axis=0)
    df_aligned = df_aligned.reset_index().set_index([tid, tcol]).sort_index()

    df_miss_raw = pd.concat(miss_frames, axis=0).sort_index()
    df_miss_raw = df_miss_raw[raw_channels].astype("uint8")
    return df_aligned, df_miss_raw
