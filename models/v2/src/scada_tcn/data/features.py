from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from ..config_schema import FeatureConfig
from ..registry import FeatureRegistry
from .scalers import RobustScalerParams, transform_robust


def wrap_angle_diff(a: pd.Series, b: pd.Series, units: str, wrap_range: str) -> pd.Series:
    diff = a - b
    if units == "deg":
        period = 360.0
        if wrap_range != "[-180,180)":
            raise ValueError(f"Unsupported wrap_range for deg: {wrap_range}")
        return ((diff + 180.0) % period) - 180.0
    if units == "rad":
        period = 2.0 * math.pi
        if wrap_range != "[-pi,pi)":
            raise ValueError(f"Unsupported wrap_range for rad: {wrap_range}")
        return ((diff + math.pi) % period) - math.pi
    raise ValueError(f"Unknown units: {units}")


def encode_angle_sin_cos(angle: pd.Series, units: str, prefix: str) -> pd.DataFrame:
    if units == "deg":
        rad = np.deg2rad(angle.astype(float))
    elif units == "rad":
        rad = angle.astype(float)
    else:
        raise ValueError(f"Unknown units: {units}")

    return pd.DataFrame(
        {f"{prefix}_sin": np.sin(rad), f"{prefix}_cos": np.cos(rad)},
        index=angle.index,
    )


def add_deltas(df_X: pd.DataFrame, df_M: pd.DataFrame, channels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_X = df_X.copy()
    out_M = df_M.copy()

    for ch in channels:
        if ch not in out_X.columns:
            raise ValueError(f"Delta base channel not found: {ch}")

        dname = f"{ch}_delta"
        g = out_X[ch].groupby(level=0, sort=False)
        delta = g.diff(1)

        mcur = out_M[ch].astype("uint8")
        mprev = mcur.groupby(level=0, sort=False).shift(1).fillna(0).astype("uint8")
        mp = (mcur & mprev).astype("uint8")

        out_X[dname] = delta
        out_M[dname] = mp

    return out_X, out_M


def add_rolling(
    df_X: pd.DataFrame,
    df_M: pd.DataFrame,
    channels: list[str],
    windows: list[int],
    stats: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_X = df_X.copy()
    out_M = df_M.copy()

    for ch in channels:
        if ch not in out_X.columns:
            raise ValueError(f"Rolling base channel not found: {ch}")

        for w in windows:
            if w <= 0:
                raise ValueError(f"Invalid rolling window: {w}")

            m = out_M[ch].astype("uint8")
            present_count = (
                m.groupby(level=0, sort=False)
                .rolling(window=w, min_periods=w)
                .sum()
                .reset_index(level=0, drop=True)
            )
            m_roll = (present_count == w).astype("uint8")

            g = out_X[ch].groupby(level=0, sort=False)
            roll = g.rolling(window=w, min_periods=w)

            if "mean" in stats:
                name = f"{ch}_roll{w}_mean"
                out_X[name] = roll.mean().reset_index(level=0, drop=True)
                out_M[name] = m_roll
            if "std" in stats:
                name = f"{ch}_roll{w}_std"
                out_X[name] = roll.std(ddof=0).reset_index(level=0, drop=True)
                out_M[name] = m_roll

    return out_X, out_M


def build_features(
    df_aligned: pd.DataFrame,
    df_miss_raw: pd.DataFrame,
    cfg: FeatureConfig,
    scaler: Optional[RobustScalerParams],
    feature_registry: Optional[FeatureRegistry],
    fill_missing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, FeatureRegistry]:
    raw = list(cfg.raw_channels)
    ang = set(cfg.angular_channels)
    units = str(cfg.angle_units)

    if not isinstance(df_aligned.index, pd.MultiIndex):
        raise ValueError("df_aligned must be indexed by (turbine_id, timestamp) MultiIndex")
    if not df_aligned.index.equals(df_miss_raw.index):
        raise ValueError("df_aligned and df_miss_raw must share identical index")

    X_parts: list[pd.DataFrame] = []
    M_parts: list[pd.DataFrame] = []

    for ch in raw:
        if ch in ang:
            enc = encode_angle_sin_cos(df_aligned[ch], units=units, prefix=ch)
            X_parts.append(enc)
            m = df_miss_raw[ch].astype("uint8")
            M_parts.append(pd.DataFrame({f"{ch}_sin": m, f"{ch}_cos": m}, index=df_aligned.index))
        else:
            X_parts.append(df_aligned[[ch]].copy())
            M_parts.append(df_miss_raw[[ch]].astype("uint8"))

    df_X = pd.concat(X_parts, axis=1)
    df_M = pd.concat(M_parts, axis=1).astype("uint8")

    if bool(cfg.yaw_error.get("enabled", False)):
        wcol = str(cfg.yaw_error["wind_dir_col"])
        ycol = str(cfg.yaw_error["nacelle_yaw_col"])
        oname = str(cfg.yaw_error.get("output_name", "yaw_error"))
        wrap_range = str(cfg.yaw_error.get("wrap_range", "[-180,180)"))

        yaw_err = wrap_angle_diff(df_aligned[wcol], df_aligned[ycol], units=units, wrap_range=wrap_range)
        myaw = (df_miss_raw[wcol].astype("uint8") & df_miss_raw[ycol].astype("uint8")).astype("uint8")

        df_X[oname] = yaw_err
        df_M[oname] = myaw

    if bool(cfg.deltas.get("enabled", False)):
        df_X, df_M = add_deltas(df_X, df_M, list(cfg.deltas.get("channels", [])))

    if bool(cfg.rolling.get("enabled", False)):
        if not bool(cfg.rolling.get("causal", True)):
            raise ValueError("rolling.causal must be true")
        df_X, df_M = add_rolling(
            df_X,
            df_M,
            channels=list(cfg.rolling.get("channels", [])),
            windows=list(cfg.rolling.get("windows", [])),
            stats=list(cfg.rolling.get("stats", [])),
        )

    derived_names = list(df_X.columns)
    if feature_registry is None:
        if bool(cfg.output_ordering.get("explicit", False)):
            names = list(cfg.output_ordering.get("feature_names", []))
            if set(names) != set(derived_names):
                raise ValueError("Explicit feature_names does not match engineered columns set")
            ordered = names
        else:
            ordered = derived_names

        feature_registry = FeatureRegistry(
            feature_names=ordered,
            raw_channel_names=raw,
            angular_channel_names=list(cfg.angular_channels),
        )
    else:
        if set(feature_registry.feature_names) != set(derived_names):
            raise ValueError("Provided registry feature_names does not match engineered columns set")

    df_X = df_X[feature_registry.feature_names]
    df_M = df_M[feature_registry.feature_names].astype("uint8")

    if scaler is not None:
        df_X = df_X.copy()
        scale_cols = list(scaler.feature_names)
        if scale_cols:
            df_X[scale_cols] = transform_robust(df_X[scale_cols], scaler)[scale_cols]

    if fill_missing:
        c = float(cfg.fill_value_c)
        df_X = df_X.where(df_M.astype(bool), other=c)

    return df_X.astype("float32"), df_M.astype("uint8"), feature_registry
