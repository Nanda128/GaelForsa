# scripts/prepare_data.py
from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Iterable

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from scada_tcn.config_schema import FeatureConfig, build_config
from scada_tcn.data import (
    build_features,
    build_flags,
    build_window_index,
    fit_robust_scaler,
    load_raw_tables,
    run_qc_align,
    save_scaler,
    split_window_index,
)
from scada_tcn.registry import save_registry
from scada_tcn.utils.io import ensure_dir, save_parquet
from scada_tcn.utils.logging import get_logger, log_kv


SPEC_SCADA_CHANNELS = [
    "Wind Speed",
    "Wind Direction",
    "Ambient Temperature",
    "Rotor Speed",
    "Generator Speed",
    "Generator Torque",
    "Active Power",
    "Reactive Power",
    "Blade Pitch Angle",
    "Gearbox Oil Temperature",
    "Generator Winding Temperature",
    "Generator Bearing Temperature",
    "Converter Temperatures",
    "Transformer Temperature",
    "Generator Current",
    "Voltage",
    "NacelleYaw",
]
SPEC_ANGULAR_CHANNELS = ["Wind Direction", "Blade Pitch Angle", "NacelleYaw"]


def _ensure_columns(df: pd.DataFrame, cols: Iterable[str], fill_value: Any = np.nan) -> pd.DataFrame:
    """Ensure df has all columns in cols; create missing ones with fill_value."""
    for c in cols:
        if c not in df.columns:
            df[c] = fill_value
    return df


def _coerce_numeric_inplace(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """Coerce selected columns to numeric (errors->NaN)."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _auto_detect_raw_channels(
    df_raw: pd.DataFrame,
    tid_col: str,
    time_col: str,
    exclude_cols: set[str],
) -> list[str]:
    """
    Auto-detect candidate raw channels:
      - all columns except {tid_col, time_col} and exclude_cols
      - coercible to numeric
      - not all-NaN after coercion
    Deterministic output: sorted by column name.
    """
    candidates = [c for c in df_raw.columns if c not in exclude_cols and c not in (tid_col, time_col)]
    # Try coercion so numeric columns stored as objects still become usable
    _coerce_numeric_inplace(df_raw, candidates)

    raw = []
    for c in candidates:
        if pd.api.types.is_numeric_dtype(df_raw[c]) and not df_raw[c].isna().all():
            raw.append(c)

    raw = sorted(set(raw))
    if not raw:
        raise ValueError(
            "AUTO feature discovery found zero usable numeric columns.\n"
            f"Excluded: {sorted(exclude_cols)}\n"
            f"All columns: {list(df_raw.columns)}"
        )
    return raw


def _train_rows_mask(df_X: pd.DataFrame, cfg_data: dict[str, Any]) -> pd.Series:
    """
    Boolean mask selecting rows used to fit scaler (training split only).
    Supports:
      - splits.method == "time"
      - splits.method == "turbine"
    Fallback: all rows.
    """
    splits = cfg_data.get("splits", {})
    method = str(splits.get("method", "time"))

    if method == "time":
        tr0, tr1 = splits.get("train_range", [None, None])
        if tr0 is None or tr1 is None:
            return pd.Series(True, index=df_X.index)

        tr0 = pd.to_datetime(tr0)
        tr1 = pd.to_datetime(tr1)

        t = pd.to_datetime(df_X.index.get_level_values(1))
        mask = (t >= tr0) & (t <= tr1)
        return pd.Series(mask, index=df_X.index)

    if method == "turbine":
        tids = sorted(df_X.index.get_level_values(0).unique().tolist())
        n = len(tids)
        ntr = int(0.7 * n)
        train_set = set(tids[:ntr])

        mask = df_X.index.get_level_values(0).isin(train_set)
        return pd.Series(mask, index=df_X.index)

    if method in ("time_auto", "time_per_turbine"):
        ratios = splits.get("ratios", [0.7, 0.15, 0.15])
        if len(ratios) != 3:
            raise ValueError(f"splits.ratios must have 3 values for {method}, got {ratios}")
        tr, va, te = [float(x) for x in ratios]
        s = tr + va + te
        if s <= 0:
            raise ValueError(f"splits.ratios must sum > 0 for {method}, got {ratios}")
        tr = tr / s

        idx = df_X.index
        tid = idx.get_level_values(0)
        t = pd.to_datetime(idx.get_level_values(1))

        keep = pd.Series(False, index=df_X.index)
        for turbine_id in pd.Index(tid).unique().tolist():
            m = tid == turbine_id
            loc = np.flatnonzero(m)
            if len(loc) == 0:
                continue
            order = loc[np.argsort(t[m].to_numpy())]
            n = len(order)
            ntr = int(n * tr)
            if n >= 3:
                ntr = max(1, min(n - 2, ntr))
            keep.iloc[order[:ntr]] = True
        return keep

    return pd.Series(True, index=df_X.index)



def _resolve_feature_config_for_dataset(
    rc,
    df_raw: pd.DataFrame,
    extra_cols: list[str],
    logger,
) -> FeatureConfig:
    """
    Returns a FeatureConfig that is compatible with the dataset.
    - If rc.features.raw_channels is empty => AUTO detect features.
    - Else => use rc.features.raw_channels, but tolerate missing by creating NaN columns.
    Also:
    - Drops angular_channels not present in raw_channels (warn)
    - If yaw_error enabled but required cols missing, auto-disable (warn)
    """
    tid = rc.data.identifiers["turbine_id_col"]
    tcol = rc.data.identifiers["time_col"]

    # Treat mapped flag columns and identifiers as excluded from features when auto-discovering
    # add near exclude definition
    label_col = str(rc.data.labels.get("label_col", "anomaly"))  # you’ll add this to config
    hard_exclude = {
        label_col,
        "anomaly",
        "event_id",
        "train_test",
        "status_type_id",
        "id",
    }
    exclude = set(extra_cols) | {tid, tcol} | hard_exclude


    raw_channels_cfg = list(rc.features.raw_channels or [])
    auto = (len(raw_channels_cfg) == 0) or (raw_channels_cfg == ["*"]) or (raw_channels_cfg == ["__AUTO__"])

    if auto:
        discovered = _auto_detect_raw_channels(df_raw, tid, tcol, exclude_cols=exclude)

        # In auto mode, the model contract is fixed to canonical SCADA channels.
        # We map dataset columns into this schema upstream and create any still-missing
        # canonical columns as NaN so model input shape stays stable across farms.
        raw_channels = list(SPEC_SCADA_CHANNELS)
        _ensure_columns(df_raw, raw_channels, fill_value=np.nan)

        matched_spec = [c for c in SPEC_SCADA_CHANNELS if c in discovered]
        missing_spec = [c for c in SPEC_SCADA_CHANNELS if c not in matched_spec]
        logger.info(
            {
                "msg": "AUTO fixed canonical raw_channels selected",
                "count": len(raw_channels),
                "selected": raw_channels,
                "matched_in_dataset": matched_spec,
                "missing_spec": missing_spec,
                "dropped_non_spec_count": int(max(0, len(discovered) - len(matched_spec))),
            }
        )
    else:
        raw_channels = list(raw_channels_cfg)
        # ensure missing columns exist so downstream code doesn't crash
        _ensure_columns(df_raw, raw_channels, fill_value=np.nan)

    # Coerce raw feature channels to numeric where possible.
    # Keep non-numeric extras (e.g., textual fault labels) untouched.
    _coerce_numeric_inplace(df_raw, raw_channels)

    # Angular channels must be subset of raw_channels
    angular_source = SPEC_ANGULAR_CHANNELS if auto else list(rc.features.angular_channels or [])
    angular = [c for c in angular_source if c in set(raw_channels)]
    dropped_ang = [c for c in angular_source if c not in set(raw_channels)]
    if dropped_ang:
        logger.info({"msg": "Dropping angular_channels not in raw_channels", "dropped": dropped_ang})

    # Yaw error requires both cols in df_raw (not just in raw_channels)
    yaw_cfg = dict(rc.features.yaw_error or {})
    wcol = str(yaw_cfg.get("wind_dir_col", "Wind Direction"))
    ycol = str(yaw_cfg.get("nacelle_yaw_col", "NacelleYaw"))
    if auto:
        yaw_cfg["enabled"] = True

    if bool(yaw_cfg.get("enabled", False)):
        if (wcol not in df_raw.columns) or (ycol not in df_raw.columns):
            logger.info(
                {
                    "msg": "Disabling yaw_error because required cols are missing in dataset",
                    "wind_dir_col": wcol,
                    "nacelle_yaw_col": ycol,
                }
            )
            yaw_cfg["enabled"] = False

    # Build a dataset-compatible FeatureConfig
    feat_cfg = replace(
        rc.features,
        raw_channels=raw_channels,
        angular_channels=angular,
        yaw_error=yaw_cfg,
    )
    return feat_cfg


def _build_labels_series(df_aligned: pd.DataFrame, rc, logger) -> tuple[pd.Series | None, list[str]]:
    labels_cfg = dict(rc.data.labels or {})
    if not bool(labels_cfg.get("enabled", False)):
        return None, []

    label_col = str(labels_cfg.get("label_col", "anomaly"))
    if label_col not in df_aligned.columns:
        logger.warning({"msg": "labels.enabled=true but label column is missing", "label_col": label_col})
        return None, []

    class_names_cfg = [str(x) for x in labels_cfg.get("class_names", [])]

    # If label column is textual, infer compact class IDs from configured class_names.
    if df_aligned[label_col].dtype == object:
        raw = df_aligned[label_col].astype(str).str.strip().str.lower()
        if class_names_cfg:
            mapping = {name.lower(): i for i, name in enumerate(class_names_cfg)}
            y = raw.map(mapping).fillna(mapping.get("other_fault", 0)).astype("int64")
            return y, class_names_cfg

        inferred = sorted(raw.dropna().unique().tolist())
        if "normal" in inferred:
            inferred = ["normal"] + [c for c in inferred if c != "normal"]
        if "other_fault" not in inferred:
            inferred.append("other_fault")
        mapping = {name: i for i, name in enumerate(inferred)}
        y = raw.map(mapping).fillna(mapping["other_fault"]).astype("int64")
        logger.info({"msg": "Inferred fault class_names from textual labels", "class_names": inferred})
        return y, inferred

    y = pd.to_numeric(df_aligned[label_col], errors="coerce").fillna(0)
    if bool(labels_cfg.get("multilabel", False)):
        logger.warning({"msg": "multilabel=true is not yet supported in prepare_data labels export; using raw numeric labels"})
        return y.astype("float32"), class_names_cfg

    n_classes = int(rc.model.heads.get("fault", {}).get("C", 2))
    if n_classes <= 2:
        y = (y > 0).astype("int64")
    else:
        y = y.clip(lower=0, upper=n_classes - 1).astype("int64")

    return y, class_names_cfg


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    rc = build_config(cfg)
    logger = get_logger("prepare_data", output_dir=rc.output_dir)

    raw_dir = rc.data.paths["raw_dir"]
    processed_dir = rc.data.paths["processed_dir"]
    ensure_dir(processed_dir)

    # ---- Determine which columns are flags (so we don't accidentally treat them as features) ----
    extra_cols: list[str] = []
    if str(rc.flags.sources.get("type", "columns")) == "columns":
        extra_cols = list(dict(rc.flags.sources.get("mapping", {})).values())
    label_col = str(rc.data.labels.get("label_col", "anomaly"))
    if bool(rc.data.labels.get("enabled", False)) and label_col:
        extra_cols.append(label_col)

    # ---- Load raw ----
    df_raw = load_raw_tables(raw_dir, rc.data)

    # ---- Resolve FeatureConfig against this dataset (AUTO or explicit, tolerating missing) ----
    feat_cfg = _resolve_feature_config_for_dataset(rc, df_raw, extra_cols=extra_cols, logger=logger)

    # ---- QC/align ----
    df_aligned, df_miss_raw = run_qc_align(df_raw, rc.data, feat_cfg, extra_cols=extra_cols)

    # ---- Features: build unscaled/unfilled for scaler fit ----
    df_X_unscaled, df_M_miss, feature_reg = build_features(
        df_aligned,
        df_miss_raw,
        feat_cfg,
        scaler=None,
        feature_registry=None,
        fill_missing=False,
    )

    # ---- Fit scaler on training split only ----
    scaler_cfg = dict(feat_cfg.scaler or {})
    scale_sincos = bool(scaler_cfg.get("scale_sincos", False))
    if scale_sincos:
        scale_cols = list(df_X_unscaled.columns)
    else:
        scale_cols = [c for c in df_X_unscaled.columns if not (c.endswith("_sin") or c.endswith("_cos"))]

    train_mask = _train_rows_mask(df_X_unscaled, cfg_data=rc.data.__dict__)
    scaler = fit_robust_scaler(
        df_X_unscaled.loc[train_mask],
        feature_cols=scale_cols,
        eps=float(scaler_cfg.get("eps", 1e-6)),
        clip_iqr=scaler_cfg.get("clip_iqr", None),
        fitted_on={
            "split_method": rc.data.splits.get("method"),
            "train_range": rc.data.splits.get("train_range"),
            "note": "AUTO features may differ per dataset run; registry is authoritative.",
        },
    )

    # ---- Features: rebuild scaled + filled (using scaler + same registry) ----
    df_X, df_M_miss2, feature_reg2 = build_features(
        df_aligned,
        df_miss_raw,
        feat_cfg,
        scaler=scaler,
        feature_registry=feature_reg,
        fill_missing=True,
    )
    df_M_miss = df_M_miss2
    feature_reg = feature_reg2

    # ---- Flags ----
    df_flags, flag_reg = build_flags(df_aligned, rc.flags, flag_registry=None)
    labels_series, class_names = _build_labels_series(df_aligned, rc, logger)

    # ---- Persist artifacts ----
    ensure_dir(os.path.join(rc.output_dir, "scalers"))
    ensure_dir(os.path.join(rc.output_dir, "registries"))
    save_scaler(scaler, os.path.join(rc.output_dir, "scalers", "robust_scaler.json"))
    save_registry(feature_reg, flag_reg, os.path.join(rc.output_dir, "registries", "registry.json"))
    if class_names:
        label_map = {"class_names": class_names, "class_to_idx": {name: i for i, name in enumerate(class_names)}}
        from scada_tcn.utils.io import save_json
        save_json(label_map, os.path.join(rc.output_dir, "registries", "fault_label_map.json"))

    # (Optional) also store the registry alongside processed data for portability
    ensure_dir(os.path.join(processed_dir, "registries"))
    save_registry(feature_reg, flag_reg, os.path.join(processed_dir, "registries", "registry.json"))

    # ---- Persist per-turbine arrays ----
    ensure_dir(os.path.join(processed_dir, "turbines"))
    for turbine_id, gX in df_X.groupby(level=0, sort=False):
        gM = df_M_miss.xs(turbine_id, level=0)
        gF = df_flags.xs(turbine_id, level=0)

        ts = gX.index.get_level_values(1).to_numpy(dtype="datetime64[ns]")
        ts_i64 = ts.astype("int64")  # ns since epoch

        base = os.path.join(processed_dir, "turbines", str(turbine_id))
        ensure_dir(base)
        np.save(os.path.join(base, "X.npy"), gX.to_numpy(dtype=np.float32))
        np.save(os.path.join(base, "M_miss.npy"), gM.to_numpy(dtype=np.uint8))
        np.save(os.path.join(base, "Flags.npy"), gF.to_numpy(dtype=np.float32))
        np.save(os.path.join(base, "timestamps.npy"), ts_i64)
        if labels_series is not None:
            gy = labels_series.xs(turbine_id, level=0)
            np.save(os.path.join(base, "labels.npy"), gy.to_numpy())

    # ---- Window index + splits ----
    df_index = build_window_index(df_X, df_M_miss, df_flags, rc.data, labels_df=labels_series)
    splits = split_window_index(df_index, rc.data)

    save_parquet(df_index, os.path.join(processed_dir, "window_index.parquet"))
    for k, v in splits.items():
        save_parquet(v, os.path.join(processed_dir, f"window_index_{k}.parquet"))

    split_pos = {}
    if "window_has_positive" in df_index.columns:
        for k, v in splits.items():
            if len(v) == 0:
                split_pos[f"windows_{k}_positive"] = 0
            else:
                split_pos[f"windows_{k}_positive"] = int(v["window_has_positive"].astype(bool).sum())

    log_kv(
        logger,
        step=0,
        payload={
            "rows_raw": int(len(df_raw)),
            "rows_aligned": int(len(df_aligned)),
            "F": int(len(feature_reg.feature_names)),
            "R": int(len(flag_reg.flag_names)),
            "windows_total": int(len(df_index)),
            "windows_train": int(len(splits.get("train", []))),
            "windows_val": int(len(splits.get("val", []))),
            "windows_test": int(len(splits.get("test", []))),
            "features_mode": "auto" if (len(rc.features.raw_channels or []) == 0) else "explicit",
            "labels_enabled": bool(labels_series is not None),
            "labels_positive_frac": float((labels_series > 0).mean()) if labels_series is not None else 0.0,
            **split_pos,
        },
    )


if __name__ == "__main__":
    main()
