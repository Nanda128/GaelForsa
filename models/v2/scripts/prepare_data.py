# scripts/prepare_data.py
from __future__ import annotations

import os

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from scada_tcn.config_schema import build_config, resolved_feature_names, resolved_flag_names
from scada_tcn.data import (
    build_features,
    build_flags,
    build_window_index,
    fit_robust_scaler,
    load_raw_tables,
    run_qc_align,
    save_scaler,
    split_window_index,
    transform_robust,
)
from scada_tcn.registry import FeatureRegistry, FlagRegistry, save_registry
from scada_tcn.utils.io import ensure_dir, save_parquet
from scada_tcn.utils.logging import get_logger, log_kv


def _ensure_multiindex(df: pd.DataFrame, turbine_id_col: str, time_col: str) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex):
        return df
    return df.set_index([turbine_id_col, time_col]).sort_index()


def _train_rows_by_time(df_X: pd.DataFrame, cfg_data: dict) -> pd.Series:
    tr0, tr1 = cfg_data["splits"]["train_range"]
    tr0 = pd.to_datetime(tr0); tr1 = pd.to_datetime(tr1)
    t = pd.to_datetime(df_X.index.get_level_values(1))
    return (t >= tr0) & (t <= tr1)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    rc = build_config(cfg)
    logger = get_logger("prepare_data", output_dir=rc.output_dir)

    raw_dir = rc.data.paths["raw_dir"]
    processed_dir = rc.data.paths["processed_dir"]
    ensure_dir(processed_dir)

    df_raw = load_raw_tables(raw_dir, rc.data)
    df_aligned, df_miss_raw = run_qc_align(df_raw, rc.data, rc.features)

    # Build features unscaled to fit scaler
    df_X_unscaled, df_M_miss, feature_reg = build_features(
    df_aligned, df_miss_raw, rc.features, scaler=None, feature_registry=None, fill_missing=False
    )


    # Fit robust scaler on train rows only (time split)
    scaler_cfg = rc.features.scaler
    sincos_cols = [c for c in df_X_unscaled.columns if c.endswith("_sin") or c.endswith("_cos")]
    scale_sincos = bool(scaler_cfg.get("scale_sincos", False))
    scale_cols = list(df_X_unscaled.columns) if scale_sincos else [c for c in df_X_unscaled.columns if c not in sincos_cols]

    train_mask = _train_rows_by_time(df_X_unscaled, rc.data.__dict__)
    scaler = fit_robust_scaler(
        df_X_unscaled.loc[train_mask],
        feature_cols=scale_cols,
        eps=float(scaler_cfg.get("eps", 1e-6)),
        clip_iqr=scaler_cfg.get("clip_iqr", None),
        fitted_on={"split": "time", "train_range": rc.data.splits.get("train_range")},
    )

    # Rebuild features using the fitted scaler, with filling enabled.
    df_X, df_M_miss2, feature_reg2 = build_features(
        df_aligned,
        df_miss_raw,
        rc.features,
        scaler=scaler,
        feature_registry=feature_reg,
        fill_missing=True,   # <- important: only fill AFTER missingness mask is computed
    )

    # Sanity: engineered missingness should be identical before/after scaling.
    # (If not, you’ve got a bug in build_features.)
    df_M_miss = df_M_miss2


    # Flags
    df_flags, flag_reg = build_flags(df_aligned, rc.flags, flag_registry=None)

    # Persist artifacts
    artifacts_dir = rc.output_dir
    ensure_dir(os.path.join(artifacts_dir, "scalers"))
    ensure_dir(os.path.join(artifacts_dir, "registries"))

    save_scaler(scaler, os.path.join(artifacts_dir, "scalers", "robust_scaler.json"))
    save_registry(feature_reg, flag_reg, os.path.join(artifacts_dir, "registries", "registry.json"))

    # Write processed per turbine arrays
    ensure_dir(os.path.join(processed_dir, "turbines"))
    for turbine_id, gX in df_X.groupby(level=0, sort=False):
        gM = df_M_miss.xs(turbine_id, level=0)
        gF = df_flags.xs(turbine_id, level=0)
        ts = gX.index.get_level_values(1).to_numpy(dtype="datetime64[ns]")

        base = os.path.join(processed_dir, "turbines", str(turbine_id))
        ensure_dir(base)
        np.save(os.path.join(base, "X.npy"), gX.to_numpy(dtype=np.float32))
        np.save(os.path.join(base, "M_miss.npy"), gM.to_numpy(dtype=np.uint8))
        np.save(os.path.join(base, "Flags.npy"), gF.to_numpy(dtype=np.float32))
        np.save(os.path.join(base, "timestamps.npy"), ts.astype("int64"))

    # Window index + splits
    df_index = build_window_index(df_X, df_M_miss, df_flags, rc.data)
    splits = split_window_index(df_index, rc.data)

    save_parquet(df_index, os.path.join(processed_dir, "window_index.parquet"))
    for k, v in splits.items():
        save_parquet(v, os.path.join(processed_dir, f"window_index_{k}.parquet"))

    log_kv(
        logger,
        step=0,
        payload={
            "rows_raw": int(len(df_raw)),
            "rows_aligned": int(len(df_aligned)),
            "F": int(len(feature_reg.feature_names)),
            "R": int(len(flag_reg.flag_names)),
            "windows_total": int(len(df_index)),
            "windows_train": int(len(splits["train"])),
            "windows_val": int(len(splits["val"])),
            "windows_test": int(len(splits["test"])),
        },
    )


if __name__ == "__main__":
    main()
