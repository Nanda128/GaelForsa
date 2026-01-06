from __future__ import annotations

from typing import Optional

import pandas as pd

from ..config_schema import FlagConfig
from ..registry import FlagRegistry


def load_external_flags(path: str, cfg: FlagConfig) -> pd.DataFrame:
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    return df


def build_flags(
    df_source: pd.DataFrame,
    cfg: FlagConfig,
    flag_registry: Optional[FlagRegistry],
) -> tuple[pd.DataFrame, FlagRegistry]:
    if not isinstance(df_source.index, pd.MultiIndex):
        raise ValueError("df_source must be indexed by (turbine_id, timestamp) MultiIndex")

    if flag_registry is None:
        flag_registry = FlagRegistry(flag_names=list(cfg.flag_names))

    if flag_registry.flag_names != list(cfg.flag_names):
        raise ValueError("Flag registry mismatch vs cfg.flag_names")

    src = cfg.sources
    ftype = str(src.get("type", "columns"))
    mapping = dict(src.get("mapping", {}))

    out = pd.DataFrame(index=df_source.index)

    if ftype == "columns":
        for fname in flag_registry.flag_names:
            col = mapping.get(fname)
            if col is None:
                raise ValueError(f"flags.sources.mapping missing for flag: {fname}")
            if col not in df_source.columns:
                # strict: flags missing -> all zeros (common in SCADA exports)
                out[fname] = 0.0
            else:
                out[fname] = df_source[col].astype("float32").fillna(0.0)
    else:
        raise ValueError(f"Unsupported flags.sources.type: {ftype}")

    out = out[flag_registry.flag_names].astype("float32")
    return out, flag_registry
