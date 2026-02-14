from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from ..config_schema import DataConfig, FeatureConfig


def _detect_separator(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s:
                for sep in (";", "\t", ",", "|"):
                    if s.count(sep) > 0:
                        return sep
                break
    return ","


def _read_table(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        sep = _detect_separator(path)
        df = pd.read_csv(path, sep=sep)
        if len(df.columns) == 1 and isinstance(df.columns[0], str) and ";" in df.columns[0]:
            df = pd.read_csv(path, sep=";")
    else:
        df = pd.read_parquet(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _farm_from_path(path: str, raw_dir: str) -> str:
    rel = os.path.relpath(os.path.dirname(path), raw_dir)
    parts = [p for p in rel.split(os.sep) if p and p != "."]
    if not parts:
        return "farm_unknown"
    return str(parts[0])




def _normalize_asset_series(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    s = s.fillna("")
    # Canonicalize integer-like float strings, e.g. "11.0" -> "11"
    s = s.str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)
    return s


def _normalize_asset_id(x: object) -> str:
    if pd.isna(x):
        return ""
    txt = str(x).strip()
    if txt == "":
        return ""
    return txt[:-2] if txt.endswith(".0") and txt[:-2].lstrip("-").isdigit() else txt


def _normalize_id_columns(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    tid = cfg.identifiers["turbine_id_col"]
    tcol = cfg.identifiers["time_col"]

    col_map = {c.lower(): c for c in df.columns}
    if tid not in df.columns:
        for cand in ("asset_id", "asset", "turbine_id", "turbine", "unit_id"):
            src = col_map.get(cand)
            if src is not None:
                df = df.rename(columns={src: tid})
                break
    if tcol not in df.columns:
        for cand in ("time_stamp", "timestamp", "time", "datetime", "date_time", "date"):
            src = col_map.get(cand)
            if src is not None:
                df = df.rename(columns={src: tcol})
                break
    return df


def _infer_fault_label(event_label: str, event_description: str) -> str:
    text = f"{event_label} {event_description}".lower()
    if any(tok in text for tok in ("normal", "none", "healthy", "ok", "no fault")):
        return "normal"

    groups: list[tuple[str, list[str]]] = [
        (
            "mechanical",
            [
                "gear", "gearbox", "planetary", "bearing", "main bearing", "rotor bearing", "rotorbearing",
                "shaft", "drivetrain", "coupling", "vibration", "blade", "yaw drive", "pitch motor",
                "pitch system", "brake disc", "mechanical",
            ],
        ),
        (
            "electrical",
            [
                "generator", "converter", "inverter", "igbt", "voltage", "current", "electrical",
                "transformer", "stator", "rotor winding", "insulation", "short circuit", "power module",
            ],
        ),
        (
            "thermal",
            [
                "temperature", "high temperature", "overheat", "overheating", "cooling", "thermal",
                "hot", "heat", "therm", "temp", "oil temperature",
            ],
        ),
        ("hydraulic", ["hydraulic", "pressure", "pump", "oil leak", "valve", "actuator"]),
        ("grid", ["grid", "curtail", "dispatch", "frequency", "voltage dip", "reactive power", "curtailment"]),
        ("sensor", ["sensor", "signal", "telemetry", "scada", "communication", "comms", "missing data", "nan"]),
        ("control", ["controller", "control", "setpoint", "trip", "shutdown", "startup", "brake", "standstill", "stopped"]),
        ("icing", ["ice", "icing", "frozen", "de-icing", "deicing"]),
    ]
    for group, toks in groups:
        if any(tok in text for tok in toks):
            return group
    return "other_fault"


def _norm_text(x: str) -> str:
    return " ".join(str(x).lower().replace("_", " ").replace("-", " ").split())


def _canonical_sensor_map(feature_desc: pd.DataFrame) -> dict[str, str]:
    if "sensor_name" not in feature_desc.columns:
        return {}
    desc_col = "description" if "description" in feature_desc.columns else "sensor_name"
    target_keywords: dict[str, list[str]] = {
        "Wind Speed": ["windspeed", "wind speed"],
        "Wind Direction": ["absolute direction", "wind direction"],
        "Ambient Temperature": ["ambient temperature"],
        "Rotor Speed": ["rotor speed"],
        "Generator Speed": ["generator speed"],
        "Generator Torque": ["generator torque", "torque"],
        "Active Power": ["active power", "power output", "actual power"],
        "Reactive Power": ["reactive power"],
        "Blade Pitch Angle": ["pitch angle"],
        "Gearbox Oil Temperature": ["gearbox oil temperature", "gearbox temperature", "gearbox oil"],
        "Generator Winding Temperature": ["winding temperature", "stator temperature"],
        "Generator Bearing Temperature": ["generator bearing temperature", "bearing temperature"],
        "Converter Temperatures": ["converter temperature", "igbt", "inverter temperature"],
        "Transformer Temperature": ["transformer temperature"],
        "Generator Current": ["generator current", "current"],
        "Voltage": ["voltage"],
        "NacelleYaw": ["nacelle yaw", "yaw position", "yaw angle"],
    }

    mapping: dict[str, str] = {}
    for _, row in feature_desc.iterrows():
        sensor = str(row.get("sensor_name", "")).strip()
        text = _norm_text(f"{row.get(desc_col, '')} {sensor}")
        if not sensor:
            continue
        best_name = None
        best_score = 0
        for cname, keys in target_keywords.items():
            score = sum(1 for k in keys if _norm_text(k) in text)
            if score > best_score:
                best_name = cname
                best_score = score
        if best_name is not None and best_score > 0 and sensor not in mapping:
            mapping[sensor] = best_name
    return mapping


def _canonical_alias_map() -> dict[str, str]:
    alias: dict[str, list[str]] = {
        "Wind Speed": ["wind speed", "windspeed", "wind_speed"],
        "Wind Direction": ["wind direction", "wind_dir", "winddirection"],
        "Ambient Temperature": ["ambient temperature", "ambient_temp", "outside temperature"],
        "Rotor Speed": ["rotor speed", "rotor_speed"],
        "Generator Speed": ["generator speed", "generator_speed", "gen speed"],
        "Generator Torque": ["generator torque", "generator_torque", "gen torque"],
        "Active Power": ["active power", "active_power", "power output", "actual power"],
        "Reactive Power": ["reactive power", "reactive_power"],
        "Blade Pitch Angle": ["blade pitch angle", "pitch angle", "blade_pitch"],
        "Gearbox Oil Temperature": ["gearbox oil temperature", "gearbox_oil_temp", "gearbox temperature"],
        "Generator Winding Temperature": ["generator winding temperature", "winding temperature", "stator temperature"],
        "Generator Bearing Temperature": ["generator bearing temperature", "bearing temperature", "generator_bearing_temp"],
        "Converter Temperatures": ["converter temperatures", "converter temperature", "igbt temperature", "inverter temperature"],
        "Transformer Temperature": ["transformer temperature", "transformer_temp"],
        "Generator Current": ["generator current", "generator_current", "gen current"],
        "Voltage": ["voltage", "generator voltage"],
        "NacelleYaw": ["nacelleyaw", "nacelle yaw", "yaw position", "yaw angle"],
    }
    out: dict[str, str] = {}
    for canonical, variants in alias.items():
        out[_norm_text(canonical)] = canonical
        for v in variants:
            out[_norm_text(v)] = canonical
    return out


def _apply_direct_canonical_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    alias = _canonical_alias_map()
    for col in list(df.columns):
        key = _norm_text(col)
        dst = alias.get(key)
        if dst is None:
            continue
        if col == dst:
            continue
        if dst not in out.columns:
            out[dst] = out[col]
        else:
            out[dst] = out[dst].where(~out[dst].isna(), out[col])
    return out


def _apply_feature_map(df: pd.DataFrame, feature_desc_path: str) -> pd.DataFrame:
    if not os.path.exists(feature_desc_path):
        return df
    fd = _read_table(feature_desc_path)
    base_map = _canonical_sensor_map(fd)
    if not base_map:
        return df

    out = df.copy()
    for src_base, dst in base_map.items():
        candidates = [f"{src_base}_avg", src_base]
        src_col = next((c for c in candidates if c in out.columns), None)
        if src_col is None:
            continue
        if dst not in out.columns:
            out[dst] = out[src_col]
        else:
            out[dst] = out[dst].where(~out[dst].isna(), out[src_col])
    return out


def _read_event_info(path: str) -> pd.DataFrame:
    ev = _read_table(path)
    cols = {c.lower(): c for c in ev.columns}

    asset_col = next((cols.get(k) for k in ("asset_id", "asset", "turbine_id", "turbine") if cols.get(k)), None)
    label_col = next((cols.get(k) for k in ("event_label", "label") if cols.get(k)), None)
    start_col = next((cols.get(k) for k in ("event_start", "start_time", "start") if cols.get(k)), None)
    end_col = next((cols.get(k) for k in ("event_end", "end_time", "end") if cols.get(k)), None)
    start_id_col = next((cols.get(k) for k in ("event_start_id", "start_id") if cols.get(k)), None)
    end_id_col = next((cols.get(k) for k in ("event_end_id", "end_id") if cols.get(k)), None)
    if asset_col is None or label_col is None or start_col is None or end_col is None:
        return pd.DataFrame()

    desc_col = next((cols.get(k) for k in ("event_description", "description") if cols.get(k)), None)
    event_id_col = cols.get("event_id")

    out = pd.DataFrame(
        {
            "asset_id": _normalize_asset_series(ev[asset_col]),
            "event_label": ev[label_col].astype(str),
            "event_start": pd.to_datetime(ev[start_col], errors="coerce"),
            "event_end": pd.to_datetime(ev[end_col], errors="coerce"),
            "event_start_id": pd.to_numeric(ev[start_id_col], errors="coerce") if start_id_col is not None else np.nan,
            "event_end_id": pd.to_numeric(ev[end_id_col], errors="coerce") if end_id_col is not None else np.nan,
            "event_description": ev[desc_col].astype(str) if desc_col is not None else "",
            "event_id": ev[event_id_col].astype(str) if event_id_col is not None else "-1",
        }
    )
    out = out.dropna(subset=["event_start", "event_end"]).reset_index(drop=True)
    out["event_start"] = out["event_start"].dt.tz_localize(None)
    out["event_end"] = out["event_end"].dt.tz_localize(None)
    return out


def load_raw_tables(raw_dir: str, cfg: DataConfig) -> pd.DataFrame:
    if not os.path.isdir(raw_dir):
        raise ValueError(f"raw_dir not found: {raw_dir}")

    files: list[str] = []
    for root, _, fns in os.walk(raw_dir):
        for fn in fns:
            if fn.endswith(".parquet") or fn.endswith(".pq") or fn.endswith(".csv"):
                p = os.path.join(root, fn)
                if fn.lower() == "event_info.csv" or fn.lower() == "feature_description.csv":
                    continue
                files.append(p)
    if not files:
        raise ValueError(f"No .parquet/.csv files found in {raw_dir}")

    tid = cfg.identifiers["turbine_id_col"]
    tcol = cfg.identifiers["time_col"]

    dfs: list[pd.DataFrame] = []
    event_infos: list[pd.DataFrame] = []
    event_paths: list[str] = []
    for root, _, fns in os.walk(raw_dir):
        for fn in fns:
            if fn.lower() == "event_info.csv":
                event_paths.append(os.path.join(root, fn))

    for p in sorted(files):
        df = _read_table(p)
        df = _normalize_id_columns(df, cfg)
        df = _apply_direct_canonical_aliases(df)
        farm_id = _farm_from_path(p, raw_dir)
        df = _apply_feature_map(df, os.path.join(raw_dir, farm_id, "feature_description.csv"))
        if tid in df.columns:
            df[tid] = _normalize_asset_series(df[tid])
        df["farm_id"] = farm_id
        dfs.append(df)

    for p in sorted(event_paths):
        ev = _read_event_info(p)
        if len(ev) == 0:
            continue
        ev["farm_id"] = _farm_from_path(p, raw_dir)
        event_infos.append(ev)

    df_raw = pd.concat(dfs, axis=0, ignore_index=True)

    if tid not in df_raw.columns or tcol not in df_raw.columns:
        raise ValueError(f"Missing required columns: {tid}, {tcol}")

    if event_infos:
        all_events = pd.concat(event_infos, axis=0, ignore_index=True)
        if "fault_label" not in df_raw.columns:
            df_raw["fault_label"] = "normal"
        if "event_id" not in df_raw.columns:
            df_raw["event_id"] = "-1"

        time_vals = pd.to_datetime(df_raw[tcol], errors="coerce")
        raw_id_vals = pd.to_numeric(df_raw.get("id"), errors="coerce") if "id" in df_raw.columns else None

        # Apply normal events first, then non-normal events so anomaly labels take precedence.
        all_events = all_events.copy()
        all_events["_fault_label"] = [
            _infer_fault_label(str(ev), str(desc))
            for ev, desc in zip(all_events["event_label"], all_events.get("event_description", ""))
        ]
        all_events["_is_non_normal"] = all_events["_fault_label"] != "normal"
        all_events = all_events.sort_values("_is_non_normal", ascending=True).reset_index(drop=True)

        for _, ev in all_events.iterrows():
            ev_asset = _normalize_asset_id(ev["asset_id"])
            ev_farm = str(ev["farm_id"]) if "farm_id" in ev else ""
            ev_fault_label = str(ev.get("_fault_label", "normal"))

            mask_asset = _normalize_asset_series(df_raw[tid]) == ev_asset
            mask_farm = df_raw["farm_id"].astype(str) == ev_farm if "farm_id" in df_raw.columns else True
            base_mask = mask_asset & mask_farm

            mask_time = (time_vals >= ev["event_start"]) & (time_vals <= ev["event_end"])
            mask = base_mask & mask_time

            if not np.any(mask) and raw_id_vals is not None:
                sid = pd.to_numeric(pd.Series([ev.get("event_start_id", np.nan)]), errors="coerce").iloc[0]
                eid = pd.to_numeric(pd.Series([ev.get("event_end_id", np.nan)]), errors="coerce").iloc[0]
                if pd.notna(sid) and pd.notna(eid):
                    lo, hi = (sid, eid) if sid <= eid else (eid, sid)
                    mask_id = (raw_id_vals >= lo) & (raw_id_vals <= hi)
                    mask = base_mask & mask_id

            if np.any(mask):
                # Avoid normal events erasing already assigned non-normal segments.
                if ev_fault_label == "normal":
                    write_mask = mask & (df_raw["fault_label"].astype(str) == "normal")
                else:
                    write_mask = mask
                if np.any(write_mask):
                    df_raw.loc[write_mask, "fault_label"] = ev_fault_label
                    df_raw.loc[write_mask, "event_id"] = str(ev["event_id"])

    if "farm_id" in df_raw.columns:
        df_raw[tid] = df_raw["farm_id"].astype(str) + "::" + _normalize_asset_series(df_raw[tid])

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
