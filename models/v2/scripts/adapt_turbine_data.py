from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# -----------------------------
# CSV parsing helpers
# -----------------------------
def detect_separator(filepath: str) -> str:
    """Detect likely separator by scanning the first non-empty line."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                first_line = line
                break
        else:
            return ","

    # Prefer common separators by likelihood in your dataset (you already saw ';')
    for sep in [";", "\t", ",", "|"]:
        if sep in first_line and first_line.count(sep) > 0:
            return sep
    return ","


def _robust_read_csv(path: str) -> pd.DataFrame:
    """
    Read CSV with detected separator.
    Includes a safety fallback: if we ended up with 1 column containing ';', re-read with ';'.
    """
    sep = detect_separator(path)
    df = pd.read_csv(path, sep=sep)

    # Fallback: classic symptom of delimiter mismatch
    if len(df.columns) == 1 and isinstance(df.columns[0], str) and ";" in df.columns[0]:
        df = pd.read_csv(path, sep=";")

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _choose_best_datetime_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Pick the candidate column that parses best as datetime."""
    best_col = None
    best_ratio = 0.0
    for c in candidates:
        if c not in df.columns:
            continue
        parsed = pd.to_datetime(df[c], errors="coerce")
        ratio = float(parsed.notna().mean())
        if ratio > best_ratio:
            best_ratio = ratio
            best_col = c
    if best_col is None or best_ratio < 0.5:
        return None
    return best_col


# -----------------------------
# Load event info / feature desc
# -----------------------------
def load_event_info(event_path: str) -> pd.DataFrame:
    """
    Load and standardize event info.

    Returns df_events with columns:
      - asset_id (str)
      - event_id (str)
      - event_label (str)
      - event_start (datetime64[ns], tz-naive)
      - event_end (datetime64[ns], tz-naive)
      - event_description (str)
    """
    df = _robust_read_csv(event_path)
    print(f"Event info columns: {list(df.columns)}")

    cols_lower = {c.lower(): c for c in df.columns}

    # Asset
    for cand in ["asset_id", "asset", "turbine_id", "turbine", "unit_id", "unit"]:
        if cand in cols_lower:
            asset_col = cols_lower[cand]
            break
    else:
        asset_col = next((c for c in df.columns if "asset" in c.lower()), None)
    if asset_col is None:
        raise KeyError(f"Could not find asset column in event file. Columns={list(df.columns)}")

    # Event id (optional)
    event_id_col = cols_lower.get("event_id")
    if event_id_col is None:
        event_id_col = next((c for c in df.columns if "event_id" in c.lower()), None)

    # Label
    label_col = cols_lower.get("event_label")
    if label_col is None:
        label_col = next((c for c in df.columns if "label" in c.lower()), None)
    if label_col is None:
        raise KeyError(f"Could not find label column in event file. Columns={list(df.columns)}")

    # Start/End: avoid *_id columns by preferring non-id candidates
    non_id_cols = [c for c in df.columns if "id" not in c.lower() or c.lower() == "event_id"]
    start_candidates = [cols_lower[k] for k in ["event_start", "start_time", "start"] if k in cols_lower]
    end_candidates = [cols_lower[k] for k in ["event_end", "end_time", "end"] if k in cols_lower]

    start_col = _choose_best_datetime_col(df, start_candidates) or _choose_best_datetime_col(df, non_id_cols)
    end_col = _choose_best_datetime_col(df, end_candidates) or _choose_best_datetime_col(df, non_id_cols)

    if start_col is None or end_col is None:
        raise KeyError(
            "Could not detect parseable event_start/event_end columns.\n"
            f"Columns={list(df.columns)}\n"
            f"Start candidates tried={start_candidates}\n"
            f"End candidates tried={end_candidates}"
        )

    # Description (optional)
    desc_col = cols_lower.get("event_description")
    if desc_col is None:
        desc_col = next((c for c in df.columns if "description" in c.lower()), None)

    out = pd.DataFrame()
    out["asset_id"] = df[asset_col].astype(str).str.strip()
    out["event_id"] = (
        df[event_id_col].astype(str).str.strip() if event_id_col is not None else np.arange(len(df)).astype(str)
    )
    out["event_label"] = df[label_col].astype(str).str.strip()

    out["event_start"] = pd.to_datetime(df[start_col], errors="coerce")
    out["event_end"] = pd.to_datetime(df[end_col], errors="coerce")

    out["event_description"] = df[desc_col].astype(str) if desc_col is not None else ""

    out = out.dropna(subset=["event_start", "event_end"]).reset_index(drop=True)
    out["event_start"] = pd.to_datetime(out["event_start"]).dt.tz_localize(None)
    out["event_end"] = pd.to_datetime(out["event_end"]).dt.tz_localize(None)

    return out


def load_feature_descriptions(feature_path: str) -> pd.DataFrame:
    df = _robust_read_csv(feature_path)

    # If the feature file is actually headerless/oddly formatted in your repo, keep compatibility:
    # If it already has expected columns, do nothing; else fall back to your earlier headerless behavior.
    if set(["sensor_name", "is_angular"]).issubset(set(df.columns)):
        print(f"Feature description columns: {list(df.columns)}")
        print(f"Loaded {len(df)} feature descriptions")
        return df

    # Headerless fallback
    raw = pd.read_csv(feature_path, sep=detect_separator(feature_path), header=None)
    if len(raw.columns) >= 5:
        raw.columns = (
            ["sensor_name", "aggregations", "description", "unit", "is_angular"]
            + [f"col_{i}" for i in range(5, len(raw.columns))]
        )
    else:
        raw.columns = ["sensor_name", "aggregations", "description", "unit", "is_angular"][: len(raw.columns)]
    print(f"Feature description columns: {list(raw.columns)}")
    print(f"Loaded {len(raw)} feature descriptions")
    return raw




def _norm_text(x: str) -> str:
    return " ".join(str(x).lower().replace("_", " ").replace("-", " ").split())


def build_canonical_sensor_map(feature_desc: pd.DataFrame) -> dict[str, str]:
    """Map raw sensor names to canonical SCADA feature names using description keywords."""
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

def identify_angular_sensors(feature_desc: pd.DataFrame) -> list[str]:
    if "is_angular" not in feature_desc.columns or "sensor_name" not in feature_desc.columns:
        return []
    angular_mask = feature_desc["is_angular"].astype(str).str.lower().isin(["true", "t", "1", "yes"])
    angular = feature_desc.loc[angular_mask, "sensor_name"].astype(str).tolist()
    print(f"Identified {len(angular)} angular sensors")
    return angular


# -----------------------------
# Turbine CSV loading + labeling
# -----------------------------
def load_turbine_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a single turbine CSV file, normalizing to columns: time_stamp, asset_id.
    Critical fix: use detected separator; your turbine files are ';' delimited.
    """
    df = _robust_read_csv(csv_path)

    # ---- detect time column ----
    preferred_time = [
        "time_stamp",
        "timestamp",
        "time",
        "datetime",
        "date_time",
        "date",
        "dt",
        "tstamp",
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    time_col = None
    for cand in preferred_time:
        if cand in cols_lower:
            time_col = cols_lower[cand]
            break

    if time_col is None:
        # Heuristic: pick best parseable datetime column among non-numeric, non-id-like columns
        best = (None, 0.0)
        for c in df.columns:
            cl = c.lower()
            if cl in ("asset_id", "asset", "turbine_id", "turbine", "unit_id", "id"):
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                continue
            parsed = pd.to_datetime(df[c], errors="coerce")
            ratio = float(parsed.notna().mean())
            if ratio > best[1]:
                best = (c, ratio)
        if best[0] is None or best[1] < 0.5:
            raise KeyError(f"Could not detect a timestamp column. Columns={list(df.columns)}")
        time_col = best[0]

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    if df[time_col].isna().all():
        raise ValueError(f"Detected time column '{time_col}' but failed to parse any datetimes.")
    df = df.dropna(subset=[time_col]).copy()

    if time_col != "time_stamp":
        df = df.rename(columns={time_col: "time_stamp"})

    # ---- detect/create asset_id ----
    preferred_asset = ["asset_id", "asset", "turbine_id", "turbine", "unit_id", "unit"]
    asset_col = None
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in preferred_asset:
        if cand in cols_lower:
            asset_col = cols_lower[cand]
            break

    if asset_col is None:
        stem = Path(csv_path).stem
        # Default: file is like "2.csv" meaning turbine 2
        df["asset_id"] = str(stem)
    else:
        if asset_col != "asset_id":
            df = df.rename(columns={asset_col: "asset_id"})

    df["asset_id"] = df["asset_id"].astype(str).str.strip()
    df["time_stamp"] = pd.to_datetime(df["time_stamp"]).dt.tz_localize(None)

    return df


def create_label_column(df: pd.DataFrame, df_events: pd.DataFrame) -> pd.DataFrame:
    """Add anomaly + coarse fault labels based on event windows.

    fault_label is inferred into a compact taxonomy so the fault head can train even when
    source event labels are heterogeneous/free-text.
    """

    def _infer_fault_label(event_label: str, event_description: str) -> str:
        text = f"{event_label} {event_description}".lower()

        normal_tokens = ["normal", "none", "healthy", "ok", "no fault"]
        if any(tok in text for tok in normal_tokens):
            return "normal"

        groups: list[tuple[str, list[str]]] = [
            ("mechanical", ["gear", "bearing", "shaft", "drivetrain", "vibration", "blade", "yaw", "pitch"]),
            ("electrical", ["generator", "converter", "inverter", "igbt", "voltage", "current", "electrical"]),
            ("thermal", ["temperature", "overheat", "cooling", "thermal", "hot"]),
            ("hydraulic", ["hydraulic", "pressure", "pump", "oil leak", "valve"]),
            ("grid", ["grid", "curtail", "dispatch", "frequency", "voltage dip"]),
            ("sensor", ["sensor", "signal", "telemetry", "scada", "communication", "comms", "missing data"]),
            ("control", ["controller", "control", "setpoint", "trip", "shutdown", "startup", "brake"]),
            ("icing", ["ice", "icing", "frozen"]),
        ]
        for group, toks in groups:
            if any(tok in text for tok in toks):
                return group

        # Keep a generalized fallback class when we cannot infer from event text.
        return "other_fault"

    df = df.copy()
    df["anomaly"] = 0
    df["event_id"] = "-1"
    df["fault_label"] = "normal"

    if len(df) == 0:
        return df

    asset_id = str(df["asset_id"].iloc[0]).strip()

    events = df_events[df_events["asset_id"].astype(str).str.strip() == asset_id]
    if len(events) == 0:
        print(f"  No events found for asset_id={asset_id}")
        return df

    print(f"  Found {len(events)} events for asset_id={asset_id}")

    for _, ev in events.iterrows():
        label = str(ev["event_label"]).strip()
        fault_label = _infer_fault_label(label, str(ev.get("event_description", "")))

        # Ensure scalar timestamps (never Series)
        start = ev["event_start"]
        end = ev["event_end"]
        if isinstance(start, pd.Series):
            start = start.iloc[0]
        if isinstance(end, pd.Series):
            end = end.iloc[0]

        start = pd.to_datetime(start, errors="coerce")
        end = pd.to_datetime(end, errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue

        start = pd.Timestamp(start).tz_localize(None)
        end = pd.Timestamp(end).tz_localize(None)

        mask = (df["time_stamp"] >= start) & (df["time_stamp"] <= end)
        n = int(mask.sum())
        if n > 0:
            # anomaly is binary but fault_label keeps semantic class for fault head.
            if fault_label != "normal":
                df.loc[mask, "anomaly"] = 1
            df.loc[mask, "event_id"] = str(ev["event_id"])
            df.loc[mask, "fault_label"] = fault_label
            print(f"    Event {ev['event_id']}: {n} rows marked as fault_label={fault_label}")

    return df


def flatten_aggregated_columns(df: pd.DataFrame, sensor_rename_map: dict[str, str] | None = None) -> pd.DataFrame:
    """
    Convert sensor_*_{avg,max,min,std} -> sensor_* using avg (fallback to other if avg missing).
    Keep metadata columns.
    """
    meta_cols = [
        "time_stamp",
        "asset_id",
        "id",
        "train_test",
        "status_type_id",
        "anomaly",
        "event_id",
        "fault_label",
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]

    sensor_bases = set()
    for col in df.columns:
        if col in meta_cols:
            continue
        parts = col.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in ["avg", "max", "min", "std"]:
            sensor_bases.add(parts[0])

    print(f"  Found {len(sensor_bases)} unique sensors")

    out = df[meta_cols].copy()
    sensor_rename_map = sensor_rename_map or {}
    for base in sorted(sensor_bases):
        avg_col = f"{base}_avg"
        if avg_col in df.columns:
            series = df[avg_col]
        else:
            series = None
            for suffix in ["max", "min", "std"]:
                alt = f"{base}_{suffix}"
                if alt in df.columns:
                    series = df[alt]
                    break
            if series is None:
                series = pd.Series(np.nan, index=df.index)

        out_name = sensor_rename_map.get(base, base)
        if out_name in out.columns:
            out[out_name] = pd.concat([out[out_name], series], axis=1).mean(axis=1, skipna=True)
        else:
            out[out_name] = series

    return out


# -----------------------------
# Main pipeline
# -----------------------------
def process_dataset(
    data_dir: str,
    event_path: str,
    feature_path: str,
    output_dir: str,
    use_avg_only: bool = True,
) -> None:
    print("Loading event info...")
    df_events = load_event_info(event_path)
    print(f"Loaded {len(df_events)} events")

    print("\nLoading feature descriptions...")
    df_feat = load_feature_descriptions(feature_path)
    angular_sensors = identify_angular_sensors(df_feat)
    sensor_rename_map = build_canonical_sensor_map(df_feat)
    if sensor_rename_map:
        print(f"Built canonical mapping for {len(sensor_rename_map)} sensors")

    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv") and f[:-4].isdigit()])
    print(f"\nFound {len(csv_files)} turbine CSV files")

    if len(csv_files) == 0:
        print(f"ERROR: No CSV files found in {data_dir}")
        return

    all_data: list[pd.DataFrame] = []

    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)
        print(f"\nProcessing {csv_file}...")

        try:
            df = load_turbine_csv(csv_path)
            print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

            df = create_label_column(df, df_events)

            if use_avg_only:
                df = flatten_aggregated_columns(df, sensor_rename_map=sensor_rename_map)
                print(f"  Flattened to {len(df.columns)} columns")

            all_data.append(df)

        except Exception as e:
            print(f"  ERROR processing {csv_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_data) == 0:
        print("\nERROR: No data was successfully processed!")
        return

    print("\nConcatenating all turbines...")
    df_all = pd.concat(all_data, axis=0, ignore_index=True)

    # Downstream expects some canonical names; keep yours consistent:
    df_all = df_all.rename(columns={"asset_id": "turbine_id", "time_stamp": "timestamp"})

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "turbine_data_merged.csv")
    print(f"\nSaving to {out_path}...")
    df_all.to_csv(out_path, index=False)

    print("\n============================================================")
    print(f"SUCCESS: {len(df_all)} rows across {df_all['turbine_id'].nunique()} turbines")
    print(f"timestamp range: {df_all['timestamp'].min()} -> {df_all['timestamp'].max()}")
    if "anomaly" in df_all.columns:
        print("anomaly counts:", df_all["anomaly"].value_counts(dropna=False).to_dict())
    print("============================================================")

    if angular_sensors:
        angular_path = os.path.join(output_dir, "angular_sensors.txt")
        with open(angular_path, "w", encoding="utf-8") as f:
            for s in angular_sensors:
                f.write(f"{s}\n")
        print(f"Saved angular sensors list to {angular_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapt wind turbine dataset format")
    parser.add_argument("--data_dir", required=True, help="Directory containing N.csv files")
    parser.add_argument("--event_path", required=True, help="Path to event_info.csv")
    parser.add_argument("--feature_path", required=True, help="Path to feature_description.csv")
    parser.add_argument("--output_dir", default="data/raw", help="Output directory (prepare_data expects data/raw)")
    parser.add_argument("--keep_all_aggs", action="store_true", help="Keep all aggregations (avg/max/min/std)")
    args = parser.parse_args()

    process_dataset(
        data_dir=args.data_dir,
        event_path=args.event_path,
        feature_path=args.feature_path,
        output_dir=args.output_dir,
        use_avg_only=not args.keep_all_aggs,
    )
