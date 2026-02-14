#!/bin/bash
set -euo pipefail

echo "=== Wind Turbine Anomaly Detection Pipeline ==="
echo ""
echo "🧭 NEW USER QUICK START"
echo "This script runs 6 steps: adapt -> prepare -> train -> infer -> evaluate -> fault-test"
echo "You only need your dataset folder (event/feature files can be auto-discovered per farm)."
echo ""

# Ensure local package imports resolve for scripts/*.
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

# Install runtime dependencies unless caller opts out.
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  echo "Installing dependencies from requirements.txt ..."
  if ! python -m pip install -r requirements.txt; then
    echo "WARNING: dependency install failed; continuing with current environment (set SKIP_INSTALL=1 to silence)." >&2
  fi
fi

# Configuration (can be overridden via env vars)
DATA_DIR="${DATA_DIR:-temp_dataset}"
EVENT_PATH="${EVENT_PATH:-event_info.csv}"
FEATURE_PATH="${FEATURE_PATH:-feature_description.csv}"
RUN_MODE="${RUN_MODE:-auto}"   # auto|adapt|direct

if [[ "$RUN_MODE" == "auto" ]]; then
  if [[ -d "$DATA_DIR" ]] && {
    find "$DATA_DIR" -mindepth 2 -maxdepth 2 -type d -name datasets | grep -q . ||
    find "$DATA_DIR" -type f -name event_info.csv | grep -q .;
  }; then
    RUN_MODE="direct"
  else
    RUN_MODE="adapt"
  fi
fi

if [[ "$RUN_MODE" == "adapt" ]]; then
  if [[ ! -f "$EVENT_PATH" || ! -f "$FEATURE_PATH" ]]; then
    AUTO_EVENT_PATH="$(find "$DATA_DIR" -type f -name event_info.csv | head -n 1 || true)"
    AUTO_FEATURE_PATH="$(find "$DATA_DIR" -type f -name feature_description.csv | head -n 1 || true)"

    if [[ -n "$AUTO_EVENT_PATH" && -n "$AUTO_FEATURE_PATH" ]]; then
      echo "Step 1: Detected per-farm metadata files in $DATA_DIR; switching to direct mode (no manual EVENT_PATH/FEATURE_PATH required)."
      RUN_MODE="direct"
    else
      if [[ ! -f "$EVENT_PATH" ]]; then
        echo "ERROR: EVENT_PATH not found: $EVENT_PATH" >&2
      fi
      if [[ ! -f "$FEATURE_PATH" ]]; then
        echo "ERROR: FEATURE_PATH not found: $FEATURE_PATH" >&2
      fi
      echo "Hint: for per-farm datasets containing event_info.csv and feature_description.csv in farm subfolders, use RUN_MODE=auto or RUN_MODE=direct." >&2
      exit 1
    fi
  fi
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: DATA_DIR not found: $DATA_DIR" >&2
  exit 1
fi

if [[ "$RUN_MODE" == "adapt" ]]; then
  # Step 1: Adapt the data format
  echo "Step 1: Adapting dataset format..."
  python scripts/adapt_turbine_data.py \
    --data_dir "$DATA_DIR" \
    --event_path "$EVENT_PATH" \
    --feature_path "$FEATURE_PATH" \
    --output_dir "data/raw"
else
  echo "Step 1: Skipped adapt step (direct temp_dataset mode detected)"
fi

# Step 2: Prepare data (QC, features, windowing)
echo "Step 2: Preparing data..."
RAW_DIR_ARG="data/raw"
if [[ "$RUN_MODE" == "direct" ]]; then
  RAW_DIR_ARG="$DATA_DIR"
fi
python scripts/auto_time_splits.py --raw_dir "$RAW_DIR_ARG" --ratios "${SPLIT_RATIOS:-0.7,0.15,0.15}" || true
if [[ -n "${PREPARE_ARGS:-}" ]]; then
  # shellcheck disable=SC2086
  python scripts/prepare_data.py data.paths.raw_dir="$RAW_DIR_ARG" data.splits.method=time_auto data.splits.ratios="[${SPLIT_RATIOS:-0.7,0.15,0.15}]" ${PREPARE_ARGS}
else
  python scripts/prepare_data.py data.paths.raw_dir="$RAW_DIR_ARG" data.splits.method=time_auto data.splits.ratios="[${SPLIT_RATIOS:-0.7,0.15,0.15}]"
fi

# Step 3: Train model
echo ""
echo "🎯 Step 3/6: Training model (watch losses/progress bars)..."
if [[ -n "${TRAIN_ARGS:-}" ]]; then
  # shellcheck disable=SC2086
  python scripts/train.py ${TRAIN_ARGS}
else
  python scripts/train.py
fi

# Step 4: Run inference
echo ""
echo "🔮 Step 4/6: Running inference (risk probabilities + anomaly scores)..."
if [[ -n "${INFER_ARGS:-}" ]]; then
  # shellcheck disable=SC2086
  python scripts/infer.py ${INFER_ARGS}
else
  python scripts/infer.py
fi

# Step 5: Evaluate
echo ""
echo "📊 Step 5/6: Evaluating summary reports..."
if [[ -n "${EVAL_ARGS:-}" ]]; then
  # shellcheck disable=SC2086
  python scripts/evaluate.py ${EVAL_ARGS}
else
  python scripts/evaluate.py
fi

# Step 6: Explicit fault-window tests
echo ""
echo "🧪 Step 6/6: Running explicit fault-head tests on fault-test windows..."
if [[ -n "${FAULT_TEST_ARGS:-}" ]]; then
  # shellcheck disable=SC2086
  python scripts/test_fault_windows.py ${FAULT_TEST_ARGS}
else
  python scripts/test_fault_windows.py
fi

echo ""
echo "=== Pipeline complete! ==="
echo "Results in: artifacts/wind_turbine/"
echo ""
echo "What to open next:"
echo "  • artifacts/wind_turbine/infer/results.parquet   (main predictions)"
echo "  • artifacts/wind_turbine/reports/                 (summary reports)"
echo "  • artifacts/wind_turbine/reports/fault_window_test_metrics.json (explicit fault-test-window metrics)"
echo "  • artifacts/wind_turbine/logs/train.jsonl         (detailed train logs)"
