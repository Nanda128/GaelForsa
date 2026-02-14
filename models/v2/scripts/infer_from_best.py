#!/usr/bin/env python
"""Simple one-command inference entrypoint for an existing best.pt checkpoint.

Example:
  PYTHONPATH=src python scripts/infer_from_best.py \
    --checkpoint /path/to/best.pt \
    --raw-dir temp_dataset
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}:{env['PYTHONPATH']}"
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def _ensure_processed(raw_dir: str, splits: str) -> None:
    proc = REPO_ROOT / "data" / "processed" / "wind_turbine"
    required = [
        proc / "window_index_train.parquet",
        proc / "window_index_val.parquet",
        proc / "window_index_test.parquet",
    ]
    if all(p.exists() for p in required):
        print("Processed window indices already exist; skipping prepare_data.")
        return

    print("Processed data missing. Running prepare_data.py ...")
    _run(
        [
            sys.executable,
            "scripts/prepare_data.py",
            f"data.paths.raw_dir={raw_dir}",
            "data.splits.method=time_auto",
            f"data.splits.ratios=[{splits}]",
        ]
    )


def _stage_checkpoint(src_ckpt: Path, output_dir: str) -> Path:
    dst = REPO_ROOT / output_dir / "checkpoints" / "best.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src_ckpt.resolve() == dst.resolve():
        print(f"Checkpoint already in place: {dst}")
        return dst

    shutil.copy2(src_ckpt, dst)
    print(f"Copied checkpoint to: {dst}")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description="Run inference from an existing best.pt checkpoint")
    ap.add_argument("--checkpoint", required=True, help="Path to best.pt")
    ap.add_argument("--raw-dir", default="temp_dataset", help="Raw dataset root used for prepare_data")
    ap.add_argument("--output-dir", default="artifacts/wind_turbine", help="Model artifact output dir")
    ap.add_argument(
        "--splits",
        default="0.7,0.15,0.15",
        help="Time-auto split ratios for prepare_data.py, comma separated",
    )
    ap.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip prepare_data.py even if processed windows are missing",
    )
    ap.add_argument(
        "--infer-overrides",
        default="",
        help="Extra Hydra overrides passed through to scripts/infer.py",
    )
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if not args.skip_prepare:
        _ensure_processed(raw_dir=args.raw_dir, splits=args.splits)

    _stage_checkpoint(ckpt, args.output_dir)

    cmd = [sys.executable, "scripts/infer.py", f"output_dir={args.output_dir}"]
    if args.infer_overrides.strip():
        cmd.extend(args.infer_overrides.strip().split())
    _run(cmd)

    print("\nInference complete.")
    print(f"Results parquet: {args.output_dir}/infer/results.parquet")
    print(f"Threshold json:  {args.output_dir}/infer/thresholds.json")


if __name__ == "__main__":
    main()
