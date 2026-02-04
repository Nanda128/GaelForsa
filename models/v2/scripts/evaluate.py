from __future__ import annotations

import os

import hydra
import pandas as pd
from omegaconf import DictConfig

from scada_tcn.config_schema import build_config
from scada_tcn.evaluation.reports import write_summary_reports


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    rc = build_config(cfg)

    infer_path = os.path.join(rc.output_dir, "infer", "results.parquet")
    df = pd.read_parquet(infer_path)

    metrics = {
        "n_rows": int(len(df)),
        "mean_s_now": float(df["s_now"].mean()) if len(df) else float("nan"),
    }

    out_dir = os.path.join(rc.output_dir, "reports")
    os.makedirs(out_dir, exist_ok=True)
    write_summary_reports(out_dir, rc, metrics, examples=None)
    print("Wrote reports to:", out_dir)


if __name__ == "__main__":
    main()
