from __future__ import annotations

import os

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig

from scada_tcn.config_schema import build_config, resolved_feature_names, resolved_flag_names
from scada_tcn.data.dataset import make_dataloaders
from scada_tcn.modeling import MultiTaskTCN
from scada_tcn.registry import load_registry
from scada_tcn.training.trainer import fit


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    rc = build_config(cfg)

    reg_path = os.path.join(rc.output_dir, "registries", "registry.json")
    feature_reg, flag_reg = load_registry(reg_path)

    # hard check ordering vs cfg-derived names (optional but recommended)
    assert feature_reg.feature_names == resolved_feature_names(rc)
    assert flag_reg.flag_names == resolved_flag_names(rc)

    proc = rc.data.paths["processed_dir"]
    df_train = pd.read_parquet(os.path.join(proc, "window_index_train.parquet"))
    df_val = pd.read_parquet(os.path.join(proc, "window_index_val.parquet"))
    df_test = pd.read_parquet(os.path.join(proc, "window_index_test.parquet"))

    loaders = make_dataloaders(
        processed_dir=proc,
        index_splits={"train": df_train, "val": df_val, "test": df_test},
        feature_reg=feature_reg,
        flag_reg=flag_reg,
        cfg_root=rc,
    )

    F = len(feature_reg.feature_names)
    R = len(flag_reg.flag_names)
    K = int(rc.data.windowing.get("K", 0))
    C = int(rc.model.heads.get("fault", {}).get("C", 1))

    F_in = 3 * F + R

    model = MultiTaskTCN(F_in=F_in, F=F, R=R, K=K, C=C, cfg_model=rc.model)

    best_ckpt = fit(
        model=model,
        loaders=loaders,
        cfg=rc,
        feature_reg=feature_reg,
        flag_reg=flag_reg,
        output_dir=rc.output_dir,
    )

    print("Best checkpoint:", best_ckpt)


if __name__ == "__main__":
    main()
