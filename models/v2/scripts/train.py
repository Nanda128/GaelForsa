# scripts/train.py
from __future__ import annotations

import os

import hydra
import pandas as pd
from omegaconf import DictConfig

from scada_tcn.config_schema import build_config
from scada_tcn.data.dataset import make_dataloaders
from scada_tcn.modeling import MultiTaskTCN
from scada_tcn.registry import load_registry
from scada_tcn.utils.io import load_json
from scada_tcn.training.trainer import fit
from scada_tcn.utils.logging import get_logger, log_kv


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    rc = build_config(cfg)
    logger = get_logger("train_entry", output_dir=rc.output_dir)

    # Load registry produced by prepare_data.py (authoritative ordering)
    reg_path = os.path.join(rc.output_dir, "registries", "registry.json")
    feature_reg, flag_reg = load_registry(reg_path)

    proc = rc.data.paths["processed_dir"]
    df_train = pd.read_parquet(os.path.join(proc, "window_index_train.parquet"))
    df_val = pd.read_parquet(os.path.join(proc, "window_index_val.parquet"))
    df_test = pd.read_parquet(os.path.join(proc, "window_index_test.parquet"))

    log_kv(
        logger,
        step=0,
        payload={
            "event": "dataset_index_summary",
            "train_windows": int(len(df_train)),
            "val_windows": int(len(df_val)),
            "test_windows": int(len(df_test)),
            "processed_dir": proc,
        },
    )

    loaders = make_dataloaders(
        processed_dir=proc,
        index_splits={"train": df_train, "val": df_val, "test": df_test},
        feature_reg=feature_reg,
        flag_reg=flag_reg,
        cfg_root=rc,
    )

    train_ds = loaders["train"].dataset
    if getattr(train_ds, "sample_class_hist", None) is not None:
        log_kv(
            logger,
            step=0,
            payload={
                "event": "train_label_distribution",
                "class_hist": dict(train_ds.sample_class_hist),
                "balanced_sampler_enabled": bool(rc.train.fault_balanced_sampling.get("enabled", False)),
            },
        )

    F = len(feature_reg.feature_names)
    R = len(flag_reg.flag_names)
    K = int(rc.data.windowing.get("K", 0))
    C = int(rc.model.heads.get("fault", {}).get("C", 1))
    label_map_path = os.path.join(rc.output_dir, "registries", "fault_label_map.json")
    if os.path.exists(label_map_path):
        lm = load_json(label_map_path)
        C = len(lm.get("class_names", [])) or C
    F_in = 3 * F + R

    model = MultiTaskTCN(F_in=F_in, F=F, R=R, K=K, C=C, cfg_model=rc.model)

    log_kv(
        logger,
        step=0,
        payload={
            "event": "model_contract",
            "F": F,
            "R": R,
            "K": K,
            "C": C,
            "F_in": F_in,
            "tcn": dict(rc.model.tcn),
            "heads": dict(rc.model.heads),
            "train": {
                "batch_size": int(rc.train.batch_size),
                "epochs": int(rc.train.epochs),
                "log_every": int(rc.train.log_every),
                "p_mask": float(rc.train.p_mask),
                "loss": dict(rc.train.loss),
            },
            "infer": {
                "p_score": float(rc.infer.p_score),
                "aggregation": dict(rc.infer.aggregation),
                "topn_features": int(rc.infer.topn_features),
            },
        },
    )

    best_ckpt = fit(
        model=model,
        loaders=loaders,
        cfg=rc,
        feature_reg=feature_reg,
        flag_reg=flag_reg,
        output_dir=rc.output_dir,
    )

    log_kv(logger, step=0, payload={"event": "train_complete", "best_ckpt": str(best_ckpt)})
    print("Best checkpoint:", best_ckpt)


if __name__ == "__main__":
    main()
