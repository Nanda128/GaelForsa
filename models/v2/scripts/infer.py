from __future__ import annotations

import os
from typing import Any, Dict, List

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from scada_tcn.config_schema import build_config
from scada_tcn.data.dataset import ScadaWindowDataset, collate_scada_batches
from scada_tcn.inference.step import infer_step
from scada_tcn.modeling import MultiTaskTCN
from scada_tcn.registry import FeatureRegistry, FlagRegistry, load_registry
from scada_tcn.utils.seed import make_torch_generator


def _load_registries_from_ckpt(ckpt: dict) -> tuple[FeatureRegistry, FlagRegistry]:
    fr = ckpt["feature_registry"]
    gr = ckpt["flag_registry"]
    return FeatureRegistry(**fr), FlagRegistry(**gr)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    rc = build_config(cfg)

    ckpt_path = os.path.join(rc.output_dir, "checkpoints", "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    feature_reg, flag_reg = _load_registries_from_ckpt(ckpt)

    F = len(feature_reg.feature_names)
    R = len(flag_reg.flag_names)
    K = int(rc.data.windowing.get("K", 0))
    C = int(rc.model.heads.get("fault", {}).get("C", 1))
    F_in = 3 * F + R

    model = MultiTaskTCN(F_in=F_in, F=F, R=R, K=K, C=C, cfg_model=rc.model)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    device = torch.device(str(rc.device))
    model.to(device)

    proc = rc.data.paths["processed_dir"]
    df_test = pd.read_parquet(os.path.join(proc, "window_index_test.parquet"))

    ds = ScadaWindowDataset(
        processed_dir=proc,
        window_index=df_test,
        feature_reg=feature_reg,
        flag_reg=flag_reg,
        cfg_data=rc.data.__dict__,
        mode="infer",
    )
    loader = DataLoader(ds, batch_size=int(rc.train.batch_size), shuffle=False, collate_fn=collate_scada_batches)

    g = make_torch_generator(int(rc.seed) + 999, device=str(device))

    rows: List[Dict[str, Any]] = []
    for batch in loader:
        batch = batch  # InferBatch
        batch.X = batch.X.to(device)
        batch.M_miss = batch.M_miss.to(device)
        batch.Flags = batch.Flags.to(device)

        out = infer_step(
            model=model,
            batch=batch,
            cfg=rc,
            feature_reg=feature_reg,
            flag_reg=flag_reg,
            generator=g,
        )

        s_now = out.s_now.detach().cpu().numpy()
        top_idx = out.top_idx.detach().cpu().numpy()

        for i in range(len(s_now)):
            tid = batch.turbine_ids[i] if batch.turbine_ids else "unknown"
            rows.append(
                {
                    "turbine_id": tid,
                    "s_now": float(s_now[i]),
                    "top_features": [feature_reg.feature_names[j] for j in top_idx[i].tolist()],
                }
            )

    out_dir = os.path.join(rc.output_dir, "infer")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(out_dir, "results.parquet"), index=False)
    print("Wrote:", os.path.join(out_dir, "results.parquet"))


if __name__ == "__main__":
    main()
