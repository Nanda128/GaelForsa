import os
import numpy as np
import torch
import pandas as pd

from scada_tcn.registry import load_registry
from scada_tcn.data.dataset import ScadaWindowDataset, collate_scada_batches

PROC = "data/processed"
REG_PATH = os.path.join("artifacts", "registries", "registry.json")

feature_reg, flag_reg = load_registry(REG_PATH)
F = len(feature_reg.feature_names)
R = len(flag_reg.flag_names)

df_train = pd.read_parquet(os.path.join(PROC, "window_index_train.parquet"))
assert len(df_train) > 0

cfg_data = {"windowing": {"L": 144, "K": 12, "stride": 12}}

ds = ScadaWindowDataset(
    processed_dir=PROC,
    window_index=df_train,
    feature_reg=feature_reg,
    flag_reg=flag_reg,
    cfg_data=cfg_data,
    mode="train",
)

batch = collate_scada_batches([ds[i] for i in range(8)])

# --- Contract checks ---
B, L, F_ = batch.X.shape
assert (B, L, F_) == (8, 144, F)
assert batch.M_miss.shape == (B, L, F)
assert batch.Flags.shape == (B, L, R)
assert batch.Y_true is not None and batch.Y_true.shape == (B, 12, F)

# --- Dtype + finiteness ---
assert batch.X.dtype == torch.float32
assert batch.M_miss.dtype == torch.float32
assert batch.Flags.dtype == torch.float32
assert batch.Y_true.dtype == torch.float32
assert torch.isfinite(batch.X).all()
assert torch.isfinite(batch.M_miss).all()
assert torch.isfinite(batch.Flags).all()
assert torch.isfinite(batch.Y_true).all()

# --- Mask semantics ---
u = torch.unique(batch.M_miss).tolist()
assert set(u).issubset({0.0, 1.0}), f"M_miss not binary: {u}"

# Missing values in X should be filled (default 0.0) at positions where M_miss == 0
fill = 0.0
missing = (batch.M_miss == 0.0)
if missing.any():
    frac = (batch.X[missing] == fill).float().mean().item()
    print("fraction filled at missing positions:", frac)
    assert frac > 0.95, "X not filled with c at missing positions"

# --- Flag sanity ---
# Flags should be 0/1-ish too (since column-based in synth)
fu = torch.unique(batch.Flags).tolist()
assert all((v in (0.0, 1.0)) for v in fu), f"Flags not binary-ish: {fu}"

print("SEMANTICS OK")
