import os
import pandas as pd
import torch

from scada_tcn.registry import load_registry
from scada_tcn.data.dataset import ScadaWindowDataset, collate_scada_batches

PROC = "data/processed"
REG_PATH = os.path.join("artifacts", "registries", "registry.json")

feature_reg, flag_reg = load_registry(REG_PATH)

df_train = pd.read_parquet(os.path.join(PROC, "window_index_train.parquet"))
assert len(df_train) > 0

# minimal cfg_data structure expected by dataset
cfg_data = {
    "windowing": {"L": 144, "K": 12, "stride": 12}
}

ds = ScadaWindowDataset(
    processed_dir=PROC,
    window_index=df_train.head(32),
    feature_reg=feature_reg,
    flag_reg=flag_reg,
    cfg_data=cfg_data,
    mode="train",
)

items = [ds[i] for i in range(8)]
batch = collate_scada_batches(items)

print("Batch X", batch.X.shape, batch.X.dtype)
print("Batch M_miss", batch.M_miss.shape, batch.M_miss.dtype, "min/max", batch.M_miss.min().item(), batch.M_miss.max().item())
print("Batch Flags", batch.Flags.shape, batch.Flags.dtype)
print("Batch Y_true", None if batch.Y_true is None else (batch.Y_true.shape, batch.Y_true.dtype))

B, L, F = batch.X.shape
assert batch.M_miss.shape == (B, L, F)
assert batch.Flags.shape[0] == B and batch.Flags.shape[1] == L
assert torch.isfinite(batch.X).all()
assert torch.isfinite(batch.M_miss).all()
assert torch.isfinite(batch.Flags).all()

# binary-ish check for M_miss
vals = torch.unique(batch.M_miss)
print("unique(M_miss) sample:", vals[:10].tolist())
assert all(v.item() in (0.0, 1.0) for v in vals), "M_miss not 0/1"

if batch.Y_true is not None:
    assert batch.Y_true.shape[0] == B and batch.Y_true.shape[2] == F

print("OK")
