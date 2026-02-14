# tcn-model

## Run inference from an existing `best.pt`

If you already have a trained checkpoint (use best.pt in artifacts/checkpoints), use this helper script:

```bash
PYTHONPATH=src python scripts/infer_from_best.py \
  --checkpoint /absolute/path/to/best.pt \ 
  --raw-dir dataset
```

What it does automatically:
- verifies / stages your checkpoint into `artifacts/wind_turbine/checkpoints/best.pt`
- runs `prepare_data.py` if processed window indices are missing
- runs `scripts/infer.py`
- writes outputs to `artifacts/wind_turbine/infer/`

Useful flags:
- `--output-dir artifacts/wind_turbine` (default)
- `--skip-prepare` (skip data prep)
- `--infer-overrides "train.batch_size=64 device=cpu"` (forward extra Hydra overrides)
