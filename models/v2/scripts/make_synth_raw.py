import os
import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

np.random.seed(0)

turbines = ["T01", "T02"]
start = pd.Timestamp("2021-01-01 00:00:00")
n = 600
dt = pd.Timedelta(minutes=10)
ts = pd.date_range(start, periods=n, freq=dt)

rows = []
for tid in turbines:
    wind = np.clip(8 + 2*np.random.randn(n), 0, 25)
    power = np.clip(500*wind + 200*np.random.randn(n), 0, 5000)
    yaw = (np.cumsum(np.random.randn(n))*0.5) % 360
    wdir = (yaw + 10*np.random.randn(n)) % 360
    pitch = np.clip(2 + 0.1*np.random.randn(n), -5, 30)

    # make missing blocks + random missing
    miss_idx = np.random.choice(n, size=60, replace=False)
    wind[miss_idx] = np.nan
    power[np.random.choice(n, size=40, replace=False)] = np.nan
    yaw[100:120] = np.nan
    wdir[200:210] = np.nan

    df = pd.DataFrame({
        "turbine_id": tid,
        "timestamp": ts,
        "Wind Speed": wind,
        "Wind Direction": wdir,
        "Ambient Temperature": 10 + 5*np.random.randn(n),
        "Rotor Speed": 10 + 2*np.random.randn(n),
        "Generator Speed": 1000 + 50*np.random.randn(n),
        "Generator Torque": 1 + 0.2*np.random.randn(n),
        "Active Power": power,
        "Reactive Power": 50*np.random.randn(n),
        "Blade Pitch Angle": pitch,
        "Gearbox Oil Temperature": 40 + 3*np.random.randn(n),
        "Generator Winding Temperature": 60 + 3*np.random.randn(n),
        "Generator Bearing Temperature": 55 + 3*np.random.randn(n),
        "Converter Temperatures": 50 + 3*np.random.randn(n),
        "Transformer Temperature": 45 + 3*np.random.randn(n),
        "Generator Current": 200 + 20*np.random.randn(n),
        "Voltage": 690 + 5*np.random.randn(n),
        "NacelleYaw": yaw,

        # flags expected by configs/flags/default.yaml
        "curtailment": (np.random.rand(n) < 0.05).astype(float),
        "stop_start": (np.random.rand(n) < 0.03).astype(float),
        "grid_event": (np.random.rand(n) < 0.02).astype(float),
    })

    # duplicate a couple timestamps to exercise dedupe
    dup = df.iloc[50:52].copy()
    rows.append(df)
    rows.append(dup)

df_all = pd.concat(rows, ignore_index=True)
path = os.path.join(RAW_DIR, "synthetic.parquet")
df_all.to_parquet(path, index=False)
print("Wrote", path, "rows=", len(df_all))
