from __future__ import annotations

from pathlib import Path

import pandas as pd

from scada_tcn.config_schema import DataConfig
from scada_tcn.data.dataset import ScadaWindowDataset
from scada_tcn.data.qc_align import load_raw_tables
from scada_tcn.registry import FeatureRegistry, FlagRegistry


def _cfg() -> DataConfig:
    return DataConfig(
        paths={},
        sampling={"dt_minutes": 10},
        windowing={"L": 2, "K": 1, "stride": 1},
        splits={"method": "time_auto", "ratios": [0.7, 0.15, 0.15]},
        identifiers={"turbine_id_col": "turbine_id", "time_col": "timestamp"},
        labels={"enabled": True, "label_col": "fault_label", "class_names": ["normal", "thermal", "other_fault"]},
        qc={},
    )


def test_load_raw_tables_reads_nested_farms_and_applies_event_labels(tmp_path: Path) -> None:
    farm_a = tmp_path / "Farm_A"
    farm_b = tmp_path / "Farm_B"
    (farm_a / "datasets").mkdir(parents=True)
    (farm_b / "datasets").mkdir(parents=True)

    df_a = pd.DataFrame(
        {
            "time_stamp": ["2023-01-01 00:00:00", "2023-01-01 00:10:00"],
            "asset_id": ["11", "11"],
            "sensor_0_avg": [1.0, 2.0],
        }
    )
    df_b = pd.DataFrame(
        {
            "time_stamp": ["2023-01-01 00:00:00", "2023-01-01 00:10:00"],
            "asset_id": ["11", "11"],
            "sensor_0_avg": [3.0, 4.0],
        }
    )
    df_a.to_csv(farm_a / "datasets" / "11.csv", sep=";", index=False)
    df_b.to_csv(farm_b / "datasets" / "11.csv", sep=";", index=False)

    pd.DataFrame(
        {
            "asset": ["11"],
            "event_id": ["68"],
            "event_label": ["anomaly"],
            "event_start": ["2023-01-01 00:00:00"],
            "event_end": ["2023-01-01 00:05:00"],
            "event_description": ["transformer high temperature"],
        }
    ).to_csv(farm_a / "event_info.csv", sep=";", index=False)
    pd.DataFrame(
        {
            "asset_id": ["11"],
            "event_id": ["7"],
            "event_label": ["anomaly"],
            "event_start": ["2023-01-01 00:00:00"],
            "event_end": ["2023-01-01 00:05:00"],
            "event_description": ["bearing vibration"],
        }
    ).to_csv(farm_b / "event_info.csv", sep=";", index=False)

    out = load_raw_tables(str(tmp_path), _cfg())

    assert "fault_label" in out.columns
    assert "event_id" in out.columns
    assert out["turbine_id"].str.contains("Farm_A::").any()
    assert out["turbine_id"].str.contains("Farm_B::").any()

    farm_a_rows = out[out["turbine_id"].str.startswith("Farm_A::")]
    farm_b_rows = out[out["turbine_id"].str.startswith("Farm_B::")]
    assert (farm_a_rows["fault_label"] == "thermal").any()
    assert (farm_b_rows["fault_label"] == "mechanical").any()


def test_train_sampler_weights_use_train_cfg(tmp_path: Path) -> None:
    base = tmp_path / "processed" / "turbines" / "T1"
    base.mkdir(parents=True)

    import numpy as np

    np.save(base / "X.npy", np.random.randn(8, 2).astype("float32"))
    np.save(base / "M_miss.npy", np.ones((8, 2), dtype="uint8"))
    np.save(base / "Flags.npy", np.zeros((8, 1), dtype="float32"))
    np.save(base / "labels.npy", np.array([0, 0, 0, 0, 1, 1, 0, 1], dtype="int64"))

    idx = pd.DataFrame(
        {
            "turbine_id": ["T1", "T1", "T1"],
            "start_pos": [0, 2, 4],
            "target_start_pos": [2, 4, 6],
        }
    )

    ds = ScadaWindowDataset(
        processed_dir=str(tmp_path / "processed"),
        window_index=idx,
        feature_reg=FeatureRegistry(feature_names=["f1", "f2"], raw_channel_names=["f1", "f2"], angular_channel_names=[]),
        flag_reg=FlagRegistry(flag_names=["r1"]),
        cfg_data={
            "windowing": {"L": 2, "K": 1},
            "sampling": {"dt_minutes": 10},
            "labels": {"label_rule": "window_any", "class_names": ["normal", "fault"]},
        },
        cfg_train={"fault_balanced_sampling": {"enabled": True, "power": 2.0, "max_class_weight_ratio": 100.0}},
        mode="train",
    )

    assert ds.sample_weights is not None
    w = ds.sample_weights.numpy()
    assert len(w) == 3
    assert w.max() > w.min()
