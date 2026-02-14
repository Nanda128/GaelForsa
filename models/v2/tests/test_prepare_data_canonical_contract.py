from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from scada_tcn.config_schema import FeatureConfig
from scada_tcn.data.qc_align import load_raw_tables
from scada_tcn.config_schema import DataConfig
from scripts.prepare_data import SPEC_ANGULAR_CHANNELS, SPEC_SCADA_CHANNELS, _resolve_feature_config_for_dataset


class _DummyLogger:
    def info(self, _msg):
        return None


def _rc() -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            identifiers={"turbine_id_col": "turbine_id", "time_col": "timestamp"},
            labels={"label_col": "fault_label"},
        ),
        features=FeatureConfig(
            raw_channels=[],
            angular_channels=[],
            angle_units="deg",
            yaw_error={"enabled": False, "wind_dir_col": "Wind Direction", "nacelle_yaw_col": "NacelleYaw"},
            deltas={"enabled": False, "channels": []},
            rolling={"enabled": False, "windows": [3], "stats": ["mean"], "channels": [], "causal": True},
            fill_value_c=0.0,
            scaler={"type": "robust", "eps": 1e-6, "clip_iqr": None, "scale_sincos": False},
            output_ordering={"explicit": False, "feature_names": []},
        ),
    )


def test_auto_mode_enforces_fixed_canonical_contract() -> None:
    df = pd.DataFrame(
        {
            "turbine_id": ["T1", "T1"],
            "timestamp": pd.to_datetime(["2023-01-01 00:00:00", "2023-01-01 00:10:00"]),
            "fault_label": ["normal", "normal"],
            "Wind Speed": [1.0, 2.0],
            "some_other_feature": [10.0, 20.0],
        }
    )

    feat_cfg = _resolve_feature_config_for_dataset(_rc(), df, extra_cols=["fault_label"], logger=_DummyLogger())

    assert feat_cfg.raw_channels == SPEC_SCADA_CHANNELS
    assert feat_cfg.angular_channels == SPEC_ANGULAR_CHANNELS
    assert feat_cfg.yaw_error["enabled"] is True


def test_load_raw_tables_applies_direct_alias_mapping(tmp_path) -> None:
    farm = tmp_path / "Farm_A" / "datasets"
    farm.mkdir(parents=True)
    pd.DataFrame(
        {
            "time_stamp": ["2023-01-01 00:00:00", "2023-01-01 00:10:00"],
            "asset_id": ["7", "7"],
            "wind_speed": [5.0, 6.0],
            "blade_pitch": [1.5, 1.7],
        }
    ).to_csv(farm / "7.csv", index=False)

    cfg = DataConfig(
        paths={},
        sampling={"dt_minutes": 10},
        windowing={"L": 2, "K": 1, "stride": 1},
        splits={"method": "time_auto", "ratios": [0.7, 0.15, 0.15]},
        identifiers={"turbine_id_col": "turbine_id", "time_col": "timestamp"},
        labels={"enabled": False},
        qc={},
    )

    out = load_raw_tables(str(tmp_path), cfg)
    assert "Wind Speed" in out.columns
    assert "Blade Pitch Angle" in out.columns
