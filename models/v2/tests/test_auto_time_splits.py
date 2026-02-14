from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import auto_time_splits as ats


def test_find_raw_files_recurses_nested_temp_dataset_layout(tmp_path: Path) -> None:
    nested = tmp_path / "Farm_A" / "datasets"
    nested.mkdir(parents=True)
    top = tmp_path / "top.csv"
    deep = nested / "11.csv"
    top.write_text("timestamp,turbine_id,val\n2024-01-01,1,1\n", encoding="utf-8")
    deep.write_text("timestamp,turbine_id,val\n2024-01-01,2,1\n", encoding="utf-8")

    files = ats._find_raw_files(str(tmp_path))

    assert str(top) in files
    assert str(deep) in files


def test_read_one_supports_semicolon_csv(tmp_path: Path) -> None:
    p = tmp_path / "raw.csv"
    p.write_text("time_stamp;asset_id;sensor_0_avg\n2023-01-01 00:00:00;11;1.0\n", encoding="utf-8")

    out = ats._read_one(str(p))

    assert list(out.columns) == ["time_stamp", "asset_id", "sensor_0_avg"]
    assert out.iloc[0]["asset_id"] == 11


def test_main_raises_for_missing_raw_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "missing"
    monkeypatch.setattr("sys.argv", ["auto_time_splits.py", "--raw_dir", str(missing)])

    with pytest.raises(ValueError, match="raw_dir not found"):
        ats.main()
