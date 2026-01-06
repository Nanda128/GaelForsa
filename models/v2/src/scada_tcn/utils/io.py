from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Callable, Optional

import pandas as pd
import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def atomic_write(path: str, write_fn: Callable[[str], None]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(path) or ".")
    os.close(fd)
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def save_json(obj: dict, path: str) -> None:
    def _write(p: str) -> None:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)

    atomic_write(path, _write)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_torch(obj: dict, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    torch.save(obj, path)


def load_torch(path: str, map_location: Optional[str] = None) -> dict:
    return torch.load(path, map_location=map_location)


def save_parquet(df: pd.DataFrame, path: str, partition_cols: Optional[list[str]] = None) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    if partition_cols:
        df.to_parquet(path, index=True, partition_cols=partition_cols)
    else:
        df.to_parquet(path, index=True)


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
