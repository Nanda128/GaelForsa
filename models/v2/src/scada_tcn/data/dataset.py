# src/scada_tcn/data/dataset.py
from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import asdict
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..contracts import InferBatch, TrainBatch
from ..registry import FeatureRegistry, FlagRegistry
from ..utils.seed import seed_worker


def _load_npy(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)


class _TurbineCache:
    def __init__(self, max_items: int) -> None:
        self.max_items = max_items
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, key: str) -> Optional[dict[str, np.ndarray]]:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value: dict[str, np.ndarray]) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)


class ScadaWindowDataset(Dataset):
    def __init__(
        self,
        processed_dir: str,
        window_index: pd.DataFrame,
        feature_reg: FeatureRegistry,
        flag_reg: FlagRegistry,
        cfg_data: dict,
        mode: Literal["train", "val", "test", "infer"],
        cache_size: int = 8,
    ) -> None:
        self.processed_dir = processed_dir
        self.index = window_index.reset_index(drop=True)
        self.feature_reg = feature_reg
        self.flag_reg = flag_reg
        self.cfg_data = cfg_data
        self.mode = mode
        self.L = int(cfg_data["windowing"]["L"])
        self.K = int(cfg_data["windowing"].get("K", 0))
        self.cache = _TurbineCache(cache_size)

    def __len__(self) -> int:
        return len(self.index)

    def _load_turbine_arrays(self, turbine_id: str) -> dict[str, np.ndarray]:
        cached = self.cache.get(turbine_id)
        if cached is not None:
            return cached

        base = os.path.join(self.processed_dir, "turbines", str(turbine_id))
        X = _load_npy(os.path.join(base, "X.npy")).astype("float32")          # (T,F)
        M = _load_npy(os.path.join(base, "M_miss.npy")).astype("uint8")      # (T,F)
        Flags = _load_npy(os.path.join(base, "Flags.npy")).astype("float32") # (T,R)
        ts_path = os.path.join(base, "timestamps.npy")
        timestamps = _load_npy(ts_path) if os.path.exists(ts_path) else None

        if X.shape[1] != len(self.feature_reg.feature_names):
            raise ValueError(f"X F mismatch for {turbine_id}: {X.shape[1]} vs {len(self.feature_reg.feature_names)}")
        if Flags.shape[1] != len(self.flag_reg.flag_names):
            raise ValueError(
                f"Flags R mismatch for {turbine_id}: {Flags.shape[1]} vs {len(self.flag_reg.flag_names)}"
            )
        if M.shape != X.shape:
            raise ValueError(f"M_miss shape mismatch for {turbine_id}: {M.shape} vs {X.shape}")

        pack = {"X": X, "M": M, "Flags": Flags}
        if timestamps is not None:
            pack["timestamps"] = timestamps
        self.cache.put(turbine_id, pack)
        return pack

    def __getitem__(self, idx: int) -> TrainBatch | InferBatch:
        row = self.index.iloc[idx]
        tid = str(row["turbine_id"])
        start = int(row["start_pos"])
        arrs = self._load_turbine_arrays(tid)

        Xw = arrs["X"][start : start + self.L]             # (L,F)
        Mw = arrs["M"][start : start + self.L]             # (L,F)
        Fw = arrs["Flags"][start : start + self.L]         # (L,R)

        X = torch.from_numpy(Xw).to(torch.float32)
        M = torch.from_numpy(Mw.astype("float32"))
        Flags = torch.from_numpy(Fw).to(torch.float32)

        # Times optional (store start_time at least)
        times = None
        if "timestamps" in arrs:
            ts = arrs["timestamps"][start : start + self.L]
            ts = ts.astype("datetime64[ns]").astype("int64")
            times = torch.from_numpy(ts)

        if self.mode == "infer":
            return InferBatch(X=X.unsqueeze(0), M_miss=M.unsqueeze(0), Flags=Flags.unsqueeze(0), turbine_ids=[tid], times=times)

        # TrainBatch: Y_true optional depending on K
        Y_true = None
        if self.K > 0:
            tstart = int(row["target_start_pos"])
            Y_true = torch.from_numpy(arrs["X"][tstart : tstart + self.K]).to(torch.float32).unsqueeze(0)

        return TrainBatch(
            X=X.unsqueeze(0),
            M_miss=M.unsqueeze(0),
            Flags=Flags.unsqueeze(0),
            turbine_ids=[tid],
            times=times,
            Y_true=Y_true,
            y=None,
            has_label=None,
        )


def collate_scada_batches(items: list[TrainBatch | InferBatch]) -> TrainBatch | InferBatch:
    # Dataset currently returns single-item batches (B=1) for simplicity; collate stacks them.
    # This stays correct even if you later return (L,*) samples.
    if not items:
        raise ValueError("Empty batch")

    first = items[0]
    is_train = isinstance(first, TrainBatch)

    X = torch.cat([it.X for it in items], dim=0)
    M = torch.cat([it.M_miss for it in items], dim=0)
    Flags = torch.cat([it.Flags for it in items], dim=0)
    turbine_ids = sum([it.turbine_ids or [] for it in items], [])

    times = None
    if all(getattr(it, "times", None) is not None for it in items):
        times = torch.stack([it.times for it in items], dim=0)  # type: ignore[arg-type]

    if not is_train:
        return InferBatch(X=X, M_miss=M, Flags=Flags, turbine_ids=turbine_ids, times=times)

    Y_true = None
    if any(getattr(it, "Y_true", None) is not None for it in items):
        Y_true = torch.cat([it.Y_true for it in items if it.Y_true is not None], dim=0)  # type: ignore[arg-type]

    return TrainBatch(X=X, M_miss=M, Flags=Flags, turbine_ids=turbine_ids, times=times, Y_true=Y_true)


def make_dataloaders(
    processed_dir: str,
    index_splits: dict[str, pd.DataFrame],
    feature_reg: FeatureRegistry,
    flag_reg: FlagRegistry,
    cfg_root: dict,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    batch_size = int(cfg_root.get("train", {}).get("batch_size", 32)) if "train" in cfg_root else 32

    for split_name, df_idx in index_splits.items():
        mode = "infer" if split_name == "infer" else split_name
        ds = ScadaWindowDataset(
            processed_dir=processed_dir,
            window_index=df_idx,
            feature_reg=feature_reg,
            flag_reg=flag_reg,
            cfg_data=cfg_root["data"],
            mode=mode,  # type: ignore[arg-type]
        )
        shuffle = split_name == "train"
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=int(cfg_root.get("num_workers", 0)),
            pin_memory=False,
            worker_init_fn=seed_worker,
            collate_fn=collate_scada_batches,
        )
    return loaders
