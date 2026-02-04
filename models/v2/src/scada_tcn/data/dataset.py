from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import asdict, is_dataclass
from typing import Any, Literal, Optional, cast

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
    return np.load(path, allow_pickle=False)


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


def _as_dict(cfg: Any) -> dict:
    if is_dataclass(cfg):
        return cast(dict, asdict(cfg))
    if hasattr(cfg, "__dict__"):
        return cast(dict, cfg.__dict__)
    return cast(dict, cfg)


class ScadaWindowDataset(Dataset):
    def __init__(
        self,
        processed_dir: str,
        window_index: pd.DataFrame,
        feature_reg: FeatureRegistry,
        flag_reg: FlagRegistry,
        cfg_data: dict[str, Any],
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

        labels_path = os.path.join(base, "labels.npy")
        labels = _load_npy(labels_path) if os.path.exists(labels_path) else None

        ts_path = os.path.join(base, "timestamps.npy")
        timestamps = _load_npy(ts_path) if os.path.exists(ts_path) else None

        if X.shape[1] != len(self.feature_reg.feature_names):
            raise ValueError(f"X F mismatch for {turbine_id}: {X.shape[1]} vs {len(self.feature_reg.feature_names)}")
        if Flags.shape[1] != len(self.flag_reg.flag_names):
            raise ValueError(f"Flags R mismatch for {turbine_id}: {Flags.shape[1]} vs {len(self.flag_reg.flag_names)}")
        if M.shape != X.shape:
            raise ValueError(f"M_miss shape mismatch for {turbine_id}: {M.shape} vs {X.shape}")

        pack: dict[str, np.ndarray] = {"X": X, "M": M, "Flags": Flags}
        if labels is not None:
            pack["labels"] = labels
        if timestamps is not None:
            pack["timestamps"] = timestamps

        self.cache.put(turbine_id, pack)
        return pack

    def __getitem__(self, idx: int) -> TrainBatch | InferBatch:
        row = self.index.iloc[idx]
        tid = str(row["turbine_id"])
        start = int(row["start_pos"])

        arrs = self._load_turbine_arrays(tid)

        Xw = arrs["X"][start : start + self.L]     # (L,F)
        Mw = arrs["M"][start : start + self.L]     # (L,F)
        Fw = arrs["Flags"][start : start + self.L] # (L,R)

        X = torch.from_numpy(Xw).to(torch.float32)
        M = torch.from_numpy(Mw.astype("float32"))
        Flags = torch.from_numpy(Fw).to(torch.float32)

        times = None
        if "timestamps" in arrs:
            times = torch.from_numpy(arrs["timestamps"][start : start + self.L])

        if self.mode == "infer":
            return InferBatch(
                X=X.unsqueeze(0),
                M_miss=M.unsqueeze(0),
                Flags=Flags.unsqueeze(0),
                turbine_ids=[tid],
                times=times,
            )

        Y_true = None
        if self.K > 0:
            tstart = int(row["target_start_pos"])
            Y_true = torch.from_numpy(arrs["X"][tstart : tstart + self.K]).to(torch.float32)

        y = None
        has_label = None
        if "labels" in arrs:
            # You can choose how to convert timeline labels to window labels.
            # For now: take the last timestep label in the input window.
            lab = arrs["labels"][start + self.L - 1]
            y = torch.from_numpy(np.array(lab))
            has_label = torch.tensor(1.0, dtype=torch.float32)

        return TrainBatch(
            X=X.unsqueeze(0),
            M_miss=M.unsqueeze(0),
            Flags=Flags.unsqueeze(0),
            turbine_ids=[tid],
            times=times,
            Y_true=Y_true.unsqueeze(0) if Y_true is not None else None,
            y=y.unsqueeze(0) if isinstance(y, torch.Tensor) and y.ndim > 0 else y,
            has_label=has_label.unsqueeze(0) if has_label is not None else None,
        )


def collate_scada_batches(items: list[TrainBatch | InferBatch]) -> TrainBatch | InferBatch:
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

    # ---- TrainBatch fields ----
    have_Y = any(getattr(it, "Y_true", None) is not None for it in items)
    if have_Y and not all(getattr(it, "Y_true", None) is not None for it in items):
        raise ValueError("Inconsistent Y_true presence within batch")

    Y_true = None
    if have_Y:
        Y_true = torch.cat([it.Y_true for it in items if it.Y_true is not None], dim=0)  # type: ignore[arg-type]

    have_y = any(getattr(it, "y", None) is not None for it in items)
    y = None
    if have_y:
        # allow missing labels; trainer will handle has_label if provided
        ys = [it.y for it in items]
        if all(v is None for v in ys):
            y = None
        else:
            ys2 = [v if v is not None else torch.tensor(0) for v in ys]  # placeholder
            y = torch.stack([v if v.ndim == 0 else v.squeeze(0) for v in ys2], dim=0)

    have_hl = any(getattr(it, "has_label", None) is not None for it in items)
    has_label = None
    if have_hl:
        hs = [it.has_label if it.has_label is not None else torch.tensor([0.0]) for it in items]
        has_label = torch.cat(hs, dim=0)

    return TrainBatch(
        X=X,
        M_miss=M,
        Flags=Flags,
        turbine_ids=turbine_ids,
        times=times,
        Y_true=Y_true,
        y=y,
        has_label=has_label,
    )


def make_dataloaders(
    processed_dir: str,
    index_splits: dict[str, pd.DataFrame],
    feature_reg: FeatureRegistry,
    flag_reg: FlagRegistry,
    cfg_root: Any,
) -> dict[str, DataLoader]:
    cfg = _as_dict(cfg_root)
    loaders: dict[str, DataLoader] = {}

    batch_size = int(cfg.get("train", {}).get("batch_size", 32))
    num_workers = int(cfg.get("num_workers", 0))

    for split_name, df_idx in index_splits.items():
        mode = "infer" if split_name == "infer" else split_name
        ds = ScadaWindowDataset(
            processed_dir=processed_dir,
            window_index=df_idx,
            feature_reg=feature_reg,
            flag_reg=flag_reg,
            cfg_data=cfg["data"],
            mode=cast(Literal["train", "val", "test", "infer"], mode),
        )
        shuffle = split_name == "train"
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
            worker_init_fn=seed_worker,
            collate_fn=collate_scada_batches,
        )

    return loaders
