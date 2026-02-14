from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import asdict, is_dataclass
from typing import Any, Literal, Optional, cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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
    if isinstance(cfg, dict):
        return cfg
    return cast(dict, getattr(cfg, "__dict__", {}))


class ScadaWindowDataset(Dataset):
    def __init__(
        self,
        processed_dir: str,
        window_index: pd.DataFrame,
        feature_reg: FeatureRegistry,
        flag_reg: FlagRegistry,
        cfg_data: dict[str, Any],
        cfg_train: Optional[dict[str, Any]],
        mode: Literal["train", "val", "test", "infer"],
        cache_size: int = 8,
    ) -> None:
        self.processed_dir = processed_dir
        self.index = window_index.reset_index(drop=True)
        self.feature_reg = feature_reg
        self.flag_reg = flag_reg
        self.cfg_data = cfg_data
        self.cfg_train = dict(cfg_train or {})
        self.mode = mode
        self.L = int(cfg_data["windowing"]["L"])
        self.K = int(cfg_data["windowing"].get("K", 0))
        self.dt_minutes = int(cfg_data.get("sampling", {}).get("dt_minutes", 10))
        self.label_cfg = dict(cfg_data.get("labels", {}))
        self.class_names = [str(x) for x in self.label_cfg.get("class_names", [])]
        self.C = max(1, len(self.class_names))
        self.horizon_days = [int(x) for x in self.label_cfg.get("horizon_days", [7, 28])]
        self.horizon_include_normal = bool(self.label_cfg.get("horizon_include_normal", False))
        self.cache = _TurbineCache(cache_size)
        self.sample_weights: torch.Tensor | None = None
        self.sample_class_hist: dict[str, int] | None = None
        if self.mode == "train":
            self.sample_weights, self.sample_class_hist = self._build_train_sample_weights()

    def _window_label(self, labels: np.ndarray, start: int) -> np.ndarray:
        rule = str(self.label_cfg.get("label_rule", "last_timestep")).strip().lower()

        input_slice = labels[start : start + self.L]
        target_start = start + self.L
        future_slice = labels[target_start : target_start + self.K] if self.K > 0 else labels[0:0]

        if rule == "last_timestep":
            seg = input_slice[-1:]
        elif rule == "window_any":
            seg = input_slice
        elif rule == "window_any_future_k":
            seg = future_slice
        elif rule == "window_any_full":
            seg = np.concatenate([input_slice, future_slice], axis=0)
        else:
            seg = input_slice[-1:]

        if seg.size == 0:
            return np.array(0, dtype=np.int64)
        seg_i = np.asarray(seg).astype(np.int64).reshape(-1)
        pos = seg_i[seg_i > 0]
        if pos.size == 0:
            return np.array(0, dtype=np.int64)

        vals, counts = np.unique(pos, return_counts=True)
        order = np.lexsort((vals, counts))
        best = vals[order[-1]]
        return np.array(int(best), dtype=np.int64)

    def _future_horizon_targets(self, labels: np.ndarray, start: int) -> np.ndarray:
        """Build multi-hot future occurrence targets for 7/28 day horizons."""
        start_future = start + self.L
        out = []
        for d in self.horizon_days:
            steps = max(1, int(round((d * 24 * 60) / max(self.dt_minutes, 1))))
            seg = np.asarray(labels[start_future : start_future + steps]).astype(np.int64).reshape(-1)
            if self.horizon_include_normal:
                vec = np.zeros((self.C,), dtype=np.float32)
                valid = seg[(seg >= 0) & (seg < self.C)]
            else:
                vec = np.zeros((max(1, self.C - 1),), dtype=np.float32)
                valid = seg[(seg > 0) & (seg < self.C)] - 1
            if valid.size > 0:
                vec[np.unique(valid)] = 1.0
            out.append(vec)
        return np.stack(out, axis=0).astype(np.float32)


    def _build_train_sample_weights(self) -> tuple[torch.Tensor | None, dict[str, int] | None]:
        if len(self.index) == 0:
            return None, None

        labels: list[int] = []
        has_label: list[bool] = []
        for row in self.index.itertuples(index=False):
            tid = str(getattr(row, "turbine_id"))
            start = int(getattr(row, "start_pos"))
            arrs = self._load_turbine_arrays(tid)
            if "labels" not in arrs:
                labels.append(0)
                has_label.append(False)
                continue
            y = int(self._window_label(arrs["labels"], start=start))
            labels.append(y)
            has_label.append(True)

        if not any(has_label):
            return None, None

        y_arr = np.asarray(labels, dtype=np.int64)
        c_max = max(self.C, int(y_arr.max(initial=0)) + 1)
        counts = np.bincount(y_arr, minlength=c_max).astype(np.float64)

        non_zero = counts[counts > 0]
        if non_zero.size == 0:
            return None, None

        fs_cfg = dict(self.cfg_train.get("fault_balanced_sampling", {}))
        strategy = str(fs_cfg.get("strategy", "binary_presence")).strip().lower()

        if strategy == "binary_presence":
            pos_mask = y_arr > 0
            n_pos = int(pos_mask.sum())
            n_neg = int((~pos_mask).sum())
            if n_pos == 0 or n_neg == 0:
                # Fallback to class-frequency weighting if only one side exists.
                strategy = "multiclass_inverse_freq"
            else:
                # Force near 50/50 normal vs non-normal sampling to avoid long
                # all-normal streaks during fault training diagnostics.
                w = np.ones(len(y_arr), dtype=np.float64)
                w[pos_mask] = 0.5 / max(n_pos, 1)
                w[~pos_mask] = 0.5 / max(n_neg, 1)
                hist = {str(i): int(c) for i, c in enumerate(counts.tolist()) if c > 0}
                return torch.as_tensor(w, dtype=torch.double), hist

        inv = np.zeros_like(counts)
        inv[counts > 0] = 1.0 / counts[counts > 0]

        power = float(fs_cfg.get("power", 1.0))
        if power != 1.0:
            inv = np.power(inv, power)

        max_ratio = fs_cfg.get("max_class_weight_ratio", None)
        if max_ratio is not None:
            max_ratio_f = float(max_ratio)
            if max_ratio_f > 0:
                nz = inv[inv > 0]
                if nz.size > 0:
                    min_w = float(nz.min())
                    inv = np.clip(inv, min_w, min_w * max_ratio_f)

        w = np.array([inv[y] if hl else 1.0 for y, hl in zip(y_arr, has_label)], dtype=np.float64)
        hist = {str(i): int(c) for i, c in enumerate(counts.tolist()) if c > 0}
        return torch.as_tensor(w, dtype=torch.double), hist

    def _window_label(self, labels: np.ndarray, start: int) -> np.ndarray:
        rule = str(self.label_cfg.get("label_rule", "last_timestep")).strip().lower()

        input_slice = labels[start : start + self.L]
        target_start = start + self.L
        future_slice = labels[target_start : target_start + self.K] if self.K > 0 else labels[0:0]

        if rule == "last_timestep":
            seg = input_slice[-1:]
        elif rule == "window_any":
            seg = input_slice
        elif rule == "window_any_future_k":
            seg = future_slice
        elif rule == "window_any_full":
            seg = np.concatenate([input_slice, future_slice], axis=0)
        else:
            seg = input_slice[-1:]

        if seg.size == 0:
            return np.array(0, dtype=np.int64)
        seg_i = np.asarray(seg).astype(np.int64).reshape(-1)
        pos = seg_i[seg_i > 0]
        if pos.size == 0:
            return np.array(0, dtype=np.int64)

        vals, counts = np.unique(pos, return_counts=True)
        order = np.lexsort((vals, counts))
        best = vals[order[-1]]
        return np.array(int(best), dtype=np.int64)

    def _future_horizon_targets(self, labels: np.ndarray, start: int) -> np.ndarray:
        """Build multi-hot future occurrence targets for 7/28 day horizons."""
        start_future = start + self.L
        out = []
        for d in self.horizon_days:
            steps = max(1, int(round((d * 24 * 60) / max(self.dt_minutes, 1))))
            seg = np.asarray(labels[start_future : start_future + steps]).astype(np.int64).reshape(-1)
            if self.horizon_include_normal:
                vec = np.zeros((self.C,), dtype=np.float32)
                valid = seg[(seg >= 0) & (seg < self.C)]
            else:
                vec = np.zeros((max(1, self.C - 1),), dtype=np.float32)
                valid = seg[(seg > 0) & (seg < self.C)] - 1
            if valid.size > 0:
                vec[np.unique(valid)] = 1.0
            out.append(vec)
        return np.stack(out, axis=0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.index)

    def _load_turbine_arrays(self, turbine_id: str) -> dict[str, np.ndarray]:
        cached = self.cache.get(turbine_id)
        if cached is not None:
            return cached

        base = os.path.join(self.processed_dir, "turbines", str(turbine_id))
        X = _load_npy(os.path.join(base, "X.npy")).astype("float32")
        M = _load_npy(os.path.join(base, "M_miss.npy")).astype("uint8")
        Flags = _load_npy(os.path.join(base, "Flags.npy")).astype("float32")

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

        Xw = arrs["X"][start : start + self.L]
        Mw = arrs["M"][start : start + self.L]
        Fw = arrs["Flags"][start : start + self.L]

        X = torch.from_numpy(Xw).to(torch.float32)
        M = torch.from_numpy(Mw.astype("float32"))
        Flags = torch.from_numpy(Fw).to(torch.float32)

        times = None
        if "timestamps" in arrs:
            times = torch.from_numpy(arrs["timestamps"][start : start + self.L])

        if self.mode == "infer":
            return InferBatch(X=X, M_miss=M, Flags=Flags, turbine_ids=[tid], times=times)

        Y_true = None
        if self.K > 0:
            tstart = int(row["target_start_pos"])
            Y_true = torch.from_numpy(arrs["X"][tstart : tstart + self.K]).to(torch.float32)

        y = None
        y_horizons = None
        has_label = torch.tensor(0.0, dtype=torch.float32)
        if "labels" in arrs:
            labels_np = arrs["labels"]
            y = torch.from_numpy(np.array(self._window_label(labels_np, start=start)))
            y_horizons = torch.from_numpy(self._future_horizon_targets(labels_np, start=start))
            has_label = torch.tensor(1.0, dtype=torch.float32)

        return TrainBatch(
            X=X,
            M_miss=M,
            Flags=Flags,
            turbine_ids=[tid],
            times=times,
            Y_true=Y_true,
            y=y,
            y_horizons=y_horizons,
            has_label=has_label,
        )


def collate_scada_batches(items: list[TrainBatch | InferBatch]) -> TrainBatch | InferBatch:
    if not items:
        raise ValueError("Empty batch")

    first = items[0]
    is_train = isinstance(first, TrainBatch)

    X = torch.stack([it.X for it in items], dim=0)
    M = torch.stack([it.M_miss for it in items], dim=0)
    Flags = torch.stack([it.Flags for it in items], dim=0)
    turbine_ids = sum([it.turbine_ids or [] for it in items], [])

    times = None
    if all(getattr(it, "times", None) is not None for it in items):
        times = torch.stack([it.times for it in items], dim=0)  # type: ignore[arg-type]

    if not is_train:
        return InferBatch(X=X, M_miss=M, Flags=Flags, turbine_ids=turbine_ids, times=times)

    have_Y = any(getattr(it, "Y_true", None) is not None for it in items)
    if have_Y and not all(getattr(it, "Y_true", None) is not None for it in items):
        raise ValueError("Inconsistent Y_true presence within batch")
    Y_true = torch.stack([it.Y_true for it in items], dim=0) if have_Y else None  # type: ignore[arg-type]

    has_label = torch.stack([getattr(it, "has_label", torch.tensor(0.0)) for it in items], dim=0).to(torch.float32)

    ys = [getattr(it, "y", None) for it in items]
    if all(v is None for v in ys):
        y = None
    else:
        ys2 = [torch.tensor(0, dtype=torch.int64) if v is None else v for v in ys]
        y = torch.stack([v if v.ndim > 0 else v.reshape(()) for v in ys2], dim=0)

    yh = [getattr(it, "y_horizons", None) for it in items]
    if all(v is None for v in yh):
        y_horizons = None
    else:
        template = next((v for v in yh if v is not None), None)
        if template is None:
            y_horizons = None
        else:
            fill = torch.zeros_like(template)
            yh2 = [fill if v is None else v for v in yh]
            y_horizons = torch.stack(yh2, dim=0)

    return TrainBatch(
        X=X,
        M_miss=M,
        Flags=Flags,
        turbine_ids=turbine_ids,
        times=times,
        Y_true=Y_true,
        y=y,
        y_horizons=y_horizons,
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

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg.get("num_workers", 0))

    loaders: dict[str, DataLoader] = {}
    for split_name, df_idx in index_splits.items():
        mode = "infer" if split_name == "infer" else split_name
        ds = ScadaWindowDataset(
            processed_dir=processed_dir,
            window_index=df_idx,
            feature_reg=feature_reg,
            flag_reg=flag_reg,
            cfg_data=cfg["data"],
            cfg_train=cfg.get("train", {}),
            mode=cast(Literal["train", "val", "test", "infer"], mode),
        )
        shuffle = split_name == "train"
        sampler = None
        fs_cfg = dict(cfg.get("train", {}).get("fault_balanced_sampling", {}))
        if split_name == "train" and bool(fs_cfg.get("enabled", False)) and ds.sample_weights is not None:
            sampler = WeightedRandomSampler(
                weights=ds.sample_weights,
                num_samples=len(ds.sample_weights),
                replacement=bool(fs_cfg.get("replacement", True)),
            )
            shuffle = False

        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=False,
            worker_init_fn=seed_worker,
            collate_fn=collate_scada_batches,
        )
    return loaders
