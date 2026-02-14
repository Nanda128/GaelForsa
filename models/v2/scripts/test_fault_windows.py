from __future__ import annotations

import os

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from scada_tcn.config_schema import build_config
from scada_tcn.data.dataset import ScadaWindowDataset, collate_scada_batches
from scada_tcn.evaluation.metrics import compute_fault_metrics
from scada_tcn.training.masking import build_model_input
from scada_tcn.modeling import MultiTaskTCN
from scada_tcn.registry import FeatureRegistry, FlagRegistry
from scada_tcn.utils.io import load_json, save_json


def _load_registries_from_ckpt(ckpt: dict) -> tuple[FeatureRegistry, FlagRegistry]:
    fr = ckpt["feature_registry"]
    gr = ckpt["flag_registry"]
    return FeatureRegistry(**fr), FlagRegistry(**gr)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    rc = build_config(cfg)

    ckpt_path = os.path.join(rc.output_dir, "checkpoints", "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint for fault-window test: {ckpt_path}")

    proc = rc.data.paths["processed_dir"]
    idx_path = os.path.join(proc, "window_index_test.parquet")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"Missing test window index: {idx_path}")

    df_test = pd.read_parquet(idx_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    feature_reg, flag_reg = _load_registries_from_ckpt(ckpt)

    class_names: list[str] = []
    label_map_path = os.path.join(rc.output_dir, "registries", "fault_label_map.json")
    if os.path.exists(label_map_path):
        lm = load_json(label_map_path)
        class_names = [str(x) for x in lm.get("class_names", [])]

    F = len(feature_reg.feature_names)
    R = len(flag_reg.flag_names)
    K = int(rc.data.windowing.get("K", 0))
    C = int(rc.model.heads.get("fault", {}).get("C", 1))
    F_in = 3 * F + R

    model = MultiTaskTCN(F_in=F_in, F=F, R=R, K=K, C=C, cfg_model=rc.model)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = ScadaWindowDataset(
        processed_dir=proc,
        window_index=df_test,
        feature_reg=feature_reg,
        flag_reg=flag_reg,
        cfg_data=(rc.data.__dict__ if hasattr(rc.data, "__dict__") else rc.data),
        cfg_train=(rc.train.__dict__ if hasattr(rc.train, "__dict__") else rc.train),
        mode="test",
    )
    dl = DataLoader(
        ds,
        batch_size=int(rc.train.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_scada_batches,
    )

    device = torch.device(str(rc.device))
    model = model.to(device)

    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dl:
            if batch.y is None:
                continue
            X = batch.X.to(device)
            M = batch.M_miss.to(device)
            Flags = batch.Flags.to(device)
            has_label = batch.has_label.to(device)
            M_ones = torch.ones_like(M)
            X_in = build_model_input(X, M, M_ones, Flags)
            out = model(X_in, return_recon=False, return_forecast=False, return_fault=True)
            if out.p_fault is None:
                continue

            mask = has_label > 0.5
            if not torch.any(mask):
                continue
            logits_all.append(out.p_fault[mask].detach().cpu().numpy())
            labels_all.append(batch.y[mask].detach().cpu().numpy())

    metrics: dict[str, float | int | str] = {
        "n_test_windows": int(len(df_test)),
    }

    if logits_all and labels_all:
        logits = np.concatenate(logits_all, axis=0)
        labels = np.concatenate(labels_all, axis=0).astype(np.int64).reshape(-1)

        pred = np.argmax(logits, axis=1)
        fault_mask = labels > 0

        metrics["n_labeled_windows"] = int(len(labels))
        metrics["n_fault_test_windows"] = int(fault_mask.sum())
        metrics["fault_window_ratio"] = float(fault_mask.mean())

        u, c = np.unique(labels, return_counts=True)
        label_hist = {str(int(k)): int(v) for k, v in zip(u.tolist(), c.tolist())}
        metrics["test_label_hist"] = label_hist
        if class_names:
            named_hist = {class_names[int(k)] if int(k) < len(class_names) else f"class_{k}": int(v) for k, v in label_hist.items()}
            metrics["test_label_hist_named"] = named_hist

        overall = compute_fault_metrics(y_true=labels, logits=logits, multilabel=False)
        metrics["fault_acc_all_test_windows"] = float(overall.get("accuracy", float("nan")))

        if np.any(fault_mask):
            labels_fault = labels[fault_mask]
            logits_fault = logits[fault_mask]
            fault_only = compute_fault_metrics(y_true=labels_fault, logits=logits_fault, multilabel=False)
            metrics["fault_acc_fault_test_windows"] = float(fault_only.get("accuracy", float("nan")))
            metrics["fault_detect_recall_on_fault_test"] = float(np.mean(pred[fault_mask] > 0))
        else:
            metrics["fault_acc_fault_test_windows"] = float("nan")
            metrics["fault_detect_recall_on_fault_test"] = float("nan")
            metrics["warning"] = "No non-normal labels in test windows; fault-only accuracy is undefined."
    else:
        metrics["n_labeled_windows"] = 0
        metrics["n_fault_test_windows"] = 0
        metrics["fault_window_ratio"] = 0.0
        metrics["fault_acc_all_test_windows"] = float("nan")
        metrics["fault_acc_fault_test_windows"] = float("nan")
        metrics["fault_detect_recall_on_fault_test"] = float("nan")
        metrics["warning"] = "No labels were available in test windows."

    out_dir = os.path.join(rc.output_dir, "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fault_window_test_metrics.json")
    save_json(metrics, out_path)

    print("\n🧪 Explicit fault-window test metrics")
    print("-" * 56)
    print(f"test windows: {metrics['n_test_windows']}")
    print(f"labeled windows: {metrics['n_labeled_windows']}")
    print(f"fault-test windows (y>0): {metrics['n_fault_test_windows']}")
    if 'test_label_hist_named' in metrics:
        print(f"test label histogram (named): {metrics['test_label_hist_named']}")
    elif 'test_label_hist' in metrics:
        print(f"test label histogram: {metrics['test_label_hist']}")
    print(f"fault_acc (all test windows): {metrics['fault_acc_all_test_windows']}")
    print(f"fault_acc (fault-test windows): {metrics['fault_acc_fault_test_windows']}")
    print(f"fault detect recall on fault-test windows: {metrics['fault_detect_recall_on_fault_test']}")
    if "warning" in metrics:
        print(f"⚠️  {metrics['warning']}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
