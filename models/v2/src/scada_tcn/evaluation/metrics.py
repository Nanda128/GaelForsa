from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_forecast_metrics(
    Y_true: np.ndarray | torch.Tensor,
    Y_hat: np.ndarray | torch.Tensor,
    mask: Optional[np.ndarray | torch.Tensor] = None,
) -> Dict[str, float]:
    yt = _to_numpy(Y_true)
    yh = _to_numpy(Y_hat)

    err = yt - yh
    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        err = err[m]

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return {"mae": mae, "rmse": rmse}


def compute_fault_metrics(
    y_true: np.ndarray | torch.Tensor,
    logits: np.ndarray | torch.Tensor,
    multilabel: bool,
    class_names: Optional[list[str]] = None,
) -> Dict[str, float]:
    """
    Basic classification metrics.
    Uses scikit-learn if available; otherwise returns a minimal subset.
    """
    yt = _to_numpy(y_true)
    lg = _to_numpy(logits)

    out: Dict[str, float] = {}

    try:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        if not multilabel:
            pred = np.argmax(lg, axis=1)
            out["accuracy"] = float(accuracy_score(yt, pred))
            out["f1_macro"] = float(f1_score(yt, pred, average="macro"))
            out["n_samples"] = float(len(yt))

            labels_present = sorted([int(x) for x in np.unique(yt)])
            for c in labels_present:
                mask = yt == c
                if np.any(mask):
                    out[f"acc_class_{c}"] = float(np.mean(pred[mask] == yt[mask]))
                    out[f"support_class_{c}"] = float(np.sum(mask))

            # AUROC one-vs-rest if possible
            try:
                # softmax
                ex = np.exp(lg - lg.max(axis=1, keepdims=True))
                proba = ex / np.clip(ex.sum(axis=1, keepdims=True), 1e-12, None)
                out["auroc_ovr"] = float(roc_auc_score(yt, proba, multi_class="ovr"))
            except Exception:
                pass
        else:
            # multilabel expects yt: (N,C) in {0,1}
            proba = 1.0 / (1.0 + np.exp(-lg))
            pred = (proba >= 0.5).astype(int)
            out["f1_micro"] = float(f1_score(yt, pred, average="micro"))
            out["f1_macro"] = float(f1_score(yt, pred, average="macro"))
            try:
                out["auroc_macro"] = float(roc_auc_score(yt, proba, average="macro"))
            except Exception:
                pass

    except Exception:
        # minimal fallback
        if not multilabel:
            pred = np.argmax(lg, axis=1)
            out["accuracy"] = float(np.mean(pred == yt))
        else:
            proba = 1.0 / (1.0 + np.exp(-lg))
            pred = (proba >= 0.5).astype(int)
            out["exact_match"] = float(np.mean(np.all(pred == yt, axis=1)))

    return out


def compute_anomaly_metrics(
    scores: np.ndarray,
    y_anom: np.ndarray,
    thresholds: Optional[list[float]] = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    s = np.asarray(scores).astype(float)
    y = np.asarray(y_anom).astype(int)

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

        out["pr_auc"] = float(average_precision_score(y, s))
        out["roc_auc"] = float(roc_auc_score(y, s))

        if thresholds is None:
            thresholds = np.quantile(s, np.linspace(0.5, 0.995, 50)).tolist()

        best = 0.0
        best_t = thresholds[0] if thresholds else float(np.median(s))
        for t in thresholds or []:
            yp = (s >= t).astype(int)
            f1 = float(f1_score(y, yp))
            if f1 > best:
                best = f1
                best_t = float(t)
        out["best_f1"] = best
        out["best_f1_threshold"] = best_t

    except Exception:
        # minimal fallback
        out["mean_score_anom"] = float(np.mean(s[y == 1])) if np.any(y == 1) else float("nan")
        out["mean_score_norm"] = float(np.mean(s[y == 0])) if np.any(y == 0) else float("nan")

    return out
