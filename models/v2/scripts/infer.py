# scripts/infer.py
from __future__ import annotations

import os
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from scada_tcn.config_schema import build_config
from scada_tcn.data.dataset import ScadaWindowDataset, collate_scada_batches
from scada_tcn.inference.alerting import initial_alert_state, update_alert_state
from scada_tcn.inference.calibration import fit_percentile_thresholds, lookup_threshold
from scada_tcn.inference.step import infer_step
from scada_tcn.modeling import MultiTaskTCN
from scada_tcn.registry import FeatureRegistry, FlagRegistry
from scada_tcn.utils.io import load_json, save_json
from scada_tcn.utils.seed import make_torch_generator


def _load_registries_from_ckpt(ckpt: dict) -> tuple[FeatureRegistry, FlagRegistry]:
    fr = ckpt["feature_registry"]
    gr = ckpt["flag_registry"]
    return FeatureRegistry(**fr), FlagRegistry(**gr)






def _get_nested(obj: Any, path: list[str], default: Any = None) -> Any:
    cur = obj
    for k in path:
        if isinstance(cur, dict):
            if k not in cur:
                return default
            cur = cur[k]
        else:
            if not hasattr(cur, k):
                return default
            cur = getattr(cur, k)
    return cur


def _infer_ckpt_k(ckpt: dict, F: int) -> int | None:
    state = ckpt.get("model_state", {})
    w = state.get("forecast_head.mlp.2.weight")
    if w is None:
        w = state.get("forecast_head.mlp.weight")
    if w is None or getattr(w, "ndim", 0) != 2:
        return None
    out_dim = int(w.shape[0])
    if F <= 0 or out_dim % F != 0:
        return None
    return int(out_dim // F)


def _infer_ckpt_c(ckpt: dict) -> int | None:
    state = ckpt.get("model_state", {})
    w = state.get("fault_head.mlp.2.weight")
    if w is None:
        w = state.get("fault_head.mlp.weight")
    if w is None or getattr(w, "ndim", 0) != 2:
        return None
    return int(w.shape[0])


def _resolve_model_dims_from_ckpt(rc, ckpt: dict, F: int) -> tuple[int, int]:
    k_cfg = int(rc.data.windowing.get("K", 0))
    c_cfg = int(rc.model.heads.get("fault", {}).get("C", 1))

    k_ckpt = _get_nested(ckpt.get("cfg", {}), ["data", "windowing", "K"], default=None)
    if k_ckpt is None:
        k_ckpt = _infer_ckpt_k(ckpt, F)
    K = int(k_ckpt) if k_ckpt is not None else k_cfg

    c_ckpt = _get_nested(ckpt.get("cfg", {}), ["model", "heads", "fault", "C"], default=None)
    if c_ckpt is None:
        c_ckpt = _infer_ckpt_c(ckpt)
    C = int(c_ckpt) if c_ckpt is not None else c_cfg

    if K != k_cfg:
        print(f"ℹ️ Using checkpoint K={K} (config requested K={k_cfg}).")
    if C != c_cfg:
        print(f"ℹ️ Using checkpoint fault classes C={C} (config requested C={c_cfg}).")
    return K, C
def _ascii_bar(v: float, width: int = 20) -> str:
    x = max(0.0, min(1.0, float(v)))
    n = int(round(x * width))
    return "█" * n + "·" * (width - n)


def _print_infer_banner(n_windows: int) -> None:
    print("\n📗 Inference (Beginner View)")
    print("-" * 56)
    print(f"Scoring held-out test windows: {n_windows}")
    print("Outputs include:")
    print("  • Recon head: s_now = 'how unusual this window looks right now'")
    print("  • Pred head: pred_shift_* = 'how different near-future forecast is from now'")
    print("  • Fault head now: p_fault + fault_class_name")
    print("  • Future fault probabilities: p_fault_<class>_7d and _28d")
    print("-" * 56)


def _print_infer_summary(res: pd.DataFrame) -> None:
    if len(res) == 0:
        print("⚠️ No inference rows produced.")
        return
    print("\n✅ Inference summary")
    print(f"Rows: {len(res)} | Turbines: {res['turbine_id'].nunique() if 'turbine_id' in res.columns else 0}")
    if 's_now' in res.columns:
        print(f"s_now (Recon anomaly-now score) mean={res['s_now'].mean():.4f} | p95={res['s_now'].quantile(0.95):.4f}")
    if 'fault_class_name' in res.columns:
        top = res['fault_class_name'].value_counts().head(3)
        print("Top predicted current fault classes:")
        for k, v in top.items():
            print(f"  - {k}: {int(v)}")


def _print_head_visual_examples(res: pd.DataFrame) -> None:
    if len(res) == 0:
        return

    print("\n🧩 Head-by-head examples from REAL inference on held-out test windows")
    print("(These are not placeholders; they are pulled from this run's results.parquet.)")

    # Recon head examples (current anomaly + explanation)
    if "s_now" in res.columns:
        print("\n1) Recon head (what looks unusual *right now*)")
        print("   s_now purpose: larger value => window is more abnormal vs learned normal patterns.")
        top_recon = res.sort_values("s_now", ascending=False).head(3)
        for _, row in top_recon.iterrows():
            score = float(row["s_now"])
            print(
                f"   - turbine={row.get('turbine_id','?')} | s_now={score:.4f} | { _ascii_bar(min(score, 1.0), width=24) }"
            )
            feats = row.get("top_features", [])
            feat_scores = row.get("top_feature_scores", [])
            if isinstance(feats, list) and len(feats) > 0:
                pairs = [f"{f} ({float(s):.3f})" for f, s in zip(feats[:3], feat_scores[:3])]
                print(f"     top contributing sensors: {', '.join(pairs)}")

    # Pred head examples (forecast change intensity)
    if "pred_shift_max" in res.columns:
        print("\n2) Pred head (how much near-future behavior is forecast to shift)")
        print("   pred_shift_max purpose: larger value => model expects sharper short-term change.")
        top_pred = res.sort_values("pred_shift_max", ascending=False).head(3)
        for _, row in top_pred.iterrows():
            vmax = float(row["pred_shift_max"])
            vmean = float(row.get("pred_shift_mean", np.nan))
            print(
                f"   - turbine={row.get('turbine_id','?')} | pred_shift_mean={vmean:.4f} | pred_shift_max={vmax:.4f}"
            )

    # Fault head examples (current class + 7d/28d risk)
    if "p_fault_max" in res.columns:
        print("\n3) Fault head (fault class risk now + forward risk horizons)")
        print("   p_fault_max purpose: confidence of most likely current fault class for this window.")
        top_fault = res.sort_values("p_fault_max", ascending=False).head(3)
        hz_cols = [c for c in res.columns if c.startswith("p_fault_") and (c.endswith("_7d") or c.endswith("_28d"))]
        for _, row in top_fault.iterrows():
            cls = row.get("fault_class_name", f"class_{int(row.get('fault_class_idx', -1))}")
            print(
                f"   - turbine={row.get('turbine_id','?')} | class_now={cls} | p_fault_max={float(row['p_fault_max']):.4f}"
            )
            if hz_cols:
                top_hz = sorted(((c, float(row[c])) for c in hz_cols), key=lambda x: x[1], reverse=True)[:2]
                hz_txt = ", ".join([f"{name}={val:.3f}" for name, val in top_hz])
                print(f"     strongest future risks: {hz_txt}")

def _regime_key(flags_last: np.ndarray, flag_names: list[str]) -> str:
    active = [name for i, name in enumerate(flag_names) if i < len(flags_last) and float(flags_last[i]) > 0.5]
    if not active:
        return "none"
    return "|".join(sorted(active))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    rc = build_config(cfg)

    ckpt_path = os.path.join(rc.output_dir, "checkpoints", "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    feature_reg, flag_reg = _load_registries_from_ckpt(ckpt)

    F = len(feature_reg.feature_names)
    R = len(flag_reg.flag_names)
    K, C = _resolve_model_dims_from_ckpt(rc, ckpt, F)
    label_map_path = os.path.join(rc.output_dir, "registries", "fault_label_map.json")
    class_names: list[str] = []
    horizon_days = [int(x) for x in rc.data.labels.get("horizon_days", [7, 28])]
    horizon_include_normal = bool(rc.data.labels.get("horizon_include_normal", False))
    if os.path.exists(label_map_path):
        lm = load_json(label_map_path)
        class_names = [str(x) for x in lm.get("class_names", [])]
        if class_names and len(class_names) != C:
            print(
                f"⚠️ label_map class count ({len(class_names)}) differs from checkpoint/model C ({C}); "
                "using checkpoint C for model load."
            )
            if len(class_names) > C:
                class_names = class_names[:C]
            else:
                class_names = class_names + [f"class_{i}" for i in range(len(class_names), C)]
    F_in = 3 * F + R

    model = MultiTaskTCN(F_in=F_in, F=F, R=R, K=K, C=C, cfg_model=rc.model)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    device = torch.device(str(rc.device))
    model.to(device)

    proc = rc.data.paths["processed_dir"]
    df_test = pd.read_parquet(os.path.join(proc, "window_index_test.parquet"))

    ds = ScadaWindowDataset(
        processed_dir=proc,
        window_index=df_test,
        feature_reg=feature_reg,
        flag_reg=flag_reg,
        cfg_data=rc.data.__dict__,
        cfg_train=rc.train.__dict__,
        mode="infer",
    )
    loader = DataLoader(
        ds,
        batch_size=int(rc.train.batch_size),
        shuffle=False,
        collate_fn=collate_scada_batches,
        num_workers=0,
    )

    _print_infer_banner(n_windows=len(ds))

    # deterministic mask-at-test scoring
    g = make_torch_generator(int(rc.seed) + 999, device=str(device))

    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="Infer", unit="batch", leave=True):
        # move tensors to device
        batch.X = batch.X.to(device)
        batch.M_miss = batch.M_miss.to(device)
        batch.Flags = batch.Flags.to(device)

        out = infer_step(
            model=model,
            batch=batch,
            cfg=rc,
            feature_reg=feature_reg,
            flag_reg=flag_reg,
            generator=g,
        )

        s_now = out.s_now.detach().cpu().numpy()
        top_idx = out.top_idx.detach().cpu().numpy()
        contrib = out.feat_contrib.detach().cpu().numpy()
        p_fault = out.outputs.p_fault.detach().cpu().numpy() if out.outputs.p_fault is not None else None
        p_fault_h = out.outputs.p_fault_horizons.detach().cpu().numpy() if out.outputs.p_fault_horizons is not None else None
        y_hat = out.outputs.Y_hat.detach().cpu().numpy() if out.outputs.Y_hat is not None else None
        x_last = batch.X[:, -1, :].detach().cpu().numpy()

        flags_last = batch.Flags[:, -1, :].detach().cpu().numpy() if batch.Flags is not None else None

        # start time from timestamps (ns int64)
        start_time = None
        if batch.times is not None:
            # batch.times: (B,L) int64 ns since epoch
            start_ns = batch.times[:, 0].detach().cpu().numpy().astype(np.int64)
            start_time = pd.to_datetime(start_ns, unit="ns", utc=True).tz_convert(None)

        for i in range(len(s_now)):
            tid = batch.turbine_ids[i] if batch.turbine_ids else "unknown"
            top_feature_names = [feature_reg.feature_names[j] for j in top_idx[i].tolist()]
            regime = _regime_key(flags_last[i], flag_reg.flag_names) if flags_last is not None else "none"

            row = {
                "row_order": int(len(rows)),
                "turbine_id": tid,
                "s_now": float(s_now[i]),
                "regime_key": regime,
                "top_features": top_feature_names,
                "top_feature_scores": [float(contrib[i, j]) for j in top_idx[i].tolist()],
            }
            if y_hat is not None:
                # forecast-head derived operational signal (no future truth needed now):
                # mean absolute forecast shift from the last observed feature vector.
                pred_shift = np.abs(y_hat[i] - x_last[i][None, :]).mean(axis=1)
                row["pred_shift_mean"] = float(pred_shift.mean())
                row["pred_shift_max"] = float(pred_shift.max())
            if p_fault is not None:
                row["p_fault"] = [float(x) for x in p_fault[i].tolist()]
                row["p_fault_max"] = float(np.max(p_fault[i]))
                row["fault_class_idx"] = int(np.argmax(p_fault[i]))
                if class_names:
                    row["fault_class_name"] = class_names[row["fault_class_idx"]]
            if p_fault_h is not None:
                row["p_fault_horizons"] = [[float(v) for v in hz] for hz in p_fault_h[i].tolist()]
                if class_names:
                    hz_names = class_names if horizon_include_normal else class_names[1:]
                else:
                    hz_names = [f"class_{j}" for j in range(p_fault_h.shape[-1])]
                for h_idx, day in enumerate(horizon_days[: p_fault_h.shape[1]]):
                    for c_idx, cname in enumerate(hz_names[: p_fault_h.shape[2]]):
                        row[f"p_fault_{cname}_{day}d"] = float(p_fault_h[i, h_idx, c_idx])
            if start_time is not None:
                row["start_time"] = start_time[i]
            rows.append(row)

    out_dir = os.path.join(rc.output_dir, "infer")
    os.makedirs(out_dir, exist_ok=True)

    res = pd.DataFrame(rows)

    # Optional calibration + alerting
    calib_cfg = rc.infer.calibration
    calibration_payload: dict[str, Any] = {
        "enabled": bool(calib_cfg.get("enabled", False)),
        "method": str(calib_cfg.get("method", "percentile")),
    }

    if bool(calib_cfg.get("enabled", False)) and len(res) > 0:
        method = str(calib_cfg.get("method", "percentile"))
        if method != "percentile":
            raise ValueError(f"Unsupported calibration method: {method}")

        percentile = float(calib_cfg.get("percentile", 0.995))
        per_turbine = bool(calib_cfg.get("per_turbine", True))
        per_regime = bool(calib_cfg.get("per_regime", True))

        calib = fit_percentile_thresholds(
            res,
            score_col="s_now",
            turbine_col="turbine_id",
            regime_col="regime_key",
            percentile=percentile,
            per_turbine=per_turbine,
            per_regime=per_regime,
        )

        calibration_payload.update(
            {
                "percentile": percentile,
                "per_turbine": per_turbine,
                "per_regime": per_regime,
                "thresholds": calib.thresholds,
            }
        )

        res["threshold"] = [
            lookup_threshold(
                calib,
                turbine_id=str(tid),
                regime_key=str(reg),
                per_turbine=per_turbine,
                per_regime=per_regime,
            )
            for tid, reg in zip(res["turbine_id"], res["regime_key"])
        ]

        # stateful alerting
        hy = rc.infer.alerting.get("hysteresis", {})
        db = rc.infer.alerting.get("debounce", {})
        T_on_default = float(hy.get("T_on", 3.0))
        T_off_default = float(hy.get("T_off", 2.0))
        N_on = int(db.get("N_on", 3))
        N_off = int(db.get("N_off", 3))

        states: dict[str, Any] = {}
        alert_on: list[bool] = []

        sort_key = "start_time" if "start_time" in res.columns else "row_order"
        for _, row in res.sort_values(["turbine_id", sort_key]).iterrows():
            tid = str(row["turbine_id"])
            states.setdefault(tid, initial_alert_state())
            thr = float(row.get("threshold", T_on_default))
            # keep hysteresis gap if threshold is dynamic
            T_on = thr
            T_off = min(thr, T_off_default)
            st = update_alert_state(
                states[tid],
                score=float(row["s_now"]),
                T_on=T_on,
                T_off=T_off,
                N_on=N_on,
                N_off=N_off,
                timestamp=row.get("start_time", None),
                top_features=None,
            )
            alert_on.append(bool(st.is_on))

        res = res.sort_values(["turbine_id", sort_key]).copy()
        res["alert_on"] = alert_on

    if "row_order" in res.columns:
        res = res.drop(columns=["row_order"])

    out_path = os.path.join(out_dir, "results.parquet")
    res.to_parquet(out_path, index=False)

    save_json(calibration_payload, os.path.join(out_dir, "calibration.json"))
    print("Wrote:", out_path)

    _print_infer_summary(res)
    _print_head_visual_examples(res)

    # Show a few easy-to-read examples for non-ML users
    demo_cols = [c for c in ["turbine_id", "s_now", "pred_shift_mean", "fault_class_name", "p_fault_max"] if c in res.columns]
    hz_cols = [c for c in res.columns if c.startswith("p_fault_") and (c.endswith("_7d") or c.endswith("_28d"))]
    demo_cols += hz_cols[:4]
    if demo_cols:
        print("\nSample rows (first 3):")
        print(res[demo_cols].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
