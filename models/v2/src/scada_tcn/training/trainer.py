from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..contracts import TrainBatch, to_device
from ..registry import FeatureRegistry, FlagRegistry
from ..utils.io import save_torch
from ..utils.logging import get_logger, log_kv
from ..utils.seed import make_torch_generator, set_global_seed
from ..evaluation.metrics import compute_fault_metrics
from ..modeling.multitask_tcn import MultiTaskTCN
from .losses import apply_regime_weight, fault_loss, forecast_loss, masked_huber_loss
from .masking import apply_mask, build_model_input, sample_mask
from .step import train_step


def _align_horizon_targets(y_h: torch.Tensor, logits_h: torch.Tensor) -> torch.Tensor:
    if y_h.ndim != 3 or logits_h.ndim != 3:
        return y_h

    b_y, h_y, c_y = y_h.shape
    b_l, h_l, c_l = logits_h.shape
    b = min(b_y, b_l)
    h = min(h_y, h_l)
    y = y_h[:b, :h, :]

    if c_y == c_l:
        return y
    if c_y == c_l + 1:
        return y[..., 1:]
    if c_l == c_y + 1:
        pad = torch.zeros((*y.shape[:2], 1), dtype=y.dtype, device=y.device)
        return torch.cat([pad, y], dim=-1)
    if c_y > c_l:
        return y[..., :c_l]
    pad = torch.zeros((*y.shape[:2], c_l - c_y), dtype=y.dtype, device=y.device)
    return torch.cat([y, pad], dim=-1)



def _emoji_bar(frac: float, width: int = 24) -> str:
    f = max(0.0, min(1.0, float(frac)))
    n = int(round(f * width))
    return "█" * n + "·" * (width - n)


def _print_train_banner(cfg, loaders: dict[str, DataLoader], device: torch.device) -> None:
    print("\n📘 Training (Beginner View)")
    print("-" * 56)
    print(f"Device: {device}")
    print(f"Epochs: {int(cfg.train.epochs)} | Batch size: {int(cfg.train.batch_size)}")
    print(f"Train batches: {len(loaders.get('train', []))} | Val batches: {len(loaders.get('val', []))}")
    print("Heads: recon + forecast + fault (+ fault_horizons if enabled)")
    print("Tip: lower total/val loss over time generally means learning is improving.")
    print("-" * 56)


def _print_epoch_summary(epoch: int, epochs: int, train_loss: float, val_loss: float, best_val: float, epoch_sec: float) -> None:
    rel = 0.0 if best_val <= 0 else min(1.0, max(0.0, val_loss / max(best_val, 1e-9)))
    print(
        f"\n✅ Epoch {epoch}/{epochs} finished | "
        f"train={train_loss:.4f} | val={val_loss:.4f} | best_val={best_val:.4f} | "
        f"time={epoch_sec:.1f}s"
    )
    print(f"Val-vs-best: [{_emoji_bar(1.0 - min(1.0, rel - 1.0 if rel > 1 else 0.0), width=18)}]")



def build_optimizer(model: torch.nn.Module, cfg_train) -> torch.optim.Optimizer:
    name = str(cfg_train.optimizer.get("name", "adamw")).lower()
    lr = float(cfg_train.optimizer.get("lr", 3e-4))
    wd = float(cfg_train.optimizer.get("weight_decay", 1e-4))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def save_checkpoint(
    model: MultiTaskTCN,
    optimizer: torch.optim.Optimizer,
    cfg,
    feature_reg: FeatureRegistry,
    flag_reg: FlagRegistry,
    scaler_ref: str,
    path: str,
    epoch: int,
    step: int,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else cfg,
        "feature_registry": feature_reg.__dict__,
        "flag_registry": flag_reg.__dict__,
        "scaler_ref": scaler_ref,
        "epoch": int(epoch),
        "step": int(step),
    }
    save_torch(payload, path)


def fit(
    model: MultiTaskTCN,
    loaders: dict[str, DataLoader],
    cfg,
    feature_reg: FeatureRegistry,
    flag_reg: FlagRegistry,
    output_dir: str,
    resume_ckpt: Optional[str] = None,
) -> str:
    set_global_seed(int(cfg.seed), deterministic=True)
    logger = get_logger("train", output_dir=output_dir)

    device = torch.device(str(cfg.device))
    model = model.to(device)

    optimizer = build_optimizer(model, cfg.train)

    def _grad_norm_l2() -> float:
        s = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g = float(p.grad.detach().norm(2).item())
                s += g * g
        return s ** 0.5

    def _param_norm_l2() -> float:
        s = 0.0
        for p in model.parameters():
            v = float(p.detach().norm(2).item())
            s += v * v
        return s ** 0.5

    def _current_lr() -> float:
        if not optimizer.param_groups:
            return 0.0
        return float(optimizer.param_groups[0].get("lr", 0.0))

    def _evaluate_loader(loader: DataLoader, epoch_seed: int) -> Dict[str, float]:
        model.eval()
        g_eval = make_torch_generator(epoch_seed, device=str(device))
        losses: dict[str, list[float]] = {"recon": [], "pred": [], "fault": [], "fault_horizons": [], "total": []}
        fault_logits_parts: list[np.ndarray] = []
        fault_label_parts: list[np.ndarray] = []

        lam = cfg.train.loss.get("lambdas", {})
        lam_r = float(lam.get("recon", 1.0))
        lam_p = float(lam.get("pred", 1.0))
        lam_f = float(lam.get("fault", 1.0))
        lam_fh = float(lam.get("fault_horizons", 1.0))
        delta = float(cfg.train.loss.get("huber_delta", 1.0))

        for vb in loader:
            batch = to_device(vb, device)
            X = batch.X
            M_miss = batch.M_miss
            Flags = batch.Flags

            # stream A (reconstruction)
            M_mask_A = sample_mask(M_miss, p_mask=float(cfg.train.p_mask), generator=g_eval)
            X_corrupt_A = apply_mask(X, M_mask_A, float(cfg.features.fill_value_c))
            X_in_A = build_model_input(X_corrupt_A, M_miss, M_mask_A, Flags)
            out_A = model(X_in_A, return_recon=True, return_forecast=False, return_fault=False)
            if out_A.X_hat is None:
                raise ValueError("Validation recon path expected X_hat but got None")

            recon_target_mask = (1.0 - M_mask_A) * M_miss
            L_rec = masked_huber_loss(X, out_A.X_hat, recon_target_mask, delta=delta)

            # stream B (forecast + fault)
            M_mask_B = torch.ones_like(M_miss)
            X_in_B = build_model_input(X, M_miss, M_mask_B, Flags)
            out_B = model(
                X_in_B,
                return_recon=False,
                return_forecast=bool(cfg.model.heads.get("forecast", {}).get("enabled", True)),
                return_fault=bool(cfg.model.heads.get("fault", {}).get("enabled", False)),
            )

            L_pred = torch.tensor(0.0, device=X.device)
            if bool(cfg.model.heads.get("forecast", {}).get("enabled", True)):
                if batch.Y_true is None or out_B.Y_hat is None:
                    raise ValueError("Forecast enabled but Y_true or Y_hat missing in validation")
                L_pred = forecast_loss(batch.Y_true, out_B.Y_hat, delta=delta)

            L_fault = torch.tensor(0.0, device=X.device)
            L_fault_horizons = torch.tensor(0.0, device=X.device)
            fault_enabled = bool(cfg.model.heads.get("fault", {}).get("enabled", False))
            if fault_enabled and out_B.p_fault is not None and batch.y is not None:
                multilabel = bool(cfg.model.heads.get("fault", {}).get("multilabel", False))
                if batch.has_label is not None:
                    mask = batch.has_label > 0.5
                    if mask.any():
                        logits_l = out_B.p_fault[mask]
                        y_l = batch.y[mask]
                        L_fault = fault_loss(logits_l, y_l, multilabel=multilabel)
                        fault_logits_parts.append(logits_l.detach().cpu().numpy())
                        fault_label_parts.append(y_l.detach().cpu().numpy())
                        if out_B.p_fault_horizons is not None and batch.y_horizons is not None:
                            y_h = _align_horizon_targets(batch.y_horizons[mask], out_B.p_fault_horizons[mask])
                            L_fault_horizons = fault_loss(out_B.p_fault_horizons[mask], y_h, multilabel=True)
                else:
                    L_fault = fault_loss(out_B.p_fault, batch.y, multilabel=multilabel)
                    fault_logits_parts.append(out_B.p_fault.detach().cpu().numpy())
                    fault_label_parts.append(batch.y.detach().cpu().numpy())
                    if out_B.p_fault_horizons is not None and batch.y_horizons is not None:
                        y_h = _align_horizon_targets(batch.y_horizons, out_B.p_fault_horizons)
                        L_fault_horizons = fault_loss(out_B.p_fault_horizons, y_h, multilabel=True)

            rw_cfg = cfg.train.loss.get("regime_weighting", {})
            rw_enabled = bool(rw_cfg.get("enabled", False))
            weight_map = dict(rw_cfg.get("weight_map", {}))

            L_rec = apply_regime_weight(L_rec, Flags, enabled=rw_enabled, weight_map=weight_map, flag_reg=flag_reg)
            L_pred = apply_regime_weight(L_pred, Flags, enabled=rw_enabled, weight_map=weight_map, flag_reg=flag_reg)
            L_fault = apply_regime_weight(L_fault, Flags, enabled=rw_enabled, weight_map=weight_map, flag_reg=flag_reg)
            L_fault_horizons = apply_regime_weight(L_fault_horizons, Flags, enabled=rw_enabled, weight_map=weight_map, flag_reg=flag_reg)

            total = lam_r * L_rec + lam_p * L_pred + lam_f * L_fault + lam_fh * L_fault_horizons

            losses["recon"].append(float(L_rec.detach().cpu().item()))
            losses["pred"].append(float(L_pred.detach().cpu().item()))
            losses["fault"].append(float(L_fault.detach().cpu().item()))
            losses["fault_horizons"].append(float(L_fault_horizons.detach().cpu().item()))
            losses["total"].append(float(total.detach().cpu().item()))

        model.train()
        out_metrics = {k: float(sum(v) / max(len(v), 1)) for k, v in losses.items()}

        if fault_logits_parts and fault_label_parts:
            fault_logits = np.concatenate(fault_logits_parts, axis=0)
            fault_labels = np.concatenate(fault_label_parts, axis=0)
            if fault_labels.ndim > 1 and fault_labels.shape[-1] == 1:
                fault_labels = fault_labels.reshape(-1)
            fault_metrics = compute_fault_metrics(
                y_true=fault_labels,
                logits=fault_logits,
                multilabel=bool(cfg.model.heads.get("fault", {}).get("multilabel", False)),
            )
            out_metrics.update({f"fault_{k}": float(v) for k, v in fault_metrics.items()})

        return out_metrics

    start_epoch = 1
    global_step = 0

    if resume_ckpt:
        ck = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"])
        if "optim_state" in ck:
            optimizer.load_state_dict(ck["optim_state"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        global_step = int(ck.get("step", 0))

    best_val = float("inf")
    best_path = os.path.join(output_dir, "checkpoints", "best.pt")
    last_path = os.path.join(output_dir, "checkpoints", "last.pt")
    os.makedirs(os.path.dirname(best_path), exist_ok=True)

    g = make_torch_generator(int(cfg.seed) + 123, device=str(device))

    _print_train_banner(cfg, loaders, device)

    log_kv(
        logger,
        step=0,
        payload={
            "event": "train_start",
            "device": str(device),
            "seed": int(cfg.seed),
            "train_batches": len(loaders.get("train", [])),
            "val_batches": len(loaders.get("val", [])),
            "batch_size": int(cfg.train.batch_size),
            "epochs": int(cfg.train.epochs),
            "optimizer": str(cfg.train.optimizer.get("name", "adamw")),
            "lr": _current_lr(),
            "weight_decay": float(cfg.train.optimizer.get("weight_decay", 0.0)),
            "p_mask": float(cfg.train.p_mask),
            "lambdas": dict(cfg.train.loss.get("lambdas", {})),
        },
    )

    for epoch in range(start_epoch, int(cfg.train.epochs) + 1):
        model.train()
        running = []
        epoch_t0 = time.time()

        pbar = tqdm(loaders["train"], desc=f"Epoch {epoch}/{int(cfg.train.epochs)}", unit="batch", leave=True)
        for batch in pbar:
            batch = to_device(batch, device)
            step_t0 = time.time()
            out = train_step(
                model=model,
                batch=batch,
                cfg=cfg,
                feature_reg=feature_reg,
                flag_reg=flag_reg,
                optimizer=optimizer,
                generator=g,
            )
            global_step += 1
            running.append(out.losses["total"])

            step_time = time.time() - step_t0
            windows_per_sec = float(batch.X.shape[0] / max(step_time, 1e-6))

            pbar.set_postfix({"total": f"{out.losses['total']:.4f}", "recon": f"{out.losses['recon']:.4f}", "pred": f"{out.losses['pred']:.4f}"})

            if global_step % int(cfg.train.log_every) == 0:
                log_kv(
                    logger,
                    step=global_step,
                    payload={
                        "epoch": epoch,
                        **out.losses,
                        **out.metrics,
                        **out.debug,
                        "lr": _current_lr(),
                        "grad_norm_l2": _grad_norm_l2(),
                        "param_norm_l2": _param_norm_l2(),
                        "step_time_sec": step_time,
                        "windows_per_sec": windows_per_sec,
                        "batch_B": int(batch.X.shape[0]),
                        "batch_L": int(batch.X.shape[1]),
                        "batch_F": int(batch.X.shape[2]),
                        "flags_R": int(batch.Flags.shape[2]),
                    },
                )

        train_loss = float(sum(running) / max(len(running), 1))

        val_loss = train_loss
        val_parts: Dict[str, float] = {}
        if "val" in loaders and len(loaders["val"]) > 0:
            with torch.no_grad():
                val_parts = _evaluate_loader(loaders["val"], epoch_seed=int(cfg.seed) + 10_000 + epoch)
                val_loss = float(val_parts.get("total", train_loss))

        scaler_ref = os.path.join(cfg.output_dir, "scalers", "robust_scaler.json")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            feature_reg=feature_reg,
            flag_reg=flag_reg,
            scaler_ref=scaler_ref,
            path=last_path,
            epoch=epoch,
            step=global_step,
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                feature_reg=feature_reg,
                flag_reg=flag_reg,
                scaler_ref=scaler_ref,
                path=best_path,
                epoch=epoch,
                step=global_step,
            )

        _print_epoch_summary(epoch, int(cfg.train.epochs), train_loss, val_loss, best_val, float(time.time() - epoch_t0))

        log_kv(
            logger,
            step=global_step,
            payload={
                "event": "epoch_end",
                "epoch": epoch,
                "train_epoch_loss": train_loss,
                "val_epoch_loss": val_loss,
                "best_val": best_val,
                "epoch_time_sec": float(time.time() - epoch_t0),
                **{f"val_{k}": v for k, v in val_parts.items()},
            },
        )

    return best_path
