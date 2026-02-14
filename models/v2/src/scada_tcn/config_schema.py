from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

try:  # allow imports in stripped envs; scripts require omegaconf/hydra installed
    from omegaconf import OmegaConf  # type: ignore
except Exception:  # pragma: no cover
    OmegaConf = None  # type: ignore


@dataclass(frozen=True)
class DataConfig:
    paths: dict[str, str]
    sampling: dict[str, Any]
    windowing: dict[str, Any]
    splits: dict[str, Any]
    identifiers: dict[str, str]
    labels: dict[str, Any]
    qc: dict[str, Any]


@dataclass(frozen=True)
class FeatureConfig:
    raw_channels: list[str]
    angular_channels: list[str]
    angle_units: str
    yaw_error: dict[str, Any]
    deltas: dict[str, Any]
    rolling: dict[str, Any]
    fill_value_c: float
    scaler: dict[str, Any]
    output_ordering: dict[str, Any]


@dataclass(frozen=True)
class FlagConfig:
    flag_names: list[str]
    sources: dict[str, Any]
    dtype: str
    causal_alignment: bool


@dataclass(frozen=True)
class ModelConfig:
    tcn: dict[str, Any]
    heads: dict[str, Any]
    receptive_field_assert: dict[str, Any]


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    epochs: int
    log_every: int
    p_mask: float
    loss: dict[str, Any]
    fault_balanced_sampling: dict[str, Any]
    optimizer: dict[str, Any]
    scheduler: Any


@dataclass(frozen=True)
class InferConfig:
    p_score: float
    aggregation: dict[str, Any]
    topn_features: int
    calibration: dict[str, Any]
    alerting: dict[str, Any]


@dataclass(frozen=True)
class RootConfig:
    experiment_name: str
    seed: int
    device: str
    output_dir: str
    data: DataConfig
    features: FeatureConfig
    flags: FlagConfig
    model: ModelConfig
    train: TrainConfig
    infer: InferConfig


def build_config(cfg: Any) -> RootConfig:
    if not isinstance(cfg, dict):
        if OmegaConf is None:
            raise ImportError("omegaconf is required to build config from a Hydra/OmegaConf object")
        cfg = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]

    data = DataConfig(**cfg["data"])
    features = FeatureConfig(**cfg["features"])
    flags = FlagConfig(**cfg["flags"])
    model = ModelConfig(**cfg["model"])
    train = TrainConfig(**cfg["train"])
    infer = InferConfig(**cfg["infer"])

    rc = RootConfig(
        experiment_name=str(cfg.get("experiment_name", "exp")),
        seed=int(cfg.get("seed", 1337)),
        device=str(cfg.get("device", "cuda")),
        output_dir=str(cfg.get("output_dir", "artifacts")),
        data=data,
        features=features,
        flags=flags,
        model=model,
        train=train,
        infer=infer,
    )
    validate_config(rc)
    return rc


def validate_config(cfg: RootConfig) -> None:
    # --- data/windowing ---
    L = int(cfg.data.windowing["L"])
    K = int(cfg.data.windowing.get("K", 0))
    stride = int(cfg.data.windowing.get("stride", 1))
    if L <= 0 or stride <= 0 or K < 0:
        raise ValueError(f"Invalid windowing: L={L}, K={K}, stride={stride}")

    dt = int(cfg.data.sampling["dt_minutes"])
    if dt <= 0:
        raise ValueError(f"Invalid dt_minutes: {dt}")

    # --- features ---
    raw = cfg.features.raw_channels
    ang = set(cfg.features.angular_channels)
    if not ang.issubset(set(raw)):
        raise ValueError("angular_channels must be a subset of raw_channels")

    if cfg.features.angle_units not in ("deg", "rad"):
        raise ValueError("features.angle_units must be 'deg' or 'rad'")

    if float(cfg.features.scaler.get("eps", 1e-6)) <= 0:
        raise ValueError("features.scaler.eps must be > 0")

    c = float(cfg.features.fill_value_c)
    if not (-50.0 <= c <= 50.0):
        raise ValueError("features.fill_value_c looks wrong for standardized space")

    # --- model heads vs tasks ---
    heads = cfg.model.heads
    recon_enabled = bool(heads.get("recon", {}).get("enabled", True))
    forecast_enabled = bool(heads.get("forecast", {}).get("enabled", True))
    fault_enabled = bool(heads.get("fault", {}).get("enabled", False))

    if forecast_enabled and K <= 0:
        raise ValueError("Forecast head enabled but K<=0")

    # labels -> fault head consistency
    labels_enabled = bool(cfg.data.labels.get("enabled", False))
    if labels_enabled and not fault_enabled:
        raise ValueError("labels.enabled=true but model.heads.fault.enabled=false")

    hz_days = list(cfg.data.labels.get("horizon_days", [7, 28]))
    if any(int(d) <= 0 for d in hz_days):
        raise ValueError(f"data.labels.horizon_days must be positive days; got {hz_days}")

    # --- masking probs ---
    p_mask = float(cfg.train.p_mask)
    if recon_enabled and not (0.0 < p_mask < 1.0):
        raise ValueError(f"train.p_mask must be in (0,1) when recon enabled; got {p_mask}")

    p_score = float(cfg.infer.p_score)
    if recon_enabled and not (0.0 < p_score < 1.0):
        raise ValueError(f"infer.p_score must be in (0,1) when recon enabled; got {p_score}")

    # --- receptive field assert ---
    rf_cfg = cfg.model.receptive_field_assert or {}
    if bool(rf_cfg.get("enabled", True)) and bool(rf_cfg.get("must_cover_L", True)):
        from .modeling.tcn import receptive_field, resolve_dilations

        tcn_cfg = dict(cfg.model.tcn)
        dilations = resolve_dilations(tcn_cfg)
        k = int(tcn_cfg.get("kernel_size", 3))
        rf = receptive_field(kernel_size=k, dilations=dilations)
        if rf < L:
            raise ValueError(f"TCN receptive field too small: RF={rf} < L={L}")


def resolved_feature_names(cfg: RootConfig) -> list[str]:
    oc = cfg.features.output_ordering
    if bool(oc.get("explicit", False)):
        names = list(oc.get("feature_names", []))
        if not names:
            raise ValueError("features.output_ordering.explicit=true but feature_names empty")
        return names

    names: list[str] = []
    raw = list(cfg.features.raw_channels)
    ang = set(cfg.features.angular_channels)
    for ch in raw:
        if ch in ang:
            names.append(f"{ch}_sin")
            names.append(f"{ch}_cos")
        else:
            names.append(ch)

    if bool(cfg.features.yaw_error.get("enabled", False)):
        names.append(str(cfg.features.yaw_error.get("output_name", "yaw_error")))

    if bool(cfg.features.deltas.get("enabled", False)):
        for base in cfg.features.deltas.get("channels", []):
            names.append(f"{base}_delta")

    if bool(cfg.features.rolling.get("enabled", False)):
        windows = list(cfg.features.rolling.get("windows", []))
        stats = list(cfg.features.rolling.get("stats", []))
        for base in cfg.features.rolling.get("channels", []):
            for w in windows:
                for st in stats:
                    names.append(f"{base}_roll{w}_{st}")

    return names


def resolved_flag_names(cfg: RootConfig) -> list[str]:
    return list(cfg.flags.flag_names)
