from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import OmegaConf


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
class RootConfig:
    experiment_name: str
    seed: int
    device: str
    output_dir: str
    data: DataConfig
    features: FeatureConfig
    flags: FlagConfig


def build_config(cfg: Any) -> RootConfig:
    if not isinstance(cfg, dict):
        cfg = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]
    data = DataConfig(**cfg["data"])
    features = FeatureConfig(**cfg["features"])
    flags = FlagConfig(**cfg["flags"])
    rc = RootConfig(
        experiment_name=str(cfg.get("experiment_name", "exp")),
        seed=int(cfg.get("seed", 1337)),
        device=str(cfg.get("device", "cpu")),
        output_dir=str(cfg.get("output_dir", "artifacts")),
        data=data,
        features=features,
        flags=flags,
    )
    validate_config(rc)
    return rc


def validate_config(cfg: RootConfig) -> None:
    L = int(cfg.data.windowing["L"])
    K = int(cfg.data.windowing.get("K", 0))
    stride = int(cfg.data.windowing.get("stride", 1))
    if L <= 0 or stride <= 0 or K < 0:
        raise ValueError(f"Invalid windowing: L={L}, K={K}, stride={stride}")

    dt = int(cfg.data.sampling["dt_minutes"])
    if dt <= 0:
        raise ValueError(f"Invalid dt_minutes: {dt}")

    raw = cfg.features.raw_channels
    ang = set(cfg.features.angular_channels)
    if not set(ang).issubset(set(raw)):
        raise ValueError("angular_channels must be a subset of raw_channels")

    if cfg.features.angle_units not in ("deg", "rad"):
        raise ValueError("features.angle_units must be 'deg' or 'rad'")

    if float(cfg.features.scaler.get("eps", 1e-6)) <= 0:
        raise ValueError("features.scaler.eps must be > 0")

    if not (0.0 <= float(cfg.features.fill_value_c) <= 10.0):
        # in standardized space; wide but catches obvious nonsense
        raise ValueError("features.fill_value_c looks wrong for standardized space")


def resolved_feature_names(cfg: RootConfig) -> list[str]:
    # Deterministic, minimal naming scheme; matches build_features() below.
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

    # Deltas and rolling add names based on already-engineered channel names.
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
