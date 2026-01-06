from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .utils.io import load_json, save_json


@dataclass(frozen=True)
class FeatureRegistry:
    feature_names: list[str]
    raw_channel_names: list[str]
    angular_channel_names: list[str]
    engineered_version: str = "v1"


@dataclass(frozen=True)
class FlagRegistry:
    flag_names: list[str]
    flags_version: str = "v1"


def save_registry(feature_reg: FeatureRegistry, flag_reg: FlagRegistry, path: str) -> None:
    payload = {
        "schema_version": 1,
        "feature_registry": asdict(feature_reg),
        "flag_registry": asdict(flag_reg),
    }
    save_json(payload, path)


def load_registry(path: str) -> tuple[FeatureRegistry, FlagRegistry]:
    payload = load_json(path)
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported registry schema_version: {payload.get('schema_version')}")
    fr = payload.get("feature_registry", {})
    gr = payload.get("flag_registry", {})
    feature_reg = FeatureRegistry(**fr)
    flag_reg = FlagRegistry(**gr)
    return feature_reg, flag_reg


def assert_registry_compatible(
    feature_reg: FeatureRegistry,
    flag_reg: FlagRegistry,
    feature_names_expected: list[str],
    flag_names_expected: list[str],
) -> None:
    if feature_reg.feature_names != feature_names_expected:
        raise ValueError(
            "Feature registry mismatch.\n"
            f"expected={feature_names_expected}\n"
            f"got={feature_reg.feature_names}"
        )
    if flag_reg.flag_names != flag_names_expected:
        raise ValueError(
            "Flag registry mismatch.\n"
            f"expected={flag_names_expected}\n"
            f"got={flag_reg.flag_names}"
        )


def feature_index_map(feature_reg: FeatureRegistry) -> dict[str, int]:
    return {n: i for i, n in enumerate(feature_reg.feature_names)}


def flag_index_map(flag_reg: FlagRegistry) -> dict[str, int]:
    return {n: i for i, n in enumerate(flag_reg.flag_names)}
