from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from ..registry import FeatureRegistry
from ..utils.io import save_json


def write_summary_reports(
    output_dir: str,
    cfg: Any,
    metrics: Dict[str, Any],
    examples: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    save_json(metrics, os.path.join(output_dir, "metrics.json"))

    if examples is not None:
        save_json(examples, os.path.join(output_dir, "examples.json"))


def build_rca_showcase(
    turbine_ids: List[str],
    timestamps: List[Any],
    scores: List[float],
    top_idx: np.ndarray,
    feature_reg: FeatureRegistry,
    topn: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(len(turbine_ids)):
        idxs = list(map(int, top_idx[i, :topn].tolist()))
        names = [feature_reg.feature_names[j] for j in idxs]
        out.append(
            {
                "turbine_id": turbine_ids[i],
                "timestamp": timestamps[i],
                "score": float(scores[i]),
                "top_features": names,
            }
        )
    return out
