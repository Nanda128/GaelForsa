from __future__ import annotations

__version__ = "0.1.0"

# Keep import surface minimal; do not import torch/pandas here.
from .contracts import (  # noqa: F401
    InferBatch,
    InferStepOutputs,
    ModelOutputs,
    TrainBatch,
    TrainStepOutputs,
)
from .registry import FeatureRegistry, FlagRegistry  # noqa: F401
