from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import datetime
from typing import Any

import torch


class _JsonlHandler(logging.Handler):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        payload = record.msg
        if not isinstance(payload, dict):
            payload = {"msg": str(payload)}
        payload = {"ts": datetime.utcnow().isoformat(), "level": record.levelname, **payload}
        line = json.dumps(payload, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def get_logger(name: str, output_dir: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(stream)

    if output_dir is not None:
        path = os.path.join(output_dir, "logs", f"{name}.jsonl")
        logger.addHandler(_JsonlHandler(path))

    return logger


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    return x


def log_kv(logger: logging.Logger, step: int, payload: dict[str, Any]) -> None:
    safe = {k: _to_jsonable(v) for k, v in payload.items()}
    safe["step"] = int(step)
    logger.info(safe)


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        kk = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, kk))
        else:
            out[kk] = v
    return out


def format_exception(e: Exception) -> str:
    return "".join(traceback.format_exception(type(e), e, e.__traceback__))
