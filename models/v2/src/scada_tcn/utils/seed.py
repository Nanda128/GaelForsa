from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    base_seed = torch.initial_seed() % (2**32)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def make_torch_generator(seed: int, device: str = "cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g
