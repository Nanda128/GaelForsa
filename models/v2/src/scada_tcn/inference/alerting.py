from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AlertState:
    is_on: bool
    on_count: int
    off_count: int
    last_score: float
    last_timestamp: Optional[Any] = None
    last_top_features: Optional[list[int]] = None


def initial_alert_state() -> AlertState:
    return AlertState(is_on=False, on_count=0, off_count=0, last_score=0.0)


def update_alert_state(
    state: AlertState,
    score: float,
    T_on: float,
    T_off: float,
    N_on: int,
    N_off: int,
    timestamp: Any | None = None,
    top_features: list[int] | None = None,
) -> AlertState:
    score = float(score)

    if not state.is_on:
        if score >= T_on:
            state.on_count += 1
        else:
            state.on_count = 0

        if state.on_count >= int(N_on):
            state.is_on = True
            state.off_count = 0
    else:
        if score <= T_off:
            state.off_count += 1
        else:
            state.off_count = 0

        if state.off_count >= int(N_off):
            state.is_on = False
            state.on_count = 0

    state.last_score = score
    state.last_timestamp = timestamp
    state.last_top_features = top_features
    return state
