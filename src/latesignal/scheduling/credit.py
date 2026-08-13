"""V1 allocation-window and credit-decision contracts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from latesignal.errors import ConsistencyError


@dataclass(frozen=True, slots=True)
class CreditWindow:
    window_id: int
    start_time: int
    end_time: int
    early_time: int
    midpoint_time: int
    deadline_time: int


@dataclass(frozen=True, slots=True)
class SpendDecision:
    window_id: int
    decision_time: int
    spend: bool
    reason: str
    score: float | None = None
    contributing_bin: int | None = None

    def as_dict(self) -> dict[str, int | float | bool | str | None]:
        return asdict(self)


def build_credit_windows(
    *,
    origin: int,
    start_day: int = 31,
    end_day: int = 90,
    window_days: int = 5,
    day_seconds: int = 86_400,
) -> tuple[CreditWindow, ...]:
    """Build the retained-final-partial-window V1 credit allocation."""

    if start_day < 0 or end_day <= start_day or window_days <= 0 or day_seconds <= 0:
        raise ValueError("Credit-window bounds are invalid")
    count = math.ceil((end_day - start_day) / window_days)
    windows: list[CreditWindow] = []
    for window_id in range(count):
        first_day = start_day + window_id * window_days
        last_day_exclusive = min(first_day + window_days, end_day)
        temporal_midpoint = (first_day + last_day_exclusive) / 2.0
        midpoint_day = math.ceil(temporal_midpoint)
        windows.append(
            CreditWindow(
                window_id=window_id,
                start_time=origin + first_day * day_seconds,
                end_time=origin + last_day_exclusive * day_seconds,
                early_time=origin + first_day * day_seconds,
                midpoint_time=origin + midpoint_day * day_seconds,
                deadline_time=origin + (last_day_exclusive - 1) * day_seconds,
            )
        )
    if windows[-1].end_time != origin + end_day * day_seconds:
        raise ConsistencyError("Final credit window does not reach the authored horizon")
    return tuple(windows)
