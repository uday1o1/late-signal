"""Burn-in-only numeric transformations."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from latesignal.errors import ConsistencyError


@dataclass(frozen=True, slots=True)
class NumericStatistic:
    lower: float
    upper: float
    mean: float
    scale: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> NumericStatistic:
        try:
            statistic = cls(
                lower=float(value["lower"]),
                upper=float(value["upper"]),
                mean=float(value["mean"]),
                scale=float(value["scale"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ConsistencyError("Numeric statistic is malformed") from error
        if statistic.lower > statistic.upper or statistic.scale <= 0.0:
            raise ConsistencyError("Numeric statistic bounds or scale are invalid")
        return statistic


def transform_numeric(raw_value: str, statistic: NumericStatistic) -> tuple[float, bool]:
    value = float(raw_value)
    if value == -1.0:
        return 0.0, True
    if value < 0.0 or not math.isfinite(value):
        raise ConsistencyError("Prepared numeric value violates the inspected sentinel policy")
    transformed = math.log1p(value)
    clipped = min(max(transformed, statistic.lower), statistic.upper)
    return (clipped - statistic.mean) / statistic.scale, False
