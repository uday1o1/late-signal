"""Event-time records that never expose eventual truth on click arrival."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ClickEvent:
    click_id: str
    click_time: int
    feature: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PositiveReveal:
    click_id: str
    available_at: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NegativeMaturity:
    click_id: str
    available_at: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TruthRecord:
    """Private synthetic truth consumed only by the oracle and evaluator."""

    click_id: str
    final_label: int
    available_at: int

    def __post_init__(self) -> None:
        if self.final_label not in {0, 1}:
            raise ValueError("final_label must be binary")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
