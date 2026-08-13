"""Delayed-method protocol shared by event-time strategies."""

from __future__ import annotations

from typing import Protocol

from latesignal.contracts.events import ClickEvent, NegativeMaturity, PositiveReveal
from latesignal.contracts.records import TrainingRecord


class DelayedMethod(Protocol):
    def on_click(self, click: ClickEvent) -> list[TrainingRecord]: ...

    def on_positive_reveal(self, label: PositiveReveal) -> list[TrainingRecord]: ...

    def on_negative_maturity(self, label: NegativeMaturity) -> list[TrainingRecord]: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: dict[str, object]) -> None: ...
