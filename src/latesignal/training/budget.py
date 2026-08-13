"""Exact core optimizer budget accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from latesignal.errors import ConsistencyError


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    credits: int
    optimizer_steps: int
    optimizer_examples: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class BudgetCounter:
    def __init__(self) -> None:
        self.credits = 0
        self.optimizer_steps = 0
        self.optimizer_examples = 0

    def record_credit(self, *, steps: int, batch_size: int) -> None:
        if steps <= 0 or batch_size <= 0:
            raise ValueError("steps and batch_size must be positive")
        self.credits += 1
        self.optimizer_steps += steps
        self.optimizer_examples += steps * batch_size

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            credits=self.credits,
            optimizer_steps=self.optimizer_steps,
            optimizer_examples=self.optimizer_examples,
        )

    def assert_exposures(self, exposure_count: int) -> None:
        if exposure_count != self.optimizer_examples:
            raise ConsistencyError(
                "Exposure ledger does not reconcile with the optimizer-example budget",
                details={
                    "exposures": exposure_count,
                    "optimizer_examples": self.optimizer_examples,
                },
            )

    def state_dict(self) -> dict[str, int]:
        return self.snapshot().as_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        def checked(key: str) -> int:
            value = state.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConsistencyError("Budget checkpoint state is malformed")
            return value

        self.credits = checked("credits")
        self.optimizer_steps = checked("optimizer_steps")
        self.optimizer_examples = checked("optimizer_examples")
