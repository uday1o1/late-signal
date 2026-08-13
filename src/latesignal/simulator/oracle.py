"""Narrow legal-availability boundary around eventual truth."""

from __future__ import annotations

from typing import Any

from latesignal.contracts.events import NegativeMaturity, PositiveReveal, TruthRecord
from latesignal.errors import ConsistencyError

Reveal = PositiveReveal | NegativeMaturity


class LabelOracle:
    """Reveal each truth once, never before its authored availability time."""

    def __init__(self, truth: tuple[TruthRecord, ...]) -> None:
        self._truth = {record.click_id: record for record in truth}
        if len(self._truth) != len(truth):
            raise ConsistencyError("Truth store contains duplicate click IDs")
        self._delivered: set[str] = set()

    @property
    def drain_time(self) -> int:
        return max(record.available_at for record in self._truth.values())

    @property
    def drained(self) -> bool:
        return len(self._delivered) == len(self._truth)

    def reveal_through(self, simulator_time: int, clicked_ids: set[str]) -> list[Reveal]:
        available = sorted(
            (
                record
                for record in self._truth.values()
                if record.click_id not in self._delivered
                and record.click_id in clicked_ids
                and record.available_at <= simulator_time
            ),
            key=lambda record: (record.available_at, record.click_id),
        )
        reveals: list[Reveal] = []
        for record in available:
            if record.available_at > simulator_time:
                raise ConsistencyError("Oracle attempted an early reveal")
            self._delivered.add(record.click_id)
            if record.final_label == 1:
                reveals.append(PositiveReveal(record.click_id, record.available_at))
            else:
                reveals.append(NegativeMaturity(record.click_id, record.available_at))
        return reveals

    def final_labels(self) -> dict[str, int]:
        if not self.drained:
            raise ConsistencyError("Final truth is unavailable until the oracle is drained")
        return {click_id: record.final_label for click_id, record in self._truth.items()}

    def state_dict(self) -> dict[str, object]:
        return {"delivered": sorted(self._delivered)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        delivered = state.get("delivered")
        if not isinstance(delivered, list):
            raise ConsistencyError("Oracle checkpoint state is malformed")
        parsed = {str(value) for value in delivered}
        if not parsed.issubset(self._truth):
            raise ConsistencyError("Oracle checkpoint references unknown truth IDs")
        self._delivered = parsed

    def truth_state_for_checkpoint(self) -> list[dict[str, Any]]:
        return [self._truth[key].as_dict() for key in sorted(self._truth)]
