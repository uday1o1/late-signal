"""Explicitly unattainable eventual-label-at-click oracle reference."""

from __future__ import annotations

from latesignal.contracts.events import ClickEvent, NegativeMaturity, PositiveReveal, TruthRecord
from latesignal.contracts.records import TrainingRecord
from latesignal.errors import ConsistencyError


class OracleReferenceMethod:
    name = "oracle_reference"
    deployable = False
    ranking_eligible = False

    def __init__(self, truth: tuple[TruthRecord, ...]) -> None:
        self._truth = {record.click_id: record.final_label for record in truth}
        if len(self._truth) != len(truth):
            raise ConsistencyError("Oracle reference received duplicate truth IDs")
        self._emitted: set[str] = set()

    def on_click(self, click: ClickEvent) -> list[TrainingRecord]:
        if click.click_id in self._emitted:
            raise ConsistencyError(f"Oracle click processed twice: {click.click_id}")
        try:
            target = self._truth[click.click_id]
        except KeyError as error:
            raise ConsistencyError(f"Oracle has no truth for click: {click.click_id}") from error
        self._emitted.add(click.click_id)
        return [
            TrainingRecord(
                record_id=f"{self.name}:{click.click_id}:privileged-final",
                click_id=click.click_id,
                available_at=click.click_time,
                status="final",
                target=float(target),
                weight=1.0,
                correction_group=None,
                source_method=self.name,
                feature=click.feature,
            )
        ]

    def on_positive_reveal(self, label: PositiveReveal) -> list[TrainingRecord]:
        del label
        return []

    def on_negative_maturity(self, label: NegativeMaturity) -> list[TrainingRecord]:
        del label
        return []

    def on_boundary(self, simulator_time: int) -> list[TrainingRecord]:
        del simulator_time
        return []

    def state_dict(self) -> dict[str, object]:
        return {"emitted": sorted(self._emitted)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        emitted = state.get("emitted")
        if not isinstance(emitted, list):
            raise ConsistencyError("Oracle-reference checkpoint state is malformed")
        parsed = {str(value) for value in emitted}
        if not parsed.issubset(self._truth):
            raise ConsistencyError("Oracle-reference checkpoint contains unknown click IDs")
        self._emitted = parsed
