"""Immediate legal-outcome method retained for the small simulator smoke path."""

from __future__ import annotations

from latesignal.contracts.events import ClickEvent, NegativeMaturity, PositiveReveal
from latesignal.contracts.records import TrainingRecord
from latesignal.errors import ConsistencyError


class MatureOutcomeMethod:
    """Emit final records when truth first becomes legally observable."""

    name = "mature_outcome"

    def __init__(self) -> None:
        self._features: dict[str, float] = {}
        self._emitted: set[str] = set()

    def on_click(self, click: ClickEvent) -> list[TrainingRecord]:
        if click.click_id in self._features:
            raise ConsistencyError(f"Click processed twice: {click.click_id}")
        self._features[click.click_id] = click.feature
        return []

    def _final_record(self, click_id: str, available_at: int, target: float) -> TrainingRecord:
        if click_id not in self._features:
            raise ConsistencyError(f"Truth arrived before click: {click_id}")
        if click_id in self._emitted:
            raise ConsistencyError(f"Truth processed twice: {click_id}")
        self._emitted.add(click_id)
        return TrainingRecord(
            record_id=f"{self.name}:{click_id}:final",
            click_id=click_id,
            available_at=available_at,
            status="final",
            target=target,
            weight=1.0,
            correction_group=None,
            source_method=self.name,
            feature=self._features[click_id],
        )

    def on_positive_reveal(self, label: PositiveReveal) -> list[TrainingRecord]:
        return [self._final_record(label.click_id, label.available_at, 1.0)]

    def on_negative_maturity(self, label: NegativeMaturity) -> list[TrainingRecord]:
        return [self._final_record(label.click_id, label.available_at, 0.0)]

    def on_boundary(self, simulator_time: int) -> list[TrainingRecord]:
        del simulator_time
        return []

    def state_dict(self) -> dict[str, object]:
        return {"features": dict(sorted(self._features.items())), "emitted": sorted(self._emitted)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        features = state.get("features")
        emitted = state.get("emitted")
        if not isinstance(features, dict) or not isinstance(emitted, list):
            raise ConsistencyError("Mature-outcome checkpoint state is malformed")
        self._features = {str(key): float(value) for key, value in features.items()}
        self._emitted = {str(value) for value in emitted}
