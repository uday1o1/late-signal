"""Immediate fake-negative BCE event stream."""

from __future__ import annotations

from latesignal.contracts.events import ClickEvent, NegativeMaturity, PositiveReveal
from latesignal.contracts.records import TrainingRecord
from latesignal.errors import ConsistencyError


class ImmediateFakeNegativeMethod:
    name = "immediate_fake_negative"

    def __init__(self) -> None:
        self._features: dict[str, float] = {}
        self._positive_corrections: set[str] = set()
        self._mature_negatives: set[str] = set()

    def on_click(self, click: ClickEvent) -> list[TrainingRecord]:
        if click.click_id in self._features:
            raise ConsistencyError(f"Click processed twice: {click.click_id}")
        self._features[click.click_id] = click.feature
        return [self._record(click.click_id, click.click_time, 0.0, "provisional")]

    def _feature(self, click_id: str) -> float:
        try:
            return self._features[click_id]
        except KeyError as error:
            raise ConsistencyError(f"Truth arrived before click: {click_id}") from error

    def _record(
        self,
        click_id: str,
        available_at: int,
        target: float,
        status: str,
    ) -> TrainingRecord:
        return TrainingRecord(
            record_id=f"{self.name}:{click_id}:{status}",
            click_id=click_id,
            available_at=available_at,
            status="provisional" if status == "provisional" else "final",
            target=target,
            weight=1.0,
            correction_group=click_id,
            source_method=self.name,
            feature=self._feature(click_id),
        )

    def on_positive_reveal(self, label: PositiveReveal) -> list[TrainingRecord]:
        self._feature(label.click_id)
        if label.click_id in self._positive_corrections:
            raise ConsistencyError(f"Positive reveal processed twice: {label.click_id}")
        self._positive_corrections.add(label.click_id)
        return [self._record(label.click_id, label.available_at, 1.0, "correction")]

    def on_negative_maturity(self, label: NegativeMaturity) -> list[TrainingRecord]:
        self._feature(label.click_id)
        if label.click_id in self._mature_negatives:
            raise ConsistencyError(f"Negative maturity processed twice: {label.click_id}")
        self._mature_negatives.add(label.click_id)
        return []

    def on_boundary(self, simulator_time: int) -> list[TrainingRecord]:
        del simulator_time
        return []

    def state_dict(self) -> dict[str, object]:
        return {
            "features": dict(sorted(self._features.items())),
            "positive_corrections": sorted(self._positive_corrections),
            "mature_negatives": sorted(self._mature_negatives),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        features = state.get("features")
        positives = state.get("positive_corrections")
        negatives = state.get("mature_negatives")
        if (
            not isinstance(features, dict)
            or not isinstance(positives, list)
            or not isinstance(negatives, list)
        ):
            raise ConsistencyError("Immediate-fake-negative checkpoint state is malformed")
        self._features = {str(key): float(value) for key, value in features.items()}
        self._positive_corrections = {str(value) for value in positives}
        self._mature_negatives = {str(value) for value in negatives}
