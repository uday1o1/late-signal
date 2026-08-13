"""Delayed Feedback Model event state and current-observation materialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from latesignal.contracts.events import ClickEvent, NegativeMaturity, PositiveReveal
from latesignal.contracts.records import TrainingRecord
from latesignal.errors import ConsistencyError

SECONDS_PER_DAY = 86_400


@dataclass(slots=True)
class _DFMState:
    click_time: int
    feature: float
    positive_at: int | None = None
    negative_matured: bool = False


@dataclass(frozen=True, slots=True)
class DFMObservation:
    click_id: str
    feature: float
    target: float
    time_days: float
    status: str


class DelayedFeedbackMethod:
    name = "dfm"

    def __init__(self, attribution_seconds: int) -> None:
        if attribution_seconds <= 0:
            raise ValueError("attribution_seconds must be positive")
        self.attribution_seconds = attribution_seconds
        self._clicks: dict[str, _DFMState] = {}

    def on_click(self, click: ClickEvent) -> list[TrainingRecord]:
        if click.click_id in self._clicks:
            raise ConsistencyError(f"Click processed twice: {click.click_id}")
        self._clicks[click.click_id] = _DFMState(click.click_time, click.feature)
        return []

    def _state(self, click_id: str) -> _DFMState:
        try:
            return self._clicks[click_id]
        except KeyError as error:
            raise ConsistencyError(f"Truth arrived before click: {click_id}") from error

    def on_positive_reveal(self, label: PositiveReveal) -> list[TrainingRecord]:
        state = self._state(label.click_id)
        if state.positive_at is not None or state.negative_matured:
            raise ConsistencyError(f"DFM truth processed twice: {label.click_id}")
        state.positive_at = label.available_at
        return []

    def on_negative_maturity(self, label: NegativeMaturity) -> list[TrainingRecord]:
        state = self._state(label.click_id)
        if state.positive_at is not None or state.negative_matured:
            raise ConsistencyError(f"DFM truth processed twice: {label.click_id}")
        state.negative_matured = True
        return []

    def on_boundary(self, simulator_time: int) -> list[TrainingRecord]:
        del simulator_time
        return []

    def eligible_click_ids(self, simulator_time: int) -> tuple[str, ...]:
        return tuple(
            click_id
            for click_id, state in sorted(self._clicks.items())
            if state.click_time <= simulator_time
        )

    def materialize(self, click_id: str, simulator_time: int) -> DFMObservation:
        state = self._state(click_id)
        if simulator_time < state.click_time:
            raise ConsistencyError("DFM observation precedes its click")
        if state.positive_at is not None and state.positive_at <= simulator_time:
            return DFMObservation(
                click_id=click_id,
                feature=state.feature,
                target=1.0,
                time_days=(state.positive_at - state.click_time) / SECONDS_PER_DAY,
                status="revealed_positive",
            )
        elapsed = min(simulator_time - state.click_time, self.attribution_seconds)
        return DFMObservation(
            click_id=click_id,
            feature=state.feature,
            target=0.0,
            time_days=elapsed / SECONDS_PER_DAY,
            status="mature_negative" if state.negative_matured else "right_censored",
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "attribution_seconds": self.attribution_seconds,
            "clicks": {key: asdict(value) for key, value in sorted(self._clicks.items())},
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("attribution_seconds") != self.attribution_seconds:
            raise ConsistencyError("DFM checkpoint configuration does not match")
        clicks = state.get("clicks")
        if not isinstance(clicks, dict):
            raise ConsistencyError("DFM checkpoint state is malformed")
        parsed: dict[str, _DFMState] = {}
        for key, value in clicks.items():
            if not isinstance(value, dict):
                raise ConsistencyError("DFM click state is malformed")
            click_time = value.get("click_time")
            feature = value.get("feature")
            positive_at = value.get("positive_at")
            negative_matured = value.get("negative_matured")
            if (
                isinstance(click_time, bool)
                or not isinstance(click_time, int)
                or isinstance(feature, bool)
                or not isinstance(feature, (int, float))
                or (
                    positive_at is not None
                    and (isinstance(positive_at, bool) or not isinstance(positive_at, int))
                )
                or not isinstance(negative_matured, bool)
            ):
                raise ConsistencyError("DFM click state is malformed")
            parsed[str(key)] = _DFMState(
                click_time=click_time,
                feature=float(feature),
                positive_at=positive_at,
                negative_matured=negative_matured,
            )
        self._clicks = parsed
