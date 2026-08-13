"""Complete-cohort waiting delayed-label strategy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from latesignal.contracts.events import ClickEvent, NegativeMaturity, PositiveReveal
from latesignal.contracts.records import TrainingRecord
from latesignal.errors import ConsistencyError


@dataclass(slots=True)
class _WaitState:
    click_time: int
    feature: float
    final_target: float | None = None
    emitted: bool = False


class CompleteWaitMethod:
    """Emit exactly one eventual-label BCE record at click plus attribution window."""

    name = "complete_wait"

    def __init__(self, attribution_seconds: int) -> None:
        if attribution_seconds <= 0:
            raise ValueError("attribution_seconds must be positive")
        self.attribution_seconds = attribution_seconds
        self._clicks: dict[str, _WaitState] = {}

    def on_click(self, click: ClickEvent) -> list[TrainingRecord]:
        if click.click_id in self._clicks:
            raise ConsistencyError(f"Click processed twice: {click.click_id}")
        self._clicks[click.click_id] = _WaitState(click.click_time, click.feature)
        return []

    def _state_for_truth(self, click_id: str) -> _WaitState:
        try:
            state = self._clicks[click_id]
        except KeyError as error:
            raise ConsistencyError(f"Truth arrived before click: {click_id}") from error
        if state.final_target is not None:
            raise ConsistencyError(f"Truth processed twice: {click_id}")
        return state

    def on_positive_reveal(self, label: PositiveReveal) -> list[TrainingRecord]:
        state = self._state_for_truth(label.click_id)
        if label.available_at > state.click_time + self.attribution_seconds:
            raise ConsistencyError("Positive reveal exceeds the attribution window")
        state.final_target = 1.0
        return []

    def on_negative_maturity(self, label: NegativeMaturity) -> list[TrainingRecord]:
        state = self._state_for_truth(label.click_id)
        if label.available_at != state.click_time + self.attribution_seconds:
            raise ConsistencyError("Negative maturity does not match the attribution window")
        state.final_target = 0.0
        return self.on_boundary(label.available_at)

    def on_boundary(self, simulator_time: int) -> list[TrainingRecord]:
        records: list[TrainingRecord] = []
        for click_id, state in sorted(self._clicks.items()):
            due = state.click_time + self.attribution_seconds
            if state.emitted or state.final_target is None or due > simulator_time:
                continue
            state.emitted = True
            records.append(
                TrainingRecord(
                    record_id=f"{self.name}:{click_id}:final",
                    click_id=click_id,
                    available_at=due,
                    status="final",
                    target=state.final_target,
                    weight=1.0,
                    correction_group=None,
                    source_method=self.name,
                    feature=state.feature,
                )
            )
        return records

    def state_dict(self) -> dict[str, object]:
        return {
            "attribution_seconds": self.attribution_seconds,
            "clicks": {key: asdict(value) for key, value in sorted(self._clicks.items())},
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("attribution_seconds") != self.attribution_seconds:
            raise ConsistencyError("Complete-wait checkpoint configuration does not match")
        clicks = state.get("clicks")
        if not isinstance(clicks, dict):
            raise ConsistencyError("Complete-wait checkpoint state is malformed")
        parsed: dict[str, _WaitState] = {}
        for key, value in clicks.items():
            if not isinstance(value, dict):
                raise ConsistencyError("Complete-wait click state is malformed")
            parsed[str(key)] = _WaitState(
                click_time=_integer(value, "click_time"),
                feature=_number(value, "feature"),
                final_target=_optional_target(value.get("final_target")),
                emitted=_boolean(value, "emitted"),
            )
        self._clicks = parsed


def _integer(value: dict[str, Any], key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ConsistencyError(f"Complete-wait {key} is malformed")
    return item


def _number(value: dict[str, Any], key: str) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise ConsistencyError(f"Complete-wait {key} is malformed")
    return float(item)


def _boolean(value: dict[str, Any], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise ConsistencyError(f"Complete-wait {key} is malformed")
    return item


def _optional_target(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (0, 1):
        raise ConsistencyError("Complete-wait final target is malformed")
    return float(value)
