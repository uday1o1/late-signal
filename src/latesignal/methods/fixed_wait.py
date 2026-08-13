"""Fixed-wait correction event stream."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from latesignal.contracts.events import ClickEvent, NegativeMaturity, PositiveReveal
from latesignal.contracts.records import TrainingRecord
from latesignal.errors import ConsistencyError


@dataclass(slots=True)
class _FixedState:
    click_time: int
    feature: float
    positive_at: int | None = None
    provisional_emitted: bool = False
    positive_emitted: bool = False
    negative_matured: bool = False


class FixedWaitMethod:
    name = "fixed_wait"

    def __init__(self, wait_seconds: int) -> None:
        if wait_seconds <= 0:
            raise ValueError("wait_seconds must be positive")
        self.wait_seconds = wait_seconds
        self._clicks: dict[str, _FixedState] = {}

    def on_click(self, click: ClickEvent) -> list[TrainingRecord]:
        if click.click_id in self._clicks:
            raise ConsistencyError(f"Click processed twice: {click.click_id}")
        self._clicks[click.click_id] = _FixedState(click.click_time, click.feature)
        return []

    def _state(self, click_id: str) -> _FixedState:
        try:
            return self._clicks[click_id]
        except KeyError as error:
            raise ConsistencyError(f"Truth arrived before click: {click_id}") from error

    def _record(
        self,
        click_id: str,
        available_at: int,
        target: float,
        kind: str,
    ) -> TrainingRecord:
        state = self._state(click_id)
        return TrainingRecord(
            record_id=f"{self.name}:{click_id}:{kind}",
            click_id=click_id,
            available_at=available_at,
            status="provisional" if kind == "provisional" else "final",
            target=target,
            weight=1.0,
            correction_group=click_id if kind in {"provisional", "correction"} else None,
            source_method=self.name,
            feature=state.feature,
        )

    def on_positive_reveal(self, label: PositiveReveal) -> list[TrainingRecord]:
        state = self._state(label.click_id)
        if state.positive_at is not None:
            raise ConsistencyError(f"Positive reveal processed twice: {label.click_id}")
        state.positive_at = label.available_at
        due = state.click_time + self.wait_seconds
        records: list[TrainingRecord] = []
        if label.available_at <= due:
            state.positive_emitted = True
            records.append(self._record(label.click_id, label.available_at, 1.0, "early-positive"))
        else:
            if not state.provisional_emitted:
                state.provisional_emitted = True
                records.append(self._record(label.click_id, due, 0.0, "provisional"))
            state.positive_emitted = True
            records.append(self._record(label.click_id, label.available_at, 1.0, "correction"))
        return records

    def on_negative_maturity(self, label: NegativeMaturity) -> list[TrainingRecord]:
        state = self._state(label.click_id)
        if state.negative_matured or state.positive_at is not None:
            raise ConsistencyError(
                f"Negative maturity processed twice or after positive: {label.click_id}"
            )
        state.negative_matured = True
        return self.on_boundary(label.available_at)

    def on_boundary(self, simulator_time: int) -> list[TrainingRecord]:
        records: list[TrainingRecord] = []
        for click_id, state in sorted(self._clicks.items()):
            due = state.click_time + self.wait_seconds
            if state.provisional_emitted or state.positive_at is not None or due > simulator_time:
                continue
            state.provisional_emitted = True
            records.append(self._record(click_id, due, 0.0, "provisional"))
        return records

    def state_dict(self) -> dict[str, object]:
        return {
            "wait_seconds": self.wait_seconds,
            "clicks": {key: asdict(value) for key, value in sorted(self._clicks.items())},
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("wait_seconds") != self.wait_seconds:
            raise ConsistencyError("Fixed-wait checkpoint configuration does not match")
        clicks = state.get("clicks")
        if not isinstance(clicks, dict):
            raise ConsistencyError("Fixed-wait checkpoint state is malformed")
        parsed: dict[str, _FixedState] = {}
        for key, value in clicks.items():
            if not isinstance(value, dict):
                raise ConsistencyError("Fixed-wait click state is malformed")
            parsed[str(key)] = _FixedState(
                click_time=_state_int(value, "click_time"),
                feature=_state_float(value, "feature"),
                positive_at=_state_optional_int(value.get("positive_at")),
                provisional_emitted=_state_bool(value, "provisional_emitted"),
                positive_emitted=_state_bool(value, "positive_emitted"),
                negative_matured=_state_bool(value, "negative_matured"),
            )
        self._clicks = parsed


def _state_int(value: dict[str, Any], key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ConsistencyError(f"Fixed-wait {key} is malformed")
    return item


def _state_float(value: dict[str, Any], key: str) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise ConsistencyError(f"Fixed-wait {key} is malformed")
    return float(item)


def _state_bool(value: dict[str, Any], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise ConsistencyError(f"Fixed-wait {key} is malformed")
    return item


def _state_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConsistencyError("Fixed-wait positive time is malformed")
    return value
