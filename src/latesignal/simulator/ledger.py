"""Append-only prediction and availability ledgers."""

from __future__ import annotations

from typing import Any

from latesignal.contracts.records import PredictionRecord, TrainingRecord
from latesignal.errors import ConsistencyError


def audit_event_trace(events: list[dict[str, object]]) -> None:
    """Reject traces where click delivery or truth precedes persisted prediction."""

    predicted: set[str] = set()
    delivered: set[str] = set()
    for expected_sequence, event in enumerate(events):
        sequence = event.get("sequence")
        kind = event.get("kind")
        click_id = event.get("click_id")
        if sequence != expected_sequence or not isinstance(kind, str):
            raise ConsistencyError("Event trace sequence or kind is malformed")
        if kind == "prediction":
            if not isinstance(click_id, str) or click_id in predicted:
                raise ConsistencyError("Event trace contains an invalid duplicate prediction")
            predicted.add(click_id)
        elif kind == "click_delivered":
            if not isinstance(click_id, str) or click_id not in predicted:
                raise ConsistencyError("Click was delivered before its prediction was persisted")
            delivered.add(click_id)
        elif kind in {"positive_reveal", "negative_maturity"}:
            if (
                not isinstance(click_id, str)
                or click_id not in predicted
                or click_id not in delivered
            ):
                raise ConsistencyError("Truth was revealed before prediction and click delivery")


class PredictionLedger:
    def __init__(self) -> None:
        self.records: list[PredictionRecord] = []
        self._ids: set[str] = set()
        self.sealed = False

    def append(self, record: PredictionRecord) -> None:
        if self.sealed:
            raise ConsistencyError("The prediction ledger is sealed")
        if record.click_id in self._ids:
            raise ConsistencyError(f"Prediction written twice: {record.click_id}")
        self.records.append(record)
        self._ids.add(record.click_id)

    def seal(self) -> None:
        self.sealed = True

    def state_dict(self) -> dict[str, object]:
        return {
            "sealed": self.sealed,
            "records": [record.as_dict() for record in self.records],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        records = state.get("records")
        sealed = state.get("sealed")
        if not isinstance(records, list) or not isinstance(sealed, bool):
            raise ConsistencyError("Prediction-ledger checkpoint state is malformed")
        parsed = [PredictionRecord.from_dict(_object(value)) for value in records]
        ids = {record.click_id for record in parsed}
        if len(ids) != len(parsed):
            raise ConsistencyError("Prediction-ledger checkpoint contains duplicate IDs")
        self.records = parsed
        self._ids = ids
        self.sealed = sealed


class AvailabilityLedger:
    def __init__(self) -> None:
        self.records: list[TrainingRecord] = []
        self._ids: set[str] = set()

    def append(self, record: TrainingRecord, simulator_time: int) -> None:
        record.assert_available(simulator_time)
        if record.record_id in self._ids:
            raise ConsistencyError(f"Training record written twice: {record.record_id}")
        self.records.append(record)
        self._ids.add(record.record_id)

    def legal_at(self, simulator_time: int) -> tuple[TrainingRecord, ...]:
        records = tuple(record for record in self.records if record.available_at <= simulator_time)
        if len(records) != len(self.records):
            raise ConsistencyError("Availability ledger contains a future record")
        return records

    def state_dict(self) -> dict[str, object]:
        return {"records": [record.as_dict() for record in self.records]}

    def load_state_dict(self, state: dict[str, object]) -> None:
        records = state.get("records")
        if not isinstance(records, list):
            raise ConsistencyError("Availability-ledger checkpoint state is malformed")
        parsed = [TrainingRecord.from_dict(_object(value)) for value in records]
        ids = {record.record_id for record in parsed}
        if len(ids) != len(parsed):
            raise ConsistencyError("Availability-ledger checkpoint contains duplicate IDs")
        self.records = parsed
        self._ids = ids


def _object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConsistencyError("Checkpoint ledger record must be an object")
    return value
