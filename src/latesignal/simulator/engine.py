"""Deterministic hourly event-time engine with atomic prediction persistence."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from latesignal.contracts.config import SyntheticRunConfig
from latesignal.contracts.events import ClickEvent, NegativeMaturity, PositiveReveal
from latesignal.contracts.records import CreditRecord, ExposureRecord, PredictionRecord
from latesignal.data.manifests import write_json_atomic
from latesignal.errors import ConsistencyError
from latesignal.experiments.synthetic import SyntheticFixture
from latesignal.methods.mature_outcome import MatureOutcomeMethod
from latesignal.models.tiny import TinyLogisticModel
from latesignal.scheduling.fixed import FixedDailyScheduler
from latesignal.simulator.ledger import AvailabilityLedger, PredictionLedger
from latesignal.simulator.oracle import LabelOracle
from latesignal.training.trainer import TinyTrainer

CheckpointHandler = Callable[["EventTimeEngine", int], None]


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


def _object(value: Any, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConsistencyError(f"{name} checkpoint state must be an object")
    return value


def _state_int(state: dict[str, object], key: str) -> int:
    value = state.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConsistencyError(f"Engine checkpoint {key} must be an integer")
    return value


def _state_list(state: dict[str, object], key: str) -> list[Any]:
    value = state.get(key)
    if not isinstance(value, list):
        raise ConsistencyError(f"Engine checkpoint {key} must be a list")
    return value


class EventTimeEngine:
    """Execute the mandatory click, reveal, schedule, update, checkpoint order."""

    def __init__(
        self,
        config: SyntheticRunConfig,
        fixture: SyntheticFixture,
        live_prediction_path: Path,
    ) -> None:
        self.config = config
        self.fixture = fixture
        self.live_prediction_path = live_prediction_path
        self.model = TinyLogisticModel()
        self.method = MatureOutcomeMethod()
        self.oracle = LabelOracle(fixture.truth)
        self.predictions = PredictionLedger()
        self.availability = AvailabilityLedger()
        self.origin = min(click.click_time for click in fixture.clicks)
        self.scheduler = FixedDailyScheduler(self.origin, config.decision_interval_seconds)
        self.trainer = TinyTrainer(
            self.model,
            learning_rate=config.training.learning_rate,
            steps_per_credit=config.training.steps_per_credit,
        )
        self.rng = random.Random(config.seed)
        self.click_cursor = 0
        self.clicked_ids: set[str] = set()
        self.next_boundary = self.origin
        self.previous_boundary = self.origin - config.boundary_seconds
        self.credit_ledger: list[CreditRecord] = []
        self.exposure_ledger: list[ExposureRecord] = []
        self.event_trace: list[dict[str, object]] = []
        self.checkpoint_count = 0
        self.last_click_time = max(click.click_time for click in fixture.clicks)
        self.final_boundary = (
            math.ceil(self.oracle.drain_time / config.boundary_seconds) * config.boundary_seconds
        )

    def _event(self, simulator_time: int, kind: str, click_id: str | None = None) -> None:
        event: dict[str, object] = {
            "sequence": len(self.event_trace),
            "simulator_time": simulator_time,
            "kind": kind,
        }
        if click_id is not None:
            event["click_id"] = click_id
        self.event_trace.append(event)

    def persist_predictions(self) -> None:
        write_json_atomic(
            self.live_prediction_path,
            self.predictions.state_dict(),
            overwrite=True,
        )

    def _clicks_through(self, boundary: int) -> list[ClickEvent]:
        batch: list[ClickEvent] = []
        while self.click_cursor < len(self.fixture.clicks):
            click = self.fixture.clicks[self.click_cursor]
            if click.click_time > boundary:
                break
            if click.click_time <= self.previous_boundary:
                raise ConsistencyError("Click cursor moved behind the simulator boundary")
            batch.append(click)
            self.click_cursor += 1
        return batch

    def _process_boundary(self, boundary: int) -> bool:
        clicks = self._clicks_through(boundary)
        for click in clicks:
            self.predictions.append(
                PredictionRecord(
                    click_id=click.click_id,
                    click_time=click.click_time,
                    probability=self.model.predict(click.feature),
                    model_version=self.model.version,
                )
            )
            self._event(boundary, "prediction", click.click_id)
        if self.click_cursor == len(self.fixture.clicks) and not self.predictions.sealed:
            self.predictions.seal()
            self._event(boundary, "prediction_ledger_sealed")
        if clicks or self.predictions.sealed:
            self.persist_predictions()

        for click in clicks:
            self.method.on_click(click)
            self.clicked_ids.add(click.click_id)
            self._event(boundary, "click_delivered", click.click_id)

        reveals = self.oracle.reveal_through(boundary, self.clicked_ids)
        for reveal in reveals:
            if isinstance(reveal, PositiveReveal):
                records = self.method.on_positive_reveal(reveal)
                kind = "positive_reveal"
            elif isinstance(reveal, NegativeMaturity):
                records = self.method.on_negative_maturity(reveal)
                kind = "negative_maturity"
            else:
                raise ConsistencyError("Oracle returned an unknown reveal type")
            self._event(boundary, kind, reveal.click_id)
            for record in records:
                self.availability.append(record, boundary)

        for record in self.method.on_boundary(boundary):
            self.availability.append(record, boundary)

        is_decision = (
            self.scheduler.is_decision_boundary(boundary) and boundary <= self.last_click_time
        )
        if is_decision:
            legal = self.availability.legal_at(boundary)
            if self.scheduler.approve(boundary, len(legal)):
                credit_id = len(self.credit_ledger)
                credit, exposures = self.trainer.spend_credit(credit_id, boundary, legal)
                self.credit_ledger.append(credit)
                self.exposure_ledger.extend(exposures)
                self._event(boundary, "credit_spent")
        return is_decision

    def run(
        self,
        checkpoint_handler: CheckpointHandler,
        *,
        stop_after_checkpoints: int | None = None,
    ) -> bool:
        while self.next_boundary <= self.final_boundary:
            boundary = self.next_boundary
            is_decision = self._process_boundary(boundary)
            self.previous_boundary = boundary
            self.next_boundary = boundary + self.config.boundary_seconds
            if is_decision:
                self.checkpoint_count += 1
                checkpoint_handler(self, boundary)
                if (
                    stop_after_checkpoints is not None
                    and self.checkpoint_count >= stop_after_checkpoints
                ):
                    return False
        if not self.predictions.sealed or not self.oracle.drained:
            raise ConsistencyError("Simulation ended before predictions sealed and truth drained")
        return True

    def state_dict(self) -> dict[str, object]:
        return {
            "click_cursor": self.click_cursor,
            "clicked_ids": sorted(self.clicked_ids),
            "next_boundary": self.next_boundary,
            "previous_boundary": self.previous_boundary,
            "checkpoint_count": self.checkpoint_count,
            "model": self.model.state_dict(),
            "method": self.method.state_dict(),
            "oracle": self.oracle.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "predictions": self.predictions.state_dict(),
            "availability": self.availability.state_dict(),
            "credit_ledger": [record.as_dict() for record in self.credit_ledger],
            "exposure_ledger": [record.as_dict() for record in self.exposure_ledger],
            "event_trace": self.event_trace,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.click_cursor = _state_int(state, "click_cursor")
        clicked_ids = _state_list(state, "clicked_ids")
        self.next_boundary = _state_int(state, "next_boundary")
        self.previous_boundary = _state_int(state, "previous_boundary")
        self.checkpoint_count = _state_int(state, "checkpoint_count")
        credit_values = _state_list(state, "credit_ledger")
        exposure_values = _state_list(state, "exposure_ledger")
        event_trace = _state_list(state, "event_trace")
        rng_state = state.get("rng_state")
        if not 0 <= self.click_cursor <= len(self.fixture.clicks):
            raise ConsistencyError("Engine checkpoint click cursor is invalid")
        self.clicked_ids = {str(value) for value in clicked_ids}
        known_clicks = {click.click_id for click in self.fixture.clicks}
        if not self.clicked_ids.issubset(known_clicks):
            raise ConsistencyError("Engine checkpoint references unknown click IDs")
        self.model.load_state_dict(_object(state["model"], "model"))
        self.method.load_state_dict(_object(state["method"], "method"))
        self.oracle.load_state_dict(_object(state["oracle"], "oracle"))
        self.scheduler.load_state_dict(_object(state["scheduler"], "scheduler"))
        self.predictions.load_state_dict(_object(state["predictions"], "predictions"))
        self.availability.load_state_dict(_object(state["availability"], "availability"))
        self.credit_ledger = [
            CreditRecord.from_dict(_dict_record(value, "credit")) for value in credit_values
        ]
        self.exposure_ledger = [
            ExposureRecord.from_dict(_dict_record(value, "exposure")) for value in exposure_values
        ]
        self.event_trace = [_dict_record(value, "event") for value in event_trace]
        try:
            self.rng.setstate(_tuple_tree(rng_state))
        except (TypeError, ValueError) as error:
            raise ConsistencyError("Engine checkpoint RNG state is malformed") from error


def _dict_record(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConsistencyError(f"{name} checkpoint record must be an object")
    return value
