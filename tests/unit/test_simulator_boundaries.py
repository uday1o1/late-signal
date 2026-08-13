from __future__ import annotations

from pathlib import Path

import pytest

from latesignal.contracts.config import load_synthetic_config
from latesignal.contracts.events import NegativeMaturity, PositiveReveal, TruthRecord
from latesignal.contracts.records import PredictionRecord, TrainingRecord
from latesignal.errors import ConsistencyError
from latesignal.experiments.runner import run_synthetic_experiment
from latesignal.experiments.synthetic import build_synthetic_fixture
from latesignal.simulator.ledger import PredictionLedger
from latesignal.simulator.oracle import LabelOracle


def test_oracle_reveals_one_unit_before_exactly_and_after() -> None:
    oracle = LabelOracle(
        (
            TruthRecord("positive", 1, 100),
            TruthRecord("negative", 0, 200),
        )
    )
    clicked = {"positive", "negative"}

    assert oracle.reveal_through(99, clicked) == []
    assert oracle.reveal_through(100, clicked) == [PositiveReveal("positive", 100)]
    assert oracle.reveal_through(101, clicked) == []
    assert oracle.reveal_through(199, clicked) == []
    assert oracle.reveal_through(200, clicked) == [NegativeMaturity("negative", 200)]
    assert oracle.reveal_through(201, clicked) == []
    assert oracle.drained is True


def test_prediction_ledger_is_immutable_after_seal() -> None:
    ledger = PredictionLedger()
    ledger.append(PredictionRecord("one", 0, 0.5, 0))
    ledger.seal()

    with pytest.raises(ConsistencyError, match="sealed"):
        ledger.append(PredictionRecord("two", 1, 0.5, 0))


def test_training_record_rejects_future_availability() -> None:
    record = TrainingRecord(
        record_id="future",
        click_id="click",
        available_at=11,
        status="final",
        target=1.0,
        weight=1.0,
        correction_group=None,
        source_method="test",
        feature=0.0,
    )

    with pytest.raises(ConsistencyError, match="before legal availability"):
        record.assert_available(10)


def test_same_time_prediction_precedes_reveal_and_clock_drains(tmp_path: Path) -> None:
    config = load_synthetic_config(Path("configs/experiments/synthetic.yaml"))
    fixture = build_synthetic_fixture(config)
    manifest = run_synthetic_experiment(config, tmp_path / "run")
    events = manifest["ledger_sha256"]
    assert isinstance(events, dict)

    import json

    event_values = json.loads((tmp_path / "run" / "events.json").read_text(encoding="utf-8"))
    same_time = [
        event
        for event in event_values
        if event.get("click_id") == "click-002" and event["simulator_time"] == 3600
    ]
    assert [event["kind"] for event in same_time] == [
        "prediction",
        "click_delivered",
        "positive_reveal",
    ]
    assert manifest["prediction_ledger_sealed"] is True
    assert manifest["truth_drained"] is True
    assert manifest["clock"]["truth_drain_boundary"] > max(
        click.click_time for click in fixture.clicks
    )
    negative_events = [event for event in event_values if event["kind"] == "negative_maturity"]
    assert len(negative_events) == 3
