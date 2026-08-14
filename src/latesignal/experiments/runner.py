"""Public synthetic run and resume orchestration."""

from __future__ import annotations

import hashlib
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from latesignal import __version__
from latesignal.contracts.config import SyntheticRunConfig, parse_synthetic_config
from latesignal.data.manifests import canonical_json_bytes, write_json_atomic
from latesignal.errors import ConfigurationError, ConsistencyError
from latesignal.evaluation.metrics import binary_metrics
from latesignal.experiments.synthetic import SyntheticFixture, build_synthetic_fixture
from latesignal.simulator.checkpoint import read_checkpoint, write_checkpoint
from latesignal.simulator.engine import EventTimeEngine
from latesignal.simulator.ledger import audit_event_trace


def _ledger_hash(records: object) -> str:
    return hashlib.sha256(canonical_json_bytes(records)).hexdigest()


def _prepare_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ConfigurationError(
            f"Output directory is not empty: {output_dir}",
            details={"output_dir": str(output_dir)},
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _checkpoint_payload(
    config: SyntheticRunConfig,
    fixture: SyntheticFixture,
    engine: EventTimeEngine,
    boundary: int,
) -> dict[str, object]:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "code_version": __version__,
        "config": config.as_dict(),
        "config_sha256": config.canonical_sha256,
        "fixture_sha256": fixture.canonical_sha256,
        "simulator_time": boundary,
        "engine": engine.state_dict(),
    }


def _write_outputs(
    output_dir: Path,
    config: SyntheticRunConfig,
    fixture: SyntheticFixture,
    engine: EventTimeEngine,
    complete: bool,
) -> dict[str, Any]:
    prediction_values = [record.as_dict() for record in engine.predictions.records]
    availability_values = [record.as_dict() for record in engine.availability.records]
    credit_values = [record.as_dict() for record in engine.credit_ledger]
    exposure_values = [record.as_dict() for record in engine.exposure_ledger]
    event_values = engine.event_trace
    audit_event_trace(event_values)
    ledgers: dict[str, object] = {
        "predictions": prediction_values,
        "availability": availability_values,
        "credits": credit_values,
        "exposures": exposure_values,
        "events": event_values,
    }
    for name, values in ledgers.items():
        write_json_atomic(output_dir / f"{name}.json", values, overwrite=True)
    ledger_hashes = {name: _ledger_hash(values) for name, values in ledgers.items()}
    metrics: dict[str, float | int] | None = None
    if complete:
        metrics = binary_metrics(tuple(engine.predictions.records), engine.oracle.final_labels())
        write_json_atomic(output_dir / "metrics.json", metrics, overwrite=True)
    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "status": "complete" if complete else "interrupted",
        "generated_at": datetime.now(UTC).isoformat(),
        "code_version": __version__,
        "config_sha256": config.canonical_sha256,
        "fixture_sha256": fixture.canonical_sha256,
        "seed": config.seed,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "clock": {
            "boundary_seconds": config.boundary_seconds,
            "last_click_time": engine.last_click_time,
            "truth_drain_boundary": engine.final_boundary,
            "next_boundary": engine.next_boundary,
        },
        "counts": {
            "predictions": len(prediction_values),
            "available_records": len(availability_values),
            "credits": len(credit_values),
            "optimizer_steps": sum(record.steps for record in engine.credit_ledger),
            "optimizer_examples": sum(record.examples for record in engine.credit_ledger),
            "checkpoints": engine.checkpoint_count,
        },
        "prediction_ledger_sealed": engine.predictions.sealed,
        "truth_drained": engine.oracle.drained,
        "ledger_sha256": ledger_hashes,
        "final_model": engine.model.state_dict(),
        "metrics": metrics,
    }
    write_json_atomic(output_dir / "manifest.json", manifest, overwrite=True)
    return manifest


def _execute(
    config: SyntheticRunConfig,
    fixture: SyntheticFixture,
    engine: EventTimeEngine,
    output_dir: Path,
    *,
    stop_after_checkpoints: int | None,
) -> dict[str, Any]:
    checkpoint_dir = output_dir / "checkpoints"

    def checkpoint_handler(current: EventTimeEngine, boundary: int) -> None:
        write_checkpoint(
            checkpoint_dir / f"checkpoint-{boundary:010d}.json",
            _checkpoint_payload(config, fixture, current, boundary),
        )

    complete = engine.run(
        checkpoint_handler,
        stop_after_checkpoints=stop_after_checkpoints,
    )
    return _write_outputs(output_dir, config, fixture, engine, complete)


def run_synthetic_experiment(
    config: SyntheticRunConfig,
    output_dir: Path,
    *,
    stop_after_checkpoints: int | None = None,
) -> dict[str, Any]:
    if stop_after_checkpoints is not None and stop_after_checkpoints <= 0:
        raise ConfigurationError("stop_after_checkpoints must be positive")
    _prepare_output(output_dir)
    fixture = build_synthetic_fixture(config)
    engine = EventTimeEngine(config, fixture, output_dir / "live-predictions.json")
    return _execute(
        config,
        fixture,
        engine,
        output_dir,
        stop_after_checkpoints=stop_after_checkpoints,
    )


def resume_synthetic_experiment(checkpoint_path: Path, output_dir: Path) -> dict[str, Any]:
    checkpoint = read_checkpoint(checkpoint_path)
    config_raw = checkpoint.get("config")
    config = parse_synthetic_config(config_raw)
    if config.canonical_sha256 != checkpoint.get("config_sha256"):
        raise ConsistencyError("Checkpoint configuration hash does not match its contents")
    fixture = build_synthetic_fixture(config)
    if fixture.canonical_sha256 != checkpoint.get("fixture_sha256"):
        raise ConsistencyError("Checkpoint fixture hash does not match the generated fixture")
    engine_state = checkpoint.get("engine")
    if not isinstance(engine_state, dict):
        raise ConsistencyError("Checkpoint engine state is malformed")
    _prepare_output(output_dir)
    engine = EventTimeEngine(config, fixture, output_dir / "live-predictions.json")
    engine.load_state_dict(engine_state)
    engine.persist_predictions()
    return _execute(
        config,
        fixture,
        engine,
        output_dir,
        stop_after_checkpoints=None,
    )
