from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from latesignal.cli import app

runner = CliRunner()


def test_public_protocol_validation_reports_exact_external_blockers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "latesignal.experiments.estimate._benchmark",
        lambda *_args, **_kwargs: {
            "requested_device": "cuda",
            "measured_device": "cpu",
            "requested_device_available": False,
            "training_examples_per_second": 10_000_000.0,
            "training_step_seconds": 0.001,
            "es_main_training_step_seconds": 0.002,
            "dfm_training_step_seconds": 0.0012,
            "prediction_examples_per_second": 10_000_000.0,
            "checkpoint_bytes": 1_000_000,
            "model_state_bytes": 400_000,
            "checkpoint_write_seconds": 0.001,
            "checkpoint_state_materialization_seconds": 0.0004,
            "checkpoint_durable_write_seconds": 0.0006,
            "final_snapshot_write_seconds": 0.001,
            "final_snapshot_verify_seconds": 0.0005,
            "prediction_artifact_bytes_per_row": 32.0,
            "exposure_artifact_bytes_per_row": 16.0,
            "peak_host_memory_gb": 1.0,
        },
    )
    monkeypatch.setattr(
        "latesignal.experiments.estimate._real_pilot",
        lambda *_: {"status": "unavailable"},
    )

    result = runner.invoke(
        app,
        [
            "protocol",
            "validate",
            "configs/experiments/final.yaml",
            "--out",
            str(tmp_path / "feasibility.json"),
            "--json",
        ],
    )

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    assert payload["matrix"]["total_runs"] == 89
    assert payload["matrix"]["total_online_credits"] == 1_883
    assert payload["selected_steps_per_credit"] is None
    assert set(payload["blockers"]) == {
        "REQUESTED_ACCELERATOR_UNAVAILABLE",
        "REAL_DATA_PILOT_REQUIRED",
        "REAL_WORKLOAD_INVENTORY_REQUIRED",
        "NO_STEPS_PER_CREDIT_CANDIDATE_FITS_CAPS",
    }
    assert (
        json.loads((tmp_path / "feasibility.json").read_text(encoding="utf-8"))["status"]
        == "blocked"
    )


def test_invalid_protocol_error_is_machine_readable(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("version: 1\nmode: final\nunknown: true\n", encoding="utf-8")

    result = runner.invoke(app, ["protocol", "estimate", str(path), "--json"])

    assert result.exit_code == 2, result.stdout
    payload = json.loads(result.stdout)
    assert payload["error"] == "INVALID_CONFIGURATION"
    assert payload["details"]["errors"]


def test_protocol_estimate_refuses_control_artifacts_in_tracked_repository_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "latesignal.experiments.estimate._benchmark",
        lambda *_args, **_kwargs: pytest.fail("benchmark must not run for an unsafe output path"),
    )
    output = Path.cwd() / "configs" / "unsafe-feasibility-output.json"

    result = runner.invoke(
        app,
        [
            "protocol",
            "estimate",
            "configs/experiments/final.yaml",
            "--out",
            str(output),
            "--json",
        ],
    )

    assert result.exit_code == 2, result.stdout
    payload = json.loads(result.stdout)
    assert payload["error"] == "INVALID_CONFIGURATION"
    assert "ignored runs root" in payload["message"]
    assert not output.exists()
