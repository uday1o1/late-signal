from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from latesignal.cli import app

runner = CliRunner()


def test_public_protocol_validation_reports_exact_external_blockers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "latesignal.experiments.estimate._benchmark",
        lambda _: {
            "requested_device": "cuda",
            "measured_device": "cpu",
            "requested_device_available": False,
            "training_examples_per_second": 10_000_000.0,
            "checkpoint_bytes": 1_000_000,
            "peak_host_memory_gb": 1.0,
        },
    )
    monkeypatch.setattr(
        "latesignal.experiments.estimate._real_pilot",
        lambda *_: {"status": "unavailable"},
    )

    result = runner.invoke(
        app,
        ["protocol", "validate", "configs/experiments/final.yaml", "--json"],
    )

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    assert payload["matrix"]["total_runs"] == 89
    assert payload["matrix"]["total_online_credits"] == 1_883
    assert payload["selected_steps_per_credit"] is None
    assert set(payload["blockers"]) == {
        "USER_RESOURCE_CAPS_REQUIRED",
        "REQUESTED_ACCELERATOR_UNAVAILABLE",
        "REAL_DATA_PILOT_REQUIRED",
        "NO_STEPS_PER_CREDIT_CANDIDATE_FITS_CAPS",
    }


def test_invalid_protocol_error_is_machine_readable(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("version: 1\nmode: final\nunknown: true\n", encoding="utf-8")

    result = runner.invoke(app, ["protocol", "estimate", str(path), "--json"])

    assert result.exit_code == 2, result.stdout
    payload = json.loads(result.stdout)
    assert payload["error"] == "INVALID_CONFIGURATION"
    assert payload["details"]["errors"]
