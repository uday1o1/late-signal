from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from latesignal.cli import app

runner = CliRunner()


def test_public_selection_command_routes_the_complete_production_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "final.yaml"
    data_manifest = tmp_path / "prepared.json"
    feature_config = tmp_path / "features.yaml"
    for path in (config, data_manifest, feature_config):
        path.write_text("{}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(config_path: Path, **kwargs: object) -> dict[str, object]:
        captured["config_path"] = config_path
        captured.update(kwargs)
        return {
            "status": "complete",
            "candidate_counts": {"model": 36, "delayed": 8, "sampler": 6, "total": 50},
            "selection_results_sha256": "a" * 64,
        }

    monkeypatch.setattr("latesignal.cli.run_production_selection", fake_run)
    output = tmp_path / "selection"
    result = runner.invoke(
        app,
        [
            "selection",
            "run",
            str(config),
            "--data-manifest",
            str(data_manifest),
            "--feature-config",
            str(feature_config),
            "--cache-root",
            str(tmp_path / "cache"),
            "--out",
            str(output),
            "--steps-per-credit",
            "100",
            "--device-uuid",
            "GPU-test",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "complete"
    assert payload["candidate_counts"]["total"] == 50
    assert captured["output_root"] == output.resolve()
    assert captured["steps_per_credit"] == 100
    assert captured["device_uuid"] == "GPU-test"


def test_public_final_command_routes_the_locked_production_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "final.yaml"
    protocol_lock = tmp_path / "protocol-lock.json"
    data_manifest = tmp_path / "prepared.json"
    feature_config = tmp_path / "features.yaml"
    for path in (config, protocol_lock, data_manifest, feature_config):
        path.write_text("{}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(config_path: Path, **kwargs: object) -> dict[str, object]:
        captured["config_path"] = config_path
        captured.update(kwargs)
        return {
            "status": "complete",
            "completed_count": 39,
            "online_runs": 33,
            "offline_runs": 6,
            "manifest_sha256": "a" * 64,
        }

    monkeypatch.setattr("latesignal.cli.run_production_final", fake_run)
    output = tmp_path / "final"
    result = runner.invoke(
        app,
        [
            "final",
            "run",
            str(config),
            "--protocol-lock",
            str(protocol_lock),
            "--data-manifest",
            str(data_manifest),
            "--feature-config",
            str(feature_config),
            "--cache-root",
            str(tmp_path / "cache"),
            "--out",
            str(output),
            "--device-uuid",
            "GPU-test",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["completed_count"] == 39
    assert payload["online_runs"] == 33
    assert payload["offline_runs"] == 6
    assert captured["protocol_lock_path"] == protocol_lock.resolve()
    assert captured["output_root"] == output.resolve()
    assert captured["device_uuid"] == "GPU-test"


def test_public_final_qualify_routes_the_locked_gate_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "final.yaml"
    protocol_lock = tmp_path / "protocol-lock.json"
    data_manifest = tmp_path / "prepared.json"
    feature_config = tmp_path / "features.yaml"
    for path in (config, protocol_lock, data_manifest, feature_config):
        path.write_text("{}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_qualify(config_path: Path, **kwargs: object) -> dict[str, object]:
        captured["config_path"] = config_path
        captured.update(kwargs)
        return {
            "status": "passed",
            "manifest_sha256": "a" * 64,
        }

    monkeypatch.setattr("latesignal.cli.run_final_qualification", fake_qualify)
    output = tmp_path / "quality-gate.json"
    result = runner.invoke(
        app,
        [
            "final",
            "qualify",
            str(config),
            "--protocol-lock",
            str(protocol_lock),
            "--data-manifest",
            str(data_manifest),
            "--feature-config",
            str(feature_config),
            "--cache-root",
            str(tmp_path / "cache"),
            "--out",
            str(output),
            "--device-uuid",
            "GPU-test",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "passed"
    assert captured["protocol_lock_path"] == protocol_lock.resolve()
    assert captured["output_path"] == output.resolve()
    assert captured["device_uuid"] == "GPU-test"


def test_public_final_aggregate_routes_quality_and_evidence_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "final.yaml"
    protocol_lock = tmp_path / "protocol-lock.json"
    data_manifest = tmp_path / "prepared.json"
    feature_config = tmp_path / "features.yaml"
    quality_gate = tmp_path / "quality-gate.json"
    output = tmp_path / "final"
    output.mkdir()
    for path in (config, protocol_lock, data_manifest, feature_config, quality_gate):
        path.write_text("{}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_aggregate(config_path: Path, **kwargs: object) -> dict[str, object]:
        captured["config_path"] = config_path
        captured.update(kwargs)
        return {
            "status": "complete",
            "scheduler_outcome": "negative_or_inconclusive",
            "manifest_sha256": "a" * 64,
        }

    monkeypatch.setattr("latesignal.cli.run_production_aggregate", fake_aggregate)
    result = runner.invoke(
        app,
        [
            "final",
            "aggregate",
            str(config),
            "--protocol-lock",
            str(protocol_lock),
            "--data-manifest",
            str(data_manifest),
            "--feature-config",
            str(feature_config),
            "--cache-root",
            str(tmp_path / "cache"),
            "--out",
            str(output),
            "--quality-gate",
            str(quality_gate),
            "--device-uuid",
            "GPU-test",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["scheduler_outcome"] == "negative_or_inconclusive"
    assert captured["quality_gate_path"] == quality_gate.resolve()
    assert captured["output_root"] == output.resolve()
