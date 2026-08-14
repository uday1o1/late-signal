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
