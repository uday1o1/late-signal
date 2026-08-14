from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from latesignal.cli import app
from latesignal.data.manifests import read_json, write_json_atomic

runner = CliRunner()
CONFIG = Path("configs/experiments/synthetic.yaml").resolve()


def test_public_run_produces_predictions_updates_metrics_and_manifest(tmp_path: Path) -> None:
    output = tmp_path / "full"

    result = runner.invoke(
        app,
        ["run", str(CONFIG), "--out", str(output), "--json"],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "complete"
    assert payload["counts"] == {
        "available_records": 6,
        "checkpoints": 2,
        "credits": 2,
        "optimizer_examples": 20,
        "optimizer_steps": 8,
        "predictions": 6,
    }
    assert payload["metrics"]["count"] == 6
    manifest = read_json(output / "manifest.json")
    assert manifest["prediction_ledger_sealed"] is True
    assert manifest["truth_drained"] is True
    assert len(list((output / "checkpoints").glob("checkpoint-*.json"))) == 2


def test_resume_from_every_checkpoint_reproduces_all_ledgers(tmp_path: Path) -> None:
    full_output = tmp_path / "full"
    full = runner.invoke(
        app,
        ["run", str(CONFIG), "--out", str(full_output), "--json"],
    )
    assert full.exit_code == 0, full.stdout
    expected = read_json(full_output / "manifest.json")
    checkpoints = sorted((full_output / "checkpoints").glob("checkpoint-*.json"))
    assert len(checkpoints) == 2

    for index, checkpoint in enumerate(checkpoints):
        resumed_output = tmp_path / f"resumed-{index}"
        resumed = runner.invoke(
            app,
            ["resume", str(checkpoint), "--out", str(resumed_output), "--json"],
        )
        assert resumed.exit_code == 0, resumed.stdout
        actual = read_json(resumed_output / "manifest.json")
        assert actual["ledger_sha256"] == expected["ledger_sha256"]
        assert actual["counts"] == expected["counts"]
        assert actual["metrics"] == expected["metrics"]
        assert actual["final_model"] == expected["final_model"]


def test_interrupted_public_run_returns_incomplete_and_can_resume(tmp_path: Path) -> None:
    interrupted_output = tmp_path / "interrupted"
    interrupted = runner.invoke(
        app,
        [
            "run",
            str(CONFIG),
            "--out",
            str(interrupted_output),
            "--stop-after-checkpoints",
            "1",
            "--json",
        ],
    )
    assert interrupted.exit_code == 4
    payload = json.loads(interrupted.stdout)
    assert payload["status"] == "interrupted"
    checkpoint = next((interrupted_output / "checkpoints").glob("checkpoint-*.json"))

    resumed_output = tmp_path / "resumed"
    resumed = runner.invoke(
        app,
        ["resume", str(checkpoint), "--out", str(resumed_output), "--json"],
    )

    assert resumed.exit_code == 0, resumed.stdout
    assert read_json(resumed_output / "manifest.json")["status"] == "complete"


def test_resume_refuses_tampered_checkpoint_configuration(tmp_path: Path) -> None:
    full_output = tmp_path / "full"
    assert runner.invoke(app, ["run", str(CONFIG), "--out", str(full_output)]).exit_code == 0
    checkpoint = next((full_output / "checkpoints").glob("checkpoint-*.json"))
    document = read_json(checkpoint)
    document["config"]["seed"] = 999
    tampered = tmp_path / "tampered.json"
    write_json_atomic(tampered, document)

    result = runner.invoke(
        app,
        ["resume", str(tampered), "--out", str(tmp_path / "resume"), "--json"],
    )

    assert result.exit_code == 5
    assert json.loads(result.stdout)["error"] == "INTERNAL_CONSISTENCY_FAILURE"


def test_resume_refuses_changed_runtime_identity(tmp_path: Path) -> None:
    interrupted = tmp_path / "interrupted"
    first = runner.invoke(
        app,
        [
            "run",
            str(CONFIG),
            "--out",
            str(interrupted),
            "--stop-after-checkpoints",
            "1",
            "--json",
        ],
    )
    assert first.exit_code == 4
    checkpoint = next((interrupted / "checkpoints").glob("checkpoint-*.json"))
    payload = read_json(checkpoint)
    runtime = payload["runtime_identity"]
    assert isinstance(runtime, dict)
    runtime["source_tree_sha256"] = "0" * 64
    write_json_atomic(checkpoint, payload, overwrite=True)

    result = runner.invoke(
        app,
        ["resume", str(checkpoint), "--out", str(tmp_path / "refused"), "--json"],
    )

    assert result.exit_code == 5
    assert "runtime identity" in json.loads(result.stdout)["message"]
