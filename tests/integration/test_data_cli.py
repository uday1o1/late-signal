from __future__ import annotations

import hashlib
import json
from pathlib import Path

from conftest import write_config
from typer.testing import CliRunner

from latesignal.cli import app

runner = CliRunner()


def test_public_cli_fetch_review_and_inspect(tmp_path: Path, valid_archive: Path) -> None:
    config = write_config(tmp_path / "data.yaml", valid_archive)
    data_root = tmp_path / "raw"

    refused = runner.invoke(
        app,
        [
            "data",
            "fetch",
            "--config",
            str(config),
            "--data-root",
            str(data_root),
            "--json",
        ],
    )
    assert refused.exit_code == 3
    assert json.loads(refused.stdout)["error"] == "LICENSE_NOT_ACCEPTED"

    first = runner.invoke(
        app,
        [
            "data",
            "fetch",
            "--accept-license",
            "--config",
            str(config),
            "--data-root",
            str(data_root),
            "--json",
        ],
    )
    assert first.exit_code == 3
    lines = [json.loads(line) for line in first.stdout.splitlines()]
    assert lines[0]["status"] == "license_notice"
    assert lines[1]["error"] == "FIRST_DOWNLOAD_REVIEW_REQUIRED"
    digest = hashlib.sha256(valid_archive.read_bytes()).hexdigest()
    assert lines[1]["details"]["sha256"] == digest

    reviewed = runner.invoke(
        app,
        [
            "data",
            "fetch",
            "--accept-license",
            "--review-sha256",
            digest,
            "--config",
            str(config),
            "--data-root",
            str(data_root),
            "--json",
        ],
    )
    assert reviewed.exit_code == 0, reviewed.stdout
    reviewed_lines = [json.loads(line) for line in reviewed.stdout.splitlines()]
    assert reviewed_lines[-1]["status"] == "verified"

    manifest = tmp_path / "processed" / "inspection.json"
    quarantine = tmp_path / "processed" / "rejected.jsonl"
    inspected = runner.invoke(
        app,
        [
            "data",
            "inspect",
            "--config",
            str(config),
            "--data-root",
            str(data_root),
            "--out",
            str(manifest),
            "--quarantine",
            str(quarantine),
            "--json",
        ],
    )
    assert inspected.exit_code == 0, inspected.stdout
    result = json.loads(inspected.stdout)
    assert result["rows"]["reconciled"] is True
    assert result["time_unit"]["selected_seconds_per_raw_unit"] == 1.0
    assert manifest.exists()


def test_cli_refuses_to_overwrite_immutable_inspection(
    tmp_path: Path, trusted_config: Path
) -> None:
    data_root = tmp_path / "raw"
    fetch = runner.invoke(
        app,
        [
            "data",
            "fetch",
            "--accept-license",
            "--config",
            str(trusted_config),
            "--data-root",
            str(data_root),
        ],
    )
    assert fetch.exit_code == 0
    manifest = tmp_path / "inspection.json"
    quarantine = tmp_path / "quarantine.jsonl"
    command = [
        "data",
        "inspect",
        "--config",
        str(trusted_config),
        "--data-root",
        str(data_root),
        "--out",
        str(manifest),
        "--quarantine",
        str(quarantine),
        "--json",
    ]
    assert runner.invoke(app, command).exit_code == 0

    repeated = runner.invoke(app, command)

    assert repeated.exit_code == 5
    assert json.loads(repeated.stdout)["error"] == "INTERNAL_CONSISTENCY_FAILURE"
