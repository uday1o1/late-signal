from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from latesignal.cli import app
from latesignal.contracts.config import load_synthetic_config
from latesignal.data.manifests import read_json, write_json_atomic
from latesignal.training.reproducibility import capture_runtime_identity

runner = CliRunner()
CONFIG = Path("configs/experiments/synthetic.yaml")


def _manifest(*, changed_metrics: bool = False) -> dict[str, object]:
    runtime = capture_runtime_identity()
    metrics = {
        "brier_score": 0.24281936191743045,
        "count": 6,
        "log_loss": 0.6784697858888467,
        "positives": 3,
    }
    if changed_metrics:
        metrics["log_loss"] = 0.1
    return {
        "version": 1,
        "kind": "synthetic-run",
        "config": str(CONFIG),
        "config_sha256": load_synthetic_config(CONFIG).canonical_sha256,
        "source_tree_sha256": runtime["source_tree_sha256"],
        "dependency_lock_sha256": runtime["dependency_lock_sha256"],
        "expected": {
            "ledger_sha256": {
                "availability": "e7788eb6e9d832855a87774a922f10b08672b7c958380741b7f8c50b6b5b67b9",
                "credits": "7eabbbaa06fdf75043ad305bb78eb1f632ff51593d1e3a5aeb943b2d8e4a2707",
                "events": "fc9250382b3d5de64a6a708f7a490d533b8762f1e4291f3f8c46ca121f224518",
                "exposures": "f2943727cfef2a6101f230fa54adbf5655403faec6d1e909bafaad48561af908",
                "predictions": "50e35c65078d05cd360c8f63bd854bcd14295e624f3a9240d87cbce717070b6c",
            },
            "counts": {
                "available_records": 6,
                "checkpoints": 2,
                "credits": 2,
                "optimizer_examples": 20,
                "optimizer_steps": 8,
                "predictions": 6,
            },
            "metrics": metrics,
        },
    }


def test_public_reproduce_matches_every_locked_synthetic_output(tmp_path: Path) -> None:
    manifest = tmp_path / "reproduction-manifest.json"
    write_json_atomic(manifest, _manifest())
    output = tmp_path / "reproduced"

    result = runner.invoke(
        app,
        ["reproduce", str(manifest), "--out", str(output), "--json"],
    )

    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["status"] == "complete"
    reproduction = read_json(output / "reproduction.json")
    assert reproduction["observed"] == reproduction["expected"]


def test_public_reproduce_retains_evidence_when_expected_output_is_wrong(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "reproduction-manifest.json"
    write_json_atomic(manifest, _manifest(changed_metrics=True))
    output = tmp_path / "mismatch"

    result = runner.invoke(
        app,
        ["reproduce", str(manifest), "--out", str(output), "--json"],
    )

    assert result.exit_code == 5, result.stdout
    assert read_json(output / "reproduction.json")["status"] == "mismatch"
