from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from latesignal.cli import app
from latesignal.data.manifests import read_json

runner = CliRunner()
CONFIG = Path("configs/experiments/study_b.synthetic.yaml").resolve()


def test_public_study_b_matches_compute_and_triggers_early(tmp_path: Path) -> None:
    output = tmp_path / "study-b"

    result = runner.invoke(app, ["run", str(CONFIG), "--out", str(output), "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "complete"
    assert payload["counts"] == {"policies": 4}
    manifest = read_json(output / "manifest.json")
    policies = manifest["policies"]
    assert {policy["scheduler"] for policy in policies} == {
        "fixed_early",
        "fixed_midpoint",
        "fixed_deadline",
        "calibration_drift",
    }
    assert {json.dumps(policy["core"], sort_keys=True) for policy in policies} == {
        json.dumps(
            {"credits": 12, "optimizer_steps": 24, "optimizer_examples": 192},
            sort_keys=True,
        )
    }
    assert {policy["monitoring_decisions"] for policy in policies} == {59}
    assert len({policy["monitoring_forward_examples"] for policy in policies}) == 1
    assert all(policy["monitoring_exposure_overlap"] == 0 for policy in policies)
    assert all(policy["exposure_rows"] == 192 for policy in policies)
    assert manifest["synthetic_shift"] == {
        "click_day": 3,
        "first_legal_monitoring_day": 34,
        "adaptive_first_spend": 34,
        "deadline_first_spend": 35,
        "triggered_earlier": True,
    }
