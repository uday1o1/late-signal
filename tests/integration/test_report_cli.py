from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from latesignal.cli import app
from latesignal.data.manifests import read_json, write_json_atomic

runner = CliRunner()
DIGEST = "a" * 64


def _report_input() -> dict[str, object]:
    metrics = {
        "count": 100,
        "positives": 20,
        "log_loss": 0.42,
        "brier_score": 0.13,
        "pr_auc": 0.61,
        "roc_auc": 0.77,
        "calibration_intercept": 0.01,
        "calibration_slope": 0.98,
        "expected_calibration_error": 0.02,
        "reliability": [
            {
                "index": index,
                "lower": index / 10,
                "upper": (index + 1) / 10,
                "count": 10,
                "positives": 2,
                "mean_probability": index / 10 + 0.05,
                "observed_rate": 0.2,
            }
            for index in range(10)
        ],
    }
    return {
        "version": 1,
        "title": "LateSignal synthetic evidence",
        "result_kind": "synthetic",
        "dataset": {
            "name": "Deterministic synthetic fixture",
            "license_id": "synthetic",
            "source_archive_sha256": None,
            "preparation_manifest_sha256": None,
            "accepted_rows": 100,
            "quarantined_rows": 0,
        },
        "protocol": {
            "lock_sha256": DIGEST,
            "code_commit": "1234567",
            "environment_sha256": DIGEST,
            "seeds": [17],
            "publication_eligible": False,
        },
        "methods": [
            {
                "method": "complete_wait",
                "deployable": True,
                "ranking_eligible": True,
                "credits": 3,
                "core_optimizer_steps": 6,
                "core_optimizer_examples": 48,
                "auxiliary_optimizer_steps": 0,
                "auxiliary_optimizer_examples": 0,
                "status": "complete",
            }
        ],
        "schedulers": [
            {
                "scheduler": "fixed_deadline",
                "seed": 17,
                "credits": 12,
                "optimizer_steps": 24,
                "optimizer_examples": 192,
                "monitoring_examples": 59_000,
                "monitoring_exposure_overlap": 0,
                "trigger_days": [35],
                "status": "complete",
            }
        ],
        "evaluations": [
            {
                "method": "complete_wait",
                "seed": 17,
                "ranking_eligible": True,
                "metrics": metrics,
            }
        ],
        "slices": [
            {
                "method": "complete_wait",
                "seed": 17,
                "dimension": "cold_user",
                "value": "true",
                "count": 40,
                "positives": 8,
                "ranking_eligible": True,
                "suppression_reason": None,
                "log_loss": 0.44,
            },
            {
                "method": "complete_wait",
                "seed": 17,
                "dimension": "device_type",
                "value": "unseen",
                "count": 0,
                "positives": 0,
                "ranking_eligible": False,
                "suppression_reason": "empty",
                "log_loss": None,
            },
        ],
        "paired_intervals": [
            {
                "control": "fixed_deadline",
                "candidate": "calibration_drift",
                "metric": "log_loss",
                "block_days": 3,
                "replicates": 2_000,
                "point_difference": -0.01,
                "lower_95": -0.02,
                "upper_95": -0.001,
                "seed_differences": [{"seed": 17, "difference": -0.01}],
            }
        ],
        "intermediate_budget": [
            {
                "method": "complete_wait",
                "budget_fraction": fraction,
                "core_examples": int(100 * fraction),
                "log_loss": 0.5 - fraction / 20,
            }
            for fraction in (0.25, 0.5, 0.75, 1.0)
        ],
        "compute": [
            {
                "method": "complete_wait",
                "log_loss": 0.42,
                "core_examples": 48,
                "total_examples": 48,
                "wall_seconds": 1.5,
                "peak_memory_gb": 0.2,
                "pareto_efficient": True,
            }
        ],
        "leakage_audit": [
            {
                "control": "prediction_before_reveal",
                "status": "passed",
                "evidence": "Boundary mutation test failed for the intended reason.",
            }
        ],
        "limitations": ["Synthetic evidence does not establish real-data quality."],
        "reproduction_commands": [
            "uv run latesignal run configs/experiments/study_a.synthetic.yaml --out runs/study-a"
        ],
        "claim": {
            "scheduler_outcome": "not_evaluated",
            "published_number_reproduction": False,
            "statement": "Synthetic qualification only. No real-data result is claimed.",
        },
    }


def test_public_report_renders_static_html_and_underlying_aggregate_tables(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json_atomic(run_dir / "report-input.json", _report_input())

    result = runner.invoke(app, ["report", str(run_dir), "--format", "html", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["aggregate_only"] is True
    report_root = run_dir / "report"
    html = (report_root / "report.html").read_text(encoding="utf-8")
    assert "Chronology and information availability" in html
    assert "Study A equal-budget methods" in html
    assert "Paired uncertainty" in html
    assert "Limitations and threats to validity" in html
    assert (report_root / "tables" / "slices.csv").exists()
    manifest = read_json(report_root / "manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["aggregate_only"] is True
    assert len(manifest["outputs"]) == 11


def test_report_refuses_unknown_row_level_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    value = _report_input()
    value["raw_rows"] = [{"user_id": "must-not-enter-report"}]
    write_json_atomic(run_dir / "report-input.json", value)

    result = runner.invoke(app, ["report", str(run_dir), "--format", "json", "--json"])

    assert result.exit_code == 2, result.stdout
    assert json.loads(result.stdout)["error"] == "INVALID_CONFIGURATION"
