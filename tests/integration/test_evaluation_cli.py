from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from latesignal.cli import app
from latesignal.data.manifests import read_json, write_json_atomic

runner = CliRunner()


def _write_input(run_dir: Path, method: str, seed: int, *, candidate: bool) -> None:
    examples = []
    for day in range(65, 90):
        for index in range(4):
            label = int(index % 2 == 0)
            probability = (0.8 if label else 0.2) if candidate else (0.65 if label else 0.35)
            prior = (day + index) % 4
            examples.append(
                {
                    "click_id": f"click-{day}-{index}",
                    "click_day": day,
                    "final_label": label,
                    "probability": probability,
                    "cold_user": prior == 0,
                    "cold_product": prior == 0,
                    "prior_user_clicks": prior,
                    "prior_product_clicks": prior,
                    "product_price_bin": "low" if index < 2 else "high",
                    "device_type": "mobile" if index % 2 else "desktop",
                    "conversion_delay_days": 2.0 if label else None,
                }
            )
    run_dir.mkdir()
    write_json_atomic(
        run_dir / "evaluation-input.json",
        {
            "method": method,
            "seed": seed,
            "ranking_eligible": True,
            "sealed": True,
            "period_first_day": 65,
            "period_last_day": 89,
            "examples": examples,
        },
    )


def test_public_evaluate_and_compare_use_sealed_matched_final_rows(tmp_path: Path) -> None:
    run_dirs: list[Path] = []
    for method, candidate in (("fixed_deadline", False), ("calibration_drift", True)):
        for seed in (17, 41, 73):
            run_dir = tmp_path / f"{method}-{seed}"
            _write_input(run_dir, method, seed, candidate=candidate)
            run_dirs.append(run_dir)

    evaluated = runner.invoke(app, ["evaluate", str(run_dirs[0]), "--json"])
    assert evaluated.exit_code == 0, evaluated.stdout
    payload = json.loads(evaluated.stdout)
    assert payload["sealed"] is True
    assert payload["period"] == [65, 89]
    assert payload["overall"]["count"] == 100
    assert read_json(run_dirs[0] / "evaluation.json")["status"] == "complete"

    compared = runner.invoke(
        app,
        [
            "compare",
            *[str(path) for path in run_dirs],
            "--control",
            "fixed_deadline",
            "--candidate",
            "calibration_drift",
            "--replicates",
            "2000",
            "--json",
        ],
    )
    assert compared.exit_code == 0, compared.stdout
    comparison = json.loads(compared.stdout)
    assert comparison["status"] == "complete"
    assert comparison["seeds"] == [17, 41, 73]
    assert comparison["paired_intervals"]["log_loss"]["3"]["upper_95"] < 0.0
