from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from latesignal.cli import app
from latesignal.data.manifests import read_json

runner = CliRunner()
CONFIG = Path("configs/experiments/study_a.synthetic.yaml").resolve()


def test_public_study_a_runs_all_methods_with_identical_core_budget(tmp_path: Path) -> None:
    output = tmp_path / "study-a"

    result = runner.invoke(app, ["run", str(CONFIG), "--out", str(output), "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "complete"
    assert payload["counts"] == {"methods": 7}
    manifest = read_json(output / "manifest.json")
    methods = manifest["methods"]
    assert {method["method"] for method in methods} == {
        "complete_wait",
        "immediate_fake_negative",
        "fixed_wait",
        "dfm",
        "fnw",
        "es_dfm",
        "oracle_reference",
    }
    assert {json.dumps(method["core"], sort_keys=True) for method in methods} == {
        json.dumps(
            {"credits": 3, "optimizer_steps": 6, "optimizer_examples": 48},
            sort_keys=True,
        )
    }
    assert {tuple(method["schedule_seconds"]) for method in methods} == {
        (31 * 86_400, 32 * 86_400, 33 * 86_400)
    }
    assert all(method["exposure_rows"] == 48 for method in methods)
    esdfm = next(method for method in methods if method["method"] == "es_dfm")
    dfm = next(method for method in methods if method["method"] == "dfm")
    oracle = next(method for method in methods if method["method"] == "oracle_reference")
    assert esdfm["auxiliary"]["optimizer_examples"] > 0
    assert dfm["auxiliary"]["forward_examples"] == 48
    assert oracle["deployable"] is False
    assert oracle["ranking_eligible"] is False
    assert manifest["claims"]["published_number_reproduction"] is False


def test_study_a_is_reproducible_for_identical_seed_and_environment(tmp_path: Path) -> None:
    outputs = []
    for name in ("first", "second"):
        output = tmp_path / name
        result = runner.invoke(app, ["run", str(CONFIG), "--out", str(output), "--json"])
        assert result.exit_code == 0, result.stdout
        outputs.append(read_json(output / "manifest.json"))

    first_methods = outputs[0]["methods"]
    second_methods = outputs[1]["methods"]
    assert [method["exposure_sha256"] for method in first_methods] == [
        method["exposure_sha256"] for method in second_methods
    ]
    assert [method["final_model_sha256"] for method in first_methods] == [
        method["final_model_sha256"] for method in second_methods
    ]
