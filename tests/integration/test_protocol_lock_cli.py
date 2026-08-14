from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from latesignal.cli import app
from latesignal.contracts.protocol import load_final_protocol
from latesignal.contracts.selection import SelectionResults
from latesignal.data.manifests import read_json, sha256_file, write_json_atomic
from latesignal.errors import ConsistencyError
from latesignal.experiments.estimate import enumerate_matrix
from latesignal.experiments.protocol_lock import verify_protocol_lock

runner = CliRunner()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _authored_configs(tmp_path: Path) -> tuple[Path, str, dict[str, int]]:
    protocol = yaml.safe_load(Path("configs/protocol.yaml").read_text(encoding="utf-8"))
    final = yaml.safe_load(Path("configs/experiments/final.yaml").read_text(encoding="utf-8"))
    protocol_path = tmp_path / "protocol.yaml"
    final_path = tmp_path / "final.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    final["protocol"] = "protocol.yaml"
    final["target_device"] = "cpu"
    final["require_real_pilot"] = True
    final["pilot"]["prepared_root"] = "processed"
    final["caps"] = {
        "max_runs": 100,
        "max_gpu_hours": 1_000.0,
        "max_working_disk_gb": 30.0,
        "max_retained_disk_gb": 30.0,
    }
    final_path.write_text(yaml.safe_dump(final, sort_keys=False), encoding="utf-8")
    final_config, protocol_config, protocol_sha256 = load_final_protocol(final_path)
    return final_path, protocol_sha256, enumerate_matrix(protocol_config, final_config)


def _result_fields(name: str, index: int, *, loss: float) -> dict[str, object]:
    return {
        "config_sha256": _digest(f"{name}-{index}"),
        "status": "complete",
        "mean_selection_log_loss": loss,
        "measured_compute_seconds": float(100 + index),
        "parameter_count": 1_000 + index,
        "failure_reason": None,
    }


def _selection(protocol_sha256: str) -> dict[str, Any]:
    model_candidates: list[dict[str, object]] = []
    index = 0
    for learning_rate in (0.0001, 0.0003, 0.001):
        for weight_decay in (0.0, 0.00001, 0.0001):
            for dropout in (0.0, 0.1):
                for feature_policy in ("compact", "large"):
                    model_candidates.append(
                        {
                            **_result_fields("model", index, loss=0.4 + index / 100.0),
                            "learning_rate": learning_rate,
                            "weight_decay": weight_decay,
                            "dropout": dropout,
                            "feature_policy": feature_policy,
                            "seed": 17,
                        }
                    )
                    index += 1
    model_candidates[1]["mean_selection_log_loss"] = 0.4000005
    model_candidates[1]["measured_compute_seconds"] = 10.0
    delayed_candidates: list[dict[str, object]] = []
    index = 0
    for method in ("fixed_wait", "es_dfm"):
        for wait_days in (1, 3, 7, 14):
            delayed_candidates.append(
                {
                    **_result_fields("delayed", index, loss=0.5 + index / 100.0),
                    "method": method,
                    "wait_days": wait_days,
                    "seed": 17,
                }
            )
            index += 1
    sampler_candidates: list[dict[str, object]] = []
    index = 0
    for window in (1, 3, 7):
        for capacity in (1_000_000, 5_000_000):
            sampler_candidates.append(
                {
                    **_result_fields("sampler", index, loss=0.6 + index / 100.0),
                    "recent_window_days": window,
                    "reservoir_capacity": capacity,
                    "seed": 17,
                }
            )
            index += 1
    return {
        "version": 1,
        "protocol_sha256": protocol_sha256,
        "window": {
            "first_click_day": 25,
            "last_click_day": 34,
            "all_labels_mature_by_day": 64,
            "embargo_outcomes_accessed": False,
            "final_period_metrics_accessed": False,
        },
        "model_candidates": model_candidates,
        "delayed_candidates": delayed_candidates,
        "sampler_candidates": sampler_candidates,
    }


def _prepared_manifest(tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    feature = processed / "features" / "click_day=0" / "part.parquet"
    feature.parent.mkdir(parents=True)
    feature.write_bytes(b"synthetic-parquet-fixture")
    sha256, size = sha256_file(feature)
    manifest_path = processed / "manifests" / "preparation.json"
    write_json_atomic(
        manifest_path,
        {
            "manifest_version": 1,
            "rows": {"reconciled": True},
            "numeric_statistics": {"fit_click_days": [0, 14]},
            "files": [
                {
                    "path": "features/click_day=0/part.parquet",
                    "sha256": sha256,
                    "bytes": size,
                }
            ],
        },
    )
    return manifest_path


def _feasibility(protocol_sha256: str, matrix: dict[str, int]) -> dict[str, Any]:
    checks = {
        "runs": True,
        "compute_hours": True,
        "working_disk": True,
        "retained_disk": True,
    }
    return {
        "feasibility_model_version": 2,
        "status": "passed",
        "protocol_sha256": protocol_sha256,
        "blockers": [],
        "matrix": matrix,
        "benchmark": {"requested_device_available": True},
        "real_data_pilot": {
            "status": "measured",
            "workload_inventory": {
                "total_click_rows_days_0_89": 1_000,
                "selection_rows_days_25_34": 100,
                "final_rows_days_65_89": 250,
            },
        },
        "selected_steps_per_credit": 500,
        "projections": [
            {
                "steps_per_credit": steps,
                "fits_caps": True,
                "cap_checks": checks,
            }
            for steps in (100, 250, 500)
        ],
    }


def test_selection_contract_rejects_embargo_access_and_incomplete_grid(
    tmp_path: Path,
) -> None:
    _, protocol_sha256, _ = _authored_configs(tmp_path)
    selection = _selection(protocol_sha256)
    selection["window"]["embargo_outcomes_accessed"] = True

    with pytest.raises(ValueError):
        SelectionResults.model_validate(selection)

    selection = _selection(protocol_sha256)
    selection["model_candidates"].pop()
    with pytest.raises(ValueError, match="exhaustive model grid"):
        SelectionResults.model_validate(selection)


def test_public_protocol_lock_hashes_selection_data_code_and_environment(tmp_path: Path) -> None:
    final_path, protocol_sha256, matrix = _authored_configs(tmp_path)
    selection_path = tmp_path / "selection.json"
    feasibility_path = tmp_path / "feasibility.json"
    data_manifest = _prepared_manifest(tmp_path)
    output = tmp_path / "protocol-lock.json"
    write_json_atomic(selection_path, _selection(protocol_sha256))
    write_json_atomic(feasibility_path, _feasibility(protocol_sha256, matrix))

    result = runner.invoke(
        app,
        [
            "protocol",
            "lock",
            str(final_path),
            "--selection",
            str(selection_path),
            "--feasibility",
            str(feasibility_path),
            "--data-manifest",
            str(data_manifest),
            "--out",
            str(output),
            "--allow-dirty",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    lock = verify_protocol_lock(output)
    assert payload["lock_sha256"] == lock["lock_sha256"]
    assert lock["publication_eligible"] is (not lock["git"]["dirty"])
    assert lock["git"]["allow_dirty_override"] is lock["git"]["dirty"]
    assert lock["selection_decisions"]["model"]["config_sha256"] == _digest("model-1")
    assert lock["selection_decisions"]["derived"] == {
        "shared_wait_days": 1,
        "study_b_method": "fixed_wait",
    }
    assert lock["data"]["verified_files"] == 1
    assert lock["final_seeds"] == [17, 41, 73]
    assert lock["selected_steps_per_credit"] == 500

    stored = read_json(output)
    stored["selected_steps_per_credit"] = 100
    write_json_atomic(output, stored, overwrite=True)
    with pytest.raises(ConsistencyError, match="does not match"):
        verify_protocol_lock(output)


def test_selection_lock_refuses_infrastructure_failure(tmp_path: Path) -> None:
    _, protocol_sha256, _ = _authored_configs(tmp_path)
    selection = _selection(protocol_sha256)
    selection["model_candidates"][0].update(
        {
            "status": "infrastructure_failed",
            "mean_selection_log_loss": None,
            "measured_compute_seconds": None,
            "parameter_count": None,
            "failure_reason": "OOM",
        }
    )
    parsed = SelectionResults.model_validate(selection)

    with pytest.raises(ConsistencyError, match="non-scientific failure"):
        from latesignal.experiments.protocol_lock import selection_decisions

        selection_decisions(parsed)
