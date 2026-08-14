from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from latesignal.contracts.protocol import ResourceCaps, load_final_protocol
from latesignal.errors import ConfigurationError
from latesignal.experiments.estimate import (
    _worst_case_workload,
    enumerate_matrix,
    estimate_protocol,
)


def _benchmark(*, checkpoint_bytes: int = 1_000_000) -> dict[str, object]:
    return {
        "requested_device": "cpu",
        "measured_device": "cpu",
        "requested_device_available": True,
        "training_examples_per_second": 10_000_000.0,
        "training_step_seconds": 0.0001,
        "es_main_training_step_seconds": 0.0002,
        "dfm_training_step_seconds": 0.00012,
        "prediction_examples_per_second": 10_000_000.0,
        "checkpoint_bytes": checkpoint_bytes,
        "model_state_bytes": 0,
        "checkpoint_write_seconds": 0.001,
        "prediction_artifact_bytes_per_row": 0.0,
        "exposure_artifact_bytes_per_row": 0.0,
        "peak_host_memory_gb": 1.0,
    }


def _pilot() -> dict[str, object]:
    return {
        "status": "measured",
        "workload_inventory": {
            "total_click_rows_days_0_89": 1_000,
            "selection_rows_days_25_34": 100,
            "final_rows_days_65_89": 250,
        },
    }


def _write_configs(tmp_path: Path, *, narrowed: bool = False) -> Path:
    protocol = yaml.safe_load(Path("configs/protocol.yaml").read_text(encoding="utf-8"))
    final = yaml.safe_load(Path("configs/experiments/final.yaml").read_text(encoding="utf-8"))
    if narrowed:
        protocol["model_selection"]["learning_rates"] = [0.0003]
    protocol_path = tmp_path / "protocol.yaml"
    final_path = tmp_path / "final.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    final["protocol"] = "protocol.yaml"
    final["target_device"] = "cpu"
    final["require_real_pilot"] = False
    final["pilot"]["prepared_root"] = "missing-processed"
    final["caps"] = {
        "max_runs": 100,
        "max_gpu_hours": 1_000.0,
        "max_working_disk_gb": 30.0,
        "max_retained_disk_gb": 30.0,
    }
    final_path.write_text(yaml.safe_dump(final, sort_keys=False), encoding="utf-8")
    return final_path


def test_locked_protocol_enumerates_the_exact_authored_matrix(tmp_path: Path) -> None:
    path = _write_configs(tmp_path)
    final, protocol, _ = load_final_protocol(path)

    matrix = enumerate_matrix(protocol, final)

    assert matrix == {
        "model_selection_runs": 36,
        "delayed_selection_runs": 8,
        "sampler_selection_runs": 6,
        "final_study_a_runs": 21,
        "final_study_b_runs": 12,
        "offline_reference_runs": 6,
        "online_runs": 83,
        "total_runs": 89,
        "total_online_credits": 1_883,
    }
    assert protocol.selection_defaults.model_method == "complete_wait"
    assert (
        protocol.selection_defaults.recent_window_days,
        protocol.selection_defaults.reservoir_capacity,
    ) == (3, 1_000_000)
    assert (
        protocol.selection_defaults.first_credit_day,
        protocol.selection_defaults.last_credit_day,
    ) == (55, 64)
    assert protocol.final_training.initialization_steps == 500
    assert _worst_case_workload(protocol, final, matrix) == {
        "selection_runs": 50,
        "final_online_runs": 33,
        "initialization_runs": 83,
        "initialization_steps": 41_500,
        "base_core_credits": 1_285,
        "es_core_credits": 421,
        "dfm_core_credits": 177,
        "auxiliary_steps": 109_200,
        "one_model_checkpoint_writes": 1_578,
        "three_model_checkpoint_writes": 471,
        "equivalent_single_model_checkpoint_writes": 2_991,
    }


def test_protocol_refuses_a_silently_narrowed_candidate_set(tmp_path: Path) -> None:
    path = _write_configs(tmp_path, narrowed=True)

    with pytest.raises(ConfigurationError, match="validation failed") as caught:
        load_final_protocol(path)

    assert "locked candidate set" in str(caught.value.details)


def test_estimator_selects_largest_quality_independent_candidate_that_fits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_configs(tmp_path)
    final, protocol, protocol_sha256 = load_final_protocol(path)
    monkeypatch.setattr("latesignal.experiments.estimate._benchmark", lambda _: _benchmark())
    monkeypatch.setattr("latesignal.experiments.estimate._real_pilot", lambda *_: _pilot())

    result = estimate_protocol(
        final,
        protocol,
        config_path=path,
        protocol_sha256=protocol_sha256,
    )

    assert result["status"] == "passed"
    assert result["selected_steps_per_credit"] == 500
    assert result["blockers"] == []
    assert result["assumptions"]["quality_metrics_used_for_steps_choice"] is False


def test_estimator_models_rolling_checkpoint_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_configs(tmp_path)
    final, protocol, protocol_sha256 = load_final_protocol(path)
    checkpoint_bytes = 1024**3
    monkeypatch.setattr(
        "latesignal.experiments.estimate._benchmark",
        lambda _: _benchmark(checkpoint_bytes=checkpoint_bytes),
    )
    monkeypatch.setattr("latesignal.experiments.estimate._real_pilot", lambda *_: _pilot())

    result = estimate_protocol(
        final,
        protocol,
        config_path=path,
        protocol_sha256=protocol_sha256,
    )

    projection = result["projections"][0]
    report_gb = 89 * 250_000 / 1024**3
    assert projection["working_disk_gb"] == pytest.approx(9.5 + 9.0 + report_gb)
    assert projection["retained_disk_gb"] == pytest.approx(report_gb)
    assert result["assumptions"]["checkpoint_working_copies"] == 3
    assert result["assumptions"]["completed_checkpoints_retained"] == 0
    assert result["assumptions"]["execution_host_has_source_artifacts"] is False


def test_estimator_rejects_k500_after_including_real_workload_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_configs(tmp_path)
    final, protocol, protocol_sha256 = load_final_protocol(path)
    final = final.model_copy(
        update={
            "caps": ResourceCaps(
                max_runs=89,
                max_gpu_hours=4.0,
                max_working_disk_gb=25.0,
                max_retained_disk_gb=2.0,
            )
        }
    )
    benchmark = _benchmark(checkpoint_bytes=1_226_380_695)
    benchmark.update(
        {
            "training_step_seconds": 0.00742,
            "es_main_training_step_seconds": 0.012,
            "dfm_training_step_seconds": 0.008,
            "prediction_examples_per_second": 3_677_150.0,
            "model_state_bytes": 409_000_000,
            "checkpoint_write_seconds": 1.067,
            "prediction_artifact_bytes_per_row": 25.0,
            "exposure_artifact_bytes_per_row": 15.0,
        }
    )
    pilot = {
        "status": "measured",
        "workload_inventory": {
            "total_click_rows_days_0_89": 15_924_859,
            "selection_rows_days_25_34": 1_770_000,
            "final_rows_days_65_89": 4_420_000,
        },
    }
    monkeypatch.setattr("latesignal.experiments.estimate._benchmark", lambda _: benchmark)
    monkeypatch.setattr("latesignal.experiments.estimate._real_pilot", lambda *_: pilot)

    result = estimate_protocol(
        final,
        protocol,
        config_path=path,
        protocol_sha256=protocol_sha256,
    )

    assert result["status"] == "passed"
    assert result["selected_steps_per_credit"] == 250
    assert result["projections"][2]["cap_checks"]["compute_hours"] is False
    assert result["projections"][1]["fits_caps"] is True
    assert result["worst_case_workload"]["auxiliary_steps"] == 109_200


def test_real_data_gate_refuses_cross_batch_extrapolation(tmp_path: Path) -> None:
    final = yaml.safe_load(Path("configs/experiments/final.yaml").read_text(encoding="utf-8"))
    final["pilot"]["benchmark_batch_size"] = 128
    final["protocol"] = "protocol.yaml"
    (tmp_path / "protocol.yaml").write_text(
        Path("configs/protocol.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    path = tmp_path / "final.yaml"
    path.write_text(yaml.safe_dump(final, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="locked training batch size"):
        load_final_protocol(path)
