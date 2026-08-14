from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import latesignal.experiments.estimate as estimate_module
from latesignal.contracts.protocol import ResourceCaps, load_final_protocol
from latesignal.errors import ConfigurationError, ConsistencyError
from latesignal.experiments.estimate import (
    _benchmark_workspace,
    _worst_case_workload,
    enumerate_matrix,
    estimate_protocol,
)

_WORK_ROOT_NAME = ".latesignal-feasibility-benchmark"


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
        "checkpoint_state_materialization_seconds": 0.0004,
        "checkpoint_durable_write_seconds": 0.0006,
        "final_snapshot_write_seconds": 0.001,
        "final_snapshot_verify_seconds": 0.0005,
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
        "one_model_checkpoint_writes": 1_560,
        "three_model_checkpoint_writes": 1_020,
        "actual_checkpoint_generations": 2_580,
        "equivalent_single_model_checkpoint_writes": 4_620,
        "final_snapshot_writes": 132,
        "checkpoint_snapshot_verifications": 4_875,
        "terminal_snapshot_verifications": 132,
        "final_snapshot_verifications": 5_007,
    }


def test_protocol_refuses_a_silently_narrowed_candidate_set(tmp_path: Path) -> None:
    path = _write_configs(tmp_path, narrowed=True)

    with pytest.raises(ConfigurationError, match="validation failed") as caught:
        load_final_protocol(path)

    assert "locked candidate set" in str(caught.value.details)


def test_checked_in_final_resource_caps_match_the_authorized_ceiling() -> None:
    final, _, _ = load_final_protocol(Path("configs/experiments/final.yaml"))

    assert final.caps.max_runs == 89
    assert final.caps.max_gpu_hours == 25.0
    assert final.caps.max_working_disk_gb == 26.0
    assert final.caps.max_retained_disk_gb == 2.0


def test_estimator_selects_largest_quality_independent_candidate_that_fits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_configs(tmp_path)
    final, protocol, protocol_sha256 = load_final_protocol(path)
    monkeypatch.setattr(
        "latesignal.experiments.estimate._benchmark", lambda *_args, **_kwargs: _benchmark()
    )
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
        lambda *_args, **_kwargs: _benchmark(checkpoint_bytes=checkpoint_bytes),
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


def test_estimator_rejects_original_cap_after_counting_durable_checkpoint_floor(
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
    monkeypatch.setattr(
        "latesignal.experiments.estimate._benchmark", lambda *_args, **_kwargs: benchmark
    )
    monkeypatch.setattr("latesignal.experiments.estimate._real_pilot", lambda *_: pilot)

    result = estimate_protocol(
        final,
        protocol,
        config_path=path,
        protocol_sha256=protocol_sha256,
    )

    assert result["status"] == "blocked"
    assert result["selected_steps_per_credit"] is None
    assert result["blockers"] == ["NO_STEPS_PER_CREDIT_CANDIDATE_FITS_CAPS"]
    assert result["projections"][2]["cap_checks"]["compute_hours"] is False
    assert result["projections"][1]["fits_caps"] is False
    assert result["projections"][0]["fits_caps"] is False
    assert (
        result["projections"][0]["workload"]["checkpoint_generation_rate_source"]
        == "authored_machine_pilot_floor"
    )
    assert result["projections"][0]["workload"]["checkpoint_pilot_floor_applied"] is True
    workload = result["projections"][0]["workload"]
    assert workload["checkpoint_pilot_floor_seconds"] == pytest.approx(2_580 * 6.75258791425052)
    assert workload["final_snapshot_write_seconds"] == pytest.approx(132 * 0.001)
    assert workload["final_snapshot_verification_seconds"] == pytest.approx(5_007 * 0.0005)
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


def test_benchmark_workspace_recovers_owned_interrupted_state_and_cleans_normal_exit(
    tmp_path: Path,
) -> None:
    root = tmp_path / _WORK_ROOT_NAME

    with (
        pytest.raises(RuntimeError, match="forced interruption"),
        _benchmark_workspace(root, filesystem_reference=tmp_path) as (owned, _),
    ):
        (owned / "rolling").mkdir()
        raise RuntimeError("forced interruption")

    assert root.is_dir()
    with _benchmark_workspace(root, filesystem_reference=tmp_path) as (owned, device):
        assert owned == root
        assert isinstance(device, int)
        (owned / "snapshots").mkdir()

    assert not root.exists()
    assert (tmp_path / f"{_WORK_ROOT_NAME}.lock").is_file()


def test_benchmark_workspace_rejects_unowned_or_unknown_content(tmp_path: Path) -> None:
    root = tmp_path / _WORK_ROOT_NAME
    root.mkdir()
    (root / "foreign.txt").write_text("not owned\n", encoding="utf-8")

    with (
        pytest.raises(ConsistencyError, match="not owned"),
        _benchmark_workspace(root, filesystem_reference=tmp_path),
    ):
        pass


def test_benchmark_workspace_rejects_concurrent_owner(tmp_path: Path) -> None:
    root = tmp_path / _WORK_ROOT_NAME

    with (
        _benchmark_workspace(root, filesystem_reference=tmp_path),
        pytest.raises(ConsistencyError, match="Another feasibility benchmark"),
        _benchmark_workspace(root, filesystem_reference=tmp_path),
    ):
        pass


def test_benchmark_workspace_rejects_filesystem_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / _WORK_ROOT_NAME
    reference = tmp_path / "reference"
    reference.mkdir()

    monkeypatch.setattr(
        "latesignal.experiments.estimate._filesystem_device",
        lambda path: 2 if path.resolve() == reference else 1,
    )

    with (
        pytest.raises(ConsistencyError, match="different filesystem"),
        _benchmark_workspace(root, filesystem_reference=reference),
    ):
        pass


def test_tiny_benchmark_executes_durable_checkpoint_and_snapshot_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_configs(tmp_path)
    final, _, _ = load_final_protocol(path)
    final = final.model_copy(
        update={
            "target_device": "cuda",
            "pilot": final.pilot.model_copy(
                update={
                    "benchmark_examples": 32,
                    "benchmark_batch_size": 8,
                    "benchmark_steps": 1,
                }
            ),
        }
    )
    monkeypatch.setattr(estimate_module.torch.cuda, "is_available", lambda: False)
    root = tmp_path / _WORK_ROOT_NAME

    result = estimate_module._benchmark(
        final,
        work_root=root,
        filesystem_reference=tmp_path,
    )

    samples = result["checkpoint_durable_write_samples_seconds"]
    assert isinstance(samples, list)
    assert len(samples) == 3
    assert all(isinstance(value, float) and value > 0.0 for value in samples)
    assert result["checkpoint_durable_write_seconds"] == max(samples[1:])
    assert result["checkpoint_state_materialization_seconds"] > 0.0
    assert result["checkpoint_write_seconds"] == pytest.approx(
        result["checkpoint_state_materialization_seconds"]
        + result["checkpoint_durable_write_seconds"]
    )
    assert result["final_snapshot_write_seconds"] > 0.0
    assert result["final_snapshot_verify_seconds"] > 0.0
    assert result["checkpoint_benchmark"]["mode"] == "production-durable-rolling-store"
    assert not root.exists()


def test_benchmark_disk_preflight_refuses_insufficient_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DiskUsage:
        free = 1

    monkeypatch.setattr(estimate_module.shutil, "disk_usage", lambda _: DiskUsage())

    with pytest.raises(ConsistencyError, match="Insufficient free disk"):
        estimate_module._require_benchmark_disk(
            tmp_path,
            checkpoint_bytes=1_000_000,
            model_state_bytes=400_000,
        )


def test_estimator_uses_component_rate_when_it_exceeds_authored_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_configs(tmp_path)
    final, protocol, protocol_sha256 = load_final_protocol(path)
    final = final.model_copy(
        update={
            "pilot": final.pilot.model_copy(update={"min_checkpoint_generation_seconds": 0.00001})
        }
    )
    monkeypatch.setattr(
        "latesignal.experiments.estimate._benchmark",
        lambda *_args, **_kwargs: _benchmark(),
    )
    monkeypatch.setattr("latesignal.experiments.estimate._real_pilot", lambda *_: _pilot())

    result = estimate_protocol(
        final,
        protocol,
        config_path=path,
        protocol_sha256=protocol_sha256,
    )

    workload = result["projections"][0]["workload"]
    assert workload["checkpoint_generation_rate_source"] == (
        "production_equivalent_component_benchmark"
    )
    assert workload["checkpoint_pilot_floor_applied"] is False
    assert workload["checkpoint_component_seconds"] == pytest.approx(4_620 * 0.001)
    assert workload["checkpoint_generation_seconds"] == pytest.approx(4_620 * 0.001)
    assert workload["final_snapshot_write_seconds"] == pytest.approx(132 * 0.001)
    assert workload["final_snapshot_verification_seconds"] == pytest.approx(5_007 * 0.0005)
    assert workload["checkpoint_seconds"] == pytest.approx(
        4_620 * 0.001 + 132 * 0.001 + 5_007 * 0.0005
    )
