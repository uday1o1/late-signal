from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from latesignal.contracts.protocol import load_final_protocol
from latesignal.errors import ConfigurationError
from latesignal.experiments.estimate import enumerate_matrix, estimate_protocol


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
    benchmark = {
        "requested_device": "cpu",
        "measured_device": "cpu",
        "requested_device_available": True,
        "training_examples_per_second": 10_000_000.0,
        "checkpoint_bytes": 1_000_000,
        "peak_host_memory_gb": 1.0,
    }
    monkeypatch.setattr("latesignal.experiments.estimate._benchmark", lambda _: benchmark)
    monkeypatch.setattr(
        "latesignal.experiments.estimate._real_pilot",
        lambda *_: {"status": "unavailable"},
    )

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
    benchmark = {
        "requested_device": "cpu",
        "measured_device": "cpu",
        "requested_device_available": True,
        "training_examples_per_second": 10_000_000.0,
        "checkpoint_bytes": checkpoint_bytes,
        "peak_host_memory_gb": 1.0,
    }
    monkeypatch.setattr("latesignal.experiments.estimate._benchmark", lambda _: benchmark)
    monkeypatch.setattr(
        "latesignal.experiments.estimate._real_pilot",
        lambda *_: {"status": "unavailable"},
    )

    result = estimate_protocol(
        final,
        protocol,
        config_path=path,
        protocol_sha256=protocol_sha256,
    )

    projection = result["projections"][0]
    report_gb = 89 * 250_000 / 1024**3
    assert projection["working_disk_gb"] == pytest.approx(15.5 + 3.0 + report_gb)
    assert projection["retained_disk_gb"] == pytest.approx(report_gb)
    assert result["assumptions"]["checkpoint_working_copies"] == 3
    assert result["assumptions"]["completed_checkpoints_retained"] == 0


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
