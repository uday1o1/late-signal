from __future__ import annotations

from pathlib import Path

import pytest

from latesignal.data.manifests import write_json_atomic
from latesignal.errors import ConsistencyError
from latesignal.experiments.feasibility_context import (
    bind_feasibility_context,
    verify_feasibility_context,
)


def _context(device_uuid: str = "GPU-exact") -> dict[str, object]:
    return {
        "version": 2,
        "git_commit": "1" * 40,
        "source_tree_sha256": "2" * 64,
        "dependency_lock_sha256": "3" * 64,
        "runtime_sha256": "4" * 64,
        "prepared_manifest_sha256": "5" * 64,
        "final_config_sha256": "6" * 64,
        "device_uuid": device_uuid,
        "device_identity": f"{device_uuid}, GPU, 610.57.04, 97887",
    }


def _measured(**values: object) -> dict[str, object]:
    return {
        "status": "passed",
        "benchmark": {"environment": {"device_uuid": "GPU-exact"}},
        **values,
    }


def test_feasibility_context_is_immutable_and_runtime_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measured = tmp_path / "measured.json"
    output = tmp_path / "feasibility.json"
    write_json_atomic(measured, _measured(selected_steps_per_credit=500))
    monkeypatch.setattr(
        "latesignal.experiments.feasibility_context._current_context",
        lambda *args, **kwargs: _context(),
    )

    bound = bind_feasibility_context(
        measured,
        output,
        repository=tmp_path,
        data_manifest_path=tmp_path / "data.json",
        final_config_path=tmp_path / "final.yaml",
        device_uuid="GPU-exact",
    )
    verified = verify_feasibility_context(
        output,
        repository=tmp_path,
        data_manifest_path=tmp_path / "data.json",
        final_config_path=tmp_path / "final.yaml",
        device_uuid="GPU-exact",
    )

    assert verified == bound
    assert bound["execution_context"]["device_uuid"] == "GPU-exact"
    assert isinstance(bound["execution_context"]["measured_payload_sha256"], str)


def test_feasibility_context_rejects_device_or_content_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measured = tmp_path / "measured.json"
    output = tmp_path / "feasibility.json"
    write_json_atomic(measured, _measured())
    monkeypatch.setattr(
        "latesignal.experiments.feasibility_context._current_context",
        lambda *args, **kwargs: _context(),
    )
    bind_feasibility_context(
        measured,
        output,
        repository=tmp_path,
        data_manifest_path=tmp_path / "data.json",
        final_config_path=tmp_path / "final.yaml",
        device_uuid="GPU-exact",
    )
    monkeypatch.setattr(
        "latesignal.experiments.feasibility_context._current_context",
        lambda *args, **kwargs: _context("GPU-other"),
    )

    with pytest.raises(ConsistencyError, match="different execution context"):
        verify_feasibility_context(
            output,
            repository=tmp_path,
            data_manifest_path=tmp_path / "data.json",
            final_config_path=tmp_path / "final.yaml",
            device_uuid="GPU-other",
        )


def test_feasibility_context_rejects_top_level_payload_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measured = tmp_path / "measured.json"
    output = tmp_path / "feasibility.json"
    write_json_atomic(
        measured,
        _measured(
            blockers=[],
            selected_steps_per_credit=500,
            projections=[{"steps_per_credit": 500, "fits_caps": True}],
        ),
    )
    monkeypatch.setattr(
        "latesignal.experiments.feasibility_context._current_context",
        lambda *args, **kwargs: _context(),
    )
    bound = bind_feasibility_context(
        measured,
        output,
        repository=tmp_path,
        data_manifest_path=tmp_path / "data.json",
        final_config_path=tmp_path / "final.yaml",
        device_uuid="GPU-exact",
    )
    bound["selected_steps_per_credit"] = 100
    write_json_atomic(output, bound, overwrite=True)

    with pytest.raises(ConsistencyError, match="payload failed its content seal"):
        verify_feasibility_context(
            output,
            repository=tmp_path,
            data_manifest_path=tmp_path / "data.json",
            final_config_path=tmp_path / "final.yaml",
            device_uuid="GPU-exact",
        )


def test_feasibility_binding_requires_the_exact_visibility_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measured = tmp_path / "measured.json"
    write_json_atomic(measured, _measured())
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)

    with pytest.raises(ConsistencyError, match="exactly one CUDA device"):
        bind_feasibility_context(
            measured,
            tmp_path / "bound.json",
            repository=tmp_path,
            data_manifest_path=tmp_path / "data.json",
            final_config_path=tmp_path / "final.yaml",
            device_uuid="GPU-exact",
        )


def test_feasibility_binding_rejects_a_different_measured_gpu(
    tmp_path: Path,
) -> None:
    measured = tmp_path / "measured.json"
    write_json_atomic(
        measured,
        {
            "status": "passed",
            "benchmark": {"environment": {"device_uuid": "GPU-other"}},
        },
    )

    with pytest.raises(ConsistencyError, match="selected exported GPU UUID"):
        bind_feasibility_context(
            measured,
            tmp_path / "bound.json",
            repository=tmp_path,
            data_manifest_path=tmp_path / "data.json",
            final_config_path=tmp_path / "final.yaml",
            device_uuid="GPU-exact",
        )
