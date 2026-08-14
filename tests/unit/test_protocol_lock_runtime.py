from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from latesignal.data.manifests import canonical_json_bytes, sha256_file, write_json_atomic
from latesignal.errors import ConsistencyError
from latesignal.experiments.protocol_lock import _selection_execution, verify_locked_final_runtime


def test_cross_commit_selection_provenance_is_hashed_and_target_bound(tmp_path: Path) -> None:
    selection_path = tmp_path / "selection" / "selection-results.json"
    write_json_atomic(selection_path, {"status": "complete"})
    selection_sha256, _ = sha256_file(selection_path)
    payload: dict[str, object] = {
        "version": 1,
        "status": "verified_cross_commit_reuse",
        "reason": "post_selection_scheduler_boundary_fix",
        "source_commit": "1" * 40,
        "target_commit": "2" * 40,
        "source_exit_stage": "cuda_resume_qualification",
        "source_exit_code": 5,
        "source_error": "Scheduler decisions must occur on daily boundaries",
        "source_protocol_lock_sha256": "3" * 64,
        "source_gpu_uuid": "GPU-test",
        "target_gpu_uuid": "GPU-test",
        "source_environment_sha256": "7" * 64,
        "source_protocol_sha256": "4" * 64,
        "source_data_manifest_sha256": "5" * 64,
        "source_final_config_file_sha256": "6" * 64,
        "source_steps_per_credit": 100,
        "selection_file_sha256": selection_sha256,
        "prior_gpu_seconds": 100,
        "reused_paths": ["selection/manifest.json", "selection/selection-results.json"],
    }
    payload["provenance_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    provenance_path = tmp_path / "selection-provenance.json"
    write_json_atomic(provenance_path, payload)

    execution = _selection_execution(
        selection_path,
        git_commit="2" * 40,
        selection_file_sha256=selection_sha256,
        protocol_sha256="4" * 64,
        data_manifest_sha256="5" * 64,
        final_config_file_sha256="6" * 64,
        selected_steps_per_credit=100,
        environment_sha256="7" * 64,
        target_gpu_uuid="GPU-test",
    )

    assert execution["mode"] == "verified_cross_commit_reuse"
    payload["source_steps_per_credit"] = 250
    payload["provenance_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    write_json_atomic(provenance_path, payload, overwrite=True)
    with pytest.raises(ConsistencyError, match="does not authorize"):
        _selection_execution(
            selection_path,
            git_commit="2" * 40,
            selection_file_sha256=selection_sha256,
            protocol_sha256="4" * 64,
            data_manifest_sha256="5" * 64,
            final_config_file_sha256="6" * 64,
            selected_steps_per_credit=100,
            environment_sha256="7" * 64,
            target_gpu_uuid="GPU-test",
        )
    payload["source_steps_per_credit"] = 100
    payload["provenance_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    payload["prior_gpu_seconds"] = 101
    write_json_atomic(provenance_path, payload, overwrite=True)
    with pytest.raises(ConsistencyError, match="does not authorize"):
        _selection_execution(
            selection_path,
            git_commit="2" * 40,
            selection_file_sha256=selection_sha256,
            protocol_sha256="4" * 64,
            data_manifest_sha256="5" * 64,
            final_config_file_sha256="6" * 64,
            selected_steps_per_credit=100,
            environment_sha256="7" * 64,
            target_gpu_uuid="GPU-test",
        )


def test_final_runtime_must_exactly_match_clean_protocol_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    git = {
        "commit": "1" * 40,
        "dirty": False,
        "dirty_paths": [],
        "allow_dirty_override": False,
    }
    environment = {"python": "3.12.3", "cuda_available": True}
    data = {"manifest_sha256": "2" * 64, "verified_files": 10}
    final_sha256 = "3" * 64
    monkeypatch.setattr(
        "latesignal.experiments.protocol_lock._git_identity",
        lambda repository, *, allow_dirty: git,
    )
    monkeypatch.setattr(
        "latesignal.experiments.protocol_lock._environment", lambda repository: environment
    )
    monkeypatch.setattr(
        "latesignal.experiments.protocol_lock._verify_prepared_data", lambda path: data
    )
    monkeypatch.setattr(
        "latesignal.experiments.protocol_lock.sha256_file",
        lambda path: (final_sha256, 1),
    )
    lock = {
        "status": "locked",
        "publication_eligible": True,
        "git": git,
        "environment": environment,
        "environment_sha256": hashlib.sha256(canonical_json_bytes(environment)).hexdigest(),
        "data": data,
        "final_config_file_sha256": final_sha256,
    }

    verified = verify_locked_final_runtime(
        lock,
        final_config_path=tmp_path / "final.yaml",
        data_manifest_path=tmp_path / "prepared.json",
        repository=tmp_path,
    )
    assert verified["git"] == git

    changed = dict(lock)
    changed["environment"] = {"python": "3.12.4", "cuda_available": True}
    with pytest.raises(ConsistencyError, match="differs from the protocol lock"):
        verify_locked_final_runtime(
            changed,
            final_config_path=tmp_path / "final.yaml",
            data_manifest_path=tmp_path / "prepared.json",
            repository=tmp_path,
        )
