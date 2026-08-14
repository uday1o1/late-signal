from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from latesignal.data.manifests import canonical_json_bytes
from latesignal.errors import ConsistencyError
from latesignal.experiments.protocol_lock import verify_locked_final_runtime


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
