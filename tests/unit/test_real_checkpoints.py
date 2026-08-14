from __future__ import annotations

from pathlib import Path

import pytest
import torch

from latesignal.errors import ConsistencyError
from latesignal.experiments.checkpoint import CheckpointIdentity, RollingCheckpointStore


def _identity(*, commit: str = "a" * 40) -> CheckpointIdentity:
    return CheckpointIdentity(
        version=1,
        phase="final",
        run_id="final-method-seed-17",
        config_sha256="1" * 64,
        protocol_sha256="2" * 64,
        protocol_lock_sha256="3" * 64,
        data_manifest_sha256="4" * 64,
        feature_policy_sha256="5" * 64,
        source_tree_sha256="6" * 64,
        dependency_lock_sha256="7" * 64,
        git_commit=commit,
        environment_sha256="8" * 64,
        device_uuid="GPU-test-uuid",
    )


def _state(value: int) -> dict[str, object]:
    return {
        "model": {"weight": torch.tensor([float(value)])},
        "optimizer": {"step": torch.tensor(value)},
        "rng": {"torch": torch.random.get_rng_state()},
        "cursors": {"day": value},
        "method": {"name": "fixed_wait"},
        "scheduler": {"credit": value},
        "sampler": {"cursor": torch.tensor([value])},
        "monitoring": {"rows": value},
        "ledgers": {"chunks": value},
        "compute": {"steps": value},
    }


def test_rolling_checkpoint_round_trip_and_bounded_retention(tmp_path: Path) -> None:
    store = RollingCheckpointStore(tmp_path / "checkpoints")
    identity = _identity()

    for value in (1, 2, 3):
        written = store.write(identity, _state(value))
        assert written.generation == value

    loaded = store.load_latest(identity)
    generations = sorted(path.name for path in store.root.glob("generation-*") if path.is_dir())
    assert loaded.generation == 3
    assert int(loaded.state["compute"]["steps"]) == 3
    assert generations == ["generation-000002", "generation-000003"]


def test_corrupt_current_falls_back_to_verified_previous(tmp_path: Path) -> None:
    store = RollingCheckpointStore(tmp_path / "checkpoints")
    identity = _identity()
    store.write(identity, _state(1))
    store.write(identity, _state(2))
    (store.root / "generation-000002" / "state.pt").write_bytes(b"truncated")

    loaded = store.load_latest(identity)

    assert loaded.generation == 1
    assert int(loaded.state["compute"]["steps"]) == 1


def test_checkpoint_refuses_identity_mismatch_and_double_corruption(tmp_path: Path) -> None:
    store = RollingCheckpointStore(tmp_path / "checkpoints")
    identity = _identity()
    store.write(identity, _state(1))
    store.write(identity, _state(2))

    with pytest.raises(ConsistencyError, match="No valid checkpoint"):
        store.load_latest(_identity(commit="b" * 40))

    for path in store.root.glob("generation-*/state.pt"):
        path.write_bytes(b"corrupt")
    with pytest.raises(ConsistencyError, match="No valid checkpoint"):
        store.load_latest(identity)
    with pytest.raises(ConsistencyError, match="refusing a silent restart"):
        store.write(identity, _state(3))


def test_checkpoint_requires_complete_state_and_final_lock(tmp_path: Path) -> None:
    store = RollingCheckpointStore(tmp_path / "checkpoints")
    state = _state(1)
    state.pop("sampler")
    with pytest.raises(ConsistencyError, match="complete recovery contract"):
        store.write(_identity(), state)

    invalid = _identity().model_copy(update={"protocol_lock_sha256": None})
    with pytest.raises(ConsistencyError, match="requires a protocol lock"):
        store.write(invalid, _state(1))
