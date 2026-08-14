from __future__ import annotations

from pathlib import Path

import pytest
import torch

from latesignal.errors import ConsistencyError
from latesignal.experiments.final_snapshots import (
    FinalSnapshotIdentity,
    FinalSnapshotStore,
)


def _identity(fraction: float, credits: int) -> FinalSnapshotIdentity:
    return FinalSnapshotIdentity(
        version=1,
        run_id="final-test",
        method="complete_wait",
        seed=17,
        config_sha256="1" * 64,
        protocol_sha256="2" * 64,
        protocol_lock_sha256="3" * 64,
        budget_fraction=fraction,
        credits_at_snapshot=credits,
        total_credits=59,
    )


def test_final_snapshot_store_seals_exact_ceil_boundaries(tmp_path: Path) -> None:
    store = FinalSnapshotStore(tmp_path / "snapshots")
    identities = (
        _identity(0.25, 15),
        _identity(0.5, 30),
        _identity(0.75, 45),
        _identity(1.0, 59),
    )
    states = [{"weight": torch.tensor([float(index), 2.0])} for index in range(len(identities))]
    written = [
        store.write(identity, state, model_version=index)
        for index, (identity, state) in enumerate(zip(identities, states, strict=True))
    ]

    verified = store.verify_exact(identities)

    assert [item.model_sha256 for item in verified] == [item.model_sha256 for item in written]
    assert [item.model_version for item in verified] == [0, 1, 2, 3]
    retried = store.write(identities[0], states[0], model_version=0)
    assert retried.manifest_sha256 == written[0].manifest_sha256


def test_final_snapshot_store_rejects_changed_retry_and_corruption(tmp_path: Path) -> None:
    store = FinalSnapshotStore(tmp_path / "snapshots")
    identity = _identity(0.25, 15)
    store.write(identity, {"weight": torch.tensor([1.0])}, model_version=15)

    with pytest.raises(ConsistencyError, match="differs"):
        store.write(identity, {"weight": torch.tensor([2.0])}, model_version=15)

    state_path = tmp_path / "snapshots" / "fraction-025" / "model.pt"
    state_path.write_bytes(b"corrupt")
    with pytest.raises(ConsistencyError, match="manifest or identity"):
        store.verify(identity)


def test_final_snapshot_identity_rejects_floor_instead_of_ceil() -> None:
    with pytest.raises(ValueError, match="ceil"):
        _identity(0.25, 14)
