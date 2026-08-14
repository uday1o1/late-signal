from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latesignal.errors import ConsistencyError
from latesignal.experiments.exposures import ExposureLedgerIdentity, ExposureLedgerWriter
from latesignal.training.production import ExposureCredit


def _identity() -> ExposureLedgerIdentity:
    return ExposureLedgerIdentity(
        version=1,
        phase="final",
        run_id="fixed-wait-seed-17",
        method="fixed_wait",
        seed=17,
        config_sha256="1" * 64,
        protocol_sha256="2" * 64,
        protocol_lock_sha256="3" * 64,
        expected_credits=2,
        steps_per_credit=2,
        batch_size=3,
    )


def _credit(credit_id: int, *, offset: int = 0) -> ExposureCredit:
    size = 6
    return ExposureCredit(
        credit_id=credit_id,
        record_keys=np.arange(offset, offset + size, dtype=np.uint64),
        sources=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.uint8),
        weights=np.linspace(0.5, 1.0, size, dtype=np.float32),
    )


def test_exposure_ledger_replays_identical_credit_then_seals(tmp_path: Path) -> None:
    root = tmp_path / "exposures"
    writer = ExposureLedgerWriter(root, _identity())
    first_path = writer.append_credit(_credit(0))
    assert writer.append_credit(_credit(0)) == first_path
    resumed = ExposureLedgerWriter(root, _identity())
    resumed.append_credit(_credit(1, offset=10))

    seal = resumed.seal()
    verified = ExposureLedgerWriter(root, _identity()).verify_seal()

    assert seal == verified
    assert seal.credits == 2
    assert seal.examples == 12
    with pytest.raises(ConsistencyError, match="immutable"):
        resumed.append_credit(_credit(1, offset=10))


def test_exposure_ledger_rejects_changed_retry_and_partial_seal(tmp_path: Path) -> None:
    writer = ExposureLedgerWriter(tmp_path / "exposures", _identity())
    writer.append_credit(_credit(0))

    with pytest.raises(ConsistencyError, match="differs from durable evidence"):
        writer.append_credit(_credit(0, offset=1))
    with pytest.raises(ConsistencyError, match="complete optimizer budget"):
        writer.seal()


def test_exposure_ledger_detects_part_tampering(tmp_path: Path) -> None:
    writer = ExposureLedgerWriter(tmp_path / "exposures", _identity())
    writer.append_credit(_credit(0))
    writer.append_credit(_credit(1, offset=10))
    writer.seal()
    (writer.root / "credit-001.parquet").write_bytes(b"corrupt")

    with pytest.raises(ConsistencyError, match="could not be verified"):
        ExposureLedgerWriter(writer.root, _identity()).verify_seal()
