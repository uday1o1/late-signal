"""Atomic chunked optimizer-exposure ledgers for production runs."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import Field, model_validator

from latesignal.contracts.protocol import StrictModel
from latesignal.data.manifests import (
    canonical_json_bytes,
    read_json,
    sha256_file,
    write_json_atomic,
)
from latesignal.errors import ConsistencyError
from latesignal.training.production import ExposureCredit

_PART = re.compile(r"^credit-(\d{3})\.parquet$")
_TEMPORARY = re.compile(r"^\.credit-(\d{3})\.parquet\.[0-9a-f]{32}\.tmp$")
_SCHEMA_METADATA_KEY = b"latesignal_store"
_SCHEMA_METADATA_VALUE = b"packed_optimizer_exposures"


class ExposureLedgerIdentity(StrictModel):
    version: Literal[1]
    phase: Literal["qualification", "selection", "final"]
    run_id: str = Field(min_length=1)
    method: str = Field(min_length=1)
    seed: int
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_lock_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    expected_credits: int = Field(gt=0, le=999)
    steps_per_credit: int = Field(gt=0, le=65_535)
    batch_size: int = Field(gt=0)

    @model_validator(mode="after")
    def phase_lock(self) -> ExposureLedgerIdentity:
        if self.phase == "final" and self.protocol_lock_sha256 is None:
            raise ValueError("Final exposure ledger requires a protocol lock")
        if self.phase != "final" and self.protocol_lock_sha256 is not None:
            raise ValueError("Pre-final exposure ledger cannot claim a protocol lock")
        return self


@dataclass(frozen=True, slots=True)
class ExposureLedgerPosition:
    credits: int
    examples: int
    entries_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "credits": self.credits,
            "examples": self.examples,
            "entries_sha256": self.entries_sha256,
        }


@dataclass(frozen=True, slots=True)
class ExposureSeal:
    seal_sha256: str
    ledger_sha256: str
    credits: int
    examples: int


def _schema(identity_sha256: str) -> pa.Schema:
    return pa.schema(
        [
            pa.field("credit_id", pa.uint16(), nullable=False),
            pa.field("step", pa.uint16(), nullable=False),
            pa.field("record_key", pa.uint64(), nullable=False),
            pa.field("source", pa.uint8(), nullable=False),
            pa.field("weight", pa.float32(), nullable=False),
        ],
        metadata={
            _SCHEMA_METADATA_KEY: _SCHEMA_METADATA_VALUE,
            b"identity_sha256": identity_sha256.encode(),
        },
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _table(exposure: ExposureCredit, identity: ExposureLedgerIdentity) -> pa.Table:
    expected = identity.steps_per_credit * identity.batch_size
    if (
        exposure.examples != expected
        or exposure.record_keys.shape != (expected,)
        or exposure.sources.shape != (expected,)
        or exposure.weights.shape != (expected,)
        or np.any(exposure.sources > 1)
        or not np.isfinite(exposure.weights).all()
        or np.any(exposure.weights < 0.0)
    ):
        raise ConsistencyError("Exposure credit does not match its locked optimizer budget")
    identity_sha256 = hashlib.sha256(
        canonical_json_bytes(identity.model_dump(mode="json"))
    ).hexdigest()
    return pa.Table.from_arrays(
        [
            pa.array(np.full(expected, exposure.credit_id, dtype=np.uint16)),
            pa.array(
                np.repeat(
                    np.arange(identity.steps_per_credit, dtype=np.uint16),
                    identity.batch_size,
                )
            ),
            pa.array(exposure.record_keys, type=pa.uint64()),
            pa.array(exposure.sources, type=pa.uint8()),
            pa.array(exposure.weights, type=pa.float32()),
        ],
        schema=_schema(identity_sha256),
    )


class ExposureLedgerWriter:
    """Write one retry-safe exposure part per completed optimizer credit."""

    def __init__(self, root: Path, identity: ExposureLedgerIdentity) -> None:
        if root.is_symlink():
            raise ConsistencyError("Exposure-ledger root cannot be a symlink")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.identity_sha256 = hashlib.sha256(
            canonical_json_bytes(identity.model_dump(mode="json"))
        ).hexdigest()
        identity_path = self.root / "identity.json"
        if identity_path.exists():
            stored = ExposureLedgerIdentity.model_validate(read_json(identity_path))
            if stored != identity:
                raise ConsistencyError("Exposure-ledger identity changed across resume")
        else:
            write_json_atomic(identity_path, identity.model_dump(mode="json"))
        self._remove_temporary_parts()
        self._validate_names()
        self._entries = self._scan_parts()

    def _validate_names(self) -> None:
        allowed = {"identity.json", "seal.json"}
        unexpected = sorted(
            path.name
            for path in self.root.iterdir()
            if path.name not in allowed and _PART.fullmatch(path.name) is None
        )
        if unexpected:
            raise ConsistencyError(
                "Exposure ledger contains an unexpected artifact",
                details={"paths": unexpected},
            )

    def _remove_temporary_parts(self) -> None:
        for path in self.root.iterdir():
            if _TEMPORARY.fullmatch(path.name) is None:
                continue
            if path.is_symlink() or path.resolve().parent != self.root:
                raise ConsistencyError("Refusing redirected exposure temporary cleanup")
            path.unlink()
        _fsync_directory(self.root)

    def _part_paths(self) -> list[Path]:
        parts: list[tuple[int, Path]] = []
        for path in self.root.iterdir():
            match = _PART.fullmatch(path.name)
            if match is not None:
                if path.is_symlink() or not path.is_file():
                    raise ConsistencyError("Exposure part is not a regular file")
                parts.append((int(match.group(1)), path))
        parts.sort()
        if [index for index, _ in parts] != list(range(len(parts))):
            raise ConsistencyError("Exposure credit parts are not contiguous")
        return [path for _, path in parts]

    def _entry(self, path: Path, credit_id: int) -> dict[str, object]:
        try:
            parquet = pq.ParquetFile(path)
            expected_schema = _schema(self.identity_sha256)
            if parquet.schema_arrow != expected_schema:
                raise ConsistencyError("Exposure part schema or identity is invalid")
            expected_rows = self.identity.steps_per_credit * self.identity.batch_size
            if parquet.metadata.num_rows != expected_rows:
                raise ConsistencyError("Exposure part has an invalid row count")
            step_counts = np.zeros(self.identity.steps_per_credit, dtype=np.int64)
            for batch in parquet.iter_batches(batch_size=65_536):
                credits = batch.column(0).to_numpy(zero_copy_only=False)
                steps = batch.column(1).to_numpy(zero_copy_only=False)
                sources = batch.column(3).to_numpy(zero_copy_only=False)
                weights = batch.column(4).to_numpy(zero_copy_only=False)
                if (
                    np.any(credits != credit_id)
                    or np.any(steps >= self.identity.steps_per_credit)
                    or np.any(sources > 1)
                    or not np.isfinite(weights).all()
                    or np.any(weights < 0.0)
                ):
                    raise ConsistencyError("Exposure part contains an invalid value")
                step_counts += np.bincount(
                    steps,
                    minlength=self.identity.steps_per_credit,
                )
            if np.any(step_counts != self.identity.batch_size):
                raise ConsistencyError("Exposure part does not preserve per-step batch size")
            sha256, size = sha256_file(path)
        except (OSError, pa.ArrowException) as error:
            raise ConsistencyError("Exposure part could not be verified") from error
        return {
            "path": path.name,
            "sha256": sha256,
            "bytes": size,
            "credit_id": credit_id,
            "rows": expected_rows,
        }

    def _scan_parts(self) -> list[dict[str, object]]:
        return [self._entry(path, credit_id) for credit_id, path in enumerate(self._part_paths())]

    def append_credit(self, exposure: ExposureCredit) -> Path:
        if (self.root / "seal.json").exists():
            raise ConsistencyError("Sealed exposure ledger is immutable")
        if not 0 <= exposure.credit_id < self.identity.expected_credits:
            raise ConsistencyError("Exposure credit lies outside the locked run budget")
        table = _table(exposure, self.identity)
        target = self.root / f"credit-{exposure.credit_id:03d}.parquet"
        if exposure.credit_id < len(self._entries):
            self._entry(target, exposure.credit_id)
            try:
                existing = pq.ParquetFile(target).read()
            except (OSError, pa.ArrowException) as error:
                raise ConsistencyError("Exposure part could not be verified") from error
            if not existing.equals(table, check_metadata=True):
                raise ConsistencyError("Retried exposure credit differs from durable evidence")
            return target
        if exposure.credit_id != len(self._entries):
            raise ConsistencyError("Exposure credits must be appended contiguously")
        temporary = self.root / f".{target.name}.{uuid.uuid4().hex}.tmp"
        try:
            pq.write_table(
                table,
                temporary,
                compression="zstd",
                use_dictionary=False,
                write_statistics=True,
            )
            with temporary.open("rb") as source:
                os.fsync(source.fileno())
            candidate = self._entry(temporary, exposure.credit_id)
            os.replace(temporary, target)
            _fsync_directory(self.root)
            candidate["path"] = target.name
            self._entries.append(candidate)
            return target
        finally:
            temporary.unlink(missing_ok=True)

    def position(self) -> ExposureLedgerPosition:
        examples = len(self._entries) * self.identity.steps_per_credit * self.identity.batch_size
        return ExposureLedgerPosition(
            credits=len(self._entries),
            examples=examples,
            entries_sha256=hashlib.sha256(canonical_json_bytes(self._entries)).hexdigest(),
        )

    def seal(self) -> ExposureSeal:
        seal_path = self.root / "seal.json"
        if seal_path.exists():
            raise ConsistencyError("Exposure ledger is already sealed")
        self._entries = self._scan_parts()
        expected_names = {"identity.json", *(path.name for path in self._part_paths())}
        if {path.name for path in self.root.iterdir()} != expected_names:
            raise ConsistencyError("Exposure ledger contains an unexpected pre-seal artifact")
        position = self.position()
        expected_examples = (
            self.identity.expected_credits
            * self.identity.steps_per_credit
            * self.identity.batch_size
        )
        if (
            position.credits != self.identity.expected_credits
            or position.examples != expected_examples
        ):
            raise ConsistencyError("Exposure ledger does not contain the complete optimizer budget")
        identity_sha256, identity_bytes = sha256_file(self.root / "identity.json")
        payload: dict[str, Any] = {
            "version": 1,
            "status": "sealed",
            "identity_sha256": identity_sha256,
            "identity_bytes": identity_bytes,
            "entries": self._entries,
            **position.as_dict(),
        }
        payload["ledger_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        payload["seal_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        write_json_atomic(seal_path, payload)
        _fsync_directory(self.root)
        return ExposureSeal(
            seal_sha256=str(payload["seal_sha256"]),
            ledger_sha256=str(payload["ledger_sha256"]),
            credits=position.credits,
            examples=position.examples,
        )

    def verify_seal(self) -> ExposureSeal:
        seal = read_json(self.root / "seal.json")
        expected_seal = seal.get("seal_sha256")
        unsigned = {key: value for key, value in seal.items() if key != "seal_sha256"}
        if (
            not isinstance(expected_seal, str)
            or hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != expected_seal
            or seal.get("status") != "sealed"
        ):
            raise ConsistencyError("Exposure seal content is invalid")
        scanned = self._scan_parts()
        if scanned != seal.get("entries"):
            raise ConsistencyError("Exposure seal does not match its ledger files")
        identity_sha256, identity_bytes = sha256_file(self.root / "identity.json")
        if (
            seal.get("identity_sha256") != identity_sha256
            or seal.get("identity_bytes") != identity_bytes
        ):
            raise ConsistencyError("Exposure seal does not match its identity file")
        expected_names = {"identity.json", "seal.json", *(path.name for path in self._part_paths())}
        if {path.name for path in self.root.iterdir()} != expected_names:
            raise ConsistencyError("Sealed exposure ledger contains an unexpected artifact")
        position = self.position()
        if any(seal.get(key) != value for key, value in position.as_dict().items()):
            raise ConsistencyError("Exposure seal position is inconsistent")
        ledger_payload = {
            key: value for key, value in seal.items() if key not in {"ledger_sha256", "seal_sha256"}
        }
        ledger_sha256 = hashlib.sha256(canonical_json_bytes(ledger_payload)).hexdigest()
        if seal.get("ledger_sha256") != ledger_sha256:
            raise ConsistencyError("Exposure ledger digest is invalid")
        return ExposureSeal(
            seal_sha256=expected_seal,
            ledger_sha256=ledger_sha256,
            credits=position.credits,
            examples=position.examples,
        )
