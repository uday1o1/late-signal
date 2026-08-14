"""Truth-free append-only prediction ledgers with independent sealing."""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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

_PART = re.compile(r"^part-(\d{6})\.parquet$")
_TEMPORARY_PART = re.compile(r"^\.part-(\d{6})\.parquet\.tmp$")
_HEX_ID = re.compile(r"^[0-9a-f]{64}$")
_SCHEMA = pa.schema(
    [
        pa.field("click_id", pa.string(), nullable=False),
        pa.field("click_day", pa.int16(), nullable=False),
        pa.field("probability", pa.float32(), nullable=False),
        pa.field("model_version", pa.int32(), nullable=False),
    ],
    metadata={b"latesignal_store": b"sealed_predictions_without_truth"},
)


class PredictionLedgerIdentity(StrictModel):
    version: Literal[1]
    kind: Literal["selection", "final_prequential", "intermediate"]
    run_id: str = Field(min_length=1)
    method: str = Field(min_length=1)
    seed: int
    period_first_day: int = Field(ge=0)
    period_last_day: int = Field(ge=0)
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_lock_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_rows: int = Field(gt=0)
    expected_ordered_id_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ranking_eligible: bool
    budget_fraction: float | None = None
    credits_at_snapshot: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def evidence_boundary(self) -> PredictionLedgerIdentity:
        if self.period_last_day < self.period_first_day:
            raise ValueError("Prediction period is reversed")
        if self.kind == "selection":
            if (
                self.protocol_lock_sha256 is not None
                or (self.period_first_day, self.period_last_day) != (25, 34)
                or self.budget_fraction is not None
                or self.credits_at_snapshot is not None
            ):
                raise ValueError("Selection prediction identity crosses its evidence boundary")
            return self
        if self.protocol_lock_sha256 is None or (self.period_first_day, self.period_last_day) != (
            65,
            89,
        ):
            raise ValueError("Final predictions require a lock and the final click period")
        if self.kind == "intermediate":
            if self.budget_fraction not in {0.25, 0.5, 0.75, 1.0}:
                raise ValueError("Intermediate predictions require a locked budget fraction")
            if self.credits_at_snapshot is None:
                raise ValueError("Intermediate predictions require a snapshot credit")
        elif self.budget_fraction is not None or self.credits_at_snapshot is not None:
            raise ValueError("Primary prequential predictions cannot claim a budget snapshot")
        return self


@dataclass(frozen=True, slots=True)
class PredictionSeal:
    root: Path
    identity: PredictionLedgerIdentity
    seal_sha256: str
    ledger_sha256: str
    rows: int
    ordered_id_sha256: str


def ordered_click_id_sha256(click_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for click_id in click_ids:
        if _HEX_ID.fullmatch(click_id) is None:
            raise ConsistencyError("Prediction click ID is not a canonical digest")
        digest.update(bytes.fromhex(click_id))
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class PredictionLedgerWriter:
    def __init__(self, root: Path, identity: PredictionLedgerIdentity) -> None:
        if root.is_symlink():
            raise ConsistencyError("Prediction-ledger root cannot be a symlink")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        identity_path = self.root / "identity.json"
        if identity_path.exists():
            stored = PredictionLedgerIdentity.model_validate(read_json(identity_path))
            if stored != identity:
                raise ConsistencyError("Prediction ledger identity changed across resume")
        else:
            write_json_atomic(identity_path, identity.model_dump(mode="json"))
        self._remove_temporary_parts()

    def _remove_temporary_parts(self) -> None:
        for path in self.root.iterdir():
            if _TEMPORARY_PART.fullmatch(path.name) is None:
                continue
            if path.is_symlink() or path.resolve().parent != self.root:
                raise ConsistencyError("Refusing redirected prediction temporary cleanup")
            path.unlink()
        _fsync_directory(self.root)

    def _parts(self) -> list[Path]:
        parts: list[tuple[int, Path]] = []
        for path in self.root.iterdir():
            match = _PART.fullmatch(path.name)
            if match is not None:
                if path.is_symlink() or not path.is_file():
                    raise ConsistencyError("Prediction part is not a regular file")
                parts.append((int(match.group(1)), path))
        parts.sort()
        if [index for index, _ in parts] != list(range(len(parts))):
            raise ConsistencyError("Prediction part sequence is not contiguous")
        return [path for _, path in parts]

    def append(
        self,
        *,
        click_ids: Sequence[str],
        click_days: Sequence[int],
        probabilities: Sequence[float],
        model_versions: Sequence[int],
    ) -> Path:
        if (self.root / "seal.json").exists():
            raise ConsistencyError("Sealed prediction ledger is immutable")
        lengths = {len(click_ids), len(click_days), len(probabilities), len(model_versions)}
        if lengths == {0} or len(lengths) != 1:
            raise ConsistencyError("Prediction columns must be nonempty and aligned")
        for click_id in click_ids:
            if _HEX_ID.fullmatch(click_id) is None:
                raise ConsistencyError("Prediction click ID is not a canonical digest")
        if any(
            isinstance(day, bool)
            or not isinstance(day, int)
            or not self.identity.period_first_day <= day <= self.identity.period_last_day
            for day in click_days
        ):
            raise ConsistencyError("Prediction click day lies outside the locked period")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in probabilities
        ):
            raise ConsistencyError("Prediction probability is invalid")
        if any(
            isinstance(version, bool) or not isinstance(version, int) or version < 0
            for version in model_versions
        ):
            raise ConsistencyError("Prediction model version is invalid")
        table = pa.Table.from_arrays(
            [
                pa.array(click_ids, type=pa.string()),
                pa.array(click_days, type=pa.int16()),
                pa.array(probabilities, type=pa.float32()),
                pa.array(model_versions, type=pa.int32()),
            ],
            schema=_SCHEMA,
        )
        index = len(self._parts())
        target = self.root / f"part-{index:06d}.parquet"
        temporary = self.root / f".{target.name}.tmp"
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
            metadata = pq.read_metadata(temporary)
            if metadata.num_rows != len(click_ids) or metadata.num_columns != len(_SCHEMA):
                raise ConsistencyError("Written prediction part did not pass metadata verification")
            os.replace(temporary, target)
            _fsync_directory(self.root)
            return target
        finally:
            temporary.unlink(missing_ok=True)

    def _scan(self) -> dict[str, object]:
        parts = self._parts()
        if not parts:
            raise ConsistencyError("Prediction ledger has no parts")
        ordered_ids = hashlib.sha256()
        seen_ids: set[bytes] = set()
        rows = 0
        files: list[dict[str, object]] = []
        for part in parts:
            try:
                parquet = pq.ParquetFile(part)
                if parquet.schema_arrow != _SCHEMA:
                    raise ConsistencyError("Prediction part violates the truth-free schema")
                file_rows = 0
                for batch in parquet.iter_batches(batch_size=65_536):
                    ids = batch.column(0).to_pylist()
                    days = batch.column(1).to_numpy(zero_copy_only=False)
                    probabilities = batch.column(2).to_numpy(zero_copy_only=False)
                    versions = batch.column(3).to_numpy(zero_copy_only=False)
                    if (
                        np.any(days < self.identity.period_first_day)
                        or np.any(days > self.identity.period_last_day)
                        or not np.isfinite(probabilities).all()
                        or np.any(probabilities < 0.0)
                        or np.any(probabilities > 1.0)
                        or np.any(versions < 0)
                    ):
                        raise ConsistencyError("Prediction part contains an invalid value")
                    for value in ids:
                        if not isinstance(value, str) or _HEX_ID.fullmatch(value) is None:
                            raise ConsistencyError("Prediction part contains an invalid click ID")
                        raw = bytes.fromhex(value)
                        if raw in seen_ids:
                            raise ConsistencyError(
                                "Prediction ledger contains a duplicate click ID"
                            )
                        seen_ids.add(raw)
                        ordered_ids.update(raw)
                    file_rows += batch.num_rows
                sha256, size = sha256_file(part)
            except (OSError, pa.ArrowException) as error:
                raise ConsistencyError("Prediction part could not be verified") from error
            files.append(
                {
                    "path": part.name,
                    "sha256": sha256,
                    "bytes": size,
                    "rows": file_rows,
                }
            )
            rows += file_rows
        return {
            "rows": rows,
            "ordered_id_sha256": ordered_ids.hexdigest(),
            "files": files,
        }

    def seal(self) -> PredictionSeal:
        seal_path = self.root / "seal.json"
        if seal_path.exists():
            raise ConsistencyError("Prediction ledger is already sealed")
        expected_names = {"identity.json", *(path.name for path in self._parts())}
        actual_names = {path.name for path in self.root.iterdir()}
        if actual_names != expected_names:
            raise ConsistencyError("Prediction ledger contains an unexpected pre-seal artifact")
        scan = self._scan()
        if (
            scan["rows"] != self.identity.expected_rows
            or scan["ordered_id_sha256"] != self.identity.expected_ordered_id_sha256
        ):
            raise ConsistencyError("Prediction ledger does not match the locked evaluation cohort")
        identity_sha256, identity_bytes = sha256_file(self.root / "identity.json")
        ledger_payload = {
            "identity_sha256": identity_sha256,
            "identity_bytes": identity_bytes,
            **scan,
        }
        ledger_sha256 = hashlib.sha256(canonical_json_bytes(ledger_payload)).hexdigest()
        payload: dict[str, object] = {
            "version": 1,
            "status": "sealed",
            "truth_joined": False,
            **ledger_payload,
            "ledger_sha256": ledger_sha256,
        }
        payload["seal_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        write_json_atomic(seal_path, payload)
        _fsync_directory(self.root)
        return PredictionSeal(
            root=self.root,
            identity=self.identity,
            seal_sha256=str(payload["seal_sha256"]),
            ledger_sha256=ledger_sha256,
            rows=int(scan["rows"]),
            ordered_id_sha256=str(scan["ordered_id_sha256"]),
        )

    def verify_seal(self) -> PredictionSeal:
        seal = read_json(self.root / "seal.json")
        expected_seal_sha256 = seal.get("seal_sha256")
        if not isinstance(expected_seal_sha256, str):
            raise ConsistencyError("Prediction seal has no digest")
        unsigned = {key: value for key, value in seal.items() if key != "seal_sha256"}
        if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != expected_seal_sha256:
            raise ConsistencyError("Prediction seal content does not match its digest")
        if seal.get("status") != "sealed" or seal.get("truth_joined") is not False:
            raise ConsistencyError("Prediction seal is not independent of truth")
        expected_names = {"identity.json", "seal.json", *(path.name for path in self._parts())}
        if {path.name for path in self.root.iterdir()} != expected_names:
            raise ConsistencyError("Sealed prediction ledger contains an unexpected artifact")
        scan = self._scan()
        identity_sha256, identity_bytes = sha256_file(self.root / "identity.json")
        ledger_payload = {
            "identity_sha256": identity_sha256,
            "identity_bytes": identity_bytes,
            **scan,
        }
        ledger_sha256 = hashlib.sha256(canonical_json_bytes(ledger_payload)).hexdigest()
        expected_fields = {
            **ledger_payload,
            "ledger_sha256": ledger_sha256,
        }
        if any(seal.get(key) != value for key, value in expected_fields.items()):
            raise ConsistencyError("Prediction seal does not match its ledger files")
        if (
            scan["rows"] != self.identity.expected_rows
            or scan["ordered_id_sha256"] != self.identity.expected_ordered_id_sha256
        ):
            raise ConsistencyError("Prediction seal does not match the locked evaluation cohort")
        return PredictionSeal(
            root=self.root,
            identity=self.identity,
            seal_sha256=expected_seal_sha256,
            ledger_sha256=ledger_sha256,
            rows=int(scan["rows"]),
            ordered_id_sha256=str(scan["ordered_id_sha256"]),
        )
