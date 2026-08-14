"""Immutable conversion-model snapshots for intermediate-budget evaluation."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, model_validator
from torch import Tensor

from latesignal.contracts.protocol import StrictModel
from latesignal.data.manifests import (
    canonical_json_bytes,
    read_json,
    sha256_file,
    write_json_atomic,
)
from latesignal.errors import ConsistencyError

_SNAPSHOT = re.compile(r"^fraction-(025|050|075|100)$")


class FinalSnapshotIdentity(StrictModel):
    version: Literal[1]
    run_id: str = Field(min_length=1)
    method: str = Field(min_length=1)
    seed: int
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    budget_fraction: float
    credits_at_snapshot: int = Field(gt=0)
    total_credits: int = Field(gt=0)

    @model_validator(mode="after")
    def exact_fraction_boundary(self) -> FinalSnapshotIdentity:
        if self.budget_fraction not in {0.25, 0.5, 0.75, 1.0}:
            raise ValueError("Final snapshot fraction is not authored")
        expected = (
            (int(self.budget_fraction * self.total_credits) + 1)
            if (
                self.budget_fraction * self.total_credits
                != int(self.budget_fraction * self.total_credits)
            )
            else int(self.budget_fraction * self.total_credits)
        )
        if self.credits_at_snapshot != expected:
            raise ValueError("Final snapshot does not use ceil(fraction * total credits)")
        return self

    @property
    def directory_name(self) -> str:
        return f"fraction-{round(self.budget_fraction * 100):03d}"


@dataclass(frozen=True, slots=True)
class VerifiedFinalSnapshot:
    root: Path
    identity: FinalSnapshotIdentity
    model_state: dict[str, Tensor]
    model_sha256: str
    state_sha256: str
    manifest_sha256: str
    model_version: int


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _model_sha256(state: dict[str, Tensor]) -> str:
    if not state:
        raise ConsistencyError("Final snapshot model state is empty")
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise ConsistencyError("Final snapshot contains a non-tensor model value")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(canonical_json_bytes(list(value.shape)))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class FinalSnapshotStore:
    """Write each budget snapshot once and verify it before later inference."""

    def __init__(self, root: Path) -> None:
        if root.is_symlink():
            raise ConsistencyError("Final snapshot root cannot be a symlink")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._validate_names()

    def _validate_names(self) -> None:
        unexpected = sorted(
            path.name
            for path in self.root.iterdir()
            if _SNAPSHOT.fullmatch(path.name) is None and not path.name.startswith(".fraction-")
        )
        if unexpected:
            raise ConsistencyError(
                "Final snapshot root contains an unexpected artifact",
                details={"paths": unexpected},
            )

    def _path(self, identity: FinalSnapshotIdentity) -> Path:
        path = self.root / identity.directory_name
        if _SNAPSHOT.fullmatch(path.name) is None or path.resolve().parent != self.root:
            raise ConsistencyError("Final snapshot identity resolves outside its store")
        return path

    def write(
        self,
        identity: FinalSnapshotIdentity,
        model_state: dict[str, Tensor],
        *,
        model_version: int,
    ) -> VerifiedFinalSnapshot:
        if model_version < 0:
            raise ConsistencyError("Final snapshot model version is invalid")
        target = self._path(identity)
        model_sha256 = _model_sha256(model_state)
        if target.exists():
            existing = self.verify(identity)
            if existing.model_sha256 != model_sha256 or existing.model_version != model_version:
                raise ConsistencyError("Retried final snapshot differs from immutable evidence")
            return existing
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=self.root))
        try:
            identity_path = temporary / "identity.json"
            write_json_atomic(identity_path, identity.model_dump(mode="json"))
            state_path = temporary / "model.pt"
            with state_path.open("xb") as output:
                torch.save(
                    {name: tensor.detach().cpu().clone() for name, tensor in model_state.items()},
                    output,
                )
                output.flush()
                os.fsync(output.fileno())
            identity_sha256, identity_bytes = sha256_file(identity_path)
            state_sha256, state_bytes = sha256_file(state_path)
            payload: dict[str, object] = {
                "version": 1,
                "status": "sealed",
                "identity_sha256": identity_sha256,
                "identity_bytes": identity_bytes,
                "state_sha256": state_sha256,
                "state_bytes": state_bytes,
                "model_sha256": model_sha256,
                "model_version": model_version,
            }
            payload["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
            write_json_atomic(temporary / "manifest.json", payload)
            _fsync_directory(temporary)
            self._verify_path(temporary, identity)
            os.replace(temporary, target)
            _fsync_directory(self.root)
            return self.verify(identity)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def _verify_path(
        self,
        path: Path,
        expected_identity: FinalSnapshotIdentity,
    ) -> VerifiedFinalSnapshot:
        if path.is_symlink() or not path.is_dir():
            raise ConsistencyError("Final snapshot is not a regular directory")
        if {item.name for item in path.iterdir()} != {
            "identity.json",
            "model.pt",
            "manifest.json",
        } or any(item.is_symlink() or not item.is_file() for item in path.iterdir()):
            raise ConsistencyError("Final snapshot has an incomplete or redirected file set")
        identity = FinalSnapshotIdentity.model_validate(read_json(path / "identity.json"))
        manifest = read_json(path / "manifest.json")
        expected_manifest = manifest.get("manifest_sha256")
        unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        identity_sha256, identity_bytes = sha256_file(path / "identity.json")
        state_sha256, state_bytes = sha256_file(path / "model.pt")
        model_version = manifest.get("model_version")
        if (
            identity != expected_identity
            or manifest.get("version") != 1
            or manifest.get("status") != "sealed"
            or not isinstance(expected_manifest, str)
            or hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != expected_manifest
            or manifest.get("identity_sha256") != identity_sha256
            or manifest.get("identity_bytes") != identity_bytes
            or manifest.get("state_sha256") != state_sha256
            or manifest.get("state_bytes") != state_bytes
            or isinstance(model_version, bool)
            or not isinstance(model_version, int)
            or model_version < 0
        ):
            raise ConsistencyError("Final snapshot manifest or identity is inconsistent")
        try:
            loaded = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        except Exception as error:
            raise ConsistencyError("Final snapshot model state could not be loaded") from error
        if not isinstance(loaded, dict) or not all(
            isinstance(name, str) and isinstance(value, Tensor) for name, value in loaded.items()
        ):
            raise ConsistencyError("Final snapshot model state is malformed")
        model_state = {str(name): value.cpu() for name, value in loaded.items()}
        model_sha256 = _model_sha256(model_state)
        if manifest.get("model_sha256") != model_sha256:
            raise ConsistencyError("Final snapshot tensors do not match their model digest")
        return VerifiedFinalSnapshot(
            root=path,
            identity=identity,
            model_state=model_state,
            model_sha256=model_sha256,
            state_sha256=state_sha256,
            manifest_sha256=expected_manifest,
            model_version=model_version,
        )

    def verify(self, identity: FinalSnapshotIdentity) -> VerifiedFinalSnapshot:
        return self._verify_path(self._path(identity), identity)

    def verify_exact(
        self,
        identities: tuple[FinalSnapshotIdentity, ...],
    ) -> tuple[VerifiedFinalSnapshot, ...]:
        expected = {identity.directory_name for identity in identities}
        actual = {path.name for path in self.root.iterdir() if _SNAPSHOT.fullmatch(path.name)}
        if actual != expected:
            raise ConsistencyError(
                "Final snapshot set is incomplete",
                details={
                    "missing": sorted(expected - actual),
                    "unexpected": sorted(actual - expected),
                },
            )
        return tuple(self.verify(identity) for identity in identities)
