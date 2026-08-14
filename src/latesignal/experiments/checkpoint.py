"""Atomic rolling binary checkpoints for real-data experiment state."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import Field

from latesignal.contracts.protocol import StrictModel
from latesignal.data.manifests import (
    canonical_json_bytes,
    read_json,
    sha256_file,
    write_json_atomic,
)
from latesignal.errors import ConsistencyError

_GENERATION = re.compile(r"^generation-(\d{6})$")
_REQUIRED_STATE_KEYS = {
    "model",
    "optimizer",
    "rng",
    "cursors",
    "method",
    "scheduler",
    "sampler",
    "monitoring",
    "ledgers",
    "compute",
}


class CheckpointIdentity(StrictModel):
    version: Literal[1]
    phase: Literal["qualification", "selection", "final"]
    run_id: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_lock_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    data_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependency_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    device_uuid: str = Field(min_length=1)

    def validate_phase(self) -> None:
        if self.phase == "final" and self.protocol_lock_sha256 is None:
            raise ConsistencyError("Final checkpoint identity requires a protocol lock")
        if self.phase != "final" and self.protocol_lock_sha256 is not None:
            raise ConsistencyError("Pre-final checkpoint cannot claim a final protocol lock")


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    generation: int
    identity: CheckpointIdentity
    state: dict[str, Any]
    manifest_sha256: str


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes(path: Path, value: bytes) -> None:
    with path.open("xb") as output:
        output.write(value)
        output.flush()
        os.fsync(output.fileno())


def _state_contract(state: dict[str, Any]) -> None:
    if set(state) != _REQUIRED_STATE_KEYS:
        raise ConsistencyError(
            "Checkpoint state does not contain the complete recovery contract",
            details={
                "missing": sorted(_REQUIRED_STATE_KEYS - set(state)),
                "unknown": sorted(set(state) - _REQUIRED_STATE_KEYS),
            },
        )
    if not all(isinstance(state[key], dict) for key in _REQUIRED_STATE_KEYS):
        raise ConsistencyError("Checkpoint recovery sections must be mappings")


class RollingCheckpointStore:
    """Retain two verified generations while using a third temporary write."""

    def __init__(self, root: Path) -> None:
        if root.is_symlink():
            raise ConsistencyError("Checkpoint root cannot be a symlink")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _generation_paths(self) -> list[tuple[int, Path]]:
        result: list[tuple[int, Path]] = []
        for path in self.root.iterdir():
            match = _GENERATION.fullmatch(path.name)
            if match is not None:
                if path.is_symlink() or not path.is_dir():
                    raise ConsistencyError("Checkpoint generation is not a regular directory")
                result.append((int(match.group(1)), path))
            elif path.name != "latest.json" and not path.name.startswith(".generation-"):
                raise ConsistencyError(f"Checkpoint root contains an unexpected entry: {path}")
        return sorted(result)

    def _verify_generation(
        self,
        generation: int,
        path: Path,
        expected_identity: CheckpointIdentity,
    ) -> LoadedCheckpoint:
        expected_identity.validate_phase()
        actual_names = {item.name for item in path.iterdir()}
        if actual_names != {"identity.json", "state.pt", "manifest.json"}:
            raise ConsistencyError("Checkpoint generation has an incomplete file set")
        if any(item.is_symlink() or not item.is_file() for item in path.iterdir()):
            raise ConsistencyError("Checkpoint generation contains a redirected input")
        manifest = read_json(path / "manifest.json")
        expected_manifest_sha256 = manifest.get("manifest_sha256")
        if not isinstance(expected_manifest_sha256, str):
            raise ConsistencyError("Checkpoint manifest has no digest")
        unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        actual_manifest_sha256 = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
        if actual_manifest_sha256 != expected_manifest_sha256:
            raise ConsistencyError("Checkpoint manifest content does not match its digest")
        if manifest.get("version") != 1 or manifest.get("generation") != generation:
            raise ConsistencyError("Checkpoint manifest generation is invalid")
        identity_sha256, identity_bytes = sha256_file(path / "identity.json")
        state_sha256, state_bytes = sha256_file(path / "state.pt")
        if (
            manifest.get("identity_sha256") != identity_sha256
            or manifest.get("identity_bytes") != identity_bytes
            or manifest.get("state_sha256") != state_sha256
            or manifest.get("state_bytes") != state_bytes
        ):
            raise ConsistencyError("Checkpoint file identity does not match its manifest")
        identity = CheckpointIdentity.model_validate(read_json(path / "identity.json"))
        identity.validate_phase()
        if identity != expected_identity:
            raise ConsistencyError("Checkpoint runtime identity does not match the current run")
        try:
            loaded = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
        except Exception as error:
            raise ConsistencyError("Checkpoint tensor state could not be loaded") from error
        if not isinstance(loaded, dict):
            raise ConsistencyError("Checkpoint tensor state is not a mapping")
        state = {str(key): value for key, value in loaded.items()}
        _state_contract(state)
        return LoadedCheckpoint(generation, identity, state, expected_manifest_sha256)

    def _verified_generations(
        self,
        expected_identity: CheckpointIdentity,
    ) -> list[LoadedCheckpoint]:
        verified: list[LoadedCheckpoint] = []
        for generation, path in reversed(self._generation_paths()):
            try:
                verified.append(self._verify_generation(generation, path, expected_identity))
            except ConsistencyError:
                continue
        return verified

    def write(
        self,
        identity: CheckpointIdentity,
        state: dict[str, Any],
    ) -> LoadedCheckpoint:
        identity.validate_phase()
        _state_contract(state)
        generations = self._generation_paths()
        generation = (generations[-1][0] + 1) if generations else 1
        previous = self._verified_generations(identity)
        if generations and not previous:
            raise ConsistencyError(
                "Existing checkpoint generations are invalid; refusing a silent restart"
            )
        previous_generation = previous[0].generation if previous else None
        temporary = Path(tempfile.mkdtemp(prefix=".generation-", dir=self.root))
        target = self.root / f"generation-{generation:06d}"
        try:
            _write_bytes(
                temporary / "identity.json", canonical_json_bytes(identity.model_dump(mode="json"))
            )
            state_path = temporary / "state.pt"
            with state_path.open("xb") as output:
                torch.save(state, output)
                output.flush()
                os.fsync(output.fileno())
            identity_sha256, identity_bytes = sha256_file(temporary / "identity.json")
            state_sha256, state_bytes = sha256_file(state_path)
            payload: dict[str, Any] = {
                "version": 1,
                "generation": generation,
                "identity_sha256": identity_sha256,
                "identity_bytes": identity_bytes,
                "state_sha256": state_sha256,
                "state_bytes": state_bytes,
            }
            payload["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
            _write_bytes(temporary / "manifest.json", canonical_json_bytes(payload))
            _fsync_directory(temporary)
            candidate = self._verify_generation(generation, temporary, identity)
            os.replace(temporary, target)
            _fsync_directory(self.root)
            write_json_atomic(
                self.root / "latest.json",
                {
                    "version": 1,
                    "current": target.name,
                    "previous": (
                        None
                        if previous_generation is None
                        else f"generation-{previous_generation:06d}"
                    ),
                    "current_manifest_sha256": candidate.manifest_sha256,
                },
                overwrite=True,
            )
            _fsync_directory(self.root)
            keep = {generation}
            if previous_generation is not None:
                keep.add(previous_generation)
            for old_generation, old_path in self._generation_paths():
                if old_generation not in keep:
                    self._remove_generation(old_path)
            return candidate
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def _remove_generation(self, path: Path) -> None:
        resolved = path.resolve()
        if resolved.parent != self.root or _GENERATION.fullmatch(resolved.name) is None:
            raise ConsistencyError("Refusing unsafe checkpoint cleanup target")
        if path.is_symlink():
            raise ConsistencyError("Refusing redirected checkpoint cleanup target")
        shutil.rmtree(path)
        _fsync_directory(self.root)

    def load_latest(self, expected_identity: CheckpointIdentity) -> LoadedCheckpoint:
        expected_identity.validate_phase()
        pointer_path = self.root / "latest.json"
        candidates: list[int] = []
        try:
            pointer = read_json(pointer_path)
            if pointer.get("version") == 1:
                for key in ("current", "previous"):
                    value = pointer.get(key)
                    if isinstance(value, str):
                        match = _GENERATION.fullmatch(value)
                        if match is not None:
                            candidates.append(int(match.group(1)))
        except ConsistencyError:
            pass
        candidates.extend(generation for generation, _ in reversed(self._generation_paths()))
        seen: set[int] = set()
        errors: list[int] = []
        paths = dict(self._generation_paths())
        for generation in candidates:
            if generation in seen or generation not in paths:
                continue
            seen.add(generation)
            try:
                return self._verify_generation(generation, paths[generation], expected_identity)
            except ConsistencyError:
                errors.append(generation)
        raise ConsistencyError(
            "No valid checkpoint generation is available",
            details={"invalid_generations": errors},
        )
