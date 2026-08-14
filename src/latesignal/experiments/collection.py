"""Aggregate-only collection manifests for completed remote studies."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Any

from latesignal.data.manifests import (
    canonical_json_bytes,
    read_json,
    sha256_file,
    write_json_atomic,
)
from latesignal.errors import ConsistencyError

_REQUIRED = (
    "feasibility.json",
    "selection/selection-results.json",
    "protocol-lock.json",
    "quality-gate.json",
    "final/final-manifest.json",
)
_ALLOWED_AGGREGATE_SUFFIXES = frozenset({".json", ".csv", ".html", ".npz"})
_FORBIDDEN_NAMES = ("checkpoint", "prediction", "probabilities", "model-weight")
_MAX_COLLECTION_BYTES = 2 * 1024**3


def _safe_relative(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ConsistencyError("Collection manifest contains an unsafe relative path")
    return relative


def _aggregate_files(root: Path) -> list[Path]:
    aggregate = root / "final" / "aggregate"
    if aggregate.is_symlink() or not aggregate.is_dir():
        raise ConsistencyError("Final aggregate directory is missing or redirected")
    files: list[Path] = []
    for current, directories, names in os.walk(aggregate, followlinks=False):
        current_path = Path(current)
        if any((current_path / name).is_symlink() for name in directories):
            raise ConsistencyError("Final aggregate contains a redirected directory")
        for name in names:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise ConsistencyError("Final aggregate contains a non-regular artifact")
            relative = path.relative_to(root).as_posix()
            if path.suffix not in _ALLOWED_AGGREGATE_SUFFIXES or any(
                token in relative.lower() for token in _FORBIDDEN_NAMES
            ):
                raise ConsistencyError("Final aggregate contains a prohibited artifact")
            files.append(path)
    if not files:
        raise ConsistencyError("Final aggregate contains no collectable evidence")
    return sorted(files)


def build_collection_manifest(job_root: Path) -> dict[str, Any]:
    """Seal the exact small aggregate-only artifacts permitted to leave the GPU host."""

    root = job_root.resolve()
    if job_root.is_symlink():
        raise ConsistencyError("Collection job root cannot be a symbolic link")
    paths = [root / relative for relative in _REQUIRED]
    provenance = root / "selection-provenance.json"
    if provenance.exists():
        paths.append(provenance)
    paths.extend(_aggregate_files(root))
    entries: list[dict[str, object]] = []
    total_bytes = 0
    for path in paths:
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ConsistencyError("Collection source is missing, redirected, or unsafe")
        digest, size = sha256_file(path)
        total_bytes += size
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "bytes": size,
            }
        )
    if total_bytes > _MAX_COLLECTION_BYTES:
        raise ConsistencyError("Aggregate-only collection exceeds the retained-artifact cap")
    payload: dict[str, Any] = {
        "version": 1,
        "status": "verified_aggregate_only",
        "files": entries,
        "file_count": len(entries),
        "total_bytes": total_bytes,
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    manifest_path = root / "collection-manifest.json"
    if manifest_path.exists():
        if read_json(manifest_path) != payload:
            raise ConsistencyError("Immutable collection manifest changed")
    else:
        write_json_atomic(manifest_path, payload)
    return payload


def verify_collection_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify a collected directory contains exactly its sealed regular files."""

    if root.is_symlink() or manifest_path.is_symlink():
        raise ConsistencyError("Collected evidence cannot use symbolic links")
    manifest = read_json(manifest_path)
    expected_sha256 = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    raw_files = manifest.get("files")
    if (
        manifest.get("status") != "verified_aggregate_only"
        or not isinstance(expected_sha256, str)
        or hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != expected_sha256
        or not isinstance(raw_files, list)
    ):
        raise ConsistencyError("Collection manifest is malformed or changed")
    expected_paths: set[str] = set()
    total_bytes = 0
    for item in raw_files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ConsistencyError("Collection manifest file entry is malformed")
        relative = _safe_relative(item["path"])
        path = root.joinpath(*relative.parts)
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ConsistencyError("Collected artifact is missing, redirected, or unsafe")
        digest, size = sha256_file(path)
        if digest != item.get("sha256") or size != item.get("bytes"):
            raise ConsistencyError("Collected artifact content changed in transit")
        expected_paths.add(relative.as_posix())
        total_bytes += size
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if (
        actual_paths != expected_paths
        or manifest.get("file_count") != len(expected_paths)
        or manifest.get("total_bytes") != total_bytes
        or total_bytes > _MAX_COLLECTION_BYTES
    ):
        raise ConsistencyError("Collected evidence does not match its exact manifest")
    return manifest
