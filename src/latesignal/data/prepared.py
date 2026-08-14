"""Verified manifest-only access to prepared restricted data."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from latesignal.data.manifests import read_json, sha256_file
from latesignal.errors import ConsistencyError

_FEATURE_DAY = re.compile(r"^features/click_day=(\d{3})/[^/]+\.parquet$")
_TRUTH_DAY = re.compile(r"^truth/(reveal|maturity)/(?:reveal|maturity)_day=(\d{3})/[^/]+\.parquet$")


@dataclass(frozen=True, slots=True)
class PreparedFile:
    relative_path: str
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class PreparedInventory:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    manifest_bytes: int
    files: tuple[PreparedFile, ...]
    total_bytes: int
    manifest: dict[str, Any]

    def content_addressed_root(self, storage_root: Path) -> Path:
        """Return the only supported shared-store location for this identity."""

        return storage_root / "sha256" / self.manifest_sha256

    def feature_files(self, *, first_day: int, last_day: int) -> tuple[Path, ...]:
        if first_day < 0 or last_day < first_day:
            raise ValueError("Feature-day bounds are invalid")
        selected: list[Path] = []
        for item in self.files:
            match = _FEATURE_DAY.fullmatch(item.relative_path)
            if match is not None and first_day <= int(match.group(1)) <= last_day:
                selected.append(self.root / item.relative_path)
        return tuple(selected)

    def truth_files(
        self,
        kind: Literal["reveal", "maturity"],
        *,
        first_day: int,
        last_day: int,
    ) -> tuple[Path, ...]:
        if first_day < 0 or last_day < first_day:
            raise ValueError("Truth-day bounds are invalid")
        selected: list[Path] = []
        for item in self.files:
            match = _TRUTH_DAY.fullmatch(item.relative_path)
            if (
                match is not None
                and match.group(1) == kind
                and first_day <= int(match.group(2)) <= last_day
            ):
                selected.append(self.root / item.relative_path)
        return tuple(selected)


def _relative_file(root: Path, relative: object) -> tuple[str, Path]:
    if not isinstance(relative, str):
        raise ConsistencyError("Prepared-data inventory path must be a string")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ConsistencyError("Prepared-data inventory path escapes its root")
    candidate = root.joinpath(*pure.parts)
    return pure.as_posix(), candidate


def _verified_file(path: Path, *, expected_sha256: str, expected_bytes: int) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ConsistencyError(f"Prepared-data file is unavailable: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ConsistencyError(f"Prepared-data input is not a regular file: {path}")
    actual_sha256, actual_bytes = sha256_file(path)
    if actual_sha256 != expected_sha256 or actual_bytes != expected_bytes:
        raise ConsistencyError(
            "Prepared-data file identity does not match its manifest",
            details={"path": str(path)},
        )


def _control_files(root: Path, manifest: dict[str, Any]) -> set[str]:
    allowed: set[str] = set()
    inspection_sha256 = manifest.get("inspection_sha256")
    if inspection_sha256 is None:
        return allowed
    if not isinstance(inspection_sha256, str):
        raise ConsistencyError("Prepared manifest has an invalid inspection identity")
    inspection_path = root / "manifests" / "inspection.json"
    try:
        inspection_bytes = inspection_path.lstat().st_size
    except OSError as error:
        raise ConsistencyError("Inspection manifest is unavailable") from error
    _verified_file(
        inspection_path,
        expected_sha256=inspection_sha256,
        expected_bytes=inspection_bytes,
    )
    inspection = read_json(inspection_path)
    allowed.add("manifests/inspection.json")
    quarantine = inspection.get("quarantine")
    if not isinstance(quarantine, dict):
        raise ConsistencyError("Inspection manifest has no quarantine identity")
    quarantine_sha256 = quarantine.get("sha256")
    quarantine_bytes = quarantine.get("bytes")
    if (
        not isinstance(quarantine_sha256, str)
        or isinstance(quarantine_bytes, bool)
        or not isinstance(quarantine_bytes, int)
    ):
        raise ConsistencyError("Inspection quarantine identity is malformed")
    quarantine_path = root / "quarantine" / "rejected.jsonl"
    _verified_file(
        quarantine_path,
        expected_sha256=quarantine_sha256,
        expected_bytes=quarantine_bytes,
    )
    allowed.add("quarantine/rejected.jsonl")
    return allowed


def _actual_files(root: Path) -> set[str]:
    files: set[str] = set()
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            candidate = current_path / directory
            if candidate.is_symlink():
                raise ConsistencyError(f"Prepared-data root contains a symlink: {candidate}")
        for name in names:
            candidate = current_path / name
            if candidate.is_symlink():
                raise ConsistencyError(f"Prepared-data root contains a symlink: {candidate}")
            files.add(candidate.relative_to(root).as_posix())
    return files


def verify_prepared_inventory(
    manifest_path: Path,
    *,
    reject_unlisted: bool = True,
) -> PreparedInventory:
    """Verify every authored byte and expose only manifest-listed data files."""

    if manifest_path.is_symlink():
        raise ConsistencyError("Prepared-data manifest cannot be a symlink")
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent.parent.resolve()
    manifest = read_json(manifest_path)
    rows = manifest.get("rows")
    numeric = manifest.get("numeric_statistics")
    raw_files = manifest.get("files")
    if (
        manifest.get("manifest_version") != 1
        or not isinstance(rows, dict)
        or rows.get("reconciled") is not True
        or not isinstance(numeric, dict)
        or numeric.get("fit_click_days") != [0, 14]
        or not isinstance(raw_files, list)
    ):
        raise ConsistencyError("Prepared-data manifest does not meet the final lock contract")

    files: list[PreparedFile] = []
    seen: set[str] = set()
    total_bytes = 0
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise ConsistencyError("Prepared-data inventory contains a malformed entry")
        relative, candidate = _relative_file(root, raw.get("path"))
        expected_sha256 = raw.get("sha256")
        expected_bytes = raw.get("bytes")
        if (
            relative in seen
            or not isinstance(expected_sha256, str)
            or isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
        ):
            raise ConsistencyError("Prepared-data inventory entry has invalid fields")
        _verified_file(
            candidate,
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
        )
        seen.add(relative)
        total_bytes += expected_bytes
        files.append(PreparedFile(relative, expected_sha256, expected_bytes))

    manifest_sha256, manifest_bytes = sha256_file(manifest_path)
    if reject_unlisted:
        manifest_relative = manifest_path.relative_to(root).as_posix()
        allowed = seen | {manifest_relative} | _control_files(root, manifest)
        unlisted = sorted(_actual_files(root) - allowed)
        if unlisted:
            raise ConsistencyError(
                "Prepared-data root contains unlisted files",
                details={"paths": unlisted},
            )
    return PreparedInventory(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        manifest_bytes=manifest_bytes,
        files=tuple(files),
        total_bytes=total_bytes,
        manifest=manifest,
    )
