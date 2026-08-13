"""License-gated, streaming, content-addressed dataset download."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, cast

from latesignal import __version__
from latesignal.data.archive import ArchiveInspection, inspect_tar_archive
from latesignal.data.config import DataConfig
from latesignal.data.manifests import read_json, sha256_file, write_json_atomic
from latesignal.errors import (
    ConsistencyError,
    DataArtifactError,
    FirstDownloadReviewRequired,
    LicenseNotAcceptedError,
)

CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class FetchNotice:
    dataset: str
    license_id: str
    official_page: str
    archive_url: str
    destination: str
    restriction: str


@dataclass(frozen=True)
class FetchResult:
    archive_path: str
    sha256: str
    bytes: int
    artifact_lock: str
    reused: bool
    archive: dict[str, object]


NoticeHandler = Callable[[FetchNotice], None]
Opener = Callable[[str], IO[bytes]]


def _default_opener(url: str) -> IO[bytes]:
    return cast(IO[bytes], urllib.request.urlopen(url, timeout=60))


def _artifact_path(data_root: Path, sha256: str, filename: str) -> Path:
    return data_root / "artifacts" / "sha256" / sha256 / filename


def _lock_payload(
    config: DataConfig,
    archive_path: Path,
    sha256: str,
    size: int,
    archive: ArchiveInspection,
    trust_source: str,
) -> dict[str, object]:
    return {
        "manifest_version": 1,
        "dataset": config.dataset.name,
        "config_sha256": config.canonical_sha256,
        "archive_path": str(archive_path.resolve()),
        "archive_sha256": sha256,
        "archive_bytes": size,
        "archive_url": config.dataset.archive_url,
        "archive": archive.as_dict(),
        "trust_source": trust_source,
        "reviewed_at": datetime.now(UTC).isoformat(),
        "code_version": __version__,
    }


def _record_acknowledgement(config: DataConfig, data_root: Path) -> Path:
    timestamp = datetime.now(UTC)
    path = (
        data_root
        / "acknowledgements"
        / f"{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex}.json"
    )
    write_json_atomic(
        path,
        {
            "manifest_version": 1,
            "dataset": config.dataset.name,
            "license_id": config.dataset.license_id,
            "official_page": config.dataset.official_page,
            "acknowledged_at": timestamp.isoformat(),
            "code_version": __version__,
            "config_sha256": config.canonical_sha256,
        },
    )
    return path


def _verify_locked_artifact(config: DataConfig, data_root: Path) -> FetchResult | None:
    lock_path = data_root / "manifests" / "artifact-lock.json"
    if not lock_path.exists():
        return None
    lock = read_json(lock_path)
    if lock.get("config_sha256") != config.canonical_sha256:
        raise ConsistencyError(
            "The existing artifact lock belongs to a different data configuration"
        )
    raw_path = lock.get("archive_path")
    expected_hash = lock.get("archive_sha256")
    expected_bytes = lock.get("archive_bytes")
    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
        raise ConsistencyError("The artifact lock is malformed")
    archive_path = Path(raw_path)
    actual_hash, actual_bytes = sha256_file(archive_path)
    if actual_hash != expected_hash or actual_bytes != expected_bytes:
        raise DataArtifactError("The locked archive no longer matches its local manifest")
    inspection = inspect_tar_archive(
        archive_path,
        config.archive_limits,
        expected_members=config.dataset.expected_members,
    )
    return FetchResult(
        archive_path=str(archive_path),
        sha256=actual_hash,
        bytes=actual_bytes,
        artifact_lock=str(lock_path),
        reused=True,
        archive=inspection.as_dict(),
    )


def _review_candidate(
    config: DataConfig, data_root: Path, reviewed_sha256: str
) -> FetchResult | None:
    candidate_path = data_root / "manifests" / "download-candidate.json"
    if not candidate_path.exists():
        return None
    candidate = read_json(candidate_path)
    if candidate.get("config_sha256") != config.canonical_sha256:
        raise ConsistencyError("The download candidate belongs to a different configuration")
    if reviewed_sha256 != candidate.get("archive_sha256"):
        raise DataArtifactError(
            "The reviewed SHA-256 does not match the retained candidate",
            details={
                "reviewed_sha256": reviewed_sha256,
                "candidate_sha256": candidate.get("archive_sha256"),
            },
        )
    raw_path = candidate.get("archive_path")
    if not isinstance(raw_path, str):
        raise ConsistencyError("The download candidate is malformed")
    archive_path = Path(raw_path)
    actual_hash, actual_bytes = sha256_file(archive_path)
    if actual_hash != reviewed_sha256 or actual_bytes != candidate.get("archive_bytes"):
        raise DataArtifactError("The retained download candidate changed before review")
    inspection = inspect_tar_archive(
        archive_path,
        config.archive_limits,
        expected_members=config.dataset.expected_members,
    )
    lock_path = data_root / "manifests" / "artifact-lock.json"
    write_json_atomic(
        lock_path,
        _lock_payload(
            config,
            archive_path,
            actual_hash,
            actual_bytes,
            inspection,
            "explicit-first-download-review",
        ),
    )
    return FetchResult(
        archive_path=str(archive_path),
        sha256=actual_hash,
        bytes=actual_bytes,
        artifact_lock=str(lock_path),
        reused=True,
        archive=inspection.as_dict(),
    )


def fetch_dataset(
    config: DataConfig,
    data_root: Path,
    *,
    accept_license: bool,
    reviewed_sha256: str | None = None,
    notice_handler: NoticeHandler | None = None,
    opener: Opener = _default_opener,
) -> FetchResult:
    """Fetch the configured archive only after acknowledgement and safety checks."""

    if not accept_license:
        raise LicenseNotAcceptedError(
            "The dataset license must be reviewed and explicitly accepted",
            details={"required_flag": "--accept-license"},
        )
    if reviewed_sha256 is not None:
        reviewed_sha256 = reviewed_sha256.lower()
        if len(reviewed_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in reviewed_sha256
        ):
            raise DataArtifactError("--review-sha256 must be a SHA-256 hex digest")

    notice = FetchNotice(
        dataset=config.dataset.name,
        license_id=config.dataset.license_id,
        official_page=config.dataset.official_page,
        archive_url=config.dataset.archive_url,
        destination=str(data_root.resolve()),
        restriction=config.dataset.noncommercial_notice,
    )
    if notice_handler is not None:
        notice_handler(notice)
    _record_acknowledgement(config, data_root)

    locked = _verify_locked_artifact(config, data_root)
    if locked is not None:
        return locked
    if reviewed_sha256 is not None:
        reviewed = _review_candidate(config, data_root, reviewed_sha256)
        if reviewed is not None:
            return reviewed

    temporary_root = data_root / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix="download-", dir=temporary_root)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as output:
            try:
                with opener(config.dataset.archive_url) as response:
                    while chunk := response.read(CHUNK_BYTES):
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            except (OSError, urllib.error.URLError) as error:
                raise DataArtifactError("Dataset download failed") from error
            output.flush()
            os.fsync(output.fileno())
        sha256 = digest.hexdigest()
        if size != config.dataset.expected_bytes:
            raise DataArtifactError(
                "Archive byte count does not match the reviewed configuration",
                details={"expected": config.dataset.expected_bytes, "actual": size},
            )
        if config.dataset.expected_sha256 is not None and sha256 != config.dataset.expected_sha256:
            raise DataArtifactError(
                "Archive SHA-256 does not match the reviewed configuration",
                details={"expected": config.dataset.expected_sha256, "actual": sha256},
            )
        inspection = inspect_tar_archive(
            temporary,
            config.archive_limits,
            expected_members=config.dataset.expected_members,
        )
        artifact_path = _artifact_path(data_root, sha256, config.dataset.archive_filename)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if artifact_path.exists():
            existing_hash, existing_bytes = sha256_file(artifact_path)
            if existing_hash != sha256 or existing_bytes != size:
                raise ConsistencyError("Content-addressed archive path contains different bytes")
        else:
            os.replace(temporary, artifact_path)

        trusted_hash = config.dataset.expected_sha256 or reviewed_sha256
        if trusted_hash == sha256:
            lock_path = data_root / "manifests" / "artifact-lock.json"
            trust_source = (
                "authored-config-sha256"
                if config.dataset.expected_sha256 is not None
                else "caller-supplied-sha256"
            )
            write_json_atomic(
                lock_path,
                _lock_payload(config, artifact_path, sha256, size, inspection, trust_source),
            )
            return FetchResult(
                archive_path=str(artifact_path),
                sha256=sha256,
                bytes=size,
                artifact_lock=str(lock_path),
                reused=False,
                archive=inspection.as_dict(),
            )

        candidate_path = data_root / "manifests" / "download-candidate.json"
        write_json_atomic(
            candidate_path,
            {
                "manifest_version": 1,
                "dataset": config.dataset.name,
                "config_sha256": config.canonical_sha256,
                "archive_path": str(artifact_path.resolve()),
                "archive_sha256": sha256,
                "archive_bytes": size,
                "archive": inspection.as_dict(),
                "downloaded_at": datetime.now(UTC).isoformat(),
                "trusted": False,
            },
            overwrite=True,
        )
        raise FirstDownloadReviewRequired(
            "The first download has no authoritative configured digest and remains untrusted",
            details={
                "sha256": sha256,
                "bytes": size,
                "archive_path": str(artifact_path),
                "resume": (f"latesignal data fetch --accept-license --review-sha256 {sha256}"),
            },
        )
    finally:
        if temporary.exists():
            temporary.unlink()
