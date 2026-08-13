"""Fail-closed tar archive inspection without extraction."""

from __future__ import annotations

import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from latesignal.data.config import ArchiveLimits
from latesignal.errors import DataArtifactError


@dataclass(frozen=True)
class ArchiveMember:
    name: str
    size: int
    kind: str


@dataclass(frozen=True)
class ArchiveInspection:
    members: tuple[ArchiveMember, ...]
    expanded_bytes: int
    compression_ratio: float

    def as_dict(self) -> dict[str, object]:
        return {
            "members": [asdict(member) for member in self.members],
            "expanded_bytes": self.expanded_bytes,
            "compression_ratio": self.compression_ratio,
        }


def _validate_name(name: str) -> None:
    candidate = PurePosixPath(name)
    if not name or name.startswith("/") or candidate.is_absolute():
        raise DataArtifactError("Archive contains an absolute or empty member path")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise DataArtifactError("Archive contains an unsafe member path", details={"member": name})


def inspect_tar_archive(
    path: Path,
    limits: ArchiveLimits,
    *,
    expected_members: tuple[str, ...] | None = None,
) -> ArchiveInspection:
    """Inspect every member and reject unsafe or unexpectedly large archives."""

    try:
        archive_bytes = path.stat().st_size
    except OSError as error:
        raise DataArtifactError(f"Could not stat archive: {path}") from error
    if archive_bytes <= 0:
        raise DataArtifactError("Archive is empty")

    members: list[ArchiveMember] = []
    names: set[str] = set()
    expanded_bytes = 0
    try:
        with tarfile.open(path, mode="r:*") as archive:
            for index, member in enumerate(archive, start=1):
                if index > limits.max_members:
                    raise DataArtifactError(
                        "Archive exceeds the configured member limit",
                        details={"max_members": limits.max_members},
                    )
                _validate_name(member.name)
                if member.name in names:
                    raise DataArtifactError(
                        "Archive contains duplicate member names",
                        details={"member": member.name},
                    )
                names.add(member.name)
                if member.issym() or member.islnk():
                    raise DataArtifactError(
                        "Archive links are forbidden", details={"member": member.name}
                    )
                if member.isdev() or member.isfifo():
                    raise DataArtifactError(
                        "Archive special files are forbidden", details={"member": member.name}
                    )
                if not (member.isfile() or member.isdir()):
                    raise DataArtifactError(
                        "Archive contains an unsupported member type",
                        details={"member": member.name, "type": repr(member.type)},
                    )
                if member.size < 0 or member.size > limits.max_member_bytes:
                    raise DataArtifactError(
                        "Archive member exceeds the configured size limit",
                        details={"member": member.name, "size": member.size},
                    )
                expanded_bytes += member.size
                if expanded_bytes > limits.max_expanded_bytes:
                    raise DataArtifactError(
                        "Archive exceeds the configured expanded-size limit",
                        details={
                            "expanded_bytes": expanded_bytes,
                            "max_expanded_bytes": limits.max_expanded_bytes,
                        },
                    )
                members.append(
                    ArchiveMember(
                        name=member.name,
                        size=member.size,
                        kind="file" if member.isfile() else "directory",
                    )
                )
    except (tarfile.TarError, OSError) as error:
        raise DataArtifactError("Archive could not be parsed as a tar file") from error

    compression_ratio = expanded_bytes / archive_bytes
    if compression_ratio > limits.max_compression_ratio:
        raise DataArtifactError(
            "Archive exceeds the configured compression-ratio limit",
            details={
                "compression_ratio": compression_ratio,
                "max_compression_ratio": limits.max_compression_ratio,
            },
        )
    if expected_members is not None:
        actual = tuple(member.name for member in members)
        if actual != expected_members:
            raise DataArtifactError(
                "Archive member list does not match the reviewed contract",
                details={"expected": list(expected_members), "actual": list(actual)},
            )
    return ArchiveInspection(
        members=tuple(members),
        expanded_bytes=expanded_bytes,
        compression_ratio=compression_ratio,
    )
