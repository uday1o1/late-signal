from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from latesignal.data.archive import inspect_tar_archive
from latesignal.data.config import ArchiveLimits
from latesignal.errors import DataArtifactError

LIMITS = ArchiveLimits(
    max_members=4,
    max_member_bytes=1024,
    max_expanded_bytes=2048,
    max_compression_ratio=100,
)


def _single_member(path: Path, name: str, payload: bytes = b"safe") -> Path:
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return path


@pytest.mark.parametrize("name", ["../escape", "/absolute", "safe/../escape"])
def test_archive_rejects_traversal_and_absolute_names(tmp_path: Path, name: str) -> None:
    archive = _single_member(tmp_path / "bad.tar.gz", name)

    with pytest.raises(DataArtifactError, match=r"unsafe|absolute"):
        inspect_tar_archive(archive, LIMITS)


def test_archive_rejects_symbolic_links(tmp_path: Path) -> None:
    archive_path = tmp_path / "link.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "target"
        archive.addfile(info)

    with pytest.raises(DataArtifactError, match="links are forbidden"):
        inspect_tar_archive(archive_path, LIMITS)


def test_archive_rejects_duplicate_member_names(tmp_path: Path) -> None:
    archive_path = tmp_path / "duplicate.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for _ in range(2):
            info = tarfile.TarInfo("same")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))

    with pytest.raises(DataArtifactError, match="duplicate"):
        inspect_tar_archive(archive_path, LIMITS)


def test_archive_rejects_expansion_ratio(tmp_path: Path) -> None:
    archive = _single_member(tmp_path / "ratio.tar.gz", "zeros", b"0" * 1000)
    strict = ArchiveLimits(
        max_members=2,
        max_member_bytes=2000,
        max_expanded_bytes=2000,
        max_compression_ratio=1.1,
    )

    with pytest.raises(DataArtifactError, match="compression-ratio"):
        inspect_tar_archive(archive, strict)


def test_archive_enforces_exact_member_order(tmp_path: Path) -> None:
    archive = _single_member(tmp_path / "members.tar.gz", "actual")

    with pytest.raises(DataArtifactError, match="member list"):
        inspect_tar_archive(archive, LIMITS, expected_members=("expected",))
