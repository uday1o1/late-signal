from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from conftest import write_config

from latesignal.data.config import load_data_config
from latesignal.data.download import fetch_dataset
from latesignal.errors import FirstDownloadReviewRequired, LicenseNotAcceptedError


def test_fetch_refuses_before_opening_url(tmp_path: Path, valid_archive: Path) -> None:
    config = load_data_config(write_config(tmp_path / "data.yaml", valid_archive))
    opened = False

    def forbidden_opener(_: str) -> object:
        nonlocal opened
        opened = True
        raise AssertionError("opener must not be called")

    with pytest.raises(LicenseNotAcceptedError):
        fetch_dataset(
            config,
            tmp_path / "raw",
            accept_license=False,
            opener=forbidden_opener,  # type: ignore[arg-type]
        )

    assert opened is False
    assert not (tmp_path / "raw" / "acknowledgements").exists()


def test_first_download_requires_explicit_hash_review(tmp_path: Path, valid_archive: Path) -> None:
    config = load_data_config(write_config(tmp_path / "data.yaml", valid_archive))
    data_root = tmp_path / "raw"
    notices = []

    with pytest.raises(FirstDownloadReviewRequired) as raised:
        fetch_dataset(
            config,
            data_root,
            accept_license=True,
            notice_handler=notices.append,
        )

    digest = hashlib.sha256(valid_archive.read_bytes()).hexdigest()
    assert raised.value.details["sha256"] == digest
    assert notices[0].dataset == config.dataset.name
    assert not (data_root / "manifests" / "artifact-lock.json").exists()

    result = fetch_dataset(
        config,
        data_root,
        accept_license=True,
        reviewed_sha256=digest,
    )

    assert result.sha256 == digest
    assert result.reused is True
    assert Path(result.artifact_lock).exists()
    assert len(list((data_root / "acknowledgements").glob("*.json"))) == 2


def test_authored_digest_locks_in_one_pass(tmp_path: Path, valid_archive: Path) -> None:
    digest = hashlib.sha256(valid_archive.read_bytes()).hexdigest()
    config = load_data_config(
        write_config(tmp_path / "data.yaml", valid_archive, expected_sha256=digest)
    )

    result = fetch_dataset(config, tmp_path / "raw", accept_license=True)

    assert result.sha256 == digest
    assert result.reused is False
