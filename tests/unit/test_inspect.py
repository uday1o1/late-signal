from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from conftest import FIELDS, raw_row, write_archive, write_config

from latesignal.data.config import load_data_config
from latesignal.data.inspect import inspect_archive
from latesignal.errors import AmbiguousTimeUnitError


def test_inspection_reconciles_rows_and_selects_unique_seconds(
    tmp_path: Path, valid_archive: Path
) -> None:
    config = load_data_config(write_config(tmp_path / "data.yaml", valid_archive))
    manifest_path = tmp_path / "inspection.json"
    quarantine_path = tmp_path / "rejected.jsonl"

    manifest = inspect_archive(
        config,
        valid_archive,
        manifest_path=manifest_path,
        quarantine_path=quarantine_path,
    )

    assert manifest["rows"] == {
        "parsed": 3,
        "accepted": 3,
        "quarantined": 0,
        "reconciled": True,
        "duplicate_raw_rows": 0,
    }
    assert manifest["time_unit"]["selected_seconds_per_raw_unit"] == 1.0
    assert manifest["click_time"]["span_days"] == 89.0
    assert manifest["positive_delay"]["max_days"] == 1.0
    assert quarantine_path.read_text(encoding="utf-8") == ""
    assert manifest_path.exists()


def test_invalid_rows_are_quarantined_by_index_and_reason(tmp_path: Path) -> None:
    base = 100
    valid_first = raw_row(sale=1, amount=10, delay=86_400, click=base, suffix="a")
    invalid = raw_row(
        sale=0,
        amount=-1,
        delay=0,
        click=base + 40 * 86_400,
        suffix="bad",
    )
    valid_last = raw_row(
        sale=0,
        amount=-1,
        delay=-1,
        click=base + 89 * 86_400,
        suffix="c",
    )
    duplicate = valid_last
    data = (
        "\t".join(FIELDS) + "\n" + "\n".join([valid_first, invalid, valid_last, duplicate]) + "\n"
    ).encode()
    archive = write_archive(tmp_path / "fixture.tar.gz", data)
    config = load_data_config(write_config(tmp_path / "data.yaml", archive))

    manifest = inspect_archive(
        config,
        archive,
        manifest_path=tmp_path / "inspection.json",
        quarantine_path=tmp_path / "rejected.jsonl",
    )

    assert manifest["rows"]["parsed"] == 4
    assert manifest["rows"]["accepted"] == 3
    assert manifest["rows"]["quarantined"] == 1
    assert manifest["rows"]["duplicate_raw_rows"] == 1
    rejected = (tmp_path / "rejected.jsonl").read_text(encoding="utf-8")
    assert '"raw_row_index": 1' in rejected
    assert "SALE_DELAY_INCONSISTENT" in rejected


def test_ambiguous_time_unit_publishes_no_outputs(tmp_path: Path) -> None:
    rows = [
        raw_row(sale=1, amount=10, delay=1, click=100, suffix="a"),
        raw_row(sale=0, amount=-1, delay=-1, click=101, suffix="b"),
    ]
    data = ("\t".join(FIELDS) + "\n" + "\n".join(rows) + "\n").encode()
    archive = write_archive(tmp_path / "short.tar.gz", data)
    config = load_data_config(write_config(tmp_path / "data.yaml", archive))
    manifest_path = tmp_path / "inspection.json"
    quarantine_path = tmp_path / "rejected.jsonl"

    with pytest.raises(AmbiguousTimeUnitError) as raised:
        inspect_archive(
            config,
            archive,
            manifest_path=manifest_path,
            quarantine_path=quarantine_path,
        )

    assert raised.value.error_code == "AMBIGUOUS_TIME_UNIT"
    assert not manifest_path.exists()
    assert not quarantine_path.exists()


def test_extracted_hash_covers_the_exact_data_member(tmp_path: Path, valid_archive: Path) -> None:
    config = load_data_config(write_config(tmp_path / "data.yaml", valid_archive))
    manifest = inspect_archive(
        config,
        valid_archive,
        manifest_path=tmp_path / "inspection.json",
        quarantine_path=tmp_path / "rejected.jsonl",
    )

    import tarfile

    with tarfile.open(valid_archive, "r:gz") as archive:
        stream = archive.extractfile("CriteoSearchData")
        assert stream is not None
        exact = stream.read()
    assert manifest["extracted_data_member"]["sha256"] == hashlib.sha256(exact).hexdigest()
    assert manifest["extracted_data_member"]["bytes"] == len(exact)
