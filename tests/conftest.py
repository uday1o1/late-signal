from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path
from typing import Any

import pytest
import yaml

FIELDS = [
    "Sale",
    "SalesAmountInEuro",
    "time_delay_for_conversion",
    "click_timestamp",
    "nb_clicks_1week",
    "product_price",
    "product_age_group",
    "device_type",
    "audience_id",
    "product_gender",
    "product_brand",
    "product_category1",
    "product_category2",
    "product_category3",
    "product_category4",
    "product_category5",
    "product_category6",
    "product_category7",
    "product_country",
    "product_id",
    "product_title",
    "partner_id",
    "user_id",
]


def raw_row(*, sale: int, amount: float, delay: float, click: float, suffix: str = "1") -> str:
    values: list[object] = [sale, amount, delay, click, 3, 19.5]
    values.extend(f"{suffix}-{index}" for index in range(17))
    assert len(values) == 23
    return "\t".join(str(value) for value in values)


def write_archive(path: Path, data: bytes, *, unsafe_name: str | None = None) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        readme = b"Synthetic Criteo-compatible fixture.\n"
        readme_info = tarfile.TarInfo("README.md")
        readme_info.size = len(readme)
        archive.addfile(readme_info, io.BytesIO(readme))
        data_info = tarfile.TarInfo(unsafe_name or "CriteoSearchData")
        data_info.size = len(data)
        archive.addfile(data_info, io.BytesIO(data))
    return path


def write_config(
    path: Path,
    archive: Path,
    *,
    has_header: bool = True,
    expected_sha256: str | None = None,
    members: list[str] | None = None,
    max_ratio: float = 100.0,
) -> Path:
    payload: dict[str, Any] = {
        "version": 1,
        "dataset": {
            "name": "Synthetic Sponsored Search Fixture",
            "license_id": "CC-BY-NC-SA-4.0",
            "official_page": "https://example.invalid/dataset",
            "archive_url": archive.resolve().as_uri(),
            "archive_filename": archive.name,
            "expected_bytes": archive.stat().st_size,
            "expected_sha256": expected_sha256,
            "expected_members": members or ["README.md", "CriteoSearchData"],
            "data_member": "CriteoSearchData",
            "noncommercial_notice": "Synthetic fixture restriction notice.",
        },
        "archive_limits": {
            "max_members": 8,
            "max_member_bytes": 1024 * 1024,
            "max_expanded_bytes": 2 * 1024 * 1024,
            "max_compression_ratio": max_ratio,
        },
        "schema": {"delimiter": "\t", "has_header": has_header, "fields": FIELDS},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def valid_archive(tmp_path: Path) -> Path:
    base = 100
    rows = [
        raw_row(sale=1, amount=10, delay=86_400, click=base, suffix="a"),
        raw_row(sale=0, amount=-1, delay=-1, click=base + 44 * 86_400, suffix="b"),
        raw_row(sale=0, amount=-1, delay=-1, click=base + 89 * 86_400, suffix="c"),
    ]
    data = ("\t".join(FIELDS) + "\n" + "\n".join(rows) + "\n").encode()
    return write_archive(tmp_path / "fixture.tar.gz", data)


@pytest.fixture
def trusted_config(tmp_path: Path, valid_archive: Path) -> Path:
    digest = hashlib.sha256(valid_archive.read_bytes()).hexdigest()
    return write_config(tmp_path / "data.yaml", valid_archive, expected_sha256=digest)
