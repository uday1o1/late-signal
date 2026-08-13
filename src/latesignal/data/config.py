"""Strict authored configuration for the data boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from latesignal.errors import ConfigurationError


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    license_id: str
    official_page: str
    archive_url: str
    archive_filename: str
    expected_bytes: int
    expected_sha256: str | None
    expected_members: tuple[str, ...]
    data_member: str
    noncommercial_notice: str


@dataclass(frozen=True)
class ArchiveLimits:
    max_members: int
    max_member_bytes: int
    max_expanded_bytes: int
    max_compression_ratio: float


@dataclass(frozen=True)
class RawSchema:
    delimiter: str
    has_header: bool
    fields: tuple[str, ...]


@dataclass(frozen=True)
class DataConfig:
    version: int
    dataset: DatasetSpec
    archive_limits: ArchiveLimits
    schema: RawSchema
    canonical_sha256: str


_TOP_KEYS = {"version", "dataset", "archive_limits", "schema"}
_DATASET_KEYS = {
    "name",
    "license_id",
    "official_page",
    "archive_url",
    "archive_filename",
    "expected_bytes",
    "expected_sha256",
    "expected_members",
    "data_member",
    "noncommercial_notice",
}
_LIMIT_KEYS = {
    "max_members",
    "max_member_bytes",
    "max_expanded_bytes",
    "max_compression_ratio",
}
_SCHEMA_KEYS = {"delimiter", "has_header", "fields"}


def _mapping(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{location} must be a mapping with string keys")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], location: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise ConfigurationError(
            f"{location} has invalid keys",
            details={"unknown": unknown, "missing": missing},
        )


def _positive_int(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{location} must be a positive integer")
    return value


def _positive_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigurationError(f"{location} must be positive")
    return float(value)


def _text(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{location} must be a nonempty string")
    return value


def _sha256_or_none(value: object, location: str) -> str | None:
    if value is None:
        return None
    result = _text(value, location).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ConfigurationError(f"{location} must be a lowercase SHA-256 hex digest")
    return result


def load_data_config(path: Path) -> DataConfig:
    """Load a fail-closed data configuration and compute its canonical hash."""

    try:
        raw_object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"Could not read data configuration: {path}") from error
    raw = _mapping(raw_object, "configuration")
    _exact_keys(raw, _TOP_KEYS, "configuration")
    if raw["version"] != 1:
        raise ConfigurationError("Only data configuration version 1 is supported")

    dataset_raw = _mapping(raw["dataset"], "dataset")
    limits_raw = _mapping(raw["archive_limits"], "archive_limits")
    schema_raw = _mapping(raw["schema"], "schema")
    _exact_keys(dataset_raw, _DATASET_KEYS, "dataset")
    _exact_keys(limits_raw, _LIMIT_KEYS, "archive_limits")
    _exact_keys(schema_raw, _SCHEMA_KEYS, "schema")

    members_raw = dataset_raw["expected_members"]
    if not isinstance(members_raw, list) or not members_raw:
        raise ConfigurationError("dataset.expected_members must be a nonempty list")
    members = tuple(_text(item, "dataset.expected_members item") for item in members_raw)
    if len(set(members)) != len(members):
        raise ConfigurationError("dataset.expected_members contains duplicates")

    fields_raw = schema_raw["fields"]
    if not isinstance(fields_raw, list) or len(fields_raw) != 23:
        raise ConfigurationError("schema.fields must contain exactly 23 fields")
    fields = tuple(_text(item, "schema.fields item") for item in fields_raw)
    if len(set(fields)) != len(fields):
        raise ConfigurationError("schema.fields contains duplicates")
    delimiter_raw = schema_raw["delimiter"]
    if not isinstance(delimiter_raw, str) or not delimiter_raw:
        raise ConfigurationError("schema.delimiter must be a nonempty string")
    delimiter = delimiter_raw
    if len(delimiter) != 1:
        raise ConfigurationError("schema.delimiter must be one character")
    if not isinstance(schema_raw["has_header"], bool):
        raise ConfigurationError("schema.has_header must be a boolean")

    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    canonical_sha256 = hashlib.sha256(canonical.encode()).hexdigest()
    return DataConfig(
        version=1,
        dataset=DatasetSpec(
            name=_text(dataset_raw["name"], "dataset.name"),
            license_id=_text(dataset_raw["license_id"], "dataset.license_id"),
            official_page=_text(dataset_raw["official_page"], "dataset.official_page"),
            archive_url=_text(dataset_raw["archive_url"], "dataset.archive_url"),
            archive_filename=_text(dataset_raw["archive_filename"], "dataset.archive_filename"),
            expected_bytes=_positive_int(dataset_raw["expected_bytes"], "dataset.expected_bytes"),
            expected_sha256=_sha256_or_none(
                dataset_raw["expected_sha256"], "dataset.expected_sha256"
            ),
            expected_members=members,
            data_member=_text(dataset_raw["data_member"], "dataset.data_member"),
            noncommercial_notice=_text(
                dataset_raw["noncommercial_notice"], "dataset.noncommercial_notice"
            ),
        ),
        archive_limits=ArchiveLimits(
            max_members=_positive_int(limits_raw["max_members"], "archive_limits.max_members"),
            max_member_bytes=_positive_int(
                limits_raw["max_member_bytes"], "archive_limits.max_member_bytes"
            ),
            max_expanded_bytes=_positive_int(
                limits_raw["max_expanded_bytes"], "archive_limits.max_expanded_bytes"
            ),
            max_compression_ratio=_positive_number(
                limits_raw["max_compression_ratio"], "archive_limits.max_compression_ratio"
            ),
        ),
        schema=RawSchema(delimiter=delimiter, has_header=schema_raw["has_header"], fields=fields),
        canonical_sha256=canonical_sha256,
    )
