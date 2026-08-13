"""Streaming raw-data inspection and fail-closed timestamp-unit inference."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sqlite3
import tarfile
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from latesignal import __version__
from latesignal.data.archive import inspect_tar_archive
from latesignal.data.config import DataConfig, RawSchema
from latesignal.data.manifests import read_json, sha256_file, write_json_atomic
from latesignal.errors import AmbiguousTimeUnitError, ConsistencyError, DataArtifactError

SECONDS_PER_DAY = 86_400.0
MATURITY_DAYS = 30.0
TIME_MULTIPLIERS = (1.0, 1e-3, 1e-6, 1e-9)
_REQUIRED_FIELDS = {
    "Sale",
    "SalesAmountInEuro",
    "time_delay_for_conversion",
    "click_timestamp",
    "nb_clicks_1week",
    "product_price",
}
_NUMERIC_FIELDS = _REQUIRED_FIELDS


@dataclass(frozen=True)
class ParsedRow:
    sale: int
    sales_amount: float
    delay: float
    click_timestamp: float


@dataclass
class AuditState:
    parsed_rows: int = 0
    accepted_rows: int = 0
    quarantined_rows: int = 0
    duplicate_rows: int = 0
    last_click_raw: float | None = None
    min_click_raw: float | None = None
    max_click_raw: float | None = None
    positive_delay_count: int = 0
    positive_delay_sum_raw: float = 0.0
    min_positive_delay_raw: float | None = None
    max_positive_delay_raw: float | None = None


def _field_stats(schema: RawSchema) -> dict[str, dict[str, int]]:
    return {
        field: {
            "missing": 0,
            "sentinel": 0,
            "invalid": 0,
            "negative": 0,
            "nonfinite": 0,
        }
        for field in schema.fields
    }


def _update_lexical_stats(
    values: list[str], schema: RawSchema, stats: dict[str, dict[str, int]]
) -> None:
    for field, raw in zip(schema.fields, values, strict=True):
        stripped = raw.strip()
        if not stripped:
            stats[field]["missing"] += 1
            continue
        if stripped == "-1":
            stats[field]["sentinel"] += 1
        if field in _NUMERIC_FIELDS:
            try:
                numeric = float(stripped)
            except ValueError:
                stats[field]["invalid"] += 1
                continue
            if not math.isfinite(numeric):
                stats[field]["nonfinite"] += 1
            elif numeric < 0:
                stats[field]["negative"] += 1


def _parse_row(
    values: list[str], schema: RawSchema
) -> tuple[ParsedRow | None, list[str], list[str]]:
    if len(values) != len(schema.fields):
        return None, ["FIELD_COUNT"], []
    by_name = dict(zip(schema.fields, values, strict=True))
    reasons: list[str] = []
    fields: list[str] = []

    def parse_float(name: str) -> float | None:
        raw = by_name[name].strip()
        try:
            value = float(raw)
        except ValueError:
            reasons.append("INVALID_NUMBER")
            fields.append(name)
            return None
        if not math.isfinite(value):
            reasons.append("NONFINITE_NUMBER")
            fields.append(name)
            return None
        return value

    raw_sale = by_name["Sale"].strip()
    try:
        sale = int(raw_sale)
    except ValueError:
        sale = -1
        reasons.append("INVALID_SALE")
        fields.append("Sale")
    if sale not in {0, 1} and "Sale" not in fields:
        reasons.append("INVALID_SALE")
        fields.append("Sale")

    amount = parse_float("SalesAmountInEuro")
    delay = parse_float("time_delay_for_conversion")
    click = parse_float("click_timestamp")
    clicks_week = parse_float("nb_clicks_1week")
    price = parse_float("product_price")

    for name in schema.fields:
        if not by_name[name].strip():
            reasons.append("EMPTY_FIELD")
            fields.append(name)

    if click is not None and click <= 0:
        reasons.append("MISSING_OR_NEGATIVE_CLICK_TIME")
        fields.append("click_timestamp")
    for name, value in (("nb_clicks_1week", clicks_week), ("product_price", price)):
        if value is not None and value < -1:
            reasons.append("INVALID_NEGATIVE_VALUE")
            fields.append(name)
    if amount is not None:
        if sale == 0 and amount != -1:
            reasons.append("SALE_AMOUNT_INCONSISTENT")
            fields.append("SalesAmountInEuro")
        if sale == 1 and amount < 0:
            reasons.append("SALE_AMOUNT_INCONSISTENT")
            fields.append("SalesAmountInEuro")
    if delay is not None:
        if sale == 0 and delay != -1:
            reasons.append("SALE_DELAY_INCONSISTENT")
            fields.append("time_delay_for_conversion")
        if sale == 1 and delay < 0:
            reasons.append("SALE_DELAY_INCONSISTENT")
            fields.append("time_delay_for_conversion")

    if reasons or amount is None or delay is None or click is None:
        return None, list(dict.fromkeys(reasons)), list(dict.fromkeys(fields))
    return (
        ParsedRow(
            sale=sale,
            sales_amount=amount,
            delay=delay,
            click_timestamp=click,
        ),
        [],
        [],
    )


def _iter_member_lines(archive_path: Path, member_name: str) -> tuple[tarfile.TarFile, IO[bytes]]:
    try:
        archive = tarfile.open(archive_path, mode="r:*")  # noqa: SIM115
        member = archive.getmember(member_name)
        if not member.isfile():
            archive.close()
            raise DataArtifactError("Configured data member is not a regular file")
        stream = archive.extractfile(member)
        if stream is None:
            archive.close()
            raise DataArtifactError("Configured data member could not be opened")
        return archive, stream
    except (tarfile.TarError, KeyError, OSError) as error:
        raise DataArtifactError(
            "Configured data member is absent or unreadable",
            details={"member": member_name},
        ) from error


def _rows(
    archive_path: Path,
    member_name: str,
    schema: RawSchema,
    *,
    hash_output: Any | None = None,
) -> Iterator[tuple[int, bytes, list[str]]]:
    archive, stream = _iter_member_lines(archive_path, member_name)
    try:
        data_index = 0
        for physical_index, raw_line in enumerate(stream):
            if hash_output is not None:
                hash_output.update(raw_line)
            try:
                text = raw_line.decode("utf-8")
                values = next(csv.reader([text], delimiter=schema.delimiter))
            except (UnicodeDecodeError, csv.Error):
                values = []
            if physical_index == 0 and schema.has_header:
                if tuple(values) != schema.fields:
                    raise DataArtifactError(
                        "Raw header does not match the checked-in schema contract",
                        details={"expected": list(schema.fields), "actual": values},
                    )
                continue
            if physical_index == 0 and not schema.has_header and tuple(values) == schema.fields:
                raise DataArtifactError(
                    "Raw data unexpectedly contains a header; review the schema contract"
                )
            yield data_index, raw_line, values
            data_index += 1
    finally:
        stream.close()
        archive.close()


def _infer_time_unit(state: AuditState) -> tuple[float, list[dict[str, object]]]:
    if state.min_click_raw is None or state.max_click_raw is None:
        raise DataArtifactError("No accepted rows remain for timestamp inference")
    interpretations: list[dict[str, object]] = []
    passing: list[float] = []
    for multiplier in TIME_MULTIPLIERS:
        span_days = (state.max_click_raw - state.min_click_raw) * multiplier / SECONDS_PER_DAY
        max_delay_days = (
            None
            if state.max_positive_delay_raw is None
            else state.max_positive_delay_raw * multiplier / SECONDS_PER_DAY
        )
        click_span_passes = 89.0 <= span_days <= 91.0
        delay_passes = max_delay_days is None or 0.0 <= max_delay_days <= MATURITY_DAYS
        passes = click_span_passes and delay_passes
        interpretations.append(
            {
                "seconds_per_raw_unit": multiplier,
                "click_span_days": span_days,
                "max_positive_delay_days": max_delay_days,
                "click_span_passes": click_span_passes,
                "delay_window_passes": delay_passes,
                "passes": passes,
            }
        )
        if passes:
            passing.append(multiplier)
    if len(passing) != 1:
        raise AmbiguousTimeUnitError(
            "Exactly one shared timestamp multiplier must satisfy the 90-day "
            "and 30-day constraints",
            details={"candidates": interpretations, "passing_count": len(passing)},
        )
    return passing[0], interpretations


def _promote_immutable(temporary: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise ConsistencyError(f"Immutable audit output already exists: {destination}") from error
    temporary.unlink()


def inspect_archive(
    config: DataConfig,
    archive_path: Path,
    *,
    manifest_path: Path,
    quarantine_path: Path,
) -> dict[str, Any]:
    """Inspect a reviewed archive using bounded-memory streaming passes."""

    if not _REQUIRED_FIELDS.issubset(config.schema.fields):
        raise ConsistencyError("The schema contract omits fields required by the inspector")
    if manifest_path.exists() or quarantine_path.exists():
        raise ConsistencyError("Inspection outputs are immutable and already exist")

    archive_sha256, archive_bytes = sha256_file(archive_path)
    archive_inspection = inspect_tar_archive(
        archive_path,
        config.archive_limits,
        expected_members=config.dataset.expected_members,
    )
    state = AuditState()
    stats = _field_stats(config.schema)
    parse_failures: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    column_counts: Counter[int] = Counter()
    monotonic_rows: list[int] = []
    sale_delay_consistency = Counter[str]()
    extracted_digest = hashlib.sha256()
    extracted_bytes = next(
        member.size
        for member in archive_inspection.members
        if member.name == config.dataset.data_member
    )

    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, quarantine_name = tempfile.mkstemp(
        prefix=f".{quarantine_path.name}.", dir=quarantine_path.parent
    )
    quarantine_temporary = Path(quarantine_name)
    database_descriptor, database_name = tempfile.mkstemp(
        prefix="latesignal-duplicates-", suffix=".sqlite", dir=quarantine_path.parent
    )
    os.close(database_descriptor)
    database_path = Path(database_name)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("CREATE TABLE rows (fingerprint BLOB PRIMARY KEY)")
        with os.fdopen(descriptor, "w", encoding="utf-8") as quarantine_output:
            for raw_index, raw_line, values in _rows(
                archive_path,
                config.dataset.data_member,
                config.schema,
                hash_output=extracted_digest,
            ):
                state.parsed_rows += 1
                column_counts[len(values)] += 1
                fingerprint = hashlib.sha256(raw_line.rstrip(b"\r\n")).digest()
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO rows(fingerprint) VALUES (?)", (fingerprint,)
                )
                if cursor.rowcount == 0:
                    state.duplicate_rows += 1
                if len(values) == len(config.schema.fields):
                    _update_lexical_stats(values, config.schema, stats)
                parsed, reasons, fields = _parse_row(values, config.schema)
                if parsed is None:
                    state.quarantined_rows += 1
                    for reason in reasons:
                        reason_counts[reason] += 1
                    for field in fields:
                        parse_failures[field] += 1
                    quarantine_output.write(
                        json.dumps(
                            {
                                "raw_row_index": raw_index,
                                "reason_codes": reasons,
                                "fields": fields,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    if "SALE_DELAY_INCONSISTENT" in reasons:
                        sale_delay_consistency["inconsistent"] += 1
                    continue
                sale_delay_consistency["consistent"] += 1
                state.accepted_rows += 1
                if (
                    state.last_click_raw is not None
                    and parsed.click_timestamp < state.last_click_raw
                ):
                    monotonic_rows.append(raw_index)
                state.last_click_raw = parsed.click_timestamp
                state.min_click_raw = (
                    parsed.click_timestamp
                    if state.min_click_raw is None
                    else min(state.min_click_raw, parsed.click_timestamp)
                )
                state.max_click_raw = (
                    parsed.click_timestamp
                    if state.max_click_raw is None
                    else max(state.max_click_raw, parsed.click_timestamp)
                )
                if parsed.sale == 1:
                    state.positive_delay_count += 1
                    state.positive_delay_sum_raw += parsed.delay
                    state.min_positive_delay_raw = (
                        parsed.delay
                        if state.min_positive_delay_raw is None
                        else min(state.min_positive_delay_raw, parsed.delay)
                    )
                    state.max_positive_delay_raw = (
                        parsed.delay
                        if state.max_positive_delay_raw is None
                        else max(state.max_positive_delay_raw, parsed.delay)
                    )
                if state.parsed_rows % 50_000 == 0:
                    connection.commit()
            quarantine_output.flush()
            os.fsync(quarantine_output.fileno())
        connection.commit()

        if state.accepted_rows + state.quarantined_rows != state.parsed_rows:
            raise ConsistencyError("Accepted and quarantined row counts do not reconcile")
        multiplier, interpretations = _infer_time_unit(state)
        assert state.min_click_raw is not None
        t0_seconds = state.min_click_raw * multiplier
        day_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        last_click_seconds = 0.0
        last_positive_reveal_seconds: float | None = None
        last_negative_maturity_seconds: float | None = None
        accepted_second_pass = 0
        for _, _, values in _rows(archive_path, config.dataset.data_member, config.schema):
            parsed, _, _ = _parse_row(values, config.schema)
            if parsed is None:
                continue
            accepted_second_pass += 1
            click_seconds = parsed.click_timestamp * multiplier
            click_day = math.floor((click_seconds - t0_seconds) / SECONDS_PER_DAY)
            day_counts[click_day][0] += 1
            day_counts[click_day][1] += parsed.sale
            last_click_seconds = max(last_click_seconds, click_seconds)
            if parsed.sale == 1:
                reveal = click_seconds + parsed.delay * multiplier
                last_positive_reveal_seconds = (
                    reveal
                    if last_positive_reveal_seconds is None
                    else max(last_positive_reveal_seconds, reveal)
                )
            else:
                maturity = click_seconds + MATURITY_DAYS * SECONDS_PER_DAY
                last_negative_maturity_seconds = (
                    maturity
                    if last_negative_maturity_seconds is None
                    else max(last_negative_maturity_seconds, maturity)
                )
        if accepted_second_pass != state.accepted_rows:
            raise ConsistencyError("Streaming inspection passes accepted different row counts")

        quarantine_sha256, quarantine_bytes = sha256_file(quarantine_temporary)
        delay_mean_raw = (
            None
            if state.positive_delay_count == 0
            else state.positive_delay_sum_raw / state.positive_delay_count
        )
        manifest: dict[str, Any] = {
            "manifest_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "code_version": __version__,
            "config_sha256": config.canonical_sha256,
            "dataset": config.dataset.name,
            "archive": {
                "path": str(archive_path.resolve()),
                "sha256": archive_sha256,
                "bytes": archive_bytes,
                **archive_inspection.as_dict(),
                "member_compression_note": (
                    "Per-member compressed sizes are unavailable in a solid gzip stream; "
                    "the manifest records the exact total expansion ratio."
                ),
            },
            "extracted_data_member": {
                "name": config.dataset.data_member,
                "sha256": extracted_digest.hexdigest(),
                "bytes": extracted_bytes,
            },
            "schema": {
                "delimiter": config.schema.delimiter,
                "has_header": config.schema.has_header,
                "field_count": len(config.schema.fields),
                "field_order": list(config.schema.fields),
                "column_count_frequencies": {
                    str(key): value for key, value in sorted(column_counts.items())
                },
            },
            "rows": {
                "parsed": state.parsed_rows,
                "accepted": state.accepted_rows,
                "quarantined": state.quarantined_rows,
                "reconciled": state.accepted_rows + state.quarantined_rows == state.parsed_rows,
                "duplicate_raw_rows": state.duplicate_rows,
            },
            "quarantine": {
                "path": str(quarantine_path.resolve()),
                "sha256": quarantine_sha256,
                "bytes": quarantine_bytes,
                "reason_counts": dict(sorted(reason_counts.items())),
                "parse_failures_by_field": dict(sorted(parse_failures.items())),
            },
            "time_unit": {
                "candidate_seconds_per_raw_unit": interpretations,
                "selected_seconds_per_raw_unit": multiplier,
                "time_origin_seconds": t0_seconds,
                "day_assignment": (
                    "click_day=floor((click_time_seconds-T0)/86400), with T0 equal to "
                    "the minimum accepted click time and intervals [D(d), D(d+1))"
                ),
            },
            "click_time": {
                "monotonic": not monotonic_rows,
                "monotonic_violation_rows": monotonic_rows,
                "min_raw": state.min_click_raw,
                "max_raw": state.max_click_raw,
                "span_days": (
                    (state.max_click_raw - state.min_click_raw) * multiplier / SECONDS_PER_DAY
                    if state.max_click_raw is not None and state.min_click_raw is not None
                    else None
                ),
            },
            "positive_delay": {
                "count": state.positive_delay_count,
                "min_raw": state.min_positive_delay_raw,
                "max_raw": state.max_positive_delay_raw,
                "mean_raw": delay_mean_raw,
                "min_days": (
                    None
                    if state.min_positive_delay_raw is None
                    else state.min_positive_delay_raw * multiplier / SECONDS_PER_DAY
                ),
                "max_days": (
                    None
                    if state.max_positive_delay_raw is None
                    else state.max_positive_delay_raw * multiplier / SECONDS_PER_DAY
                ),
                "mean_days": (
                    None
                    if delay_mean_raw is None
                    else delay_mean_raw * multiplier / SECONDS_PER_DAY
                ),
            },
            "sale_delay_consistency": dict(sorted(sale_delay_consistency.items())),
            "value_audit_by_field": stats,
            "click_days": [
                {
                    "click_day": day,
                    "rows": counts[0],
                    "positives": counts[1],
                    "conversion_rate": counts[1] / counts[0],
                }
                for day, counts in sorted(day_counts.items())
            ],
            "event_horizon_seconds": {
                "last_click": last_click_seconds,
                "last_positive_reveal": last_positive_reveal_seconds,
                "last_negative_maturity": last_negative_maturity_seconds,
            },
        }
        _promote_immutable(quarantine_temporary, quarantine_path)
        try:
            write_json_atomic(manifest_path, manifest)
        except Exception:
            quarantine_path.unlink(missing_ok=True)
            raise
        return manifest
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)
        quarantine_temporary.unlink(missing_ok=True)


def inspect_locked_archive(
    config: DataConfig,
    data_root: Path,
    *,
    manifest_path: Path,
    quarantine_path: Path,
) -> dict[str, Any]:
    """Resolve and verify the trusted local artifact before inspecting it."""

    lock_path = data_root / "manifests" / "artifact-lock.json"
    if not lock_path.exists():
        raise DataArtifactError(
            "No reviewed artifact lock exists; complete the data fetch review first"
        )
    lock = read_json(lock_path)
    if lock.get("config_sha256") != config.canonical_sha256:
        raise ConsistencyError("The artifact lock does not match the data configuration")
    raw_archive_path = lock.get("archive_path")
    if not isinstance(raw_archive_path, str):
        raise ConsistencyError("The artifact lock does not contain an archive path")
    archive_path = Path(raw_archive_path)
    actual_sha256, actual_bytes = sha256_file(archive_path)
    if actual_sha256 != lock.get("archive_sha256") or actual_bytes != lock.get("archive_bytes"):
        raise DataArtifactError("The archive does not match its reviewed artifact lock")
    return inspect_archive(
        config,
        archive_path,
        manifest_path=manifest_path,
        quarantine_path=quarantine_path,
    )
