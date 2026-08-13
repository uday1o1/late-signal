"""Bounded-memory preparation into physically isolated Parquet stores."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from latesignal import __version__
from latesignal.data.config import DataConfig
from latesignal.data.inspect import (
    MATURITY_DAYS,
    SECONDS_PER_DAY,
    iter_raw_rows,
    parse_raw_row,
)
from latesignal.data.manifests import read_json, sha256_file, write_json_atomic
from latesignal.data.schema import CATEGORICAL_CLICK_FIELDS, NUMERIC_CLICK_FIELDS
from latesignal.errors import ConsistencyError, DataArtifactError
from latesignal.features.hashing import categorical_bucket, click_id
from latesignal.features.numeric import NumericStatistic, transform_numeric
from latesignal.features.online_history import OnlineHistory
from latesignal.features.policy import FeaturePolicy


def _scan_accepted(path: Path, config: DataConfig) -> pl.LazyFrame:
    schema: dict[str, Any] = {"raw_row_index": pl.UInt64}
    schema.update({field: pl.String for field in config.schema.fields})
    return pl.scan_csv(
        path,
        separator=config.schema.delimiter,
        has_header=True,
        schema=schema,
        infer_schema=False,
        ignore_errors=False,
        low_memory=True,
        rechunk=False,
        truncate_ragged_lines=False,
    )


def _numeric_log_expression(field: str) -> pl.Expr:
    numeric = pl.col(field).cast(pl.Float64, strict=True)
    return pl.when(numeric >= 0.0).then(numeric.log1p()).otherwise(None)


def fit_numeric_statistics(
    accepted_path: Path,
    config: DataConfig,
    policy: FeaturePolicy,
    *,
    seconds_per_raw_unit: float,
    time_origin_seconds: float,
) -> dict[str, NumericStatistic]:
    """Fit clipping and standardization only on authored burn-in click days."""

    click_seconds = pl.col("click_timestamp").cast(pl.Float64) * seconds_per_raw_unit
    lazy = _scan_accepted(accepted_path, config).with_columns(
        ((click_seconds - time_origin_seconds) / SECONDS_PER_DAY)
        .floor()
        .cast(pl.Int16)
        .alias("click_day"),
        *[
            _numeric_log_expression(field).alias(f"{field}_log")
            for field in sorted(NUMERIC_CLICK_FIELDS)
        ],
    )
    burn_in = lazy.filter(pl.col("click_day") <= policy.burn_in_last_day)
    quantile_expressions: list[pl.Expr] = []
    for field in sorted(NUMERIC_CLICK_FIELDS):
        quantile_expressions.extend(
            [
                pl.col(f"{field}_log")
                .quantile(policy.numeric_lower_quantile)
                .alias(f"{field}_lower"),
                pl.col(f"{field}_log")
                .quantile(policy.numeric_upper_quantile)
                .alias(f"{field}_upper"),
            ]
        )
    quantiles = burn_in.select(quantile_expressions).collect(engine="streaming").row(0, named=True)
    bounds: dict[str, tuple[float, float]] = {}
    for field in sorted(NUMERIC_CLICK_FIELDS):
        lower = quantiles[f"{field}_lower"]
        upper = quantiles[f"{field}_upper"]
        if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
            raise DataArtifactError(f"Burn-in has no valid values for {field}")
        bounds[field] = (float(lower), float(upper))

    moment_expressions: list[pl.Expr] = []
    for field, (lower, upper) in bounds.items():
        clipped = pl.col(f"{field}_log").clip(lower, upper)
        moment_expressions.extend(
            [
                clipped.mean().alias(f"{field}_mean"),
                clipped.std(ddof=0).alias(f"{field}_scale"),
            ]
        )
    moments = burn_in.select(moment_expressions).collect(engine="streaming").row(0, named=True)
    result: dict[str, NumericStatistic] = {}
    for field, (lower, upper) in bounds.items():
        mean = moments[f"{field}_mean"]
        scale = moments[f"{field}_scale"]
        if not isinstance(mean, (int, float)):
            raise DataArtifactError(f"Burn-in mean is unavailable for {field}")
        parsed_scale = float(scale) if isinstance(scale, (int, float)) else 0.0
        if not math.isfinite(parsed_scale) or parsed_scale <= 0.0:
            parsed_scale = 1.0
        result[field] = NumericStatistic(lower, upper, float(mean), parsed_scale)
    return result


def _feature_schema(config: DataConfig, policy: FeaturePolicy, inspection_sha256: str) -> pa.Schema:
    categorical = [field for field in config.schema.fields if field in CATEGORICAL_CLICK_FIELDS]
    numeric = [field for field in config.schema.fields if field in NUMERIC_CLICK_FIELDS]
    fields = [
        pa.field("click_id", pa.string(), nullable=False),
        pa.field("click_time_seconds", pa.float64(), nullable=False),
        pa.field("click_day", pa.int16(), nullable=False),
    ]
    fields.extend(pa.field(field, pa.string(), nullable=False) for field in categorical)
    fields.extend(pa.field(field, pa.float64(), nullable=False) for field in numeric)
    fields.extend(pa.field(f"{field}_bucket", pa.uint32(), nullable=False) for field in categorical)
    for field in numeric:
        fields.extend(
            [
                pa.field(f"{field}_value", pa.float32(), nullable=False),
                pa.field(f"{field}_missing", pa.bool_(), nullable=False),
            ]
        )
    fields.extend(
        [
            pa.field("cold_user", pa.bool_(), nullable=False),
            pa.field("cold_product", pa.bool_(), nullable=False),
            pa.field("prior_user_clicks", pa.int64(), nullable=False),
            pa.field("prior_product_clicks", pa.int64(), nullable=False),
        ]
    )
    return pa.schema(
        fields,
        metadata={
            b"latesignal_store": b"click_time_features",
            b"feature_policy_sha256": policy.canonical_sha256.encode(),
            b"inspection_sha256": inspection_sha256.encode(),
        },
    )


def _truth_schema(inspection_sha256: str) -> pa.Schema:
    return pa.schema(
        [
            pa.field("click_id", pa.string(), nullable=False),
            pa.field("final_label", pa.int8(), nullable=False),
            pa.field("click_time_seconds", pa.float64(), nullable=False),
            pa.field("available_at_seconds", pa.float64(), nullable=False),
        ],
        metadata={
            b"latesignal_store": b"eventual_truth",
            b"inspection_sha256": inspection_sha256.encode(),
        },
    )


def _write_partition(
    root: Path,
    partition_name: str,
    day: int,
    part: int,
    rows: list[dict[str, object]],
    schema: pa.Schema,
) -> Path:
    partition = root / f"{partition_name}={day:03d}"
    partition.mkdir(parents=True, exist_ok=True)
    path = partition / f"part-{part:05d}.parquet"
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        version="2.6",
        use_dictionary=True,
        write_statistics=True,
        store_schema=True,
    )
    return path


def _sanitize_accepted_rows(
    archive_path: Path,
    sanitized_path: Path,
    config: DataConfig,
    expected_accepted: int,
    expected_quarantined: int,
) -> tuple[int, int]:
    accepted = 0
    quarantined = 0
    with sanitized_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(
            output,
            delimiter=config.schema.delimiter,
            lineterminator="\n",
        )
        writer.writerow(("raw_row_index", *config.schema.fields))
        for raw_index, _, values in iter_raw_rows(
            archive_path, config.dataset.data_member, config.schema
        ):
            parsed, _, _ = parse_raw_row(values, config.schema)
            if parsed is None:
                quarantined += 1
                continue
            accepted += 1
            writer.writerow((raw_index, *values))
        output.flush()
        os.fsync(output.fileno())
    if accepted != expected_accepted or quarantined != expected_quarantined:
        raise ConsistencyError(
            "Preparation row classification does not match inspection",
            details={
                "expected_accepted": expected_accepted,
                "actual_accepted": accepted,
                "expected_quarantined": expected_quarantined,
                "actual_quarantined": quarantined,
            },
        )
    return accepted, quarantined


def _write_quarantine_parquet(source: Path, destination: Path) -> int:
    schema = pa.schema(
        [
            pa.field("raw_row_index", pa.int64(), nullable=False),
            pa.field("reason_codes", pa.list_(pa.string()), nullable=False),
            pa.field("fields", pa.list_(pa.string()), nullable=False),
        ],
        metadata={b"latesignal_store": b"quarantine"},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    count = 0
    with pq.ParquetWriter(destination, schema, compression="zstd") as writer:
        with source.open(encoding="utf-8") as input_file:
            for line in input_file:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ConsistencyError("Quarantine JSONL contains a non-object")
                rows.append(value)
                count += 1
                if len(rows) >= 10_000:
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                    rows.clear()
        if rows or count == 0:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    return count


def _file_inventory(root: Path) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for path in sorted(root.rglob("*.parquet")):
        sha256, size = sha256_file(path)
        inventory.append({"path": str(path.relative_to(root)), "sha256": sha256, "bytes": size})
    return inventory


def prepare_data(
    config: DataConfig,
    policy: FeaturePolicy,
    *,
    data_root: Path,
    inspection_path: Path,
    output_root: Path,
    batch_rows: int = 65_536,
) -> dict[str, Any]:
    """Prepare reviewed rows through bounded Polars batches and explicit Arrow schemas."""

    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    inspection = read_json(inspection_path)
    lock = read_json(data_root / "manifests" / "artifact-lock.json")
    if inspection.get("config_sha256") != config.canonical_sha256:
        raise ConsistencyError("Inspection manifest does not match the data configuration")
    archive_info = inspection.get("archive")
    extracted_info = inspection.get("extracted_data_member")
    row_info = inspection.get("rows")
    time_info = inspection.get("time_unit")
    click_info = inspection.get("click_time")
    quarantine_info = inspection.get("quarantine")
    if not all(
        isinstance(value, dict)
        for value in (
            archive_info,
            extracted_info,
            row_info,
            time_info,
            click_info,
            quarantine_info,
        )
    ):
        raise ConsistencyError("Inspection manifest is missing required sections")
    assert isinstance(archive_info, dict)
    assert isinstance(extracted_info, dict)
    assert isinstance(row_info, dict)
    assert isinstance(time_info, dict)
    assert isinstance(click_info, dict)
    assert isinstance(quarantine_info, dict)
    if click_info.get("monotonic") is not True:
        raise DataArtifactError("Preparation requires monotonic click order for past-only history")
    archive_path_raw = lock.get("archive_path")
    if not isinstance(archive_path_raw, str):
        raise ConsistencyError("Artifact lock has no archive path")
    archive_path = Path(archive_path_raw)
    archive_sha256, archive_bytes = sha256_file(archive_path)
    if (
        archive_sha256 != lock.get("archive_sha256")
        or archive_bytes != lock.get("archive_bytes")
        or archive_sha256 != archive_info.get("sha256")
    ):
        raise DataArtifactError("Archive identity changed after inspection")
    raw_file_sha256 = extracted_info.get("sha256")
    if not isinstance(raw_file_sha256, str):
        raise ConsistencyError("Inspection manifest has no extracted file hash")
    accepted_expected = row_info.get("accepted")
    quarantined_expected = row_info.get("quarantined")
    multiplier = time_info.get("selected_seconds_per_raw_unit")
    time_origin = time_info.get("time_origin_seconds")
    quarantine_path_raw = quarantine_info.get("path")
    if (
        isinstance(accepted_expected, bool)
        or not isinstance(accepted_expected, int)
        or isinstance(quarantined_expected, bool)
        or not isinstance(quarantined_expected, int)
        or not isinstance(multiplier, (int, float))
        or not isinstance(time_origin, (int, float))
        or not isinstance(quarantine_path_raw, str)
    ):
        raise ConsistencyError("Inspection counts, times, or quarantine path are malformed")

    targets = [
        output_root / "features",
        output_root / "truth",
        output_root / "quarantine" / "rejected.parquet",
        output_root / "manifests" / "preparation.json",
    ]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise ConsistencyError("Prepared outputs already exist", details={"paths": existing})
    output_root.mkdir(parents=True, exist_ok=True)
    stage = output_root / f".prepare-{uuid.uuid4().hex}"
    stage.mkdir()
    promoted: list[Path] = []
    try:
        sanitized_path = stage / "accepted.tsv"
        _sanitize_accepted_rows(
            archive_path,
            sanitized_path,
            config,
            accepted_expected,
            quarantined_expected,
        )
        statistics = fit_numeric_statistics(
            sanitized_path,
            config,
            policy,
            seconds_per_raw_unit=float(multiplier),
            time_origin_seconds=float(time_origin),
        )
        inspection_sha256, _ = sha256_file(inspection_path)
        feature_schema = _feature_schema(config, policy, inspection_sha256)
        truth_schema = _truth_schema(inspection_sha256)
        feature_root = stage / "features"
        reveal_root = stage / "truth" / "reveal"
        maturity_root = stage / "truth" / "maturity"
        categorical = [field for field in config.schema.fields if field in CATEGORICAL_CLICK_FIELDS]
        numeric = [field for field in config.schema.fields if field in NUMERIC_CLICK_FIELDS]
        history = OnlineHistory()
        feature_count = 0
        truth_count = 0
        last_click_seconds: float | None = None
        for part, batch in enumerate(
            _scan_accepted(sanitized_path, config).collect_batches(
                chunk_size=batch_rows,
                maintain_order=True,
                engine="streaming",
            )
        ):
            feature_groups: dict[int, list[dict[str, object]]] = defaultdict(list)
            reveal_groups: dict[int, list[dict[str, object]]] = defaultdict(list)
            maturity_groups: dict[int, list[dict[str, object]]] = defaultdict(list)
            for row in batch.iter_rows(named=True):
                raw_index = int(row["raw_row_index"])
                click_seconds = float(row["click_timestamp"]) * float(multiplier)
                if last_click_seconds is not None and click_seconds < last_click_seconds:
                    raise ConsistencyError("Accepted rows are not monotonic during preparation")
                last_click_seconds = click_seconds
                click_day = math.floor((click_seconds - float(time_origin)) / SECONDS_PER_DAY)
                identifier = click_id(raw_file_sha256, raw_index)
                history_values = history.observe(str(row["user_id"]), str(row["product_id"]))
                feature_row: dict[str, object] = {
                    "click_id": identifier,
                    "click_time_seconds": click_seconds,
                    "click_day": click_day,
                    **{field: str(row[field]) for field in categorical},
                    **{field: float(row[field]) for field in numeric},
                    **history_values,
                }
                for field in categorical:
                    feature_row[f"{field}_bucket"] = categorical_bucket(
                        field,
                        str(row[field]),
                        policy.field_seed,
                        policy.bucket_count(field),
                    )
                for field in numeric:
                    value, missing = transform_numeric(str(row[field]), statistics[field])
                    feature_row[f"{field}_value"] = value
                    feature_row[f"{field}_missing"] = missing
                feature_groups[click_day].append(feature_row)

                sale = int(row["Sale"])
                delay = float(row["time_delay_for_conversion"])
                available_at = (
                    click_seconds + delay * float(multiplier)
                    if sale == 1
                    else click_seconds + MATURITY_DAYS * SECONDS_PER_DAY
                )
                availability_day = math.floor((available_at - float(time_origin)) / SECONDS_PER_DAY)
                truth_row: dict[str, object] = {
                    "click_id": identifier,
                    "final_label": sale,
                    "click_time_seconds": click_seconds,
                    "available_at_seconds": available_at,
                }
                target = reveal_groups if sale == 1 else maturity_groups
                target[availability_day].append(truth_row)
                feature_count += 1
                truth_count += 1
            for day, rows in feature_groups.items():
                _write_partition(feature_root, "click_day", day, part, rows, feature_schema)
            for day, rows in reveal_groups.items():
                _write_partition(reveal_root, "reveal_day", day, part, rows, truth_schema)
            for day, rows in maturity_groups.items():
                _write_partition(maturity_root, "maturity_day", day, part, rows, truth_schema)
        if feature_count != accepted_expected or truth_count != accepted_expected:
            raise ConsistencyError("Prepared feature and truth counts do not reconcile")

        quarantine_destination = stage / "quarantine" / "rejected.parquet"
        quarantine_count = _write_quarantine_parquet(
            Path(quarantine_path_raw), quarantine_destination
        )
        if quarantine_count != quarantined_expected:
            raise ConsistencyError("Prepared quarantine count does not match inspection")
        inventory = _file_inventory(stage)
        manifest: dict[str, Any] = {
            "manifest_version": 1,
            "code_version": __version__,
            "config_sha256": config.canonical_sha256,
            "feature_policy_sha256": policy.canonical_sha256,
            "inspection_sha256": inspection_sha256,
            "source": {
                "archive_sha256": archive_sha256,
                "raw_file_sha256": raw_file_sha256,
            },
            "rows": {
                "inspection_accepted": accepted_expected,
                "inspection_quarantined": quarantined_expected,
                "features": feature_count,
                "truth": truth_count,
                "quarantine": quarantine_count,
                "reconciled": feature_count == truth_count == accepted_expected,
            },
            "streaming": {
                "engine": "polars",
                "batch_rows": batch_rows,
                "source_materialized_in_memory": False,
            },
            "numeric_statistics": {
                "fit_click_days": [0, policy.burn_in_last_day],
                "fields": {field: value.as_dict() for field, value in statistics.items()},
            },
            "schemas": {
                "features": str(feature_schema),
                "truth": str(truth_schema),
            },
            "files": inventory,
        }

        feature_target = output_root / "features"
        truth_target = output_root / "truth"
        quarantine_target = output_root / "quarantine" / "rejected.parquet"
        quarantine_target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(feature_root, feature_target)
        promoted.append(feature_target)
        os.replace(stage / "truth", truth_target)
        promoted.append(truth_target)
        os.replace(quarantine_destination, quarantine_target)
        promoted.append(quarantine_target)
        write_json_atomic(output_root / "manifests" / "preparation.json", manifest)
        return manifest
    except Exception:
        for path in reversed(promoted):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
