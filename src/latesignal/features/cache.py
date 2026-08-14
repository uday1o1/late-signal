"""Content-addressed truth-free feature caches for production experiments."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from latesignal.data.manifests import (
    canonical_json_bytes,
    read_json,
    sha256_file,
    write_json_atomic,
)
from latesignal.data.prepared import PreparedInventory, verify_prepared_inventory
from latesignal.data.schema import (
    CATEGORICAL_CLICK_FIELDS,
    NUMERIC_CLICK_FIELDS,
    OUTCOME_FIELDS,
    TIMING_FIELDS,
)
from latesignal.errors import ConsistencyError
from latesignal.features.hashing import categorical_bucket
from latesignal.features.policy import FeaturePolicy

FeaturePolicyName = Literal["compact", "large"]

_PART = re.compile(r"^parts/click_day=(\d{3})/part-(\d{5})\.parquet$")
_TEMPORARY_PART = re.compile(r"^parts/click_day=\d{3}/\.part-\d{5}\.parquet\.[0-9a-f]{32}\.tmp$")
_HIGH_CARDINALITY_FIELDS = frozenset(
    {
        "audience_id",
        "product_brand",
        "product_id",
        "product_title",
        "partner_id",
        "user_id",
    }
)
_POLICY_BUCKETS: dict[FeaturePolicyName, tuple[int, int]] = {
    "compact": (2**18, 2**12),
    "large": (2**20, 2**14),
}
_SLICE_COLUMNS = (
    "cold_user",
    "cold_product",
    "prior_user_clicks",
    "prior_product_clicks",
    "product_price",
    "device_type",
)


@dataclass(frozen=True, slots=True)
class RuntimeFeaturePolicy:
    name: FeaturePolicyName
    field_seed: int
    high_cardinality_fields: frozenset[str]
    high_cardinality_buckets: int
    other_categorical_buckets: int
    canonical_sha256: str

    def bucket_count(self, field: str) -> int:
        if field not in CATEGORICAL_CLICK_FIELDS:
            raise ConsistencyError(f"Unknown categorical feature: {field}")
        if field in self.high_cardinality_fields:
            return self.high_cardinality_buckets
        return self.other_categorical_buckets


@dataclass(frozen=True, slots=True)
class FeatureCache:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    policy: RuntimeFeaturePolicy
    prepared_manifest_sha256: str
    files: tuple[Path, ...]
    rows: int
    ordered_id_sha256: str
    total_bytes: int


def runtime_feature_policy(
    authored: FeaturePolicy,
    name: FeaturePolicyName,
) -> RuntimeFeaturePolicy:
    """Bind an authored seed and field grouping to one locked selection policy."""

    if authored.high_cardinality_fields != _HIGH_CARDINALITY_FIELDS:
        raise ConsistencyError("Authored high-cardinality grouping does not match the protocol")
    high_buckets, other_buckets = _POLICY_BUCKETS[name]
    payload = {
        "version": 1,
        "name": name,
        "field_seed": authored.field_seed,
        "high_cardinality_fields": sorted(authored.high_cardinality_fields),
        "high_cardinality_buckets": high_buckets,
        "other_categorical_buckets": other_buckets,
    }
    return RuntimeFeaturePolicy(
        name=name,
        field_seed=authored.field_seed,
        high_cardinality_fields=authored.high_cardinality_fields,
        high_cardinality_buckets=high_buckets,
        other_categorical_buckets=other_buckets,
        canonical_sha256=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
    )


def feature_cache_root(
    storage_root: Path,
    *,
    prepared_manifest_sha256: str,
    policy_sha256: str,
) -> Path:
    """Return the only supported cache path for one data and policy identity."""

    return storage_root / "sha256" / prepared_manifest_sha256 / policy_sha256


def _cache_schema(policy: RuntimeFeaturePolicy, prepared_sha256: str) -> pa.Schema:
    fields = [
        pa.field("click_id", pa.binary(32), nullable=False),
        pa.field("click_time_seconds", pa.float64(), nullable=False),
        pa.field("click_day", pa.int16(), nullable=False),
    ]
    fields.extend(
        pa.field(f"{field}_bucket", pa.uint32(), nullable=False)
        for field in sorted(CATEGORICAL_CLICK_FIELDS)
    )
    for field in sorted(NUMERIC_CLICK_FIELDS):
        fields.extend(
            (
                pa.field(f"{field}_value", pa.float32(), nullable=False),
                pa.field(f"{field}_missing", pa.bool_(), nullable=False),
            )
        )
    fields.extend(
        (
            pa.field("cold_user", pa.bool_(), nullable=False),
            pa.field("cold_product", pa.bool_(), nullable=False),
            pa.field("prior_user_clicks", pa.int64(), nullable=False),
            pa.field("prior_product_clicks", pa.int64(), nullable=False),
            pa.field("product_price", pa.float64(), nullable=False),
            pa.field("device_type", pa.string(), nullable=False),
        )
    )
    return pa.schema(
        fields,
        metadata={
            b"latesignal_store": b"truth_free_runtime_features",
            b"prepared_manifest_sha256": prepared_sha256.encode(),
            b"runtime_feature_policy_sha256": policy.canonical_sha256.encode(),
        },
    )


def _required_source_columns() -> tuple[str, ...]:
    columns = ["click_id", "click_time_seconds", "click_day"]
    columns.extend(sorted(CATEGORICAL_CLICK_FIELDS))
    for field in sorted(NUMERIC_CLICK_FIELDS):
        columns.extend((f"{field}_value", f"{field}_missing"))
    columns.extend(_SLICE_COLUMNS)
    return tuple(dict.fromkeys(columns))


def _source_relative(path: Path, inventory: PreparedInventory) -> str:
    try:
        relative = path.relative_to(inventory.root).as_posix()
    except ValueError as error:
        raise ConsistencyError("Prepared feature path escapes its verified root") from error
    if not relative.startswith("features/click_day="):
        raise ConsistencyError("Prepared inventory exposed a non-feature cache input")
    return relative


def _target_relative(source_relative: str) -> str:
    source = PurePosixPath(source_relative)
    if len(source.parts) != 3:
        raise ConsistencyError("Prepared feature partition path is malformed")
    target = PurePosixPath("parts", source.parts[1], source.parts[2])
    if _PART.fullmatch(target.as_posix()) is None:
        raise ConsistencyError("Prepared feature part name is not canonical")
    return target.as_posix()


def _source_table(
    source: Path,
    *,
    policy: RuntimeFeaturePolicy,
    schema: pa.Schema,
) -> pa.Table:
    try:
        parquet = pq.ParquetFile(source)
        source_names = set(parquet.schema_arrow.names)
        forbidden = (OUTCOME_FIELDS | TIMING_FIELDS | {"final_label", "available_at_seconds"}) & (
            source_names - {"click_time_seconds"}
        )
        required = set(_required_source_columns())
        if forbidden or not required.issubset(source_names):
            raise ConsistencyError(
                "Runtime feature cache input violates the click-time boundary",
                details={
                    "forbidden": sorted(forbidden),
                    "missing": sorted(required - source_names),
                },
            )
        table = parquet.read(columns=list(_required_source_columns()))
    except (OSError, pa.ArrowException) as error:
        raise ConsistencyError("Prepared feature part could not be read") from error
    raw_ids = table["click_id"].to_pylist()
    try:
        click_ids = [bytes.fromhex(value) for value in raw_ids]
    except (TypeError, ValueError) as error:
        raise ConsistencyError("Prepared feature part contains an invalid click ID") from error
    if any(len(value) != 32 for value in click_ids):
        raise ConsistencyError("Prepared feature part contains an invalid click ID")
    arrays: list[pa.Array | pa.ChunkedArray] = [
        pa.array(click_ids, type=pa.binary(32)),
        table["click_time_seconds"],
        table["click_day"],
    ]
    for field in sorted(CATEGORICAL_CLICK_FIELDS):
        values = table[field].to_pylist()
        arrays.append(
            pa.array(
                [
                    categorical_bucket(
                        field,
                        value,
                        policy.field_seed,
                        policy.bucket_count(field),
                    )
                    for value in values
                ],
                type=pa.uint32(),
            )
        )
    for field in sorted(NUMERIC_CLICK_FIELDS):
        arrays.extend((table[f"{field}_value"], table[f"{field}_missing"]))
    arrays.extend(table[column] for column in _SLICE_COLUMNS)
    return pa.Table.from_arrays(arrays, schema=schema)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_part(table: pa.Table, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
        )
        with temporary.open("rb") as source:
            os.fsync(source.fileno())
        metadata = pq.read_metadata(temporary)
        if metadata.num_rows != table.num_rows or metadata.num_columns != table.num_columns:
            raise ConsistencyError("Runtime feature cache write failed metadata verification")
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_temporary_parts(root: Path) -> None:
    for path in root.rglob("*.tmp"):
        relative = path.relative_to(root).as_posix()
        if _TEMPORARY_PART.fullmatch(relative) is None:
            continue
        if path.is_symlink() or path.resolve().parent != path.parent.resolve():
            raise ConsistencyError("Refusing redirected runtime feature temporary cleanup")
        path.unlink()


def build_feature_cache(
    prepared_manifest_path: Path,
    *,
    authored_policy: FeaturePolicy,
    policy_name: FeaturePolicyName,
    storage_root: Path,
) -> FeatureCache:
    """Materialize one selected hashing policy without opening eventual truth."""

    inventory = verify_prepared_inventory(prepared_manifest_path)
    policy = runtime_feature_policy(authored_policy, policy_name)
    root = feature_cache_root(
        storage_root,
        prepared_manifest_sha256=inventory.manifest_sha256,
        policy_sha256=policy.canonical_sha256,
    )
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        return verify_feature_cache(
            manifest_path,
            expected_prepared_sha256=inventory.manifest_sha256,
            expected_policy=policy,
        )
    if root.is_symlink():
        raise ConsistencyError("Runtime feature cache root cannot be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    _remove_temporary_parts(root)
    schema = _cache_schema(policy, inventory.manifest_sha256)
    ordered_ids = hashlib.sha256()
    rows = 0
    files: list[dict[str, object]] = []
    feature_files = inventory.feature_files(first_day=0, last_day=90)
    if not feature_files:
        raise ConsistencyError("Prepared inventory contains no feature files")
    for source in feature_files:
        relative = _target_relative(_source_relative(source, inventory))
        target = root.joinpath(*PurePosixPath(relative).parts)
        table = _source_table(source, policy=policy, schema=schema)
        for value in table["click_id"].to_pylist():
            if not isinstance(value, bytes) or len(value) != 32:
                raise ConsistencyError("Runtime feature cache contains an invalid click ID")
            ordered_ids.update(value)
        _write_part(table, target)
        sha256, size = sha256_file(target)
        files.append(
            {
                "path": relative,
                "sha256": sha256,
                "bytes": size,
                "rows": table.num_rows,
            }
        )
        rows += table.num_rows
    expected_rows = inventory.manifest.get("rows", {}).get("features")
    if rows != expected_rows:
        raise ConsistencyError("Runtime feature cache row count does not match preparation")
    payload: dict[str, Any] = {
        "manifest_version": 1,
        "status": "complete",
        "truth_free": True,
        "prepared_manifest_sha256": inventory.manifest_sha256,
        "policy": {
            "name": policy.name,
            "field_seed": policy.field_seed,
            "high_cardinality_fields": sorted(policy.high_cardinality_fields),
            "high_cardinality_buckets": policy.high_cardinality_buckets,
            "other_categorical_buckets": policy.other_categorical_buckets,
            "canonical_sha256": policy.canonical_sha256,
        },
        "rows": rows,
        "ordered_id_sha256": ordered_ids.hexdigest(),
        "files": files,
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    write_json_atomic(manifest_path, payload)
    _fsync_directory(root)
    return verify_feature_cache(
        manifest_path,
        expected_prepared_sha256=inventory.manifest_sha256,
        expected_policy=policy,
    )


def _safe_cache_file(root: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str):
        raise ConsistencyError("Runtime feature cache path is malformed")
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or _PART.fullmatch(relative.as_posix()) is None
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ConsistencyError("Runtime feature cache path escapes its root")
    return root.joinpath(*relative.parts)


def verify_feature_cache(
    manifest_path: Path,
    *,
    expected_prepared_sha256: str,
    expected_policy: RuntimeFeaturePolicy,
) -> FeatureCache:
    """Rehash all runtime feature parts and reject unlisted or redirected artifacts."""

    if manifest_path.is_symlink():
        raise ConsistencyError("Runtime feature cache manifest cannot be a symlink")
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    value = read_json(manifest_path)
    claimed_manifest_sha256 = value.get("manifest_sha256")
    unsigned = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if (
        value.get("manifest_version") != 1
        or value.get("status") != "complete"
        or value.get("truth_free") is not True
        or value.get("prepared_manifest_sha256") != expected_prepared_sha256
        or not isinstance(claimed_manifest_sha256, str)
        or hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != claimed_manifest_sha256
    ):
        raise ConsistencyError("Runtime feature cache manifest identity is invalid")
    raw_policy = value.get("policy")
    if (
        not isinstance(raw_policy, dict)
        or raw_policy.get("canonical_sha256") != expected_policy.canonical_sha256
    ):
        raise ConsistencyError("Runtime feature cache policy identity changed")
    raw_files = value.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ConsistencyError("Runtime feature cache has no file inventory")
    schema = _cache_schema(expected_policy, expected_prepared_sha256)
    paths: list[Path] = []
    total_bytes = 0
    total_rows = 0
    seen: set[str] = set()
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise ConsistencyError("Runtime feature cache inventory entry is malformed")
        path = _safe_cache_file(root, raw.get("path"))
        relative = path.relative_to(root).as_posix()
        if relative in seen or path.is_symlink() or not path.is_file():
            raise ConsistencyError("Runtime feature cache part is missing or redirected")
        seen.add(relative)
        try:
            parquet = pq.ParquetFile(path)
        except (OSError, pa.ArrowException) as error:
            raise ConsistencyError("Runtime feature cache part could not be read") from error
        sha256, size = sha256_file(path)
        expected_rows = raw.get("rows")
        if (
            parquet.schema_arrow != schema
            or sha256 != raw.get("sha256")
            or size != raw.get("bytes")
            or isinstance(expected_rows, bool)
            or not isinstance(expected_rows, int)
            or parquet.metadata.num_rows != expected_rows
        ):
            raise ConsistencyError("Runtime feature cache part identity changed")
        paths.append(path)
        total_bytes += size
        total_rows += expected_rows
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != seen:
        raise ConsistencyError("Runtime feature cache contains unlisted artifacts")
    if total_rows != value.get("rows"):
        raise ConsistencyError("Runtime feature cache manifest row total is inconsistent")
    ordered_id_sha256 = value.get("ordered_id_sha256")
    if not isinstance(ordered_id_sha256, str):
        raise ConsistencyError("Runtime feature cache has no ordered ID digest")
    file_sha256, _ = sha256_file(manifest_path)
    return FeatureCache(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=file_sha256,
        policy=expected_policy,
        prepared_manifest_sha256=expected_prepared_sha256,
        files=tuple(paths),
        rows=total_rows,
        ordered_id_sha256=ordered_id_sha256,
        total_bytes=total_bytes,
    )
