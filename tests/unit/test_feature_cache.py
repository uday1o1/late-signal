from __future__ import annotations

from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch

from latesignal.data.manifests import sha256_file, write_json_atomic
from latesignal.data.schema import CATEGORICAL_CLICK_FIELDS, NUMERIC_CLICK_FIELDS
from latesignal.errors import ConsistencyError
from latesignal.features.cache import (
    build_feature_cache,
    runtime_feature_policy,
    verify_feature_cache,
)
from latesignal.features.hashing import categorical_bucket
from latesignal.features.policy import FeaturePolicy
from latesignal.features.store import RuntimeFeatureStore


def _policy() -> FeaturePolicy:
    return FeaturePolicy(
        version=1,
        field_seed=20260813,
        policy="compact",
        high_cardinality_fields=frozenset(
            {
                "audience_id",
                "product_brand",
                "product_id",
                "product_title",
                "partner_id",
                "user_id",
            }
        ),
        high_cardinality_buckets=2**18,
        other_categorical_buckets=2**12,
        burn_in_last_day=14,
        numeric_lower_quantile=0.01,
        numeric_upper_quantile=0.99,
        canonical_sha256="a" * 64,
    )


def _prepared(root: Path, *, add_truth_column: bool = False) -> Path:
    feature = root / "features" / "click_day=000" / "part-00000.parquet"
    feature.parent.mkdir(parents=True)
    rows: dict[str, list[object]] = {
        "click_id": ["1" * 64, "2" * 62 + "00"],
        "click_time_seconds": [0.0, 1.0],
        "click_day": [0, 0],
        "cold_user": [True, False],
        "cold_product": [True, False],
        "prior_user_clicks": [0, 1],
        "prior_product_clicks": [0, 1],
        "product_price": [10.0, 20.0],
        "device_type": ["desktop", "mobile"],
    }
    for field in sorted(CATEGORICAL_CLICK_FIELDS):
        rows.setdefault(field, [f"{field}-a", f"{field}-b"])
    for field in sorted(NUMERIC_CLICK_FIELDS):
        rows[f"{field}_value"] = [0.1, 0.2]
        rows[f"{field}_missing"] = [False, False]
    if add_truth_column:
        rows["final_label"] = [0, 1]
    pq.write_table(pa.table(rows), feature)
    feature_sha256, feature_bytes = sha256_file(feature)
    manifest = root / "manifests" / "preparation.json"
    write_json_atomic(
        manifest,
        {
            "manifest_version": 1,
            "rows": {"features": 2, "reconciled": True},
            "numeric_statistics": {"fit_click_days": [0, 14]},
            "files": [
                {
                    "path": "features/click_day=000/part-00000.parquet",
                    "sha256": feature_sha256,
                    "bytes": feature_bytes,
                }
            ],
        },
    )
    return manifest


def test_feature_cache_materializes_locked_hash_policies_and_resumes(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path / "prepared")
    policy = _policy()

    compact = build_feature_cache(
        prepared,
        authored_policy=policy,
        policy_name="compact",
        storage_root=tmp_path / "cache",
    )
    large = build_feature_cache(
        prepared,
        authored_policy=policy,
        policy_name="large",
        storage_root=tmp_path / "cache",
    )
    resumed = build_feature_cache(
        prepared,
        authored_policy=policy,
        policy_name="compact",
        storage_root=tmp_path / "cache",
    )

    assert compact.manifest_sha256 == resumed.manifest_sha256
    assert compact.rows == large.rows == 2
    assert compact.root != large.root
    compact_table = pq.ParquetFile(compact.files[0]).read()
    large_table = pq.ParquetFile(large.files[0]).read()
    expected_compact = categorical_bucket("user_id", "user_id-a", policy.field_seed, 2**18)
    expected_large = categorical_bucket("user_id", "user_id-a", policy.field_seed, 2**20)
    assert compact_table["user_id_bucket"][0].as_py() == expected_compact
    assert large_table["user_id_bucket"][0].as_py() == expected_large
    assert compact_table["click_id"][0].as_py() == bytes.fromhex("1" * 64)
    assert "final_label" not in compact_table.column_names
    assert "user_id" not in compact_table.column_names


def test_feature_cache_detects_modified_output(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path / "prepared")
    policy = _policy()
    cache = build_feature_cache(
        prepared,
        authored_policy=policy,
        policy_name="compact",
        storage_root=tmp_path / "cache",
    )
    cache.files[0].write_bytes(b"corrupt")

    with pytest.raises(ConsistencyError, match="could not be read"):
        verify_feature_cache(
            cache.manifest_path,
            expected_prepared_sha256=cache.prepared_manifest_sha256,
            expected_policy=runtime_feature_policy(policy, "compact"),
        )


def test_feature_cache_rejects_truth_bearing_source(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path / "prepared", add_truth_column=True)

    with pytest.raises(ConsistencyError, match="click-time boundary"):
        build_feature_cache(
            prepared,
            authored_policy=_policy(),
            policy_name="compact",
            storage_root=tmp_path / "cache",
        )


def test_runtime_feature_store_serves_stable_references_and_tensors(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path / "prepared")
    policy = _policy()
    cache = build_feature_cache(
        prepared,
        authored_policy=policy,
        policy_name="large",
        storage_root=tmp_path / "cache",
    )

    store = RuntimeFeatureStore(cache)
    references = store.references_for_ids([bytes.fromhex("2" * 62 + "00"), bytes.fromhex("1" * 64)])
    batch = store.tensor_batch(references)

    assert references.tolist() == [1, 0]
    assert store.references_for_day(0).tolist() == [0, 1]
    assert batch.numeric.shape == (2, 4)
    assert tuple(batch.categorical) == tuple(sorted(CATEGORICAL_CLICK_FIELDS))
    assert batch.categorical["user_id"].dtype == torch.int64
    assert store.device_types == ("desktop", "mobile")
