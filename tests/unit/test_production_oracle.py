from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from latesignal.data.manifests import sha256_file, write_json_atomic
from latesignal.errors import ConsistencyError
from latesignal.simulator.production_oracle import (
    SECONDS_PER_DAY,
    ProductionTruthCursor,
    load_production_truth,
)


class _Features:
    def __init__(self, prepared_manifest_sha256: str) -> None:
        self.prepared_manifest_sha256 = prepared_manifest_sha256
        self.click_ids = np.asarray(
            [bytes.fromhex("1" * 64), bytes.fromhex("2" * 64), bytes.fromhex("3" * 64)],
            dtype="V32",
        )
        self.click_times = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
        self.click_days = np.asarray([0, 0, 25], dtype=np.int16)
        self._lookup = {bytes(value): index for index, value in enumerate(self.click_ids)}

    def references_for_ids(self, click_ids: list[bytes]) -> np.ndarray:
        return np.asarray([self._lookup[value] for value in click_ids], dtype=np.int32)


def _write_truth(
    path: Path,
    *,
    click_id: str,
    label: int,
    click_time: float,
    available_at: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("click_id", pa.string(), nullable=False),
            pa.field("final_label", pa.int8(), nullable=False),
            pa.field("click_time_seconds", pa.float64(), nullable=False),
            pa.field("available_at_seconds", pa.float64(), nullable=False),
        ],
        metadata={b"latesignal_store": b"eventual_truth"},
    )
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array([click_id]),
                pa.array([label], type=pa.int8()),
                pa.array([click_time]),
                pa.array([available_at]),
            ],
            schema=schema,
        ),
        path,
    )


def _prepared(root: Path) -> Path:
    files = [
        ("truth/reveal/reveal_day=000/part-00000.parquet", "1" * 64, 1, 0.0, 1.0),
        (
            "truth/maturity/maturity_day=030/part-00000.parquet",
            "2" * 64,
            0,
            1.0,
            1.0 + 30 * SECONDS_PER_DAY,
        ),
        ("truth/reveal/reveal_day=025/part-00000.parquet", "3" * 64, 1, 2.0, 5.0),
    ]
    inventory: list[dict[str, object]] = []
    for relative, click_id, label, click_time, available_at in files:
        path = root / relative
        _write_truth(
            path,
            click_id=click_id,
            label=label,
            click_time=click_time,
            available_at=available_at,
        )
        sha256, size = sha256_file(path)
        inventory.append({"path": relative, "sha256": sha256, "bytes": size})
    manifest = root / "manifests" / "preparation.json"
    write_json_atomic(
        manifest,
        {
            "manifest_version": 1,
            "rows": {"features": 3, "truth": 3, "reconciled": True},
            "numeric_statistics": {"fit_click_days": [0, 14]},
            "files": inventory,
        },
    )
    return manifest


def test_production_truth_cursor_reveals_due_training_period_only_and_resumes(
    tmp_path: Path,
) -> None:
    manifest = _prepared(tmp_path / "prepared")
    manifest_sha256, _ = sha256_file(manifest)
    features = _Features(manifest_sha256)
    store = load_production_truth(manifest, features)
    cursor = store.cursor(features.click_days, first_click_day=0, last_click_day=24)

    first = cursor.reveal_through(5.0)
    state = cursor.state_dict()
    final = cursor.reveal_through(31 * SECONDS_PER_DAY)
    resumed = ProductionTruthCursor(
        store,
        features.click_days,
        first_click_day=0,
        last_click_day=24,
    )
    resumed.load_state_dict(state)

    assert first.feature_refs.tolist() == [0]
    assert final.feature_refs.tolist() == [1]
    assert resumed.reveal_through(31 * SECONDS_PER_DAY).feature_refs.tolist() == [1]
    assert store.final_labels.tolist() == [1, 0, 1]
    assert store.conversion_delay_days[0] == pytest.approx(1.0 / SECONDS_PER_DAY)


def test_production_truth_rejects_data_identity_mismatch(tmp_path: Path) -> None:
    manifest = _prepared(tmp_path / "prepared")

    with pytest.raises(ConsistencyError, match="different prepared identities"):
        load_production_truth(manifest, _Features("f" * 64))
