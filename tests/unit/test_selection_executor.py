from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from latesignal.errors import ConsistencyError
from latesignal.experiments.production_selection import ProductionSelectionPlan
from latesignal.experiments.selection_executor import (
    ProductionSelectionExecutor,
    _safe_remove_child,
)
from latesignal.features.store import FeatureTensorBatch
from latesignal.models.conversion_mlp import CategoricalSpec
from latesignal.simulator.production_oracle import SECONDS_PER_DAY, ProductionTruthStore

FIELDS = tuple(f"field_{index}" for index in range(17))


class _Features:
    def __init__(self, feature_policy_sha256: str) -> None:
        self.prepared_manifest_sha256 = "4" * 64
        self.feature_policy_sha256 = feature_policy_sha256
        self.click_days = np.repeat(np.arange(35, dtype=np.int16), 2)
        offsets = np.tile(np.asarray([0.0, 10.0]), 35)
        self.click_times = self.click_days.astype(np.float64) * SECONDS_PER_DAY + offsets
        self.click_ids = np.asarray(
            [index.to_bytes(32, "big") for index in range(self.click_days.size)],
            dtype="V32",
        )
        self.values = np.arange(self.click_days.size, dtype=np.int64) % 8

    @property
    def categorical_specs(self) -> dict[str, CategoricalSpec]:
        return {field: CategoricalSpec(8, 2) for field in FIELDS}

    def tensor_batch(self, references: np.ndarray) -> FeatureTensorBatch:
        values = torch.from_numpy(self.values[references])
        return FeatureTensorBatch(
            categorical={field: values.clone() for field in FIELDS},
            numeric=torch.zeros((references.size, 4), dtype=torch.float32),
        )

    def references_for_day(self, day: int) -> np.ndarray:
        return np.flatnonzero(self.click_days == day).astype(np.int32)

    def references_for_ids(self, click_ids: list[bytes]) -> np.ndarray:
        lookup = {bytes(value): index for index, value in enumerate(self.click_ids)}
        return np.asarray([lookup[value] for value in click_ids], dtype=np.int32)


def _truth(features: _Features) -> ProductionTruthStore:
    labels = (np.arange(features.click_days.size) % 3 == 0).astype(np.int8)
    available = features.click_times + np.where(labels == 1, 1.0, 30 * SECONDS_PER_DAY)
    delays = np.where(labels == 1, 1.0 / SECONDS_PER_DAY, np.nan).astype(np.float32)
    order = np.lexsort((np.arange(labels.size), available))
    references = np.arange(labels.size, dtype=np.int32)
    return ProductionTruthStore(
        prepared_manifest_sha256=features.prepared_manifest_sha256,
        final_labels=labels,
        available_at=available,
        conversion_delay_days=delays,
        event_feature_refs=references[order],
        event_available_at=available[order],
        event_labels=labels[order],
    )


def _plan() -> ProductionSelectionPlan:
    return ProductionSelectionPlan(
        version=1,
        phase="qualification",
        stage="model",
        run_id="executor-test",
        method="complete_wait",
        seed=17,
        wait_days=None,
        learning_rate=0.001,
        weight_decay=0.0,
        dropout=0.1,
        gradient_norm_clip=5.0,
        initialization_steps=1,
        steps_per_credit=1,
        batch_size=4,
        recent_window_days=3,
        reservoir_capacity=1_000_000,
        prediction_batch_size=2,
        protocol_sha256="1" * 64,
        data_manifest_sha256="4" * 64,
        feature_policy_sha256="2" * 64,
        device="cpu",
    )


def _runtime() -> dict[str, object]:
    return {
        "source_tree_sha256": "5" * 64,
        "dependency_lock_sha256": "6" * 64,
        "git_commit": "7" * 40,
        "runtime_sha256": "8" * 64,
        "git_dirty": False,
    }


def test_executor_compacts_verified_candidate_evidence_and_resumes(tmp_path: Path) -> None:
    compact = _Features("2" * 64)
    large = _Features("3" * 64)
    executor = ProductionSelectionExecutor(
        output_root=tmp_path / "selection",
        features={"compact": compact, "large": large},
        truth=_truth(compact),
        monitoring_mask=np.zeros(compact.click_days.size, dtype=np.bool_),
        runtime_identity=_runtime(),
        device_uuid="cpu-test",
    )
    plan = _plan()

    prepared = executor.prepare(plan, feature_policy="compact")
    candidate_root = tmp_path / "selection" / "model" / "candidates" / plan.run_id

    assert prepared.status == "complete"
    assert not (candidate_root / "checkpoints").exists()
    assert not (candidate_root / "exposures").exists()
    assert (candidate_root / "predictions" / "seal.json").is_file()
    assert (candidate_root / "training-retention.json").is_file()

    candidate = executor.score(prepared)
    assert candidate.status == "complete"
    assert not (candidate_root / "predictions").exists()
    assert (candidate_root / "selection-evaluation.json").is_file()
    assert (candidate_root / "candidate-result.json").is_file()

    repeated_prepared = executor.prepare(plan, feature_policy="compact")
    repeated_candidate = executor.score(repeated_prepared)
    assert repeated_prepared == prepared
    assert repeated_candidate == candidate


def test_selection_retention_refuses_a_redirected_prune_target(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate"
    external = tmp_path / "external"
    candidate_root.mkdir()
    external.mkdir()
    marker = external / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    (candidate_root / "predictions").symlink_to(external, target_is_directory=True)

    with pytest.raises(ConsistencyError, match="redirected"):
        _safe_remove_child(candidate_root, "predictions")

    assert marker.read_text(encoding="utf-8") == "keep"
