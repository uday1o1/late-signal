from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from latesignal.experiments.production_qualification import (
    _qualification_plan,
    _rehearse_resume,
)
from latesignal.features.store import FeatureTensorBatch
from latesignal.models.conversion_mlp import CategoricalSpec
from latesignal.simulator.production_oracle import SECONDS_PER_DAY

FIELDS = tuple(f"field_{index}" for index in range(17))


class _RealSchemaFeatures:
    def __init__(self) -> None:
        per_day = 8
        rows = 91 * per_day
        self.prepared_manifest_sha256 = "4" * 64
        self.feature_policy_sha256 = "5" * 64
        self.click_days = np.repeat(np.arange(91, dtype=np.int16), per_day)
        self.click_times = self.click_days.astype(np.float64) * SECONDS_PER_DAY + np.tile(
            np.arange(per_day, dtype=np.float64), 91
        )
        self.click_ids = np.asarray(
            [(index + 1).to_bytes(32, "big") for index in range(rows)],
            dtype="V32",
        )
        values = np.arange(rows, dtype=np.uint32) % 8
        self.categorical = np.repeat(values[:, None], len(FIELDS), axis=1)
        self.numeric = np.zeros((rows, 4), dtype=np.float32)

    @property
    def categorical_specs(self) -> dict[str, CategoricalSpec]:
        return {field: CategoricalSpec(8, 2) for field in FIELDS}

    def references_for_day(self, day: int) -> np.ndarray:
        return np.flatnonzero(self.click_days == day).astype(np.int32)

    def tensor_batch(self, references: np.ndarray) -> FeatureTensorBatch:
        refs = np.asarray(references, dtype=np.int64)
        return FeatureTensorBatch(
            categorical={
                field: torch.from_numpy(self.categorical[refs, column].astype(np.int64))
                for column, field in enumerate(FIELDS)
            },
            numeric=torch.from_numpy(self.numeric[refs]),
        )


def _runtime_identity() -> dict[str, object]:
    return {
        "source_tree_sha256": "6" * 64,
        "dependency_lock_sha256": "7" * 64,
        "git_commit": "8" * 40,
        "runtime_sha256": "9" * 64,
        "git_dirty": False,
    }


def test_qualification_plan_binds_the_selected_feature_policy() -> None:
    plan = _qualification_plan(
        protocol_sha256="1" * 64,
        protocol_lock_sha256="2" * 64,
        data_manifest_sha256="4" * 64,
        feature_policy_sha256="5" * 64,
        feature_policy="large",
        device="cpu",
    )

    assert plan.feature_policy == "large"
    assert plan.feature_policy_sha256 == "5" * 64


def test_real_schema_qualification_checkpoint_resume_is_exact(tmp_path: Path) -> None:
    evidence = _rehearse_resume(
        _RealSchemaFeatures(),  # type: ignore[arg-type]
        protocol_sha256="1" * 64,
        protocol_lock_sha256="2" * 64,
        runtime_identity=_runtime_identity(),
        device_uuid="cpu-test",
        temporary_root=tmp_path,
        feature_policy="large",
        device="cpu",
    )

    assert evidence["status"] == "passed"
    assert evidence["real_schema_rows"] == 728
    assert evidence["interruption_day"] == 60
    assert len(evidence["intermediate_prediction_ledger_sha256"]) == 4
    assert not list(tmp_path.glob("**/checkpoints/*.pt"))
