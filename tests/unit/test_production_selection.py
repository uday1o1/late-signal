from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch

from latesignal.experiments.checkpoint import CheckpointIdentity, RollingCheckpointStore
from latesignal.experiments.production_selection import (
    ProductionSelectionController,
    ProductionSelectionPlan,
    SelectionMethod,
)
from latesignal.features.store import FeatureTensorBatch
from latesignal.models.conversion_mlp import CategoricalSpec
from latesignal.simulator.production_oracle import SECONDS_PER_DAY, ProductionTruthStore

FIELDS = tuple(f"field_{index}" for index in range(17))


class _Features:
    def __init__(self) -> None:
        per_day = 2
        self.prepared_manifest_sha256 = "4" * 64
        self.click_days = np.repeat(np.arange(35, dtype=np.int16), per_day)
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
        numeric = torch.zeros((references.size, 4), dtype=torch.float32)
        numeric[:, 0] = values.float() / 8.0
        return FeatureTensorBatch(
            categorical={field: values.clone() for field in FIELDS},
            numeric=numeric,
        )

    def references_for_day(self, day: int) -> np.ndarray:
        return np.flatnonzero(self.click_days == day).astype(np.int32)


def _truth(features: _Features) -> ProductionTruthStore:
    labels = (np.arange(features.click_days.size) % 3 == 0).astype(np.int8)
    available = features.click_times + np.where(labels == 1, 1.0, 30 * SECONDS_PER_DAY)
    delays = np.where(labels == 1, 1.0 / SECONDS_PER_DAY, np.nan).astype(np.float32)
    order = np.lexsort((np.arange(labels.size), available))
    refs = np.arange(labels.size, dtype=np.int32)
    return ProductionTruthStore(
        prepared_manifest_sha256=features.prepared_manifest_sha256,
        final_labels=labels,
        available_at=available,
        conversion_delay_days=delays,
        event_feature_refs=refs[order],
        event_available_at=available[order],
        event_labels=labels[order],
    )


def _plan(*, method: SelectionMethod = "complete_wait") -> ProductionSelectionPlan:
    return ProductionSelectionPlan(
        version=1,
        phase="qualification",
        stage="delayed" if method in {"fixed_wait", "es_dfm"} else "model",
        run_id="model-candidate-test",
        method=method,
        seed=17,
        wait_days=1 if method in {"fixed_wait", "es_dfm"} else None,
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
        feature_policy_sha256="5" * 64,
        device="cpu",
    )


def _checkpoint_identity(plan: ProductionSelectionPlan) -> CheckpointIdentity:
    return CheckpointIdentity(
        version=1,
        phase=plan.phase,
        run_id=plan.run_id,
        config_sha256=plan.canonical_sha256,
        protocol_sha256=plan.protocol_sha256,
        data_manifest_sha256=plan.data_manifest_sha256,
        feature_policy_sha256=plan.feature_policy_sha256,
        source_tree_sha256="6" * 64,
        dependency_lock_sha256="7" * 64,
        git_commit="8" * 40,
        environment_sha256="9" * 64,
        device_uuid="cpu-test",
    )


def _controller(
    root: Path,
    *,
    resume: bool = False,
    monitoring_mask: np.ndarray | None = None,
) -> ProductionSelectionController:
    features = _Features()
    plan = _plan()
    if monitoring_mask is None:
        monitoring_mask = np.zeros(features.click_days.size, dtype=np.bool_)
    return ProductionSelectionController(
        plan=plan,
        features=features,
        truth=_truth(features),
        monitoring_mask=monitoring_mask,
        output_root=root,
        checkpoint_identity=_checkpoint_identity(plan),
        resume=resume,
    )


def _prediction_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((root / "predictions").glob("part-*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def test_selection_controller_seals_retrospective_predictions_before_truth(tmp_path: Path) -> None:
    manifest = _controller(tmp_path / "run").run()

    assert manifest["status"] == "complete"
    assert manifest["selection_mode"] == "retrospective_chronological"
    assert manifest["truth_joined"] is False
    assert manifest["training_click_days"] == [0, 24]
    assert manifest["scoring_click_days"] == [25, 34]
    assert manifest["credit_days"] == [55, 64]
    assert manifest["credits"] == 10
    assert manifest["core_optimizer_steps"] == 10
    rows = _prediction_rows(tmp_path / "run")
    assert len(rows) == 20
    assert {int(row["click_day"]) for row in rows} == set(range(25, 35))
    assert {int(row["model_version"]) for row in rows} == {10}


def test_selection_controller_resume_matches_uninterrupted_ledgers(tmp_path: Path) -> None:
    uninterrupted = _controller(tmp_path / "uninterrupted").run()
    scenarios = (
        ("after-initialization", True, None),
        ("after-credit-3", False, 3),
        ("after-credit-10", False, 10),
    )
    for name, stop_after_initialization, stop_after_credits in scenarios:
        root = tmp_path / name
        interrupted = _controller(root).run(
            stop_after_initialization=stop_after_initialization,
            stop_after_credits=stop_after_credits,
        )
        assert interrupted["status"] == "interrupted_after_checkpoint"

        resumed = _controller(root, resume=True).run()

        assert resumed["prediction_ledger_sha256"] == uninterrupted["prediction_ledger_sha256"]
        assert resumed["exposure_ledger_sha256"] == uninterrupted["exposure_ledger_sha256"]
        assert _prediction_rows(root) == _prediction_rows(tmp_path / "uninterrupted")


def test_selection_controller_never_exposes_monitoring_records(tmp_path: Path) -> None:
    monitoring = np.zeros(70, dtype=np.bool_)
    monitoring[::2] = True

    _controller(tmp_path / "run", monitoring_mask=monitoring).run()

    exposed: list[int] = []
    for path in sorted((tmp_path / "run" / "exposures").glob("credit-*.parquet")):
        keys = pq.read_table(path, columns=["record_key"])["record_key"].to_numpy()
        exposed.extend((keys >> np.uint64(8)).astype(np.int64).tolist())
    assert exposed
    assert not monitoring[np.asarray(exposed)].any()


def test_esdfm_selection_initializes_auxiliaries_on_day_zero_before_core_credit(
    tmp_path: Path,
) -> None:
    features = _Features()
    plan = _plan(method="es_dfm")
    controller = ProductionSelectionController(
        plan=plan,
        features=features,
        truth=_truth(features),
        monitoring_mask=np.zeros(features.click_days.size, dtype=np.bool_),
        output_root=tmp_path / "run",
        checkpoint_identity=_checkpoint_identity(plan),
    )

    result = controller.run(stop_after_credits=1)

    assert result["status"] == "interrupted_after_checkpoint"
    assert controller.auxiliary is not None
    assert controller.auxiliary.q_tn.work_units == 2
    assert controller.auxiliary.q_tn.optimizer_steps == 600
    assert controller.initialization_trainer is None
    assert controller.q_tn_initialization_store is None
    assert controller.q_dp_initialization_store is None
    auxiliary = controller.compute["auxiliary"]
    assert isinstance(auxiliary, list)
    assert auxiliary[0]["main_credit_id"] is None
    assert auxiliary[0]["steps"] == 1_000
    checkpoint = RollingCheckpointStore(tmp_path / "run" / "checkpoints").load_latest(
        _checkpoint_identity(plan)
    )
    assert set(checkpoint.state["model"]) == {"main", "q_tn", "q_dp"}
    assert set(checkpoint.state["method"]) == {
        "main_store",
        "main",
        "q_tn_store",
        "q_dp_store",
    }


def test_publication_selection_plan_rejects_qualification_shortcuts() -> None:
    value = _plan().model_dump(mode="json")
    value["phase"] = "selection"

    with pytest.raises(ValueError, match="authored training grid"):
        ProductionSelectionPlan.model_validate(value)
