from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import torch

from latesignal.experiments.checkpoint import CheckpointIdentity
from latesignal.experiments.production_final import ProductionFinalPlan
from latesignal.experiments.production_final_controller import ProductionFinalController
from latesignal.features.store import FeatureTensorBatch
from latesignal.models.conversion_mlp import CategoricalSpec
from latesignal.simulator.production_oracle import SECONDS_PER_DAY, ProductionTruthStore

FIELDS = tuple(f"field_{index}" for index in range(17))


class _Features:
    def __init__(self) -> None:
        per_day = 2
        self.prepared_manifest_sha256 = "4" * 64
        self.feature_policy_sha256 = "5" * 64
        self.click_days = np.repeat(np.arange(90, dtype=np.int16), per_day)
        offsets = np.tile(np.asarray([0.0, 10.0]), 90)
        self.click_times = self.click_days.astype(np.float64) * SECONDS_PER_DAY + offsets
        self.click_ids = np.asarray(
            [(index + 1).to_bytes(32, "big") for index in range(self.click_days.size)],
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

    def references_for_ids(self, click_ids: list[bytes]) -> np.ndarray:
        lookup = {bytes(value): index for index, value in enumerate(self.click_ids)}
        return np.asarray([lookup[value] for value in click_ids], dtype=np.int32)


def _truth(features: _Features) -> ProductionTruthStore:
    labels = (np.arange(features.click_days.size) % 3 == 0).astype(np.int8)
    available = features.click_times + np.where(labels == 1, 1.0, 30 * SECONDS_PER_DAY)
    delays = np.where(labels == 1, 1.0 / SECONDS_PER_DAY, np.nan).astype(np.float32)
    refs = np.arange(labels.size, dtype=np.int32)
    order = np.lexsort((refs, available))
    return ProductionTruthStore(
        prepared_manifest_sha256=features.prepared_manifest_sha256,
        final_labels=labels,
        available_at=available,
        conversion_delay_days=delays,
        event_feature_refs=refs[order],
        event_available_at=available[order],
        event_labels=labels[order],
    )


def _plan(
    *,
    study: str = "study_a",
    method: str = "complete_wait",
    scheduler: str = "fixed_daily",
) -> ProductionFinalPlan:
    return ProductionFinalPlan.model_validate(
        {
            "version": 1,
            "phase": "qualification",
            "study": study,
            "run_id": ("study-a-" if study == "study_a" else "study-b-") + "0" * 16,
            "method": method,
            "scheduler": scheduler,
            "seed": 17,
            "wait_days": 1 if method in {"fixed_wait", "es_dfm"} else None,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "dropout": 0.0,
            "gradient_norm_clip": 5.0,
            "initialization_steps": 1,
            "steps_per_credit": 1,
            "credits": 59 if study == "study_a" else 12,
            "batch_size": 4,
            "recent_window_days": 3,
            "reservoir_capacity": 1_000_000,
            "feature_policy": "compact",
            "prediction_batch_size": 8,
            "first_decision_day": 31,
            "last_decision_day": 89,
            "evaluation_first_click_day": 65,
            "evaluation_last_click_day": 89,
            "intermediate_budget_fractions": (0.25, 0.5, 0.75, 1.0),
            "deployable": method != "oracle_reference",
            "ranking_eligible": method != "oracle_reference",
            "device": "cpu",
            "protocol_sha256": "1" * 64,
            "protocol_lock_sha256": "2" * 64,
            "selection_decisions_sha256": "3" * 64,
            "data_manifest_sha256": "4" * 64,
            "feature_policy_sha256": "5" * 64,
        }
    )


def _checkpoint_identity(plan: ProductionFinalPlan) -> CheckpointIdentity:
    return CheckpointIdentity(
        version=1,
        phase="qualification",
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
    plan: ProductionFinalPlan | None = None,
    resume: bool = False,
) -> ProductionFinalController:
    features = _Features()
    selected_plan = _plan() if plan is None else plan
    monitoring = np.zeros(features.click_days.size, dtype=np.bool_)
    monitoring[::2] = True
    return ProductionFinalController(
        plan=selected_plan,
        features=features,
        truth=_truth(features),
        monitoring_mask=monitoring,
        output_root=root,
        checkpoint_identity=_checkpoint_identity(selected_plan),
        resume=resume,
    )


def _prediction_rows(root: Path, relative: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((root / relative).glob("part-*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def test_final_controller_seals_prequential_and_intermediate_evidence(tmp_path: Path) -> None:
    root = tmp_path / "run"
    manifest = _controller(root).run()

    assert manifest["status"] == "complete"
    assert manifest["truth_joined"] is False
    assert manifest["primary_evaluation_mode"] == "prequential"
    assert manifest["credits"] == 59
    assert manifest["core_optimizer_steps"] == 59
    assert manifest["core_optimizer_examples"] == 236
    assert manifest["primary_prediction_rows"] == 50
    assert len(manifest["intermediate_predictions"]) == 4
    assert {item["credits_at_snapshot"] for item in manifest["intermediate_predictions"]} == {
        15,
        30,
        45,
        59,
    }
    primary = _prediction_rows(root, "predictions/primary")
    assert len(primary) == 50
    assert {int(item["click_day"]) for item in primary} == set(range(65, 90))
    assert len({int(item["model_version"]) for item in primary}) > 1
    ordered = {item["ordered_id_sha256"] for item in manifest["intermediate_predictions"]}
    assert len(ordered) == 1


def test_final_controller_resume_matches_at_every_checkpoint_boundary(tmp_path: Path) -> None:
    uninterrupted_root = tmp_path / "uninterrupted"
    uninterrupted = _controller(uninterrupted_root).run()
    resumed_root = tmp_path / "resumed"
    for day in range(31, 90):
        stopped = _controller(resumed_root, resume=day > 31).run(stop_after_decision_day=day)
        assert stopped["status"] == "interrupted_after_checkpoint"
        assert stopped["decision_day"] == day
    sealed = _controller(resumed_root, resume=True).run(stop_after_seal=True)
    assert sealed["status"] == "interrupted_after_checkpoint"
    assert sealed["sealed"] is True

    resumed = _controller(resumed_root, resume=True).run()

    assert (
        resumed["primary_prediction_ledger_sha256"]
        == uninterrupted["primary_prediction_ledger_sha256"]
    )
    assert resumed["exposure_ledger_sha256"] == uninterrupted["exposure_ledger_sha256"]
    assert [item["prediction_ledger_sha256"] for item in resumed["intermediate_predictions"]] == [
        item["prediction_ledger_sha256"] for item in uninterrupted["intermediate_predictions"]
    ]
    assert _prediction_rows(resumed_root, "predictions/primary") == _prediction_rows(
        uninterrupted_root,
        "predictions/primary",
    )


def test_study_b_controller_records_all_daily_monitoring_and_equal_credits(
    tmp_path: Path,
) -> None:
    outcomes: list[dict[str, object]] = []
    for scheduler in (
        "fixed_early",
        "fixed_midpoint",
        "fixed_deadline",
        "calibration_drift",
    ):
        plan = _plan(study="study_b", method="fixed_wait", scheduler=scheduler)
        outcomes.append(_controller(tmp_path / scheduler, plan=plan).run())

    assert {item["credits"] for item in outcomes} == {12}
    assert {item["core_optimizer_examples"] for item in outcomes} == {48}
    assert {len(item["scheduler_audit"]["decisions"]) for item in outcomes} == {59}
    assert {item["monitoring_audit"]["last_decision_day"] for item in outcomes} == {89}


@pytest.mark.parametrize(
    "method",
    [
        "immediate_fake_negative",
        "fixed_wait",
        "dfm",
        "fnw",
        "es_dfm",
        "oracle_reference",
    ],
)
def test_final_controller_initializes_every_production_method(
    tmp_path: Path,
    method: str,
) -> None:
    plan = _plan(method=method)
    controller = _controller(tmp_path / method, plan=plan)

    result = controller.run(stop_after_decision_day=31)

    assert result["status"] == "interrupted_after_checkpoint"
    assert result["credits"] == 1
    assert controller.initialization_model_sha256 is not None
    assert controller.initialization_trainer is None
    if method == "es_dfm":
        assert controller.auxiliary is not None
        assert controller.auxiliary.q_tn.work_units == 1
        assert controller.auxiliary.q_tn.optimizer_steps == 500
