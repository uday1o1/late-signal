from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np

from latesignal.experiments.production_final import FinalPlanInputs, ProductionFinalPlan
from latesignal.experiments.production_offline import (
    ProductionOfflineExecutor,
    ProductionOfflinePlan,
    _split,
    offline_reference_plans,
    run_offline_references,
)
from latesignal.features.store import RuntimeFeatureStore
from latesignal.simulator.production_oracle import ProductionTruthStore


class _Policy:
    def bucket_count(self, field: str) -> int:
        assert field == "device_type"
        return 8


class _Features:
    def __init__(self) -> None:
        rows = 91 * 2
        self.click_days = np.repeat(np.arange(91, dtype=np.int16), 2)
        self.click_times = self.click_days.astype(np.float64) * 86_400 + np.tile(
            np.asarray([0.0, 100.0]), 91
        )
        self.click_ids = np.asarray(
            [hashlib.sha256(f"offline-{index}".encode()).digest() for index in range(rows)],
            dtype="V32",
        )
        self.categorical_fields = ("device_type",)
        self.numeric_fields = ("product_price",)
        self.categorical = (np.arange(rows, dtype=np.uint32) % 8).reshape(rows, 1)
        values = np.linspace(-1.0, 1.0, rows, dtype=np.float32)
        self.numeric = np.column_stack((values, np.zeros(rows, dtype=np.float32)))
        self.prior_user_clicks = np.arange(rows, dtype=np.int64) % 7
        self.prior_product_clicks = np.arange(rows, dtype=np.int64) % 5
        self.cold_user = self.prior_user_clicks == 0
        self.cold_product = self.prior_product_clicks == 0
        self.cache = SimpleNamespace(policy=_Policy())
        self.prepared_manifest_sha256 = "4" * 64
        self.feature_policy_sha256 = "5" * 64


def _truth(features: _Features) -> ProductionTruthStore:
    labels = (np.arange(features.click_days.size) % 2).astype(np.int8)
    available = features.click_times + np.where(labels == 1, 86_400.0, 30 * 86_400.0)
    delay = np.where(labels == 1, 1.0, np.nan).astype(np.float32)
    order = np.lexsort((np.arange(labels.size), available))
    return ProductionTruthStore(
        prepared_manifest_sha256=features.prepared_manifest_sha256,
        final_labels=labels,
        available_at=available,
        conversion_delay_days=delay,
        event_feature_refs=order.astype(np.int32),
        event_available_at=available[order],
        event_labels=labels[order],
    )


def _common_plan() -> ProductionFinalPlan:
    return ProductionFinalPlan.model_validate(
        {
            "version": 1,
            "phase": "qualification",
            "study": "study_a",
            "run_id": "study-a-" + "0" * 16,
            "method": "complete_wait",
            "scheduler": "fixed_daily",
            "seed": 17,
            "wait_days": None,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "dropout": 0.0,
            "gradient_norm_clip": 5.0,
            "initialization_steps": 1,
            "steps_per_credit": 1,
            "credits": 59,
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
            "deployable": True,
            "ranking_eligible": True,
            "device": "cpu",
            "protocol_sha256": "1" * 64,
            "protocol_lock_sha256": "2" * 64,
            "selection_decisions_sha256": "3" * 64,
            "data_manifest_sha256": "4" * 64,
            "feature_policy_sha256": "5" * 64,
        }
    )


def _plans() -> tuple[ProductionOfflinePlan, ...]:
    common = _common_plan()
    plans: list[ProductionOfflinePlan] = []
    for name in ("mature_logistic_regression", "mature_lightgbm"):
        for seed in (17, 41, 73):
            plans.append(
                ProductionOfflinePlan.model_validate(
                    {
                        "version": 1,
                        "phase": "final",
                        "run_id": f"offline-{len(plans):016x}",
                        "name": name,
                        "seed": seed,
                        "training_first_click_day": 0,
                        "training_last_click_day": 34,
                        "training_cutoff_day": 65,
                        "evaluation_first_click_day": 65,
                        "evaluation_last_click_day": 89,
                        "monitoring_excluded": True,
                        "ranking_eligible": False,
                        "protocol_sha256": common.protocol_sha256,
                        "protocol_lock_sha256": common.protocol_lock_sha256,
                        "data_manifest_sha256": common.data_manifest_sha256,
                        "feature_policy_sha256": common.feature_policy_sha256,
                    }
                )
            )
    return tuple(plans)


def test_offline_split_excludes_monitoring_and_hides_evaluation_truth() -> None:
    features = _Features()
    truth = _truth(features)
    monitoring = np.zeros(features.click_days.size, dtype=np.bool_)
    monitoring[0] = True

    split, evaluation_refs = _split(cast(RuntimeFeatureStore, features), truth, monitoring)

    assert split.train_labels.size == 69
    assert split.train_click_times.min() == 100.0
    assert not split.evaluation_labels.any()
    assert truth.final_labels[evaluation_refs].any()
    assert split.evaluation_features.shape[0] == 50


def test_offline_matrix_runs_seals_and_resumes_without_refitting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    features = _Features()
    truth = _truth(features)
    monitoring = np.zeros(features.click_days.size, dtype=np.bool_)
    plans = _plans()
    monkeypatch.setattr(
        "latesignal.experiments.production_offline.offline_reference_plans",
        lambda _: plans,
    )
    executor = ProductionOfflineExecutor(
        output_root=tmp_path,
        features=cast(RuntimeFeatureStore, features),
        truth=truth,
        monitoring_mask=monitoring,
        runtime_identity={"git_dirty": False, "runtime_sha256": "6" * 64},
    )
    inputs = cast(FinalPlanInputs, object())

    manifest = run_offline_references(inputs, tmp_path, executor=executor)
    repeated = run_offline_references(inputs, tmp_path, executor=executor)

    assert manifest == repeated
    assert manifest["completed_count"] == 6
    for plan in plans:
        root = tmp_path / "offline" / "runs" / plan.run_id
        prediction = __import__("json").loads((root / "manifest.json").read_text(encoding="utf-8"))
        evaluation = __import__("json").loads(
            (root / "evaluation.json").read_text(encoding="utf-8")
        )
        assert prediction["truth_joined"] is False
        assert evaluation["truth_joined"] is True
        assert evaluation["ranking_eligible"] is False


def test_offline_plan_expands_exact_two_by_three_grid(monkeypatch) -> None:
    monkeypatch.setattr(
        "latesignal.experiments.production_offline.final_online_plans",
        lambda _: (_common_plan(),),
    )

    plans = offline_reference_plans(cast(FinalPlanInputs, object()))

    assert len(plans) == 6
    assert {plan.name for plan in plans} == {
        "mature_logistic_regression",
        "mature_lightgbm",
    }
    assert {plan.seed for plan in plans} == {17, 41, 73}
