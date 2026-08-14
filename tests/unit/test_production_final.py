from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from latesignal.contracts.protocol import load_final_protocol
from latesignal.errors import ConsistencyError
from latesignal.experiments.production_final import (
    FinalPlanInputs,
    ProductionFinalPlan,
    final_online_plans,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _complete_fields(name: str, loss: float) -> dict[str, object]:
    return {
        "config_sha256": _digest(name),
        "status": "complete",
        "mean_selection_log_loss": loss,
        "measured_compute_seconds": 1.0,
        "parameter_count": 100,
        "failure_reason": None,
    }


def _inputs() -> FinalPlanInputs:
    _, protocol, protocol_sha256 = load_final_protocol(Path("configs/experiments/final.yaml"))
    decisions = {
        "model": {
            **_complete_fields("model", 0.1),
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "dropout": 0.1,
            "feature_policy": "large",
            "seed": 17,
        },
        "delayed": {
            **_complete_fields("delayed", 0.1),
            "method": "es_dfm",
            "wait_days": 7,
            "seed": 17,
        },
        "sampler": {
            **_complete_fields("sampler", 0.1),
            "recent_window_days": 1,
            "reservoir_capacity": 5_000_000,
            "seed": 17,
        },
        "derived": {"shared_wait_days": 7, "study_b_method": "es_dfm"},
        "tie_policy": {
            "metric_tolerance": 1e-6,
            "order": [
                "mean_selection_log_loss",
                "measured_compute_seconds",
                "parameter_count",
                "config_sha256",
            ],
        },
    }
    lock: dict[str, object] = {
        "status": "locked",
        "locked_before_final_scoring": True,
        "protocol_sha256": protocol_sha256,
        "lock_sha256": _digest("lock"),
        "data": {"manifest_sha256": _digest("data")},
        "selection_decisions": decisions,
        "selected_steps_per_credit": 100,
        "final_seeds": [17, 41, 73],
    }
    return FinalPlanInputs(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        protocol_lock=lock,
        feature_policy_sha256={"compact": _digest("compact"), "large": _digest("large")},
    )


def test_final_lock_expands_to_exact_online_matrix() -> None:
    plans = final_online_plans(_inputs())

    study_a = [plan for plan in plans if plan.study == "study_a"]
    study_b = [plan for plan in plans if plan.study == "study_b"]
    assert len(study_a) == 21
    assert len(study_b) == 12
    assert {plan.seed for plan in study_a} == {17, 41, 73}
    assert {plan.method for plan in study_a} == {
        "complete_wait",
        "immediate_fake_negative",
        "fixed_wait",
        "dfm",
        "fnw",
        "es_dfm",
        "oracle_reference",
    }
    assert {plan.scheduler for plan in study_b} == {
        "fixed_early",
        "fixed_midpoint",
        "fixed_deadline",
        "calibration_drift",
    }
    assert all(plan.method == "es_dfm" and plan.wait_days == 7 for plan in study_b)
    assert all(plan.wait_days == 7 for plan in study_a if plan.method in {"fixed_wait", "es_dfm"})
    assert all(
        plan.wait_days is None for plan in study_a if plan.method not in {"fixed_wait", "es_dfm"}
    )
    assert all(
        (plan.deployable, plan.ranking_eligible)
        == ((False, False) if plan.method == "oracle_reference" else (True, True))
        for plan in study_a
    )
    assert {plan.credits for plan in study_a} == {59}
    assert {plan.credits for plan in study_b} == {12}
    assert len({plan.run_id for plan in plans}) == 33
    assert all(plan.intermediate_budget_fractions == (0.25, 0.5, 0.75, 1.0) for plan in plans)


def test_final_lock_refuses_derived_method_or_wait_drift() -> None:
    inputs = _inputs()
    decisions = inputs.protocol_lock["selection_decisions"]
    assert isinstance(decisions, dict)
    derived = decisions["derived"]
    assert isinstance(derived, dict)
    derived["shared_wait_days"] = 14

    with pytest.raises(ConsistencyError, match="inconsistent selection decisions"):
        final_online_plans(inputs)


def test_final_plan_refuses_oracle_in_deployable_ranking() -> None:
    oracle = next(
        plan for plan in final_online_plans(_inputs()) if plan.method == "oracle_reference"
    )
    payload = oracle.model_dump(mode="python")
    payload["deployable"] = True

    with pytest.raises(ValueError, match="oracle"):
        ProductionFinalPlan.model_validate(payload)
