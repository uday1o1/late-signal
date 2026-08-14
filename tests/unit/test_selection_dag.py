from __future__ import annotations

from pathlib import Path

from latesignal.contracts.protocol import load_final_protocol
from latesignal.experiments.production_selection import ProductionSelectionPlan
from latesignal.experiments.protocol_lock import select_candidate, selection_decisions
from latesignal.experiments.selection_dag import (
    SelectionPlanInputs,
    completed_delayed_candidate,
    completed_model_candidate,
    completed_sampler_candidate,
    delayed_selection_plans,
    model_selection_plans,
    sampler_selection_plans,
    selection_results,
)


def _inputs() -> SelectionPlanInputs:
    _, protocol, protocol_sha256 = load_final_protocol(Path("configs/experiments/final.yaml"))
    return SelectionPlanInputs(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        data_manifest_sha256="1" * 64,
        feature_policy_sha256={"compact": "2" * 64, "large": "3" * 64},
        steps_per_credit=100,
        device="cuda",
    )


def _evidence(
    plan: ProductionSelectionPlan, loss: float
) -> tuple[dict[str, object], dict[str, object]]:
    config_sha256 = plan.canonical_sha256
    manifest = {
        "status": "complete",
        "config_sha256": config_sha256,
        "measured_compute_seconds": 10.0,
        "parameter_count": 100,
    }
    evaluation = {
        "status": "complete",
        "truth_joined": True,
        "config_sha256": config_sha256,
        "metrics": {"log_loss": loss},
    }
    return manifest, evaluation


def test_selection_dag_enumerates_exact_authored_stages_and_dependencies() -> None:
    inputs = _inputs()
    model_plans = model_selection_plans(inputs)
    assert len(model_plans) == 36
    assert len({plan.canonical_sha256 for plan in model_plans}) == 36
    assert {plan.method for plan in model_plans} == {"complete_wait"}
    assert {(plan.recent_window_days, plan.reservoir_capacity) for plan in model_plans} == {
        (3, 1_000_000)
    }

    model_candidates = []
    for index, plan in enumerate(model_plans):
        feature_policy = "compact" if plan.feature_policy_sha256 == "2" * 64 else "large"
        manifest, evaluation = _evidence(plan, 0.1 if index == 11 else 0.2 + index / 1000)
        model_candidates.append(
            completed_model_candidate(
                plan,
                feature_policy=feature_policy,
                manifest=manifest,
                evaluation=evaluation,
            )
        )
    selected_model = select_candidate(model_candidates)

    delayed_plans = delayed_selection_plans(inputs, selected_model)
    assert len(delayed_plans) == 8
    assert all(plan.learning_rate == selected_model.learning_rate for plan in delayed_plans)
    assert all(plan.weight_decay == selected_model.weight_decay for plan in delayed_plans)
    assert all(plan.dropout == selected_model.dropout for plan in delayed_plans)
    assert all(
        plan.feature_policy_sha256 == inputs.feature_policy_sha256[selected_model.feature_policy]
        for plan in delayed_plans
    )
    delayed_candidates = []
    for index, plan in enumerate(delayed_plans):
        manifest, evaluation = _evidence(plan, 0.05 if index == 6 else 0.3 + index / 1000)
        delayed_candidates.append(
            completed_delayed_candidate(plan, manifest=manifest, evaluation=evaluation)
        )
    selected_delayed = select_candidate(delayed_candidates)

    sampler_plans = sampler_selection_plans(inputs, selected_model, selected_delayed)
    assert len(sampler_plans) == 6
    assert all(plan.method == selected_delayed.method for plan in sampler_plans)
    assert all(plan.wait_days == selected_delayed.wait_days for plan in sampler_plans)
    sampler_candidates = []
    for index, plan in enumerate(sampler_plans):
        manifest, evaluation = _evidence(plan, 0.01 if index == 4 else 0.4 + index / 1000)
        sampler_candidates.append(
            completed_sampler_candidate(plan, manifest=manifest, evaluation=evaluation)
        )

    results = selection_results(
        inputs,
        model_candidates=model_candidates,
        delayed_candidates=delayed_candidates,
        sampler_candidates=sampler_candidates,
    )
    decisions = selection_decisions(results)

    assert (
        len(results.model_candidates)
        + len(results.delayed_candidates)
        + len(results.sampler_candidates)
        == 50
    )
    assert decisions["derived"] == {
        "shared_wait_days": selected_delayed.wait_days,
        "study_b_method": selected_delayed.method,
    }
