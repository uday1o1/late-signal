from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

import pytest

from latesignal.data.manifests import canonical_json_bytes, read_json, write_json_atomic
from latesignal.errors import ConsistencyError
from latesignal.experiments.final_coordinator import run_final_online_coordinator
from latesignal.experiments.production_final import FinalPlanInputs, ProductionFinalPlan


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _plans() -> tuple[ProductionFinalPlan, ...]:
    plans: list[ProductionFinalPlan] = []
    for index in range(33):
        study = "study_a" if index < 21 else "study_b"
        scheduler = "fixed_daily" if study == "study_a" else "fixed_deadline"
        method = "complete_wait" if study == "study_a" else "fixed_wait"
        plans.append(
            ProductionFinalPlan.model_validate(
                {
                    "version": 1,
                    "phase": "qualification",
                    "study": study,
                    "run_id": (
                        ("study-a-" if study == "study_a" else "study-b-") + f"{index:016x}"
                    ),
                    "method": method,
                    "scheduler": scheduler,
                    "seed": index,
                    "wait_days": 3 if method == "fixed_wait" else None,
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
        )
    return tuple(plans)


class _Executor:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.fail_at = fail_at
        self.calls: list[str] = []

    def execute(self, plan: ProductionFinalPlan) -> dict[str, object]:
        index = int(plan.run_id[-16:], 16)
        self.calls.append(plan.run_id)
        if index == self.fail_at:
            raise RuntimeError("simulated interruption")
        return {
            "manifest": {
                "status": "complete",
                "run_id": plan.run_id,
                "config_sha256": plan.canonical_sha256,
                "manifest_sha256": _digest(f"manifest-{plan.run_id}"),
            },
            "evaluation": {
                "status": "complete",
                "truth_joined": True,
                "run_id": plan.run_id,
                "config_sha256": plan.canonical_sha256,
                "evaluation_sha256": _digest(f"evaluation-{plan.run_id}"),
            },
            "retention": {
                "status": "verified_and_pruned",
                "run_id": plan.run_id,
                "config_sha256": plan.canonical_sha256,
                "retention_sha256": _digest(f"retention-{plan.run_id}"),
            },
        }


def test_final_coordinator_resumes_and_reverifies_every_completed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plans = _plans()
    monkeypatch.setattr(
        "latesignal.experiments.final_coordinator.expand_final_online_plans",
        lambda _: plans,
    )
    inputs = cast(FinalPlanInputs, object())
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_final_online_coordinator(inputs, tmp_path, executor=_Executor(fail_at=5))
    interrupted = read_json(tmp_path / "online-state.json")
    assert interrupted["status"] == "running"
    assert interrupted["completed_count"] == 5

    resumed_executor = _Executor()
    completed = run_final_online_coordinator(inputs, tmp_path, executor=resumed_executor)
    repeated = run_final_online_coordinator(inputs, tmp_path, executor=_Executor())

    assert len(resumed_executor.calls) == 33
    assert completed == repeated
    assert completed["completed_count"] == 33
    assert completed["study_a_runs"] == 21
    assert completed["study_b_runs"] == 12
    assert read_json(tmp_path / "online-state.json")["status"] == "complete"


def test_final_coordinator_refuses_changed_resume_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plans = _plans()
    monkeypatch.setattr(
        "latesignal.experiments.final_coordinator.expand_final_online_plans",
        lambda _: plans,
    )
    inputs = cast(FinalPlanInputs, object())
    run_final_online_coordinator(inputs, tmp_path, executor=_Executor())
    state = read_json(tmp_path / "online-state.json")
    state["completed_runs"][0]["retention_sha256"] = "f" * 64
    unsigned = {key: value for key, value in state.items() if key != "state_sha256"}
    state["state_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    write_json_atomic(tmp_path / "online-state.json", state, overwrite=True)

    with pytest.raises(ConsistencyError, match="evidence changed"):
        run_final_online_coordinator(inputs, tmp_path, executor=_Executor())
