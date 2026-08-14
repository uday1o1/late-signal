from __future__ import annotations

from pathlib import Path

from latesignal.contracts.protocol import load_final_protocol
from latesignal.contracts.selection import DelayedCandidate, ModelCandidate, SamplerCandidate
from latesignal.experiments.production_selection import ProductionSelectionPlan
from latesignal.experiments.selection_coordinator import (
    Candidate,
    PreparedSelectionCandidate,
    run_selection_coordinator,
)
from latesignal.experiments.selection_dag import (
    SelectionPlanInputs,
    completed_delayed_candidate,
    completed_model_candidate,
    completed_sampler_candidate,
)
from latesignal.features.cache import FeaturePolicyName


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


class _Executor:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def prepare(
        self,
        plan: ProductionSelectionPlan,
        *,
        feature_policy: FeaturePolicyName,
    ) -> PreparedSelectionCandidate:
        self.events.append(("prepare", plan.stage))
        return PreparedSelectionCandidate(plan, feature_policy, "complete")

    def score(self, prepared: PreparedSelectionCandidate) -> Candidate:
        plan = prepared.plan
        self.events.append(("score", plan.stage))
        loss = {
            "model": int(plan.run_id[-2:], 16) / 1000 + 0.2,
            "delayed": 0.1 if plan.method == "es_dfm" and plan.wait_days == 7 else 0.3,
            "sampler": (
                0.05
                if plan.recent_window_days == 3 and plan.reservoir_capacity == 5_000_000
                else 0.4
            ),
        }[plan.stage]
        manifest: dict[str, object] = {
            "status": "complete",
            "config_sha256": plan.canonical_sha256,
            "measured_compute_seconds": 10.0,
            "parameter_count": 100,
        }
        evaluation: dict[str, object] = {
            "status": "complete",
            "truth_joined": True,
            "config_sha256": plan.canonical_sha256,
            "metrics": {"log_loss": loss},
        }
        if plan.stage == "model":
            return completed_model_candidate(
                plan,
                feature_policy=prepared.feature_policy,
                manifest=manifest,
                evaluation=evaluation,
            )
        if plan.stage == "delayed":
            return completed_delayed_candidate(
                plan,
                manifest=manifest,
                evaluation=evaluation,
            )
        return completed_sampler_candidate(
            plan,
            manifest=manifest,
            evaluation=evaluation,
        )


class _UnexpectedExecutor:
    def prepare(
        self,
        plan: ProductionSelectionPlan,
        *,
        feature_policy: FeaturePolicyName,
    ) -> PreparedSelectionCandidate:
        raise AssertionError((plan, feature_policy))

    def score(
        self, prepared: PreparedSelectionCandidate
    ) -> ModelCandidate | DelayedCandidate | SamplerCandidate:
        raise AssertionError(prepared)


def test_coordinator_freezes_every_stage_before_reading_its_scores(tmp_path: Path) -> None:
    executor = _Executor()

    results = run_selection_coordinator(_inputs(), tmp_path / "selection", executor=executor)

    assert len(results.model_candidates) == 36
    assert len(results.delayed_candidates) == 8
    assert len(results.sampler_candidates) == 6
    assert executor.events[:36] == [("prepare", "model")] * 36
    assert executor.events[36:72] == [("score", "model")] * 36
    assert executor.events[72:80] == [("prepare", "delayed")] * 8
    assert executor.events[80:88] == [("score", "delayed")] * 8
    assert executor.events[88:94] == [("prepare", "sampler")] * 6
    assert executor.events[94:] == [("score", "sampler")] * 6

    repeated = run_selection_coordinator(
        _inputs(),
        tmp_path / "selection",
        executor=_UnexpectedExecutor(),
    )
    assert repeated == results

    (tmp_path / "selection" / "selection-results.json").unlink()
    (tmp_path / "selection" / "manifest.json").unlink()
    recovered = run_selection_coordinator(
        _inputs(),
        tmp_path / "selection",
        executor=_UnexpectedExecutor(),
    )
    assert recovered == results
