"""Frozen staged candidate graph for production selection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, cast

from latesignal.contracts.protocol import ProtocolDefinition
from latesignal.contracts.selection import (
    DelayedCandidate,
    ModelCandidate,
    SamplerCandidate,
    SelectionResults,
    SelectionWindow,
)
from latesignal.data.manifests import canonical_json_bytes
from latesignal.errors import ConsistencyError
from latesignal.experiments.production_selection import (
    ProductionSelectionPlan,
    SelectionMethod,
)
from latesignal.features.cache import FeaturePolicyName

SelectionDevice = Literal["cpu", "cuda"]


@dataclass(frozen=True, slots=True)
class SelectionPlanInputs:
    protocol: ProtocolDefinition
    protocol_sha256: str
    data_manifest_sha256: str
    feature_policy_sha256: dict[FeaturePolicyName, str]
    steps_per_credit: int
    device: SelectionDevice

    def validate(self) -> None:
        digests = {
            self.protocol_sha256,
            self.data_manifest_sha256,
            *self.feature_policy_sha256.values(),
        }
        if (
            set(self.feature_policy_sha256) != {"compact", "large"}
            or any(
                len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
                for value in digests
            )
            or self.steps_per_credit not in self.protocol.final_training.steps_per_credit_candidates
        ):
            raise ConsistencyError("Selection plan inputs violate the authored protocol")


def _run_id(stage: str, payload: dict[str, object]) -> str:
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"{stage}-{digest[:16]}"


def _plan(
    inputs: SelectionPlanInputs,
    *,
    stage: Literal["model", "delayed", "sampler"],
    method: SelectionMethod,
    wait_days: Literal[1, 3, 7, 14] | None,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    feature_policy: FeaturePolicyName,
    recent_window_days: Literal[1, 3, 7],
    reservoir_capacity: Literal[1_000_000, 5_000_000],
) -> ProductionSelectionPlan:
    semantic: dict[str, object] = {
        "stage": stage,
        "method": method,
        "wait_days": wait_days,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "feature_policy": feature_policy,
        "feature_policy_sha256": inputs.feature_policy_sha256[feature_policy],
        "recent_window_days": recent_window_days,
        "reservoir_capacity": reservoir_capacity,
        "steps_per_credit": inputs.steps_per_credit,
        "protocol_sha256": inputs.protocol_sha256,
        "data_manifest_sha256": inputs.data_manifest_sha256,
    }
    return ProductionSelectionPlan(
        version=1,
        phase="selection",
        stage=stage,
        run_id=_run_id(stage, semantic),
        method=method,
        seed=17,
        wait_days=wait_days,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        gradient_norm_clip=5.0,
        initialization_steps=inputs.protocol.final_training.initialization_steps,
        steps_per_credit=inputs.steps_per_credit,
        batch_size=inputs.protocol.final_training.batch_size,
        recent_window_days=recent_window_days,
        reservoir_capacity=reservoir_capacity,
        prediction_batch_size=65_536,
        protocol_sha256=inputs.protocol_sha256,
        data_manifest_sha256=inputs.data_manifest_sha256,
        feature_policy_sha256=inputs.feature_policy_sha256[feature_policy],
        device=inputs.device,
    )


def model_selection_plans(inputs: SelectionPlanInputs) -> tuple[ProductionSelectionPlan, ...]:
    inputs.validate()
    protocol = inputs.protocol
    defaults = protocol.selection_defaults
    return tuple(
        _plan(
            inputs,
            stage="model",
            method="complete_wait",
            wait_days=None,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
            feature_policy=feature_policy,
            recent_window_days=defaults.recent_window_days,
            reservoir_capacity=defaults.reservoir_capacity,
        )
        for learning_rate in protocol.model_selection.learning_rates
        for weight_decay in protocol.model_selection.weight_decays
        for dropout in protocol.model_selection.dropouts
        for feature_policy in protocol.model_selection.feature_policies
    )


def delayed_selection_plans(
    inputs: SelectionPlanInputs,
    selected_model: ModelCandidate,
) -> tuple[ProductionSelectionPlan, ...]:
    inputs.validate()
    if selected_model.status != "complete":
        raise ConsistencyError("Delayed selection requires a completed model winner")
    defaults = inputs.protocol.selection_defaults
    return tuple(
        _plan(
            inputs,
            stage="delayed",
            method=method,
            wait_days=wait_days,
            learning_rate=selected_model.learning_rate,
            weight_decay=selected_model.weight_decay,
            dropout=selected_model.dropout,
            feature_policy=selected_model.feature_policy,
            recent_window_days=defaults.recent_window_days,
            reservoir_capacity=defaults.reservoir_capacity,
        )
        for method in inputs.protocol.delayed_selection.methods
        for wait_days in inputs.protocol.delayed_selection.wait_days
    )


def sampler_selection_plans(
    inputs: SelectionPlanInputs,
    selected_model: ModelCandidate,
    selected_delayed: DelayedCandidate,
) -> tuple[ProductionSelectionPlan, ...]:
    inputs.validate()
    if selected_model.status != "complete" or selected_delayed.status != "complete":
        raise ConsistencyError("Sampler selection requires completed upstream winners")
    return tuple(
        _plan(
            inputs,
            stage="sampler",
            method=selected_delayed.method,
            wait_days=selected_delayed.wait_days,
            learning_rate=selected_model.learning_rate,
            weight_decay=selected_model.weight_decay,
            dropout=selected_model.dropout,
            feature_policy=selected_model.feature_policy,
            recent_window_days=recent_window_days,
            reservoir_capacity=reservoir_capacity,
        )
        for recent_window_days in inputs.protocol.sampler_selection.recent_window_days
        for reservoir_capacity in inputs.protocol.sampler_selection.reservoir_capacities
    )


def _complete_measurements(
    plan: ProductionSelectionPlan,
    manifest: dict[str, object],
    evaluation: dict[str, object],
) -> tuple[float, float, int]:
    metrics = evaluation.get("metrics")
    loss = metrics.get("log_loss") if isinstance(metrics, dict) else None
    compute = manifest.get("measured_compute_seconds")
    parameters = manifest.get("parameter_count")
    if (
        manifest.get("status") != "complete"
        or manifest.get("config_sha256") != plan.canonical_sha256
        or evaluation.get("status") != "complete"
        or evaluation.get("truth_joined") is not True
        or evaluation.get("config_sha256") != plan.canonical_sha256
        or isinstance(loss, bool)
        or not isinstance(loss, (int, float))
        or isinstance(compute, bool)
        or not isinstance(compute, (int, float))
        or isinstance(parameters, bool)
        or not isinstance(parameters, int)
    ):
        raise ConsistencyError("Completed selection candidate evidence is inconsistent")
    return float(loss), float(compute), parameters


def completed_model_candidate(
    plan: ProductionSelectionPlan,
    *,
    feature_policy: FeaturePolicyName,
    manifest: dict[str, object],
    evaluation: dict[str, object],
) -> ModelCandidate:
    if plan.stage != "model":
        raise ConsistencyError("Model result received a non-model candidate plan")
    loss, compute, parameters = _complete_measurements(plan, manifest, evaluation)
    return ModelCandidate(
        config_sha256=plan.canonical_sha256,
        status="complete",
        mean_selection_log_loss=loss,
        measured_compute_seconds=compute,
        parameter_count=parameters,
        failure_reason=None,
        learning_rate=plan.learning_rate,
        weight_decay=plan.weight_decay,
        dropout=plan.dropout,
        feature_policy=feature_policy,
        seed=17,
    )


def completed_delayed_candidate(
    plan: ProductionSelectionPlan,
    *,
    manifest: dict[str, object],
    evaluation: dict[str, object],
) -> DelayedCandidate:
    if plan.stage != "delayed" or plan.method == "complete_wait" or plan.wait_days is None:
        raise ConsistencyError("Delayed result received a non-delayed candidate plan")
    loss, compute, parameters = _complete_measurements(plan, manifest, evaluation)
    return DelayedCandidate(
        config_sha256=plan.canonical_sha256,
        status="complete",
        mean_selection_log_loss=loss,
        measured_compute_seconds=compute,
        parameter_count=parameters,
        failure_reason=None,
        method=plan.method,
        wait_days=plan.wait_days,
        seed=17,
    )


def completed_sampler_candidate(
    plan: ProductionSelectionPlan,
    *,
    manifest: dict[str, object],
    evaluation: dict[str, object],
) -> SamplerCandidate:
    if plan.stage != "sampler":
        raise ConsistencyError("Sampler result received a non-sampler candidate plan")
    if plan.reservoir_capacity not in {1_000_000, 5_000_000}:
        raise ConsistencyError("Sampler result contains an unauthorized reservoir capacity")
    loss, compute, parameters = _complete_measurements(plan, manifest, evaluation)
    return SamplerCandidate(
        config_sha256=plan.canonical_sha256,
        status="complete",
        mean_selection_log_loss=loss,
        measured_compute_seconds=compute,
        parameter_count=parameters,
        failure_reason=None,
        recent_window_days=plan.recent_window_days,
        reservoir_capacity=cast(Literal[1_000_000, 5_000_000], plan.reservoir_capacity),
        seed=17,
    )


def selection_results(
    inputs: SelectionPlanInputs,
    *,
    model_candidates: list[ModelCandidate],
    delayed_candidates: list[DelayedCandidate],
    sampler_candidates: list[SamplerCandidate],
) -> SelectionResults:
    inputs.validate()
    return SelectionResults(
        version=1,
        protocol_sha256=inputs.protocol_sha256,
        window=SelectionWindow(
            first_click_day=25,
            last_click_day=34,
            all_labels_mature_by_day=64,
            embargo_outcomes_accessed=False,
            final_period_metrics_accessed=False,
        ),
        model_candidates=model_candidates,
        delayed_candidates=delayed_candidates,
        sampler_candidates=sampler_candidates,
    )
