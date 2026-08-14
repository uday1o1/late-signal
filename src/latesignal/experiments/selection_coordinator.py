"""Durable coordinator for the frozen staged production-selection graph."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import TypeAdapter, ValidationError

from latesignal.contracts.selection import (
    DelayedCandidate,
    ModelCandidate,
    SamplerCandidate,
    SelectionResults,
)
from latesignal.data.manifests import canonical_json_bytes, read_json, write_json_atomic
from latesignal.errors import ConsistencyError
from latesignal.experiments.production_selection import ProductionSelectionPlan
from latesignal.experiments.protocol_lock import select_candidate, selection_decisions
from latesignal.experiments.selection_dag import (
    SelectionPlanInputs,
    delayed_selection_plans,
    model_selection_plans,
    sampler_selection_plans,
    selection_results,
)
from latesignal.features.cache import FeaturePolicyName

StageName = Literal["model", "delayed", "sampler"]
Candidate = ModelCandidate | DelayedCandidate | SamplerCandidate


@dataclass(frozen=True, slots=True)
class PreparedSelectionCandidate:
    plan: ProductionSelectionPlan
    feature_policy: FeaturePolicyName
    status: Literal["complete", "protocol_invalid"]
    failure_reason: str | None = None


class SelectionCandidateExecutor(Protocol):
    def prepare(
        self,
        plan: ProductionSelectionPlan,
        *,
        feature_policy: FeaturePolicyName,
    ) -> PreparedSelectionCandidate: ...

    def score(self, prepared: PreparedSelectionCandidate) -> Candidate: ...


def _hashed_payload(payload: dict[str, object], digest_name: str) -> dict[str, object]:
    result = dict(payload)
    result[digest_name] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def _verify_hashed_payload(
    value: dict[str, object],
    *,
    digest_name: str,
    description: str,
) -> None:
    expected = value.get(digest_name)
    unsigned = {key: item for key, item in value.items() if key != digest_name}
    if (
        not isinstance(expected, str)
        or hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != expected
    ):
        raise ConsistencyError(f"{description} content does not match its digest")


def _write_or_verify(path: Path, value: dict[str, object], *, digest_name: str) -> None:
    if path.exists():
        stored = read_json(path)
        _verify_hashed_payload(stored, digest_name=digest_name, description=path.name)
        if stored != value:
            raise ConsistencyError(f"Immutable selection artifact changed: {path.name}")
        return
    write_json_atomic(path, value)


def _policy_for_model_plan(
    plan: ProductionSelectionPlan,
    inputs: SelectionPlanInputs,
) -> FeaturePolicyName:
    matches = [
        name
        for name, digest in inputs.feature_policy_sha256.items()
        if digest == plan.feature_policy_sha256
    ]
    if len(matches) != 1:
        raise ConsistencyError("Model candidate feature policy identity is ambiguous")
    return matches[0]


def _validate_stage_candidates[CandidateT: Candidate](
    stage: StageName,
    plans: tuple[ProductionSelectionPlan, ...],
    candidates: list[CandidateT],
) -> None:
    if (
        len(candidates) != len(plans)
        or {candidate.config_sha256 for candidate in candidates}
        != {plan.canonical_sha256 for plan in plans}
        or any(candidate.status not in {"complete", "protocol_invalid"} for candidate in candidates)
    ):
        raise ConsistencyError(f"Selection {stage} outcomes do not match the frozen stage plan")


def _stage_plan(
    stage: StageName,
    plans: tuple[ProductionSelectionPlan, ...],
    feature_policies: tuple[FeaturePolicyName, ...],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": 1,
        "status": "frozen",
        "stage": stage,
        "candidates": [
            {
                "feature_policy": feature_policy,
                "plan": plan.model_dump(mode="json"),
                "config_sha256": plan.canonical_sha256,
            }
            for plan, feature_policy in zip(plans, feature_policies, strict=True)
        ],
    }
    return _hashed_payload(payload, "stage_plan_sha256")


def _stage_results[CandidateT: Candidate](
    stage: StageName,
    candidates: list[CandidateT],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": 1,
        "status": "complete",
        "stage": stage,
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
    }
    return _hashed_payload(payload, "stage_results_sha256")


def _load_stage_results[CandidateModel: ModelCandidate | DelayedCandidate | SamplerCandidate](
    path: Path,
    *,
    stage: StageName,
    model: type[CandidateModel],
) -> list[CandidateModel]:
    value = read_json(path)
    _verify_hashed_payload(value, digest_name="stage_results_sha256", description=path.name)
    raw = value.get("candidates")
    if (
        value.get("version") != 1
        or value.get("status") != "complete"
        or value.get("stage") != stage
        or not isinstance(raw, list)
    ):
        raise ConsistencyError(f"Selection {stage} result artifact is malformed")
    try:
        return TypeAdapter(list[model]).validate_python(raw)  # type: ignore[valid-type]
    except ValidationError as error:
        raise ConsistencyError(f"Selection {stage} result candidates are malformed") from error


def _run_stage[CandidateModel: ModelCandidate | DelayedCandidate | SamplerCandidate](
    root: Path,
    *,
    stage: StageName,
    plans: tuple[ProductionSelectionPlan, ...],
    feature_policies: tuple[FeaturePolicyName, ...],
    model: type[CandidateModel],
    executor: SelectionCandidateExecutor,
) -> list[CandidateModel]:
    stage_root = root / stage
    stage_root.mkdir(parents=True, exist_ok=True)
    plan_artifact = _stage_plan(stage, plans, feature_policies)
    _write_or_verify(
        stage_root / "stage-plan.json",
        plan_artifact,
        digest_name="stage_plan_sha256",
    )
    result_path = stage_root / "stage-results.json"
    if result_path.exists():
        candidates = _load_stage_results(result_path, stage=stage, model=model)
        _validate_stage_candidates(stage, plans, candidates)
        return candidates
    prepared = [
        executor.prepare(plan, feature_policy=feature_policy)
        for plan, feature_policy in zip(plans, feature_policies, strict=True)
    ]
    if any(item.plan != plan for item, plan in zip(prepared, plans, strict=True)):
        raise ConsistencyError(f"Selection {stage} executor changed a frozen candidate plan")
    raw_candidates = [executor.score(item) for item in prepared]
    if any(not isinstance(candidate, model) for candidate in raw_candidates):
        raise ConsistencyError(f"Selection {stage} executor returned the wrong result type")
    candidates = [candidate for candidate in raw_candidates if isinstance(candidate, model)]
    _validate_stage_candidates(stage, plans, candidates)
    _write_or_verify(
        result_path,
        _stage_results(stage, candidates),
        digest_name="stage_results_sha256",
    )
    return candidates


def run_selection_coordinator(
    inputs: SelectionPlanInputs,
    output_root: Path,
    *,
    executor: SelectionCandidateExecutor,
) -> SelectionResults:
    """Run every frozen stage, scoring only after every stage candidate is sealed."""

    inputs.validate()
    if output_root.is_symlink():
        raise ConsistencyError("Selection coordinator output root cannot be a symlink")
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    final_path = root / "selection-results.json"
    manifest_path = root / "manifest.json"
    if final_path.exists() or manifest_path.exists():
        if not final_path.exists() or not manifest_path.exists():
            raise ConsistencyError("Completed selection coordinator artifacts are incomplete")
        stored = SelectionResults.model_validate(read_json(final_path))
        manifest = read_json(manifest_path)
        _verify_hashed_payload(
            manifest,
            digest_name="manifest_sha256",
            description="selection coordinator manifest",
        )
        if (
            stored.protocol_sha256 != inputs.protocol_sha256
            or manifest.get("selection_results_sha256")
            != hashlib.sha256(canonical_json_bytes(stored.model_dump(mode="json"))).hexdigest()
            or manifest.get("decisions") != selection_decisions(stored)
        ):
            raise ConsistencyError("Completed selection coordinator identity changed")
        return stored

    model_plans = model_selection_plans(inputs)
    model_policies = tuple(_policy_for_model_plan(plan, inputs) for plan in model_plans)
    model_candidates = _run_stage(
        root,
        stage="model",
        plans=model_plans,
        feature_policies=model_policies,
        model=ModelCandidate,
        executor=executor,
    )
    selected_model = select_candidate(model_candidates)

    delayed_plans = delayed_selection_plans(inputs, selected_model)
    delayed_policies = (selected_model.feature_policy,) * len(delayed_plans)
    delayed_candidates = _run_stage(
        root,
        stage="delayed",
        plans=delayed_plans,
        feature_policies=delayed_policies,
        model=DelayedCandidate,
        executor=executor,
    )
    selected_delayed = select_candidate(delayed_candidates)

    sampler_plans = sampler_selection_plans(inputs, selected_model, selected_delayed)
    sampler_policies = (selected_model.feature_policy,) * len(sampler_plans)
    sampler_candidates = _run_stage(
        root,
        stage="sampler",
        plans=sampler_plans,
        feature_policies=sampler_policies,
        model=SamplerCandidate,
        executor=executor,
    )
    result = selection_results(
        inputs,
        model_candidates=model_candidates,
        delayed_candidates=delayed_candidates,
        sampler_candidates=sampler_candidates,
    )
    result_payload = result.model_dump(mode="json")
    write_json_atomic(final_path, result_payload)
    results_sha256 = hashlib.sha256(canonical_json_bytes(result_payload)).hexdigest()
    final_manifest: dict[str, object] = {
        "version": 1,
        "status": "complete",
        "protocol_sha256": inputs.protocol_sha256,
        "candidate_counts": {"model": 36, "delayed": 8, "sampler": 6, "total": 50},
        "selection_results_sha256": results_sha256,
        "decisions": selection_decisions(result),
    }
    write_json_atomic(
        manifest_path,
        _hashed_payload(final_manifest, "manifest_sha256"),
    )
    return result
