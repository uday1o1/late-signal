"""Durable coordinator for the exact locked final online matrix."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol

from latesignal.data.manifests import canonical_json_bytes, read_json, write_json_atomic
from latesignal.errors import ConsistencyError
from latesignal.experiments.production_final import FinalPlanInputs, ProductionFinalPlan
from latesignal.experiments.production_final import final_online_plans as expand_final_online_plans


class FinalRunExecutor(Protocol):
    """Execute or verify one immutable final run."""

    def execute(self, plan: ProductionFinalPlan) -> dict[str, object]: ...


def _hashed(value: dict[str, object], digest_name: str) -> dict[str, object]:
    result = dict(value)
    result[digest_name] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return result


def _verify_hash(
    value: dict[str, Any],
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


def _write_or_verify(
    path: Path,
    value: dict[str, object],
    *,
    digest_name: str,
) -> None:
    if path.exists():
        stored = read_json(path)
        _verify_hash(stored, digest_name=digest_name, description=path.name)
        if stored != value:
            raise ConsistencyError(f"Immutable final artifact changed: {path.name}")
        return
    write_json_atomic(path, value)


def _plan_manifest(plans: tuple[ProductionFinalPlan, ...]) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": 1,
        "status": "frozen",
        "online_runs": 33,
        "study_a_runs": 21,
        "study_b_runs": 12,
        "plans": [
            {
                "run_id": plan.run_id,
                "config_sha256": plan.canonical_sha256,
                "plan": plan.model_dump(mode="json"),
            }
            for plan in plans
        ],
    }
    return _hashed(payload, "plan_manifest_sha256")


def _outcome(plan: ProductionFinalPlan, result: dict[str, object]) -> dict[str, object]:
    manifest = result.get("manifest")
    evaluation = result.get("evaluation")
    retention = result.get("retention")
    if not all(isinstance(value, dict) for value in (manifest, evaluation, retention)):
        raise ConsistencyError("Final executor returned incomplete evidence")
    assert isinstance(manifest, dict)
    assert isinstance(evaluation, dict)
    assert isinstance(retention, dict)
    if (
        manifest.get("status") != "complete"
        or evaluation.get("status") != "complete"
        or evaluation.get("truth_joined") is not True
        or retention.get("status") != "verified_and_pruned"
        or any(value.get("run_id") != plan.run_id for value in (manifest, evaluation, retention))
        or any(
            value.get("config_sha256") != plan.canonical_sha256
            for value in (manifest, evaluation, retention)
        )
        or not all(
            isinstance(value, str) and len(value) == 64
            for value in (
                manifest.get("manifest_sha256"),
                evaluation.get("evaluation_sha256"),
                retention.get("retention_sha256"),
            )
        )
    ):
        raise ConsistencyError("Final executor evidence does not match the frozen run")
    return {
        "index": 0,
        "study": plan.study,
        "run_id": plan.run_id,
        "config_sha256": plan.canonical_sha256,
        "manifest_sha256": manifest["manifest_sha256"],
        "evaluation_sha256": evaluation["evaluation_sha256"],
        "retention_sha256": retention["retention_sha256"],
    }


def _state(
    plan_manifest_sha256: str,
    completed: list[dict[str, object]],
    *,
    status: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": 1,
        "status": status,
        "plan_manifest_sha256": plan_manifest_sha256,
        "completed_runs": completed,
        "completed_count": len(completed),
        "next_index": len(completed),
        "total_runs": 33,
    }
    return _hashed(payload, "state_sha256")


def _load_state(path: Path, *, plan_manifest_sha256: str) -> list[dict[str, object]]:
    if not path.exists():
        return []
    value = read_json(path)
    _verify_hash(value, digest_name="state_sha256", description="final coordinator state")
    completed = value.get("completed_runs")
    if (
        value.get("version") != 1
        or value.get("status") not in {"running", "complete"}
        or value.get("plan_manifest_sha256") != plan_manifest_sha256
        or not isinstance(completed, list)
        or value.get("completed_count") != len(completed)
        or value.get("next_index") != len(completed)
        or value.get("total_runs") != 33
        or len(completed) > 33
        or not all(isinstance(item, dict) for item in completed)
    ):
        raise ConsistencyError("Final coordinator state is malformed or changed")
    return [dict(item) for item in completed if isinstance(item, dict)]


def run_final_online_coordinator(
    inputs: FinalPlanInputs,
    output_root: Path,
    *,
    executor: FinalRunExecutor,
) -> dict[str, Any]:
    """Run or verify all 33 final online experiments in frozen order."""

    if output_root.is_symlink():
        raise ConsistencyError("Final coordinator output root cannot be a symlink")
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    plans = expand_final_online_plans(inputs)
    plan = _plan_manifest(plans)
    plan_path = root / "online-plan.json"
    _write_or_verify(plan_path, plan, digest_name="plan_manifest_sha256")
    plan_sha256 = plan["plan_manifest_sha256"]
    assert isinstance(plan_sha256, str)
    state_path = root / "online-state.json"
    completed = _load_state(state_path, plan_manifest_sha256=plan_sha256)
    for index, frozen in enumerate(plans):
        outcome = _outcome(frozen, executor.execute(frozen))
        outcome["index"] = index
        if index < len(completed):
            if completed[index] != outcome:
                raise ConsistencyError("Completed final run evidence changed during resume")
            continue
        completed.append(outcome)
        write_json_atomic(
            state_path,
            _state(plan_sha256, completed, status="running"),
            overwrite=True,
        )
    payload: dict[str, object] = {
        "version": 1,
        "status": "complete",
        "plan_manifest_sha256": plan_sha256,
        "completed_runs": completed,
        "completed_count": len(completed),
        "study_a_runs": sum(item["study"] == "study_a" for item in completed),
        "study_b_runs": sum(item["study"] == "study_b" for item in completed),
    }
    manifest = _hashed(payload, "manifest_sha256")
    _write_or_verify(root / "online-manifest.json", manifest, digest_name="manifest_sha256")
    write_json_atomic(
        state_path,
        _state(plan_sha256, completed, status="complete"),
        overwrite=True,
    )
    return read_json(root / "online-manifest.json")
