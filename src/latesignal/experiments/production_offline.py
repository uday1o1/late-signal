"""Locked mature-label offline references for the production final study."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from latesignal.contracts.protocol import StrictModel
from latesignal.data.manifests import (
    canonical_json_bytes,
    read_json,
    sha256_file,
    write_json_atomic,
)
from latesignal.errors import ConsistencyError
from latesignal.evaluation.metrics import classification_metrics
from latesignal.experiments.production_final import FinalPlanInputs, final_online_plans
from latesignal.features.store import RuntimeFeatureStore
from latesignal.models.lightgbm import run_lightgbm_reference
from latesignal.models.logistic import run_logistic_reference
from latesignal.models.offline import MatureOfflineSplit
from latesignal.simulator.production_oracle import SECONDS_PER_DAY, ProductionTruthStore

OfflineReferenceName = Literal["mature_logistic_regression", "mature_lightgbm"]
_OFFLINE_NAMES: tuple[OfflineReferenceName, ...] = (
    "mature_logistic_regression",
    "mature_lightgbm",
)
_FINAL_SEEDS = (17, 41, 73)


class ProductionOfflinePlan(StrictModel):
    """One immutable mature-label offline sanity-reference run."""

    version: Literal[1]
    phase: Literal["final"]
    run_id: str = Field(pattern=r"^offline-[0-9a-f]{16}$")
    name: OfflineReferenceName
    seed: Literal[17, 41, 73]
    training_first_click_day: Literal[0]
    training_last_click_day: Literal[34]
    training_cutoff_day: Literal[65]
    evaluation_first_click_day: Literal[65]
    evaluation_last_click_day: Literal[89]
    monitoring_excluded: Literal[True]
    ranking_eligible: Literal[False]
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def final_contract(self) -> ProductionOfflinePlan:
        if self.seed not in _FINAL_SEEDS:
            raise ValueError("Offline reference seed is outside the final protocol")
        return self

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()


def offline_reference_plans(inputs: FinalPlanInputs) -> tuple[ProductionOfflinePlan, ...]:
    """Expand the protocol lock into the exact two-by-three offline matrix."""

    common = final_online_plans(inputs)[0]
    plans: list[ProductionOfflinePlan] = []
    for name in _OFFLINE_NAMES:
        for seed in _FINAL_SEEDS:
            semantic: dict[str, object] = {
                "version": 1,
                "phase": "final",
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
            run_digest = hashlib.sha256(canonical_json_bytes(semantic)).hexdigest()
            plans.append(
                ProductionOfflinePlan.model_validate(
                    {**semantic, "run_id": f"offline-{run_digest[:16]}"}
                )
            )
    if len(plans) != 6 or len({plan.canonical_sha256 for plan in plans}) != 6:
        raise ConsistencyError("Offline matrix is not the exact authored two-by-three grid")
    return tuple(plans)


def _offline_features(
    features: RuntimeFeatureStore,
    references: NDArray[np.int32],
) -> NDArray[np.float64]:
    categorical_count = len(features.categorical_fields)
    numeric_count = features.numeric.shape[1]
    result = np.empty((references.size, categorical_count + numeric_count + 4), dtype=np.float64)
    for column, field in enumerate(features.categorical_fields):
        result[:, column] = features.categorical[
            references, column
        ] / features.cache.policy.bucket_count(field)
    result[:, categorical_count : categorical_count + numeric_count] = features.numeric[references]
    cursor = categorical_count + numeric_count
    result[:, cursor] = np.log1p(features.prior_user_clicks[references])
    result[:, cursor + 1] = np.log1p(features.prior_product_clicks[references])
    result[:, cursor + 2] = features.cold_user[references]
    result[:, cursor + 3] = features.cold_product[references]
    if not np.isfinite(result).all():
        raise ConsistencyError("Offline feature matrix contains a non-finite value")
    return result


def _split(
    features: RuntimeFeatureStore,
    truth: ProductionTruthStore,
    monitoring_mask: NDArray[np.bool_],
) -> tuple[MatureOfflineSplit, NDArray[np.int32]]:
    if monitoring_mask.shape != features.click_days.shape:
        raise ConsistencyError("Offline monitoring exclusion has the wrong shape")
    train_refs = np.flatnonzero((features.click_days <= 34) & ~monitoring_mask).astype(np.int32)
    evaluation_refs = np.flatnonzero(
        (features.click_days >= 65) & (features.click_days <= 89)
    ).astype(np.int32)
    origin = float(features.click_times[0])
    cutoff = float(np.nextafter(origin + 65 * SECONDS_PER_DAY, -np.inf))
    split = MatureOfflineSplit(
        train_features=_offline_features(features, train_refs),
        train_labels=truth.final_labels[train_refs].astype(np.int64),
        train_click_times=features.click_times[train_refs].astype(np.float64),
        train_available_at=truth.available_at[train_refs].astype(np.float64),
        evaluation_features=_offline_features(features, evaluation_refs),
        evaluation_labels=np.zeros(evaluation_refs.size, dtype=np.int64),
        evaluation_click_times=features.click_times[evaluation_refs].astype(np.float64),
        training_cutoff=cutoff,
        maturity_window=30 * SECONDS_PER_DAY,
    )
    return split, evaluation_refs


def _verify_hashed(value: dict[str, Any], digest_name: str, description: str) -> None:
    expected = value.get(digest_name)
    unsigned = {key: item for key, item in value.items() if key != digest_name}
    if (
        not isinstance(expected, str)
        or hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != expected
    ):
        raise ConsistencyError(f"{description} does not match its digest")


def _write_probabilities(path: Path, values: NDArray[np.float64]) -> tuple[str, int]:
    temporary = path.parent / f".{path.name}.tmp"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("xb") as output:
            np.save(output, values.astype(np.float32), allow_pickle=False)
            output.flush()
            os.fsync(output.fileno())
        loaded = np.load(temporary, allow_pickle=False)
        if loaded.dtype != np.float32 or not np.array_equal(loaded, values.astype(np.float32)):
            raise ConsistencyError("Offline probability round trip changed values")
        os.replace(temporary, path)
        return sha256_file(path)
    finally:
        temporary.unlink(missing_ok=True)


class ProductionOfflineExecutor:
    """Fit, seal, and evaluate one non-ranking offline reference."""

    def __init__(
        self,
        *,
        output_root: Path,
        features: RuntimeFeatureStore,
        truth: ProductionTruthStore,
        monitoring_mask: NDArray[np.bool_],
        runtime_identity: dict[str, Any],
    ) -> None:
        if (
            features.prepared_manifest_sha256 != truth.prepared_manifest_sha256
            or runtime_identity.get("git_dirty") is not False
        ):
            raise ConsistencyError("Offline executor identities are incomplete")
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.features = features
        self.truth = truth
        self.monitoring_mask = np.array(monitoring_mask, dtype=np.bool_, copy=True)
        self.runtime_identity = dict(runtime_identity)
        self._prepared: tuple[MatureOfflineSplit, NDArray[np.int32]] | None = None

    def _root(self, plan: ProductionOfflinePlan) -> Path:
        parent = self.output_root / "offline" / "runs"
        root = parent / plan.run_id
        if root.resolve().parent != parent.resolve() or root.is_symlink():
            raise ConsistencyError("Offline run root is redirected or malformed")
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _load_verified(self, root: Path, plan: ProductionOfflinePlan) -> dict[str, object]:
        manifest = read_json(root / "manifest.json")
        evaluation = read_json(root / "evaluation.json")
        _verify_hashed(manifest, "manifest_sha256", "Offline prediction manifest")
        _verify_hashed(evaluation, "evaluation_sha256", "Offline evaluation")
        probabilities_path = root / "probabilities.npy"
        sha256, size = sha256_file(probabilities_path)
        values = np.load(probabilities_path, allow_pickle=False)
        evaluation_refs = np.flatnonzero(
            (self.features.click_days >= 65) & (self.features.click_days <= 89)
        ).astype(np.int32)
        if (
            manifest.get("status") != "sealed_before_truth_evaluation"
            or manifest.get("truth_joined") is not False
            or evaluation.get("status") != "complete"
            or evaluation.get("truth_joined") is not True
            or any(value.get("run_id") != plan.run_id for value in (manifest, evaluation))
            or any(
                value.get("config_sha256") != plan.canonical_sha256
                for value in (manifest, evaluation)
            )
            or manifest.get("probabilities_sha256") != sha256
            or manifest.get("probabilities_bytes") != size
            or manifest.get("runtime_sha256") != self.runtime_identity.get("runtime_sha256")
            or manifest.get("ordered_evaluation_id_sha256")
            != hashlib.sha256(self.features.click_ids[evaluation_refs].tobytes()).hexdigest()
            or values.dtype != np.float32
            or values.shape != (evaluation_refs.size,)
            or evaluation.get("prediction_manifest_sha256") != manifest.get("manifest_sha256")
        ):
            raise ConsistencyError("Offline reference evidence is inconsistent")
        return {"manifest": manifest, "evaluation": evaluation}

    def execute(self, plan: ProductionOfflinePlan) -> dict[str, object]:
        if (
            plan.data_manifest_sha256 != self.features.prepared_manifest_sha256
            or plan.feature_policy_sha256 != self.features.feature_policy_sha256
        ):
            raise ConsistencyError("Offline plan does not match the selected data and features")
        root = self._root(plan)
        if (root / "evaluation.json").exists():
            return self._load_verified(root, plan)
        if self._prepared is None:
            self._prepared = _split(self.features, self.truth, self.monitoring_mask)
        split, evaluation_refs = self._prepared
        probability_path = root / "probabilities.npy"
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            started = time.perf_counter()
            if plan.name == "mature_logistic_regression":
                prediction = run_logistic_reference(split, seed=plan.seed)
            else:
                prediction = run_lightgbm_reference(split, seed=plan.seed)
            elapsed = time.perf_counter() - started
            if prediction.ranking_eligible or prediction.name != plan.name:
                raise ConsistencyError("Offline model returned an invalid reference prediction")
            probability_sha256, probability_bytes = _write_probabilities(
                probability_path, prediction.probabilities
            )
            feature_schema = {
                "categorical_fields": list(self.features.categorical_fields),
                "numeric_fields": list(self.features.numeric_fields),
                "history_fields": [
                    "log1p_prior_user_clicks",
                    "log1p_prior_product_clicks",
                    "cold_user",
                    "cold_product",
                ],
            }
            payload: dict[str, object] = {
                "version": 1,
                "status": "sealed_before_truth_evaluation",
                "truth_joined": False,
                "run_id": plan.run_id,
                "name": plan.name,
                "seed": plan.seed,
                "ranking_eligible": False,
                "plan": plan.model_dump(mode="json"),
                "config_sha256": plan.canonical_sha256,
                "protocol_sha256": plan.protocol_sha256,
                "protocol_lock_sha256": plan.protocol_lock_sha256,
                "data_manifest_sha256": plan.data_manifest_sha256,
                "feature_policy_sha256": plan.feature_policy_sha256,
                "feature_schema": feature_schema,
                "feature_schema_sha256": hashlib.sha256(
                    canonical_json_bytes(feature_schema)
                ).hexdigest(),
                "ordered_evaluation_id_sha256": hashlib.sha256(
                    self.features.click_ids[evaluation_refs].tobytes()
                ).hexdigest(),
                "training_examples": int(split.train_labels.size),
                "evaluation_examples": int(evaluation_refs.size),
                "training_seconds": elapsed,
                "runtime_sha256": self.runtime_identity.get("runtime_sha256"),
                "probabilities_path": probability_path.name,
                "probabilities_dtype": "float32",
                "probabilities_sha256": probability_sha256,
                "probabilities_bytes": probability_bytes,
            }
            payload["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
            write_json_atomic(manifest_path, payload)
        manifest = read_json(manifest_path)
        _verify_hashed(manifest, "manifest_sha256", "Offline prediction manifest")
        probability_sha256, probability_bytes = sha256_file(probability_path)
        probabilities = np.load(probability_path, allow_pickle=False)
        if (
            manifest.get("status") != "sealed_before_truth_evaluation"
            or manifest.get("truth_joined") is not False
            or manifest.get("run_id") != plan.run_id
            or manifest.get("config_sha256") != plan.canonical_sha256
            or manifest.get("protocol_lock_sha256") != plan.protocol_lock_sha256
            or manifest.get("data_manifest_sha256") != plan.data_manifest_sha256
            or manifest.get("feature_policy_sha256") != plan.feature_policy_sha256
            or manifest.get("ordered_evaluation_id_sha256")
            != hashlib.sha256(self.features.click_ids[evaluation_refs].tobytes()).hexdigest()
            or manifest.get("evaluation_examples") != evaluation_refs.size
            or manifest.get("runtime_sha256") != self.runtime_identity.get("runtime_sha256")
            or manifest.get("probabilities_dtype") != "float32"
            or manifest.get("probabilities_path") != probability_path.name
            or probabilities.dtype != np.float32
            or probabilities.shape != (evaluation_refs.size,)
            or manifest.get("probabilities_sha256") != probability_sha256
            or manifest.get("probabilities_bytes") != probability_bytes
        ):
            raise ConsistencyError("Offline sealed probabilities are malformed")
        labels = self.truth.final_labels[evaluation_refs].astype(np.int64)
        evaluation: dict[str, object] = {
            "version": 1,
            "status": "complete",
            "truth_joined": True,
            "run_id": plan.run_id,
            "name": plan.name,
            "seed": plan.seed,
            "ranking_eligible": False,
            "config_sha256": plan.canonical_sha256,
            "protocol_sha256": plan.protocol_sha256,
            "protocol_lock_sha256": plan.protocol_lock_sha256,
            "data_manifest_sha256": plan.data_manifest_sha256,
            "prediction_manifest_sha256": manifest["manifest_sha256"],
            "rows": int(labels.size),
            "period": [65, 89],
            "overall": classification_metrics(labels, probabilities),
        }
        evaluation["evaluation_sha256"] = hashlib.sha256(
            canonical_json_bytes(evaluation)
        ).hexdigest()
        write_json_atomic(root / "evaluation.json", evaluation)
        return self._load_verified(root, plan)


def run_offline_references(
    inputs: FinalPlanInputs,
    output_root: Path,
    *,
    executor: ProductionOfflineExecutor,
) -> dict[str, Any]:
    """Run or verify all six locked offline references in deterministic order."""

    if output_root.is_symlink():
        raise ConsistencyError("Offline coordinator output root cannot be a symlink")
    output_root.mkdir(parents=True, exist_ok=True)
    plans = offline_reference_plans(inputs)
    plan_payload: dict[str, object] = {
        "version": 1,
        "status": "frozen",
        "runs": [
            {
                "run_id": plan.run_id,
                "config_sha256": plan.canonical_sha256,
                "plan": plan.model_dump(mode="json"),
            }
            for plan in plans
        ],
    }
    plan_payload["plan_sha256"] = hashlib.sha256(canonical_json_bytes(plan_payload)).hexdigest()
    plan_path = output_root / "offline-plan.json"
    if plan_path.exists():
        stored_plan = read_json(plan_path)
        _verify_hashed(stored_plan, "plan_sha256", "Offline plan")
        if stored_plan != plan_payload:
            raise ConsistencyError("Immutable offline plan changed")
    else:
        write_json_atomic(plan_path, plan_payload)
    state_path = output_root / "offline-state.json"
    previous_completed: list[dict[str, object]] = []
    if state_path.exists():
        previous_state = read_json(state_path)
        _verify_hashed(previous_state, "state_sha256", "Offline coordinator state")
        raw_completed = previous_state.get("completed_runs")
        if (
            previous_state.get("status") not in {"running", "complete"}
            or previous_state.get("plan_sha256") != plan_payload["plan_sha256"]
            or not isinstance(raw_completed, list)
            or not all(isinstance(item, dict) for item in raw_completed)
            or previous_state.get("completed_count") != len(raw_completed)
            or len(raw_completed) > 6
            or (previous_state.get("status") == "complete" and len(raw_completed) != 6)
        ):
            raise ConsistencyError("Offline coordinator state is malformed")
        previous_completed = [dict(item) for item in raw_completed if isinstance(item, dict)]
    completed: list[dict[str, object]] = []
    for index, plan in enumerate(plans):
        result = executor.execute(plan)
        manifest = result["manifest"]
        evaluation = result["evaluation"]
        assert isinstance(manifest, dict) and isinstance(evaluation, dict)
        outcome = {
            "run_id": plan.run_id,
            "name": plan.name,
            "seed": plan.seed,
            "config_sha256": plan.canonical_sha256,
            "manifest_sha256": manifest["manifest_sha256"],
            "evaluation_sha256": evaluation["evaluation_sha256"],
        }
        if index < len(previous_completed) and previous_completed[index] != outcome:
            raise ConsistencyError("Completed offline evidence changed during resume")
        completed.append(outcome)
        state: dict[str, object] = {
            "version": 1,
            "status": "running",
            "plan_sha256": plan_payload["plan_sha256"],
            "completed_count": len(completed),
            "completed_runs": completed,
        }
        state["state_sha256"] = hashlib.sha256(canonical_json_bytes(state)).hexdigest()
        write_json_atomic(state_path, state, overwrite=True)
    payload: dict[str, object] = {
        "version": 1,
        "status": "complete",
        "plan_sha256": plan_payload["plan_sha256"],
        "completed_count": len(completed),
        "completed_runs": completed,
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    manifest_path = output_root / "offline-manifest.json"
    if manifest_path.exists():
        stored = read_json(manifest_path)
        _verify_hashed(stored, "manifest_sha256", "Offline coordinator manifest")
        if stored != payload:
            raise ConsistencyError("Offline coordinator evidence changed")
    else:
        write_json_atomic(manifest_path, payload)
    state = {
        "version": 1,
        "status": "complete",
        "plan_sha256": plan_payload["plan_sha256"],
        "completed_count": len(completed),
        "completed_runs": completed,
    }
    state["state_sha256"] = hashlib.sha256(canonical_json_bytes(state)).hexdigest()
    write_json_atomic(state_path, state, overwrite=True)
    return read_json(manifest_path)
