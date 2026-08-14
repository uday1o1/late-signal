"""Production candidate execution, evidence compaction, and resume handling."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from latesignal.contracts.selection import DelayedCandidate, ModelCandidate, SamplerCandidate
from latesignal.data.manifests import canonical_json_bytes, read_json, write_json_atomic
from latesignal.errors import ConsistencyError
from latesignal.experiments.checkpoint import CheckpointIdentity, RollingCheckpointStore
from latesignal.experiments.exposures import ExposureLedgerIdentity, ExposureLedgerWriter
from latesignal.experiments.predictions import PredictionLedgerIdentity, PredictionLedgerWriter
from latesignal.experiments.production_selection import (
    ProductionSelectionController,
    ProductionSelectionPlan,
    SelectionFeatureStore,
)
from latesignal.experiments.selection_coordinator import (
    Candidate,
    PreparedSelectionCandidate,
)
from latesignal.experiments.selection_dag import (
    completed_delayed_candidate,
    completed_model_candidate,
    completed_sampler_candidate,
)
from latesignal.experiments.selection_evaluation import (
    evaluate_selection_candidate,
    verify_selection_run_manifest,
)
from latesignal.features.cache import FeaturePolicyName
from latesignal.simulator.production_oracle import ProductionTruthStore

_SCIENTIFIC_FAILURES = {
    "INSUFFICIENT_LEGAL_POOL",
    "INSUFFICIENT_LEGAL_AUXILIARY_POOL",
}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hashed_payload(payload: dict[str, object], digest_name: str) -> dict[str, object]:
    result = dict(payload)
    result[digest_name] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def _verify_hashed_payload(
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
        raise ConsistencyError(f"{description} does not match its digest")


def _safe_remove_child(root: Path, name: str) -> None:
    if name not in {"checkpoints", "exposures", "predictions"}:
        raise ConsistencyError("Selection retention requested an unauthorized removal")
    resolved_root = root.resolve()
    target = resolved_root / name
    if not target.exists():
        return
    if target.is_symlink() or not target.is_dir() or target.resolve().parent != resolved_root:
        raise ConsistencyError("Selection retention target is redirected or malformed")
    shutil.rmtree(target)
    _fsync_directory(resolved_root)


class ProductionSelectionExecutor:
    """Execute frozen candidates while retaining only verified aggregate evidence."""

    def __init__(
        self,
        *,
        output_root: Path,
        features: dict[FeaturePolicyName, SelectionFeatureStore],
        truth: ProductionTruthStore,
        monitoring_mask: NDArray[np.bool_],
        runtime_identity: dict[str, Any],
        device_uuid: str,
    ) -> None:
        if output_root.is_symlink() or set(features) != {"compact", "large"}:
            raise ConsistencyError("Production selection executor inputs are incomplete")
        compact = features["compact"]
        large = features["large"]
        if (
            compact.prepared_manifest_sha256 != large.prepared_manifest_sha256
            or compact.prepared_manifest_sha256 != truth.prepared_manifest_sha256
            or not np.array_equal(compact.click_ids, large.click_ids)
            or not np.array_equal(compact.click_times, large.click_times)
            or not np.array_equal(compact.click_days, large.click_days)
            or monitoring_mask.shape != compact.click_days.shape
        ):
            raise ConsistencyError("Selection feature policies or truth do not share one cohort")
        required_runtime = {
            "source_tree_sha256",
            "dependency_lock_sha256",
            "git_commit",
            "runtime_sha256",
            "git_dirty",
        }
        if (
            not required_runtime.issubset(runtime_identity)
            or runtime_identity.get("git_dirty") is not False
            or not device_uuid
        ):
            raise ConsistencyError("Selection requires a clean complete runtime identity")
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.features = features
        self.truth = truth
        self.monitoring_mask = np.array(monitoring_mask, dtype=np.bool_, copy=True)
        self.runtime_identity = dict(runtime_identity)
        self.device_uuid = device_uuid

    def _root(self, plan: ProductionSelectionPlan) -> Path:
        root = self.output_root / plan.stage / "candidates" / plan.run_id
        if root.resolve().parent != (self.output_root / plan.stage / "candidates").resolve():
            raise ConsistencyError("Selection candidate run ID escapes its stage root")
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _checkpoint_identity(self, plan: ProductionSelectionPlan) -> CheckpointIdentity:
        runtime = self.runtime_identity
        return CheckpointIdentity(
            version=1,
            phase=plan.phase,
            run_id=plan.run_id,
            config_sha256=plan.canonical_sha256,
            protocol_sha256=plan.protocol_sha256,
            data_manifest_sha256=plan.data_manifest_sha256,
            feature_policy_sha256=plan.feature_policy_sha256,
            source_tree_sha256=str(runtime["source_tree_sha256"]),
            dependency_lock_sha256=str(runtime["dependency_lock_sha256"]),
            git_commit=str(runtime["git_commit"]),
            environment_sha256=str(runtime["runtime_sha256"]),
            device_uuid=self.device_uuid,
        )

    def _verify_retention(
        self,
        root: Path,
        plan: ProductionSelectionPlan,
    ) -> dict[str, Any]:
        receipt = read_json(root / "training-retention.json")
        _verify_hashed_payload(
            receipt,
            digest_name="retention_sha256",
            description="Selection training-retention receipt",
        )
        if (
            receipt.get("status") != "verified_and_pruned"
            or receipt.get("config_sha256") != plan.canonical_sha256
            or receipt.get("pruned") != ["checkpoints", "exposures"]
        ):
            raise ConsistencyError("Selection training-retention receipt is inconsistent")
        return receipt

    def _compact_training_evidence(
        self,
        root: Path,
        plan: ProductionSelectionPlan,
        checkpoint_identity: CheckpointIdentity,
    ) -> None:
        receipt_path = root / "training-retention.json"
        if receipt_path.exists():
            self._verify_retention(root, plan)
            _safe_remove_child(root, "checkpoints")
            _safe_remove_child(root, "exposures")
            return
        manifest = verify_selection_run_manifest(root / "manifest.json")
        prediction_identity = PredictionLedgerIdentity.model_validate(
            read_json(root / "predictions" / "identity.json")
        )
        prediction_seal = PredictionLedgerWriter(
            root / "predictions", prediction_identity
        ).verify_seal()
        exposure_identity = ExposureLedgerIdentity.model_validate(
            read_json(root / "exposures" / "identity.json")
        )
        exposure_seal = ExposureLedgerWriter(root / "exposures", exposure_identity).verify_seal()
        checkpoint = RollingCheckpointStore(root / "checkpoints").load_latest(checkpoint_identity)
        if (
            manifest.get("config_sha256") != plan.canonical_sha256
            or manifest.get("prediction_seal_sha256") != prediction_seal.seal_sha256
            or manifest.get("exposure_seal_sha256") != exposure_seal.seal_sha256
        ):
            raise ConsistencyError("Selection retention sources do not match the run manifest")
        payload: dict[str, object] = {
            "version": 1,
            "status": "verified_and_pruned",
            "config_sha256": plan.canonical_sha256,
            "manifest_sha256": manifest["manifest_sha256"],
            "prediction_seal_sha256": prediction_seal.seal_sha256,
            "prediction_ledger_sha256": prediction_seal.ledger_sha256,
            "exposure_seal_sha256": exposure_seal.seal_sha256,
            "exposure_ledger_sha256": exposure_seal.ledger_sha256,
            "exposure_examples": exposure_seal.examples,
            "checkpoint_generation": checkpoint.generation,
            "checkpoint_manifest_sha256": checkpoint.manifest_sha256,
            "pruned": ["checkpoints", "exposures"],
        }
        write_json_atomic(
            receipt_path,
            _hashed_payload(payload, "retention_sha256"),
        )
        _fsync_directory(root)
        _safe_remove_child(root, "checkpoints")
        _safe_remove_child(root, "exposures")

    def _prepared_receipt(
        self,
        root: Path,
        *,
        plan: ProductionSelectionPlan,
        feature_policy: FeaturePolicyName,
        status: str,
        failure_reason: str | None,
    ) -> PreparedSelectionCandidate:
        path = root / "prepared.json"
        payload: dict[str, object] = {
            "version": 1,
            "status": status,
            "failure_reason": failure_reason,
            "feature_policy": feature_policy,
            "plan": plan.model_dump(mode="json"),
            "config_sha256": plan.canonical_sha256,
        }
        artifact = _hashed_payload(payload, "prepared_sha256")
        if path.exists():
            stored = read_json(path)
            _verify_hashed_payload(
                stored,
                digest_name="prepared_sha256",
                description="Selection prepared-candidate receipt",
            )
            if stored != artifact:
                raise ConsistencyError("Selection prepared-candidate receipt changed")
        else:
            write_json_atomic(path, artifact)
        if status not in {"complete", "protocol_invalid"}:
            raise ConsistencyError("Selection prepared-candidate status is invalid")
        return PreparedSelectionCandidate(
            plan=plan,
            feature_policy=feature_policy,
            status=cast(Literal["complete", "protocol_invalid"], status),
            failure_reason=failure_reason,
        )

    def prepare(
        self,
        plan: ProductionSelectionPlan,
        *,
        feature_policy: FeaturePolicyName,
    ) -> PreparedSelectionCandidate:
        features = self.features[feature_policy]
        if plan.feature_policy_sha256 != features.feature_policy_sha256:
            raise ConsistencyError("Selection candidate uses the wrong feature policy")
        root = self._root(plan)
        prepared_path = root / "prepared.json"
        if prepared_path.exists():
            stored = read_json(prepared_path)
            status = stored.get("status")
            reason = stored.get("failure_reason")
            if not isinstance(status, str) or (reason is not None and not isinstance(reason, str)):
                raise ConsistencyError("Selection prepared-candidate receipt is malformed")
            prepared = self._prepared_receipt(
                root,
                plan=plan,
                feature_policy=feature_policy,
                status=status,
                failure_reason=reason,
            )
            if prepared.status == "complete":
                verify_selection_run_manifest(root / "manifest.json")
                self._verify_retention(root, plan)
                if not (root / "candidate-result.json").exists():
                    PredictionLedgerWriter(
                        root / "predictions",
                        PredictionLedgerIdentity.model_validate(
                            read_json(root / "predictions" / "identity.json")
                        ),
                    ).verify_seal()
            return prepared
        checkpoint_identity = self._checkpoint_identity(plan)
        manifest_path = root / "manifest.json"
        try:
            if not manifest_path.exists():
                resume = (root / "checkpoints" / "latest.json").exists()
                ProductionSelectionController(
                    plan=plan,
                    features=features,
                    truth=self.truth,
                    monitoring_mask=self.monitoring_mask,
                    output_root=root,
                    checkpoint_identity=checkpoint_identity,
                    resume=resume,
                ).run()
            self._compact_training_evidence(root, plan, checkpoint_identity)
        except ConsistencyError as error:
            if error.message not in _SCIENTIFIC_FAILURES:
                raise
            return self._prepared_receipt(
                root,
                plan=plan,
                feature_policy=feature_policy,
                status="protocol_invalid",
                failure_reason=error.message,
            )
        return self._prepared_receipt(
            root,
            plan=plan,
            feature_policy=feature_policy,
            status="complete",
            failure_reason=None,
        )

    def _failed_candidate(self, prepared: PreparedSelectionCandidate) -> Candidate:
        plan = prepared.plan
        common = {
            "config_sha256": plan.canonical_sha256,
            "status": "protocol_invalid",
            "mean_selection_log_loss": None,
            "measured_compute_seconds": None,
            "parameter_count": None,
            "failure_reason": prepared.failure_reason,
            "seed": 17,
        }
        if plan.stage == "model":
            return ModelCandidate.model_validate(
                {
                    **common,
                    "learning_rate": plan.learning_rate,
                    "weight_decay": plan.weight_decay,
                    "dropout": plan.dropout,
                    "feature_policy": prepared.feature_policy,
                }
            )
        if (
            plan.stage == "delayed"
            and plan.method != "complete_wait"
            and plan.wait_days is not None
        ):
            return DelayedCandidate.model_validate(
                {
                    **common,
                    "method": plan.method,
                    "wait_days": plan.wait_days,
                }
            )
        return SamplerCandidate.model_validate(
            {
                **common,
                "recent_window_days": plan.recent_window_days,
                "reservoir_capacity": plan.reservoir_capacity,
            }
        )

    def _load_result(
        self,
        path: Path,
        prepared: PreparedSelectionCandidate,
    ) -> Candidate:
        value = read_json(path)
        _verify_hashed_payload(
            value,
            digest_name="candidate_result_sha256",
            description="Selection candidate-result receipt",
        )
        raw = value.get("candidate")
        if (
            not isinstance(raw, dict)
            or value.get("config_sha256") != prepared.plan.canonical_sha256
        ):
            raise ConsistencyError("Selection candidate-result receipt is malformed")
        model: type[Candidate]
        if prepared.plan.stage == "model":
            model = ModelCandidate
        elif prepared.plan.stage == "delayed":
            model = DelayedCandidate
        else:
            model = SamplerCandidate
        candidate = model.model_validate(raw)
        if candidate.config_sha256 != prepared.plan.canonical_sha256:
            raise ConsistencyError("Selection candidate result changed its frozen plan")
        return candidate

    def score(self, prepared: PreparedSelectionCandidate) -> Candidate:
        root = self._root(prepared.plan)
        result_path = root / "candidate-result.json"
        if result_path.exists():
            candidate = self._load_result(result_path, prepared)
            _safe_remove_child(root, "predictions")
            return candidate
        if prepared.status == "protocol_invalid":
            candidate = self._failed_candidate(prepared)
            evaluation_sha256: str | None = None
        else:
            manifest = verify_selection_run_manifest(root / "manifest.json")
            evaluation = evaluate_selection_candidate(
                root,
                truth=self.truth,
                features=self.features[prepared.feature_policy],
                require_ranking_eligible=prepared.plan.phase == "selection",
            )
            evaluation_sha256 = cast(str, evaluation["evaluation_sha256"])
            if prepared.plan.stage == "model":
                candidate = completed_model_candidate(
                    prepared.plan,
                    feature_policy=prepared.feature_policy,
                    manifest=manifest,
                    evaluation=evaluation,
                )
            elif prepared.plan.stage == "delayed":
                candidate = completed_delayed_candidate(
                    prepared.plan,
                    manifest=manifest,
                    evaluation=evaluation,
                )
            else:
                candidate = completed_sampler_candidate(
                    prepared.plan,
                    manifest=manifest,
                    evaluation=evaluation,
                )
        payload: dict[str, object] = {
            "version": 1,
            "status": "aggregated",
            "config_sha256": prepared.plan.canonical_sha256,
            "evaluation_sha256": evaluation_sha256,
            "candidate": candidate.model_dump(mode="json"),
            "pruned_after_aggregate_verification": ["predictions"],
        }
        write_json_atomic(
            result_path,
            _hashed_payload(payload, "candidate_result_sha256"),
        )
        _fsync_directory(root)
        _safe_remove_child(root, "predictions")
        return candidate
