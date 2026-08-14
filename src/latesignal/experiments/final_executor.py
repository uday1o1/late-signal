"""Sequential final-run execution with verified evidence compaction and pruning."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from latesignal.data.manifests import (
    canonical_json_bytes,
    read_json,
    sha256_file,
    write_json_atomic,
)
from latesignal.errors import ConsistencyError
from latesignal.experiments.checkpoint import CheckpointIdentity, RollingCheckpointStore
from latesignal.experiments.exposures import ExposureLedgerIdentity, ExposureLedgerWriter
from latesignal.experiments.final_evaluation import (
    FinalEvaluationFeatures,
    evaluate_final_run,
    verify_final_run_manifest,
)
from latesignal.experiments.final_snapshots import (
    FinalSnapshotIdentity,
    FinalSnapshotStore,
)
from latesignal.experiments.predictions import PredictionLedgerIdentity, PredictionLedgerWriter
from latesignal.experiments.production_final import ProductionFinalPlan
from latesignal.experiments.production_final_controller import ProductionFinalController
from latesignal.experiments.production_selection import SelectionFeatureStore
from latesignal.simulator.production_oracle import ProductionTruthStore

_PRUNABLE = frozenset({"checkpoints", "exposures", "predictions", "snapshots"})


class FinalExecutionFeatures(SelectionFeatureStore, FinalEvaluationFeatures, Protocol):
    """Combined truth-free feature interface used by final training and evaluation."""


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def _safe_remove_children(root: Path, names: tuple[str, ...]) -> None:
    if not names or len(names) != len(set(names)) or not set(names).issubset(_PRUNABLE):
        raise ConsistencyError("Final retention requested an unauthorized removal set")
    resolved_root = root.resolve()
    for name in names:
        target = resolved_root / name
        if not target.exists():
            continue
        if target.is_symlink() or not target.is_dir() or target.resolve().parent != resolved_root:
            raise ConsistencyError("Final retention target is redirected or malformed")
        shutil.rmtree(target)
        _fsync_directory(resolved_root)


class ProductionFinalExecutor:
    """Run, evaluate, compact, and safely prune one final online candidate."""

    def __init__(
        self,
        *,
        output_root: Path,
        features: FinalExecutionFeatures,
        truth: ProductionTruthStore,
        monitoring_mask: NDArray[np.bool_],
        runtime_identity: dict[str, Any],
        device_uuid: str,
    ) -> None:
        required_runtime = {
            "source_tree_sha256",
            "dependency_lock_sha256",
            "git_commit",
            "runtime_sha256",
            "git_dirty",
        }
        if (
            output_root.is_symlink()
            or features.prepared_manifest_sha256 != truth.prepared_manifest_sha256
            or monitoring_mask.shape != features.click_days.shape
            or not required_runtime.issubset(runtime_identity)
            or runtime_identity.get("git_dirty") is not False
            or not device_uuid
        ):
            raise ConsistencyError("Final executor inputs or runtime identity are incomplete")
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.features = features
        self.truth = truth
        self.monitoring_mask = np.array(monitoring_mask, dtype=np.bool_, copy=True)
        self.runtime_identity = dict(runtime_identity)
        self.device_uuid = device_uuid

    def _root(self, plan: ProductionFinalPlan) -> Path:
        study = "study-a" if plan.study == "study_a" else "study-b"
        parent = self.output_root / study / "runs"
        root = parent / plan.run_id
        if root.resolve().parent != parent.resolve():
            raise ConsistencyError("Final run ID escapes its study root")
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _checkpoint_identity(self, plan: ProductionFinalPlan) -> CheckpointIdentity:
        runtime = self.runtime_identity
        return CheckpointIdentity(
            version=1,
            phase=plan.phase,
            run_id=plan.run_id,
            config_sha256=plan.canonical_sha256,
            protocol_sha256=plan.protocol_sha256,
            protocol_lock_sha256=(plan.protocol_lock_sha256 if plan.phase == "final" else None),
            data_manifest_sha256=plan.data_manifest_sha256,
            feature_policy_sha256=plan.feature_policy_sha256,
            source_tree_sha256=str(runtime["source_tree_sha256"]),
            dependency_lock_sha256=str(runtime["dependency_lock_sha256"]),
            git_commit=str(runtime["git_commit"]),
            environment_sha256=str(runtime["runtime_sha256"]),
            device_uuid=self.device_uuid,
        )

    def _verify_compact(self, root: Path, plan: ProductionFinalPlan) -> dict[str, Any]:
        compact = read_json(root / "compact" / "manifest.json")
        _verify_hashed_payload(
            compact,
            digest_name="manifest_sha256",
            description="Compact final primary manifest",
        )
        path = root / "compact" / "primary-probabilities.npy"
        sha256, size = sha256_file(path)
        if (
            compact.get("status") != "verified_compact_primary"
            or compact.get("config_sha256") != plan.canonical_sha256
            or compact.get("probabilities_sha256") != sha256
            or compact.get("probabilities_bytes") != size
        ):
            raise ConsistencyError("Compact final primary evidence is inconsistent")
        return compact

    def _verify_retention(self, root: Path, plan: ProductionFinalPlan) -> dict[str, Any]:
        receipt = read_json(root / "retention.json")
        _verify_hashed_payload(
            receipt,
            digest_name="retention_sha256",
            description="Final retention receipt",
        )
        if (
            receipt.get("status") != "verified_and_pruned"
            or receipt.get("config_sha256") != plan.canonical_sha256
            or receipt.get("pruned") != sorted(_PRUNABLE)
        ):
            raise ConsistencyError("Final retention receipt is inconsistent")
        self._verify_compact(root, plan)
        return receipt

    def _snapshot_identities(self, plan: ProductionFinalPlan) -> tuple[FinalSnapshotIdentity, ...]:
        import math

        return tuple(
            FinalSnapshotIdentity(
                version=1,
                run_id=plan.run_id,
                method=plan.method,
                seed=plan.seed,
                config_sha256=plan.canonical_sha256,
                protocol_sha256=plan.protocol_sha256,
                protocol_lock_sha256=plan.protocol_lock_sha256,
                budget_fraction=fraction,
                credits_at_snapshot=math.ceil(fraction * plan.credits),
                total_credits=plan.credits,
            )
            for fraction in plan.intermediate_budget_fractions
        )

    def _compact_and_prune(
        self,
        root: Path,
        plan: ProductionFinalPlan,
        checkpoint_identity: CheckpointIdentity,
    ) -> dict[str, Any]:
        receipt_path = root / "retention.json"
        if receipt_path.exists():
            receipt = self._verify_retention(root, plan)
            _safe_remove_children(root, tuple(sorted(_PRUNABLE)))
            return receipt
        manifest = verify_final_run_manifest(root / "manifest.json")
        evaluation = read_json(root / "evaluation.json")
        _verify_hashed_payload(
            evaluation,
            digest_name="evaluation_sha256",
            description="Final evaluation",
        )
        compact = self._verify_compact(root, plan)
        primary_root = root / "predictions" / "primary"
        primary_identity = PredictionLedgerIdentity.model_validate(
            read_json(primary_root / "identity.json")
        )
        primary_seal = PredictionLedgerWriter(primary_root, primary_identity).verify_seal()
        exposure_identity = ExposureLedgerIdentity.model_validate(
            read_json(root / "exposures" / "identity.json")
        )
        exposure_seal = ExposureLedgerWriter(root / "exposures", exposure_identity).verify_seal()
        checkpoint = RollingCheckpointStore(root / "checkpoints").load_latest(checkpoint_identity)
        snapshots = FinalSnapshotStore(root / "snapshots").verify_exact(
            self._snapshot_identities(plan)
        )
        intermediate = manifest.get("intermediate_predictions")
        if not isinstance(intermediate, list) or len(intermediate) != 4:
            raise ConsistencyError("Final retention source has incomplete intermediate evidence")
        intermediate_seals: list[dict[str, object]] = []
        for entry in intermediate:
            if not isinstance(entry, dict):
                raise ConsistencyError("Final retention intermediate entry is malformed")
            fraction = entry.get("budget_fraction")
            if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
                raise ConsistencyError("Final retention fraction is malformed")
            name = f"fraction-{round(float(fraction) * 100):03d}"
            ledger_root = root / "predictions" / "intermediate" / name
            identity = PredictionLedgerIdentity.model_validate(
                read_json(ledger_root / "identity.json")
            )
            seal = PredictionLedgerWriter(ledger_root, identity).verify_seal()
            intermediate_seals.append(
                {
                    "budget_fraction": float(fraction),
                    "seal_sha256": seal.seal_sha256,
                    "ledger_sha256": seal.ledger_sha256,
                    "rows": seal.rows,
                }
            )
        if (
            manifest.get("config_sha256") != plan.canonical_sha256
            or manifest.get("primary_prediction_seal_sha256") != primary_seal.seal_sha256
            or evaluation.get("prediction_seal_sha256") != primary_seal.seal_sha256
            or evaluation.get("compact_primary_manifest_sha256") != compact.get("manifest_sha256")
            or manifest.get("exposure_seal_sha256") != exposure_seal.seal_sha256
        ):
            raise ConsistencyError("Final retention evidence chain does not align")
        payload: dict[str, object] = {
            "version": 1,
            "status": "verified_and_pruned",
            "run_id": plan.run_id,
            "config_sha256": plan.canonical_sha256,
            "manifest_sha256": manifest["manifest_sha256"],
            "evaluation_sha256": evaluation["evaluation_sha256"],
            "compact_primary_manifest_sha256": compact["manifest_sha256"],
            "primary_prediction_seal_sha256": primary_seal.seal_sha256,
            "primary_prediction_ledger_sha256": primary_seal.ledger_sha256,
            "intermediate_prediction_seals": intermediate_seals,
            "exposure_seal_sha256": exposure_seal.seal_sha256,
            "exposure_ledger_sha256": exposure_seal.ledger_sha256,
            "checkpoint_generation": checkpoint.generation,
            "checkpoint_manifest_sha256": checkpoint.manifest_sha256,
            "snapshot_manifest_sha256": [item.manifest_sha256 for item in snapshots],
            "pruned": sorted(_PRUNABLE),
        }
        payload["retention_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        write_json_atomic(receipt_path, payload)
        _fsync_directory(root)
        _safe_remove_children(root, tuple(sorted(_PRUNABLE)))
        return payload

    def execute(self, plan: ProductionFinalPlan) -> dict[str, object]:
        if plan.data_manifest_sha256 != self.features.prepared_manifest_sha256:
            raise ConsistencyError("Final plan uses a different prepared dataset")
        root = self._root(plan)
        checkpoint_identity = self._checkpoint_identity(plan)
        if (root / "retention.json").exists():
            manifest = verify_final_run_manifest(root / "manifest.json")
            evaluation = read_json(root / "evaluation.json")
            retention = self._verify_retention(root, plan)
            return {
                "manifest": manifest,
                "evaluation": evaluation,
                "retention": retention,
            }
        if not (root / "manifest.json").exists():
            resume = (root / "checkpoints" / "latest.json").exists()
            ProductionFinalController(
                plan=plan,
                features=self.features,
                truth=self.truth,
                monitoring_mask=self.monitoring_mask,
                output_root=root,
                checkpoint_identity=checkpoint_identity,
                resume=resume,
            ).run()
        manifest = verify_final_run_manifest(root / "manifest.json")
        evaluation = evaluate_final_run(root, truth=self.truth, features=self.features)
        retention = self._compact_and_prune(root, plan, checkpoint_identity)
        return {"manifest": manifest, "evaluation": evaluation, "retention": retention}
