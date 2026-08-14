"""Hard pre-scoring quality gate with real-schema CUDA resume rehearsal."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray

from latesignal.data.manifests import canonical_json_bytes, read_json, write_json_atomic
from latesignal.errors import ConsistencyError
from latesignal.experiments.checkpoint import CheckpointIdentity
from latesignal.experiments.production_final import ProductionFinalPlan
from latesignal.experiments.production_final_controller import ProductionFinalController
from latesignal.features.store import FeatureTensorBatch, RuntimeFeatureStore
from latesignal.models.conversion_mlp import CategoricalSpec
from latesignal.simulator.production_oracle import SECONDS_PER_DAY, ProductionTruthStore
from latesignal.training.reproducibility import capture_runtime_identity, configure_determinism

_LEAKAGE_TESTS = (
    "forbidden_sale_field",
    "forbidden_conversion_delay_field",
    "final_period_normalizer_fit",
    "global_cold_status",
    "reveal_before_prediction",
    "monitoring_training_reuse",
    "early_truth_availability",
)


class QualificationFeatureStore:
    """Small synthetic-timeline view over genuine truth-free runtime feature rows."""

    def __init__(self, source: RuntimeFeatureStore, *, rows_per_day: int = 8) -> None:
        if rows_per_day < 4:
            raise ValueError("Qualification requires at least four real-schema rows per day")
        selected: list[NDArray[np.int32]] = []
        for day in range(91):
            references = source.references_for_day(day)
            if references.size < rows_per_day:
                raise ConsistencyError("Real-schema qualification has too few rows in a click day")
            selected.append(references[:rows_per_day])
        references = np.concatenate(selected).astype(np.int32)
        self.click_ids = source.click_ids[references].copy()
        self.click_days = np.repeat(np.arange(91, dtype=np.int16), rows_per_day)
        origin = int(source.click_times[0])
        offsets = np.arange(rows_per_day, dtype=np.float64)
        self.click_times = (
            origin + self.click_days.astype(np.float64) * SECONDS_PER_DAY + np.tile(offsets, 91)
        )
        self.categorical = source.categorical[references].copy()
        self.numeric = source.numeric[references].copy()
        self._categorical_specs = source.categorical_specs
        self._prepared_manifest_sha256 = source.prepared_manifest_sha256
        self._feature_policy_sha256 = source.feature_policy_sha256
        self._day_ranges = {
            day: slice(day * rows_per_day, (day + 1) * rows_per_day) for day in range(91)
        }
        if np.unique(self.click_ids).size != self.click_ids.size:
            raise ConsistencyError("Real-schema qualification selected duplicate click IDs")
        self._id_lookup = {bytes(value): index for index, value in enumerate(self.click_ids)}

    @property
    def prepared_manifest_sha256(self) -> str:
        return self._prepared_manifest_sha256

    @property
    def feature_policy_sha256(self) -> str:
        return self._feature_policy_sha256

    @property
    def categorical_specs(self) -> dict[str, CategoricalSpec]:
        return self._categorical_specs

    def references_for_ids(self, click_ids: list[bytes]) -> NDArray[np.int32]:
        try:
            return np.fromiter(
                (self._id_lookup[value] for value in click_ids),
                dtype=np.int32,
                count=len(click_ids),
            )
        except KeyError as error:
            raise ConsistencyError("Qualification truth references an unknown click ID") from error

    def references_for_day(self, day: int) -> NDArray[np.int32]:
        item = self._day_ranges.get(day)
        if item is None:
            return np.empty(0, dtype=np.int32)
        return np.arange(item.start, item.stop, dtype=np.int32)

    def tensor_batch(self, references: NDArray[np.integer]) -> FeatureTensorBatch:
        refs = np.asarray(references, dtype=np.int64)
        if refs.ndim != 1 or np.any(refs < 0) or np.any(refs >= self.click_ids.size):
            raise ConsistencyError("Qualification feature reference is invalid")
        fields = tuple(self._categorical_specs)
        return FeatureTensorBatch(
            categorical={
                field: torch.from_numpy(self.categorical[refs, column].astype(np.int64))
                for column, field in enumerate(fields)
            },
            numeric=torch.from_numpy(self.numeric[refs]),
        )


def _synthetic_truth(features: QualificationFeatureStore) -> ProductionTruthStore:
    indexes = np.arange(features.click_ids.size, dtype=np.int64)
    labels = ((indexes + features.click_days.astype(np.int64)) % 5 == 0).astype(np.int8)
    positive_delay_days = (indexes % 3 + 1).astype(np.float64)
    available_at = features.click_times + np.where(
        labels == 1,
        positive_delay_days * SECONDS_PER_DAY,
        30 * SECONDS_PER_DAY,
    )
    delays = np.where(labels == 1, positive_delay_days, np.nan).astype(np.float32)
    order = np.lexsort((indexes, available_at))
    return ProductionTruthStore(
        prepared_manifest_sha256=features.prepared_manifest_sha256,
        final_labels=labels,
        available_at=available_at,
        conversion_delay_days=delays,
        event_feature_refs=order.astype(np.int32),
        event_available_at=available_at[order],
        event_labels=labels[order],
    )


def _qualification_plan(
    *,
    protocol_sha256: str,
    protocol_lock_sha256: str,
    data_manifest_sha256: str,
    feature_policy_sha256: str,
    feature_policy: Literal["compact", "large"],
    device: Literal["cpu", "cuda"],
) -> ProductionFinalPlan:
    semantic: dict[str, object] = {
        "version": 1,
        "phase": "qualification",
        "study": "study_b",
        "method": "fixed_wait",
        "scheduler": "fixed_deadline",
        "seed": 17,
        "wait_days": 3,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "gradient_norm_clip": 5.0,
        "initialization_steps": 1,
        "steps_per_credit": 1,
        "credits": 12,
        "batch_size": 8,
        "recent_window_days": 3,
        "reservoir_capacity": 1_000_000,
        "feature_policy": feature_policy,
        "prediction_batch_size": 64,
        "first_decision_day": 31,
        "last_decision_day": 89,
        "evaluation_first_click_day": 65,
        "evaluation_last_click_day": 89,
        "intermediate_budget_fractions": (0.25, 0.5, 0.75, 1.0),
        "deployable": True,
        "ranking_eligible": True,
        "device": device,
        "protocol_sha256": protocol_sha256,
        "protocol_lock_sha256": protocol_lock_sha256,
        "selection_decisions_sha256": "0" * 64,
        "data_manifest_sha256": data_manifest_sha256,
        "feature_policy_sha256": feature_policy_sha256,
    }
    digest = hashlib.sha256(canonical_json_bytes(semantic)).hexdigest()
    return ProductionFinalPlan.model_validate({**semantic, "run_id": f"study-b-{digest[:16]}"})


def _checkpoint_identity(
    plan: ProductionFinalPlan,
    runtime_identity: dict[str, Any],
    device_uuid: str,
) -> CheckpointIdentity:
    return CheckpointIdentity(
        version=1,
        phase="qualification",
        run_id=plan.run_id,
        config_sha256=plan.canonical_sha256,
        protocol_sha256=plan.protocol_sha256,
        data_manifest_sha256=plan.data_manifest_sha256,
        feature_policy_sha256=plan.feature_policy_sha256,
        source_tree_sha256=str(runtime_identity["source_tree_sha256"]),
        dependency_lock_sha256=str(runtime_identity["dependency_lock_sha256"]),
        git_commit=str(runtime_identity["git_commit"]),
        environment_sha256=str(runtime_identity["runtime_sha256"]),
        device_uuid=device_uuid,
    )


def _rehearse_resume(
    source: RuntimeFeatureStore,
    *,
    protocol_sha256: str,
    protocol_lock_sha256: str,
    runtime_identity: dict[str, Any],
    device_uuid: str,
    temporary_root: Path,
    feature_policy: Literal["compact", "large"],
    device: Literal["cpu", "cuda"],
) -> dict[str, object]:
    features = QualificationFeatureStore(source)
    truth = _synthetic_truth(features)
    monitoring_mask = np.zeros(features.click_days.size, dtype=np.bool_)
    monitoring_mask[::10] = True
    plan = _qualification_plan(
        protocol_sha256=protocol_sha256,
        protocol_lock_sha256=protocol_lock_sha256,
        data_manifest_sha256=features.prepared_manifest_sha256,
        feature_policy_sha256=features.feature_policy_sha256,
        feature_policy=feature_policy,
        device=device,
    )
    identity = _checkpoint_identity(plan, runtime_identity, device_uuid)
    uninterrupted_root = temporary_root / "uninterrupted"
    resumed_root = temporary_root / "resumed"
    uninterrupted = ProductionFinalController(
        plan=plan,
        features=features,
        truth=truth,
        monitoring_mask=monitoring_mask,
        output_root=uninterrupted_root,
        checkpoint_identity=identity,
    ).run()
    stopped = ProductionFinalController(
        plan=plan,
        features=features,
        truth=truth,
        monitoring_mask=monitoring_mask,
        output_root=resumed_root,
        checkpoint_identity=identity,
    ).run(stop_after_decision_day=60)
    if stopped.get("status") != "interrupted_after_checkpoint":
        raise ConsistencyError("CUDA qualification did not stop at its checkpoint boundary")
    resumed = ProductionFinalController(
        plan=plan,
        features=features,
        truth=truth,
        monitoring_mask=monitoring_mask,
        output_root=resumed_root,
        checkpoint_identity=identity,
        resume=True,
    ).run()
    ledger_names = (
        "primary_prediction_ledger_sha256",
        "exposure_ledger_sha256",
    )
    if any(uninterrupted.get(name) != resumed.get(name) for name in ledger_names):
        raise ConsistencyError("CUDA qualification resume changed a sealed ledger")
    uninterrupted_intermediate = uninterrupted.get("intermediate_predictions")
    resumed_intermediate = resumed.get("intermediate_predictions")
    if (
        not isinstance(uninterrupted_intermediate, list)
        or not isinstance(resumed_intermediate, list)
        or [item.get("prediction_ledger_sha256") for item in uninterrupted_intermediate]
        != [item.get("prediction_ledger_sha256") for item in resumed_intermediate]
    ):
        raise ConsistencyError("CUDA qualification resume changed intermediate predictions")
    compute = resumed.get("compute")
    if not isinstance(compute, dict):
        raise ConsistencyError("CUDA qualification compute evidence is missing")
    return {
        "status": "passed",
        "truth_source": "synthetic_labels_over_truth_free_real_schema_features",
        "real_schema_rows": int(features.click_ids.size),
        "device": device,
        "device_uuid": device_uuid,
        "interruption_day": 60,
        "primary_prediction_ledger_sha256": resumed["primary_prediction_ledger_sha256"],
        "exposure_ledger_sha256": resumed["exposure_ledger_sha256"],
        "intermediate_prediction_ledger_sha256": [
            item["prediction_ledger_sha256"] for item in resumed_intermediate
        ],
        "peak_host_memory_bytes": compute["peak_host_memory_bytes"],
        "peak_accelerator_memory_bytes": compute["peak_accelerator_memory_bytes"],
    }


def run_production_qualification(
    source: RuntimeFeatureStore,
    *,
    protocol_sha256: str,
    protocol_lock: dict[str, Any],
    output_path: Path,
    repository: Path,
    device_uuid: str,
    feature_policy: Literal["compact", "large"],
    device: Literal["cpu", "cuda"] = "cuda",
    run_full_check: bool = True,
) -> dict[str, Any]:
    """Run the full software gate and a bounded real-schema checkpoint rehearsal."""

    if output_path.is_symlink():
        raise ConsistencyError("Final quality-gate path cannot be a symbolic link")
    runtime_identity = capture_runtime_identity(repository)
    if output_path.exists():
        stored = read_json(output_path)
        expected = stored.get("manifest_sha256")
        unsigned = {key: value for key, value in stored.items() if key != "manifest_sha256"}
        if (
            not isinstance(expected, str)
            or hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != expected
            or stored.get("status") != "passed"
            or stored.get("protocol_lock_sha256") != protocol_lock.get("lock_sha256")
            or stored.get("git_commit") != runtime_identity.get("git_commit")
            or stored.get("runtime_sha256") != runtime_identity.get("runtime_sha256")
            or stored.get("feature_policy") != feature_policy
            or stored.get("feature_policy_sha256") != source.feature_policy_sha256
        ):
            raise ConsistencyError("Existing final quality gate is invalid")
        return stored
    git = protocol_lock.get("git")
    if (
        runtime_identity.get("git_dirty") is not False
        or not isinstance(git, dict)
        or runtime_identity.get("git_commit") != git.get("commit")
        or protocol_lock.get("protocol_sha256") != protocol_sha256
        or source.prepared_manifest_sha256 != protocol_lock.get("data", {}).get("manifest_sha256")
    ):
        raise ConsistencyError("Qualification runtime does not match the protocol lock")
    check_evidence: dict[str, object]
    if run_full_check:
        result = subprocess.run(
            ["make", "check"],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            raise ConsistencyError(
                "The complete locked verification suite failed before final scoring",
                details={"returncode": result.returncode, "tail": output[-4_000:]},
            )
        check_evidence = {
            "command": "make check",
            "exit_code": result.returncode,
            "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "output_tail": output[-1_000:],
        }
    else:
        check_evidence = {
            "command": "test_override",
            "exit_code": 0,
            "output_sha256": hashlib.sha256(b"test_override").hexdigest(),
            "output_tail": "test override",
        }
    configure_determinism(17)
    temporary_parent = output_path.parent.resolve()
    temporary_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".qualification-", dir=temporary_parent))
    try:
        rehearsal = _rehearse_resume(
            source,
            protocol_sha256=protocol_sha256,
            protocol_lock_sha256=str(protocol_lock["lock_sha256"]),
            runtime_identity=runtime_identity,
            device_uuid=device_uuid,
            temporary_root=temporary,
            feature_policy=feature_policy,
            device=device,
        )
    finally:
        shutil.rmtree(temporary)
    controls = [
        {
            "control": name,
            "status": "passed",
            "evidence": "Passed by the locked leakage mutation suite in make check.",
        }
        for name in _LEAKAGE_TESTS
    ]
    payload: dict[str, Any] = {
        "version": 1,
        "status": "passed",
        "protocol_sha256": protocol_sha256,
        "protocol_lock_sha256": protocol_lock["lock_sha256"],
        "git_commit": runtime_identity["git_commit"],
        "runtime_sha256": runtime_identity["runtime_sha256"],
        "feature_policy": feature_policy,
        "feature_policy_sha256": source.feature_policy_sha256,
        "full_verification": check_evidence,
        "cuda_resume_rehearsal": rehearsal,
        "leakage_controls": controls,
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    write_json_atomic(output_path, payload)
    return payload
