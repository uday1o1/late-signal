"""Resumable event-time controller for locked production Study A and Study B runs."""

from __future__ import annotations

import copy
import hashlib
import math
import resource
import sys
import time
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from latesignal.data.manifests import canonical_json_bytes, write_json_atomic
from latesignal.errors import ConsistencyError
from latesignal.experiments.checkpoint import CheckpointIdentity, RollingCheckpointStore
from latesignal.experiments.exposures import ExposureLedgerIdentity, ExposureLedgerWriter
from latesignal.experiments.final_snapshots import (
    FinalSnapshotIdentity,
    FinalSnapshotStore,
)
from latesignal.experiments.predictions import PredictionLedgerIdentity, PredictionLedgerWriter
from latesignal.experiments.production_final import ProductionFinalPlan
from latesignal.experiments.production_selection import SelectionFeatureStore
from latesignal.methods.production import (
    MethodBoundaryResult,
    PackedDelayedMethod,
)
from latesignal.scheduling.base import CreditScheduler
from latesignal.scheduling.calibration_drift import CalibrationDriftCreditScheduler
from latesignal.scheduling.credit import SpendDecision, build_credit_windows
from latesignal.scheduling.fixed import FixedWindowScheduler
from latesignal.scheduling.production import DailyRangeCreditScheduler, PackedMonitoringState
from latesignal.simulator.production_oracle import (
    SECONDS_PER_DAY,
    ProductionTruthCursor,
    ProductionTruthStore,
    TruthEventBatch,
)
from latesignal.training.packed import (
    PackedDeterministicSampler,
    PackedRecordStore,
    RecordKind,
    packed_record_batch,
)
from latesignal.training.production import PackedConversionTrainer, ProductionTrainingConfig
from latesignal.training.production_dfm import PackedDFMTrainer
from latesignal.training.production_esdfm import ESDFMAuxiliaryPair

HOUR_SECONDS = 3_600
FINAL_FIRST_CLICK_DAY = 65
FINAL_LAST_CLICK_DAY = 89
FINAL_FIRST_DECISION_DAY = 31
FINAL_LAST_DECISION_DAY = 89
FINAL_SEAL_DAY = 90
FINAL_TRUTH_DRAIN_DAY = 120


class _FinalTrainer(Protocol):
    model: nn.Module
    budget: Any
    model_version: int

    def predict(self, references: NDArray[np.integer]) -> NDArray[np.float32]: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class _PackedOracleReference:
    """Emit privileged eventual labels at click time only for the explicit oracle run."""

    name = "oracle_reference"
    deployable = False
    ranking_eligible = False

    def __init__(
        self,
        *,
        click_times: NDArray[np.float64],
        final_labels: NDArray[np.int8],
        monitoring_mask: NDArray[np.bool_],
        main_store: PackedRecordStore,
    ) -> None:
        if (
            click_times.ndim != 1
            or click_times.size == 0
            or final_labels.shape != click_times.shape
            or monitoring_mask.shape != click_times.shape
            or main_store.feature_count != click_times.size
            or np.any(np.diff(click_times) < 0.0)
            or np.any((final_labels != 0) & (final_labels != 1))
        ):
            raise ConsistencyError("Production oracle input contract is invalid")
        self.click_times = np.array(click_times, dtype=np.float64, copy=True)
        self.final_labels = np.array(final_labels, dtype=np.int8, copy=True)
        self.monitoring_mask = np.array(monitoring_mask, dtype=np.bool_, copy=True)
        self.main_store = main_store
        self.click_cursor = 0
        self.last_time: float | None = None
        self.config_sha256 = hashlib.sha256(
            canonical_json_bytes(
                {
                    "name": self.name,
                    "click_times_sha256": hashlib.sha256(self.click_times.tobytes()).hexdigest(),
                    "final_labels_sha256": hashlib.sha256(self.final_labels.tobytes()).hexdigest(),
                    "monitoring_sha256": hashlib.sha256(
                        np.packbits(self.monitoring_mask, bitorder="little").tobytes()
                    ).hexdigest(),
                }
            )
        ).hexdigest()

    def process_boundary(
        self,
        *,
        boundary: float,
        click_refs: NDArray[np.int32],
        truth: TruthEventBatch,
    ) -> MethodBoundaryResult:
        del truth
        expected = np.arange(
            self.click_cursor,
            self.click_cursor + click_refs.size,
            dtype=np.int32,
        )
        if (
            not np.isfinite(boundary)
            or (self.last_time is not None and boundary < self.last_time)
            or not np.array_equal(click_refs, expected)
            or np.any(self.click_times[click_refs] > boundary)
        ):
            raise ConsistencyError("Production oracle click cursor is not chronological")
        self.click_cursor += click_refs.size
        legal_refs = click_refs[~self.monitoring_mask[click_refs]]
        if legal_refs.size:
            self.main_store.append(
                packed_record_batch(
                    feature_refs=legal_refs,
                    available_at=self.click_times[legal_refs],
                    targets=self.final_labels[legal_refs].astype(np.float32),
                    kinds=np.full(legal_refs.size, RecordKind.FINAL, dtype=np.uint8),
                ),
                simulator_time=boundary,
            )
        self.last_time = boundary
        return MethodBoundaryResult(
            main_records=int(legal_refs.size),
            q_tn_records=0,
            q_dp_records=0,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "config_sha256": self.config_sha256,
            "click_cursor": self.click_cursor,
            "last_time": self.last_time,
            "main_store": self.main_store.rebuild_token(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        cursor = state.get("click_cursor")
        last_time = state.get("last_time")
        if (
            set(state)
            != {
                "version",
                "config_sha256",
                "click_cursor",
                "last_time",
                "main_store",
            }
            or state.get("version") != 1
            or state.get("config_sha256") != self.config_sha256
            or isinstance(cursor, bool)
            or not isinstance(cursor, int)
            or not 0 <= cursor <= self.click_times.size
            or (
                last_time is not None
                and (
                    isinstance(last_time, bool)
                    or not isinstance(last_time, (int, float))
                    or not np.isfinite(float(last_time))
                )
            )
            or state.get("main_store") != self.main_store.rebuild_token()
        ):
            raise ConsistencyError("Production oracle checkpoint state is malformed")
        if last_time is None:
            if cursor != 0:
                raise ConsistencyError("Production oracle has a cursor without time")
        elif cursor != int(np.searchsorted(self.click_times, float(last_time), side="right")):
            raise ConsistencyError("Production oracle checkpoint cursor is inconsistent")
        self.click_cursor = cursor
        self.last_time = None if last_time is None else float(last_time)


def _model_sha256(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(canonical_json_bytes(list(value.shape)))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _split_component_state(
    name: str,
    state: dict[str, object],
    *,
    model: dict[str, object],
    optimizer: dict[str, object],
    rng: dict[str, object],
) -> None:
    required = {"model", "optimizer", "cpu_rng_state", "cuda_rng_state"}
    if not required.issubset(state):
        raise ConsistencyError(f"Final checkpoint component {name} is incomplete")
    model[name] = state["model"]
    optimizer[name] = {
        "optimizer": state["optimizer"],
        "metadata": {key: value for key, value in state.items() if key not in required},
    }
    rng[name] = {
        "cpu_rng_state": state["cpu_rng_state"],
        "cuda_rng_state": state["cuda_rng_state"],
    }


def _join_component_state(
    name: str,
    *,
    model: dict[str, Any],
    optimizer: dict[str, Any],
    rng: dict[str, Any],
) -> dict[str, Any]:
    model_state = model.get(name)
    optimizer_section = optimizer.get(name)
    rng_section = rng.get(name)
    if (
        not isinstance(model_state, dict)
        or not isinstance(optimizer_section, dict)
        or not isinstance(rng_section, dict)
        or not isinstance(optimizer_section.get("optimizer"), dict)
        or not isinstance(optimizer_section.get("metadata"), dict)
        or not isinstance(rng_section.get("cpu_rng_state"), Tensor)
        or not isinstance(rng_section.get("cuda_rng_state"), list)
    ):
        raise ConsistencyError(f"Final checkpoint component {name} is malformed")
    return {
        **optimizer_section["metadata"],
        "model": model_state,
        "optimizer": optimizer_section["optimizer"],
        "cpu_rng_state": rng_section["cpu_rng_state"],
        "cuda_rng_state": rng_section["cuda_rng_state"],
    }


class ProductionFinalController:
    """Replay one locked final run with daily recovery and truth-free prediction seals."""

    def __init__(
        self,
        *,
        plan: ProductionFinalPlan,
        features: SelectionFeatureStore,
        truth: ProductionTruthStore,
        monitoring_mask: NDArray[np.bool_],
        output_root: Path,
        checkpoint_identity: CheckpointIdentity,
        resume: bool = False,
    ) -> None:
        if output_root.is_symlink():
            raise ConsistencyError("Final output root cannot be a symlink")
        if (output_root / "manifest.json").exists():
            raise ConsistencyError("Completed final run cannot be overwritten")
        if output_root.exists() and any(output_root.iterdir()) and not resume:
            raise ConsistencyError("Final output root is not empty")
        if (
            features.prepared_manifest_sha256 != plan.data_manifest_sha256
            or truth.prepared_manifest_sha256 != plan.data_manifest_sha256
            or monitoring_mask.shape != features.click_days.shape
            or features.click_ids.shape != features.click_days.shape
            or features.click_times.shape != features.click_days.shape
            or features.click_days.size == 0
            or features.click_days[0] != 0
            or np.any(np.diff(features.click_times) < 0.0)
            or np.any(np.diff(features.click_days) < 0)
        ):
            raise ConsistencyError("Final data identities or feature order changed")
        expected_lock = plan.protocol_lock_sha256 if plan.phase == "final" else None
        if (
            checkpoint_identity.phase != plan.phase
            or checkpoint_identity.config_sha256 != plan.canonical_sha256
            or checkpoint_identity.protocol_sha256 != plan.protocol_sha256
            or checkpoint_identity.protocol_lock_sha256 != expected_lock
            or checkpoint_identity.data_manifest_sha256 != plan.data_manifest_sha256
            or checkpoint_identity.feature_policy_sha256 != plan.feature_policy_sha256
        ):
            raise ConsistencyError("Final checkpoint identity does not match the run plan")
        checkpoint_identity.validate_phase()
        self.plan = plan
        self.features = features
        self.truth_store = truth
        self.monitoring_mask = np.array(monitoring_mask, dtype=np.bool_, copy=True)
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_identity = checkpoint_identity
        self.origin = int(features.click_times[0])
        if float(self.origin) != float(features.click_times[0]):
            raise ConsistencyError("Final time origin must be an integral second")
        self.training_limit = int(
            np.searchsorted(features.click_days, FINAL_LAST_CLICK_DAY + 1, side="left")
        )
        self.final_refs = np.flatnonzero(
            (features.click_days >= FINAL_FIRST_CLICK_DAY)
            & (features.click_days <= FINAL_LAST_CLICK_DAY)
        ).astype(np.int32)
        if self.training_limit == 0 or self.final_refs.size == 0:
            raise ConsistencyError("Final authored training or evaluation period is empty")
        covered_days = set(np.unique(features.click_days[: self.training_limit]).tolist())
        if covered_days != set(range(FINAL_LAST_CLICK_DAY + 1)):
            raise ConsistencyError("Final data does not cover every authored click day")
        self._build_components(include_initialization=not resume)
        self.checkpoints = RollingCheckpointStore(self.output_root / "checkpoints")
        self.exposures = ExposureLedgerWriter(
            self.output_root / "exposures",
            ExposureLedgerIdentity(
                version=1,
                phase=plan.phase,
                run_id=plan.run_id,
                method=plan.method,
                seed=plan.seed,
                config_sha256=plan.canonical_sha256,
                protocol_sha256=plan.protocol_sha256,
                protocol_lock_sha256=expected_lock,
                expected_credits=plan.credits,
                steps_per_credit=plan.steps_per_credit,
                batch_size=plan.batch_size,
            ),
        )
        final_ids_sha256 = hashlib.sha256(
            self.features.click_ids[self.final_refs].tobytes()
        ).hexdigest()
        self.primary_predictions = PredictionLedgerWriter(
            self.output_root / "predictions" / "primary",
            PredictionLedgerIdentity(
                version=1,
                kind="final_prequential",
                run_id=plan.run_id,
                method=plan.method,
                seed=plan.seed,
                period_first_day=FINAL_FIRST_CLICK_DAY,
                period_last_day=FINAL_LAST_CLICK_DAY,
                protocol_sha256=plan.protocol_sha256,
                protocol_lock_sha256=plan.protocol_lock_sha256,
                config_sha256=plan.canonical_sha256,
                data_manifest_sha256=plan.data_manifest_sha256,
                expected_rows=int(self.final_refs.size),
                expected_ordered_id_sha256=final_ids_sha256,
                ranking_eligible=plan.ranking_eligible,
            ),
        )
        self.snapshots = FinalSnapshotStore(self.output_root / "snapshots")
        self.snapshot_identities = tuple(
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
        self.next_boundary_index = 0
        self.primary_part_index = 0
        self.initialization_complete = False
        self.initialization_model_sha256: str | None = None
        self.compute: dict[str, Any] = {
            "initialization_seconds": 0.0,
            "core_training_seconds": 0.0,
            "auxiliary_training_seconds": 0.0,
            "primary_prediction_seconds": 0.0,
            "intermediate_prediction_seconds": 0.0,
            "monitoring_prediction_seconds": 0.0,
            "checkpoint_seconds": 0.0,
            "snapshot_seconds": 0.0,
            "peak_host_memory_bytes": 0,
            "peak_accelerator_memory_bytes": 0,
            "feature_rows": 0,
            "truth_events": 0,
            "main_records": 0,
            "q_tn_records": 0,
            "q_dp_records": 0,
            "credits": [],
            "auxiliary": [],
            "snapshots": [],
        }
        if resume:
            self._restore_latest()
        elif (self.output_root / "checkpoints" / "latest.json").exists():
            raise ConsistencyError("Existing final checkpoint requires resume mode")
        if self.plan.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _update_peak_memory(self) -> None:
        resident = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            resident *= 1024
        self.compute["peak_host_memory_bytes"] = max(
            int(self.compute.get("peak_host_memory_bytes", 0)), resident
        )
        if self.plan.device == "cuda" and torch.cuda.is_available():
            accelerator = int(torch.cuda.max_memory_allocated())
            self.compute["peak_accelerator_memory_bytes"] = max(
                int(self.compute.get("peak_accelerator_memory_bytes", 0)), accelerator
            )

    @property
    def _main_method(self) -> PackedDelayedMethod | _PackedOracleReference:
        return self.method

    def _build_components(self, *, include_initialization: bool) -> None:
        count = int(self.features.click_days.size)
        self.main_store = PackedRecordStore(feature_count=count)
        self.q_tn_store = (
            PackedRecordStore(feature_count=count) if self.plan.method == "es_dfm" else None
        )
        self.q_dp_store = (
            PackedRecordStore(feature_count=count) if self.plan.method == "es_dfm" else None
        )
        if self.plan.method == "oracle_reference":
            self.method: PackedDelayedMethod | _PackedOracleReference = _PackedOracleReference(
                click_times=self.features.click_times,
                final_labels=self.truth_store.final_labels,
                monitoring_mask=self.monitoring_mask,
                main_store=self.main_store,
            )
        else:
            self.method = PackedDelayedMethod(
                self.plan.method,
                click_times=self.features.click_times,
                monitoring_mask=self.monitoring_mask,
                main_store=self.main_store,
                wait_days=self.plan.wait_days,
                q_tn_store=self.q_tn_store,
                q_dp_store=self.q_dp_store,
            )
        recent_seconds = self.plan.recent_window_days * SECONDS_PER_DAY
        self.main_sampler = PackedDeterministicSampler(
            self.main_store,
            seed=self.plan.seed,
            recent_window_seconds=recent_seconds,
            reservoir_capacity=self.plan.reservoir_capacity,
        )
        self.q_tn_sampler = (
            None
            if self.q_tn_store is None
            else PackedDeterministicSampler(
                self.q_tn_store,
                seed=self.plan.seed + 1000,
                recent_window_seconds=recent_seconds,
                reservoir_capacity=self.plan.reservoir_capacity,
            )
        )
        self.q_dp_sampler = (
            None
            if self.q_dp_store is None
            else PackedDeterministicSampler(
                self.q_dp_store,
                seed=self.plan.seed + 2000,
                recent_window_seconds=recent_seconds,
                reservoir_capacity=self.plan.reservoir_capacity,
            )
        )
        self.initialization_store: PackedRecordStore | None = None
        self.initialization_method: PackedDelayedMethod | None = None
        self.initialization_sampler: PackedDeterministicSampler | None = None
        self.initialization_trainer: PackedConversionTrainer | None = None
        self.q_tn_initialization_store: PackedRecordStore | None = None
        self.q_dp_initialization_store: PackedRecordStore | None = None
        self.q_tn_initialization_sampler: PackedDeterministicSampler | None = None
        self.q_dp_initialization_sampler: PackedDeterministicSampler | None = None
        if include_initialization:
            initialization_exclusion = self.monitoring_mask | (self.features.click_days != 0)
            self.initialization_store = PackedRecordStore(feature_count=count)
            self.initialization_method = PackedDelayedMethod(
                "complete_wait",
                click_times=self.features.click_times,
                monitoring_mask=initialization_exclusion,
                main_store=self.initialization_store,
            )
            self.initialization_sampler = PackedDeterministicSampler(
                self.initialization_store,
                seed=self.plan.seed,
                recent_window_seconds=recent_seconds,
                reservoir_capacity=self.plan.reservoir_capacity,
            )
            if self.plan.method == "es_dfm":
                self.q_tn_initialization_store = PackedRecordStore(feature_count=count)
                self.q_dp_initialization_store = PackedRecordStore(feature_count=count)
                self.q_tn_initialization_sampler = PackedDeterministicSampler(
                    self.q_tn_initialization_store,
                    seed=self.plan.seed + 1000,
                    recent_window_seconds=recent_seconds,
                    reservoir_capacity=self.plan.reservoir_capacity,
                )
                self.q_dp_initialization_sampler = PackedDeterministicSampler(
                    self.q_dp_initialization_store,
                    seed=self.plan.seed + 2000,
                    recent_window_seconds=recent_seconds,
                    reservoir_capacity=self.plan.reservoir_capacity,
                )
            initialization_config = ProductionTrainingConfig(
                learning_rate=self.plan.learning_rate,
                weight_decay=self.plan.weight_decay,
                dropout=self.plan.dropout,
                gradient_norm_clip=self.plan.gradient_norm_clip,
                steps_per_credit=self.plan.initialization_steps,
                batch_size=self.plan.batch_size,
                loss_mode="bce",
            )
            self.initialization_trainer = PackedConversionTrainer.create(
                self.features,
                initialization_config,
                seed=self.plan.seed,
                device=self.plan.device,
            )
        loss_mode: Literal["bce", "fnw", "es_dfm"] = (
            "fnw"
            if self.plan.method == "fnw"
            else "es_dfm"
            if self.plan.method == "es_dfm"
            else "bce"
        )
        main_config = ProductionTrainingConfig(
            learning_rate=self.plan.learning_rate,
            weight_decay=self.plan.weight_decay,
            dropout=self.plan.dropout,
            gradient_norm_clip=self.plan.gradient_norm_clip,
            steps_per_credit=self.plan.steps_per_credit,
            batch_size=self.plan.batch_size,
            loss_mode=loss_mode,
        )
        self.main_trainer: PackedConversionTrainer | PackedDFMTrainer
        if self.plan.method == "dfm":
            assert isinstance(self.method, PackedDelayedMethod)
            self.main_trainer = PackedDFMTrainer.create(
                self.features,
                self.method,
                main_config,
                seed=self.plan.seed,
                device=self.plan.device,
            )
        else:
            self.main_trainer = PackedConversionTrainer.create(
                self.features,
                main_config,
                seed=self.plan.seed,
                device=self.plan.device,
            )
        self.auxiliary = (
            ESDFMAuxiliaryPair.create(
                self.features,
                training_seed=self.plan.seed,
                dropout=self.plan.dropout,
                batch_size=self.plan.batch_size,
                device=self.plan.device,
            )
            if self.plan.method == "es_dfm"
            else None
        )
        self.truth_cursor: ProductionTruthCursor = self.truth_store.cursor(
            self.features.click_days,
            first_click_day=0,
            last_click_day=FINAL_LAST_CLICK_DAY,
        )
        self.monitoring = PackedMonitoringState(
            click_days=self.features.click_days,
            monitoring_mask=self.monitoring_mask,
            inference_batch_size=self.plan.prediction_batch_size,
        )
        if self.plan.study == "study_a":
            self.scheduler: DailyRangeCreditScheduler | CreditScheduler = DailyRangeCreditScheduler(
                origin=self.origin,
                first_day=FINAL_FIRST_DECISION_DAY,
                last_day=FINAL_LAST_DECISION_DAY,
            )
        else:
            windows = build_credit_windows(origin=self.origin)
            if self.plan.scheduler.startswith("fixed_"):
                policy = self.plan.scheduler.removeprefix("fixed_")
                self.scheduler = FixedWindowScheduler(windows, policy=policy)
            elif self.plan.scheduler == "calibration_drift":
                self.scheduler = CalibrationDriftCreditScheduler(windows, threshold=3.0)
            else:
                raise ConsistencyError("Unknown final Study B scheduler")

    def _copy_day_zero_auxiliary_records(
        self,
        source: PackedRecordStore,
        target: PackedRecordStore,
        *,
        simulator_time: float,
    ) -> None:
        indices = np.flatnonzero(self.features.click_days[source.feature_refs[: len(source)]] == 0)
        if indices.size == 0:
            raise ConsistencyError("INSUFFICIENT_LEGAL_AUXILIARY_POOL")
        target.append(
            packed_record_batch(
                feature_refs=source.feature_refs[indices],
                available_at=source.available_at[indices],
                targets=source.targets[indices],
                kinds=source.kinds[indices],
            ),
            simulator_time=simulator_time,
        )

    def _apply_initialization(self, boundary: int) -> None:
        if self.initialization_complete:
            raise ConsistencyError("Final shared initialization was applied twice")
        if self.initialization_trainer is None or self.initialization_sampler is None:
            raise ConsistencyError("Final shared initialization components are absent")
        started = time.perf_counter()
        result = self.initialization_trainer.spend_credit(
            credit_id=0,
            decision_time=boundary,
            sampler=self.initialization_sampler,
        )
        initialization = copy.deepcopy(self.initialization_trainer.model.state_dict())
        if isinstance(self.main_trainer, PackedDFMTrainer):
            self.main_trainer.model.conversion.load_state_dict(initialization)
        else:
            self.main_trainer.model.load_state_dict(initialization)
        self.initialization_model_sha256 = _model_sha256(initialization)
        self.compute["initialization_seconds"] += time.perf_counter() - started
        self.compute["initialization_steps"] = result.steps
        self.compute["initialization_examples"] = result.examples
        if self.auxiliary is not None:
            assert self.q_tn_store is not None and self.q_dp_store is not None
            assert self.q_tn_initialization_store is not None
            assert self.q_dp_initialization_store is not None
            assert self.q_tn_initialization_sampler is not None
            assert self.q_dp_initialization_sampler is not None
            self._copy_day_zero_auxiliary_records(
                self.q_tn_store,
                self.q_tn_initialization_store,
                simulator_time=boundary,
            )
            self._copy_day_zero_auxiliary_records(
                self.q_dp_store,
                self.q_dp_initialization_store,
                simulator_time=boundary,
            )
            auxiliary_started = time.perf_counter()
            try:
                work = self.auxiliary.update(
                    credit_id=0,
                    decision_time=boundary,
                    q_tn_sampler=self.q_tn_initialization_sampler,
                    q_dp_sampler=self.q_dp_initialization_sampler,
                )
            except ConsistencyError as error:
                if "INSUFFICIENT_LEGAL_POOL" in str(error):
                    raise ConsistencyError("INSUFFICIENT_LEGAL_AUXILIARY_POOL") from error
                raise
            self.compute["auxiliary_training_seconds"] += time.perf_counter() - auxiliary_started
            self.compute["auxiliary"].append(
                {
                    "work_id": 0,
                    "main_credit_id": None,
                    "decision_time": boundary,
                    "steps": work.auxiliary_steps,
                    "examples": work.auxiliary_examples,
                    "q_tn_loss": work.q_tn.mean_loss,
                    "q_dp_loss": work.q_dp.mean_loss,
                }
            )
        self.initialization_complete = True
        self.initialization_store = None
        self.initialization_method = None
        self.initialization_sampler = None
        self.initialization_trainer = None
        self.q_tn_initialization_store = None
        self.q_dp_initialization_store = None
        self.q_tn_initialization_sampler = None
        self.q_dp_initialization_sampler = None

    def _snapshot_due(self) -> None:
        credits = self.main_trainer.budget.credits
        identities = [
            identity
            for identity in self.snapshot_identities
            if identity.credits_at_snapshot == credits
        ]
        if not identities:
            return
        state = cast(dict[str, Tensor], self.main_trainer.model.state_dict())
        for identity in identities:
            started = time.perf_counter()
            snapshot = self.snapshots.write(
                identity,
                state,
                model_version=self.main_trainer.model_version,
            )
            self.compute["snapshot_seconds"] += time.perf_counter() - started
            snapshots = self.compute["snapshots"]
            assert isinstance(snapshots, list)
            entry = {
                "budget_fraction": identity.budget_fraction,
                "credits_at_snapshot": identity.credits_at_snapshot,
                "model_version": snapshot.model_version,
                "model_sha256": snapshot.model_sha256,
                "manifest_sha256": snapshot.manifest_sha256,
            }
            existing = next(
                (
                    value
                    for value in snapshots
                    if value.get("budget_fraction") == identity.budget_fraction
                ),
                None,
            )
            if existing is not None and existing != entry:
                raise ConsistencyError("Final snapshot compute receipt changed across resume")
            if existing is None:
                snapshots.append(entry)

    def _spend_credit(self, boundary: int) -> None:
        credit_id = self.main_trainer.budget.credits
        if self.auxiliary is not None and boundary != self.origin + 31 * int(SECONDS_PER_DAY):
            assert self.q_tn_sampler is not None and self.q_dp_sampler is not None
            auxiliary_started = time.perf_counter()
            try:
                work = self.auxiliary.update(
                    credit_id=self.auxiliary.q_tn.work_units,
                    decision_time=boundary,
                    q_tn_sampler=self.q_tn_sampler,
                    q_dp_sampler=self.q_dp_sampler,
                )
            except ConsistencyError as error:
                if "INSUFFICIENT_LEGAL_POOL" in str(error):
                    raise ConsistencyError("INSUFFICIENT_LEGAL_AUXILIARY_POOL") from error
                raise
            self.compute["auxiliary_training_seconds"] += time.perf_counter() - auxiliary_started
            self.compute["auxiliary"].append(
                {
                    "work_id": work.credit_id,
                    "main_credit_id": credit_id,
                    "decision_time": boundary,
                    "steps": work.auxiliary_steps,
                    "examples": work.auxiliary_examples,
                    "q_tn_loss": work.q_tn.mean_loss,
                    "q_dp_loss": work.q_dp.mean_loss,
                }
            )
        training_started = time.perf_counter()
        if isinstance(self.main_trainer, PackedDFMTrainer):
            result = self.main_trainer.spend_credit(
                credit_id=credit_id,
                decision_time=boundary,
                sampler=self.main_sampler,
            )
        else:
            result = self.main_trainer.spend_credit(
                credit_id=credit_id,
                decision_time=boundary,
                sampler=self.main_sampler,
                auxiliary_provider=self.auxiliary,
            )
        self.compute["core_training_seconds"] += time.perf_counter() - training_started
        self.exposures.append_credit(result.exposure)
        self.compute["credits"].append(
            {
                "credit_id": credit_id,
                "decision_time": boundary,
                "steps": result.steps,
                "examples": result.examples,
                "mean_loss": result.mean_loss,
            }
        )
        self._snapshot_due()

    def _decision(self, *, day: int, boundary: int) -> SpendDecision:
        if isinstance(self.scheduler, DailyRangeCreditScheduler):
            return self.scheduler.decide(boundary)
        started = time.perf_counter()
        evidence = self.monitoring.evidence(
            decision_day=day,
            predictor=self.main_trainer,
            model_checkpoint_sha256=_model_sha256(
                cast(dict[str, Tensor], self.main_trainer.model.state_dict())
            ),
        )
        self.compute["monitoring_prediction_seconds"] += time.perf_counter() - started
        return self.scheduler.decide(boundary, evidence)

    def _predict_primary(self, click_refs: NDArray[np.int32]) -> None:
        if (self.primary_predictions.root / "seal.json").exists() or click_refs.size == 0:
            return
        mask = (self.features.click_days[click_refs] >= FINAL_FIRST_CLICK_DAY) & (
            self.features.click_days[click_refs] <= FINAL_LAST_CLICK_DAY
        )
        refs = click_refs[mask]
        if refs.size == 0:
            return
        started = time.perf_counter()
        for start in range(0, refs.size, self.plan.prediction_batch_size):
            batch = refs[start : start + self.plan.prediction_batch_size]
            probabilities = self.main_trainer.predict(batch)
            days = self.features.click_days[batch]
            self.primary_predictions.append(
                part_index=self.primary_part_index,
                click_ids=[bytes(value).hex() for value in self.features.click_ids[batch]],
                click_days=days.astype(int).tolist(),
                probabilities=probabilities.tolist(),
                model_versions=[self.main_trainer.model_version] * int(batch.size),
            )
            self.primary_part_index += 1
        self.compute["primary_prediction_seconds"] += time.perf_counter() - started

    def _component_sections(self) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        model: dict[str, object] = {}
        optimizer: dict[str, object] = {}
        rng: dict[str, object] = {}
        _split_component_state(
            "main",
            self.main_trainer.state_dict(),
            model=model,
            optimizer=optimizer,
            rng=rng,
        )
        if self.auxiliary is not None:
            state = self.auxiliary.state_dict()
            for name in ("q_tn", "q_dp"):
                value = state[name]
                assert isinstance(value, dict)
                _split_component_state(name, value, model=model, optimizer=optimizer, rng=rng)
        return model, optimizer, rng

    def _snapshot_receipts(self) -> dict[str, str]:
        receipts: dict[str, str] = {}
        for identity in self.snapshot_identities:
            path = self.snapshots.root / identity.directory_name
            if not path.exists():
                continue
            receipts[identity.directory_name] = self.snapshots.verify(identity).manifest_sha256
        return receipts

    def _monitoring_audit(self) -> dict[str, object]:
        return {
            "config_sha256": self.monitoring.config_sha256,
            "membership_sha256": self.monitoring.membership_sha256,
            "labels_sha256": hashlib.sha256(self.monitoring.labels.tobytes()).hexdigest(),
            "reserved_examples": int(np.count_nonzero(self.monitoring.monitoring_mask)),
            "last_decision_day": self.monitoring.last_decision_day,
            "inference_examples": self.monitoring.inference_examples,
            "evidence_log": copy.deepcopy(self.monitoring.evidence_log),
        }

    def _state(self, *, next_boundary_index: int) -> dict[str, Any]:
        model, optimizer, rng = self._component_sections()
        return {
            "model": model,
            "optimizer": optimizer,
            "rng": rng,
            "cursors": {
                "origin": self.origin,
                "next_boundary_index": next_boundary_index,
                "training_limit": self.training_limit,
                "truth": self.truth_cursor.state_dict(),
            },
            "method": {
                "main_store": self.main_store.state_dict(),
                "main": self.method.state_dict(),
                "q_tn_store": None if self.q_tn_store is None else self.q_tn_store.state_dict(),
                "q_dp_store": None if self.q_dp_store is None else self.q_dp_store.state_dict(),
            },
            "scheduler": self.scheduler.state_dict(),
            "sampler": {
                "main": self.main_sampler.state_dict(),
                "q_tn": None if self.q_tn_sampler is None else self.q_tn_sampler.state_dict(),
                "q_dp": None if self.q_dp_sampler is None else self.q_dp_sampler.state_dict(),
            },
            "monitoring": self.monitoring.state_dict(),
            "ledgers": {
                "exposures": self.exposures.position().as_dict(),
                "primary_predictions": self.primary_predictions.position().as_dict(),
                "snapshots": self._snapshot_receipts(),
            },
            "compute": {
                **self.compute,
                "initialization_complete": self.initialization_complete,
                "initialization_model_sha256": self.initialization_model_sha256,
            },
        }

    def _write_checkpoint(self, *, next_boundary_index: int) -> None:
        if not self.initialization_complete or self.initialization_trainer is not None:
            raise ConsistencyError("Final checkpoint preceded compact shared initialization")
        started = time.perf_counter()
        self._update_peak_memory()
        self.checkpoints.write(
            self.checkpoint_identity,
            self._state(next_boundary_index=next_boundary_index),
        )
        self.compute["checkpoint_seconds"] += time.perf_counter() - started

    def _restore_latest(self) -> None:
        state = self.checkpoints.load_latest(self.checkpoint_identity).state
        method = state["method"]
        sampler = state["sampler"]
        cursors = state["cursors"]
        ledgers = state["ledgers"]
        compute = state["compute"]
        model = state["model"]
        optimizer = state["optimizer"]
        rng = state["rng"]
        if not all(
            isinstance(value, dict)
            for value in (method, sampler, cursors, ledgers, compute, model, optimizer, rng)
        ):
            raise ConsistencyError("Final checkpoint recovery section is malformed")
        assert isinstance(method, dict)
        assert isinstance(sampler, dict)
        assert isinstance(cursors, dict)
        assert isinstance(ledgers, dict)
        assert isinstance(compute, dict)
        assert isinstance(model, dict)
        assert isinstance(optimizer, dict)
        assert isinstance(rng, dict)
        if set(method) != {"main_store", "main", "q_tn_store", "q_dp_store"} or set(sampler) != {
            "main",
            "q_tn",
            "q_dp",
        }:
            raise ConsistencyError("Final checkpoint component set changed")
        main_store = method.get("main_store")
        main_method = method.get("main")
        if not isinstance(main_store, dict) or not isinstance(main_method, dict):
            raise ConsistencyError("Final checkpoint method state is malformed")
        self.main_store.load_state_dict(main_store)
        if self.q_tn_store is not None and self.q_dp_store is not None:
            q_tn_store = method.get("q_tn_store")
            q_dp_store = method.get("q_dp_store")
            if not isinstance(q_tn_store, dict) or not isinstance(q_dp_store, dict):
                raise ConsistencyError("Final checkpoint auxiliary stores are malformed")
            self.q_tn_store.load_state_dict(q_tn_store)
            self.q_dp_store.load_state_dict(q_dp_store)
        elif any(method.get(name) is not None for name in ("q_tn_store", "q_dp_store")):
            raise ConsistencyError("Final checkpoint has unexpected auxiliary stores")
        self.method.load_state_dict(main_method)
        expected_components = {"main", "q_tn", "q_dp"} if self.auxiliary else {"main"}
        if (
            set(model) != expected_components
            or set(optimizer) != expected_components
            or set(rng) != expected_components
        ):
            raise ConsistencyError("Final checkpoint trainer component set changed")
        self.main_trainer.load_state_dict(
            _join_component_state("main", model=model, optimizer=optimizer, rng=rng)
        )
        if self.auxiliary is not None:
            self.auxiliary.load_state_dict(
                {
                    "q_tn": _join_component_state(
                        "q_tn", model=model, optimizer=optimizer, rng=rng
                    ),
                    "q_dp": _join_component_state(
                        "q_dp", model=model, optimizer=optimizer, rng=rng
                    ),
                }
            )
        main_sampler = sampler.get("main")
        if not isinstance(main_sampler, dict):
            raise ConsistencyError("Final checkpoint sampler state is malformed")
        self.main_sampler.load_state_dict(main_sampler)
        if self.q_tn_sampler is not None and self.q_dp_sampler is not None:
            q_tn_sampler = sampler.get("q_tn")
            q_dp_sampler = sampler.get("q_dp")
            if not isinstance(q_tn_sampler, dict) or not isinstance(q_dp_sampler, dict):
                raise ConsistencyError("Final checkpoint auxiliary samplers are malformed")
            self.q_tn_sampler.load_state_dict(q_tn_sampler)
            self.q_dp_sampler.load_state_dict(q_dp_sampler)
        elif any(sampler.get(name) is not None for name in ("q_tn", "q_dp")):
            raise ConsistencyError("Final checkpoint has unexpected auxiliary samplers")
        scheduler_state = state["scheduler"]
        monitoring_state = state["monitoring"]
        truth_state = cursors.get("truth")
        if not all(
            isinstance(value, dict) for value in (scheduler_state, monitoring_state, truth_state)
        ):
            raise ConsistencyError("Final checkpoint scheduler or monitoring state is malformed")
        assert isinstance(scheduler_state, dict)
        assert isinstance(monitoring_state, dict)
        assert isinstance(truth_state, dict)
        self.scheduler.load_state_dict(scheduler_state)
        self.monitoring.load_state_dict(monitoring_state)
        self.truth_cursor.load_state_dict(truth_state)
        next_boundary = cursors.get("next_boundary_index")
        if (
            cursors.get("origin") != self.origin
            or cursors.get("training_limit") != self.training_limit
            or isinstance(next_boundary, bool)
            or not isinstance(next_boundary, int)
            or not 31 * 24 < next_boundary <= FINAL_SEAL_DAY * 24 + 1
        ):
            raise ConsistencyError("Final checkpoint cursors are inconsistent")
        expected_next = (
            0
            if self.method.last_time is None
            else round((self.method.last_time - self.origin) / HOUR_SECONDS) + 1
        )
        expected_click_cursor = min(
            self.training_limit,
            int(
                np.searchsorted(
                    self.features.click_times,
                    self.origin + (next_boundary - 1) * HOUR_SECONDS,
                    side="right",
                )
            ),
        )
        if (
            next_boundary != expected_next
            or self.method.click_cursor != expected_click_cursor
            or self.truth_cursor.last_time != self.method.last_time
        ):
            raise ConsistencyError("Final checkpoint event-time position is inconsistent")
        exposure_position = ledgers.get("exposures")
        prediction_position = ledgers.get("primary_predictions")
        snapshot_receipts = ledgers.get("snapshots")
        if (
            not isinstance(exposure_position, dict)
            or not isinstance(prediction_position, dict)
            or not isinstance(snapshot_receipts, dict)
            or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in snapshot_receipts.items()
            )
        ):
            raise ConsistencyError("Final checkpoint ledger positions are malformed")
        exposure_credits = exposure_position.get("credits")
        exposure_examples = exposure_position.get("examples")
        prediction_parts = prediction_position.get("parts")
        if (
            isinstance(exposure_credits, bool)
            or not isinstance(exposure_credits, int)
            or isinstance(exposure_examples, bool)
            or not isinstance(exposure_examples, int)
            or isinstance(prediction_parts, bool)
            or not isinstance(prediction_parts, int)
            or exposure_credits != self.main_trainer.budget.credits
            or exposure_examples != self.main_trainer.budget.optimizer_examples
            or self.exposures.position().credits < exposure_credits
            or self.primary_predictions.position().parts < prediction_parts
        ):
            raise ConsistencyError("Final checkpoint ledger accounting is inconsistent")
        for name, digest in snapshot_receipts.items():
            identity = next(
                (item for item in self.snapshot_identities if item.directory_name == name),
                None,
            )
            if identity is None or self.snapshots.verify(identity).manifest_sha256 != digest:
                raise ConsistencyError("Final checkpoint snapshot receipt is inconsistent")
        initialization_complete = compute.pop("initialization_complete", None)
        initialization_sha256 = compute.pop("initialization_model_sha256", None)
        if (
            initialization_complete is not True
            or not isinstance(initialization_sha256, str)
            or len(initialization_sha256) != 64
        ):
            raise ConsistencyError("Final checkpoint initialization evidence is malformed")
        self.compute = compute
        self.initialization_complete = True
        self.initialization_model_sha256 = initialization_sha256
        self.next_boundary_index = next_boundary
        self.primary_part_index = prediction_parts
        credits = self.compute.get("credits")
        snapshots = self.compute.get("snapshots")
        if (
            not isinstance(credits, list)
            or len(credits) != self.main_trainer.budget.credits
            or not isinstance(snapshots, list)
            or self.compute.get("initialization_steps") != self.plan.initialization_steps
            or self.compute.get("initialization_examples")
            != self.plan.initialization_steps * self.plan.batch_size
        ):
            raise ConsistencyError("Final checkpoint budget and compute evidence diverged")

    def _seal_primary_and_exposures(self) -> dict[str, object]:
        primary = (
            self.primary_predictions.verify_seal()
            if (self.primary_predictions.root / "seal.json").exists()
            else self.primary_predictions.seal()
        )
        exposures = (
            self.exposures.verify_seal()
            if (self.exposures.root / "seal.json").exists()
            else self.exposures.seal()
        )
        return {
            "primary_prediction_seal_sha256": primary.seal_sha256,
            "primary_prediction_ledger_sha256": primary.ledger_sha256,
            "primary_prediction_rows": primary.rows,
            "exposure_seal_sha256": exposures.seal_sha256,
            "exposure_ledger_sha256": exposures.ledger_sha256,
            "exposure_examples": exposures.examples,
        }

    def _intermediate_predictions(self) -> list[dict[str, object]]:
        snapshots = self.snapshots.verify_exact(self.snapshot_identities)
        final_state = copy.deepcopy(self.main_trainer.model.state_dict())
        evidence: list[dict[str, object]] = []
        try:
            for snapshot in snapshots:
                identity = snapshot.identity
                ledger = PredictionLedgerWriter(
                    self.output_root / "predictions" / "intermediate" / identity.directory_name,
                    PredictionLedgerIdentity(
                        version=1,
                        kind="intermediate",
                        run_id=self.plan.run_id,
                        method=self.plan.method,
                        seed=self.plan.seed,
                        period_first_day=FINAL_FIRST_CLICK_DAY,
                        period_last_day=FINAL_LAST_CLICK_DAY,
                        protocol_sha256=self.plan.protocol_sha256,
                        protocol_lock_sha256=self.plan.protocol_lock_sha256,
                        config_sha256=self.plan.canonical_sha256,
                        data_manifest_sha256=self.plan.data_manifest_sha256,
                        expected_rows=int(self.final_refs.size),
                        expected_ordered_id_sha256=hashlib.sha256(
                            self.features.click_ids[self.final_refs].tobytes()
                        ).hexdigest(),
                        ranking_eligible=False,
                        budget_fraction=identity.budget_fraction,
                        credits_at_snapshot=identity.credits_at_snapshot,
                    ),
                )
                if (ledger.root / "seal.json").exists():
                    seal = ledger.verify_seal()
                else:
                    self.main_trainer.model.load_state_dict(snapshot.model_state)
                    started = time.perf_counter()
                    part_index = 0
                    for day in range(FINAL_FIRST_CLICK_DAY, FINAL_LAST_CLICK_DAY + 1):
                        day_refs = self.features.references_for_day(day)
                        for start in range(0, day_refs.size, self.plan.prediction_batch_size):
                            refs = day_refs[start : start + self.plan.prediction_batch_size]
                            probabilities = self.main_trainer.predict(refs)
                            ledger.append(
                                part_index=part_index,
                                click_ids=[
                                    bytes(value).hex() for value in self.features.click_ids[refs]
                                ],
                                click_days=[day] * int(refs.size),
                                probabilities=probabilities.tolist(),
                                model_versions=[snapshot.model_version] * int(refs.size),
                            )
                            part_index += 1
                    self.compute["intermediate_prediction_seconds"] += time.perf_counter() - started
                    seal = ledger.seal()
                evidence.append(
                    {
                        "budget_fraction": identity.budget_fraction,
                        "credits_at_snapshot": identity.credits_at_snapshot,
                        "mode": "retrospective_inference_only",
                        "ranking_eligible": False,
                        "model_sha256": snapshot.model_sha256,
                        "model_version": snapshot.model_version,
                        "prediction_seal_sha256": seal.seal_sha256,
                        "prediction_ledger_sha256": seal.ledger_sha256,
                        "prediction_rows": seal.rows,
                        "ordered_id_sha256": seal.ordered_id_sha256,
                    }
                )
        finally:
            self.main_trainer.model.load_state_dict(final_state)
        return evidence

    def run(
        self,
        *,
        stop_after_decision_day: int | None = None,
        stop_after_seal: bool = False,
    ) -> dict[str, object]:
        if stop_after_decision_day is not None and not (
            FINAL_FIRST_DECISION_DAY <= stop_after_decision_day <= FINAL_LAST_DECISION_DAY
        ):
            raise ValueError("Final interruption day must lie in [31, 89]")
        last_boundary_index = FINAL_TRUTH_DRAIN_DAY * 24
        seal_evidence: dict[str, object] | None = None
        while self.next_boundary_index <= last_boundary_index:
            boundary_index = self.next_boundary_index
            boundary = self.origin + boundary_index * HOUR_SECONDS
            end = min(
                self.training_limit,
                int(np.searchsorted(self.features.click_times, boundary, side="right")),
            )
            click_refs = np.arange(self.method.click_cursor, end, dtype=np.int32)
            self._predict_primary(click_refs)
            truth = self.truth_cursor.reveal_through(boundary)
            self.monitoring.observe_truth(truth)
            result = self.method.process_boundary(
                boundary=boundary,
                click_refs=click_refs,
                truth=truth,
            )
            self.compute["feature_rows"] += int(click_refs.size)
            self.compute["truth_events"] += len(truth)
            self.compute["main_records"] += result.main_records
            self.compute["q_tn_records"] += result.q_tn_records
            self.compute["q_dp_records"] += result.q_dp_records
            if not self.initialization_complete:
                if self.initialization_method is None:
                    raise ConsistencyError("Final shared initialization stream is absent")
                self.initialization_method.process_boundary(
                    boundary=boundary,
                    click_refs=click_refs,
                    truth=truth,
                )
            next_boundary = boundary_index + 1
            day = boundary_index // 24
            if boundary_index == FINAL_FIRST_DECISION_DAY * 24:
                self._apply_initialization(boundary)
            if (
                boundary_index % 24 == 0
                and FINAL_FIRST_DECISION_DAY <= day <= FINAL_LAST_DECISION_DAY
            ):
                if not self.initialization_complete:
                    raise ConsistencyError("Final decision preceded shared initialization")
                decision = self._decision(day=day, boundary=boundary)
                if decision.spend:
                    self._spend_credit(boundary)
                self._write_checkpoint(next_boundary_index=next_boundary)
                if stop_after_decision_day == day:
                    self.next_boundary_index = next_boundary
                    return {
                        "status": "interrupted_after_checkpoint",
                        "decision_day": day,
                        "credits": self.main_trainer.budget.credits,
                        "next_boundary_index": next_boundary,
                    }
            if boundary_index == FINAL_SEAL_DAY * 24:
                self.scheduler.assert_complete()
                if self.main_trainer.budget.credits != self.plan.credits:
                    raise ConsistencyError("Final scheduler and optimizer budgets diverged")
                seal_evidence = self._seal_primary_and_exposures()
                self._write_checkpoint(next_boundary_index=next_boundary)
                if stop_after_seal:
                    self.next_boundary_index = next_boundary
                    return {
                        "status": "interrupted_after_checkpoint",
                        "decision_day": None,
                        "sealed": True,
                        "credits": self.main_trainer.budget.credits,
                        "next_boundary_index": next_boundary,
                    }
            self.next_boundary_index = next_boundary
        if self.method.click_cursor != self.training_limit:
            raise ConsistencyError("Final training feature cursor is incomplete")
        if self.compute["truth_events"] != self.training_limit:
            raise ConsistencyError("Final training-period truth did not drain completely")
        if seal_evidence is None:
            seal_evidence = self._seal_primary_and_exposures()
        intermediate = self._intermediate_predictions()
        snapshots = self.compute.get("snapshots")
        if not isinstance(snapshots, list) or len(snapshots) != 4:
            raise ConsistencyError("Final run did not freeze all intermediate budget snapshots")
        parameter_count = sum(
            parameter.numel() for parameter in self.main_trainer.model.parameters()
        )
        if self.auxiliary is not None:
            parameter_count += sum(
                parameter.numel()
                for model in (self.auxiliary.q_tn.model, self.auxiliary.q_dp.model)
                for parameter in model.parameters()
            )
        self._update_peak_memory()
        manifest: dict[str, object] = {
            "version": 1,
            "status": "complete",
            "phase": self.plan.phase,
            "study": self.plan.study,
            "run_id": self.plan.run_id,
            "method": self.plan.method,
            "scheduler": self.plan.scheduler,
            "plan": self.plan.model_dump(mode="json"),
            "config_sha256": self.plan.canonical_sha256,
            "protocol_sha256": self.plan.protocol_sha256,
            "protocol_lock_sha256": self.plan.protocol_lock_sha256,
            "truth_joined": False,
            "primary_evaluation_mode": "prequential",
            "evaluation_click_days": [FINAL_FIRST_CLICK_DAY, FINAL_LAST_CLICK_DAY],
            "decision_days": [FINAL_FIRST_DECISION_DAY, FINAL_LAST_DECISION_DAY],
            "credits": self.main_trainer.budget.credits,
            "core_optimizer_steps": self.main_trainer.budget.optimizer_steps,
            "core_optimizer_examples": self.main_trainer.budget.optimizer_examples,
            "initialization_model_sha256": self.initialization_model_sha256,
            "parameter_count": parameter_count,
            "scheduler_audit": self.scheduler.state_dict(),
            "monitoring_audit": self._monitoring_audit(),
            "intermediate_predictions": intermediate,
            "compute": self.compute,
            **seal_evidence,
        }
        manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
        write_json_atomic(self.output_root / "manifest.json", manifest)
        return manifest
