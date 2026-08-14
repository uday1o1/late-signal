"""Resumable real-data controller for retrospective chronological selection."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import Field, model_validator

from latesignal.contracts.protocol import StrictModel
from latesignal.data.manifests import canonical_json_bytes, write_json_atomic
from latesignal.errors import ConsistencyError
from latesignal.experiments.checkpoint import CheckpointIdentity, RollingCheckpointStore
from latesignal.experiments.exposures import ExposureLedgerIdentity, ExposureLedgerWriter
from latesignal.experiments.predictions import PredictionLedgerIdentity, PredictionLedgerWriter
from latesignal.features.store import FeatureTensorBatch
from latesignal.methods.production import PackedDelayedMethod
from latesignal.models.conversion_mlp import CategoricalSpec
from latesignal.scheduling.production import DailyRangeCreditScheduler
from latesignal.simulator.production_oracle import (
    SECONDS_PER_DAY,
    ProductionTruthCursor,
    ProductionTruthStore,
)
from latesignal.training.packed import (
    PackedDeterministicSampler,
    PackedRecordStore,
    packed_record_batch,
)
from latesignal.training.production import (
    PackedConversionTrainer,
    ProductionTrainingConfig,
)
from latesignal.training.production_esdfm import ESDFMAuxiliaryPair

HOUR_SECONDS = 3_600
SELECTION_TRAINING_LAST_DAY = 24
SELECTION_SCORING_FIRST_DAY = 25
SELECTION_SCORING_LAST_DAY = 34
SELECTION_FIRST_CREDIT_DAY = 55
SELECTION_LAST_CREDIT_DAY = 64

SelectionMethod = Literal["complete_wait", "fixed_wait", "es_dfm"]


class SelectionFeatureStore(Protocol):
    prepared_manifest_sha256: str
    feature_policy_sha256: str
    click_ids: NDArray[np.void]
    click_times: NDArray[np.float64]
    click_days: NDArray[np.int16]

    @property
    def categorical_specs(self) -> dict[str, CategoricalSpec]: ...

    def tensor_batch(self, references: NDArray[np.integer]) -> FeatureTensorBatch: ...

    def references_for_day(self, day: int) -> NDArray[np.int32]: ...

    def references_for_ids(self, click_ids: list[bytes]) -> NDArray[np.int32]: ...


class ProductionSelectionPlan(StrictModel):
    """One canonical candidate run from the frozen staged-selection DAG."""

    version: Literal[1]
    phase: Literal["qualification", "selection"]
    stage: Literal["model", "delayed", "sampler"]
    run_id: str = Field(min_length=1)
    method: SelectionMethod
    seed: Literal[17]
    wait_days: Literal[1, 3, 7, 14] | None
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    dropout: float = Field(ge=0.0, lt=1.0)
    gradient_norm_clip: float = Field(gt=0.0)
    initialization_steps: int = Field(gt=0)
    steps_per_credit: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    recent_window_days: Literal[1, 3, 7]
    reservoir_capacity: int = Field(gt=0)
    prediction_batch_size: int = Field(gt=0, le=65_536)
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    device: Literal["cpu", "cuda"]

    @model_validator(mode="after")
    def method_wait_contract(self) -> ProductionSelectionPlan:
        if (self.method in {"fixed_wait", "es_dfm"}) != (self.wait_days is not None):
            raise ValueError("Selection method and wait duration do not align")
        if self.stage == "model" and (
            self.method != "complete_wait"
            or self.recent_window_days != 3
            or self.reservoir_capacity != 1_000_000
        ):
            raise ValueError("Model selection must use the authored nuisance defaults")
        if self.stage == "delayed" and (
            self.method not in {"fixed_wait", "es_dfm"}
            or self.recent_window_days != 3
            or self.reservoir_capacity != 1_000_000
        ):
            raise ValueError("Delayed selection must use the authored nuisance defaults")
        if self.stage == "sampler" and self.method not in {"fixed_wait", "es_dfm"}:
            raise ValueError("Sampler selection requires the selected delayed method")
        if self.phase == "selection" and (
            self.learning_rate not in {0.0001, 0.0003, 0.001}
            or self.weight_decay not in {0.0, 0.00001, 0.0001}
            or self.dropout not in {0.0, 0.1}
            or self.gradient_norm_clip != 5.0
            or self.initialization_steps != 500
            or self.steps_per_credit not in {100, 250, 500}
            or self.batch_size != 2048
            or self.reservoir_capacity not in {1_000_000, 5_000_000}
            or self.prediction_batch_size != 65_536
            or self.device != "cuda"
        ):
            raise ValueError("Publication selection plan violates the authored training grid")
        return self

    @property
    def canonical_sha256(self) -> str:
        payload = self.model_dump(mode="json")
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _loss_mode(method: SelectionMethod) -> Literal["bce", "es_dfm"]:
    return "es_dfm" if method == "es_dfm" else "bce"


def _model_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _measured_compute_seconds(compute: dict[str, Any]) -> float:
    names = (
        "initialization_seconds",
        "core_training_seconds",
        "auxiliary_training_seconds",
        "prediction_seconds",
    )
    total = 0.0
    for name in names:
        value = compute.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConsistencyError("Selection compute measurements are malformed")
        total += float(value)
    return total


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
        raise ConsistencyError(f"Checkpoint component {name} is incomplete")
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
        or not isinstance(rng_section.get("cpu_rng_state"), torch.Tensor)
        or not isinstance(rng_section.get("cuda_rng_state"), list)
    ):
        raise ConsistencyError(f"Checkpoint component {name} is malformed")
    return {
        **optimizer_section["metadata"],
        "model": model_state,
        "optimizer": optimizer_section["optimizer"],
        "cpu_rng_state": rng_section["cpu_rng_state"],
        "cuda_rng_state": rng_section["cuda_rng_state"],
    }


class ProductionSelectionController:
    """Run one selection candidate without exposing held-out truth before sealing."""

    def __init__(
        self,
        *,
        plan: ProductionSelectionPlan,
        features: SelectionFeatureStore,
        truth: ProductionTruthStore,
        monitoring_mask: NDArray[np.bool_],
        output_root: Path,
        checkpoint_identity: CheckpointIdentity,
        resume: bool = False,
    ) -> None:
        if output_root.is_symlink():
            raise ConsistencyError("Selection output root cannot be a symlink")
        if (output_root / "manifest.json").exists():
            raise ConsistencyError("Completed selection run cannot be overwritten")
        if output_root.exists() and any(output_root.iterdir()) and not resume:
            raise ConsistencyError("Selection output root is not empty")
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
            raise ConsistencyError("Selection data identities or feature order changed")
        if (
            checkpoint_identity.phase != plan.phase
            or checkpoint_identity.config_sha256 != plan.canonical_sha256
            or checkpoint_identity.protocol_sha256 != plan.protocol_sha256
            or checkpoint_identity.protocol_lock_sha256 is not None
            or checkpoint_identity.data_manifest_sha256 != plan.data_manifest_sha256
            or checkpoint_identity.feature_policy_sha256 != plan.feature_policy_sha256
        ):
            raise ConsistencyError("Selection checkpoint identity does not match the run plan")
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
            raise ConsistencyError("Selection time origin must be an integral second")
        self.training_limit = int(
            np.searchsorted(
                features.click_days,
                SELECTION_TRAINING_LAST_DAY + 1,
                side="left",
            )
        )
        if self.training_limit == 0:
            raise ConsistencyError("Selection training period contains no clicks")
        self.scoring_refs = np.flatnonzero(
            (features.click_days >= SELECTION_SCORING_FIRST_DAY)
            & (features.click_days <= SELECTION_SCORING_LAST_DAY)
        ).astype(np.int32)
        if self.scoring_refs.size == 0:
            raise ConsistencyError("Selection scoring period contains no clicks")
        required_days = set(range(SELECTION_SCORING_LAST_DAY + 1))
        covered_days = set(
            np.unique(features.click_days[: int(self.scoring_refs[-1]) + 1]).tolist()
        )
        if covered_days != required_days:
            raise ConsistencyError("Selection data does not cover every authored click day")
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
                expected_credits=10,
                steps_per_credit=plan.steps_per_credit,
                batch_size=plan.batch_size,
            ),
        )
        scoring_ids_sha256 = hashlib.sha256(
            self.features.click_ids[self.scoring_refs].tobytes()
        ).hexdigest()
        self.predictions = PredictionLedgerWriter(
            self.output_root / "predictions",
            PredictionLedgerIdentity(
                version=1,
                kind="selection",
                run_id=plan.run_id,
                method=plan.method,
                seed=plan.seed,
                period_first_day=SELECTION_SCORING_FIRST_DAY,
                period_last_day=SELECTION_SCORING_LAST_DAY,
                protocol_sha256=plan.protocol_sha256,
                config_sha256=plan.canonical_sha256,
                data_manifest_sha256=plan.data_manifest_sha256,
                expected_rows=int(self.scoring_refs.size),
                expected_ordered_id_sha256=scoring_ids_sha256,
                ranking_eligible=plan.phase == "selection",
            ),
        )
        self.next_boundary_index = 0
        self.initialization_complete = False
        self.initialization_model_sha256: str | None = None
        self.compute: dict[str, Any] = {
            "initialization_seconds": 0.0,
            "core_training_seconds": 0.0,
            "auxiliary_training_seconds": 0.0,
            "prediction_seconds": 0.0,
            "feature_rows": 0,
            "truth_events": 0,
            "main_records": 0,
            "q_tn_records": 0,
            "q_dp_records": 0,
            "credits": [],
            "auxiliary": [],
        }
        if resume:
            self._restore_latest()
        elif (self.output_root / "checkpoints" / "latest.json").exists():
            raise ConsistencyError("Existing selection checkpoint requires resume mode")

    def _build_components(self, *, include_initialization: bool) -> None:
        count = int(self.features.click_days.size)
        exclusion = self.monitoring_mask | (self.features.click_days > SELECTION_TRAINING_LAST_DAY)
        self.main_store = PackedRecordStore(feature_count=count)
        self.q_tn_store = (
            PackedRecordStore(feature_count=count) if self.plan.method == "es_dfm" else None
        )
        self.q_dp_store = (
            PackedRecordStore(feature_count=count) if self.plan.method == "es_dfm" else None
        )
        self.method = PackedDelayedMethod(
            self.plan.method,
            click_times=self.features.click_times,
            monitoring_mask=exclusion,
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
            initial_config = ProductionTrainingConfig(
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
                initial_config,
                seed=self.plan.seed,
                device=self.plan.device,
            )
        main_config = ProductionTrainingConfig(
            learning_rate=self.plan.learning_rate,
            weight_decay=self.plan.weight_decay,
            dropout=self.plan.dropout,
            gradient_norm_clip=self.plan.gradient_norm_clip,
            steps_per_credit=self.plan.steps_per_credit,
            batch_size=self.plan.batch_size,
            loss_mode=_loss_mode(self.plan.method),
        )
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
            last_click_day=SELECTION_TRAINING_LAST_DAY,
        )
        self.scheduler = DailyRangeCreditScheduler(
            origin=self.origin,
            first_day=SELECTION_FIRST_CREDIT_DAY,
            last_day=SELECTION_LAST_CREDIT_DAY,
        )

    def _apply_initialization(self) -> None:
        if self.initialization_complete:
            raise ConsistencyError("Shared initialization was applied twice")
        if self.initialization_trainer is None or self.initialization_sampler is None:
            raise ConsistencyError("Shared initialization components are absent")
        started = time.perf_counter()
        result = self.initialization_trainer.spend_credit(
            credit_id=0,
            decision_time=self.origin + 31 * SECONDS_PER_DAY,
            sampler=self.initialization_sampler,
        )
        initialization = self.initialization_trainer.model.state_dict()
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
                simulator_time=self.origin + 31 * SECONDS_PER_DAY,
            )
            self._copy_day_zero_auxiliary_records(
                self.q_dp_store,
                self.q_dp_initialization_store,
                simulator_time=self.origin + 31 * SECONDS_PER_DAY,
            )
            auxiliary_started = time.perf_counter()
            try:
                auxiliary = self.auxiliary.update(
                    credit_id=0,
                    decision_time=self.origin + 31 * SECONDS_PER_DAY,
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
                    "decision_time": self.origin + 31 * SECONDS_PER_DAY,
                    "steps": auxiliary.auxiliary_steps,
                    "examples": auxiliary.auxiliary_examples,
                    "q_tn_loss": auxiliary.q_tn.mean_loss,
                    "q_dp_loss": auxiliary.q_dp.mean_loss,
                }
            )
        self.initialization_complete = True
        self._release_initialization_components()

    def _release_initialization_components(self) -> None:
        self.initialization_store = None
        self.initialization_method = None
        self.initialization_sampler = None
        self.initialization_trainer = None
        self.q_tn_initialization_store = None
        self.q_dp_initialization_store = None
        self.q_tn_initialization_sampler = None
        self.q_dp_initialization_sampler = None

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

    def _spend_credit(self, boundary: int) -> None:
        decision = self.scheduler.decide(boundary)
        if not decision.spend:
            raise ConsistencyError("Selection daily scheduler refused an authored credit")
        credit_id = self.scheduler.spent_count - 1
        if self.auxiliary is not None:
            assert self.q_tn_sampler is not None and self.q_dp_sampler is not None
            auxiliary_started = time.perf_counter()
            try:
                auxiliary = self.auxiliary.update(
                    credit_id=credit_id + 1,
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
                    "work_id": credit_id + 1,
                    "main_credit_id": credit_id,
                    "decision_time": boundary,
                    "steps": auxiliary.auxiliary_steps,
                    "examples": auxiliary.auxiliary_examples,
                    "q_tn_loss": auxiliary.q_tn.mean_loss,
                    "q_dp_loss": auxiliary.q_dp.mean_loss,
                }
            )
        training_started = time.perf_counter()
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
            auxiliary_state = self.auxiliary.state_dict()
            for name, key in (("q_tn", "q_tn"), ("q_dp", "q_dp")):
                value = auxiliary_state[key]
                assert isinstance(value, dict)
                _split_component_state(
                    name,
                    value,
                    model=model,
                    optimizer=optimizer,
                    rng=rng,
                )
        return model, optimizer, rng

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
            "monitoring": {
                "membership_sha256": hashlib.sha256(
                    np.packbits(self.monitoring_mask, bitorder="little").tobytes()
                ).hexdigest(),
                "excluded_examples": int(np.count_nonzero(self.monitoring_mask)),
            },
            "ledgers": {
                "exposures": self.exposures.position().as_dict(),
                "predictions": self.predictions.position().as_dict(),
            },
            "compute": {
                **self.compute,
                "initialization_complete": self.initialization_complete,
                "initialization_model_sha256": self.initialization_model_sha256,
            },
        }

    def _write_checkpoint(self, *, next_boundary_index: int) -> None:
        if not self.initialization_complete or self.initialization_trainer is not None:
            raise ConsistencyError("Selection checkpoint preceded compact initialization")
        self.checkpoints.write(
            self.checkpoint_identity,
            self._state(next_boundary_index=next_boundary_index),
        )

    def _restore_latest(self) -> None:
        loaded = self.checkpoints.load_latest(self.checkpoint_identity)
        state = loaded.state
        method = state["method"]
        sampler = state["sampler"]
        cursors = state["cursors"]
        monitoring = state["monitoring"]
        ledgers = state["ledgers"]
        compute = state["compute"]
        for section in (method, sampler, cursors, monitoring, ledgers, compute):
            if not isinstance(section, dict):
                raise ConsistencyError("Selection checkpoint recovery section is malformed")
        expected_method_keys = {
            "main_store",
            "main",
            "q_tn_store",
            "q_dp_store",
        }
        expected_sampler_keys = {
            "main",
            "q_tn",
            "q_dp",
        }
        if set(method) != expected_method_keys or set(sampler) != expected_sampler_keys:
            raise ConsistencyError("Selection checkpoint component set changed")
        main_store = method.get("main_store")
        main_method = method.get("main")
        if not isinstance(main_store, dict) or not isinstance(main_method, dict):
            raise ConsistencyError("Selection checkpoint method state is malformed")
        assert isinstance(main_store, dict)
        assert isinstance(main_method, dict)
        self.main_store.load_state_dict(main_store)
        if self.q_tn_store is not None and self.q_dp_store is not None:
            q_tn_store = method.get("q_tn_store")
            q_dp_store = method.get("q_dp_store")
            if not isinstance(q_tn_store, dict) or not isinstance(q_dp_store, dict):
                raise ConsistencyError("Selection checkpoint auxiliary stores are malformed")
            self.q_tn_store.load_state_dict(q_tn_store)
            self.q_dp_store.load_state_dict(q_dp_store)
        elif any(method.get(key) is not None for key in ("q_tn_store", "q_dp_store")):
            raise ConsistencyError("Selection checkpoint has unexpected auxiliary stores")
        self.method.load_state_dict(main_method)
        model = state["model"]
        optimizer = state["optimizer"]
        rng = state["rng"]
        if not all(isinstance(value, dict) for value in (model, optimizer, rng)):
            raise ConsistencyError("Selection checkpoint trainer sections are malformed")
        assert isinstance(model, dict) and isinstance(optimizer, dict) and isinstance(rng, dict)
        expected_components = {"main"}
        if self.auxiliary is not None:
            expected_components.update({"q_tn", "q_dp"})
        if (
            set(model) != expected_components
            or set(optimizer) != expected_components
            or set(rng) != expected_components
        ):
            raise ConsistencyError("Selection checkpoint trainer component set changed")
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
            raise ConsistencyError("Selection checkpoint core sampler is malformed")
        self.main_sampler.load_state_dict(main_sampler)
        if self.q_tn_sampler is not None and self.q_dp_sampler is not None:
            q_tn_sampler = sampler.get("q_tn")
            q_dp_sampler = sampler.get("q_dp")
            if not isinstance(q_tn_sampler, dict) or not isinstance(q_dp_sampler, dict):
                raise ConsistencyError("Selection checkpoint auxiliary samplers are malformed")
            assert isinstance(q_tn_sampler, dict)
            assert isinstance(q_dp_sampler, dict)
            self.q_tn_sampler.load_state_dict(q_tn_sampler)
            self.q_dp_sampler.load_state_dict(q_dp_sampler)
        elif any(sampler.get(key) is not None for key in ("q_tn", "q_dp")):
            raise ConsistencyError("Selection checkpoint has unexpected auxiliary samplers")
        scheduler_state = state["scheduler"]
        truth_state = cursors.get("truth")
        if not isinstance(scheduler_state, dict) or not isinstance(truth_state, dict):
            raise ConsistencyError("Selection checkpoint scheduler or truth cursor is malformed")
        self.scheduler.load_state_dict(scheduler_state)
        self.truth_cursor.load_state_dict(truth_state)
        next_boundary = cursors.get("next_boundary_index")
        if (
            cursors.get("origin") != self.origin
            or cursors.get("training_limit") != self.training_limit
            or isinstance(next_boundary, bool)
            or not isinstance(next_boundary, int)
            or not 0 < next_boundary <= SELECTION_LAST_CREDIT_DAY * 24 + 1
        ):
            raise ConsistencyError("Selection checkpoint cursors are inconsistent")
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
            raise ConsistencyError("Selection checkpoint event-time position is inconsistent")
        expected_membership = hashlib.sha256(
            np.packbits(self.monitoring_mask, bitorder="little").tobytes()
        ).hexdigest()
        if monitoring.get("membership_sha256") != expected_membership or monitoring.get(
            "excluded_examples"
        ) != int(np.count_nonzero(self.monitoring_mask)):
            raise ConsistencyError("Selection checkpoint monitoring identity changed")
        exposure_position = ledgers.get("exposures")
        prediction_position = ledgers.get("predictions")
        if not isinstance(exposure_position, dict) or not isinstance(prediction_position, dict):
            raise ConsistencyError("Selection checkpoint ledger positions are malformed")
        checkpoint_exposure_credits = exposure_position.get("credits")
        checkpoint_exposure_examples = exposure_position.get("examples")
        checkpoint_prediction_parts = prediction_position.get("parts")
        if (
            isinstance(checkpoint_exposure_credits, bool)
            or not isinstance(checkpoint_exposure_credits, int)
            or isinstance(checkpoint_exposure_examples, bool)
            or not isinstance(checkpoint_exposure_examples, int)
            or isinstance(checkpoint_prediction_parts, bool)
            or not isinstance(checkpoint_prediction_parts, int)
            or checkpoint_exposure_credits != self.scheduler.spent_count
            or checkpoint_exposure_examples != self.main_trainer.budget.optimizer_examples
        ):
            raise ConsistencyError("Selection checkpoint ledger accounting is inconsistent")
        if (
            self.exposures.position().credits < checkpoint_exposure_credits
            or self.predictions.position().parts < checkpoint_prediction_parts
        ):
            raise ConsistencyError("Durable selection ledger is behind its checkpoint")
        initialization_complete = compute.pop("initialization_complete", None)
        initialization_sha256 = compute.pop("initialization_model_sha256", None)
        if (
            initialization_complete is not True
            or not isinstance(initialization_sha256, str)
            or len(initialization_sha256) != 64
            or any(character not in "0123456789abcdef" for character in initialization_sha256)
        ):
            raise ConsistencyError("Selection checkpoint compute state is malformed")
        self.compute = compute
        self.initialization_complete = initialization_complete
        self.initialization_model_sha256 = initialization_sha256
        self.next_boundary_index = next_boundary
        credits = self.compute.get("credits")
        auxiliary_work = self.compute.get("auxiliary")
        if (
            self.initialization_complete is not True
            or self.compute.get("initialization_steps") != self.plan.initialization_steps
            or self.compute.get("initialization_examples")
            != self.plan.initialization_steps * self.plan.batch_size
            or not isinstance(credits, list)
            or len(credits) != self.scheduler.spent_count
            or not isinstance(auxiliary_work, list)
            or self.main_trainer.budget.credits != self.scheduler.spent_count
        ):
            raise ConsistencyError("Selection checkpoint budget and scheduler diverged")
        if self.auxiliary is not None:
            expected_auxiliary_work = self.scheduler.spent_count + 1
            if (
                self.auxiliary.q_tn.work_units != expected_auxiliary_work
                or self.auxiliary.q_dp.work_units != expected_auxiliary_work
                or len(auxiliary_work) != expected_auxiliary_work
            ):
                raise ConsistencyError("Selection auxiliary and core work units diverged")
        if self.auxiliary is None and auxiliary_work:
            raise ConsistencyError("Non-ES selection checkpoint contains auxiliary work")

    def _predict_and_seal(self) -> dict[str, object]:
        prediction_seal_path = self.predictions.root / "seal.json"
        if prediction_seal_path.exists():
            prediction_seal = self.predictions.verify_seal()
        else:
            started = time.perf_counter()
            part_index = 0
            model_version = self.main_trainer.model_version
            for day in range(SELECTION_SCORING_FIRST_DAY, SELECTION_SCORING_LAST_DAY + 1):
                day_refs = self.features.references_for_day(day)
                for start in range(0, day_refs.size, self.plan.prediction_batch_size):
                    refs = day_refs[start : start + self.plan.prediction_batch_size]
                    probabilities = self.main_trainer.predict(refs)
                    self.predictions.append(
                        part_index=part_index,
                        click_ids=[bytes(value).hex() for value in self.features.click_ids[refs]],
                        click_days=[day] * int(refs.size),
                        probabilities=probabilities.tolist(),
                        model_versions=[model_version] * int(refs.size),
                    )
                    part_index += 1
            self.compute["prediction_seconds"] += time.perf_counter() - started
            prediction_seal = self.predictions.seal()
        exposure_seal_path = self.exposures.root / "seal.json"
        exposure_seal = (
            self.exposures.verify_seal() if exposure_seal_path.exists() else self.exposures.seal()
        )
        return {
            "prediction_seal_sha256": prediction_seal.seal_sha256,
            "prediction_ledger_sha256": prediction_seal.ledger_sha256,
            "prediction_rows": prediction_seal.rows,
            "exposure_seal_sha256": exposure_seal.seal_sha256,
            "exposure_ledger_sha256": exposure_seal.ledger_sha256,
            "exposure_examples": exposure_seal.examples,
        }

    def run(
        self,
        *,
        stop_after_initialization: bool = False,
        stop_after_credits: int | None = None,
    ) -> dict[str, object]:
        if stop_after_credits is not None and not 0 < stop_after_credits <= 10:
            raise ValueError("Selection interruption credit must lie in [1, 10]")
        last_boundary_index = SELECTION_LAST_CREDIT_DAY * 24
        while self.next_boundary_index <= last_boundary_index:
            boundary_index = self.next_boundary_index
            boundary = self.origin + boundary_index * HOUR_SECONDS
            end = min(
                self.training_limit,
                int(np.searchsorted(self.features.click_times, boundary, side="right")),
            )
            click_refs = np.arange(self.method.click_cursor, end, dtype=np.int32)
            truth = self.truth_cursor.reveal_through(boundary)
            boundary_result = self.method.process_boundary(
                boundary=boundary,
                click_refs=click_refs,
                truth=truth,
            )
            self.compute["feature_rows"] += int(click_refs.size)
            self.compute["truth_events"] += len(truth)
            self.compute["main_records"] += boundary_result.main_records
            self.compute["q_tn_records"] += boundary_result.q_tn_records
            self.compute["q_dp_records"] += boundary_result.q_dp_records
            if not self.initialization_complete:
                if self.initialization_method is None:
                    raise ConsistencyError("Shared initialization stream is absent")
                self.initialization_method.process_boundary(
                    boundary=boundary,
                    click_refs=click_refs,
                    truth=truth,
                )
            next_boundary = boundary_index + 1
            if boundary_index == 31 * 24:
                self._apply_initialization()
                self._write_checkpoint(next_boundary_index=next_boundary)
                if stop_after_initialization:
                    self.next_boundary_index = next_boundary
                    return {
                        "status": "interrupted_after_checkpoint",
                        "credits": 0,
                        "next_boundary_index": self.next_boundary_index,
                    }
            day = boundary_index // 24
            if (
                boundary_index % 24 == 0
                and SELECTION_FIRST_CREDIT_DAY <= day <= SELECTION_LAST_CREDIT_DAY
            ):
                if not self.initialization_complete:
                    raise ConsistencyError("Selection credit preceded shared initialization")
                self._spend_credit(boundary)
                self._write_checkpoint(next_boundary_index=next_boundary)
                if stop_after_credits == self.scheduler.spent_count:
                    self.next_boundary_index = next_boundary
                    return {
                        "status": "interrupted_after_checkpoint",
                        "credits": self.scheduler.spent_count,
                        "next_boundary_index": self.next_boundary_index,
                    }
            self.next_boundary_index = next_boundary
        self.scheduler.assert_complete()
        if self.method.click_cursor != self.training_limit:
            raise ConsistencyError("Selection training feature cursor is incomplete")
        evidence = self._predict_and_seal()
        self._write_checkpoint(next_boundary_index=self.next_boundary_index)
        parameter_count = self.main_trainer.model.parameter_count
        if self.auxiliary is not None:
            parameter_count += (
                self.auxiliary.q_tn.model.parameter_count
                + self.auxiliary.q_dp.model.parameter_count
            )
        manifest: dict[str, object] = {
            "version": 1,
            "status": "complete",
            "phase": self.plan.phase,
            "selection_mode": "retrospective_chronological",
            "truth_joined": False,
            "run_id": self.plan.run_id,
            "stage": self.plan.stage,
            "plan": self.plan.model_dump(mode="json"),
            "config_sha256": self.plan.canonical_sha256,
            "protocol_sha256": self.plan.protocol_sha256,
            "training_click_days": [0, SELECTION_TRAINING_LAST_DAY],
            "scoring_click_days": [
                SELECTION_SCORING_FIRST_DAY,
                SELECTION_SCORING_LAST_DAY,
            ],
            "credit_days": [SELECTION_FIRST_CREDIT_DAY, SELECTION_LAST_CREDIT_DAY],
            "credits": self.scheduler.spent_count,
            "core_optimizer_steps": self.main_trainer.budget.optimizer_steps,
            "core_optimizer_examples": self.main_trainer.budget.optimizer_examples,
            "initialization_model_sha256": self.initialization_model_sha256,
            "measured_compute_seconds": _measured_compute_seconds(self.compute),
            "parameter_count": parameter_count,
            "compute": self.compute,
            **evidence,
        }
        manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
        write_json_atomic(self.output_root / "manifest.json", manifest)
        return manifest
