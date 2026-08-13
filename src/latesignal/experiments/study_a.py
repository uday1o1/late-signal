"""Synthetic fixed-schedule qualification for every Study A method."""

from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from latesignal.contracts.events import ClickEvent, NegativeMaturity, PositiveReveal, TruthRecord
from latesignal.contracts.records import ExposureRecord, TrainingRecord
from latesignal.contracts.study_a import StudyAConfig
from latesignal.data.manifests import canonical_json_bytes, write_json_atomic
from latesignal.data.schema import CATEGORICAL_CLICK_FIELDS
from latesignal.errors import ConfigurationError, ConsistencyError
from latesignal.methods.base import DelayedMethod
from latesignal.methods.complete_wait import CompleteWaitMethod
from latesignal.methods.dfm import DelayedFeedbackMethod
from latesignal.methods.es_dfm import AuxiliaryRecord, ESDFMMethod
from latesignal.methods.fixed_wait import FixedWaitMethod
from latesignal.methods.fnw import FakeNegativeWeightedMethod
from latesignal.methods.immediate_fake_negative import ImmediateFakeNegativeMethod
from latesignal.methods.losses import dfm_loss, esdfm_loss, fnw_loss
from latesignal.methods.oracle_reference import OracleReferenceMethod
from latesignal.models.conversion_mlp import CategoricalSpec, ConversionMLP
from latesignal.models.dfm import DelayedFeedbackMLP
from latesignal.training.budget import BudgetCounter
from latesignal.training.reproducibility import configure_determinism
from latesignal.training.sampler import DeterministicSampler
from latesignal.training.trainer import MLPTrainer, ModelBatch, TrainingLoss

SECONDS_PER_DAY = 86_400
METHOD_NAMES = (
    "complete_wait",
    "immediate_fake_negative",
    "fixed_wait",
    "dfm",
    "fnw",
    "es_dfm",
    "oracle_reference",
)
FIELDS = tuple(sorted(CATEGORICAL_CLICK_FIELDS))


@dataclass(frozen=True, slots=True)
class _Fixture:
    clicks: tuple[ClickEvent, ...]
    truth: tuple[TruthRecord, ...]


@dataclass(slots=True)
class _Cursors:
    click: int = 0
    truth: int = 0


@dataclass(slots=True)
class _AuxiliaryTraining:
    q_tn: ConversionMLP
    q_dp: ConversionMLP
    q_tn_optimizer: torch.optim.AdamW
    q_dp_optimizer: torch.optim.AdamW
    rng: random.Random
    steps: int = 0
    examples: int = 0


def _fixture(config: StudyAConfig) -> _Fixture:
    clicks: list[ClickEvent] = []
    truth: list[TruthRecord] = []
    for day in range(34):
        for within_day in range(4):
            index = day * 4 + within_day
            click_id = f"study-a-{index:04d}"
            click_time = day * SECONDS_PER_DAY + within_day
            feature = 1.0 if within_day in {0, 1} else -1.0
            final_label = int(within_day in {0, 1})
            if within_day == 0:
                available_at = click_time + 6 * 3_600
            elif within_day == 1:
                available_at = click_time + 2 * SECONDS_PER_DAY
            else:
                available_at = click_time + config.attribution_seconds
            clicks.append(ClickEvent(click_id, click_time, feature))
            truth.append(TruthRecord(click_id, final_label, available_at))
    return _Fixture(tuple(clicks), tuple(sorted(truth, key=lambda item: item.available_at)))


def _new_conversion_model() -> ConversionMLP:
    return ConversionMLP(
        {field: CategoricalSpec(bucket_count=8, embedding_dim=2) for field in FIELDS},
        dropout=0.0,
    )


def _encode(records: tuple[TrainingRecord, ...]) -> ModelBatch:
    category = torch.tensor(
        [int((record.feature + 2.0) * 2.0) % 8 for record in records],
        dtype=torch.long,
    )
    numeric = torch.tensor(
        [[record.feature, abs(record.feature), 0.0, 0.0] for record in records],
        dtype=torch.float32,
    )
    return ModelBatch(
        categorical={field: (category + index) % 8 for index, field in enumerate(FIELDS)},
        numeric=numeric,
        targets=torch.tensor([record.target for record in records], dtype=torch.float32),
        weights=torch.tensor([record.weight for record in records], dtype=torch.float32),
        record_ids=tuple(record.record_id for record in records),
    )


def _model_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _initial_checkpoint(config: StudyAConfig, fixture: _Fixture) -> dict[str, Tensor]:
    configure_determinism(config.seed)
    model = _new_conversion_model()
    day_zero_ids = {
        click.click_id for click in fixture.clicks if click.click_time < SECONDS_PER_DAY
    }
    features = {click.click_id: click.feature for click in fixture.clicks}
    records = tuple(
        TrainingRecord(
            record_id=f"initial:{truth.click_id}",
            click_id=truth.click_id,
            available_at=config.attribution_seconds,
            status="final",
            target=float(truth.final_label),
            weight=1.0,
            correction_group=None,
            source_method="shared_initialization",
            feature=features[truth.click_id],
        )
        for truth in fixture.truth
        if truth.click_id in day_zero_ids
    )
    sampler = DeterministicSampler(
        seed=config.seed,
        recent_window_seconds=config.recent_window_seconds,
        reservoir_capacity=config.reservoir_capacity,
    )
    initialization_time = 31 * SECONDS_PER_DAY
    for record in records:
        sampler.add(record, initialization_time)
    trainer = MLPTrainer(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_norm_clip=config.gradient_norm_clip,
        steps_per_credit=config.initialization_steps,
        batch_size=config.batch_size,
        encoder=_encode,
    )
    trainer.spend_credit(credit_id=-1, decision_time=initialization_time, sampler=sampler)
    return copy.deepcopy(model.state_dict())


def _method(name: str, config: StudyAConfig, truth: tuple[TruthRecord, ...]) -> DelayedMethod:
    if name == "complete_wait":
        return CompleteWaitMethod(config.attribution_seconds)
    if name == "immediate_fake_negative":
        return ImmediateFakeNegativeMethod()
    if name == "fixed_wait":
        return FixedWaitMethod(config.wait_seconds)
    if name == "dfm":
        return DelayedFeedbackMethod(config.attribution_seconds)
    if name == "fnw":
        return FakeNegativeWeightedMethod()
    if name == "es_dfm":
        return ESDFMMethod(
            wait_seconds=config.wait_seconds,
            attribution_seconds=config.attribution_seconds,
        )
    if name == "oracle_reference":
        return OracleReferenceMethod(truth)
    raise ConsistencyError(f"Unknown Study A method: {name}")


def _append_records(
    records: list[TrainingRecord],
    sampler: DeterministicSampler,
    simulator_time: int,
) -> None:
    for record in records:
        record.assert_available(simulator_time)
        sampler.add(record, simulator_time)


def _deliver_through(
    method: DelayedMethod,
    sampler: DeterministicSampler,
    fixture: _Fixture,
    cursors: _Cursors,
    simulator_time: int,
) -> None:
    while cursors.click < len(fixture.clicks):
        click = fixture.clicks[cursors.click]
        if click.click_time > simulator_time:
            break
        emitted = method.on_click(click)
        _append_records(emitted, sampler, simulator_time)
        if isinstance(method, DelayedFeedbackMethod):
            sampler.add(
                TrainingRecord(
                    record_id=f"dfm:{click.click_id}:current",
                    click_id=click.click_id,
                    available_at=click.click_time,
                    status="provisional",
                    target=0.0,
                    weight=1.0,
                    correction_group=None,
                    source_method="dfm",
                    feature=click.feature,
                ),
                simulator_time,
            )
        cursors.click += 1
    while cursors.truth < len(fixture.truth):
        truth = fixture.truth[cursors.truth]
        if truth.available_at > simulator_time:
            break
        if truth.final_label == 1:
            emitted = method.on_positive_reveal(PositiveReveal(truth.click_id, truth.available_at))
        else:
            emitted = method.on_negative_maturity(
                NegativeMaturity(truth.click_id, truth.available_at)
            )
        _append_records(emitted, sampler, simulator_time)
        cursors.truth += 1
    _append_records(method.on_boundary(simulator_time), sampler, simulator_time)


def _standard_trainer(
    name: str,
    config: StudyAConfig,
    initialization: dict[str, Tensor],
    auxiliary: _AuxiliaryTraining | None,
) -> MLPTrainer:
    model = _new_conversion_model()
    model.load_state_dict(initialization)
    loss_function = None
    if name == "fnw":

        def fnw_training(logits: Tensor, batch: ModelBatch) -> TrainingLoss:
            result = fnw_loss(logits, batch.targets)
            return TrainingLoss(result.loss, result.weights)

        loss_function = fnw_training
    elif name == "es_dfm":
        if auxiliary is None:
            raise ConsistencyError("ES-DFM auxiliary state is unavailable")

        def esdfm_training(logits: Tensor, batch: ModelBatch) -> TrainingLoss:
            auxiliary.q_tn.eval()
            auxiliary.q_dp.eval()
            with torch.no_grad():
                q_tn_logits = auxiliary.q_tn(batch.categorical, batch.numeric)
                q_dp_logits = auxiliary.q_dp(batch.categorical, batch.numeric)
            result = esdfm_loss(logits, batch.targets, q_tn_logits, q_dp_logits)
            return TrainingLoss(result.loss, result.weights)

        loss_function = esdfm_training
    return MLPTrainer(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_norm_clip=config.gradient_norm_clip,
        steps_per_credit=config.steps_per_credit,
        batch_size=config.batch_size,
        encoder=_encode,
        loss_function=loss_function,
    )


def _new_auxiliary(config: StudyAConfig) -> _AuxiliaryTraining:
    q_tn = _new_conversion_model()
    q_dp = _new_conversion_model()
    return _AuxiliaryTraining(
        q_tn=q_tn,
        q_dp=q_dp,
        q_tn_optimizer=torch.optim.AdamW(q_tn.parameters(), lr=3e-4, weight_decay=1e-4),
        q_dp_optimizer=torch.optim.AdamW(q_dp.parameters(), lr=3e-4, weight_decay=1e-4),
        rng=random.Random(config.seed + 1_000),
    )


def _aux_batch(records: list[AuxiliaryRecord], target: str) -> ModelBatch:
    converted = tuple(
        TrainingRecord(
            record_id=f"aux:{target}:{record.click_id}",
            click_id=record.click_id,
            available_at=record.available_at,
            status="final",
            target=float(getattr(record, target)),
            weight=1.0,
            correction_group=None,
            source_method=f"es_dfm_{target}",
            feature=record.feature,
        )
        for record in records
    )
    return _encode(converted)


def _train_aux_model(
    model: ConversionMLP,
    optimizer: torch.optim.AdamW,
    records: list[AuxiliaryRecord],
    target: str,
    *,
    steps: int,
    batch_size: int,
    rng: random.Random,
) -> None:
    if not records:
        raise ConsistencyError("INSUFFICIENT_LEGAL_AUXILIARY_POOL")
    model.train()
    for _ in range(steps):
        selected = [rng.choice(records) for _ in range(batch_size)]
        batch = _aux_batch(selected, target)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.categorical, batch.numeric)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, batch.targets)
        loss.backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()


def _update_auxiliary(
    method: ESDFMMethod,
    auxiliary: _AuxiliaryTraining,
    config: StudyAConfig,
    simulator_time: int,
    credit_id: int,
) -> None:
    frozen = list(method.auxiliary_records(simulator_time))
    q_tn_pool = [record for record in frozen if record.q_tn_target is not None]
    steps = (
        config.auxiliary_initialization_steps if credit_id == 0 else config.auxiliary_later_steps
    )
    _train_aux_model(
        auxiliary.q_tn,
        auxiliary.q_tn_optimizer,
        q_tn_pool,
        "q_tn_target",
        steps=steps,
        batch_size=config.batch_size,
        rng=auxiliary.rng,
    )
    _train_aux_model(
        auxiliary.q_dp,
        auxiliary.q_dp_optimizer,
        frozen,
        "q_dp_target",
        steps=steps,
        batch_size=config.batch_size,
        rng=auxiliary.rng,
    )
    auxiliary.steps += 2 * steps
    auxiliary.examples += 2 * steps * config.batch_size


def _train_dfm_credit(
    method: DelayedFeedbackMethod,
    model: DelayedFeedbackMLP,
    optimizer: torch.optim.AdamW,
    sampler: DeterministicSampler,
    config: StudyAConfig,
    *,
    credit_id: int,
    simulator_time: int,
    budget: BudgetCounter,
    exposures: list[ExposureRecord],
) -> None:
    model.train()
    for step in range(config.steps_per_credit):
        sampled = sampler.sample(simulator_time=simulator_time, batch_size=config.batch_size)
        observations = [
            method.materialize(item.record.click_id, simulator_time) for item in sampled
        ]
        converted = tuple(
            TrainingRecord(
                record_id=f"dfm:{item.click_id}:current",
                click_id=item.click_id,
                available_at=simulator_time,
                status="final" if item.target == 1.0 else "provisional",
                target=item.target,
                weight=1.0,
                correction_group=None,
                source_method="dfm",
                feature=item.feature,
            )
            for item in observations
        )
        batch = _encode(converted)
        conversion_logits, rate_logits = model(batch.categorical, batch.numeric)
        time_days = torch.tensor([item.time_days for item in observations], dtype=torch.float32)
        loss = dfm_loss(conversion_logits, rate_logits, batch.targets, time_days)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_norm_clip)
        optimizer.step()
        exposures.extend(
            ExposureRecord(
                credit_id=credit_id,
                step=step,
                record_id=record.record_id,
                weight=1.0,
            )
            for record in converted
        )
    budget.record_credit(steps=config.steps_per_credit, batch_size=config.batch_size)
    budget.assert_exposures(len(exposures))


def _run_method(
    name: str,
    config: StudyAConfig,
    fixture: _Fixture,
    initialization: dict[str, Tensor],
    output_root: Path,
) -> dict[str, Any]:
    configure_determinism(config.seed)
    method = _method(name, config, fixture.truth)
    sampler = DeterministicSampler(
        seed=config.seed,
        recent_window_seconds=config.recent_window_seconds,
        reservoir_capacity=config.reservoir_capacity,
    )
    cursors = _Cursors()
    auxiliary = _new_auxiliary(config) if name == "es_dfm" else None
    trainer: MLPTrainer | None = None
    dfm_model: DelayedFeedbackMLP | None = None
    dfm_optimizer: torch.optim.AdamW | None = None
    dfm_budget = BudgetCounter()
    dfm_exposures: list[ExposureRecord] = []
    if name == "dfm":
        conversion = _new_conversion_model()
        conversion.load_state_dict(initialization)
        dfm_model = DelayedFeedbackMLP(conversion)
        dfm_optimizer = torch.optim.AdamW(
            dfm_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        trainer = _standard_trainer(name, config, initialization, auxiliary)

    schedule = tuple((31 + credit_id) * SECONDS_PER_DAY for credit_id in range(config.credits))
    for credit_id, decision_time in enumerate(schedule):
        _deliver_through(method, sampler, fixture, cursors, decision_time)
        if isinstance(method, ESDFMMethod):
            if auxiliary is None:
                raise ConsistencyError("ES-DFM auxiliary state is unavailable")
            _update_auxiliary(method, auxiliary, config, decision_time, credit_id)
        if name == "dfm":
            if not isinstance(method, DelayedFeedbackMethod):
                raise ConsistencyError("DFM runner has the wrong method type")
            if dfm_model is None or dfm_optimizer is None:
                raise ConsistencyError("DFM model state is unavailable")
            _train_dfm_credit(
                method,
                dfm_model,
                dfm_optimizer,
                sampler,
                config,
                credit_id=credit_id,
                simulator_time=decision_time,
                budget=dfm_budget,
                exposures=dfm_exposures,
            )
        else:
            if trainer is None:
                raise ConsistencyError("Standard Study A trainer is unavailable")
            trainer.spend_credit(
                credit_id=credit_id,
                decision_time=decision_time,
                sampler=sampler,
            )

    if trainer is not None:
        budget = trainer.budget.snapshot()
        exposures = trainer.exposures
        final_model_hash = _model_hash(trainer.model)
    else:
        budget = dfm_budget.snapshot()
        exposures = dfm_exposures
        if dfm_model is None:
            raise ConsistencyError("DFM final model state is unavailable")
        final_model_hash = _model_hash(dfm_model)
    expected_examples = config.credits * config.steps_per_credit * config.batch_size
    if (
        budget.credits != config.credits
        or budget.optimizer_steps != config.credits * config.steps_per_credit
        or budget.optimizer_examples != expected_examples
        or len(exposures) != expected_examples
    ):
        raise ConsistencyError(f"Study A core budget did not reconcile for {name}")
    exposure_values = [record.as_dict() for record in exposures]
    method_root = output_root / name
    method_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(method_root / "exposures.json", exposure_values)
    auxiliary_compute: dict[str, int] = {
        "optimizer_steps": 0,
        "optimizer_examples": 0,
        "forward_examples": 0,
        "backward_examples": 0,
    }
    if name == "dfm":
        auxiliary_compute["forward_examples"] = expected_examples
        auxiliary_compute["backward_examples"] = expected_examples
    if auxiliary is not None:
        auxiliary_compute["optimizer_steps"] = auxiliary.steps
        auxiliary_compute["optimizer_examples"] = auxiliary.examples
        auxiliary_compute["forward_examples"] = auxiliary.examples + expected_examples * 2
        auxiliary_compute["backward_examples"] = auxiliary.examples
    return {
        "method": name,
        "deployable": name != "oracle_reference",
        "ranking_eligible": name != "oracle_reference",
        "schedule_seconds": list(schedule),
        "shared_initialization": True,
        "core": budget.as_dict(),
        "exposure_rows": len(exposures),
        "exposure_sha256": hashlib.sha256(canonical_json_bytes(exposure_values)).hexdigest(),
        "auxiliary": auxiliary_compute,
        "final_model_sha256": final_model_hash,
        "published_number_reproduction": False,
    }


def run_study_a(config: StudyAConfig, output_root: Path) -> dict[str, Any]:
    """Run every method through the same synthetic fixed-budget qualification."""

    if output_root.exists() and any(output_root.iterdir()):
        raise ConfigurationError(f"Output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    fixture = _fixture(config)
    initialization = _initial_checkpoint(config, fixture)
    initialization_model = _new_conversion_model()
    initialization_model.load_state_dict(initialization)
    methods = [
        _run_method(name, config, fixture, initialization, output_root) for name in METHOD_NAMES
    ]
    core_budgets = {canonical_json_bytes(method["core"]) for method in methods}
    schedules = {canonical_json_bytes(method["schedule_seconds"]) for method in methods}
    if len(core_budgets) != 1 or len(schedules) != 1:
        raise ConsistencyError("Study A methods do not share one schedule and core budget")
    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "status": "complete",
        "mode": config.mode,
        "config": config.as_dict(),
        "config_sha256": config.canonical_sha256,
        "shared_initialization_sha256": _model_hash(initialization_model),
        "method_count": len(methods),
        "methods": methods,
        "claims": {
            "synthetic_qualification_only": True,
            "published_number_reproduction": False,
        },
    }
    write_json_atomic(output_root / "manifest.json", manifest)
    return manifest
