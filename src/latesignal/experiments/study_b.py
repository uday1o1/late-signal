"""Synthetic compute-matched qualification for Study B schedulers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from latesignal.contracts.records import ExposureRecord, TrainingRecord
from latesignal.contracts.study_b import StudyBConfig
from latesignal.data.manifests import canonical_json_bytes, write_json_atomic
from latesignal.errors import ConfigurationError, ConsistencyError
from latesignal.models.tiny import TinyLogisticModel
from latesignal.scheduling.base import CreditScheduler
from latesignal.scheduling.calibration_drift import CalibrationDriftCreditScheduler
from latesignal.scheduling.credit import build_credit_windows
from latesignal.scheduling.fixed import FixedWindowScheduler
from latesignal.scheduling.monitoring import (
    MonitoringExample,
    calibration_evidence,
    is_monitoring_member,
)
from latesignal.training.budget import BudgetCounter
from latesignal.training.sampler import DeterministicSampler

DAY = 86_400
POLICIES = ("fixed_early", "fixed_midpoint", "fixed_deadline", "calibration_drift")


@dataclass(frozen=True, slots=True)
class _MonitoringTemplate:
    click_id: str
    click_day: int
    final_label: int
    feature: float


def _next_identifier(prefix: str, start: int, seed: int, *, monitoring: bool) -> tuple[str, int]:
    candidate = start
    while True:
        click_id = f"{prefix}-{candidate}"
        if is_monitoring_member(click_id, seed) is monitoring:
            return click_id, candidate + 1
        candidate += 1


def _monitoring_templates(config: StudyBConfig) -> tuple[_MonitoringTemplate, ...]:
    templates: list[_MonitoringTemplate] = []
    for click_day in range(1, 60):
        candidate = 0
        for index in range(config.monitoring_examples_per_day):
            click_id, candidate = _next_identifier(
                f"monitor-{click_day}",
                candidate,
                config.monitor_seed,
                monitoring=True,
            )
            templates.append(
                _MonitoringTemplate(
                    click_id=click_id,
                    click_day=click_day,
                    final_label=(1 if click_day == config.shift_click_day else int(index % 2 == 0)),
                    feature=0.0,
                )
            )
    return tuple(templates)


def _training_records(config: StudyBConfig) -> tuple[TrainingRecord, ...]:
    records: list[TrainingRecord] = []
    candidate = 0
    for click_day in range(60):
        for index in range(16):
            click_id, candidate = _next_identifier(
                "training",
                candidate,
                config.monitor_seed,
                monitoring=False,
            )
            target = float(index % 2 == 0)
            records.append(
                TrainingRecord(
                    record_id=f"complete_wait:{click_id}:final",
                    click_id=click_id,
                    available_at=(click_day + 30) * DAY,
                    status="final",
                    target=target,
                    weight=1.0,
                    correction_group=None,
                    source_method="complete_wait",
                    feature=1.0 if target == 1.0 else -1.0,
                )
            )
    return tuple(sorted(records, key=lambda item: (item.available_at, item.record_id)))


def _model_hash(model: TinyLogisticModel) -> str:
    return hashlib.sha256(canonical_json_bytes(model.state_dict())).hexdigest()


def _scheduler(name: str, config: StudyBConfig) -> CreditScheduler:
    windows = build_credit_windows(origin=0, window_days=config.window_days)
    if name.startswith("fixed_"):
        return FixedWindowScheduler(windows, policy=name.removeprefix("fixed_"))
    if name == "calibration_drift":
        return CalibrationDriftCreditScheduler(windows, threshold=config.threshold)
    raise ConsistencyError(f"Unknown Study B scheduler: {name}")


def _run_policy(
    name: str,
    config: StudyBConfig,
    monitoring: tuple[_MonitoringTemplate, ...],
    training: tuple[TrainingRecord, ...],
    initialization: dict[str, object],
    output_root: Path,
) -> dict[str, Any]:
    scheduler = _scheduler(name, config)
    model = TinyLogisticModel()
    model.load_state_dict(initialization)
    monitoring_ids = frozenset(item.click_id for item in monitoring)
    sampler = DeterministicSampler(
        seed=config.seed,
        recent_window_seconds=config.recent_window_seconds,
        reservoir_capacity=config.reservoir_capacity,
        excluded_click_ids=monitoring_ids,
    )
    budget = BudgetCounter()
    exposures: list[ExposureRecord] = []
    training_cursor = 0
    monitoring_forward_examples = 0
    for decision_day in range(31, 90):
        decision_time = decision_day * DAY
        while (
            training_cursor < len(training)
            and training[training_cursor].available_at <= decision_time
        ):
            sampler.add(training[training_cursor], decision_time)
            training_cursor += 1
        examples = tuple(
            MonitoringExample(
                click_id=item.click_id,
                click_day=item.click_day,
                final_label=item.final_label,
                probability=model.predict(item.feature),
            )
            for item in monitoring
        )
        evidence = calibration_evidence(
            examples,
            decision_day=decision_day,
            model_checkpoint_sha256=_model_hash(model),
            monitor_seed=config.monitor_seed,
        )
        monitoring_forward_examples += evidence.monitoring_examples
        decision = scheduler.decide(decision_time, evidence)
        if not decision.spend:
            continue
        credit_id = budget.credits
        for step in range(config.steps_per_credit):
            sampled = sampler.sample(
                simulator_time=decision_time,
                batch_size=config.batch_size,
            )
            records = tuple(item.record for item in sampled)
            model.train_step(records, config.learning_rate)
            exposures.extend(
                ExposureRecord(
                    credit_id=credit_id,
                    step=step,
                    record_id=record.record_id,
                    weight=record.weight,
                )
                for record in records
            )
        budget.record_credit(steps=config.steps_per_credit, batch_size=config.batch_size)
        budget.assert_exposures(len(exposures))
    scheduler.assert_complete()
    if any(record.record_id.split(":")[1] in monitoring_ids for record in exposures):
        raise ConsistencyError("Monitoring ID entered the Study B exposure ledger")
    expected_credits = len(scheduler.windows)
    expected_examples = expected_credits * config.steps_per_credit * config.batch_size
    snapshot = budget.snapshot()
    if snapshot.credits != expected_credits or snapshot.optimizer_examples != expected_examples:
        raise ConsistencyError(f"Study B budget did not reconcile for {name}")
    policy_root = output_root / name
    policy_root.mkdir(parents=True, exist_ok=True)
    exposure_values = [record.as_dict() for record in exposures]
    write_json_atomic(policy_root / "exposures.json", exposure_values)
    write_json_atomic(policy_root / "scheduler-audit.json", scheduler.state_dict())
    spend_decisions = [decision for decision in scheduler.decisions if decision.spend]
    return {
        "scheduler": name,
        "core": snapshot.as_dict(),
        "spend_times": [decision.decision_time for decision in spend_decisions],
        "spend_reasons": [decision.reason for decision in spend_decisions],
        "monitoring_decisions": len(scheduler.decisions),
        "monitoring_forward_examples": monitoring_forward_examples,
        "monitoring_exposure_overlap": 0,
        "exposure_rows": len(exposures),
        "exposure_sha256": hashlib.sha256(canonical_json_bytes(exposure_values)).hexdigest(),
        "final_model_sha256": _model_hash(model),
    }


def run_study_b(config: StudyBConfig, output_root: Path) -> dict[str, Any]:
    """Qualify fixed and calibration schedulers under identical core budgets."""

    if output_root.exists() and any(output_root.iterdir()):
        raise ConfigurationError(f"Output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    monitoring = _monitoring_templates(config)
    training = _training_records(config)
    initialization_model = TinyLogisticModel()
    day_zero = tuple(record for record in training if record.available_at == 30 * DAY)
    for _ in range(4):
        initialization_model.train_step(day_zero, config.learning_rate)
    initialization = initialization_model.state_dict()
    policies = [
        _run_policy(name, config, monitoring, training, initialization, output_root)
        for name in POLICIES
    ]
    budgets = {canonical_json_bytes(policy["core"]) for policy in policies}
    monitoring_counts = {policy["monitoring_forward_examples"] for policy in policies}
    if len(budgets) != 1 or len(monitoring_counts) != 1:
        raise ConsistencyError("Study B policies do not share core or monitoring compute")
    adaptive = next(policy for policy in policies if policy["scheduler"] == "calibration_drift")
    deadline = next(policy for policy in policies if policy["scheduler"] == "fixed_deadline")
    if adaptive["spend_times"][0] >= deadline["spend_times"][0]:
        raise ConsistencyError("Synthetic calibration shift did not trigger before the deadline")
    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "status": "complete",
        "mode": config.mode,
        "config": config.as_dict(),
        "config_sha256": config.canonical_sha256,
        "shared_initialization_sha256": _model_hash(initialization_model),
        "policy_count": len(policies),
        "policies": policies,
        "synthetic_shift": {
            "click_day": config.shift_click_day,
            "first_legal_monitoring_day": config.shift_click_day + 30,
            "adaptive_first_spend": adaptive["spend_times"][0] // DAY,
            "deadline_first_spend": deadline["spend_times"][0] // DAY,
            "triggered_earlier": True,
        },
    }
    write_json_atomic(output_root / "manifest.json", manifest)
    return manifest
