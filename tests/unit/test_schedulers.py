from __future__ import annotations

import math

import pytest

from latesignal.contracts.records import TrainingRecord
from latesignal.errors import ConsistencyError
from latesignal.scheduling.calibration_drift import CalibrationDriftCreditScheduler
from latesignal.scheduling.credit import build_credit_windows
from latesignal.scheduling.fixed import FixedWindowScheduler
from latesignal.scheduling.monitoring import (
    CalibrationEvidence,
    MonitoringExample,
    calibration_evidence,
    is_monitoring_member,
)
from latesignal.training.sampler import DeterministicSampler

DAY = 86_400
SEED = 20260813


def _empty_evidence(day: int) -> CalibrationEvidence:
    return calibration_evidence(
        (),
        decision_day=day,
        model_checkpoint_sha256="a" * 64,
        monitor_seed=SEED,
    )


def _member_ids(prefix: str, count: int) -> list[str]:
    result: list[str] = []
    candidate = 0
    while len(result) < count:
        click_id = f"{prefix}-{candidate}"
        if is_monitoring_member(click_id, SEED):
            result.append(click_id)
        candidate += 1
    return result


def _cohort(day: int, *, shifted: bool) -> list[MonitoringExample]:
    identifiers = _member_ids(f"day-{day}", 1_000)
    return [
        MonitoringExample(
            click_id=click_id,
            click_day=day,
            final_label=1 if shifted or index % 2 == 0 else 0,
            probability=0.5,
        )
        for index, click_id in enumerate(identifiers)
    ]


def test_credit_windows_retain_final_partial_window() -> None:
    windows = build_credit_windows(origin=0)

    assert len(windows) == 12
    assert (windows[0].early_time, windows[0].midpoint_time, windows[0].deadline_time) == (
        31 * DAY,
        34 * DAY,
        35 * DAY,
    )
    assert (windows[-1].start_time, windows[-1].end_time, windows[-1].deadline_time) == (
        86 * DAY,
        90 * DAY,
        89 * DAY,
    )


@pytest.mark.parametrize(
    ("policy", "expected_first"),
    [("early", 31), ("midpoint", 34), ("deadline", 35)],
)
def test_fixed_schedulers_spend_exactly_once_per_window(policy: str, expected_first: int) -> None:
    scheduler = FixedWindowScheduler(build_credit_windows(origin=0), policy=policy)

    for day in range(31, 90):
        scheduler.decide(day * DAY, _empty_evidence(day))
    scheduler.assert_complete()
    spends = [decision for decision in scheduler.decisions if decision.spend]

    assert len(spends) == 12
    assert spends[0].decision_time == expected_first * DAY
    assert len(scheduler.monitoring_log) == 59


def test_calibration_residual_matches_hand_calculation() -> None:
    examples = tuple(_cohort(0, shifted=True) + _cohort(1, shifted=True))

    evidence = calibration_evidence(
        examples,
        decision_day=31,
        model_checkpoint_sha256="b" * 64,
        monitor_seed=SEED,
    )

    assert evidence.monitoring_examples == 1_000
    assert evidence.contributing_bin == 5
    assert evidence.score == pytest.approx(500.0 / math.sqrt(250.0 + 1e-8))
    selected = evidence.bins[5]
    assert selected.count == 1_000
    assert selected.positives == 1_000
    assert selected.signed_residual_sum == 500.0
    assert selected.variance_sum == 250.0
    assert evidence.monitoring_cohort_last_day == 0


def test_monitoring_excludes_cohort_that_only_starts_maturing_at_boundary() -> None:
    evidence = calibration_evidence(
        tuple(_cohort(0, shifted=False) + _cohort(1, shifted=True)),
        decision_day=31,
        model_checkpoint_sha256="d" * 64,
        monitor_seed=SEED,
    )

    assert evidence.monitoring_cohort_last_day == 0
    assert evidence.monitoring_examples == 1_000
    assert evidence.bins[5].positives == 500


def test_calibration_scheduler_triggers_early_on_mature_shift_and_is_reproducible() -> None:
    examples = tuple(
        _cohort(1, shifted=False) + _cohort(2, shifted=False) + _cohort(3, shifted=True)
    )
    spend_times: list[int] = []
    audit_logs: list[list[dict[str, object]]] = []
    for _ in range(2):
        scheduler = CalibrationDriftCreditScheduler(build_credit_windows(origin=0))
        for day in range(31, 36):
            evidence = calibration_evidence(
                examples,
                decision_day=day,
                model_checkpoint_sha256="c" * 64,
                monitor_seed=SEED,
            )
            scheduler.decide(day * DAY, evidence)
        spends = [decision for decision in scheduler.decisions if decision.spend]
        spend_times.append(spends[0].decision_time)
        audit_logs.append(scheduler.evidence_log)

    assert spend_times == [34 * DAY, 34 * DAY]
    assert audit_logs[0] == audit_logs[1]
    assert audit_logs[0][3]["monitoring_cohort_last_day"] == 3
    assert audit_logs[0][3]["model_checkpoint_sha256"] == "c" * 64


def test_unsupported_calibration_cannot_trigger_and_deadline_forces_spend() -> None:
    scheduler = CalibrationDriftCreditScheduler(build_credit_windows(origin=0))

    for day in range(31, 36):
        decision = scheduler.decide(day * DAY, _empty_evidence(day))

    assert decision.spend is True
    assert decision.reason == "forced_deadline"
    assert decision.score is None


def test_monitoring_membership_is_stable_and_sampler_rejects_monitoring_ids() -> None:
    click_id = _member_ids("protected", 1)[0]
    assert is_monitoring_member(click_id, SEED)
    sampler = DeterministicSampler(
        seed=17,
        recent_window_seconds=DAY,
        reservoir_capacity=10,
        excluded_click_ids=frozenset({click_id}),
    )
    record = TrainingRecord(
        record_id="monitoring-record",
        click_id=click_id,
        available_at=0,
        status="final",
        target=1.0,
        weight=1.0,
        correction_group=None,
        source_method="test",
        feature=0.0,
    )

    with pytest.raises(ConsistencyError, match="Monitoring record"):
        sampler.add(record, 0)
