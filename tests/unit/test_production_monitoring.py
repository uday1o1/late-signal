from __future__ import annotations

import numpy as np
import pytest

from latesignal.errors import ConsistencyError
from latesignal.scheduling.production import PackedMonitoringState
from latesignal.simulator.production_oracle import TruthEventBatch


class _Predictor:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def predict(self, references: np.ndarray) -> np.ndarray:
        self.batch_sizes.append(references.size)
        return np.full(references.size, 0.5, dtype=np.float32)


def _truth(refs: np.ndarray, labels: np.ndarray) -> TruthEventBatch:
    return TruthEventBatch(
        feature_refs=refs.astype(np.int32),
        available_at=np.zeros(refs.size, dtype=np.float64),
        labels=labels.astype(np.int8),
    )


def test_packed_monitoring_uses_only_newest_fully_mature_cohorts() -> None:
    per_day = 1_000
    days = np.repeat(np.arange(8, dtype=np.int16), per_day)
    monitoring = np.ones(days.size, dtype=np.bool_)
    state = PackedMonitoringState(
        click_days=days,
        monitoring_mask=monitoring,
        inference_batch_size=256,
    )
    refs = np.arange(days.size, dtype=np.int32)
    labels = np.tile(np.arange(per_day) % 2, 8)
    state.observe_truth(_truth(refs, labels))

    predictor = _Predictor()
    day_31 = state.evidence(
        decision_day=31,
        predictor=predictor,
        model_checkpoint_sha256="a" * 64,
    )
    later = PackedMonitoringState(
        click_days=days,
        monitoring_mask=monitoring,
        inference_batch_size=256,
    )
    later.observe_truth(_truth(refs, labels))
    day_38 = None
    for decision_day in range(31, 39):
        day_38 = later.evidence(
            decision_day=decision_day,
            predictor=_Predictor(),
            model_checkpoint_sha256="b" * 64,
        )
    assert day_38 is not None

    assert (day_31.monitoring_cohort_first_day, day_31.monitoring_cohort_last_day) == (-6, 0)
    assert day_31.monitoring_examples == 1_000
    assert day_31.score == 0.0
    assert (day_38.monitoring_cohort_first_day, day_38.monitoring_cohort_last_day) == (1, 7)
    assert day_38.monitoring_examples == 7_000
    assert state.inference_examples == 1_000
    assert max(predictor.batch_sizes) <= 256


def test_packed_monitoring_checkpoint_round_trip() -> None:
    days = np.zeros(1_000, dtype=np.int16)
    monitoring = np.ones(1_000, dtype=np.bool_)
    state = PackedMonitoringState(click_days=days, monitoring_mask=monitoring)
    refs = np.arange(1_000, dtype=np.int32)
    state.observe_truth(_truth(refs, refs % 2))
    expected = state.evidence(
        decision_day=31,
        predictor=_Predictor(),
        model_checkpoint_sha256="a" * 64,
    )
    restored = PackedMonitoringState(click_days=days, monitoring_mask=monitoring)
    restored.load_state_dict(state.state_dict())

    assert restored.evidence_log == [expected.as_dict()]
    assert restored.inference_examples == 1_000


def test_packed_monitoring_rejects_skipped_days_and_immature_truth() -> None:
    days = np.repeat(np.arange(2, dtype=np.int16), 1_000)
    monitoring = np.ones(days.size, dtype=np.bool_)
    incomplete = PackedMonitoringState(click_days=days, monitoring_mask=monitoring)
    incomplete_refs = np.arange(999, dtype=np.int32)
    incomplete.observe_truth(_truth(incomplete_refs, incomplete_refs % 2))

    with pytest.raises(ConsistencyError, match="fully mature"):
        incomplete.evidence(
            decision_day=31,
            predictor=_Predictor(),
            model_checkpoint_sha256="a" * 64,
        )
    state = PackedMonitoringState(click_days=days, monitoring_mask=monitoring)
    refs = np.arange(1_000, dtype=np.int32)
    state.observe_truth(_truth(refs, refs % 2))
    state.evidence(
        decision_day=31,
        predictor=_Predictor(),
        model_checkpoint_sha256="a" * 64,
    )
    with pytest.raises(ConsistencyError, match="every day"):
        state.evidence(
            decision_day=33,
            predictor=_Predictor(),
            model_checkpoint_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    ("refs", "labels"),
    [
        (np.array([0, 0]), np.array([0, 0])),
        (np.array([0]), np.array([2])),
        (np.array([-1]), np.array([0])),
    ],
)
def test_packed_monitoring_rejects_malformed_truth(
    refs: np.ndarray,
    labels: np.ndarray,
) -> None:
    state = PackedMonitoringState(
        click_days=np.zeros(2, dtype=np.int16),
        monitoring_mask=np.ones(2, dtype=np.bool_),
    )

    with pytest.raises(ConsistencyError, match="malformed"):
        state.observe_truth(_truth(refs, labels))


def test_packed_monitoring_rejects_inconsistent_checkpoint_log() -> None:
    state = PackedMonitoringState(
        click_days=np.zeros(1_000, dtype=np.int16),
        monitoring_mask=np.ones(1_000, dtype=np.bool_),
    )
    refs = np.arange(1_000, dtype=np.int32)
    state.observe_truth(_truth(refs, refs % 2))
    state.evidence(
        decision_day=31,
        predictor=_Predictor(),
        model_checkpoint_sha256="a" * 64,
    )
    checkpoint = state.state_dict()
    log = checkpoint["evidence_log"]
    assert isinstance(log, list)
    assert isinstance(log[0], dict)
    log[0]["monitoring_examples"] = 999

    restored = PackedMonitoringState(
        click_days=np.zeros(1_000, dtype=np.int16),
        monitoring_mask=np.ones(1_000, dtype=np.bool_),
    )
    with pytest.raises(ConsistencyError, match="inconsistent"):
        restored.load_state_dict(checkpoint)
    assert state.evidence_log[0]["monitoring_examples"] == 1_000


def test_packed_monitoring_checkpoint_binds_click_day_identity() -> None:
    click_days = np.zeros(1_000, dtype=np.int16)
    monitoring = np.ones(1_000, dtype=np.bool_)
    state = PackedMonitoringState(click_days=click_days, monitoring_mask=monitoring)
    checkpoint = state.state_dict()
    click_days[:] = 1
    monitoring[:] = False

    assert np.all(state.click_days == 0)
    assert state.monitoring_mask.all()
    changed = PackedMonitoringState(
        click_days=np.ones(1_000, dtype=np.int16),
        monitoring_mask=np.ones(1_000, dtype=np.bool_),
    )
    with pytest.raises(ConsistencyError, match="malformed"):
        changed.load_state_dict(checkpoint)
