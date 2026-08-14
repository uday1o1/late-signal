from __future__ import annotations

import numpy as np
import pytest

from latesignal.errors import ConsistencyError
from latesignal.methods.production import PackedDelayedMethod
from latesignal.simulator.production_oracle import SECONDS_PER_DAY, TruthEventBatch
from latesignal.training.packed import PackedRecordStore, RecordKind


def _truth(refs: list[int], times: list[float], labels: list[int]) -> TruthEventBatch:
    return TruthEventBatch(
        feature_refs=np.asarray(refs, dtype=np.int32),
        available_at=np.asarray(times, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int8),
    )


def _empty_truth() -> TruthEventBatch:
    return _truth([], [], [])


def test_fixed_wait_orders_early_late_and_unresolved_records() -> None:
    day = SECONDS_PER_DAY
    click_times = np.asarray([0.0, 0.5 * day, day], dtype=np.float64)
    store = PackedRecordStore(feature_count=3)
    method = PackedDelayedMethod(
        "fixed_wait",
        click_times=click_times,
        monitoring_mask=np.zeros(3, dtype=np.bool_),
        main_store=store,
        wait_days=1,
    )

    method.process_boundary(
        boundary=2 * day,
        click_refs=np.arange(3, dtype=np.int32),
        truth=_truth([0, 1], [day, 1.6 * day], [1, 1]),
    )

    assert store.feature_refs[: len(store)].tolist() == [0, 1, 1, 2]
    assert store.kinds[: len(store)].tolist() == [
        RecordKind.EARLY_POSITIVE,
        RecordKind.PROVISIONAL,
        RecordKind.CORRECTION,
        RecordKind.PROVISIONAL,
    ]
    assert store.available_at[: len(store)].tolist() == pytest.approx(
        [day, 1.5 * day, 1.6 * day, 2 * day]
    )


def test_immediate_and_dfm_exclude_monitoring_from_training() -> None:
    click_times = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    monitoring = np.asarray([False, True, False])
    immediate_store = PackedRecordStore(feature_count=3)
    immediate = PackedDelayedMethod(
        "immediate_fake_negative",
        click_times=click_times,
        monitoring_mask=monitoring,
        main_store=immediate_store,
    )
    immediate.process_boundary(
        boundary=3.0,
        click_refs=np.arange(3, dtype=np.int32),
        truth=_truth([0, 1], [2.5, 2.5], [1, 1]),
    )
    assert immediate_store.feature_refs[: len(immediate_store)].tolist() == [0, 2, 0]
    assert immediate_store.kinds[: len(immediate_store)].tolist() == [
        RecordKind.PROVISIONAL,
        RecordKind.PROVISIONAL,
        RecordKind.CORRECTION,
    ]

    dfm_store = PackedRecordStore(feature_count=3)
    dfm = PackedDelayedMethod(
        "dfm",
        click_times=click_times,
        monitoring_mask=monitoring,
        main_store=dfm_store,
    )
    dfm.process_boundary(
        boundary=3.0,
        click_refs=np.arange(3, dtype=np.int32),
        truth=_empty_truth(),
    )
    assert dfm_store.feature_refs[: len(dfm_store)].tolist() == [0, 2]
    assert dfm_store.kinds[: len(dfm_store)].tolist() == [
        RecordKind.DFM_CLICK,
        RecordKind.DFM_CLICK,
    ]


def test_esdfm_emits_only_legally_mature_auxiliary_targets() -> None:
    day = SECONDS_PER_DAY
    click_times = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    main = PackedRecordStore(feature_count=3)
    q_tn = PackedRecordStore(feature_count=3)
    q_dp = PackedRecordStore(feature_count=3)
    method = PackedDelayedMethod(
        "es_dfm",
        click_times=click_times,
        monitoring_mask=np.zeros(3, dtype=np.bool_),
        main_store=main,
        q_tn_store=q_tn,
        q_dp_store=q_dp,
        wait_days=1,
    )
    method.process_boundary(
        boundary=31 * day,
        click_refs=np.arange(3, dtype=np.int32),
        truth=_truth([0, 1, 2], [0.5 * day, 2 * day, 30 * day + 2.0], [1, 1, 0]),
    )

    assert q_dp.feature_refs[: len(q_dp)].tolist() == [0, 1, 2]
    assert q_dp.targets[: len(q_dp)].tolist() == [0.0, 1.0, 0.0]
    assert q_tn.feature_refs[: len(q_tn)].tolist() == [1, 2]
    assert q_tn.targets[: len(q_tn)].tolist() == [0.0, 1.0]


def test_production_method_resume_and_truth_order_fail_closed() -> None:
    day = SECONDS_PER_DAY
    click_times = np.asarray([0.0, 1.0], dtype=np.float64)
    store = PackedRecordStore(feature_count=2)
    method = PackedDelayedMethod(
        "complete_wait",
        click_times=click_times,
        monitoring_mask=np.zeros(2, dtype=np.bool_),
        main_store=store,
    )
    method.process_boundary(
        boundary=1.0,
        click_refs=np.arange(2, dtype=np.int32),
        truth=_truth([0], [1.0], [1]),
    )
    assert len(store) == 0
    state = method.state_dict()
    restored_store = PackedRecordStore(feature_count=2)
    restored_store.load_state_dict(store.state_dict())
    restored = PackedDelayedMethod(
        "complete_wait",
        click_times=click_times,
        monitoring_mask=np.zeros(2, dtype=np.bool_),
        main_store=restored_store,
    )
    restored.load_state_dict(state)
    restored.process_boundary(
        boundary=30 * day + 2.0,
        click_refs=np.asarray([], dtype=np.int32),
        truth=_truth([1], [30 * day + 1.0], [0]),
    )
    assert restored_store.targets[: len(restored_store)].tolist() == [1.0, 0.0]
    assert restored_store.available_at[: len(restored_store)].tolist() == [
        30 * day,
        30 * day + 1.0,
    ]

    with pytest.raises(ConsistencyError, match="before click"):
        PackedDelayedMethod(
            "complete_wait",
            click_times=click_times,
            monitoring_mask=np.zeros(2, dtype=np.bool_),
            main_store=PackedRecordStore(feature_count=2),
        ).process_boundary(
            boundary=0.0,
            click_refs=np.asarray([], dtype=np.int32),
            truth=_truth([0], [0.0], [1]),
        )

    with pytest.raises(ConsistencyError, match="before click"):
        PackedDelayedMethod(
            "complete_wait",
            click_times=click_times,
            monitoring_mask=np.zeros(2, dtype=np.bool_),
            main_store=PackedRecordStore(feature_count=2),
        ).process_boundary(
            boundary=0.0,
            click_refs=np.asarray([0], dtype=np.int32),
            truth=_truth([0], [1.0], [1]),
        )


def test_production_method_rejects_truth_shifted_before_legal_availability() -> None:
    day = SECONDS_PER_DAY
    click_times = np.asarray([day], dtype=np.float64)

    with pytest.raises(ConsistencyError, match="legal availability"):
        PackedDelayedMethod(
            "complete_wait",
            click_times=click_times,
            monitoring_mask=np.zeros(1, dtype=np.bool_),
            main_store=PackedRecordStore(feature_count=1),
        ).process_boundary(
            boundary=day,
            click_refs=np.asarray([0], dtype=np.int32),
            truth=_truth([0], [day - 1.0], [1]),
        )

    with pytest.raises(ConsistencyError, match="legal availability"):
        PackedDelayedMethod(
            "complete_wait",
            click_times=click_times,
            monitoring_mask=np.zeros(1, dtype=np.bool_),
            main_store=PackedRecordStore(feature_count=1),
        ).process_boundary(
            boundary=31 * day,
            click_refs=np.asarray([0], dtype=np.int32),
            truth=_truth([0], [30 * day], [0]),
        )
