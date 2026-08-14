from __future__ import annotations

import numpy as np
import pytest

from latesignal.errors import ConsistencyError
from latesignal.training.packed import (
    PackedDeterministicSampler,
    PackedRecordStore,
    RecordKind,
    packed_record_batch,
)


def _append(
    store: PackedRecordStore,
    refs: list[int],
    times: list[float],
    *,
    kind: RecordKind = RecordKind.FINAL,
) -> None:
    store.append(
        packed_record_batch(
            feature_refs=np.asarray(refs),
            available_at=np.asarray(times),
            targets=np.asarray([float(index % 2) for index in refs]),
            kinds=np.full(len(refs), kind),
        ),
        simulator_time=max(times),
    )


def test_packed_sampler_separates_recent_and_reservoir_records() -> None:
    store = PackedRecordStore(feature_count=20, initial_capacity=2)
    _append(store, list(range(10)), [float(index) for index in range(10)])
    sampler = PackedDeterministicSampler(
        store,
        seed=17,
        recent_window_seconds=3.0,
        reservoir_capacity=4,
    )

    sample = sampler.sample(simulator_time=9.0, batch_size=8)

    assert sample.sources.tolist().count(0) == 4
    assert sample.sources.tolist().count(1) == 4
    assert all(index >= 6 for index in sample.store_indices[:4])
    assert all(index < 6 for index in sample.store_indices[4:])

    recent_only = PackedDeterministicSampler(
        store,
        seed=17,
        recent_window_seconds=20.0,
        reservoir_capacity=4,
    ).sample(simulator_time=9.0, batch_size=4)
    assert recent_only.sources.tolist() == [0, 0, 0, 0]


def test_packed_sampler_is_stable_across_append_chunking_and_resume() -> None:
    stores: list[PackedRecordStore] = []
    first = PackedRecordStore(feature_count=20)
    _append(first, list(range(10)), [float(index) for index in range(10)])
    stores.append(first)
    second = PackedRecordStore(feature_count=20)
    _append(second, list(range(5)), [float(index) for index in range(5)])
    _append(second, list(range(5, 10)), [float(index) for index in range(5, 10)])
    stores.append(second)
    samplers = [
        PackedDeterministicSampler(
            store,
            seed=41,
            recent_window_seconds=2.0,
            reservoir_capacity=3,
        )
        for store in stores
    ]

    first_sample = samplers[0].sample(simulator_time=9.0, batch_size=10)
    second_sample = samplers[1].sample(simulator_time=9.0, batch_size=10)
    resumed = PackedDeterministicSampler(
        stores[0],
        seed=41,
        recent_window_seconds=2.0,
        reservoir_capacity=3,
    )
    resumed.load_state_dict(samplers[0].state_dict())

    assert stores[0].sha256 == stores[1].sha256
    assert first_sample.store_indices.tolist() == second_sample.store_indices.tolist()
    assert (
        resumed.sample(simulator_time=9.0, batch_size=10).store_indices.tolist()
        == samplers[0].sample(simulator_time=9.0, batch_size=10).store_indices.tolist()
    )


def test_packed_store_round_trips_and_rejects_duplicate_or_future_records() -> None:
    store = PackedRecordStore(feature_count=4)
    _append(store, [0, 1], [1.0, 2.0])
    restored = PackedRecordStore(feature_count=4)
    restored.load_state_dict(store.state_dict())

    assert restored.rebuild_token() == store.rebuild_token()
    with pytest.raises(ConsistencyError, match="emitted twice"):
        _append(store, [0], [3.0])
    with pytest.raises(ConsistencyError, match="future"):
        store.append(
            packed_record_batch(
                feature_refs=np.asarray([2]),
                available_at=np.asarray([5.0]),
                targets=np.asarray([0.0]),
                kinds=np.asarray([RecordKind.FINAL]),
            ),
            simulator_time=4.0,
        )


def test_packed_sampler_rejects_empty_pool_and_forged_reservoir() -> None:
    store = PackedRecordStore(feature_count=3)
    sampler = PackedDeterministicSampler(
        store,
        seed=73,
        recent_window_seconds=1.0,
        reservoir_capacity=1,
    )
    with pytest.raises(ConsistencyError, match="INSUFFICIENT_LEGAL_POOL"):
        sampler.sample(simulator_time=0.0, batch_size=2)

    _append(store, [0, 1, 2], [0.0, 1.0, 2.0])
    sampler.sample(simulator_time=3.0, batch_size=2)
    state = sampler.state_dict()
    state["reservoir"] = state["reservoir"] + 1  # type: ignore[operator]
    with pytest.raises(ConsistencyError, match="reservoir is invalid"):
        PackedDeterministicSampler(
            store,
            seed=73,
            recent_window_seconds=1.0,
            reservoir_capacity=1,
        ).load_state_dict(state)
