"""Packed legal-record storage and deterministic recent-reservoir sampling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from latesignal.errors import ConsistencyError

_UINT64_MASK = (1 << 64) - 1
_PACKED_DTYPE = np.dtype(
    [
        ("feature_ref", "<i4"),
        ("available_at", "<f8"),
        ("target", "<f4"),
        ("kind", "u1"),
    ],
    align=False,
)


class RecordKind(IntEnum):
    FINAL = 0
    PROVISIONAL = 1
    CORRECTION = 2
    EARLY_POSITIVE = 3
    DFM_CLICK = 4
    ORACLE_FINAL = 5


@dataclass(frozen=True, slots=True)
class PackedRecordBatch:
    feature_refs: NDArray[np.int32]
    available_at: NDArray[np.float64]
    targets: NDArray[np.float32]
    kinds: NDArray[np.uint8]

    def __len__(self) -> int:
        return int(self.feature_refs.size)


@dataclass(frozen=True, slots=True)
class PackedSample:
    store_indices: NDArray[np.int32]
    feature_refs: NDArray[np.int32]
    targets: NDArray[np.float32]
    kinds: NDArray[np.uint8]
    sources: NDArray[np.uint8]

    @property
    def record_keys(self) -> NDArray[np.uint64]:
        return record_keys(self.feature_refs, self.kinds)


def packed_record_batch(
    *,
    feature_refs: NDArray[np.integer],
    available_at: NDArray[np.floating],
    targets: NDArray[np.floating],
    kinds: NDArray[np.integer],
) -> PackedRecordBatch:
    return PackedRecordBatch(
        feature_refs=np.asarray(feature_refs, dtype=np.int32),
        available_at=np.asarray(available_at, dtype=np.float64),
        targets=np.asarray(targets, dtype=np.float32),
        kinds=np.asarray(kinds, dtype=np.uint8),
    )


def record_keys(
    feature_refs: NDArray[np.integer],
    kinds: NDArray[np.integer],
) -> NDArray[np.uint64]:
    refs = np.asarray(feature_refs, dtype=np.uint64)
    parsed_kinds = np.asarray(kinds, dtype=np.uint64)
    return (refs << np.uint64(8)) | parsed_kinds


def _mix64(values: NDArray[np.uint64]) -> NDArray[np.uint64]:
    mixed = values.copy()
    mixed ^= mixed >> np.uint64(30)
    mixed *= np.uint64(0xBF58476D1CE4E5B9)
    mixed ^= mixed >> np.uint64(27)
    mixed *= np.uint64(0x94D049BB133111EB)
    mixed ^= mixed >> np.uint64(31)
    return mixed


def _record_bytes(batch: PackedRecordBatch) -> bytes:
    values = np.empty(len(batch), dtype=_PACKED_DTYPE)
    values["feature_ref"] = batch.feature_refs
    values["available_at"] = batch.available_at
    values["target"] = batch.targets
    values["kind"] = batch.kinds
    return values.tobytes()


class PackedRecordStore:
    """Append legal records into compact arrays with dense duplicate protection."""

    def __init__(self, *, feature_count: int, initial_capacity: int = 65_536) -> None:
        if feature_count <= 0 or initial_capacity <= 0:
            raise ValueError("Packed record-store capacities must be positive")
        self.feature_count = feature_count
        self._capacity = initial_capacity
        self._size = 0
        self.feature_refs = np.empty(initial_capacity, dtype=np.int32)
        self.available_at = np.empty(initial_capacity, dtype=np.float64)
        self.targets = np.empty(initial_capacity, dtype=np.float32)
        self.kinds = np.empty(initial_capacity, dtype=np.uint8)
        self._emitted_kind_mask = np.zeros(feature_count, dtype=np.uint16)
        self._digest = hashlib.sha256()

    def __len__(self) -> int:
        return self._size

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()

    def _ensure_capacity(self, required: int) -> None:
        if required <= self._capacity:
            return
        capacity = self._capacity
        while capacity < required:
            capacity = max(capacity * 2, required)
        self.feature_refs = np.resize(self.feature_refs, capacity)
        self.available_at = np.resize(self.available_at, capacity)
        self.targets = np.resize(self.targets, capacity)
        self.kinds = np.resize(self.kinds, capacity)
        self._capacity = capacity

    def append(self, batch: PackedRecordBatch, *, simulator_time: float) -> slice:
        lengths = {
            batch.feature_refs.size,
            batch.available_at.size,
            batch.targets.size,
            batch.kinds.size,
        }
        if lengths == {0} or len(lengths) != 1:
            raise ConsistencyError("Packed training columns must be nonempty and aligned")
        if (
            np.any(batch.feature_refs < 0)
            or np.any(batch.feature_refs >= self.feature_count)
            or not np.isfinite(batch.available_at).all()
            or np.any(batch.available_at > simulator_time)
            or np.any(np.diff(batch.available_at) < 0.0)
            or not np.isfinite(batch.targets).all()
            or np.any(batch.targets < 0.0)
            or np.any(batch.targets > 1.0)
            or np.any(batch.kinds >= 16)
        ):
            raise ConsistencyError("Packed training batch contains an invalid or future record")
        if self._size and batch.available_at[0] < self.available_at[self._size - 1]:
            raise ConsistencyError("Packed training availability cannot move backward")
        keys = record_keys(batch.feature_refs, batch.kinds)
        if np.unique(keys).size != keys.size:
            raise ConsistencyError("Packed training batch contains duplicate record keys")
        bits = np.left_shift(np.uint16(1), batch.kinds.astype(np.uint16))
        if np.any(np.bitwise_and(self._emitted_kind_mask[batch.feature_refs], bits)):
            raise ConsistencyError("Packed training record was emitted twice")
        start = self._size
        end = start + len(batch)
        self._ensure_capacity(end)
        self.feature_refs[start:end] = batch.feature_refs
        self.available_at[start:end] = batch.available_at
        self.targets[start:end] = batch.targets
        self.kinds[start:end] = batch.kinds
        np.bitwise_or.at(self._emitted_kind_mask, batch.feature_refs, bits)
        self._digest.update(_record_bytes(batch))
        self._size = end
        return slice(start, end)

    def take(
        self,
        indices: NDArray[np.integer],
        *,
        sources: NDArray[np.integer] | None = None,
    ) -> PackedSample:
        parsed = np.asarray(indices, dtype=np.int32)
        if parsed.ndim != 1 or np.any(parsed < 0) or np.any(parsed >= self._size):
            raise ConsistencyError("Packed sampler selected an invalid record index")
        if sources is None:
            parsed_sources = np.zeros(parsed.size, dtype=np.uint8)
        else:
            parsed_sources = np.asarray(sources, dtype=np.uint8)
            if parsed_sources.shape != parsed.shape or np.any(parsed_sources > 1):
                raise ConsistencyError("Packed sampler sources are invalid")
        return PackedSample(
            store_indices=parsed,
            feature_refs=self.feature_refs[parsed],
            targets=self.targets[parsed],
            kinds=self.kinds[parsed],
            sources=parsed_sources,
        )

    def rebuild_token(self) -> dict[str, object]:
        return {
            "feature_count": self.feature_count,
            "records": self._size,
            "records_sha256": self.sha256,
        }

    def state_dict(self) -> dict[str, object]:
        end = self._size
        return {
            **self.rebuild_token(),
            "feature_refs": torch.from_numpy(self.feature_refs[:end].copy()),
            "available_at": torch.from_numpy(self.available_at[:end].copy()),
            "targets": torch.from_numpy(self.targets[:end].copy()),
            "kinds": torch.from_numpy(self.kinds[:end].copy()),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("feature_count") != self.feature_count:
            raise ConsistencyError("Packed record-store feature identity changed")
        tensors = tuple(
            state.get(name) for name in ("feature_refs", "available_at", "targets", "kinds")
        )
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise ConsistencyError("Packed record-store checkpoint tensors are malformed")
        refs_tensor, available_tensor, targets_tensor, kinds_tensor = tensors
        assert isinstance(refs_tensor, torch.Tensor)
        assert isinstance(available_tensor, torch.Tensor)
        assert isinstance(targets_tensor, torch.Tensor)
        assert isinstance(kinds_tensor, torch.Tensor)
        batch = packed_record_batch(
            feature_refs=refs_tensor.cpu().numpy(),
            available_at=available_tensor.cpu().numpy(),
            targets=targets_tensor.cpu().numpy(),
            kinds=kinds_tensor.cpu().numpy(),
        )
        replacement = PackedRecordStore(
            feature_count=self.feature_count,
            initial_capacity=max(1, len(batch)),
        )
        simulator_time = float(batch.available_at[-1]) if len(batch) else 0.0
        if len(batch):
            replacement.append(batch, simulator_time=simulator_time)
        if replacement.rebuild_token() != {
            "feature_count": state.get("feature_count"),
            "records": state.get("records"),
            "records_sha256": state.get("records_sha256"),
        }:
            raise ConsistencyError("Packed record-store checkpoint digest is invalid")
        self.__dict__.update(replacement.__dict__)


class _CounterRNG:
    def __init__(self, seed: int) -> None:
        self.seed = seed & _UINT64_MASK
        self.counter = 0

    def draw(self, *, high: int, size: int) -> NDArray[np.int64]:
        if high < 0 or size < 0:
            raise ValueError("Counter RNG bounds are invalid")
        if size == 0:
            return np.empty(0, dtype=np.int64)
        if high == 0:
            raise ValueError("Counter RNG cannot draw from an empty range")
        offsets = np.arange(self.counter, self.counter + size, dtype=np.uint64)
        raw = _mix64(offsets ^ np.uint64(self.seed))
        self.counter += size
        return np.asarray(raw % np.uint64(high), dtype=np.int64)


class PackedDeterministicSampler:
    """Sample half recent records and half from a priority reservoir."""

    def __init__(
        self,
        store: PackedRecordStore,
        *,
        seed: int,
        recent_window_seconds: float,
        reservoir_capacity: int,
    ) -> None:
        if recent_window_seconds <= 0.0 or reservoir_capacity <= 0:
            raise ValueError("Packed sampler limits must be positive")
        self.store = store
        self.seed = seed
        self.recent_window_seconds = float(recent_window_seconds)
        self.reservoir_capacity = reservoir_capacity
        self._older_cursor = 0
        self._reservoir = np.empty(0, dtype=np.int32)
        self._reservoir_priorities = np.empty(0, dtype=np.uint64)
        self._rng = _CounterRNG(seed ^ 0xA5A5A5A5A5A5A5A5)
        self._last_time: float | None = None

    def _priorities(self, indices: NDArray[np.int32]) -> NDArray[np.uint64]:
        keys = record_keys(self.store.feature_refs[indices], self.store.kinds[indices])
        return _mix64(keys ^ np.uint64(self.seed & _UINT64_MASK))

    def _bounded_reservoir(
        self,
        indices: NDArray[np.int32],
        priorities: NDArray[np.uint64],
    ) -> tuple[NDArray[np.int32], NDArray[np.uint64]]:
        if indices.size <= self.reservoir_capacity:
            return indices, priorities
        boundary = np.partition(priorities, self.reservoir_capacity - 1)[
            self.reservoir_capacity - 1
        ]
        lower = np.flatnonzero(priorities < boundary)
        remaining = self.reservoir_capacity - lower.size
        equal = np.flatnonzero(priorities == boundary)
        if remaining < equal.size:
            equal_keys = record_keys(
                self.store.feature_refs[indices[equal]],
                self.store.kinds[indices[equal]],
            )
            equal = equal[np.argsort(equal_keys, kind="stable")[:remaining]]
        selected_mask = np.zeros(indices.size, dtype=np.bool_)
        selected_mask[lower] = True
        selected_mask[equal[:remaining]] = True
        selected = np.flatnonzero(selected_mask)
        return indices[selected], priorities[selected]

    def advance(self, simulator_time: float) -> None:
        if not np.isfinite(simulator_time):
            raise ConsistencyError("Packed sampler time must be finite")
        if self._last_time is not None and simulator_time < self._last_time:
            raise ConsistencyError("Packed sampler time cannot move backward")
        if len(self.store) and self.store.available_at[len(self.store) - 1] > simulator_time:
            raise ConsistencyError("Packed sampler store contains a future record")
        cutoff = simulator_time - self.recent_window_seconds
        recent_start = int(
            np.searchsorted(
                self.store.available_at[: len(self.store)],
                cutoff,
                side="left",
            )
        )
        if recent_start > self._older_cursor:
            new_indices = np.arange(self._older_cursor, recent_start, dtype=np.int32)
            new_priorities = self._priorities(new_indices)
            candidates = np.concatenate((self._reservoir, new_indices))
            priorities = np.concatenate((self._reservoir_priorities, new_priorities))
            self._reservoir, self._reservoir_priorities = self._bounded_reservoir(
                candidates, priorities
            )
            self._older_cursor = recent_start
        self._last_time = simulator_time

    def sample(self, *, simulator_time: float, batch_size: int) -> PackedSample:
        if batch_size <= 0:
            raise ValueError("Packed sampler batch size must be positive")
        self.advance(simulator_time)
        recent_size = len(self.store) - self._older_cursor
        reservoir_size = self._reservoir.size
        if recent_size == 0 and reservoir_size == 0:
            raise ConsistencyError("INSUFFICIENT_LEGAL_POOL")
        recent_count = batch_size // 2
        reservoir_count = batch_size - recent_count
        if recent_size == 0:
            recent_count = 0
            reservoir_count = batch_size
        elif reservoir_size == 0:
            recent_count = batch_size
            reservoir_count = 0
        recent = self._older_cursor + self._rng.draw(
            high=recent_size,
            size=recent_count,
        )
        older_offsets = self._rng.draw(high=reservoir_size, size=reservoir_count)
        older = self._reservoir[older_offsets]
        indices = np.concatenate((recent.astype(np.int32), older))
        sources = np.concatenate(
            (
                np.zeros(recent_count, dtype=np.uint8),
                np.ones(reservoir_count, dtype=np.uint8),
            )
        )
        return self.store.take(indices, sources=sources)

    def state_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "recent_window_seconds": self.recent_window_seconds,
            "reservoir_capacity": self.reservoir_capacity,
            "older_cursor": self._older_cursor,
            "reservoir": torch.from_numpy(self._reservoir.copy()),
            "reservoir_priorities": torch.from_numpy(self._reservoir_priorities.copy()),
            "rng_counter": self._rng.counter,
            "last_time": self._last_time,
            "store": self.store.rebuild_token(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (
            state.get("seed") != self.seed
            or state.get("recent_window_seconds") != self.recent_window_seconds
            or state.get("reservoir_capacity") != self.reservoir_capacity
            or state.get("store") != self.store.rebuild_token()
        ):
            raise ConsistencyError("Packed sampler checkpoint identity changed")
        older_cursor = state.get("older_cursor")
        rng_counter = state.get("rng_counter")
        last_time = state.get("last_time")
        reservoir = state.get("reservoir")
        priorities = state.get("reservoir_priorities")
        if (
            isinstance(older_cursor, bool)
            or not isinstance(older_cursor, int)
            or not 0 <= older_cursor <= len(self.store)
            or isinstance(rng_counter, bool)
            or not isinstance(rng_counter, int)
            or rng_counter < 0
            or (last_time is not None and not isinstance(last_time, (int, float)))
            or not isinstance(reservoir, torch.Tensor)
            or not isinstance(priorities, torch.Tensor)
        ):
            raise ConsistencyError("Packed sampler checkpoint state is malformed")
        parsed_reservoir = np.asarray(reservoir.cpu().numpy(), dtype=np.int32)
        parsed_priorities = np.asarray(priorities.cpu().numpy(), dtype=np.uint64)
        all_older = np.arange(older_cursor, dtype=np.int32)
        expected_reservoir, expected_priorities = self._bounded_reservoir(
            all_older,
            self._priorities(all_older),
        )
        if not np.array_equal(parsed_reservoir, expected_reservoir) or not np.array_equal(
            parsed_priorities, expected_priorities
        ):
            raise ConsistencyError("Packed sampler checkpoint reservoir is invalid")
        self._older_cursor = older_cursor
        self._reservoir = parsed_reservoir
        self._reservoir_priorities = parsed_priorities
        self._rng.counter = rng_counter
        self._last_time = None if last_time is None else float(last_time)
