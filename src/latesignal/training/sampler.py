"""Deterministic recent-plus-reservoir sampling."""

from __future__ import annotations

import hashlib
import heapq
import random
from dataclasses import dataclass
from typing import Any

from latesignal.contracts.records import TrainingRecord
from latesignal.errors import ConsistencyError


@dataclass(frozen=True, slots=True)
class SampledRecord:
    record: TrainingRecord
    source: str


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


class DeterministicSampler:
    """Sample half recent and half from a deterministic uniform-priority reservoir."""

    def __init__(
        self,
        *,
        seed: int,
        recent_window_seconds: int,
        reservoir_capacity: int,
        excluded_click_ids: frozenset[str] = frozenset(),
    ) -> None:
        if recent_window_seconds <= 0 or reservoir_capacity <= 0:
            raise ValueError("Sampler window and reservoir capacity must be positive")
        self.seed = seed
        self.recent_window_seconds = recent_window_seconds
        self.reservoir_capacity = reservoir_capacity
        self.excluded_click_ids = excluded_click_ids
        self._recent: list[TrainingRecord] = []
        self._reservoir: dict[str, TrainingRecord] = {}
        self._reservoir_heap: list[tuple[int, str]] = []
        self._reservoir_values: tuple[TrainingRecord, ...] | None = None
        self._ids: set[str] = set()
        self._rng = random.Random(seed)
        self._last_time: int | None = None

    def add(self, record: TrainingRecord, simulator_time: int) -> None:
        record.assert_available(simulator_time)
        if record.click_id in self.excluded_click_ids:
            raise ConsistencyError(f"Monitoring record cannot enter training: {record.click_id}")
        self._advance(simulator_time)
        if record.record_id in self._ids:
            raise ConsistencyError(f"Sampler record added twice: {record.record_id}")
        self._ids.add(record.record_id)
        cutoff = simulator_time - self.recent_window_seconds
        if record.available_at >= cutoff:
            self._recent.append(record)
        else:
            self._consider_reservoir(record)

    def _priority(self, record_id: str) -> int:
        return int.from_bytes(hashlib.sha256(f"{self.seed}:{record_id}".encode()).digest())

    def _consider_reservoir(self, record: TrainingRecord) -> None:
        priority = self._priority(record.record_id)
        if len(self._reservoir) < self.reservoir_capacity:
            self._reservoir[record.record_id] = record
            heapq.heappush(self._reservoir_heap, (-priority, record.record_id))
            self._reservoir_values = None
            return
        worst_negative, worst_id = self._reservoir_heap[0]
        if priority >= -worst_negative:
            self._ids.remove(record.record_id)
            return
        heapq.heapreplace(self._reservoir_heap, (-priority, record.record_id))
        del self._reservoir[worst_id]
        self._ids.remove(worst_id)
        self._reservoir[record.record_id] = record
        self._reservoir_values = None

    def _advance(self, simulator_time: int) -> None:
        if self._last_time is not None and simulator_time < self._last_time:
            raise ConsistencyError("Sampler time cannot move backward")
        if self._last_time == simulator_time:
            return
        self._last_time = simulator_time
        cutoff = simulator_time - self.recent_window_seconds
        retained: list[TrainingRecord] = []
        for record in self._recent:
            if record.available_at >= cutoff:
                retained.append(record)
            else:
                self._consider_reservoir(record)
        self._recent = retained

    def sample(self, *, simulator_time: int, batch_size: int) -> list[SampledRecord]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._advance(simulator_time)
        for record in self._recent:
            record.assert_available(simulator_time)
        for record in self._reservoir.values():
            record.assert_available(simulator_time)
        if self._reservoir_values is None:
            self._reservoir_values = tuple(self._reservoir.values())
        reservoir = self._reservoir_values
        if not self._recent and not reservoir:
            raise ConsistencyError("INSUFFICIENT_LEGAL_POOL")
        recent_count = batch_size // 2
        older_count = batch_size - recent_count
        if not self._recent:
            older_count = batch_size
            recent_count = 0
        if not reservoir:
            recent_count = batch_size
            older_count = 0
        sampled = [
            SampledRecord(self._rng.choice(self._recent), "recent") for _ in range(recent_count)
        ]
        sampled.extend(
            SampledRecord(self._rng.choice(reservoir), "reservoir") for _ in range(older_count)
        )
        self._rng.shuffle(sampled)
        return sampled

    def state_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "recent_window_seconds": self.recent_window_seconds,
            "reservoir_capacity": self.reservoir_capacity,
            "excluded_click_ids": sorted(self.excluded_click_ids),
            "recent_records": [record.as_dict() for record in self._recent],
            "reservoir_records": [record.as_dict() for record in self._reservoir.values()],
            "rng_state": self._rng.getstate(),
            "last_time": self._last_time,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if (
            state.get("seed") != self.seed
            or state.get("recent_window_seconds") != self.recent_window_seconds
            or state.get("reservoir_capacity") != self.reservoir_capacity
            or state.get("excluded_click_ids") != sorted(self.excluded_click_ids)
        ):
            raise ConsistencyError("Sampler checkpoint does not match configuration")
        recent_records = state.get("recent_records")
        reservoir_records = state.get("reservoir_records")
        last_time = state.get("last_time")
        if (
            not isinstance(recent_records, list)
            or not isinstance(reservoir_records, list)
            or (
                last_time is not None
                and (isinstance(last_time, bool) or not isinstance(last_time, int))
            )
        ):
            raise ConsistencyError("Sampler checkpoint records are malformed")
        self._recent = []
        self._reservoir = {}
        self._reservoir_heap = []
        self._reservoir_values = None
        self._ids = set()
        for value in recent_records:
            if not isinstance(value, dict):
                raise ConsistencyError("Sampler checkpoint record is malformed")
            record = TrainingRecord.from_dict(value)
            if record.record_id in self._ids:
                raise ConsistencyError("Sampler checkpoint contains duplicate record IDs")
            self._recent.append(record)
            self._ids.add(record.record_id)
        for value in reservoir_records:
            if not isinstance(value, dict):
                raise ConsistencyError("Sampler checkpoint record is malformed")
            record = TrainingRecord.from_dict(value)
            if record.record_id in self._ids:
                raise ConsistencyError("Sampler checkpoint contains duplicate record IDs")
            self._reservoir[record.record_id] = record
            heapq.heappush(
                self._reservoir_heap,
                (-self._priority(record.record_id), record.record_id),
            )
            self._ids.add(record.record_id)
        self._last_time = last_time
        try:
            self._rng.setstate(_tuple_tree(state.get("rng_state")))
        except (TypeError, ValueError) as error:
            raise ConsistencyError("Sampler checkpoint RNG state is malformed") from error
