"""Restricted real-data truth loading and event-time reveal cursors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from numpy.typing import NDArray

from latesignal.data.prepared import PreparedInventory, verify_prepared_inventory
from latesignal.errors import ConsistencyError

SECONDS_PER_DAY = 86_400.0
ATTRIBUTION_DAYS = 30.0


class TruthFeatureIndex(Protocol):
    click_ids: NDArray[np.void]
    click_times: NDArray[np.float64]
    click_days: NDArray[np.int16]

    @property
    def prepared_manifest_sha256(self) -> str: ...

    def references_for_ids(self, click_ids: list[bytes]) -> NDArray[np.int32]: ...


@dataclass(frozen=True, slots=True)
class TruthEventBatch:
    feature_refs: NDArray[np.int32]
    available_at: NDArray[np.float64]
    labels: NDArray[np.int8]

    def __len__(self) -> int:
        return int(self.feature_refs.size)


@dataclass(frozen=True, slots=True)
class ProductionTruthStore:
    prepared_manifest_sha256: str
    final_labels: NDArray[np.int8]
    available_at: NDArray[np.float64]
    conversion_delay_days: NDArray[np.float32]
    event_feature_refs: NDArray[np.int32]
    event_available_at: NDArray[np.float64]
    event_labels: NDArray[np.int8]

    @property
    def rows(self) -> int:
        return int(self.final_labels.size)

    def cursor(
        self,
        click_days: NDArray[np.int16],
        *,
        first_click_day: int,
        last_click_day: int,
    ) -> ProductionTruthCursor:
        if click_days.shape != self.final_labels.shape:
            raise ConsistencyError("Truth cursor feature-day identity changed")
        return ProductionTruthCursor(
            self,
            click_days,
            first_click_day=first_click_day,
            last_click_day=last_click_day,
        )


def _truth_files_by_day(inventory: PreparedInventory) -> dict[int, list[tuple[int, Path]]]:
    result: dict[int, list[tuple[int, Path]]] = {}
    kinds: tuple[Literal["reveal", "maturity"], ...] = ("reveal", "maturity")
    for kind_index, kind in enumerate(kinds):
        for day in range(121):
            for path in inventory.truth_files(kind, first_day=day, last_day=day):
                result.setdefault(day, []).append((kind_index, path))
    return result


def _read_truth_file(
    path: Path,
    *,
    expected_label: int,
    features: TruthFeatureIndex,
) -> tuple[NDArray[np.int32], NDArray[np.float64], NDArray[np.int8]]:
    try:
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        required = {
            "click_id",
            "final_label",
            "click_time_seconds",
            "available_at_seconds",
        }
        if (
            set(schema.names) != required
            or schema.metadata is None
            or schema.metadata.get(b"latesignal_store") != b"eventual_truth"
        ):
            raise ConsistencyError("Truth part violates the restricted oracle schema")
        table = parquet.read(columns=sorted(required))
    except (OSError, pa.ArrowException) as error:
        raise ConsistencyError("Truth part could not be read") from error
    raw_ids = table["click_id"].to_pylist()
    try:
        click_ids = [bytes.fromhex(value) for value in raw_ids]
    except (TypeError, ValueError) as error:
        raise ConsistencyError("Truth part contains an invalid click ID") from error
    if any(len(value) != 32 for value in click_ids):
        raise ConsistencyError("Truth part contains an invalid click ID")
    refs = features.references_for_ids(click_ids)
    labels = np.asarray(
        table["final_label"].to_numpy(zero_copy_only=False),
        dtype=np.int8,
    )
    click_times = np.asarray(
        table["click_time_seconds"].to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    available = np.asarray(
        table["available_at_seconds"].to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    delays = (available - click_times) / SECONDS_PER_DAY
    if (
        refs.size != labels.size
        or np.any(labels != expected_label)
        or not np.array_equal(click_times, features.click_times[refs])
        or not np.isfinite(available).all()
        or np.any(delays < 0.0)
        or np.any(delays > ATTRIBUTION_DAYS + 1e-9)
        or (expected_label == 0 and not np.allclose(delays, ATTRIBUTION_DAYS, atol=1e-9))
    ):
        raise ConsistencyError("Truth part violates label availability semantics")
    return refs, available, labels


def load_production_truth(
    prepared_manifest_path: Path,
    features: TruthFeatureIndex,
) -> ProductionTruthStore:
    """Verify and compact the private truth store without exposing it to learners."""

    inventory = verify_prepared_inventory(prepared_manifest_path)
    if inventory.manifest_sha256 != features.prepared_manifest_sha256:
        raise ConsistencyError("Truth and runtime features use different prepared identities")
    expected_rows = inventory.manifest.get("rows", {}).get("truth")
    if (
        isinstance(expected_rows, bool)
        or not isinstance(expected_rows, int)
        or expected_rows != features.click_ids.size
    ):
        raise ConsistencyError("Prepared truth count does not match runtime features")
    final_labels = np.full(expected_rows, -1, dtype=np.int8)
    available_at = np.full(expected_rows, np.nan, dtype=np.float64)
    delay_days = np.full(expected_rows, np.nan, dtype=np.float32)
    event_refs = np.empty(expected_rows, dtype=np.int32)
    event_available = np.empty(expected_rows, dtype=np.float64)
    event_labels = np.empty(expected_rows, dtype=np.int8)
    seen = np.zeros(expected_rows, dtype=np.bool_)
    cursor = 0
    for _, entries in sorted(_truth_files_by_day(inventory).items()):
        day_refs: list[NDArray[np.int32]] = []
        day_available: list[NDArray[np.float64]] = []
        day_labels: list[NDArray[np.int8]] = []
        for kind_index, path in sorted(entries, key=lambda item: (item[0], str(item[1]))):
            refs, times, labels = _read_truth_file(
                path,
                expected_label=1 if kind_index == 0 else 0,
                features=features,
            )
            if np.any(seen[refs]) or np.unique(refs).size != refs.size:
                raise ConsistencyError("Prepared truth contains a duplicate click ID")
            seen[refs] = True
            final_labels[refs] = labels
            available_at[refs] = times
            delay_days[refs] = np.where(
                labels == 1,
                (times - features.click_times[refs]) / SECONDS_PER_DAY,
                np.nan,
            )
            day_refs.append(refs)
            day_available.append(times)
            day_labels.append(labels)
        refs = np.concatenate(day_refs)
        times = np.concatenate(day_available)
        labels = np.concatenate(day_labels)
        order = np.lexsort((refs, times))
        count = refs.size
        end = cursor + count
        event_refs[cursor:end] = refs[order]
        event_available[cursor:end] = times[order]
        event_labels[cursor:end] = labels[order]
        cursor = end
    if cursor != expected_rows or not seen.all() or np.any(final_labels < 0):
        raise ConsistencyError("Prepared truth does not reconcile one-to-one with features")
    if np.any(np.diff(event_available) < 0.0):
        raise ConsistencyError("Prepared truth events are not globally chronological")
    return ProductionTruthStore(
        prepared_manifest_sha256=inventory.manifest_sha256,
        final_labels=final_labels,
        available_at=available_at,
        conversion_delay_days=delay_days,
        event_feature_refs=event_refs,
        event_available_at=event_available,
        event_labels=event_labels,
    )


class ProductionTruthCursor:
    """Reveal only due truth for an authored click-day training period."""

    def __init__(
        self,
        store: ProductionTruthStore,
        click_days: NDArray[np.int16],
        *,
        first_click_day: int,
        last_click_day: int,
    ) -> None:
        if first_click_day < 0 or last_click_day < first_click_day:
            raise ValueError("Truth cursor click-day period is invalid")
        self.store = store
        self.click_days = click_days
        self.first_click_day = first_click_day
        self.last_click_day = last_click_day
        self.event_cursor = 0
        self.last_time: float | None = None

    @property
    def drained(self) -> bool:
        return self.event_cursor == self.store.event_feature_refs.size

    def reveal_through(self, simulator_time: float) -> TruthEventBatch:
        if not np.isfinite(simulator_time):
            raise ConsistencyError("Truth cursor time must be finite")
        if self.last_time is not None and simulator_time < self.last_time:
            raise ConsistencyError("Truth cursor time cannot move backward")
        end = int(
            np.searchsorted(
                self.store.event_available_at,
                simulator_time,
                side="right",
            )
        )
        refs = self.store.event_feature_refs[self.event_cursor : end]
        mask = (self.click_days[refs] >= self.first_click_day) & (
            self.click_days[refs] <= self.last_click_day
        )
        result = TruthEventBatch(
            feature_refs=refs[mask],
            available_at=self.store.event_available_at[self.event_cursor : end][mask],
            labels=self.store.event_labels[self.event_cursor : end][mask],
        )
        self.event_cursor = end
        self.last_time = simulator_time
        return result

    def state_dict(self) -> dict[str, object]:
        return {
            "prepared_manifest_sha256": self.store.prepared_manifest_sha256,
            "first_click_day": self.first_click_day,
            "last_click_day": self.last_click_day,
            "event_cursor": self.event_cursor,
            "last_time": self.last_time,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        event_cursor = state.get("event_cursor")
        last_time = state.get("last_time")
        if (
            state.get("prepared_manifest_sha256") != self.store.prepared_manifest_sha256
            or state.get("first_click_day") != self.first_click_day
            or state.get("last_click_day") != self.last_click_day
            or isinstance(event_cursor, bool)
            or not isinstance(event_cursor, int)
            or not 0 <= event_cursor <= self.store.event_feature_refs.size
            or (last_time is not None and not isinstance(last_time, (int, float)))
        ):
            raise ConsistencyError("Truth cursor checkpoint state is malformed")
        if last_time is not None:
            expected_cursor = int(
                np.searchsorted(
                    self.store.event_available_at,
                    float(last_time),
                    side="right",
                )
            )
            if event_cursor != expected_cursor:
                raise ConsistencyError("Truth cursor checkpoint position is inconsistent")
        elif event_cursor != 0:
            raise ConsistencyError("Truth cursor checkpoint has a cursor without time")
        self.event_cursor = event_cursor
        self.last_time = None if last_time is None else float(last_time)
