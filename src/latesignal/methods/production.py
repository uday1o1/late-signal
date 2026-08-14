"""Vectorized delayed-label state machines for production event replay."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray

from latesignal.data.manifests import canonical_json_bytes
from latesignal.errors import ConsistencyError
from latesignal.simulator.production_oracle import SECONDS_PER_DAY, TruthEventBatch
from latesignal.training.packed import (
    PackedRecordBatch,
    PackedRecordStore,
    RecordKind,
    packed_record_batch,
)

ProductionMethodName = Literal[
    "complete_wait",
    "immediate_fake_negative",
    "fixed_wait",
    "dfm",
    "fnw",
    "es_dfm",
]

_UNRESOLVED = np.uint8(0)
_POSITIVE = np.uint8(1)
_NEGATIVE = np.uint8(2)


@dataclass(frozen=True, slots=True)
class MethodBoundaryResult:
    main_records: int
    q_tn_records: int
    q_dp_records: int


@dataclass(frozen=True, slots=True)
class DFMObservationBatch:
    targets: NDArray[np.float32]
    time_days: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class _PendingRecords:
    refs: list[NDArray[np.int32]]
    available: list[NDArray[np.float64]]
    targets: list[NDArray[np.float32]]
    kinds: list[NDArray[np.uint8]]

    @classmethod
    def empty(cls) -> _PendingRecords:
        return cls([], [], [], [])

    def add(
        self,
        refs: NDArray[np.int32],
        available: NDArray[np.float64],
        targets: NDArray[np.float32],
        kind: RecordKind,
    ) -> None:
        if refs.size == 0:
            return
        self.refs.append(refs)
        self.available.append(available)
        self.targets.append(targets)
        self.kinds.append(np.full(refs.size, kind, dtype=np.uint8))

    def batch(self) -> PackedRecordBatch | None:
        if not self.refs:
            return None
        refs = np.concatenate(self.refs)
        available = np.concatenate(self.available)
        targets = np.concatenate(self.targets)
        kinds = np.concatenate(self.kinds)
        order = np.lexsort((refs, kinds, available))
        return packed_record_batch(
            feature_refs=refs[order],
            available_at=available[order],
            targets=targets[order],
            kinds=kinds[order],
        )


class PackedDelayedMethod:
    """Emit legal records without per-click Python state or future truth access."""

    def __init__(
        self,
        name: ProductionMethodName,
        *,
        click_times: NDArray[np.float64],
        monitoring_mask: NDArray[np.bool_],
        main_store: PackedRecordStore,
        wait_days: int | None = None,
        attribution_days: int = 30,
        q_tn_store: PackedRecordStore | None = None,
        q_dp_store: PackedRecordStore | None = None,
    ) -> None:
        if click_times.ndim != 1 or click_times.size == 0 or np.any(np.diff(click_times) < 0.0):
            raise ValueError("Production method click times must be chronological")
        if monitoring_mask.shape != click_times.shape:
            raise ValueError("Production method monitoring mask shape is invalid")
        if main_store.feature_count != click_times.size:
            raise ValueError("Production method record store has a different feature count")
        if attribution_days != 30:
            raise ValueError("Production method attribution window must remain 30 days")
        if name in {"fixed_wait", "es_dfm"}:
            if wait_days not in {1, 3, 7, 14}:
                raise ValueError("Fixed-wait production methods require a locked wait")
        elif wait_days is not None:
            raise ValueError("Only fixed-wait production methods accept wait days")
        if name == "es_dfm":
            if q_tn_store is None or q_dp_store is None:
                raise ValueError("ES-DFM requires both auxiliary record stores")
            if (
                q_tn_store.feature_count != click_times.size
                or q_dp_store.feature_count != click_times.size
            ):
                raise ValueError("ES-DFM auxiliary stores have a different feature count")
        elif q_tn_store is not None or q_dp_store is not None:
            raise ValueError("Only ES-DFM accepts auxiliary record stores")
        self.name = name
        self.click_times = np.array(click_times, dtype=np.float64, copy=True)
        self.monitoring_mask = np.array(monitoring_mask, dtype=np.bool_, copy=True)
        self.main_store = main_store
        self.q_tn_store = q_tn_store
        self.q_dp_store = q_dp_store
        self.wait_seconds = None if wait_days is None else wait_days * SECONDS_PER_DAY
        self.attribution_seconds = attribution_days * SECONDS_PER_DAY
        self.outcome_state = np.zeros(click_times.size, dtype=np.uint8)
        self.positive_delay_days = np.full(click_times.size, np.nan, dtype=np.float32)
        self.click_cursor = 0
        self.due_cursor = 0
        self.maturity_cursor = 0
        self.last_time: float | None = None
        self.monitoring_sha256 = hashlib.sha256(
            np.packbits(self.monitoring_mask, bitorder="little").tobytes()
        ).hexdigest()
        self.click_times_sha256 = hashlib.sha256(self.click_times.tobytes()).hexdigest()
        self.config_sha256 = hashlib.sha256(
            canonical_json_bytes(
                {
                    "name": name,
                    "wait_seconds": self.wait_seconds,
                    "attribution_seconds": self.attribution_seconds,
                    "monitoring_sha256": self.monitoring_sha256,
                    "click_times_sha256": self.click_times_sha256,
                }
            )
        ).hexdigest()

    def _legal(self, refs: NDArray[np.int32]) -> NDArray[np.bool_]:
        return ~self.monitoring_mask[refs]

    def _register_clicks(
        self,
        refs: NDArray[np.int32],
        boundary: float,
        pending: _PendingRecords,
    ) -> None:
        expected = np.arange(self.click_cursor, self.click_cursor + refs.size, dtype=np.int32)
        if (
            refs.ndim != 1
            or not np.array_equal(refs, expected)
            or np.any(self.click_times[refs] > boundary)
        ):
            raise ConsistencyError("Production method click cursor is not chronological")
        self.click_cursor += refs.size
        legal = self._legal(refs)
        legal_refs = refs[legal]
        if self.name in {"immediate_fake_negative", "fnw"}:
            pending.add(
                legal_refs,
                self.click_times[legal_refs],
                np.zeros(legal_refs.size, dtype=np.float32),
                RecordKind.PROVISIONAL,
            )
        elif self.name == "dfm":
            pending.add(
                legal_refs,
                self.click_times[legal_refs],
                np.zeros(legal_refs.size, dtype=np.float32),
                RecordKind.DFM_CLICK,
            )

    def _register_truth(
        self,
        truth: TruthEventBatch,
        boundary: float,
        pending: _PendingRecords,
    ) -> None:
        refs = truth.feature_refs
        if (
            refs.ndim != 1
            or truth.available_at.shape != refs.shape
            or truth.labels.shape != refs.shape
            or np.any(refs < 0)
            or np.any(refs >= self.click_cursor)
            or np.unique(refs).size != refs.size
            or np.any(self.outcome_state[refs] != _UNRESOLVED)
            or not np.isfinite(truth.available_at).all()
            or np.any(truth.available_at > boundary)
        ):
            raise ConsistencyError("Production truth arrived before click or more than once")
        positives = truth.labels == 1
        negatives = truth.labels == 0
        maturity_times = self.click_times[refs] + self.attribution_seconds
        if (
            not np.all(positives | negatives)
            or np.any(np.diff(truth.available_at) < 0.0)
            or np.any(truth.available_at < self.click_times[refs])
            or np.any(truth.available_at[positives] > maturity_times[positives])
            or not np.allclose(
                truth.available_at[negatives],
                maturity_times[negatives],
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise ConsistencyError("Production truth violates legal availability")
        positive_refs = refs[positives]
        negative_refs = refs[negatives]
        self.outcome_state[positive_refs] = _POSITIVE
        self.outcome_state[negative_refs] = _NEGATIVE
        self.positive_delay_days[positive_refs] = (
            (truth.available_at[positives] - self.click_times[positive_refs]) / SECONDS_PER_DAY
        ).astype(np.float32)
        legal = self._legal(refs)
        legal_positive = positives & legal
        if self.name in {"immediate_fake_negative", "fnw"}:
            pending.add(
                refs[legal_positive],
                truth.available_at[legal_positive],
                np.ones(np.count_nonzero(legal_positive), dtype=np.float32),
                RecordKind.CORRECTION,
            )
        elif self.name in {"fixed_wait", "es_dfm"}:
            assert self.wait_seconds is not None
            due = self.click_times[refs] + self.wait_seconds
            early = legal_positive & (truth.available_at <= due)
            late = legal_positive & ~early
            pending.add(
                refs[early],
                truth.available_at[early],
                np.ones(np.count_nonzero(early), dtype=np.float32),
                RecordKind.EARLY_POSITIVE,
            )
            pending.add(
                refs[late],
                truth.available_at[late],
                np.ones(np.count_nonzero(late), dtype=np.float32),
                RecordKind.CORRECTION,
            )

    def _emit_due(self, boundary: float, pending: _PendingRecords) -> None:
        if self.wait_seconds is None:
            return
        end = int(
            np.searchsorted(
                self.click_times[: self.click_cursor],
                boundary - self.wait_seconds,
                side="right",
            )
        )
        refs = np.arange(self.due_cursor, end, dtype=np.int32)
        if refs.size:
            positive_after_due = (self.outcome_state[refs] != _POSITIVE) | (
                self.positive_delay_days[refs] * SECONDS_PER_DAY > self.wait_seconds
            )
            legal = self._legal(refs) & positive_after_due
            selected = refs[legal]
            pending.add(
                selected,
                self.click_times[selected] + self.wait_seconds,
                np.zeros(selected.size, dtype=np.float32),
                RecordKind.PROVISIONAL,
            )
        self.due_cursor = end

    def _emit_mature(
        self,
        boundary: float,
        pending: _PendingRecords,
    ) -> tuple[int, int]:
        if self.name not in {"complete_wait", "es_dfm"}:
            return 0, 0
        end = int(
            np.searchsorted(
                self.click_times[: self.click_cursor],
                boundary - self.attribution_seconds,
                side="right",
            )
        )
        refs = np.arange(self.maturity_cursor, end, dtype=np.int32)
        if refs.size == 0:
            self.maturity_cursor = end
            return 0, 0
        if np.any(self.outcome_state[refs] == _UNRESOLVED):
            raise ConsistencyError("Method maturity arrived before legal final truth")
        legal_refs = refs[self._legal(refs)]
        times = self.click_times[legal_refs] + self.attribution_seconds
        if self.name == "complete_wait":
            pending.add(
                legal_refs,
                times,
                (self.outcome_state[legal_refs] == _POSITIVE).astype(np.float32),
                RecordKind.FINAL,
            )
            self.maturity_cursor = end
            return 0, 0
        assert self.q_tn_store is not None and self.q_dp_store is not None
        assert self.wait_seconds is not None
        delayed = (self.outcome_state[legal_refs] == _POSITIVE) & (
            self.positive_delay_days[legal_refs] * SECONDS_PER_DAY > self.wait_seconds
        )
        early = (self.outcome_state[legal_refs] == _POSITIVE) & ~delayed
        q_dp = packed_record_batch(
            feature_refs=legal_refs,
            available_at=times,
            targets=delayed.astype(np.float32),
            kinds=np.full(legal_refs.size, RecordKind.FINAL, dtype=np.uint8),
        )
        q_tn_refs = legal_refs[~early]
        q_tn = packed_record_batch(
            feature_refs=q_tn_refs,
            available_at=times[~early],
            targets=(~delayed[~early]).astype(np.float32),
            kinds=np.full(q_tn_refs.size, RecordKind.FINAL, dtype=np.uint8),
        )
        if len(q_dp):
            self.q_dp_store.append(q_dp, simulator_time=boundary)
        if len(q_tn):
            self.q_tn_store.append(q_tn, simulator_time=boundary)
        self.maturity_cursor = end
        return len(q_tn), len(q_dp)

    def process_boundary(
        self,
        *,
        boundary: float,
        click_refs: NDArray[np.int32],
        truth: TruthEventBatch,
    ) -> MethodBoundaryResult:
        if not np.isfinite(boundary) or (self.last_time is not None and boundary < self.last_time):
            raise ConsistencyError("Production method boundary cannot move backward")
        pending = _PendingRecords.empty()
        self._register_clicks(click_refs, boundary, pending)
        self._register_truth(truth, boundary, pending)
        self._emit_due(boundary, pending)
        q_tn_records, q_dp_records = self._emit_mature(boundary, pending)
        batch = pending.batch()
        if batch is not None:
            self.main_store.append(batch, simulator_time=boundary)
        self.last_time = boundary
        return MethodBoundaryResult(
            main_records=0 if batch is None else len(batch),
            q_tn_records=q_tn_records,
            q_dp_records=q_dp_records,
        )

    def dfm_observations(
        self,
        refs: NDArray[np.int32],
        *,
        simulator_time: float,
    ) -> DFMObservationBatch:
        if self.name != "dfm":
            raise ConsistencyError("Only DFM can materialize current censored observations")
        parsed = np.asarray(refs, dtype=np.int32)
        if (
            parsed.ndim != 1
            or np.any(parsed < 0)
            or np.any(parsed >= self.click_cursor)
            or np.any(self.click_times[parsed] > simulator_time)
        ):
            raise ConsistencyError("DFM sample contains a future or unknown click")
        positive = self.outcome_state[parsed] == _POSITIVE
        elapsed = np.minimum(
            (simulator_time - self.click_times[parsed]) / SECONDS_PER_DAY,
            self.attribution_seconds / SECONDS_PER_DAY,
        ).astype(np.float32)
        times = np.where(positive, self.positive_delay_days[parsed], elapsed).astype(np.float32)
        return DFMObservationBatch(
            targets=positive.astype(np.float32),
            time_days=times,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "config_sha256": self.config_sha256,
            "click_cursor": self.click_cursor,
            "due_cursor": self.due_cursor,
            "maturity_cursor": self.maturity_cursor,
            "last_time": self.last_time,
            "outcome_state": torch.from_numpy(self.outcome_state.copy()),
            "positive_delay_days": torch.from_numpy(self.positive_delay_days.copy()),
            "main_store": self.main_store.rebuild_token(),
            "q_tn_store": None if self.q_tn_store is None else self.q_tn_store.rebuild_token(),
            "q_dp_store": None if self.q_dp_store is None else self.q_dp_store.rebuild_token(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        cursors = tuple(state.get(key) for key in ("click_cursor", "due_cursor", "maturity_cursor"))
        last_time = state.get("last_time")
        outcome = state.get("outcome_state")
        delay = state.get("positive_delay_days")
        if (
            state.get("version") != 1
            or state.get("config_sha256") != self.config_sha256
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in cursors)
            or not isinstance(outcome, torch.Tensor)
            or not isinstance(delay, torch.Tensor)
            or (
                last_time is not None
                and (not isinstance(last_time, (int, float)) or not np.isfinite(float(last_time)))
            )
            or state.get("main_store") != self.main_store.rebuild_token()
            or state.get("q_tn_store")
            != (None if self.q_tn_store is None else self.q_tn_store.rebuild_token())
            or state.get("q_dp_store")
            != (None if self.q_dp_store is None else self.q_dp_store.rebuild_token())
        ):
            raise ConsistencyError("Production method checkpoint identity is malformed")
        click_cursor, due_cursor, maturity_cursor = cursors
        assert isinstance(click_cursor, int)
        assert isinstance(due_cursor, int)
        assert isinstance(maturity_cursor, int)
        parsed_outcome = np.asarray(outcome.cpu().numpy(), dtype=np.uint8)
        parsed_delay = np.asarray(delay.cpu().numpy(), dtype=np.float32)
        if (
            parsed_outcome.shape != self.outcome_state.shape
            or parsed_delay.shape != self.positive_delay_days.shape
            or not 0 <= due_cursor <= click_cursor <= self.click_times.size
            or not 0 <= maturity_cursor <= click_cursor
            or np.any(parsed_outcome > _NEGATIVE)
            or np.any(np.isfinite(parsed_delay) != (parsed_outcome == _POSITIVE))
            or np.any(parsed_delay[parsed_outcome == _POSITIVE] < 0.0)
            or np.any(parsed_delay[parsed_outcome == _POSITIVE] > 30.0)
            or np.any(parsed_outcome[click_cursor:] != _UNRESOLVED)
        ):
            raise ConsistencyError("Production method checkpoint arrays are inconsistent")
        if last_time is not None:
            expected_due = (
                0
                if self.wait_seconds is None
                else int(
                    np.searchsorted(
                        self.click_times[:click_cursor],
                        float(last_time) - self.wait_seconds,
                        side="right",
                    )
                )
            )
            expected_maturity = (
                int(
                    np.searchsorted(
                        self.click_times[:click_cursor],
                        float(last_time) - self.attribution_seconds,
                        side="right",
                    )
                )
                if self.name in {"complete_wait", "es_dfm"}
                else 0
            )
            if due_cursor != expected_due or maturity_cursor != expected_maturity:
                raise ConsistencyError("Production method checkpoint cursors are inconsistent")
        elif any(value != 0 for value in (click_cursor, due_cursor, maturity_cursor)):
            raise ConsistencyError("Production method checkpoint has cursors without time")
        self.click_cursor = click_cursor
        self.due_cursor = due_cursor
        self.maturity_cursor = maturity_cursor
        self.last_time = None if last_time is None else float(last_time)
        self.outcome_state = parsed_outcome
        self.positive_delay_days = parsed_delay
