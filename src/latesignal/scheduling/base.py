"""Common state and safety checks for windowed credit schedulers."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Protocol

from latesignal.errors import ConsistencyError
from latesignal.scheduling.credit import CreditWindow, SpendDecision
from latesignal.scheduling.monitoring import CalibrationEvidence


class CreditScheduler(Protocol):
    name: str
    windows: tuple[CreditWindow, ...]
    decisions: list[SpendDecision]

    def decide(
        self,
        simulator_time: int,
        evidence: CalibrationEvidence | None,
    ) -> SpendDecision: ...

    def assert_complete(self) -> None: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class WindowedScheduler:
    name = "windowed"

    def __init__(self, windows: tuple[CreditWindow, ...], *, day_seconds: int = 86_400) -> None:
        if not windows:
            raise ValueError("At least one credit window is required")
        if day_seconds <= 0:
            raise ValueError("day_seconds must be positive")
        self.windows = windows
        self.day_seconds = day_seconds
        self.decisions: list[SpendDecision] = []
        self._spent_windows: set[int] = set()
        self._last_decision_time: int | None = None

    @property
    def spent_count(self) -> int:
        return len(self._spent_windows)

    def _window_at(self, simulator_time: int) -> CreditWindow:
        first_boundary = self.windows[0].start_time
        if (simulator_time - first_boundary) % self.day_seconds:
            raise ConsistencyError("Scheduler decisions must occur on daily boundaries")
        if self._last_decision_time is not None and simulator_time <= self._last_decision_time:
            raise ConsistencyError("Scheduler decision time must increase strictly")
        for window in self.windows:
            if window.start_time <= simulator_time < window.end_time:
                self._last_decision_time = simulator_time
                return window
        raise ConsistencyError("Scheduler decision lies outside the adaptive horizon")

    def _window_for_time(self, simulator_time: int) -> CreditWindow | None:
        return next(
            (
                window
                for window in self.windows
                if window.start_time <= simulator_time < window.end_time
            ),
            None,
        )

    def _record(
        self,
        window: CreditWindow,
        simulator_time: int,
        *,
        spend: bool,
        reason: str,
        score: float | None = None,
        contributing_bin: int | None = None,
    ) -> SpendDecision:
        if spend:
            if window.window_id in self._spent_windows:
                raise ConsistencyError(f"Credit spent twice in window {window.window_id}")
            self._spent_windows.add(window.window_id)
        decision = SpendDecision(
            window_id=window.window_id,
            decision_time=simulator_time,
            spend=spend,
            reason=reason,
            score=score,
            contributing_bin=contributing_bin,
        )
        self.decisions.append(decision)
        return decision

    def assert_complete(self) -> None:
        expected = {window.window_id for window in self.windows}
        if self._spent_windows != expected:
            raise ConsistencyError(
                "Scheduler did not spend exactly one credit per window",
                details={
                    "missing": sorted(expected - self._spent_windows),
                    "unexpected": sorted(self._spent_windows - expected),
                },
            )

    def state_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "day_seconds": self.day_seconds,
            "windows": [asdict(window) for window in self.windows],
            "spent_windows": sorted(self._spent_windows),
            "last_decision_time": self._last_decision_time,
            "decisions": [decision.as_dict() for decision in self.decisions],
        }

    def _load_common_state(
        self,
        state: dict[str, Any],
        *,
        extra_keys: set[str],
    ) -> list[SpendDecision]:
        base_keys = {
            "name",
            "day_seconds",
            "windows",
            "spent_windows",
            "last_decision_time",
            "decisions",
        }
        if (
            set(state) != base_keys | extra_keys
            or state.get("name") != self.name
            or state.get("day_seconds") != self.day_seconds
            or state.get("windows") != [asdict(window) for window in self.windows]
        ):
            raise ConsistencyError("Scheduler checkpoint identity changed")
        spent = state.get("spent_windows")
        last_time = state.get("last_decision_time")
        raw_decisions = state.get("decisions")
        if (
            not isinstance(spent, list)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in spent)
            or spent != sorted(set(spent))
            or (
                last_time is not None
                and (isinstance(last_time, bool) or not isinstance(last_time, int))
            )
            or not isinstance(raw_decisions, list)
            or not all(isinstance(value, dict) for value in raw_decisions)
        ):
            raise ConsistencyError("Scheduler checkpoint state is malformed")
        decisions: list[SpendDecision] = []
        expected_keys = {
            "window_id",
            "decision_time",
            "spend",
            "reason",
            "score",
            "contributing_bin",
        }
        for raw in raw_decisions:
            window_id = raw.get("window_id")
            decision_time = raw.get("decision_time")
            spend = raw.get("spend")
            reason = raw.get("reason")
            score = raw.get("score")
            contributing_bin = raw.get("contributing_bin")
            if (
                set(raw) != expected_keys
                or isinstance(window_id, bool)
                or not isinstance(window_id, int)
                or isinstance(decision_time, bool)
                or not isinstance(decision_time, int)
                or not isinstance(spend, bool)
                or not isinstance(reason, str)
                or (
                    score is not None
                    and (
                        isinstance(score, bool)
                        or not isinstance(score, (int, float))
                        or not math.isfinite(float(score))
                        or float(score) < 0.0
                    )
                )
                or (
                    contributing_bin is not None
                    and (
                        isinstance(contributing_bin, bool)
                        or not isinstance(contributing_bin, int)
                        or not 0 <= contributing_bin < 10
                    )
                )
            ):
                raise ConsistencyError("Scheduler checkpoint decision is malformed")
            window = self._window_for_time(decision_time)
            if window is None or window.window_id != window_id:
                raise ConsistencyError("Scheduler checkpoint decision uses the wrong window")
            decisions.append(
                SpendDecision(
                    window_id=window_id,
                    decision_time=decision_time,
                    spend=spend,
                    reason=reason,
                    score=None if score is None else float(score),
                    contributing_bin=contributing_bin,
                )
            )
        if decisions:
            expected_times = list(
                range(
                    self.windows[0].start_time,
                    decisions[-1].decision_time + self.day_seconds,
                    self.day_seconds,
                )
            )
            if [value.decision_time for value in decisions] != expected_times:
                raise ConsistencyError(
                    "Scheduler checkpoint decisions are not daily and contiguous"
                )
            if last_time != decisions[-1].decision_time:
                raise ConsistencyError("Scheduler checkpoint last decision is inconsistent")
        elif last_time is not None:
            raise ConsistencyError("Scheduler checkpoint has time without decisions")
        spent_from_decisions = [value.window_id for value in decisions if value.spend]
        if len(spent_from_decisions) != len(set(spent_from_decisions)) or spent != sorted(
            spent_from_decisions
        ):
            raise ConsistencyError("Scheduler checkpoint spent windows are inconsistent")
        self.decisions = decisions
        self._spent_windows = set(spent)
        self._last_decision_time = last_time
        return decisions

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._load_common_state(state, extra_keys=set())
