"""Common state and safety checks for windowed credit schedulers."""

from __future__ import annotations

from typing import Protocol

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
        if simulator_time % self.day_seconds:
            raise ConsistencyError("Scheduler decisions must occur on daily boundaries")
        if self._last_decision_time is not None and simulator_time <= self._last_decision_time:
            raise ConsistencyError("Scheduler decision time must increase strictly")
        self._last_decision_time = simulator_time
        for window in self.windows:
            if window.start_time <= simulator_time < window.end_time:
                return window
        raise ConsistencyError("Scheduler decision lies outside the adaptive horizon")

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
            "spent_windows": sorted(self._spent_windows),
            "last_decision_time": self._last_decision_time,
            "decisions": [decision.as_dict() for decision in self.decisions],
        }
