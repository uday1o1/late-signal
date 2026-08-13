"""Auditable calibration-drift credit scheduler."""

from __future__ import annotations

from latesignal.scheduling.base import WindowedScheduler
from latesignal.scheduling.credit import CreditWindow, SpendDecision
from latesignal.scheduling.monitoring import CalibrationEvidence


class CalibrationDriftCreditScheduler(WindowedScheduler):
    name = "calibration_drift"

    def __init__(
        self,
        windows: tuple[CreditWindow, ...],
        *,
        threshold: float = 3.0,
        day_seconds: int = 86_400,
    ) -> None:
        super().__init__(windows, day_seconds=day_seconds)
        if threshold <= 0.0:
            raise ValueError("Calibration trigger threshold must be positive")
        self.threshold = threshold
        self.evidence_log: list[dict[str, object]] = []

    def decide(
        self,
        simulator_time: int,
        evidence: CalibrationEvidence | None,
    ) -> SpendDecision:
        window = self._window_at(simulator_time)
        if evidence is None:
            raise ValueError("Calibration scheduler requires daily evidence")
        self.evidence_log.append(evidence.as_dict())
        if window.window_id in self._spent_windows:
            return self._record(window, simulator_time, spend=False, reason="already_spent")
        if evidence.score is not None and evidence.score > self.threshold:
            return self._record(
                window,
                simulator_time,
                spend=True,
                reason="calibration_trigger",
                score=evidence.score,
                contributing_bin=evidence.contributing_bin,
            )
        if simulator_time == window.deadline_time:
            return self._record(
                window,
                simulator_time,
                spend=True,
                reason="forced_deadline",
                score=evidence.score,
                contributing_bin=evidence.contributing_bin,
            )
        return self._record(
            window,
            simulator_time,
            spend=False,
            reason="waiting",
            score=evidence.score,
            contributing_bin=evidence.contributing_bin,
        )

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state.update({"threshold": self.threshold, "evidence_log": self.evidence_log})
        return state
