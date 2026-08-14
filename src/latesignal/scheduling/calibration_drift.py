"""Auditable calibration-drift credit scheduler."""

from __future__ import annotations

import copy
import math
from typing import Any

from latesignal.errors import ConsistencyError
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
        state.update(
            {"threshold": self.threshold, "evidence_log": copy.deepcopy(self.evidence_log)}
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("threshold") != self.threshold:
            raise ConsistencyError("Calibration scheduler checkpoint threshold changed")
        evidence_log = state.get("evidence_log")
        decisions = self._load_common_state(
            state,
            extra_keys={"threshold", "evidence_log"},
        )
        if (
            not isinstance(evidence_log, list)
            or not all(isinstance(value, dict) for value in evidence_log)
            or len(evidence_log) != len(decisions)
        ):
            raise ConsistencyError("Calibration scheduler evidence checkpoint is malformed")
        spent: set[int] = set()
        origin = self.windows[0].start_time - 31 * self.day_seconds
        for decision, evidence in zip(decisions, evidence_log, strict=True):
            raw_score = evidence.get("score")
            raw_bin = evidence.get("contributing_bin")
            expected_day = (decision.decision_time - origin) // self.day_seconds
            if (
                evidence.get("decision_day") != expected_day
                or (
                    raw_score is not None
                    and (
                        isinstance(raw_score, bool)
                        or not isinstance(raw_score, (int, float))
                        or not math.isfinite(float(raw_score))
                        or float(raw_score) < 0.0
                    )
                )
                or (
                    raw_bin is not None
                    and (
                        isinstance(raw_bin, bool)
                        or not isinstance(raw_bin, int)
                        or not 0 <= raw_bin < 10
                    )
                )
            ):
                raise ConsistencyError("Calibration scheduler evidence entry is malformed")
            score = None if raw_score is None else float(raw_score)
            window = self.windows[decision.window_id]
            already_spent = decision.window_id in spent
            trigger = not already_spent and score is not None and score > self.threshold
            deadline = (
                not already_spent and not trigger and decision.decision_time == window.deadline_time
            )
            expected_spend = trigger or deadline
            expected_reason = (
                "already_spent"
                if already_spent
                else "calibration_trigger"
                if trigger
                else "forced_deadline"
                if deadline
                else "waiting"
            )
            expected_score = None if already_spent else score
            expected_bin = None if already_spent else raw_bin
            if (
                decision.spend != expected_spend
                or decision.reason != expected_reason
                or decision.score != expected_score
                or decision.contributing_bin != expected_bin
            ):
                raise ConsistencyError("Calibration scheduler checkpoint decision is inconsistent")
            if decision.spend:
                spent.add(decision.window_id)
        self.evidence_log = copy.deepcopy(evidence_log)
