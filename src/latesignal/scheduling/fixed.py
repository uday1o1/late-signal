"""Fixed daily scheduler for the synthetic vertical slice."""

from __future__ import annotations

from latesignal.errors import ConsistencyError
from latesignal.scheduling.base import WindowedScheduler
from latesignal.scheduling.credit import CreditWindow, SpendDecision
from latesignal.scheduling.monitoring import CalibrationEvidence


class FixedWindowScheduler(WindowedScheduler):
    """Spend at the authored early, midpoint, or deadline boundary."""

    def __init__(
        self,
        windows: tuple[CreditWindow, ...],
        *,
        policy: str,
        day_seconds: int = 86_400,
    ) -> None:
        super().__init__(windows, day_seconds=day_seconds)
        if policy not in {"early", "midpoint", "deadline"}:
            raise ValueError("Unknown fixed credit policy")
        self.policy = policy
        self.name = f"fixed_{policy}"
        self.monitoring_log: list[dict[str, object]] = []

    def decide(
        self,
        simulator_time: int,
        evidence: CalibrationEvidence | None,
    ) -> SpendDecision:
        window = self._window_at(simulator_time)
        if evidence is None:
            raise ValueError("Fixed schedulers require common daily monitoring evidence")
        self.monitoring_log.append(evidence.as_dict())
        if window.window_id in self._spent_windows:
            return self._record(window, simulator_time, spend=False, reason="already_spent")
        target_time = {
            "early": window.early_time,
            "midpoint": window.midpoint_time,
            "deadline": window.deadline_time,
        }[self.policy]
        if simulator_time == target_time:
            return self._record(
                window,
                simulator_time,
                spend=True,
                reason=f"fixed_{self.policy}",
            )
        return self._record(window, simulator_time, spend=False, reason="waiting")

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state.update({"policy": self.policy, "monitoring_log": self.monitoring_log})
        return state


class FixedDailyScheduler:
    """Approve one credit at every aligned daily boundary with a legal pool."""

    def __init__(self, origin: int, interval_seconds: int) -> None:
        self.origin = origin
        self.interval_seconds = interval_seconds
        self.spent_times: list[int] = []

    def is_decision_boundary(self, simulator_time: int) -> bool:
        return (
            simulator_time > self.origin
            and (simulator_time - self.origin) % self.interval_seconds == 0
        )

    def approve(self, simulator_time: int, legal_pool_size: int) -> bool:
        if not self.is_decision_boundary(simulator_time) or legal_pool_size == 0:
            return False
        if simulator_time in self.spent_times:
            raise ConsistencyError(f"Credit spent twice at {simulator_time}")
        self.spent_times.append(simulator_time)
        return True

    def state_dict(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "interval_seconds": self.interval_seconds,
            "spent_times": list(self.spent_times),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        origin_value = state.get("origin")
        interval_value = state.get("interval_seconds")
        spent = state.get("spent_times")
        if (
            isinstance(origin_value, bool)
            or not isinstance(origin_value, int)
            or isinstance(interval_value, bool)
            or not isinstance(interval_value, int)
        ):
            raise ConsistencyError("Scheduler checkpoint state is malformed")
        origin = origin_value
        interval = interval_value
        if (
            origin != self.origin
            or interval != self.interval_seconds
            or not isinstance(spent, list)
        ):
            raise ConsistencyError("Scheduler checkpoint does not match configuration")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in spent):
            raise ConsistencyError("Scheduler spent times are malformed")
        self.spent_times = list(spent)
