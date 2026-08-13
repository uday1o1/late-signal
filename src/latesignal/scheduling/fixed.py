"""Fixed daily scheduler for the synthetic vertical slice."""

from __future__ import annotations

from latesignal.errors import ConsistencyError


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
