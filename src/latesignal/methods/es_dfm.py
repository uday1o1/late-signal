"""Constant-wait ES-DFM event and legal auxiliary-label state."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from latesignal.contracts.events import ClickEvent, NegativeMaturity, PositiveReveal
from latesignal.contracts.records import TrainingRecord
from latesignal.errors import ConsistencyError
from latesignal.methods.fixed_wait import FixedWaitMethod


@dataclass(slots=True)
class _AuxState:
    click_time: int
    feature: float
    positive_at: int | None = None
    negative_matured: bool = False
    auxiliary_emitted: bool = False


@dataclass(frozen=True, slots=True)
class AuxiliaryRecord:
    click_id: str
    available_at: int
    feature: float
    q_tn_target: float | None
    q_dp_target: float


class ESDFMMethod:
    name = "es_dfm"

    def __init__(self, *, wait_seconds: int, attribution_seconds: int) -> None:
        if wait_seconds <= 0 or attribution_seconds <= wait_seconds:
            raise ValueError("ES-DFM requires 0 < wait_seconds < attribution_seconds")
        self.wait_seconds = wait_seconds
        self.attribution_seconds = attribution_seconds
        self._main = FixedWaitMethod(wait_seconds)
        self._states: dict[str, _AuxState] = {}
        self._auxiliary: list[AuxiliaryRecord] = []

    def on_click(self, click: ClickEvent) -> list[TrainingRecord]:
        if click.click_id in self._states:
            raise ConsistencyError(f"Click processed twice: {click.click_id}")
        self._states[click.click_id] = _AuxState(click.click_time, click.feature)
        return self._retag(self._main.on_click(click))

    def _state(self, click_id: str) -> _AuxState:
        try:
            return self._states[click_id]
        except KeyError as error:
            raise ConsistencyError(f"Truth arrived before click: {click_id}") from error

    def on_positive_reveal(self, label: PositiveReveal) -> list[TrainingRecord]:
        state = self._state(label.click_id)
        if state.positive_at is not None or state.negative_matured:
            raise ConsistencyError(f"ES-DFM truth processed twice: {label.click_id}")
        state.positive_at = label.available_at
        return self._retag(self._main.on_positive_reveal(label))

    def on_negative_maturity(self, label: NegativeMaturity) -> list[TrainingRecord]:
        state = self._state(label.click_id)
        if state.positive_at is not None or state.negative_matured:
            raise ConsistencyError(f"ES-DFM truth processed twice: {label.click_id}")
        state.negative_matured = True
        records = self._main.on_negative_maturity(label)
        return self._retag(records + self.on_boundary(label.available_at))

    def on_boundary(self, simulator_time: int) -> list[TrainingRecord]:
        records = self._retag(self._main.on_boundary(simulator_time))
        for click_id, state in sorted(self._states.items()):
            maturity = state.click_time + self.attribution_seconds
            if state.auxiliary_emitted or maturity > simulator_time:
                continue
            if state.positive_at is None and not state.negative_matured:
                continue
            delayed_positive = (
                state.positive_at is not None
                and state.positive_at > state.click_time + self.wait_seconds
            )
            early_positive = (
                state.positive_at is not None
                and state.positive_at <= state.click_time + self.wait_seconds
            )
            state.auxiliary_emitted = True
            self._auxiliary.append(
                AuxiliaryRecord(
                    click_id=click_id,
                    available_at=maturity,
                    feature=state.feature,
                    q_tn_target=None if early_positive else float(not delayed_positive),
                    q_dp_target=float(delayed_positive),
                )
            )
        return records

    def auxiliary_records(self, simulator_time: int) -> tuple[AuxiliaryRecord, ...]:
        if any(record.available_at > simulator_time for record in self._auxiliary):
            raise ConsistencyError("ES-DFM auxiliary pool contains future truth")
        return tuple(record for record in self._auxiliary if record.available_at <= simulator_time)

    def _retag(self, records: list[TrainingRecord]) -> list[TrainingRecord]:
        return [
            TrainingRecord(
                record_id=record.record_id.replace("fixed_wait:", f"{self.name}:", 1),
                click_id=record.click_id,
                available_at=record.available_at,
                status=record.status,
                target=record.target,
                weight=record.weight,
                correction_group=record.correction_group,
                source_method=self.name,
                feature=record.feature,
            )
            for record in records
        ]

    def state_dict(self) -> dict[str, object]:
        return {
            "wait_seconds": self.wait_seconds,
            "attribution_seconds": self.attribution_seconds,
            "main": self._main.state_dict(),
            "states": {key: asdict(value) for key, value in sorted(self._states.items())},
            "auxiliary": [asdict(value) for value in self._auxiliary],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if (
            state.get("wait_seconds") != self.wait_seconds
            or state.get("attribution_seconds") != self.attribution_seconds
        ):
            raise ConsistencyError("ES-DFM checkpoint configuration does not match")
        main = state.get("main")
        states = state.get("states")
        auxiliary = state.get("auxiliary")
        if (
            not isinstance(main, dict)
            or not isinstance(states, dict)
            or not isinstance(auxiliary, list)
        ):
            raise ConsistencyError("ES-DFM checkpoint state is malformed")
        self._main.load_state_dict(main)
        parsed_states: dict[str, _AuxState] = {}
        for key, value in states.items():
            if not isinstance(value, dict):
                raise ConsistencyError("ES-DFM auxiliary state is malformed")
            try:
                parsed_states[str(key)] = _AuxState(
                    click_time=int(value["click_time"]),
                    feature=float(value["feature"]),
                    positive_at=(
                        None if value["positive_at"] is None else int(value["positive_at"])
                    ),
                    negative_matured=bool(value["negative_matured"]),
                    auxiliary_emitted=bool(value["auxiliary_emitted"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ConsistencyError("ES-DFM auxiliary state is malformed") from error
        parsed_auxiliary: list[AuxiliaryRecord] = []
        for value in auxiliary:
            if not isinstance(value, dict):
                raise ConsistencyError("ES-DFM auxiliary record is malformed")
            try:
                parsed_auxiliary.append(
                    AuxiliaryRecord(
                        click_id=str(value["click_id"]),
                        available_at=int(value["available_at"]),
                        feature=float(value["feature"]),
                        q_tn_target=(
                            None if value["q_tn_target"] is None else float(value["q_tn_target"])
                        ),
                        q_dp_target=float(value["q_dp_target"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ConsistencyError("ES-DFM auxiliary record is malformed") from error
        self._states = parsed_states
        self._auxiliary = parsed_auxiliary
