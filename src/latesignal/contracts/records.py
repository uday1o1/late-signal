"""Training, prediction, and compute-ledger records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from latesignal.errors import ConsistencyError


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    record_id: str
    click_id: str
    available_at: int
    status: Literal["final", "provisional"]
    target: float
    weight: float
    correction_group: str | None
    source_method: str
    feature: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.target <= 1.0:
            raise ValueError("target must lie in [0, 1]")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def assert_available(self, simulator_time: int) -> None:
        if self.available_at > simulator_time:
            raise ConsistencyError(
                "Training record arrived before legal availability",
                details={
                    "record_id": self.record_id,
                    "available_at": self.available_at,
                    "simulator_time": simulator_time,
                },
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainingRecord:
        return cls(
            record_id=str(value["record_id"]),
            click_id=str(value["click_id"]),
            available_at=int(value["available_at"]),
            status=value["status"],
            target=float(value["target"]),
            weight=float(value["weight"]),
            correction_group=(
                None if value["correction_group"] is None else str(value["correction_group"])
            ),
            source_method=str(value["source_method"]),
            feature=float(value["feature"]),
        )


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    click_id: str
    click_time: int
    probability: float
    model_version: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PredictionRecord:
        return cls(
            click_id=str(value["click_id"]),
            click_time=int(value["click_time"]),
            probability=float(value["probability"]),
            model_version=int(value["model_version"]),
        )


@dataclass(frozen=True, slots=True)
class CreditRecord:
    credit_id: int
    decision_time: int
    steps: int
    examples: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CreditRecord:
        return cls(
            credit_id=int(value["credit_id"]),
            decision_time=int(value["decision_time"]),
            steps=int(value["steps"]),
            examples=int(value["examples"]),
        )


@dataclass(frozen=True, slots=True)
class ExposureRecord:
    credit_id: int
    step: int
    record_id: str
    weight: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExposureRecord:
        return cls(
            credit_id=int(value["credit_id"]),
            step=int(value["step"]),
            record_id=str(value["record_id"]),
            weight=float(value["weight"]),
        )
