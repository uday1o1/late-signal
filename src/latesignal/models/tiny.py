"""Deterministic scalar logistic model for the CPU vertical slice."""

from __future__ import annotations

import math

from latesignal.contracts.records import TrainingRecord
from latesignal.errors import ConsistencyError


def stable_sigmoid(logit: float) -> float:
    if logit >= 0.0:
        inverse = math.exp(-logit)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


class TinyLogisticModel:
    """One-feature model whose updates are deterministic on CPU."""

    def __init__(self) -> None:
        self.weight = 0.0
        self.bias = 0.0
        self.version = 0

    def predict(self, feature: float) -> float:
        return stable_sigmoid(self.weight * feature + self.bias)

    def train_step(self, records: tuple[TrainingRecord, ...], learning_rate: float) -> None:
        if not records:
            raise ConsistencyError("A training step requires at least one legal record")
        weight_gradient = 0.0
        bias_gradient = 0.0
        total_weight = 0.0
        for record in records:
            probability = self.predict(record.feature)
            residual = (probability - record.target) * record.weight
            weight_gradient += residual * record.feature
            bias_gradient += residual
            total_weight += record.weight
        if total_weight <= 0.0:
            raise ConsistencyError("A training batch must have positive total weight")
        self.weight -= learning_rate * weight_gradient / total_weight
        self.bias -= learning_rate * bias_gradient / total_weight
        self.version += 1

    def state_dict(self) -> dict[str, object]:
        return {"weight": self.weight, "bias": self.bias, "version": self.version}

    def load_state_dict(self, state: dict[str, object]) -> None:
        weight = state.get("weight")
        bias = state.get("bias")
        version = state.get("version")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or isinstance(bias, bool)
            or not isinstance(bias, (int, float))
            or isinstance(version, bool)
            or not isinstance(version, int)
        ):
            raise ConsistencyError("Tiny-model checkpoint state is malformed")
        self.weight = float(weight)
        self.bias = float(bias)
        self.version = version
