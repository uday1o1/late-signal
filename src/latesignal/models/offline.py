"""Chronological mature-label contract shared by offline sanity references."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from latesignal.errors import ConsistencyError

FloatMatrix = NDArray[np.float64]
FloatVector = NDArray[np.float64]
IntVector = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class MatureOfflineSplit:
    """A train/evaluation split whose training truth was legal at its cutoff."""

    train_features: FloatMatrix
    train_labels: IntVector
    train_click_times: FloatVector
    train_available_at: FloatVector
    evaluation_features: FloatMatrix
    evaluation_labels: IntVector
    evaluation_click_times: FloatVector
    training_cutoff: float
    maturity_window: float

    def __post_init__(self) -> None:
        self._validate_partition(
            "training",
            self.train_features,
            self.train_labels,
            self.train_click_times,
        )
        self._validate_partition(
            "evaluation",
            self.evaluation_features,
            self.evaluation_labels,
            self.evaluation_click_times,
        )
        if self.train_available_at.shape != self.train_click_times.shape:
            raise ConsistencyError("Training availability has an invalid shape")
        if not np.isfinite(self.train_available_at).all():
            raise ConsistencyError("Training availability contains a non-finite value")
        if np.any(self.train_click_times > self.training_cutoff):
            raise ConsistencyError("Offline training includes a future click")
        if not math.isfinite(self.maturity_window) or self.maturity_window <= 0.0:
            raise ConsistencyError("Offline maturity window must be finite and positive")
        if np.any(self.train_click_times + self.maturity_window > self.training_cutoff):
            raise ConsistencyError("Offline training includes an incomplete click cohort")
        if np.any(self.train_available_at > self.training_cutoff):
            raise ConsistencyError("Offline training includes an immature label")
        if np.any(self.evaluation_click_times <= self.training_cutoff):
            raise ConsistencyError("Offline evaluation is not strictly chronological")
        if np.unique(self.train_labels).size != 2:
            raise ConsistencyError("Offline training requires both binary classes")

    @staticmethod
    def _validate_partition(
        name: str,
        features: FloatMatrix,
        labels: IntVector,
        click_times: FloatVector,
    ) -> None:
        if features.ndim != 2 or features.shape[0] == 0:
            raise ConsistencyError(f"Offline {name} features must be a nonempty matrix")
        if labels.shape != (features.shape[0],) or click_times.shape != labels.shape:
            raise ConsistencyError(f"Offline {name} arrays have inconsistent shapes")
        if not np.isfinite(features).all() or not np.isfinite(click_times).all():
            raise ConsistencyError(f"Offline {name} data contains a non-finite value")
        if not np.isin(labels, (0, 1)).all():
            raise ConsistencyError(f"Offline {name} labels must be binary")


@dataclass(frozen=True, slots=True)
class OfflineReferencePrediction:
    name: str
    probabilities: FloatVector
    labels: IntVector
    training_examples: int
    evaluation_examples: int
    training_cutoff: float
    ranking_eligible: bool = False

    def __post_init__(self) -> None:
        if self.probabilities.shape != self.labels.shape:
            raise ConsistencyError("Offline prediction and label shapes do not match")
        if not np.isfinite(self.probabilities).all():
            raise ConsistencyError("Offline probabilities contain a non-finite value")
        if np.any((self.probabilities < 0.0) | (self.probabilities > 1.0)):
            raise ConsistencyError("Offline probabilities lie outside [0, 1]")
        if self.ranking_eligible:
            raise ConsistencyError("Offline references cannot enter the online ranking")
