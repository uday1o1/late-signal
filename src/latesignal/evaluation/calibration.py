"""Locked-bin calibration evaluation for binary predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from latesignal.errors import ConsistencyError


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    index: int
    lower: float
    upper: float
    count: int
    positives: int
    mean_probability: float | None
    observed_rate: float | None

    def as_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    count: int
    positives: int
    intercept: float | None
    slope: float | None
    expected_calibration_error: float
    bins: tuple[ReliabilityBin, ...]


def evaluate_calibration(
    labels: ArrayLike,
    probabilities: ArrayLike,
    *,
    bin_count: int = 10,
) -> CalibrationResult:
    """Evaluate predeclared equal-width bins and logistic calibration coefficients."""

    target = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    if target.ndim != 1 or probability.shape != target.shape or target.size == 0:
        raise ConsistencyError("Calibration arrays must be nonempty matching vectors")
    if not np.isin(target, (0, 1)).all():
        raise ConsistencyError("Calibration labels must be binary")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ConsistencyError("Calibration probabilities must be finite and lie in [0, 1]")
    if bin_count <= 1:
        raise ValueError("bin_count must exceed one")

    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    intercept: float | None = None
    slope: float | None = None
    if np.unique(target).size == 2:
        calibrator = LogisticRegression(
            C=1e12,
            fit_intercept=True,
            max_iter=2_000,
            solver="lbfgs",
        )
        calibrator.fit(logit, target)
        intercept = float(calibrator.intercept_[0])
        slope = float(calibrator.coef_[0, 0])

    indices = np.minimum((probability * bin_count).astype(np.int64), bin_count - 1)
    bins: list[ReliabilityBin] = []
    weighted_error = 0.0
    for index in range(bin_count):
        mask = indices == index
        count = int(mask.sum())
        positives = int(target[mask].sum())
        mean_probability = float(probability[mask].mean()) if count else None
        observed_rate = float(target[mask].mean()) if count else None
        if count and mean_probability is not None and observed_rate is not None:
            weighted_error += count * abs(mean_probability - observed_rate)
        bins.append(
            ReliabilityBin(
                index=index,
                lower=index / bin_count,
                upper=(index + 1) / bin_count,
                count=count,
                positives=positives,
                mean_probability=mean_probability,
                observed_rate=observed_rate,
            )
        )
    return CalibrationResult(
        count=int(target.size),
        positives=int(target.sum()),
        intercept=intercept,
        slope=slope,
        expected_calibration_error=weighted_error / target.size,
        bins=tuple(bins),
    )
