"""Deterministic mature monitoring split and calibration residuals."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import xxhash

from latesignal.errors import ConsistencyError


@dataclass(frozen=True, slots=True)
class MonitoringExample:
    click_id: str
    click_day: int
    final_label: int
    probability: float

    def __post_init__(self) -> None:
        if self.final_label not in {0, 1}:
            raise ValueError("Monitoring label must be binary")
        if not math.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise ValueError("Monitoring probability must be finite and lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class ResidualBin:
    index: int
    count: int
    positives: int
    probability_sum: float
    signed_residual_sum: float
    variance_sum: float
    eligible: bool
    standardized_residual: float | None

    def as_dict(self) -> dict[str, int | float | bool | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CalibrationEvidence:
    decision_day: int
    model_checkpoint_sha256: str
    monitoring_cohort_first_day: int
    monitoring_cohort_last_day: int
    monitoring_examples: int
    score: float | None
    contributing_bin: int | None
    bins: tuple[ResidualBin, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "decision_day": self.decision_day,
            "model_checkpoint_sha256": self.model_checkpoint_sha256,
            "monitoring_cohort_first_day": self.monitoring_cohort_first_day,
            "monitoring_cohort_last_day": self.monitoring_cohort_last_day,
            "monitoring_examples": self.monitoring_examples,
            "score": self.score,
            "contributing_bin": self.contributing_bin,
            "bins": [value.as_dict() for value in self.bins],
        }


def is_monitoring_member(click_id: str, seed: int) -> bool:
    """Apply the locked deterministic 10 percent monitoring assignment."""

    return xxhash.xxh64_intdigest(click_id.encode(), seed=seed) % 10 == 0


def calibration_evidence(
    examples: tuple[MonitoringExample, ...],
    *,
    decision_day: int,
    model_checkpoint_sha256: str,
    monitor_seed: int,
    maturity_days: int = 30,
    monitoring_window_days: int = 7,
    bin_count: int = 10,
    minimum_bin_examples: int = 1_000,
    minimum_variance: float = 25.0,
    epsilon: float = 1e-8,
) -> CalibrationEvidence:
    """Compute the maximum eligible fixed-bin mature calibration residual."""

    if decision_day < maturity_days or monitoring_window_days <= 0 or bin_count <= 1:
        raise ValueError("Calibration monitoring bounds are invalid")
    if minimum_bin_examples <= 0 or minimum_variance < 0.0 or epsilon <= 0.0:
        raise ValueError("Calibration support settings are invalid")
    newest_day = decision_day - maturity_days
    first_day = newest_day - monitoring_window_days + 1
    selected = tuple(
        example
        for example in examples
        if first_day <= example.click_day <= newest_day
        and is_monitoring_member(example.click_id, monitor_seed)
    )
    if any(example.click_day > newest_day for example in selected):
        raise ConsistencyError("Monitoring included an immature click cohort")
    accumulators = [
        {"count": 0, "positives": 0, "probability_sum": 0.0, "residual": 0.0, "variance": 0.0}
        for _ in range(bin_count)
    ]
    for example in selected:
        index = min(int(example.probability * bin_count), bin_count - 1)
        value = accumulators[index]
        value["count"] += 1
        value["positives"] += example.final_label
        value["probability_sum"] += example.probability
        value["residual"] += example.final_label - example.probability
        value["variance"] += example.probability * (1.0 - example.probability)
    bins: list[ResidualBin] = []
    candidates: list[tuple[float, int]] = []
    for index, value in enumerate(accumulators):
        count = int(value["count"])
        variance = float(value["variance"])
        eligible = count >= minimum_bin_examples and variance >= minimum_variance
        residual = (
            abs(float(value["residual"])) / math.sqrt(variance + epsilon) if eligible else None
        )
        if residual is not None:
            candidates.append((residual, index))
        bins.append(
            ResidualBin(
                index=index,
                count=count,
                positives=int(value["positives"]),
                probability_sum=float(value["probability_sum"]),
                signed_residual_sum=float(value["residual"]),
                variance_sum=variance,
                eligible=eligible,
                standardized_residual=residual,
            )
        )
    score: float | None = None
    contributing_bin: int | None = None
    if candidates:
        score, contributing_bin = max(candidates, key=lambda item: (item[0], -item[1]))
    return CalibrationEvidence(
        decision_day=decision_day,
        model_checkpoint_sha256=model_checkpoint_sha256,
        monitoring_cohort_first_day=first_day,
        monitoring_cohort_last_day=newest_day,
        monitoring_examples=len(selected),
        score=score,
        contributing_bin=contributing_bin,
        bins=tuple(bins),
    )
