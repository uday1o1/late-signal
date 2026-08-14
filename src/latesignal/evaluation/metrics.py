"""Core binary metrics for the synthetic vertical slice."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    roc_auc_score,
)

from latesignal.contracts.records import PredictionRecord
from latesignal.errors import ConsistencyError
from latesignal.evaluation.calibration import evaluate_calibration


def classification_metrics(
    labels: ArrayLike,
    probabilities: ArrayLike,
) -> dict[str, float | int | list[dict[str, int | float | None]] | None]:
    """Compute the locked overall binary metric suite without fabricating AUCs."""

    target = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    if target.ndim != 1 or probability.shape != target.shape or target.size == 0:
        raise ConsistencyError("Metric arrays must be nonempty matching vectors")
    if not np.isin(target, (0, 1)).all():
        raise ConsistencyError("Metric labels must be binary")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ConsistencyError("Metric probabilities must be finite and lie in [0, 1]")
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    losses = -(target * np.log(clipped) + (1 - target) * np.log1p(-clipped))
    brier = np.square(probability - target)
    both_classes = np.unique(target).size == 2
    calibration = evaluate_calibration(target, probability)
    return {
        "count": int(target.size),
        "positives": int(target.sum()),
        "log_loss": float(losses.mean()),
        "brier_score": float(brier.mean()),
        "pr_auc": float(average_precision_score(target, probability)) if both_classes else None,
        "roc_auc": float(roc_auc_score(target, probability)) if both_classes else None,
        "calibration_intercept": calibration.intercept,
        "calibration_slope": calibration.slope,
        "expected_calibration_error": calibration.expected_calibration_error,
        "reliability": [item.as_dict() for item in calibration.bins],
    }


def binary_metrics(
    predictions: tuple[PredictionRecord, ...], final_labels: dict[str, int]
) -> dict[str, float | int]:
    if not predictions:
        raise ConsistencyError("Evaluation requires at least one prediction")
    prediction_ids = {prediction.click_id for prediction in predictions}
    if prediction_ids != set(final_labels):
        raise ConsistencyError(
            "Prediction and final-truth IDs do not match",
            details={
                "missing_predictions": sorted(set(final_labels) - prediction_ids),
                "missing_truth": sorted(prediction_ids - set(final_labels)),
            },
        )
    log_loss = 0.0
    brier = 0.0
    positives = 0
    for prediction in predictions:
        target = final_labels[prediction.click_id]
        positives += target
        probability = min(max(prediction.probability, 1e-12), 1.0 - 1e-12)
        log_loss -= target * math.log(probability) + (1 - target) * math.log1p(-probability)
        brier += (probability - target) ** 2
    count = len(predictions)
    return {
        "count": count,
        "positives": positives,
        "log_loss": log_loss / count,
        "brier_score": brier / count,
    }
