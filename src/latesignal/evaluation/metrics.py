"""Core binary metrics for the synthetic vertical slice."""

from __future__ import annotations

import math

from latesignal.contracts.records import PredictionRecord
from latesignal.errors import ConsistencyError


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
