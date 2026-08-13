"""Mature-label logistic-regression sanity reference."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from latesignal.models.offline import MatureOfflineSplit, OfflineReferencePrediction


def run_logistic_reference(
    split: MatureOfflineSplit,
    *,
    seed: int,
) -> OfflineReferencePrediction:
    """Fit only on labels mature at the chronological training cutoff."""

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1_000, random_state=seed, solver="lbfgs"),
    )
    model.fit(split.train_features, split.train_labels)
    probabilities = model.predict_proba(split.evaluation_features)[:, 1]
    return OfflineReferencePrediction(
        name="mature_logistic_regression",
        probabilities=probabilities,
        labels=split.evaluation_labels.copy(),
        training_examples=split.train_labels.size,
        evaluation_examples=split.evaluation_labels.size,
        training_cutoff=split.training_cutoff,
    )
