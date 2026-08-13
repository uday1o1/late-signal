"""Mature-label LightGBM sanity reference."""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier

from latesignal.models.offline import MatureOfflineSplit, OfflineReferencePrediction


def run_lightgbm_reference(
    split: MatureOfflineSplit,
    *,
    seed: int,
    estimators: int = 100,
) -> OfflineReferencePrediction:
    """Fit a deterministic CPU reference on chronologically legal mature truth."""

    if estimators <= 0:
        raise ValueError("estimators must be positive")
    model = LGBMClassifier(
        objective="binary",
        n_estimators=estimators,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(split.train_features, split.train_labels)
    probabilities = np.asarray(model.predict_proba(split.evaluation_features), dtype=np.float64)[
        :, 1
    ]
    return OfflineReferencePrediction(
        name="mature_lightgbm",
        probabilities=probabilities,
        labels=split.evaluation_labels.copy(),
        training_examples=split.train_labels.size,
        evaluation_examples=split.evaluation_labels.size,
        training_cutoff=split.training_cutoff,
    )
