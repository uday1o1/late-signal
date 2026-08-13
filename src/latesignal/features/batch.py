"""Model-batch construction that enforces the authored allowlist."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from latesignal.features.policy import FeaturePolicy


def validate_training_batch(batch: Mapping[str, Sequence[object]], policy: FeaturePolicy) -> None:
    """Fail closed if a missing, extra, outcome, or timing column enters training."""

    policy.validate_training_columns(batch.keys())
    lengths = {len(values) for values in batch.values()}
    if len(lengths) > 1:
        raise ValueError("Training batch columns must have equal lengths")
