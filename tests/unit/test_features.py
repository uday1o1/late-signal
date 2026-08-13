from __future__ import annotations

from pathlib import Path

import pytest

from latesignal.errors import ConsistencyError
from latesignal.features.batch import validate_training_batch
from latesignal.features.hashing import categorical_bucket, click_id
from latesignal.features.online_history import OnlineHistory
from latesignal.features.policy import load_feature_policy


def test_hashing_is_field_specific_and_stable() -> None:
    first = categorical_bucket("user_id", "same", 17, 2**20)
    repeated = categorical_bucket("user_id", "same", 17, 2**20)
    other_field = categorical_bucket("product_id", "same", 17, 2**20)

    assert first == repeated
    assert first != other_field
    assert click_id("a" * 64, 7) == click_id("a" * 64, 7)
    assert click_id("a" * 64, 7) != click_id("a" * 64, 8)


def test_online_history_uses_only_strictly_prior_clicks() -> None:
    history = OnlineHistory()

    first = history.observe("user", "product")
    second = history.observe("user", "product")

    assert first == {
        "cold_user": True,
        "cold_product": True,
        "prior_user_clicks": 0,
        "prior_product_clicks": 0,
    }
    assert second == {
        "cold_user": False,
        "cold_product": False,
        "prior_user_clicks": 1,
        "prior_product_clicks": 1,
    }


@pytest.mark.parametrize("injected", ["Sale", "time_delay_for_conversion", "reveal_time"])
def test_training_batch_rejects_outcome_and_timing_fields(injected: str) -> None:
    policy = load_feature_policy(Path("configs/features.yaml"))
    valid = {column: [0] for column in policy.model_columns}
    valid[injected] = [1]

    with pytest.raises(ConsistencyError, match="allowlist"):
        validate_training_batch(valid, policy)


def test_training_batch_requires_every_authored_model_field() -> None:
    policy = load_feature_policy(Path("configs/features.yaml"))
    incomplete = {column: [0] for column in policy.model_columns}
    incomplete.pop(next(iter(incomplete)))

    with pytest.raises(ConsistencyError, match="allowlist"):
        validate_training_batch(incomplete, policy)
