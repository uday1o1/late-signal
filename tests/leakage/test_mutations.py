from __future__ import annotations

import copy
from pathlib import Path

import pytest

from latesignal.contracts.records import TrainingRecord
from latesignal.data.manifests import sha256_file, write_json_atomic
from latesignal.errors import ConsistencyError
from latesignal.experiments.protocol_lock import _verify_prepared_data
from latesignal.features.batch import validate_training_batch
from latesignal.features.online_history import OnlineHistory
from latesignal.features.policy import load_feature_policy
from latesignal.scheduling.monitoring import is_monitoring_member
from latesignal.simulator.ledger import audit_event_trace
from latesignal.training.sampler import DeterministicSampler


def _record(click_id: str, *, available_at: int = 10) -> TrainingRecord:
    return TrainingRecord(
        record_id=f"record:{click_id}",
        click_id=click_id,
        available_at=available_at,
        status="final",
        target=1.0,
        weight=1.0,
        correction_group=None,
        source_method="leakage-test",
        feature=1.0,
    )


def _legal_trace() -> list[dict[str, object]]:
    return [
        {"sequence": 0, "simulator_time": 10, "kind": "prediction", "click_id": "click"},
        {
            "sequence": 1,
            "simulator_time": 10,
            "kind": "click_delivered",
            "click_id": "click",
        },
        {
            "sequence": 2,
            "simulator_time": 10,
            "kind": "positive_reveal",
            "click_id": "click",
        },
    ]


def _prepared_manifest(tmp_path: Path, *, fit_last_day: int) -> Path:
    root = tmp_path / f"processed-{fit_last_day}"
    data_file = root / "features" / "click_day=0" / "part.parquet"
    data_file.parent.mkdir(parents=True)
    data_file.write_bytes(b"bounded leakage fixture")
    digest, size = sha256_file(data_file)
    manifest = root / "manifests" / "preparation.json"
    write_json_atomic(
        manifest,
        {
            "manifest_version": 1,
            "rows": {"reconciled": True},
            "numeric_statistics": {"fit_click_days": [0, fit_last_day]},
            "files": [
                {
                    "path": "features/click_day=0/part.parquet",
                    "sha256": digest,
                    "bytes": size,
                }
            ],
        },
    )
    return manifest


@pytest.mark.parametrize("forbidden", ["Sale", "time_delay_for_conversion"])
def test_outcome_field_mutation_fails_while_exact_allowlist_passes(forbidden: str) -> None:
    policy = load_feature_policy(Path("configs/features.yaml"))
    legal = {column: [0] for column in policy.model_columns}
    validate_training_batch(legal, policy)

    mutated = {**legal, forbidden: [1]}
    with pytest.raises(ConsistencyError, match="allowlist"):
        validate_training_batch(mutated, policy)


def test_reveal_before_prediction_mutation_fails_while_legal_trace_passes() -> None:
    legal = _legal_trace()
    audit_event_trace(legal)
    mutated = copy.deepcopy(legal)
    mutated[0]["kind"] = "positive_reveal"
    mutated[2]["kind"] = "prediction"

    with pytest.raises(ConsistencyError, match="revealed before"):
        audit_event_trace(mutated)


def test_global_cold_status_mutation_diverges_from_past_only_control() -> None:
    history = OnlineHistory()
    legal = [history.observe("same-user", product) for product in ("one", "two")]
    globally_counted = [
        {"cold_user": False, "prior_user_clicks": 2},
        {"cold_user": False, "prior_user_clicks": 2},
    ]

    assert legal[0]["cold_user"] is True
    assert legal[0]["prior_user_clicks"] == 0
    assert legal != globally_counted


def test_monitoring_reuse_mutation_fails_while_training_member_passes() -> None:
    seed = 20260813
    monitoring_id = next(
        f"monitor-{index}"
        for index in range(10_000)
        if is_monitoring_member(f"monitor-{index}", seed)
    )
    training_id = next(
        f"training-{index}"
        for index in range(10_000)
        if not is_monitoring_member(f"training-{index}", seed)
    )
    sampler = DeterministicSampler(
        seed=17,
        recent_window_seconds=86_400,
        reservoir_capacity=10,
        excluded_click_ids=frozenset({monitoring_id}),
    )
    sampler.add(_record(training_id), simulator_time=10)

    with pytest.raises(ConsistencyError, match="Monitoring record"):
        sampler.add(_record(monitoring_id), simulator_time=10)


def test_early_truth_mutation_fails_at_the_exact_availability_boundary() -> None:
    record = _record("truth", available_at=10)
    record.assert_available(10)

    with pytest.raises(ConsistencyError, match="before legal availability"):
        record.assert_available(9)


def test_final_period_normalizer_mutation_fails_while_burn_in_manifest_passes(
    tmp_path: Path,
) -> None:
    legal = _prepared_manifest(tmp_path, fit_last_day=14)
    assert _verify_prepared_data(legal)["verified_files"] == 1
    mutated = _prepared_manifest(tmp_path, fit_last_day=89)

    with pytest.raises(ConsistencyError, match="final lock contract"):
        _verify_prepared_data(mutated)
