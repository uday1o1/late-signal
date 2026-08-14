from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import-untyped]

from latesignal.contracts.results import EvaluationDataset, EvaluationExample
from latesignal.errors import ConsistencyError
from latesignal.evaluation.analysis import (
    BudgetQualityPoint,
    compute_pareto_table,
    validate_intermediate_budget,
)
from latesignal.evaluation.bootstrap import (
    _metric,
    compact_bootstrap_metrics,
    contiguous_day_bootstrap_weights,
    paired_block_bootstrap,
    paired_compact_interval,
)
from latesignal.evaluation.compare import compare_methods, scheduler_success
from latesignal.evaluation.metrics import classification_metrics
from latesignal.evaluation.slices import evaluate_slices

SEEDS = (17, 41, 73)


def _datasets(method: str, *, quality: str) -> tuple[EvaluationDataset, ...]:
    datasets: list[EvaluationDataset] = []
    for seed in SEEDS:
        examples: list[EvaluationExample] = []
        for day in range(65, 90):
            for index in range(4):
                label = int(index % 2 == 0)
                if quality == "control":
                    probability = 0.65 if label else 0.35
                elif quality == "candidate":
                    probability = 0.8 if label else 0.2
                elif quality == "identical":
                    probability = 0.65 if label else 0.35
                else:
                    raise AssertionError("unknown fixture quality")
                prior = (day + index) % 12
                examples.append(
                    EvaluationExample(
                        click_id=f"click-{day}-{index}",
                        click_day=day,
                        final_label=label,
                        probability=probability,
                        cold_user=prior == 0,
                        cold_product=prior == 0,
                        prior_user_clicks=prior,
                        prior_product_clicks=prior,
                        product_price_bin="medium" if index < 2 else "high",
                        device_type="mobile" if index % 2 else "desktop",
                        conversion_delay_days=float(index + 1) if label else None,
                    )
                )
        datasets.append(
            EvaluationDataset(
                method=method,
                seed=seed,
                examples=tuple(examples),
                ranking_eligible=True,
                sealed=True,
            )
        )
    return tuple(datasets)


def test_metric_suite_does_not_fabricate_one_class_auc() -> None:
    metrics = classification_metrics([1, 1, 1], [0.8, 0.7, 0.9])

    assert metrics["count"] == 3
    assert metrics["pr_auc"] is None
    assert metrics["roc_auc"] is None
    assert metrics["calibration_intercept"] is None
    assert metrics["calibration_slope"] is None


def test_unsealed_or_out_of_period_evaluation_is_rejected() -> None:
    example = EvaluationExample(
        click_id="click",
        click_day=64,
        final_label=0,
        probability=0.5,
        cold_user=True,
        cold_product=True,
        prior_user_clicks=0,
        prior_product_clicks=0,
        product_price_bin="low",
        device_type="mobile",
        conversion_delay_days=None,
    )

    with pytest.raises(ConsistencyError, match="unsealed"):
        EvaluationDataset("method", 17, (example,), True, False)
    with pytest.raises(ConsistencyError, match="outside"):
        EvaluationDataset("method", 17, (example,), True, True)


def test_malformed_evaluation_row_is_rejected_instead_of_dropped() -> None:
    with pytest.raises(ConsistencyError, match="Every evaluation"):
        EvaluationDataset.from_dict(
            {
                "method": "method",
                "seed": 17,
                "ranking_eligible": True,
                "sealed": True,
                "examples": ["not-an-object"],
            }
        )


def test_optimized_bootstrap_auc_metrics_match_sklearn_with_ties() -> None:
    labels = [0, 1, 0, 1, 1, 0, 0, 1]
    probabilities = [0.1, 0.8, 0.4, 0.8, 0.4, 0.1, 0.7, 0.7]
    rows = [
        EvaluationExample(
            click_id=f"metric-{index}",
            click_day=65,
            final_label=label,
            probability=probability,
            cold_user=True,
            cold_product=True,
            prior_user_clicks=0,
            prior_product_clicks=0,
            product_price_bin="low",
            device_type="mobile",
            conversion_delay_days=1.0 if label else None,
        )
        for index, (label, probability) in enumerate(zip(labels, probabilities, strict=True))
    ]

    assert _metric(rows, "pr_auc") == pytest.approx(average_precision_score(labels, probabilities))
    assert _metric(rows, "roc_auc") == pytest.approx(roc_auc_score(labels, probabilities))


def test_compact_bootstrap_matches_explicit_resampling_with_ties() -> None:
    days = np.repeat(np.arange(65, 90, dtype=np.int16), 4)
    labels = np.tile(np.asarray([0, 1, 0, 1], dtype=np.int8), 25)
    probabilities = np.tile(np.asarray([0.1, 0.8, 0.4, 0.8], dtype=np.float64), 25)

    result = compact_bootstrap_metrics(
        labels,
        probabilities,
        days,
        block_days=3,
        replicates=2_000,
        batch_replicates=31,
    )

    assert result.point["pr_auc"] == pytest.approx(
        average_precision_score(labels, probabilities), abs=1e-12
    )
    assert result.point["roc_auc"] == pytest.approx(roc_auc_score(labels, probabilities), abs=1e-12)
    rows = [
        EvaluationExample(
            click_id=f"compact-{index}",
            click_day=int(day),
            final_label=int(label),
            probability=float(probability),
            cold_user=True,
            cold_product=True,
            prior_user_clicks=0,
            prior_product_clicks=0,
            product_price_bin="low",
            device_type="mobile",
            conversion_delay_days=1.0 if label else None,
        )
        for index, (day, label, probability) in enumerate(
            zip(days, labels, probabilities, strict=True)
        )
    ]
    for replicate in range(5):
        sampled = [
            row
            for day_index, count in enumerate(result.day_weights[replicate])
            for _ in range(int(count))
            for row in rows
            if row.click_day == 65 + day_index
        ]
        for metric in ("log_loss", "brier_score", "pr_auc", "roc_auc"):
            assert result.replicates[metric][replicate] == pytest.approx(
                _metric(sampled, metric), abs=1e-12
            )


def test_compact_bootstrap_day_weights_are_deterministic_and_bounded() -> None:
    days = np.arange(65, 90, dtype=np.int16)

    first = contiguous_day_bootstrap_weights(days, block_days=7, replicates=2_000)
    repeated = contiguous_day_bootstrap_weights(days, block_days=7, replicates=2_000)

    np.testing.assert_array_equal(first, repeated)
    assert first.shape == (2_000, 25)
    assert np.all(first.sum(axis=1) == 25)
    assert first.max() <= 25


def test_compact_paired_interval_averages_matched_seed_replicates() -> None:
    days = np.repeat(np.arange(65, 90, dtype=np.int16), 4)
    labels = np.tile(np.asarray([0, 1, 0, 1], dtype=np.int8), 25)
    control_probability = np.where(labels == 1, 0.65, 0.35).astype(np.float64)
    candidate_probability = np.where(labels == 1, 0.8, 0.2).astype(np.float64)
    control = {
        seed: compact_bootstrap_metrics(
            labels,
            control_probability,
            days,
            block_days=3,
            replicates=2_000,
            batch_replicates=29,
        )
        for seed in SEEDS
    }
    candidate = {
        seed: compact_bootstrap_metrics(
            labels,
            candidate_probability,
            days,
            block_days=3,
            replicates=2_000,
            batch_replicates=29,
        )
        for seed in SEEDS
    }

    interval = paired_compact_interval(control, candidate, metric="log_loss")

    assert interval.point_difference < 0.0
    assert interval.upper_95 < 0.0
    assert [item.seed for item in interval.seed_differences] == [17, 41, 73]


def test_empty_and_low_support_slices_are_reported_without_metrics() -> None:
    rows = _datasets("control", quality="control")[0].examples[:12]

    slices = evaluate_slices(
        rows,
        minimum_examples=10,
        minimum_positives=2,
        declared_device_types=("mobile", "desktop", "tablet"),
        declared_price_bins=("low", "medium", "high"),
    )
    tablet = next(
        item for item in slices if item.dimension == "device_type" and item.value == "tablet"
    )
    low_price = next(
        item for item in slices if item.dimension == "product_price_bin" and item.value == "low"
    )
    mobile = next(
        item for item in slices if item.dimension == "device_type" and item.value == "mobile"
    )

    assert tablet.count == 0 and tablet.metrics is None and tablet.suppression_reason == "empty"
    assert low_price.count == 0 and low_price.metrics is None
    assert mobile.count == 6 and mobile.metrics is None
    assert mobile.suppression_reason == "insufficient_examples"


def test_identical_paired_bootstrap_is_exactly_zero() -> None:
    control = _datasets("control", quality="control")
    identical = _datasets("identical", quality="identical")

    interval = paired_block_bootstrap(
        control,
        identical,
        metric="log_loss",
        replicates=2_000,
    )

    assert interval.point_difference == 0.0
    assert interval.lower_95 == 0.0
    assert interval.upper_95 == 0.0
    assert all(item.difference == 0.0 for item in interval.seed_differences)


def test_paired_bootstrap_detects_uniformly_better_candidate() -> None:
    control = _datasets("control", quality="control")
    candidate = _datasets("candidate", quality="candidate")

    interval = paired_block_bootstrap(
        control,
        candidate,
        metric="brier_score",
        block_days=3,
        replicates=2_000,
    )

    assert interval.point_difference < 0.0
    assert interval.upper_95 < 0.0
    assert all(item.difference < 0.0 for item in interval.seed_differences)


def test_comparison_exposes_seeds_sensitivities_and_locked_success_criterion() -> None:
    comparison = compare_methods(
        _datasets("fixed_deadline", quality="control"),
        _datasets("calibration_drift", quality="candidate"),
        replicates=2_000,
    )
    outcome = scheduler_success(comparison, identical_core_budget=True)

    assert comparison["seeds"] == [17, 41, 73]
    assert len(comparison["seed_metrics"]) == 6
    assert set(comparison["paired_intervals"]["log_loss"]) == {"1", "3", "7"}
    assert comparison["consistent_favorable_log_loss_sign"] is True
    assert outcome["supported"] is True


def test_intermediate_budget_and_compute_pareto_tables() -> None:
    points = tuple(
        BudgetQualityPoint("method", fraction, int(100 * fraction), 0.8 - fraction / 10)
        for fraction in (0.25, 0.5, 0.75, 1.0)
    )
    table = validate_intermediate_budget(points)
    pareto = compute_pareto_table(
        (
            ("cheap", 0.5, 100, 100),
            ("balanced", 0.4, 100, 150),
            ("dominated", 0.6, 100, 200),
        )
    )

    assert [item["budget_fraction"] for item in table] == [0.25, 0.5, 0.75, 1.0]
    assert {item["method"] for item in pareto if item["pareto_efficient"]} == {
        "cheap",
        "balanced",
    }
    assert math.isfinite(float(table[-1]["log_loss"]))
