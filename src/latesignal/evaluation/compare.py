"""Seed-aware paired final-period method comparison."""

from __future__ import annotations

from typing import Any

from latesignal.contracts.results import EvaluationDataset
from latesignal.errors import ConsistencyError
from latesignal.evaluation.bootstrap import paired_block_bootstrap
from latesignal.evaluation.metrics import classification_metrics


def compare_methods(
    control: tuple[EvaluationDataset, ...],
    candidate: tuple[EvaluationDataset, ...],
    *,
    replicates: int = 2_000,
    bootstrap_seed: int = 20260813,
) -> dict[str, Any]:
    """Compare matched seeds with primary and sensitivity block intervals."""

    if len(control) < 3 or len(candidate) < 3:
        raise ConsistencyError("Final paired comparison requires at least three seeds")
    if any(not dataset.ranking_eligible for dataset in control + candidate):
        raise ConsistencyError("Non-ranking references cannot enter deployable comparison")
    metric_names = ("log_loss", "brier_score", "pr_auc", "roc_auc")
    intervals: dict[str, dict[str, object]] = {}
    for metric in metric_names:
        intervals[metric] = {
            str(block): paired_block_bootstrap(
                control,
                candidate,
                metric=metric,
                block_days=block,
                replicates=replicates,
                bootstrap_seed=bootstrap_seed,
            ).as_dict()
            for block in (1, 3, 7)
        }
    seed_metrics: list[dict[str, object]] = []
    for dataset in (*control, *candidate):
        seed_metrics.append(
            {
                "method": dataset.method,
                "seed": dataset.seed,
                "metrics": classification_metrics(
                    [row.final_label for row in dataset.examples],
                    [row.probability for row in dataset.examples],
                ),
            }
        )
    primary_log_loss = intervals["log_loss"]["3"]
    assert isinstance(primary_log_loss, dict)
    seed_differences = primary_log_loss["seed_differences"]
    assert isinstance(seed_differences, list)
    consistent_favorable_sign = all(
        isinstance(item, dict) and float(item["difference"]) < 0.0 for item in seed_differences
    )
    return {
        "control": control[0].method,
        "candidate": candidate[0].method,
        "seeds": sorted(dataset.seed for dataset in control),
        "seed_metrics": seed_metrics,
        "paired_intervals": intervals,
        "consistent_favorable_log_loss_sign": consistent_favorable_sign,
    }


def scheduler_success(
    comparison: dict[str, Any],
    *,
    identical_core_budget: bool,
) -> dict[str, object]:
    """Evaluate the predeclared scheduler support criterion without relaxing it."""

    intervals = comparison["paired_intervals"]
    assert isinstance(intervals, dict)
    log_loss = intervals["log_loss"]
    brier = intervals["brier_score"]
    assert isinstance(log_loss, dict) and isinstance(brier, dict)
    primary_log = log_loss["3"]
    primary_brier = brier["3"]
    assert isinstance(primary_log, dict) and isinstance(primary_brier, dict)
    seed_metrics = comparison["seed_metrics"]
    assert isinstance(seed_metrics, list)
    control = str(comparison["control"])
    candidate = str(comparison["candidate"])
    control_ece = [
        float(item["metrics"]["expected_calibration_error"])
        for item in seed_metrics
        if isinstance(item, dict) and item.get("method") == control
    ]
    candidate_ece = [
        float(item["metrics"]["expected_calibration_error"])
        for item in seed_metrics
        if isinstance(item, dict) and item.get("method") == candidate
    ]
    if len(control_ece) != len(candidate_ece) or not control_ece:
        raise ConsistencyError("Scheduler comparison ECE seeds do not align")
    ece_degradation = sum(candidate_ece) / len(candidate_ece) - sum(control_ece) / len(control_ece)
    sensitivity_not_reversed = all(
        isinstance(log_loss[str(block)], dict)
        and float(log_loss[str(block)]["point_difference"]) < 0.0
        for block in (1, 7)
    )
    conditions = {
        "log_loss_interval_below_zero": float(primary_log["upper_95"]) < 0.0,
        "brier_upper_degradation_within_limit": float(primary_brier["upper_95"]) <= 0.0005,
        "ece_point_degradation_within_limit": ece_degradation <= 0.002,
        "identical_core_budget": identical_core_budget,
        "consistent_across_seeds": comparison["consistent_favorable_log_loss_sign"] is True,
        "sensitivity_not_reversed": sensitivity_not_reversed,
    }
    return {
        "supported": all(conditions.values()),
        "conditions": conditions,
        "ece_point_degradation": ece_degradation,
        "outcome": "supported" if all(conditions.values()) else "negative_or_inconclusive",
    }
