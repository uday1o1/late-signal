"""Paired contiguous-day block bootstrap across matched seeds."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from latesignal.contracts.results import EvaluationDataset, EvaluationExample
from latesignal.errors import ConsistencyError


@dataclass(frozen=True, slots=True)
class SeedDifference:
    seed: int
    control: float
    candidate: float
    difference: float


@dataclass(frozen=True, slots=True)
class PairedInterval:
    metric: str
    block_days: int
    replicates: int
    point_difference: float
    lower_95: float
    upper_95: float
    seed_differences: tuple[SeedDifference, ...]

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["seed_differences"] = [asdict(item) for item in self.seed_differences]
        return value


def _metric(rows: list[EvaluationExample], name: str) -> float:
    target = np.asarray([row.final_label for row in rows], dtype=np.int64)
    probability = np.asarray([row.probability for row in rows], dtype=np.float64)
    if name == "log_loss":
        clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
        return float(np.mean(-(target * np.log(clipped) + (1 - target) * np.log1p(-clipped))))
    if name == "brier_score":
        return float(np.mean(np.square(probability - target)))
    if np.unique(target).size != 2:
        raise ConsistencyError(f"Bootstrap metric {name} is undefined on a one-class resample")
    if name == "pr_auc":
        order = np.argsort(-probability, kind="stable")
        sorted_probability = probability[order]
        sorted_target = target[order]
        boundaries = np.r_[np.flatnonzero(np.diff(sorted_probability)) + 1, target.size]
        cumulative_true = np.cumsum(sorted_target)[boundaries - 1]
        precision = cumulative_true / boundaries
        true_increments = np.diff(np.r_[0, cumulative_true])
        return float(np.sum(true_increments * precision) / target.sum())
    if name == "roc_auc":
        order = np.argsort(probability, kind="stable")
        sorted_probability = probability[order]
        ranks = np.arange(1, target.size + 1, dtype=np.float64)
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_probability)) + 1]
        ends = np.r_[starts[1:], target.size]
        for start, end in zip(starts, ends, strict=True):
            ranks[start:end] = (start + 1 + end) / 2.0
        ranked_target = target[order]
        positives = int(target.sum())
        negatives = target.size - positives
        rank_sum = float(ranks[ranked_target == 1].sum())
        return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    raise ValueError(f"Unsupported bootstrap metric: {name}")


def _aligned_pairs(
    control: tuple[EvaluationDataset, ...],
    candidate: tuple[EvaluationDataset, ...],
) -> dict[int, tuple[dict[str, EvaluationExample], dict[str, EvaluationExample]]]:
    control_by_seed = {dataset.seed: dataset for dataset in control}
    candidate_by_seed = {dataset.seed: dataset for dataset in candidate}
    if set(control_by_seed) != set(candidate_by_seed) or len(control_by_seed) < 1:
        raise ConsistencyError("Bootstrap methods do not have the same seeds")
    aligned: dict[int, tuple[dict[str, EvaluationExample], dict[str, EvaluationExample]]] = {}
    common_ids: set[str] | None = None
    for seed in sorted(control_by_seed):
        control_rows = {row.click_id: row for row in control_by_seed[seed].examples}
        candidate_rows = {row.click_id: row for row in candidate_by_seed[seed].examples}
        if set(control_rows) != set(candidate_rows):
            raise ConsistencyError(f"Bootstrap methods have different click IDs for seed {seed}")
        if any(
            control_rows[key].final_label != candidate_rows[key].final_label
            or control_rows[key].click_day != candidate_rows[key].click_day
            for key in control_rows
        ):
            raise ConsistencyError("Bootstrap matched rows disagree on truth or click day")
        if common_ids is None:
            common_ids = set(control_rows)
        elif set(control_rows) != common_ids:
            raise ConsistencyError("Bootstrap seeds do not share identical final click IDs")
        aligned[seed] = control_rows, candidate_rows
    return aligned


def paired_block_bootstrap(
    control: tuple[EvaluationDataset, ...],
    candidate: tuple[EvaluationDataset, ...],
    *,
    metric: str,
    block_days: int = 3,
    replicates: int = 2_000,
    bootstrap_seed: int = 20260813,
) -> PairedInterval:
    """Bootstrap candidate-minus-control differences with joint day resampling."""

    if metric not in {"log_loss", "brier_score", "pr_auc", "roc_auc"}:
        raise ValueError("Unsupported paired bootstrap metric")
    if block_days <= 0 or replicates < 2_000:
        raise ValueError("Bootstrap requires positive blocks and at least 2,000 replicates")
    aligned = _aligned_pairs(control, candidate)
    first_control = next(iter(aligned.values()))[0]
    days = sorted({row.click_day for row in first_control.values()})
    if not days or block_days > len(days):
        raise ConsistencyError("Bootstrap block is longer than the observed day sequence")
    if days != list(range(days[0], days[-1] + 1)):
        raise ConsistencyError("Bootstrap requires contiguous observed click days")
    starts = np.arange(0, len(days) - block_days + 1)
    blocks_needed = int(np.ceil(len(days) / block_days))
    rng = np.random.default_rng(bootstrap_seed)
    seed_points: list[SeedDifference] = []
    rows_by_seed: dict[
        int, tuple[dict[int, list[EvaluationExample]], dict[int, list[EvaluationExample]]]
    ] = {}
    for seed, (control_rows, candidate_rows) in aligned.items():
        ordered_control = list(control_rows.values())
        ordered_candidate = list(candidate_rows.values())
        control_point = _metric(ordered_control, metric)
        candidate_point = _metric(ordered_candidate, metric)
        seed_points.append(
            SeedDifference(seed, control_point, candidate_point, candidate_point - control_point)
        )
        control_by_day: dict[int, list[EvaluationExample]] = {}
        candidate_by_day: dict[int, list[EvaluationExample]] = {}
        for row in control_rows.values():
            control_by_day.setdefault(row.click_day, []).append(row)
        for row in candidate_rows.values():
            candidate_by_day.setdefault(row.click_day, []).append(row)
        rows_by_seed[seed] = control_by_day, candidate_by_day
    replicate_values = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sampled_days = [
            days[index]
            for start in sampled_starts
            for index in range(int(start), int(start) + block_days)
        ][: len(days)]
        differences: list[float] = []
        for control_by_day, candidate_by_day in rows_by_seed.values():
            sampled_control = [row for day in sampled_days for row in control_by_day[day]]
            sampled_candidate = [row for day in sampled_days for row in candidate_by_day[day]]
            differences.append(
                _metric(sampled_candidate, metric) - _metric(sampled_control, metric)
            )
        replicate_values[replicate] = float(np.mean(differences))
    lower, upper = np.quantile(replicate_values, [0.025, 0.975])
    return PairedInterval(
        metric=metric,
        block_days=block_days,
        replicates=replicates,
        point_difference=float(np.mean([item.difference for item in seed_points])),
        lower_95=float(lower),
        upper_95=float(upper),
        seed_differences=tuple(seed_points),
    )
