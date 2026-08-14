"""Paired contiguous-day block bootstrap across matched seeds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

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


@dataclass(frozen=True, slots=True)
class CompactBootstrapMetrics:
    """Point metrics and exact day-weighted bootstrap replicates for one run."""

    block_days: int
    bootstrap_seed: int
    day_weights: NDArray[np.int16]
    point: dict[str, float]
    replicates: dict[str, NDArray[np.float64]]


def contiguous_day_bootstrap_weights(
    observed_days: NDArray[np.integer],
    *,
    block_days: int,
    replicates: int,
    bootstrap_seed: int = 20260813,
) -> NDArray[np.int16]:
    """Generate the locked truncated contiguous-block day multiplicities."""

    days = np.unique(np.asarray(observed_days, dtype=np.int64))
    if (
        days.size == 0
        or block_days <= 0
        or block_days > days.size
        or replicates < 2_000
        or not np.array_equal(days, np.arange(days[0], days[-1] + 1))
    ):
        raise ConsistencyError("Bootstrap requires a contiguous day sequence and valid bounds")
    starts = np.arange(days.size - block_days + 1, dtype=np.int64)
    blocks_needed = int(np.ceil(days.size / block_days))
    rng = np.random.default_rng(bootstrap_seed)
    sampled_starts = rng.choice(starts, size=(replicates, blocks_needed), replace=True)
    sampled_offsets = np.arange(block_days, dtype=np.int64)
    sampled_positions = (sampled_starts[:, :, None] + sampled_offsets).reshape(replicates, -1)
    sampled_positions = sampled_positions[:, : days.size]
    weights = np.zeros((replicates, days.size), dtype=np.int16)
    row_indexes = np.repeat(np.arange(replicates, dtype=np.int64), days.size)
    np.add.at(weights, (row_indexes, sampled_positions.reshape(-1)), 1)
    if np.any(weights < 0) or not np.all(weights.sum(axis=1) == days.size):
        raise ConsistencyError("Bootstrap day multiplicities did not reconcile")
    return weights


def _weighted_auc_replicates(
    labels: NDArray[np.int8],
    probabilities: NDArray[np.float64],
    day_codes: NDArray[np.int64],
    day_weights: NDArray[np.int16],
    *,
    device: Literal["cpu", "cuda"],
    batch_replicates: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if batch_replicates <= 0:
        raise ValueError("Bootstrap replicate batch must be positive")
    if device == "cuda" and not torch.cuda.is_available():
        raise ConsistencyError("CUDA bootstrap was requested without an available accelerator")
    torch_device = torch.device(device)
    order = np.argsort(-probabilities, kind="stable")
    ordered_probability = probabilities[order]
    group_ends = np.r_[np.flatnonzero(np.diff(ordered_probability)) + 1, labels.size] - 1
    ordered_days = torch.from_numpy(day_codes[order]).to(torch_device, dtype=torch.long)
    ordered_labels = torch.from_numpy(labels[order]).to(torch_device, dtype=torch.float64)
    ends = torch.from_numpy(group_ends.astype(np.int64)).to(torch_device)
    pr_values = np.empty(day_weights.shape[0], dtype=np.float64)
    roc_values = np.empty(day_weights.shape[0], dtype=np.float64)
    for first in range(0, day_weights.shape[0], batch_replicates):
        last = min(first + batch_replicates, day_weights.shape[0])
        weights = torch.from_numpy(day_weights[first:last].astype(np.float64)).to(torch_device)
        row_weights = weights.index_select(1, ordered_days)
        cumulative_total = torch.cumsum(row_weights, dim=1)
        row_weights.mul_(ordered_labels)
        cumulative_positive = torch.cumsum(row_weights, dim=1)
        total_at_end = cumulative_total.index_select(1, ends)
        positive_at_end = cumulative_positive.index_select(1, ends)
        del cumulative_total, cumulative_positive, row_weights, weights
        zeros = torch.zeros((last - first, 1), dtype=torch.float64, device=torch_device)
        group_positive = torch.diff(positive_at_end, dim=1, prepend=zeros)
        group_total = torch.diff(total_at_end, dim=1, prepend=zeros)
        group_negative = group_total - group_positive
        total_positive = positive_at_end[:, -1]
        total_negative = total_at_end[:, -1] - total_positive
        if torch.any(total_positive <= 0.0) or torch.any(total_negative <= 0.0):
            raise ConsistencyError("Bootstrap produced a one-class resample")
        precision = torch.where(
            total_at_end > 0.0,
            positive_at_end / total_at_end,
            torch.zeros_like(total_at_end),
        )
        pr_batch = torch.sum(group_positive * precision, dim=1) / total_positive
        cumulative_negative = total_at_end - positive_at_end
        lower_negative = total_negative[:, None] - cumulative_negative
        roc_batch = torch.sum(group_positive * (lower_negative + 0.5 * group_negative), dim=1) / (
            total_positive * total_negative
        )
        pr_values[first:last] = pr_batch.detach().cpu().numpy()
        roc_values[first:last] = roc_batch.detach().cpu().numpy()
        del (
            total_at_end,
            positive_at_end,
            group_positive,
            group_total,
            group_negative,
            total_positive,
            total_negative,
            precision,
            cumulative_negative,
            lower_negative,
            pr_batch,
            roc_batch,
        )
    return pr_values, roc_values


def compact_bootstrap_metrics(
    labels: NDArray[np.integer],
    probabilities: NDArray[np.floating],
    click_days: NDArray[np.integer],
    *,
    block_days: Literal[1, 3, 7],
    replicates: int = 2_000,
    bootstrap_seed: int = 20260813,
    device: Literal["cpu", "cuda"] = "cpu",
    batch_replicates: int | None = None,
) -> CompactBootstrapMetrics:
    """Recompute all four paired metrics from compact arrays under exact day resampling."""

    target = np.asarray(labels, dtype=np.int8)
    probability = np.asarray(probabilities, dtype=np.float64)
    days = np.asarray(click_days, dtype=np.int64)
    if (
        target.ndim != 1
        or probability.shape != target.shape
        or days.shape != target.shape
        or target.size == 0
        or not np.isin(target, (0, 1)).all()
        or np.unique(target).size != 2
        or not np.isfinite(probability).all()
        or np.any((probability < 0.0) | (probability > 1.0))
    ):
        raise ConsistencyError("Compact bootstrap arrays are malformed")
    observed_days = np.unique(days)
    weights = contiguous_day_bootstrap_weights(
        observed_days,
        block_days=block_days,
        replicates=replicates,
        bootstrap_seed=bootstrap_seed,
    )
    day_codes = days - observed_days[0]
    if np.any(day_codes < 0) or np.any(day_codes >= observed_days.size):
        raise ConsistencyError("Compact bootstrap day coding is invalid")
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    loss = -(target * np.log(clipped) + (1 - target) * np.log1p(-clipped))
    brier = np.square(probability - target)
    day_count = np.bincount(day_codes, minlength=observed_days.size).astype(np.float64)
    day_loss = np.bincount(day_codes, weights=loss, minlength=observed_days.size)
    day_brier = np.bincount(day_codes, weights=brier, minlength=observed_days.size)
    weighted_count = weights @ day_count
    log_replicates = (weights @ day_loss) / weighted_count
    brier_replicates = (weights @ day_brier) / weighted_count
    auc_weights = np.vstack((np.ones((1, observed_days.size), dtype=np.int16), weights))
    pr_auc, roc_auc = _weighted_auc_replicates(
        target,
        probability,
        day_codes,
        auc_weights,
        device=device,
        batch_replicates=(128 if device == "cuda" else 16)
        if batch_replicates is None
        else batch_replicates,
    )
    return CompactBootstrapMetrics(
        block_days=block_days,
        bootstrap_seed=bootstrap_seed,
        day_weights=weights,
        point={
            "log_loss": float(loss.mean()),
            "brier_score": float(brier.mean()),
            "pr_auc": float(pr_auc[0]),
            "roc_auc": float(roc_auc[0]),
        },
        replicates={
            "log_loss": log_replicates,
            "brier_score": brier_replicates,
            "pr_auc": pr_auc[1:],
            "roc_auc": roc_auc[1:],
        },
    )


def paired_compact_interval(
    control: dict[int, CompactBootstrapMetrics],
    candidate: dict[int, CompactBootstrapMetrics],
    *,
    metric: Literal["log_loss", "brier_score", "pr_auc", "roc_auc"],
) -> PairedInterval:
    """Aggregate matched compact replicate arrays across the three final seeds."""

    if set(control) != set(candidate) or len(control) < 3:
        raise ConsistencyError("Compact paired comparison requires the same three or more seeds")
    seed_differences: list[SeedDifference] = []
    replicate_differences: list[NDArray[np.float64]] = []
    block_days: int | None = None
    replicates: int | None = None
    for seed in sorted(control):
        control_result = control[seed]
        candidate_result = candidate[seed]
        if (
            control_result.block_days != candidate_result.block_days
            or control_result.bootstrap_seed != candidate_result.bootstrap_seed
            or not np.array_equal(control_result.day_weights, candidate_result.day_weights)
            or metric not in control_result.point
            or metric not in candidate_result.point
            or metric not in control_result.replicates
            or metric not in candidate_result.replicates
            or control_result.replicates[metric].shape != candidate_result.replicates[metric].shape
        ):
            raise ConsistencyError("Compact paired bootstrap identities do not align")
        if block_days is None:
            block_days = control_result.block_days
            replicates = control_result.replicates[metric].size
        elif (
            block_days != control_result.block_days
            or replicates != control_result.replicates[metric].size
        ):
            raise ConsistencyError("Compact paired bootstrap seeds use different protocols")
        control_point = control_result.point[metric]
        candidate_point = candidate_result.point[metric]
        seed_differences.append(
            SeedDifference(seed, control_point, candidate_point, candidate_point - control_point)
        )
        replicate_differences.append(
            candidate_result.replicates[metric] - control_result.replicates[metric]
        )
    assert block_days is not None and replicates is not None
    seed_averaged = np.mean(np.stack(replicate_differences), axis=0)
    lower, upper = np.quantile(seed_averaged, [0.025, 0.975])
    return PairedInterval(
        metric=metric,
        block_days=block_days,
        replicates=replicates,
        point_difference=float(np.mean([item.difference for item in seed_differences])),
        lower_95=float(lower),
        upper_95=float(upper),
        seed_differences=tuple(seed_differences),
    )


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
