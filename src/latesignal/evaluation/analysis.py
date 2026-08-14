"""Intermediate-budget and compute-Pareto result tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise

from latesignal.errors import ConsistencyError


@dataclass(frozen=True, slots=True)
class BudgetQualityPoint:
    method: str
    budget_fraction: float
    core_examples: int
    log_loss: float


@dataclass(frozen=True, slots=True)
class ComputePoint:
    method: str
    log_loss: float
    core_examples: int
    total_examples: int
    pareto_efficient: bool


def validate_intermediate_budget(points: tuple[BudgetQualityPoint, ...]) -> list[dict[str, object]]:
    required = {0.25, 0.5, 0.75, 1.0}
    methods = {point.method for point in points}
    for method in methods:
        fractions = {point.budget_fraction for point in points if point.method == method}
        if fractions != required:
            raise ConsistencyError(f"Intermediate budget checkpoints are incomplete for {method}")
        ordered = sorted(
            (point for point in points if point.method == method),
            key=lambda point: point.budget_fraction,
        )
        if any(
            later.core_examples <= earlier.core_examples for earlier, later in pairwise(ordered)
        ):
            raise ConsistencyError("Intermediate core-example budgets must increase strictly")
    return [
        asdict(point)
        for point in sorted(points, key=lambda item: (item.method, item.budget_fraction))
    ]


def compute_pareto_table(
    values: tuple[tuple[str, float, int, int], ...],
) -> list[dict[str, object]]:
    """Mark methods undominated on lower log loss and lower total examples."""

    if any(core < 0 or total < core for _, _, core, total in values):
        raise ConsistencyError("Compute table contains an invalid example count")
    result: list[ComputePoint] = []
    for method, log_loss, core, total in values:
        dominated = any(
            other_loss <= log_loss
            and other_total <= total
            and (other_loss < log_loss or other_total < total)
            for other_method, other_loss, _, other_total in values
            if other_method != method
        )
        result.append(ComputePoint(method, log_loss, core, total, not dominated))
    return [
        asdict(item)
        for item in sorted(result, key=lambda item: (item.total_examples, item.log_loss))
    ]
