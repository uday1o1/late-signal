"""Prediction-time-safe final-period slice evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import cast

from latesignal.contracts.results import EvaluationExample
from latesignal.evaluation.metrics import classification_metrics


@dataclass(frozen=True, slots=True)
class SliceResult:
    dimension: str
    value: str
    count: int
    positives: int
    ranking_eligible: bool
    suppression_reason: str | None
    metrics: dict[str, object] | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def frequency_band(count: int) -> str:
    if count < 0:
        raise ValueError("Frequency count must be nonnegative")
    if count == 0:
        return "0"
    if count <= 2:
        return "1-2"
    if count <= 9:
        return "3-9"
    return "10+"


def delay_band(delay_days: float) -> str:
    if not 0.0 <= delay_days <= 30.0:
        raise ValueError("Conversion delay must lie in [0, 30]")
    if delay_days < 1.0:
        return "[0,1)"
    if delay_days < 3.0:
        return "[1,3)"
    if delay_days < 7.0:
        return "[3,7)"
    if delay_days < 14.0:
        return "[7,14)"
    return "[14,30]"


def click_day_block(click_day: int, *, first_day: int = 65, block_days: int = 5) -> str:
    start = first_day + ((click_day - first_day) // block_days) * block_days
    return f"{start}-{min(start + block_days - 1, 89)}"


def evaluate_slices(
    examples: tuple[EvaluationExample, ...],
    *,
    minimum_examples: int = 10_000,
    minimum_positives: int = 100,
    declared_device_types: tuple[str, ...] = (),
    declared_price_bins: tuple[str, ...] = (),
) -> list[SliceResult]:
    """Evaluate required slices while retaining empty and suppressed support rows."""

    groups: dict[tuple[str, str], list[EvaluationExample]] = {}

    def add(dimension: str, value: str, example: EvaluationExample) -> None:
        groups.setdefault((dimension, value), []).append(example)

    for example in examples:
        add("cold_user", str(example.cold_user).lower(), example)
        add("cold_product", str(example.cold_product).lower(), example)
        add("user_frequency", frequency_band(example.prior_user_clicks), example)
        add("product_frequency", frequency_band(example.prior_product_clicks), example)
        add("product_price_bin", example.product_price_bin, example)
        add("device_type", example.device_type, example)
        add("click_day_block", click_day_block(example.click_day), example)
        if example.final_label == 1 and example.conversion_delay_days is not None:
            add("positive_conversion_delay", delay_band(example.conversion_delay_days), example)
    for value in ("false", "true"):
        groups.setdefault(("cold_user", value), [])
        groups.setdefault(("cold_product", value), [])
    for dimension in ("user_frequency", "product_frequency"):
        for value in ("0", "1-2", "3-9", "10+"):
            groups.setdefault((dimension, value), [])
    for value in ("[0,1)", "[1,3)", "[3,7)", "[7,14)", "[14,30]"):
        groups.setdefault(("positive_conversion_delay", value), [])
    for value in declared_device_types:
        groups.setdefault(("device_type", value), [])
    for value in declared_price_bins:
        groups.setdefault(("product_price_bin", value), [])

    results: list[SliceResult] = []
    for (dimension, value), rows in sorted(groups.items()):
        count = len(rows)
        positives = sum(row.final_label for row in rows)
        eligible = count >= minimum_examples and positives >= minimum_positives
        if count == 0:
            reason = "empty"
        elif count < minimum_examples:
            reason = "insufficient_examples"
        elif positives < minimum_positives:
            reason = "insufficient_positives"
        else:
            reason = None
        metrics: dict[str, object] | None = None
        if eligible:
            metrics = cast(
                dict[str, object],
                classification_metrics(
                    [row.final_label for row in rows],
                    [row.probability for row in rows],
                ),
            )
        results.append(
            SliceResult(
                dimension=dimension,
                value=value,
                count=count,
                positives=positives,
                ranking_eligible=eligible,
                suppression_reason=reason,
                metrics=metrics,
            )
        )
    return results
