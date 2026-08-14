"""Strict aggregate report input contracts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from latesignal.contracts.protocol import StrictModel
from latesignal.data.manifests import read_json
from latesignal.errors import ConfigurationError


class DatasetSummary(StrictModel):
    name: str = Field(min_length=1)
    license_id: Literal["CC-BY-NC-SA-4.0", "synthetic"]
    source_archive_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    preparation_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    accepted_rows: int = Field(ge=0)
    quarantined_rows: int = Field(ge=0)


class ProtocolSummary(StrictModel):
    lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_commit: str = Field(min_length=7)
    environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seeds: list[int] = Field(min_length=1)
    publication_eligible: bool


class ReliabilitySummary(StrictModel):
    index: int = Field(ge=0, le=9)
    lower: float = Field(ge=0.0, le=1.0)
    upper: float = Field(ge=0.0, le=1.0)
    count: int = Field(ge=0)
    positives: int = Field(ge=0)
    mean_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    observed_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class MetricSummary(StrictModel):
    count: int = Field(gt=0)
    positives: int = Field(ge=0)
    log_loss: float = Field(ge=0.0)
    brier_score: float = Field(ge=0.0, le=1.0)
    pr_auc: float | None = Field(default=None, ge=0.0, le=1.0)
    roc_auc: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_intercept: float | None = None
    calibration_slope: float | None = None
    expected_calibration_error: float = Field(ge=0.0, le=1.0)
    reliability: list[ReliabilitySummary] = Field(min_length=10, max_length=10)

    @model_validator(mode="after")
    def finite_metrics(self) -> MetricSummary:
        values = (
            self.log_loss,
            self.brier_score,
            self.pr_auc,
            self.roc_auc,
            self.calibration_intercept,
            self.calibration_slope,
            self.expected_calibration_error,
        )
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("Report metrics must be finite")
        if self.positives > self.count:
            raise ValueError("Report positive count cannot exceed total count")
        if [item.index for item in self.reliability] != list(range(10)):
            raise ValueError("Report reliability bins must be ordered from zero through nine")
        if sum(item.count for item in self.reliability) != self.count:
            raise ValueError("Report reliability-bin counts must reconcile")
        return self


class EvaluationSummary(StrictModel):
    method: str = Field(min_length=1)
    seed: int
    ranking_eligible: bool
    metrics: MetricSummary


class SliceSummary(StrictModel):
    method: str = Field(min_length=1)
    seed: int
    dimension: str = Field(min_length=1)
    value: str = Field(min_length=1)
    count: int = Field(ge=0)
    positives: int = Field(ge=0)
    ranking_eligible: bool
    suppression_reason: str | None
    log_loss: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def suppression_is_explicit(self) -> SliceSummary:
        if self.positives > self.count:
            raise ValueError("Slice positive count cannot exceed total count")
        if self.ranking_eligible == (self.suppression_reason is not None):
            raise ValueError("Slice eligibility and suppression reason disagree")
        if self.ranking_eligible != (self.log_loss is not None):
            raise ValueError("Only eligible slices may report a metric")
        return self


class SeedDifferenceSummary(StrictModel):
    seed: int
    difference: float


class PairedSummary(StrictModel):
    control: str = Field(min_length=1)
    candidate: str = Field(min_length=1)
    metric: Literal["log_loss", "brier_score", "pr_auc", "roc_auc"]
    block_days: Literal[1, 3, 7]
    replicates: int = Field(ge=2_000)
    point_difference: float
    lower_95: float
    upper_95: float
    seed_differences: list[SeedDifferenceSummary] = Field(min_length=1)

    @model_validator(mode="after")
    def interval_is_ordered(self) -> PairedSummary:
        values = (self.point_difference, self.lower_95, self.upper_95)
        if not all(math.isfinite(value) for value in values) or self.lower_95 > self.upper_95:
            raise ValueError("Paired interval is nonfinite or reversed")
        return self


class MethodBudgetSummary(StrictModel):
    method: str = Field(min_length=1)
    deployable: bool
    ranking_eligible: bool
    credits: int = Field(ge=0)
    core_optimizer_steps: int = Field(ge=0)
    core_optimizer_examples: int = Field(ge=0)
    auxiliary_optimizer_steps: int = Field(ge=0)
    auxiliary_optimizer_examples: int = Field(ge=0)
    status: Literal["complete", "incomplete", "failed"]


class SchedulerSummary(StrictModel):
    scheduler: str = Field(min_length=1)
    seed: int
    credits: int = Field(ge=0)
    optimizer_steps: int = Field(ge=0)
    optimizer_examples: int = Field(ge=0)
    monitoring_examples: int = Field(ge=0)
    monitoring_exposure_overlap: Literal[0]
    trigger_days: list[int]
    status: Literal["complete", "incomplete", "failed"]


class BudgetQualitySummary(StrictModel):
    method: str = Field(min_length=1)
    budget_fraction: Literal[0.25, 0.5, 0.75, 1.0]  # type: ignore[valid-type]
    core_examples: int = Field(ge=0)
    log_loss: float = Field(ge=0.0)


class ComputeSummary(StrictModel):
    method: str = Field(min_length=1)
    log_loss: float = Field(ge=0.0)
    core_examples: int = Field(ge=0)
    total_examples: int = Field(ge=0)
    wall_seconds: float = Field(ge=0.0)
    peak_memory_gb: float = Field(ge=0.0)
    pareto_efficient: bool


class LeakageAuditSummary(StrictModel):
    control: str = Field(min_length=1)
    status: Literal["passed", "failed"]
    evidence: str = Field(min_length=1)


class ClaimSummary(StrictModel):
    scheduler_outcome: Literal["supported", "negative_or_inconclusive", "not_evaluated"]
    published_number_reproduction: Literal[False]
    statement: str = Field(min_length=1)


class ReportInput(StrictModel):
    version: Literal[1]
    title: str = Field(min_length=1)
    result_kind: Literal["synthetic", "final"]
    dataset: DatasetSummary
    protocol: ProtocolSummary
    methods: list[MethodBudgetSummary]
    schedulers: list[SchedulerSummary]
    evaluations: list[EvaluationSummary]
    slices: list[SliceSummary]
    paired_intervals: list[PairedSummary]
    intermediate_budget: list[BudgetQualitySummary]
    compute: list[ComputeSummary]
    leakage_audit: list[LeakageAuditSummary]
    limitations: list[str] = Field(min_length=1)
    reproduction_commands: list[str] = Field(min_length=1)
    claim: ClaimSummary

    @model_validator(mode="after")
    def final_publication_gate(self) -> ReportInput:
        if len(self.protocol.seeds) != len(set(self.protocol.seeds)):
            raise ValueError("Report seeds must be unique")
        if self.result_kind == "synthetic":
            if self.dataset.license_id != "synthetic":
                raise ValueError("Synthetic reports must use the synthetic dataset identity")
            return self
        if self.dataset.license_id != "CC-BY-NC-SA-4.0":
            raise ValueError("Final reports must retain the dataset license identity")
        if set(self.protocol.seeds) != {17, 41, 73} or not self.protocol.publication_eligible:
            raise ValueError(
                "Final reports require the locked seeds and a publication-eligible lock"
            )
        if any(item.status != "complete" for item in self.methods) or any(
            item.status != "complete" for item in self.schedulers
        ):
            raise ValueError("Final reports cannot describe incomplete runs as complete evidence")
        required_methods = {
            "complete_wait",
            "immediate_fake_negative",
            "fixed_wait",
            "dfm",
            "fnw",
            "es_dfm",
            "oracle_reference",
        }
        required_schedulers = {
            "fixed_early",
            "fixed_midpoint",
            "fixed_deadline",
            "calibration_drift",
        }
        if {item.method for item in self.methods} != required_methods:
            raise ValueError("Final report does not contain the complete method suite")
        if {item.scheduler for item in self.schedulers} != required_schedulers:
            raise ValueError("Final report does not contain the complete scheduler suite")
        required_slices = {
            "cold_user",
            "cold_product",
            "user_frequency",
            "product_frequency",
            "product_price_bin",
            "device_type",
            "positive_conversion_delay",
            "click_day_block",
        }
        if not required_slices.issubset({item.dimension for item in self.slices}):
            raise ValueError("Final report does not contain every required slice dimension")
        required_leakage_controls = {
            "forbidden_sale_field",
            "forbidden_conversion_delay_field",
            "final_period_normalizer_fit",
            "global_cold_status",
            "reveal_before_prediction",
            "monitoring_training_reuse",
            "early_truth_availability",
        }
        passed_controls = {item.control for item in self.leakage_audit if item.status == "passed"}
        if not required_leakage_controls.issubset(passed_controls):
            raise ValueError("Final report does not contain every passing leakage mutation")
        for method in {item.method for item in self.intermediate_budget}:
            fractions = {
                item.budget_fraction for item in self.intermediate_budget if item.method == method
            }
            if fractions != {0.25, 0.5, 0.75, 1.0}:
                raise ValueError("Final report has incomplete intermediate-budget evidence")
        return self


def load_report_input(path: Path) -> ReportInput:
    try:
        return ReportInput.model_validate(read_json(path))
    except ValidationError as error:
        raise ConfigurationError(
            "Aggregate report input validation failed",
            details={
                "errors": error.errors(
                    include_context=False,
                    include_input=False,
                    include_url=False,
                )
            },
        ) from error
