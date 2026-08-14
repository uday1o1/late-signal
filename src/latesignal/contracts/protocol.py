"""Strict Pydantic contracts for protocol and final-matrix feasibility."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from latesignal.errors import ConfigurationError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class SelectionDefaults(StrictModel):
    model_method: Literal["complete_wait"]
    recent_window_days: Literal[1, 3, 7]
    reservoir_capacity: Literal[1_000_000, 5_000_000]
    training_last_click_day: Literal[24]
    scoring_first_click_day: Literal[25]
    scoring_last_click_day: Literal[34]
    first_credit_day: Literal[55]
    last_credit_day: Literal[64]


class ModelSelection(StrictModel):
    learning_rates: list[float] = Field(min_length=1)
    weight_decays: list[float] = Field(min_length=1)
    dropouts: list[float] = Field(min_length=1)
    feature_policies: list[Literal["compact", "large"]] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    credits_per_run: int = Field(gt=0)


class DelayedSelection(StrictModel):
    methods: list[Literal["fixed_wait", "es_dfm"]] = Field(min_length=1)
    wait_days: list[Literal[1, 3, 7, 14]] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    credits_per_run: int = Field(gt=0)


class SamplerSelection(StrictModel):
    recent_window_days: list[Literal[1, 3, 7]] = Field(min_length=1)
    reservoir_capacities: list[Literal[1_000_000, 5_000_000]] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    credits_per_run: int = Field(gt=0)


class FinalTraining(StrictModel):
    seeds: list[int] = Field(min_length=3)
    steps_per_credit_candidates: list[Literal[100, 250, 500]] = Field(min_length=1)
    batch_size: int = Field(gt=0)
    study_a_credits: int = Field(gt=0)
    study_b_credits: int = Field(gt=0)
    bootstrap_replicates: int = Field(ge=2_000)
    bootstrap_primary_block_days: Literal[3]
    bootstrap_sensitivity_block_days: list[Literal[1, 7]] = Field(min_length=2, max_length=2)


class ProtocolDefinition(StrictModel):
    version: Literal[1]
    selection_defaults: SelectionDefaults
    model_selection: ModelSelection
    delayed_selection: DelayedSelection
    sampler_selection: SamplerSelection
    final_training: FinalTraining

    @model_validator(mode="after")
    def unique_authored_values(self) -> ProtocolDefinition:
        defaults = self.selection_defaults
        selection_credit_count = defaults.last_credit_day - defaults.first_credit_day + 1
        if selection_credit_count != 10:
            raise ValueError("Selection credit boundaries must contain exactly ten days")
        if any(
            stage.credits_per_run != selection_credit_count
            for stage in (
                self.model_selection,
                self.delayed_selection,
                self.sampler_selection,
            )
        ):
            raise ValueError("Every selection stage must use the authored credit boundaries")
        lists: tuple[Sequence[object], ...] = (
            self.model_selection.learning_rates,
            self.model_selection.weight_decays,
            self.model_selection.dropouts,
            self.model_selection.feature_policies,
            self.model_selection.seeds,
            self.delayed_selection.methods,
            self.delayed_selection.wait_days,
            self.delayed_selection.seeds,
            self.sampler_selection.recent_window_days,
            self.sampler_selection.reservoir_capacities,
            self.sampler_selection.seeds,
            self.final_training.seeds,
            self.final_training.steps_per_credit_candidates,
        )
        if any(len(values) != len(set(values)) for values in lists):
            raise ValueError("Protocol candidate lists must not contain duplicates")
        expected: tuple[tuple[str, set[object], set[object]], ...] = (
            (
                "model learning rates",
                set(self.model_selection.learning_rates),
                {0.0001, 0.0003, 0.001},
            ),
            (
                "model weight decays",
                set(self.model_selection.weight_decays),
                {0.0, 0.00001, 0.0001},
            ),
            ("model dropouts", set(self.model_selection.dropouts), {0.0, 0.1}),
            (
                "feature policies",
                set(self.model_selection.feature_policies),
                {"compact", "large"},
            ),
            ("delayed methods", set(self.delayed_selection.methods), {"fixed_wait", "es_dfm"}),
            ("delayed wait days", set(self.delayed_selection.wait_days), {1, 3, 7, 14}),
            (
                "sampler windows",
                set(self.sampler_selection.recent_window_days),
                {1, 3, 7},
            ),
            (
                "sampler capacities",
                set(self.sampler_selection.reservoir_capacities),
                {1_000_000, 5_000_000},
            ),
            (
                "steps per credit",
                set(self.final_training.steps_per_credit_candidates),
                {100, 250, 500},
            ),
        )
        for name, actual, locked in expected:
            if actual != locked:
                raise ValueError(f"Protocol {name} must match the locked candidate set")
        for name, seeds in (
            ("model selection", self.model_selection.seeds),
            ("delayed selection", self.delayed_selection.seeds),
            ("sampler selection", self.sampler_selection.seeds),
        ):
            if seeds != [17]:
                raise ValueError(f"{name} seed must be exactly 17")
        if set(self.final_training.seeds) != {17, 41, 73}:
            raise ValueError("Final training seeds must be exactly 17, 41, and 73")
        return self


class PilotConfig(StrictModel):
    prepared_root: str
    max_click_days: int = Field(gt=0, le=2)
    benchmark_examples: int = Field(gt=0, le=100_000)
    benchmark_batch_size: int = Field(gt=0, le=2048)
    benchmark_steps: int = Field(gt=0, le=100)
    assumed_source_archive_gb: float = Field(gt=0)
    assumed_expanded_source_gb: float = Field(gt=0)
    assumed_prepared_data_gb: float = Field(gt=0)
    temporary_margin_gb: float = Field(gt=0)
    assumed_host_memory_gb: float = Field(gt=0)


class ResourceCaps(StrictModel):
    max_runs: int | None
    max_gpu_hours: float | None
    max_working_disk_gb: float | None
    max_retained_disk_gb: float | None

    @model_validator(mode="after")
    def positive_when_present(self) -> ResourceCaps:
        if self.max_runs is not None and self.max_runs <= 0:
            raise ValueError("max_runs must be positive")
        for name in ("max_gpu_hours", "max_working_disk_gb", "max_retained_disk_gb"):
            value = getattr(self, name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive")
        return self

    @property
    def complete(self) -> bool:
        return all(
            value is not None
            for value in (
                self.max_runs,
                self.max_gpu_hours,
                self.max_working_disk_gb,
                self.max_retained_disk_gb,
            )
        )


MethodName = Literal[
    "complete_wait",
    "immediate_fake_negative",
    "fixed_wait",
    "dfm",
    "fnw",
    "es_dfm",
    "oracle_reference",
]
SchedulerName = Literal[
    "fixed_early",
    "fixed_midpoint",
    "fixed_deadline",
    "calibration_drift",
]
OfflineName = Literal["mature_logistic_regression", "mature_lightgbm"]


class FinalExperimentConfig(StrictModel):
    version: Literal[1]
    mode: Literal["final"]
    protocol: str
    methods: list[MethodName] = Field(min_length=1)
    schedulers: list[SchedulerName] = Field(min_length=1)
    offline_references: list[OfflineName] = Field(min_length=1)
    target_device: Literal["cpu", "cuda"]
    require_real_pilot: bool
    pilot: PilotConfig
    caps: ResourceCaps

    @model_validator(mode="after")
    def unique_matrix_axes(self) -> FinalExperimentConfig:
        if any(
            len(values) != len(set(values))
            for values in (self.methods, self.schedulers, self.offline_references)
        ):
            raise ValueError("Final matrix axes must not contain duplicates")
        if set(self.methods) != {
            "complete_wait",
            "immediate_fake_negative",
            "fixed_wait",
            "dfm",
            "fnw",
            "es_dfm",
            "oracle_reference",
        }:
            raise ValueError("Final method axis must contain the complete locked V1 suite")
        if set(self.schedulers) != {
            "fixed_early",
            "fixed_midpoint",
            "fixed_deadline",
            "calibration_drift",
        }:
            raise ValueError("Final scheduler axis must contain all locked policies")
        if set(self.offline_references) != {
            "mature_logistic_regression",
            "mature_lightgbm",
        }:
            raise ValueError("Final offline-reference axis must contain both locked references")
        return self


def _load_yaml(path: Path) -> object:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"Could not read authored configuration: {path}") from error


def load_final_protocol(path: Path) -> tuple[FinalExperimentConfig, ProtocolDefinition, str]:
    try:
        final = FinalExperimentConfig.model_validate(_load_yaml(path))
        protocol_path = (path.parent / final.protocol).resolve()
        protocol = ProtocolDefinition.model_validate(_load_yaml(protocol_path))
    except ValidationError as error:
        raise ConfigurationError(
            "Authored protocol validation failed",
            details={
                "errors": error.errors(
                    include_context=False,
                    include_input=False,
                    include_url=False,
                )
            },
        ) from error
    if (
        final.require_real_pilot
        and final.pilot.benchmark_batch_size != protocol.final_training.batch_size
    ):
        raise ConfigurationError(
            "Final throughput benchmark must use the locked training batch size",
            details={
                "benchmark_batch_size": final.pilot.benchmark_batch_size,
                "training_batch_size": protocol.final_training.batch_size,
            },
        )
    canonical = json.dumps(
        {"final": final.model_dump(mode="json"), "protocol": protocol.model_dump(mode="json")},
        sort_keys=True,
        separators=(",", ":"),
    )
    return final, protocol, hashlib.sha256(canonical.encode()).hexdigest()
