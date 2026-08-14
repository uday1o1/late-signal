"""Strict selection-period result contracts for protocol freezing."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from latesignal.contracts.protocol import StrictModel
from latesignal.errors import ConfigurationError

Digest = str


class SelectionWindow(StrictModel):
    first_click_day: Literal[25]
    last_click_day: Literal[34]
    all_labels_mature_by_day: Literal[64]
    embargo_outcomes_accessed: Literal[False]
    final_period_metrics_accessed: Literal[False]


class CandidateResult(StrictModel):
    config_sha256: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["complete", "failed", "incomplete"]
    mean_selection_log_loss: float | None
    measured_compute_seconds: float | None = Field(ge=0.0)
    parameter_count: int | None = Field(ge=0)
    failure_reason: str | None

    @model_validator(mode="after")
    def result_fields_match_status(self) -> CandidateResult:
        if self.status == "complete":
            values = (
                self.mean_selection_log_loss,
                self.measured_compute_seconds,
                self.parameter_count,
            )
            if any(value is None for value in values) or self.failure_reason is not None:
                raise ValueError("A complete candidate requires metrics and no failure reason")
            assert self.mean_selection_log_loss is not None
            assert self.measured_compute_seconds is not None
            if not math.isfinite(self.mean_selection_log_loss) or not math.isfinite(
                self.measured_compute_seconds
            ):
                raise ValueError("Complete candidate metrics must be finite")
        elif not self.failure_reason:
            raise ValueError("A failed or incomplete candidate requires a failure reason")
        return self


class ModelCandidate(CandidateResult):
    learning_rate: float
    weight_decay: float
    dropout: float
    feature_policy: Literal["compact", "large"]
    seed: Literal[17]


class DelayedCandidate(CandidateResult):
    method: Literal["fixed_wait", "es_dfm"]
    wait_days: Literal[1, 3, 7, 14]
    seed: Literal[17]


class SamplerCandidate(CandidateResult):
    recent_window_days: Literal[1, 3, 7]
    reservoir_capacity: Literal[1_000_000, 5_000_000]
    seed: Literal[17]


class SelectionResults(StrictModel):
    version: Literal[1]
    protocol_sha256: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    window: SelectionWindow
    model_candidates: list[ModelCandidate]
    delayed_candidates: list[DelayedCandidate]
    sampler_candidates: list[SamplerCandidate]

    @model_validator(mode="after")
    def exhaustive_candidate_grids(self) -> SelectionResults:
        model_keys = {
            (
                item.learning_rate,
                item.weight_decay,
                item.dropout,
                item.feature_policy,
                item.seed,
            )
            for item in self.model_candidates
        }
        expected_model = {
            (learning_rate, weight_decay, dropout, feature_policy, 17)
            for learning_rate in (0.0001, 0.0003, 0.001)
            for weight_decay in (0.0, 0.00001, 0.0001)
            for dropout in (0.0, 0.1)
            for feature_policy in ("compact", "large")
        }
        delayed_keys = {
            (item.method, item.wait_days, item.seed) for item in self.delayed_candidates
        }
        expected_delayed = {
            (method, wait_days, 17)
            for method in ("fixed_wait", "es_dfm")
            for wait_days in (1, 3, 7, 14)
        }
        sampler_keys = {
            (item.recent_window_days, item.reservoir_capacity, item.seed)
            for item in self.sampler_candidates
        }
        expected_sampler = {
            (window, capacity, 17) for window in (1, 3, 7) for capacity in (1_000_000, 5_000_000)
        }
        checks = (
            ("model", model_keys, expected_model, len(self.model_candidates)),
            ("delayed", delayed_keys, expected_delayed, len(self.delayed_candidates)),
            ("sampler", sampler_keys, expected_sampler, len(self.sampler_candidates)),
        )
        for name, actual, expected, authored_count in checks:
            if actual != expected or authored_count != len(expected):
                raise ValueError(f"Selection results do not contain the exhaustive {name} grid")
        digests = [
            item.config_sha256
            for item in (
                *self.model_candidates,
                *self.delayed_candidates,
                *self.sampler_candidates,
            )
        ]
        if len(digests) != len(set(digests)):
            raise ValueError("Selection candidate configuration hashes must be unique")
        return self


def load_selection_results(path: Path) -> SelectionResults:
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
        return SelectionResults.model_validate(value)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise ConfigurationError(f"Could not read selection results: {path}") from error
    except ValidationError as error:
        raise ConfigurationError(
            "Selection results validation failed",
            details={
                "errors": error.errors(
                    include_context=False,
                    include_input=False,
                    include_url=False,
                )
            },
        ) from error
