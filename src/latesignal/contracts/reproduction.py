"""Strict clean-checkout synthetic reproduction manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from latesignal.contracts.protocol import StrictModel
from latesignal.data.manifests import read_json
from latesignal.errors import ConfigurationError


class ExpectedCounts(StrictModel):
    predictions: int = Field(ge=0)
    available_records: int = Field(ge=0)
    credits: int = Field(ge=0)
    optimizer_steps: int = Field(ge=0)
    optimizer_examples: int = Field(ge=0)
    checkpoints: int = Field(ge=0)


class ExpectedMetrics(StrictModel):
    count: int = Field(gt=0)
    positives: int = Field(ge=0)
    log_loss: float = Field(ge=0.0)
    brier_score: float = Field(ge=0.0, le=1.0)


class ExpectedSyntheticResult(StrictModel):
    ledger_sha256: dict[
        Literal["availability", "credits", "events", "exposures", "predictions"],
        str,
    ]
    counts: ExpectedCounts
    metrics: ExpectedMetrics


class ReproductionManifest(StrictModel):
    version: Literal[1]
    kind: Literal["synthetic-run"]
    config: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependency_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected: ExpectedSyntheticResult


def load_reproduction_manifest(path: Path) -> ReproductionManifest:
    try:
        manifest = ReproductionManifest.model_validate(read_json(path))
    except ValidationError as error:
        raise ConfigurationError(
            "Reproduction manifest validation failed",
            details={
                "errors": error.errors(
                    include_context=False,
                    include_input=False,
                    include_url=False,
                )
            },
        ) from error
    if set(manifest.expected.ledger_sha256) != {
        "availability",
        "credits",
        "events",
        "exposures",
        "predictions",
    }:
        raise ConfigurationError("Reproduction manifest has an incomplete ledger hash set")
    return manifest
