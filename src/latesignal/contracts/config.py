"""Strict synthetic vertical-slice configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from latesignal.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    learning_rate: float
    steps_per_credit: int


@dataclass(frozen=True, slots=True)
class SyntheticRunConfig:
    version: int
    mode: str
    seed: int
    boundary_seconds: int
    decision_interval_seconds: int
    maturity_seconds: int
    training: TrainingConfig
    canonical_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "mode": self.mode,
            "seed": self.seed,
            "boundary_seconds": self.boundary_seconds,
            "decision_interval_seconds": self.decision_interval_seconds,
            "maturity_seconds": self.maturity_seconds,
            "training": {
                "learning_rate": self.training.learning_rate,
                "steps_per_credit": self.training.steps_per_credit,
            },
        }


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def parse_synthetic_config(raw: object) -> SyntheticRunConfig:
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise ConfigurationError("Synthetic configuration must be a mapping")
    required = {
        "version",
        "mode",
        "seed",
        "boundary_seconds",
        "decision_interval_seconds",
        "maturity_seconds",
        "training",
    }
    if set(raw) != required:
        raise ConfigurationError(
            "Synthetic configuration has invalid keys",
            details={
                "missing": sorted(required - set(raw)),
                "unknown": sorted(set(raw) - required),
            },
        )
    if raw["version"] != 1 or raw["mode"] != "synthetic":
        raise ConfigurationError("Only synthetic configuration version 1 is supported")
    if isinstance(raw["seed"], bool) or not isinstance(raw["seed"], int):
        raise ConfigurationError("seed must be an integer")
    training = raw["training"]
    if not isinstance(training, dict) or set(training) != {
        "learning_rate",
        "steps_per_credit",
    }:
        raise ConfigurationError("training has invalid keys")
    learning_rate = training["learning_rate"]
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not 0.0 < learning_rate <= 1.0
    ):
        raise ConfigurationError("training.learning_rate must lie in (0, 1]")
    canonical_mapping = {
        "version": 1,
        "mode": "synthetic",
        "seed": raw["seed"],
        "boundary_seconds": _positive_int(raw["boundary_seconds"], "boundary_seconds"),
        "decision_interval_seconds": _positive_int(
            raw["decision_interval_seconds"], "decision_interval_seconds"
        ),
        "maturity_seconds": _positive_int(raw["maturity_seconds"], "maturity_seconds"),
        "training": {
            "learning_rate": float(learning_rate),
            "steps_per_credit": _positive_int(
                training["steps_per_credit"], "training.steps_per_credit"
            ),
        },
    }
    if canonical_mapping["decision_interval_seconds"] % canonical_mapping["boundary_seconds"]:
        raise ConfigurationError("decision_interval_seconds must align to boundary_seconds")
    canonical = json.dumps(canonical_mapping, sort_keys=True, separators=(",", ":"))
    return SyntheticRunConfig(
        version=1,
        mode="synthetic",
        seed=int(raw["seed"]),
        boundary_seconds=int(canonical_mapping["boundary_seconds"]),
        decision_interval_seconds=int(canonical_mapping["decision_interval_seconds"]),
        maturity_seconds=int(canonical_mapping["maturity_seconds"]),
        training=TrainingConfig(
            learning_rate=float(learning_rate),
            steps_per_credit=int(canonical_mapping["training"]["steps_per_credit"]),
        ),
        canonical_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
    )


def load_synthetic_config(path: Path) -> SyntheticRunConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"Could not read run configuration: {path}") from error
    return parse_synthetic_config(raw)
