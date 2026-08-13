"""Strict configuration for the synthetic Study A qualification path."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from latesignal.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class StudyAConfig:
    version: int
    mode: str
    seed: int
    attribution_seconds: int
    wait_seconds: int
    credits: int
    initialization_steps: int
    steps_per_credit: int
    batch_size: int
    recent_window_seconds: int
    reservoir_capacity: int
    learning_rate: float
    weight_decay: float
    gradient_norm_clip: float
    auxiliary_initialization_steps: int
    auxiliary_later_steps: int
    canonical_sha256: str

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("canonical_sha256")
        return value


_KEYS = {
    "version",
    "mode",
    "seed",
    "attribution_seconds",
    "wait_seconds",
    "credits",
    "initialization_steps",
    "steps_per_credit",
    "batch_size",
    "recent_window_seconds",
    "reservoir_capacity",
    "learning_rate",
    "weight_decay",
    "gradient_norm_clip",
    "auxiliary_initialization_steps",
    "auxiliary_later_steps",
}


def _positive_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{key} must be a positive integer")
    return value


def _number(raw: dict[str, Any], key: str, *, allow_zero: bool = False) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{key} must be numeric")
    parsed = float(value)
    if parsed < 0.0 or (not allow_zero and parsed == 0.0):
        raise ConfigurationError(f"{key} must be {'nonnegative' if allow_zero else 'positive'}")
    return parsed


def parse_study_a_config(value: object) -> StudyAConfig:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError("Study A configuration must be a mapping")
    raw: dict[str, Any] = value
    if set(raw) != _KEYS:
        raise ConfigurationError(
            "Study A configuration has invalid keys",
            details={"missing": sorted(_KEYS - set(raw)), "unknown": sorted(set(raw) - _KEYS)},
        )
    if raw["version"] != 1 or raw["mode"] != "synthetic-study-a":
        raise ConfigurationError("Only synthetic Study A configuration version 1 is supported")
    seed = raw["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigurationError("seed must be an integer")
    attribution_seconds = _positive_int(raw, "attribution_seconds")
    wait_seconds = _positive_int(raw, "wait_seconds")
    if wait_seconds >= attribution_seconds:
        raise ConfigurationError("wait_seconds must be shorter than attribution_seconds")
    parsed: dict[str, object] = {
        "version": 1,
        "mode": "synthetic-study-a",
        "seed": seed,
        "attribution_seconds": attribution_seconds,
        "wait_seconds": wait_seconds,
        "credits": _positive_int(raw, "credits"),
        "initialization_steps": _positive_int(raw, "initialization_steps"),
        "steps_per_credit": _positive_int(raw, "steps_per_credit"),
        "batch_size": _positive_int(raw, "batch_size"),
        "recent_window_seconds": _positive_int(raw, "recent_window_seconds"),
        "reservoir_capacity": _positive_int(raw, "reservoir_capacity"),
        "learning_rate": _number(raw, "learning_rate"),
        "weight_decay": _number(raw, "weight_decay", allow_zero=True),
        "gradient_norm_clip": _number(raw, "gradient_norm_clip"),
        "auxiliary_initialization_steps": _positive_int(raw, "auxiliary_initialization_steps"),
        "auxiliary_later_steps": _positive_int(raw, "auxiliary_later_steps"),
    }
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return StudyAConfig(
        **parsed,  # type: ignore[arg-type]
        canonical_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
    )


def load_study_a_config(path: Path) -> StudyAConfig:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"Could not read Study A configuration: {path}") from error
    return parse_study_a_config(value)
