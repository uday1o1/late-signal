"""Strict synthetic Study B qualification configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from latesignal.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class StudyBConfig:
    version: int
    mode: str
    seed: int
    window_days: int
    steps_per_credit: int
    batch_size: int
    recent_window_seconds: int
    reservoir_capacity: int
    learning_rate: float
    monitor_seed: int
    monitoring_examples_per_day: int
    shift_click_day: int
    threshold: float
    canonical_sha256: str

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("canonical_sha256")
        return value


_KEYS = {
    "version",
    "mode",
    "seed",
    "window_days",
    "steps_per_credit",
    "batch_size",
    "recent_window_seconds",
    "reservoir_capacity",
    "learning_rate",
    "monitor_seed",
    "monitoring_examples_per_day",
    "shift_click_day",
    "threshold",
}


def _integer(raw: dict[str, Any], key: str, *, minimum: int = 1) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{key} must be an integer of at least {minimum}")
    return value


def _positive_number(raw: dict[str, Any], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigurationError(f"{key} must be positive")
    return float(value)


def parse_study_b_config(value: object) -> StudyBConfig:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError("Study B configuration must be a mapping")
    raw: dict[str, Any] = value
    if set(raw) != _KEYS:
        raise ConfigurationError(
            "Study B configuration has invalid keys",
            details={"missing": sorted(_KEYS - set(raw)), "unknown": sorted(set(raw) - _KEYS)},
        )
    if raw["version"] != 1 or raw["mode"] != "synthetic-study-b":
        raise ConfigurationError("Only synthetic Study B configuration version 1 is supported")
    seed = raw["seed"]
    monitor_seed = raw["monitor_seed"]
    if any(isinstance(item, bool) or not isinstance(item, int) for item in (seed, monitor_seed)):
        raise ConfigurationError("Study B seeds must be integers")
    parsed: dict[str, object] = {
        "version": 1,
        "mode": "synthetic-study-b",
        "seed": seed,
        "window_days": _integer(raw, "window_days"),
        "steps_per_credit": _integer(raw, "steps_per_credit"),
        "batch_size": _integer(raw, "batch_size"),
        "recent_window_seconds": _integer(raw, "recent_window_seconds"),
        "reservoir_capacity": _integer(raw, "reservoir_capacity"),
        "learning_rate": _positive_number(raw, "learning_rate"),
        "monitor_seed": monitor_seed,
        "monitoring_examples_per_day": _integer(raw, "monitoring_examples_per_day", minimum=1_000),
        "shift_click_day": _integer(raw, "shift_click_day", minimum=0),
        "threshold": _positive_number(raw, "threshold"),
    }
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return StudyBConfig(
        **parsed,  # type: ignore[arg-type]
        canonical_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
    )


def load_study_b_config(path: Path) -> StudyBConfig:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"Could not read Study B configuration: {path}") from error
    return parse_study_b_config(value)
