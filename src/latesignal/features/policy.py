"""Strict authored feature policy and training-batch allowlist."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from latesignal.data.schema import CATEGORICAL_CLICK_FIELDS, NUMERIC_CLICK_FIELDS
from latesignal.errors import ConfigurationError, ConsistencyError


@dataclass(frozen=True, slots=True)
class FeaturePolicy:
    version: int
    field_seed: int
    policy: str
    high_cardinality_fields: frozenset[str]
    high_cardinality_buckets: int
    other_categorical_buckets: int
    burn_in_last_day: int
    numeric_lower_quantile: float
    numeric_upper_quantile: float
    canonical_sha256: str

    @property
    def model_columns(self) -> frozenset[str]:
        categorical = {f"{field}_bucket" for field in CATEGORICAL_CLICK_FIELDS}
        numeric = {f"{field}_value" for field in NUMERIC_CLICK_FIELDS}
        missing = {f"{field}_missing" for field in NUMERIC_CLICK_FIELDS}
        return frozenset(categorical | numeric | missing)

    def bucket_count(self, field_name: str) -> int:
        if field_name not in CATEGORICAL_CLICK_FIELDS:
            raise ConsistencyError(f"Unknown categorical feature: {field_name}")
        if field_name in self.high_cardinality_fields:
            return self.high_cardinality_buckets
        return self.other_categorical_buckets

    def validate_training_columns(self, columns: Collection[str]) -> None:
        supplied = set(columns)
        forbidden = sorted(supplied - self.model_columns)
        missing = sorted(self.model_columns - supplied)
        if forbidden or missing:
            raise ConsistencyError(
                "Training batch violates the click-time feature allowlist",
                details={"forbidden": forbidden, "missing": missing},
            )


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{name} must be an integer of at least {minimum}")
    return value


def _fraction(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ConfigurationError(f"{name} must lie in [0, 1]")
    return result


def load_feature_policy(path: Path) -> FeaturePolicy:
    try:
        raw_object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"Could not read feature policy: {path}") from error
    if not isinstance(raw_object, dict) or not all(isinstance(key, str) for key in raw_object):
        raise ConfigurationError("Feature policy must be a mapping")
    raw: dict[str, Any] = raw_object
    required = {
        "version",
        "field_seed",
        "policy",
        "high_cardinality_fields",
        "high_cardinality_buckets",
        "other_categorical_buckets",
        "burn_in_last_day",
        "numeric_lower_quantile",
        "numeric_upper_quantile",
    }
    if set(raw) != required:
        raise ConfigurationError(
            "Feature policy has invalid keys",
            details={
                "missing": sorted(required - set(raw)),
                "unknown": sorted(set(raw) - required),
            },
        )
    if raw["version"] != 1 or raw["policy"] not in {"compact", "large"}:
        raise ConfigurationError("Unsupported feature policy version or name")
    high_raw = raw["high_cardinality_fields"]
    if not isinstance(high_raw, list) or not all(isinstance(item, str) for item in high_raw):
        raise ConfigurationError("high_cardinality_fields must be a string list")
    high = frozenset(high_raw)
    if not high.issubset(CATEGORICAL_CLICK_FIELDS):
        raise ConfigurationError("high_cardinality_fields contains an unknown field")
    lower = _fraction(raw["numeric_lower_quantile"], "numeric_lower_quantile")
    upper = _fraction(raw["numeric_upper_quantile"], "numeric_upper_quantile")
    if lower >= upper:
        raise ConfigurationError("Numeric quantile bounds must be strictly increasing")
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return FeaturePolicy(
        version=1,
        field_seed=_integer(raw["field_seed"], "field_seed"),
        policy=str(raw["policy"]),
        high_cardinality_fields=high,
        high_cardinality_buckets=_integer(
            raw["high_cardinality_buckets"], "high_cardinality_buckets", minimum=1
        ),
        other_categorical_buckets=_integer(
            raw["other_categorical_buckets"], "other_categorical_buckets", minimum=1
        ),
        burn_in_last_day=_integer(raw["burn_in_last_day"], "burn_in_last_day"),
        numeric_lower_quantile=lower,
        numeric_upper_quantile=upper,
        canonical_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
    )
