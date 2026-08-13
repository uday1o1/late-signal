"""Deterministic delayed-label fixture with exact boundary cases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from latesignal.contracts.config import SyntheticRunConfig
from latesignal.contracts.events import ClickEvent, TruthRecord


@dataclass(frozen=True, slots=True)
class SyntheticFixture:
    clicks: tuple[ClickEvent, ...]
    truth: tuple[TruthRecord, ...]
    canonical_sha256: str


def build_synthetic_fixture(config: SyntheticRunConfig) -> SyntheticFixture:
    day = config.decision_interval_seconds
    maturity = config.maturity_seconds
    clicks = (
        ClickEvent("click-000", 0, 1.0),
        ClickEvent("click-001", config.boundary_seconds - 1, -1.0),
        ClickEvent("click-002", config.boundary_seconds, 0.5),
        ClickEvent("click-003", day + 1, 1.5),
        ClickEvent("click-004", 2 * day - 1, -1.5),
        ClickEvent("click-005", 2 * day + 1, -0.5),
    )
    truth = (
        TruthRecord("click-000", 1, config.boundary_seconds),
        TruthRecord("click-001", 0, config.boundary_seconds - 1 + maturity),
        TruthRecord("click-002", 1, config.boundary_seconds),
        TruthRecord("click-003", 1, day + config.boundary_seconds),
        TruthRecord("click-004", 0, 2 * day - 1 + maturity),
        TruthRecord("click-005", 0, 2 * day + 1 + maturity),
    )
    payload = {
        "clicks": [click.as_dict() for click in clicks],
        "truth": [record.as_dict() for record in truth],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return SyntheticFixture(
        clicks=clicks,
        truth=truth,
        canonical_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
    )
