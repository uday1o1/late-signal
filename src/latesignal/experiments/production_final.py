"""Immutable run plans for the locked production final studies."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic import Field, model_validator

from latesignal.contracts.protocol import ProtocolDefinition, StrictModel
from latesignal.contracts.selection import DelayedCandidate, ModelCandidate, SamplerCandidate
from latesignal.data.manifests import canonical_json_bytes
from latesignal.errors import ConsistencyError

FinalStudy = Literal["study_a", "study_b"]
FinalMethod = Literal[
    "complete_wait",
    "immediate_fake_negative",
    "fixed_wait",
    "dfm",
    "fnw",
    "es_dfm",
    "oracle_reference",
]
FinalScheduler = Literal[
    "fixed_daily",
    "fixed_early",
    "fixed_midpoint",
    "fixed_deadline",
    "calibration_drift",
]
FeaturePolicyName = Literal["compact", "large"]

_STUDY_A_METHODS: tuple[FinalMethod, ...] = (
    "complete_wait",
    "immediate_fake_negative",
    "fixed_wait",
    "dfm",
    "fnw",
    "es_dfm",
    "oracle_reference",
)
_STUDY_B_SCHEDULERS: tuple[FinalScheduler, ...] = (
    "fixed_early",
    "fixed_midpoint",
    "fixed_deadline",
    "calibration_drift",
)
_FINAL_SEEDS = (17, 41, 73)


class ProductionFinalPlan(StrictModel):
    """One canonical online run derived only from the verified protocol lock."""

    version: Literal[1]
    phase: Literal["qualification", "final"]
    study: FinalStudy
    run_id: str = Field(pattern=r"^(study-a|study-b)-[0-9a-f]{16}$")
    method: FinalMethod
    scheduler: FinalScheduler
    seed: int
    wait_days: Literal[1, 3, 7, 14] | None
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    dropout: float = Field(ge=0.0, lt=1.0)
    gradient_norm_clip: float = Field(gt=0.0)
    initialization_steps: int = Field(gt=0)
    steps_per_credit: int = Field(gt=0)
    credits: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    recent_window_days: Literal[1, 3, 7]
    reservoir_capacity: Literal[1_000_000, 5_000_000]
    feature_policy: FeaturePolicyName
    prediction_batch_size: int = Field(gt=0, le=65_536)
    first_decision_day: Literal[31]
    last_decision_day: Literal[89]
    evaluation_first_click_day: Literal[65]
    evaluation_last_click_day: Literal[89]
    intermediate_budget_fractions: tuple[float, ...]
    deployable: bool
    ranking_eligible: bool
    device: Literal["cpu", "cuda"]
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_decisions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def authored_final_contract(self) -> ProductionFinalPlan:
        needs_wait = self.method in {"fixed_wait", "es_dfm"}
        if needs_wait != (self.wait_days is not None):
            raise ValueError("Final delayed method and shared wait duration do not align")
        if self.gradient_norm_clip != 5.0:
            raise ValueError("Final gradient clipping changed")
        if self.intermediate_budget_fractions != (0.25, 0.5, 0.75, 1.0):
            raise ValueError("Final intermediate budget fractions changed")
        oracle = self.method == "oracle_reference"
        if oracle == self.deployable or oracle == self.ranking_eligible:
            raise ValueError("The oracle must be excluded from deployable ranking")
        if self.study == "study_a":
            if self.scheduler != "fixed_daily" or self.credits != 59:
                raise ValueError("Study A requires the fixed 59-credit daily schedule")
        elif (
            self.scheduler not in _STUDY_B_SCHEDULERS
            or self.method not in {"fixed_wait", "es_dfm"}
            or self.credits != 12
            or oracle
        ):
            raise ValueError("Study B requires the selected delayed method and 12-credit policy")
        if self.phase == "final" and (
            self.seed not in _FINAL_SEEDS
            or self.initialization_steps != 500
            or self.steps_per_credit not in {100, 250, 500}
            or self.batch_size != 2048
            or self.prediction_batch_size != 65_536
            or self.device != "cuda"
        ):
            raise ValueError("Publication final plan violates the authored training protocol")
        return self

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()


@dataclass(frozen=True, slots=True)
class FinalPlanInputs:
    protocol: ProtocolDefinition
    protocol_sha256: str
    protocol_lock: dict[str, Any]
    feature_policy_sha256: dict[FeaturePolicyName, str]

    def locked_decisions(
        self,
    ) -> tuple[ModelCandidate, DelayedCandidate, SamplerCandidate, str, str, int]:
        lock = self.protocol_lock
        decisions = lock.get("selection_decisions")
        data = lock.get("data")
        if (
            lock.get("status") != "locked"
            or lock.get("locked_before_final_scoring") is not True
            or lock.get("protocol_sha256") != self.protocol_sha256
            or not isinstance(decisions, dict)
            or not isinstance(data, dict)
            or set(self.feature_policy_sha256) != {"compact", "large"}
        ):
            raise ConsistencyError("Final plan inputs do not match one verified protocol lock")
        try:
            model = ModelCandidate.model_validate(decisions.get("model"))
            delayed = DelayedCandidate.model_validate(decisions.get("delayed"))
            sampler = SamplerCandidate.model_validate(decisions.get("sampler"))
        except ValueError as error:
            raise ConsistencyError("Final selection decisions are malformed") from error
        derived = decisions.get("derived")
        data_sha256 = data.get("manifest_sha256")
        lock_sha256 = lock.get("lock_sha256")
        steps = lock.get("selected_steps_per_credit")
        if (
            model.status != "complete"
            or delayed.status != "complete"
            or sampler.status != "complete"
            or not isinstance(derived, dict)
            or derived
            != {
                "shared_wait_days": delayed.wait_days,
                "study_b_method": delayed.method,
            }
            or lock.get("final_seeds") != list(_FINAL_SEEDS)
            or steps not in self.protocol.final_training.steps_per_credit_candidates
            or not isinstance(lock_sha256, str)
            or not isinstance(data_sha256, str)
        ):
            raise ConsistencyError("Final protocol lock has inconsistent selection decisions")
        hashes = {
            self.protocol_sha256,
            lock_sha256,
            data_sha256,
            *self.feature_policy_sha256.values(),
        }
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise ConsistencyError("Final protocol lock contains an invalid content identity")
        return model, delayed, sampler, lock_sha256, data_sha256, cast(int, steps)


def _run_id(study: FinalStudy, semantic: dict[str, object]) -> str:
    digest = hashlib.sha256(canonical_json_bytes(semantic)).hexdigest()
    prefix = "study-a" if study == "study_a" else "study-b"
    return f"{prefix}-{digest[:16]}"


def final_online_plans(inputs: FinalPlanInputs) -> tuple[ProductionFinalPlan, ...]:
    """Expand the lock into the exact 21 plus 12 final online run matrix."""

    model, delayed, sampler, lock_sha256, data_sha256, steps = inputs.locked_decisions()
    decisions = inputs.protocol_lock["selection_decisions"]
    assert isinstance(decisions, dict)
    decisions_sha256 = hashlib.sha256(canonical_json_bytes(decisions)).hexdigest()
    shared: dict[str, object] = {
        "version": 1,
        "phase": "final",
        "learning_rate": model.learning_rate,
        "weight_decay": model.weight_decay,
        "dropout": model.dropout,
        "gradient_norm_clip": 5.0,
        "initialization_steps": inputs.protocol.final_training.initialization_steps,
        "steps_per_credit": steps,
        "batch_size": inputs.protocol.final_training.batch_size,
        "recent_window_days": sampler.recent_window_days,
        "reservoir_capacity": sampler.reservoir_capacity,
        "feature_policy": model.feature_policy,
        "prediction_batch_size": 65_536,
        "first_decision_day": 31,
        "last_decision_day": 89,
        "evaluation_first_click_day": 65,
        "evaluation_last_click_day": 89,
        "intermediate_budget_fractions": (0.25, 0.5, 0.75, 1.0),
        "device": "cuda",
        "protocol_sha256": inputs.protocol_sha256,
        "protocol_lock_sha256": lock_sha256,
        "selection_decisions_sha256": decisions_sha256,
        "data_manifest_sha256": data_sha256,
        "feature_policy_sha256": inputs.feature_policy_sha256[model.feature_policy],
    }
    plans: list[ProductionFinalPlan] = []
    for method in _STUDY_A_METHODS:
        wait_days = delayed.wait_days if method in {"fixed_wait", "es_dfm"} else None
        for seed in _FINAL_SEEDS:
            semantic = {
                **shared,
                "study": "study_a",
                "method": method,
                "scheduler": "fixed_daily",
                "seed": seed,
                "wait_days": wait_days,
                "credits": inputs.protocol.final_training.study_a_credits,
                "deployable": method != "oracle_reference",
                "ranking_eligible": method != "oracle_reference",
            }
            plans.append(
                ProductionFinalPlan.model_validate(
                    {**semantic, "run_id": _run_id("study_a", semantic)}
                )
            )
    for scheduler in _STUDY_B_SCHEDULERS:
        for seed in _FINAL_SEEDS:
            semantic = {
                **shared,
                "study": "study_b",
                "method": delayed.method,
                "scheduler": scheduler,
                "seed": seed,
                "wait_days": delayed.wait_days,
                "credits": inputs.protocol.final_training.study_b_credits,
                "deployable": True,
                "ranking_eligible": True,
            }
            plans.append(
                ProductionFinalPlan.model_validate(
                    {**semantic, "run_id": _run_id("study_b", semantic)}
                )
            )
    if len(plans) != 33 or len({plan.canonical_sha256 for plan in plans}) != 33:
        raise ConsistencyError("Final online matrix is not the exact authored 21 plus 12 DAG")
    return tuple(plans)
