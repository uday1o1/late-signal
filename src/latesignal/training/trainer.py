"""Deterministic legal-record trainers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from latesignal.contracts.records import CreditRecord, ExposureRecord, TrainingRecord
from latesignal.errors import ConsistencyError
from latesignal.models.conversion_mlp import ConversionMLP
from latesignal.models.tiny import TinyLogisticModel
from latesignal.training.budget import BudgetCounter
from latesignal.training.sampler import DeterministicSampler


@dataclass(frozen=True, slots=True)
class ModelBatch:
    categorical: dict[str, Tensor]
    numeric: Tensor
    targets: Tensor
    weights: Tensor
    record_ids: tuple[str, ...]

    def to(self, device: torch.device) -> ModelBatch:
        return ModelBatch(
            categorical={field: values.to(device) for field, values in self.categorical.items()},
            numeric=self.numeric.to(device),
            targets=self.targets.to(device),
            weights=self.weights.to(device),
            record_ids=self.record_ids,
        )


BatchEncoder = Callable[[tuple[TrainingRecord, ...]], ModelBatch]


class MLPTrainer:
    """Shared AdamW trainer with exact budget and exposure accounting."""

    def __init__(
        self,
        model: ConversionMLP,
        *,
        learning_rate: float,
        weight_decay: float,
        gradient_norm_clip: float,
        steps_per_credit: int,
        batch_size: int,
        encoder: BatchEncoder,
        device: str | torch.device = "cpu",
    ) -> None:
        if learning_rate <= 0.0 or weight_decay < 0.0 or gradient_norm_clip <= 0.0:
            raise ValueError("Optimizer settings are invalid")
        if steps_per_credit <= 0 or batch_size <= 0:
            raise ValueError("Training budget settings must be positive")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.gradient_norm_clip = gradient_norm_clip
        self.steps_per_credit = steps_per_credit
        self.batch_size = batch_size
        self.encoder = encoder
        self.budget = BudgetCounter()
        self.exposures: list[ExposureRecord] = []

    def spend_credit(
        self,
        *,
        credit_id: int,
        decision_time: int,
        sampler: DeterministicSampler,
    ) -> CreditRecord:
        self.model.train()
        for step in range(self.steps_per_credit):
            sampled = sampler.sample(simulator_time=decision_time, batch_size=self.batch_size)
            records = tuple(item.record for item in sampled)
            batch = self.encoder(records).to(self.device)
            expected_ids = tuple(record.record_id for record in records)
            if batch.record_ids != expected_ids:
                raise ConsistencyError("Batch encoder changed sampled record order or identity")
            if batch.targets.shape != (self.batch_size,) or batch.weights.shape != (
                self.batch_size,
            ):
                raise ConsistencyError("Batch encoder returned invalid target or weight shapes")
            logits = self.model(batch.categorical, batch.numeric)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits,
                batch.targets,
                reduction="none",
            )
            weight_sum = batch.weights.sum()
            if not bool(torch.isfinite(weight_sum)) or float(weight_sum.item()) <= 0.0:
                raise ConsistencyError("Training batch has no finite positive total weight")
            loss = (losses * batch.weights).sum() / weight_sum
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_norm_clip)
            self.optimizer.step()
            self.exposures.extend(
                ExposureRecord(
                    credit_id=credit_id,
                    step=step,
                    record_id=record.record_id,
                    weight=record.weight,
                )
                for record in records
            )
        self.budget.record_credit(steps=self.steps_per_credit, batch_size=self.batch_size)
        self.budget.assert_exposures(len(self.exposures))
        return CreditRecord(
            credit_id=credit_id,
            decision_time=decision_time,
            steps=self.steps_per_credit,
            examples=self.steps_per_credit * self.batch_size,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "budget": self.budget.state_dict(),
            "exposures": [record.as_dict() for record in self.exposures],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        model = state.get("model")
        optimizer = state.get("optimizer")
        budget = state.get("budget")
        exposures = state.get("exposures")
        if not isinstance(model, dict) or not isinstance(optimizer, dict):
            raise ConsistencyError("Trainer checkpoint model or optimizer is malformed")
        if not isinstance(budget, dict) or not isinstance(exposures, list):
            raise ConsistencyError("Trainer checkpoint ledger state is malformed")
        self.model.load_state_dict(model)
        self.optimizer.load_state_dict(optimizer)
        self.budget.load_state_dict(budget)
        self.exposures = []
        for value in exposures:
            if not isinstance(value, dict):
                raise ConsistencyError("Trainer checkpoint exposure is malformed")
            self.exposures.append(ExposureRecord.from_dict(value))
        self.budget.assert_exposures(len(self.exposures))


class TinyTrainer:
    def __init__(
        self,
        model: TinyLogisticModel,
        *,
        learning_rate: float,
        steps_per_credit: int,
    ) -> None:
        self.model = model
        self.learning_rate = learning_rate
        self.steps_per_credit = steps_per_credit

    def spend_credit(
        self,
        credit_id: int,
        decision_time: int,
        records: tuple[TrainingRecord, ...],
    ) -> tuple[CreditRecord, list[ExposureRecord]]:
        for record in records:
            record.assert_available(decision_time)
        exposures: list[ExposureRecord] = []
        for step in range(self.steps_per_credit):
            self.model.train_step(records, self.learning_rate)
            exposures.extend(
                ExposureRecord(
                    credit_id=credit_id,
                    step=step,
                    record_id=record.record_id,
                    weight=record.weight,
                )
                for record in records
            )
        credit = CreditRecord(
            credit_id=credit_id,
            decision_time=decision_time,
            steps=self.steps_per_credit,
            examples=self.steps_per_credit * len(records),
        )
        return credit, exposures
