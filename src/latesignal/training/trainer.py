"""Deterministic legal-record trainer for the synthetic vertical slice."""

from __future__ import annotations

from latesignal.contracts.records import CreditRecord, ExposureRecord, TrainingRecord
from latesignal.models.tiny import TinyLogisticModel


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
