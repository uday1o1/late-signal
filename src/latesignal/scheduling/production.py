"""Packed mature-cohort monitoring for production scheduler decisions."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from latesignal.data.manifests import canonical_json_bytes
from latesignal.errors import ConsistencyError
from latesignal.scheduling.monitoring import CalibrationEvidence, ResidualBin
from latesignal.simulator.production_oracle import TruthEventBatch


class MonitoringPredictor(Protocol):
    def predict(self, references: NDArray[np.integer]) -> NDArray[np.float32]: ...


class PackedMonitoringState:
    """Retain only legal reserved labels and compute fixed-bin evidence in batches."""

    def __init__(
        self,
        *,
        click_days: NDArray[np.int16],
        monitoring_mask: NDArray[np.bool_],
        inference_batch_size: int = 65_536,
    ) -> None:
        if (
            click_days.ndim != 1
            or monitoring_mask.shape != click_days.shape
            or click_days.size == 0
            or inference_batch_size <= 0
        ):
            raise ValueError("Packed monitoring feature contract is invalid")
        self.click_days = np.array(click_days, dtype=np.int16, copy=True)
        self.monitoring_mask = np.array(monitoring_mask, dtype=np.bool_, copy=True)
        self.inference_batch_size = inference_batch_size
        self.labels = np.full(click_days.size, -1, dtype=np.int8)
        self.last_decision_day: int | None = None
        self.inference_examples = 0
        self.evidence_log: list[dict[str, object]] = []
        self.membership_sha256 = hashlib.sha256(
            np.packbits(self.monitoring_mask, bitorder="little").tobytes()
        ).hexdigest()
        self.click_days_sha256 = hashlib.sha256(self.click_days.tobytes()).hexdigest()
        self.config_sha256 = hashlib.sha256(
            canonical_json_bytes(
                {
                    "membership_sha256": self.membership_sha256,
                    "click_days_sha256": self.click_days_sha256,
                    "inference_batch_size": inference_batch_size,
                    "maturity_days": 30,
                    "window_days": 7,
                    "bins": 10,
                    "minimum_bin_examples": 1000,
                    "minimum_variance": 25.0,
                    "epsilon": 1e-8,
                }
            )
        ).hexdigest()

    def observe_truth(self, truth: TruthEventBatch) -> None:
        refs = truth.feature_refs
        if (
            refs.ndim != 1
            or truth.labels.shape != refs.shape
            or truth.available_at.shape != refs.shape
            or np.any(refs < 0)
            or np.any(refs >= self.labels.size)
            or np.unique(refs).size != refs.size
            or np.any((truth.labels != 0) & (truth.labels != 1))
        ):
            raise ConsistencyError("Packed monitoring truth batch is malformed")
        selected = refs[self.monitoring_mask[refs]]
        selected_labels = truth.labels[self.monitoring_mask[refs]]
        if np.any(self.labels[selected] != -1):
            raise ConsistencyError("Packed monitoring truth was observed twice")
        self.labels[selected] = selected_labels

    def _references(self, decision_day: int) -> tuple[NDArray[np.int32], int, int]:
        if decision_day <= 30:
            raise ValueError("Packed monitoring begins only after one full maturity window")
        newest_day = decision_day - 31
        first_day = newest_day - 6
        mask = (
            self.monitoring_mask & (self.click_days >= first_day) & (self.click_days <= newest_day)
        )
        refs = np.flatnonzero(mask).astype(np.int32)
        if np.any(self.labels[refs] < 0):
            raise ConsistencyError("Packed monitoring cohort is not fully mature")
        return refs, first_day, newest_day

    def evidence(
        self,
        *,
        decision_day: int,
        predictor: MonitoringPredictor,
        model_checkpoint_sha256: str,
    ) -> CalibrationEvidence:
        if len(model_checkpoint_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in model_checkpoint_sha256
        ):
            raise ConsistencyError("Monitoring model checkpoint digest is invalid")
        expected_day = 31 if self.last_decision_day is None else self.last_decision_day + 1
        if decision_day != expected_day:
            raise ConsistencyError("Packed monitoring decisions must cover every day from D31")
        refs, first_day, newest_day = self._references(decision_day)
        probabilities = np.empty(refs.size, dtype=np.float64)
        for start in range(0, refs.size, self.inference_batch_size):
            end = min(start + self.inference_batch_size, refs.size)
            values = np.asarray(predictor.predict(refs[start:end]), dtype=np.float64)
            if values.shape != (end - start,):
                raise ConsistencyError("Monitoring predictor returned an invalid shape")
            probabilities[start:end] = values
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
        ):
            raise ConsistencyError("Monitoring predictor returned an invalid probability")
        targets = self.labels[refs].astype(np.float64)
        bin_indexes = np.minimum((probabilities * 10).astype(np.int64), 9)
        counts = np.bincount(bin_indexes, minlength=10)
        positives = np.bincount(bin_indexes, weights=targets, minlength=10)
        probability_sums = np.bincount(bin_indexes, weights=probabilities, minlength=10)
        residual_sums = np.bincount(
            bin_indexes,
            weights=targets - probabilities,
            minlength=10,
        )
        variance_sums = np.bincount(
            bin_indexes,
            weights=probabilities * (1.0 - probabilities),
            minlength=10,
        )
        bins: list[ResidualBin] = []
        candidates: list[tuple[float, int]] = []
        for index in range(10):
            eligible = counts[index] >= 1000 and variance_sums[index] >= 25.0
            residual = (
                abs(float(residual_sums[index])) / float(np.sqrt(variance_sums[index] + 1e-8))
                if eligible
                else None
            )
            if residual is not None:
                candidates.append((residual, index))
            bins.append(
                ResidualBin(
                    index=index,
                    count=int(counts[index]),
                    positives=int(positives[index]),
                    probability_sum=float(probability_sums[index]),
                    signed_residual_sum=float(residual_sums[index]),
                    variance_sum=float(variance_sums[index]),
                    eligible=bool(eligible),
                    standardized_residual=residual,
                )
            )
        score: float | None = None
        contributing_bin: int | None = None
        if candidates:
            score, contributing_bin = max(candidates, key=lambda item: (item[0], -item[1]))
        result = CalibrationEvidence(
            decision_day=decision_day,
            model_checkpoint_sha256=model_checkpoint_sha256,
            monitoring_cohort_first_day=first_day,
            monitoring_cohort_last_day=newest_day,
            monitoring_examples=refs.size,
            score=score,
            contributing_bin=contributing_bin,
            bins=tuple(bins),
        )
        self.last_decision_day = decision_day
        self.inference_examples += refs.size
        self.evidence_log.append(result.as_dict())
        return result

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "config_sha256": self.config_sha256,
            "labels": torch.from_numpy(self.labels.copy()),
            "last_decision_day": self.last_decision_day,
            "inference_examples": self.inference_examples,
            "evidence_log": copy.deepcopy(self.evidence_log),
        }

    @staticmethod
    def _validate_evidence_log(
        evidence_log: list[dict[str, object]],
        *,
        last_day: int | None,
        inference_examples: int,
    ) -> bool:
        expected_entries = 0 if last_day is None else last_day - 30
        if expected_entries < 0 or len(evidence_log) != expected_entries:
            return False
        logged_examples = 0
        for offset, entry in enumerate(evidence_log):
            decision_day = entry.get("decision_day")
            examples = entry.get("monitoring_examples")
            first_day = entry.get("monitoring_cohort_first_day")
            newest_day = entry.get("monitoring_cohort_last_day")
            digest = entry.get("model_checkpoint_sha256")
            bins = entry.get("bins")
            if (
                decision_day != 31 + offset
                or isinstance(examples, bool)
                or not isinstance(examples, int)
                or examples < 0
                or first_day != decision_day - 37
                or newest_day != decision_day - 31
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(value not in "0123456789abcdef" for value in digest)
                or not isinstance(bins, list)
                or len(bins) != 10
            ):
                return False
            logged_examples += examples
        return logged_examples == inference_examples

    def load_state_dict(self, state: dict[str, Any]) -> None:
        labels = state.get("labels")
        last_day = state.get("last_decision_day")
        examples = state.get("inference_examples")
        evidence_log = state.get("evidence_log")
        if (
            state.get("version") != 1
            or state.get("config_sha256") != self.config_sha256
            or not isinstance(labels, torch.Tensor)
            or labels.dtype != torch.int8
            or (
                last_day is not None
                and (isinstance(last_day, bool) or not isinstance(last_day, int))
            )
            or isinstance(examples, bool)
            or not isinstance(examples, int)
            or examples < 0
            or not isinstance(evidence_log, list)
            or not all(isinstance(value, dict) for value in evidence_log)
        ):
            raise ConsistencyError("Packed monitoring checkpoint state is malformed")
        parsed_labels = np.asarray(labels.cpu().numpy(), dtype=np.int8)
        if (
            parsed_labels.shape != self.labels.shape
            or np.any(parsed_labels < -1)
            or np.any(parsed_labels > 1)
            or np.any(parsed_labels[~self.monitoring_mask] != -1)
            or not self._validate_evidence_log(
                evidence_log,
                last_day=last_day,
                inference_examples=examples,
            )
        ):
            raise ConsistencyError("Packed monitoring checkpoint arrays are inconsistent")
        self.labels = parsed_labels
        self.last_decision_day = last_day
        self.inference_examples = examples
        self.evidence_log = copy.deepcopy(evidence_log)
