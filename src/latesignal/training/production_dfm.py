"""Production trainer for the Delayed Feedback Model transfer."""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from latesignal.data.manifests import canonical_json_bytes
from latesignal.errors import ConsistencyError
from latesignal.features.store import FeatureTensorBatch
from latesignal.methods.losses import dfm_loss
from latesignal.methods.production import DFMObservationBatch
from latesignal.models.conversion_mlp import ConversionMLP
from latesignal.models.dfm import DelayedFeedbackMLP
from latesignal.training.budget import BudgetCounter
from latesignal.training.packed import PackedDeterministicSampler
from latesignal.training.production import (
    CreditTrainingResult,
    ExposureCredit,
    FeatureProvider,
    ProductionTrainingConfig,
    _snapshot_state,
    require_training_device,
)
from latesignal.training.reproducibility import configure_determinism


class DFMObservationProvider(Protocol):
    def dfm_observations(
        self,
        refs: NDArray[np.int32],
        *,
        simulator_time: float,
    ) -> DFMObservationBatch: ...


class PackedDFMTrainer:
    """Train conversion and exponential-delay heads under the matched core budget."""

    def __init__(
        self,
        model: DelayedFeedbackMLP,
        features: FeatureProvider,
        observations: DFMObservationProvider,
        config: ProductionTrainingConfig,
        *,
        seed: int,
        device: str | torch.device,
    ) -> None:
        if config.loss_mode != "bce":
            raise ValueError("DFM uses its authored censored likelihood, not a weighted BCE mode")
        self.device = require_training_device(device)
        self.features = features
        self.observations = observations
        self.config = config
        self.seed = seed
        self.config_sha256 = hashlib.sha256(
            canonical_json_bytes({"trainer": "dfm", "training_sha256": config.canonical_sha256})
        ).hexdigest()
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.budget = BudgetCounter()
        self.model_version = 0
        self._cpu_rng_state = torch.get_rng_state().clone()
        self._cuda_rng_state = (
            [value.clone() for value in torch.cuda.get_rng_state_all()]
            if self.device.type == "cuda"
            else []
        )

    @classmethod
    def create(
        cls,
        features: FeatureProvider,
        observations: DFMObservationProvider,
        config: ProductionTrainingConfig,
        *,
        seed: int,
        device: str | torch.device,
    ) -> PackedDFMTrainer:
        parsed_device = require_training_device(device)
        configure_determinism(seed)
        conversion = ConversionMLP(features.categorical_specs, dropout=config.dropout)
        return cls(
            DelayedFeedbackMLP(conversion),
            features,
            observations,
            config,
            seed=seed,
            device=parsed_device,
        )

    def _activate_rng(self) -> None:
        torch.set_rng_state(self._cpu_rng_state)
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(self._cuda_rng_state)

    def _capture_rng(self) -> None:
        self._cpu_rng_state = torch.get_rng_state().clone()
        if self.device.type == "cuda":
            self._cuda_rng_state = [value.clone() for value in torch.cuda.get_rng_state_all()]

    def spend_credit(
        self,
        *,
        credit_id: int,
        decision_time: float,
        sampler: PackedDeterministicSampler,
    ) -> CreditTrainingResult:
        if credit_id != self.budget.credits:
            raise ConsistencyError("DFM credit ID is not contiguous")
        example_count = self.config.steps_per_credit * self.config.batch_size
        exposure_keys = np.empty(example_count, dtype=np.uint64)
        exposure_sources = np.empty(example_count, dtype=np.uint8)
        exposure_weights = np.ones(example_count, dtype=np.float32)
        loss_sum = 0.0
        self._activate_rng()
        self.model.train()
        for step in range(self.config.steps_per_credit):
            sample = sampler.sample(
                simulator_time=decision_time,
                batch_size=self.config.batch_size,
            )
            features: FeatureTensorBatch = self.features.tensor_batch(sample.feature_refs).to(
                self.device
            )
            observation = self.observations.dfm_observations(
                sample.feature_refs,
                simulator_time=decision_time,
            )
            targets = torch.from_numpy(observation.targets).to(self.device)
            time_days = torch.from_numpy(observation.time_days).to(self.device)
            if targets.shape != (self.config.batch_size,) or time_days.shape != targets.shape:
                raise ConsistencyError("DFM observation batch has an invalid shape")
            conversion_logits, rate_logits = self.model(features.categorical, features.numeric)
            loss = dfm_loss(conversion_logits, rate_logits, targets, time_days)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise ConsistencyError("DFM training loss is not a finite scalar")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            gradient_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_norm_clip,
                error_if_nonfinite=True,
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise ConsistencyError("DFM training gradient norm is not finite")
            self.optimizer.step()
            start = step * self.config.batch_size
            end = start + self.config.batch_size
            exposure_keys[start:end] = sample.record_keys
            exposure_sources[start:end] = sample.sources
            loss_sum += float(loss.detach().cpu().item())
            self.model_version += 1
        self._capture_rng()
        self.budget.record_credit(
            steps=self.config.steps_per_credit,
            batch_size=self.config.batch_size,
        )
        return CreditTrainingResult(
            credit_id=credit_id,
            decision_time=decision_time,
            steps=self.config.steps_per_credit,
            examples=example_count,
            mean_loss=loss_sum / self.config.steps_per_credit,
            exposure=ExposureCredit(
                credit_id=credit_id,
                record_keys=exposure_keys,
                sources=exposure_sources,
                weights=exposure_weights,
            ),
        )

    def predict(self, references: NDArray[np.integer]) -> NDArray[np.float32]:
        self.model.eval()
        features = self.features.tensor_batch(references).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(features.categorical, features.numeric)
            probabilities = torch.sigmoid(logits)
        if not bool(torch.isfinite(probabilities).all()):
            raise ConsistencyError("DFM prediction contains a nonfinite probability")
        return np.asarray(probabilities.cpu().numpy(), dtype=np.float32)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "config_sha256": self.config_sha256,
            "seed": self.seed,
            "device_type": self.device.type,
            "model": _snapshot_state(self.model.state_dict()),
            "optimizer": _snapshot_state(self.optimizer.state_dict()),
            "budget": self.budget.state_dict(),
            "model_version": self.model_version,
            "cpu_rng_state": self._cpu_rng_state.clone(),
            "cuda_rng_state": [value.clone() for value in self._cuda_rng_state],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (
            state.get("version") != 1
            or state.get("config_sha256") != self.config_sha256
            or state.get("seed") != self.seed
            or state.get("device_type") != self.device.type
        ):
            raise ConsistencyError("DFM trainer checkpoint identity changed")
        model = state.get("model")
        optimizer = state.get("optimizer")
        budget = state.get("budget")
        model_version = state.get("model_version")
        cpu_rng = state.get("cpu_rng_state")
        cuda_rng = state.get("cuda_rng_state")
        if (
            not isinstance(model, dict)
            or not isinstance(optimizer, dict)
            or not isinstance(budget, dict)
            or isinstance(model_version, bool)
            or not isinstance(model_version, int)
            or model_version < 0
            or not isinstance(cpu_rng, Tensor)
            or not isinstance(cuda_rng, list)
            or not all(isinstance(value, Tensor) for value in cuda_rng)
        ):
            raise ConsistencyError("DFM trainer checkpoint state is malformed")
        if self.device.type == "cuda" and len(cuda_rng) != torch.cuda.device_count():
            raise ConsistencyError("DFM CUDA RNG checkpoint does not match the host")
        if self.device.type == "cpu" and cuda_rng:
            raise ConsistencyError("CPU DFM checkpoint unexpectedly contains CUDA RNG state")
        self.model.load_state_dict(model)
        self.optimizer.load_state_dict(optimizer)
        self.budget.load_state_dict(budget)
        self.model_version = model_version
        if self.model_version != self.budget.optimizer_steps:
            raise ConsistencyError("DFM model version does not match optimizer steps")
        self._cpu_rng_state = cpu_rng.cpu().clone()
        self._cuda_rng_state = [value.cpu().clone() for value in cuda_rng]
