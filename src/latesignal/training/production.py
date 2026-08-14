"""Production-scale deterministic trainers over packed legal records."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from latesignal.data.manifests import canonical_json_bytes
from latesignal.errors import ConfigurationError, ConsistencyError
from latesignal.features.store import FeatureTensorBatch
from latesignal.methods.losses import esdfm_loss, fnw_loss
from latesignal.models.conversion_mlp import CategoricalSpec, ConversionMLP
from latesignal.training.budget import BudgetCounter
from latesignal.training.packed import PackedDeterministicSampler, PackedSample
from latesignal.training.reproducibility import configure_determinism

LossMode = Literal["bce", "fnw", "es_dfm"]


class FeatureProvider(Protocol):
    @property
    def categorical_specs(self) -> dict[str, CategoricalSpec]: ...

    def tensor_batch(self, references: NDArray[np.integer]) -> FeatureTensorBatch: ...


class AuxiliaryLogitProvider(Protocol):
    def logits(self, features: FeatureTensorBatch) -> tuple[Tensor, Tensor]: ...


def _snapshot_state(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_state(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class ProductionTrainingConfig:
    learning_rate: float
    weight_decay: float
    dropout: float
    gradient_norm_clip: float
    steps_per_credit: int
    batch_size: int
    loss_mode: LossMode

    def __post_init__(self) -> None:
        if (
            self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or not 0.0 <= self.dropout < 1.0
            or self.gradient_norm_clip <= 0.0
            or self.steps_per_credit <= 0
            or self.batch_size <= 0
        ):
            raise ValueError("Production training configuration is invalid")

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "dropout": self.dropout,
                    "gradient_norm_clip": self.gradient_norm_clip,
                    "steps_per_credit": self.steps_per_credit,
                    "batch_size": self.batch_size,
                    "loss_mode": self.loss_mode,
                }
            )
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class ExposureCredit:
    credit_id: int
    record_keys: NDArray[np.uint64]
    sources: NDArray[np.uint8]
    weights: NDArray[np.float32]

    @property
    def examples(self) -> int:
        return int(self.record_keys.size)


@dataclass(frozen=True, slots=True)
class CreditTrainingResult:
    credit_id: int
    decision_time: float
    steps: int
    examples: int
    mean_loss: float
    exposure: ExposureCredit


def require_training_device(device: str | torch.device) -> torch.device:
    parsed = torch.device(device)
    if parsed.type not in {"cpu", "cuda"}:
        raise ConfigurationError("Production training supports only CPU or CUDA")
    if parsed.type == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            raise ConfigurationError(
                "CUDA training requires CUBLAS_WORKSPACE_CONFIG before process launch"
            )
        if not torch.cuda.is_available():
            raise ConfigurationError("The required CUDA training device is unavailable")
    return parsed


class PackedConversionTrainer:
    """Train the locked conversion MLP without retaining Python exposure objects."""

    def __init__(
        self,
        model: ConversionMLP,
        features: FeatureProvider,
        config: ProductionTrainingConfig,
        *,
        seed: int,
        device: str | torch.device,
    ) -> None:
        self.device = require_training_device(device)
        self.features = features
        self.config = config
        self.seed = seed
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
        config: ProductionTrainingConfig,
        *,
        seed: int,
        device: str | torch.device,
    ) -> PackedConversionTrainer:
        parsed_device = require_training_device(device)
        configure_determinism(seed)
        model = ConversionMLP(features.categorical_specs, dropout=config.dropout)
        return cls(model, features, config, seed=seed, device=parsed_device)

    def _batch(self, sample: PackedSample) -> tuple[FeatureTensorBatch, Tensor]:
        features = self.features.tensor_batch(sample.feature_refs).to(self.device)
        targets = torch.from_numpy(sample.targets).to(self.device)
        if targets.shape != (self.config.batch_size,):
            raise ConsistencyError("Packed training target batch has an invalid shape")
        return features, targets

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
        auxiliary_provider: AuxiliaryLogitProvider | None = None,
    ) -> CreditTrainingResult:
        if credit_id != self.budget.credits:
            raise ConsistencyError("Production credit ID is not contiguous")
        if (self.config.loss_mode == "es_dfm") != (auxiliary_provider is not None):
            raise ConsistencyError("ES-DFM auxiliary logits do not match the configured loss")
        examples = self.config.steps_per_credit * self.config.batch_size
        exposure_keys = np.empty(examples, dtype=np.uint64)
        exposure_sources = np.empty(examples, dtype=np.uint8)
        exposure_weights = np.empty(examples, dtype=np.float32)
        loss_sum = 0.0
        self._activate_rng()
        self.model.train()
        for step in range(self.config.steps_per_credit):
            sample = sampler.sample(
                simulator_time=decision_time,
                batch_size=self.config.batch_size,
            )
            features, targets = self._batch(sample)
            logits = self.model(features.categorical, features.numeric)
            if self.config.loss_mode == "fnw":
                weighted = fnw_loss(logits, targets)
                loss = weighted.loss
                weights = weighted.weights
            elif self.config.loss_mode == "es_dfm":
                assert auxiliary_provider is not None
                q_tn_logits, q_dp_logits = auxiliary_provider.logits(features)
                if q_tn_logits.shape != targets.shape or q_dp_logits.shape != targets.shape:
                    raise ConsistencyError("ES-DFM auxiliary logits have an invalid shape")
                weighted = esdfm_loss(logits, targets, q_tn_logits, q_dp_logits)
                loss = weighted.loss
                weights = weighted.weights
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
                weights = torch.ones_like(targets)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise ConsistencyError("Production training loss is not a finite scalar")
            if weights.shape != targets.shape or not bool(torch.isfinite(weights).all()):
                raise ConsistencyError("Production exposure weights are invalid")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            gradient_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_norm_clip,
                error_if_nonfinite=True,
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise ConsistencyError("Production training gradient norm is not finite")
            self.optimizer.step()
            start = step * self.config.batch_size
            end = start + self.config.batch_size
            exposure_keys[start:end] = sample.record_keys
            exposure_sources[start:end] = sample.sources
            exposure_weights[start:end] = weights.detach().cpu().numpy()
            loss_sum += float(loss.detach().cpu().item())
            self.model_version += 1
        self._capture_rng()
        self.budget.record_credit(
            steps=self.config.steps_per_credit,
            batch_size=self.config.batch_size,
        )
        exposure = ExposureCredit(
            credit_id=credit_id,
            record_keys=exposure_keys,
            sources=exposure_sources,
            weights=exposure_weights,
        )
        if exposure.examples != examples:
            raise ConsistencyError("Production exposure ledger does not match its credit")
        return CreditTrainingResult(
            credit_id=credit_id,
            decision_time=decision_time,
            steps=self.config.steps_per_credit,
            examples=examples,
            mean_loss=loss_sum / self.config.steps_per_credit,
            exposure=exposure,
        )

    def predict(self, references: NDArray[np.integer]) -> NDArray[np.float32]:
        self.model.eval()
        features = self.features.tensor_batch(references).to(self.device)
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(features.categorical, features.numeric))
        if not bool(torch.isfinite(probabilities).all()):
            raise ConsistencyError("Production prediction contains a nonfinite probability")
        return np.asarray(probabilities.cpu().numpy(), dtype=np.float32)

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "config_sha256": self.config.canonical_sha256,
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
            or state.get("config_sha256") != self.config.canonical_sha256
            or state.get("seed") != self.seed
            or state.get("device_type") != self.device.type
        ):
            raise ConsistencyError("Production trainer checkpoint identity changed")
        model = state.get("model")
        optimizer = state.get("optimizer")
        budget = state.get("budget")
        model_version = state.get("model_version")
        cpu_rng_state = state.get("cpu_rng_state")
        cuda_rng_state = state.get("cuda_rng_state")
        if (
            not isinstance(model, dict)
            or not isinstance(optimizer, dict)
            or not isinstance(budget, dict)
            or isinstance(model_version, bool)
            or not isinstance(model_version, int)
            or model_version < 0
            or not isinstance(cpu_rng_state, Tensor)
            or not isinstance(cuda_rng_state, list)
            or not all(isinstance(value, Tensor) for value in cuda_rng_state)
        ):
            raise ConsistencyError("Production trainer checkpoint state is malformed")
        if self.device.type == "cuda" and len(cuda_rng_state) != torch.cuda.device_count():
            raise ConsistencyError("Production CUDA RNG checkpoint does not match the host")
        if self.device.type == "cpu" and cuda_rng_state:
            raise ConsistencyError("CPU trainer checkpoint unexpectedly contains CUDA RNG state")
        self.model.load_state_dict(model)
        self.optimizer.load_state_dict(optimizer)
        self.budget.load_state_dict(budget)
        self.model_version = model_version
        if self.model_version != self.budget.optimizer_steps:
            raise ConsistencyError("Production model version does not match optimizer steps")
        self._cpu_rng_state = cpu_rng_state.cpu().clone()
        self._cuda_rng_state = [value.cpu().clone() for value in cuda_rng_state]
