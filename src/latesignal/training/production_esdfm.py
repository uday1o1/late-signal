"""Production ES-DFM auxiliary models and locked update sequence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from latesignal.data.manifests import canonical_json_bytes
from latesignal.errors import ConsistencyError
from latesignal.features.store import FeatureTensorBatch
from latesignal.models.conversion_mlp import ConversionMLP
from latesignal.training.packed import PackedDeterministicSampler
from latesignal.training.production import FeatureProvider, _snapshot_state, require_training_device
from latesignal.training.reproducibility import configure_determinism

AuxiliaryRole = Literal["q_tn", "q_dp"]


@dataclass(frozen=True, slots=True)
class AuxiliaryWorkResult:
    role: AuxiliaryRole
    work_id: int
    steps: int
    examples: int
    mean_loss: float
    record_keys: NDArray[np.uint64]
    sources: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class ESDFMUpdateResult:
    credit_id: int
    q_tn: AuxiliaryWorkResult
    q_dp: AuxiliaryWorkResult

    @property
    def auxiliary_steps(self) -> int:
        return self.q_tn.steps + self.q_dp.steps

    @property
    def auxiliary_examples(self) -> int:
        return self.q_tn.examples + self.q_dp.examples


class PackedAuxiliaryTrainer:
    """Train one locked ES-DFM auxiliary probability model."""

    def __init__(
        self,
        role: AuxiliaryRole,
        model: ConversionMLP,
        features: FeatureProvider,
        *,
        seed: int,
        dropout: float,
        batch_size: int,
        device: str | torch.device,
    ) -> None:
        if batch_size <= 0 or not 0.0 <= dropout < 1.0:
            raise ValueError("ES-DFM auxiliary configuration is invalid")
        self.role = role
        self.device = require_training_device(device)
        self.features = features
        self.seed = seed
        self.dropout = dropout
        self.batch_size = batch_size
        self.config_sha256 = hashlib.sha256(
            canonical_json_bytes(
                {
                    "role": role,
                    "seed": seed,
                    "dropout": dropout,
                    "batch_size": batch_size,
                    "optimizer": "AdamW",
                    "learning_rate": 0.0003,
                    "weight_decay": 0.0001,
                    "gradient_norm_clip": 5.0,
                    "initial_steps": 500,
                    "later_steps": 100,
                }
            )
        ).hexdigest()
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.0003,
            weight_decay=0.0001,
        )
        self.work_units = 0
        self.optimizer_steps = 0
        self.optimizer_examples = 0
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
        role: AuxiliaryRole,
        features: FeatureProvider,
        *,
        seed: int,
        dropout: float,
        batch_size: int,
        device: str | torch.device,
    ) -> PackedAuxiliaryTrainer:
        parsed_device = require_training_device(device)
        configure_determinism(seed)
        model = ConversionMLP(features.categorical_specs, dropout=dropout)
        return cls(
            role,
            model,
            features,
            seed=seed,
            dropout=dropout,
            batch_size=batch_size,
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

    def train_work_unit(
        self,
        *,
        work_id: int,
        decision_time: float,
        sampler: PackedDeterministicSampler,
    ) -> AuxiliaryWorkResult:
        if work_id != self.work_units:
            raise ConsistencyError("ES-DFM auxiliary work ID is not contiguous")
        steps = 500 if work_id == 0 else 100
        examples = steps * self.batch_size
        record_keys = np.empty(examples, dtype=np.uint64)
        sources = np.empty(examples, dtype=np.uint8)
        loss_sum = 0.0
        self._activate_rng()
        self.model.train()
        for step in range(steps):
            sample = sampler.sample(
                simulator_time=decision_time,
                batch_size=self.batch_size,
            )
            features = self.features.tensor_batch(sample.feature_refs).to(self.device)
            targets = torch.from_numpy(sample.targets).to(self.device)
            logits = self.model(features.categorical, features.numeric)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise ConsistencyError("ES-DFM auxiliary loss is not a finite scalar")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            gradient_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(),
                5.0,
                error_if_nonfinite=True,
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise ConsistencyError("ES-DFM auxiliary gradient norm is not finite")
            self.optimizer.step()
            start = step * self.batch_size
            end = start + self.batch_size
            record_keys[start:end] = sample.record_keys
            sources[start:end] = sample.sources
            loss_sum += float(loss.detach().cpu().item())
            self.model_version += 1
        self._capture_rng()
        self.work_units += 1
        self.optimizer_steps += steps
        self.optimizer_examples += examples
        return AuxiliaryWorkResult(
            role=self.role,
            work_id=work_id,
            steps=steps,
            examples=examples,
            mean_loss=loss_sum / steps,
            record_keys=record_keys,
            sources=sources,
        )

    def logits(self, features: FeatureTensorBatch) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            result = self.model(features.categorical, features.numeric)
        if not bool(torch.isfinite(result).all()):
            raise ConsistencyError("ES-DFM auxiliary prediction is not finite")
        return cast(Tensor, result)

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "config_sha256": self.config_sha256,
            "role": self.role,
            "seed": self.seed,
            "device_type": self.device.type,
            "model": _snapshot_state(self.model.state_dict()),
            "optimizer": _snapshot_state(self.optimizer.state_dict()),
            "work_units": self.work_units,
            "optimizer_steps": self.optimizer_steps,
            "optimizer_examples": self.optimizer_examples,
            "model_version": self.model_version,
            "cpu_rng_state": self._cpu_rng_state.clone(),
            "cuda_rng_state": [value.clone() for value in self._cuda_rng_state],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (
            state.get("version") != 1
            or state.get("config_sha256") != self.config_sha256
            or state.get("role") != self.role
            or state.get("seed") != self.seed
            or state.get("device_type") != self.device.type
        ):
            raise ConsistencyError("ES-DFM auxiliary checkpoint identity changed")
        model = state.get("model")
        optimizer = state.get("optimizer")
        cpu_rng = state.get("cpu_rng_state")
        cuda_rng = state.get("cuda_rng_state")
        counters = tuple(
            state.get(key)
            for key in (
                "work_units",
                "optimizer_steps",
                "optimizer_examples",
                "model_version",
            )
        )
        if (
            not isinstance(model, dict)
            or not isinstance(optimizer, dict)
            or not isinstance(cpu_rng, Tensor)
            or not isinstance(cuda_rng, list)
            or not all(isinstance(value, Tensor) for value in cuda_rng)
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in counters
            )
        ):
            raise ConsistencyError("ES-DFM auxiliary checkpoint state is malformed")
        work_units, steps, examples, model_version = counters
        assert isinstance(work_units, int)
        assert isinstance(steps, int)
        assert isinstance(examples, int)
        assert isinstance(model_version, int)
        expected_steps = 0 if work_units == 0 else 500 + (work_units - 1) * 100
        if (
            steps != expected_steps
            or examples != steps * self.batch_size
            or model_version != steps
            or (self.device.type == "cuda" and len(cuda_rng) != torch.cuda.device_count())
            or (self.device.type == "cpu" and bool(cuda_rng))
        ):
            raise ConsistencyError("ES-DFM auxiliary checkpoint counters are inconsistent")
        self.model.load_state_dict(model)
        self.optimizer.load_state_dict(optimizer)
        self.work_units = work_units
        self.optimizer_steps = steps
        self.optimizer_examples = examples
        self.model_version = model_version
        self._cpu_rng_state = cpu_rng.cpu().clone()
        self._cuda_rng_state = [value.cpu().clone() for value in cuda_rng]


class ESDFMAuxiliaryPair:
    """Update q_tn then q_dp and freeze both before every main credit."""

    def __init__(self, q_tn: PackedAuxiliaryTrainer, q_dp: PackedAuxiliaryTrainer) -> None:
        if q_tn.role != "q_tn" or q_dp.role != "q_dp":
            raise ValueError("ES-DFM auxiliary pair roles are reversed")
        if q_tn.device != q_dp.device or q_tn.batch_size != q_dp.batch_size:
            raise ValueError("ES-DFM auxiliary pair configurations do not align")
        self.q_tn = q_tn
        self.q_dp = q_dp

    @classmethod
    def create(
        cls,
        features: FeatureProvider,
        *,
        training_seed: int,
        dropout: float,
        batch_size: int,
        device: str | torch.device,
    ) -> ESDFMAuxiliaryPair:
        return cls(
            PackedAuxiliaryTrainer.create(
                "q_tn",
                features,
                seed=training_seed + 1000,
                dropout=dropout,
                batch_size=batch_size,
                device=device,
            ),
            PackedAuxiliaryTrainer.create(
                "q_dp",
                features,
                seed=training_seed + 2000,
                dropout=dropout,
                batch_size=batch_size,
                device=device,
            ),
        )

    def update(
        self,
        *,
        credit_id: int,
        decision_time: float,
        q_tn_sampler: PackedDeterministicSampler,
        q_dp_sampler: PackedDeterministicSampler,
    ) -> ESDFMUpdateResult:
        if credit_id != self.q_tn.work_units or credit_id != self.q_dp.work_units:
            raise ConsistencyError("ES-DFM auxiliary pair credit is not contiguous")
        q_tn = self.q_tn.train_work_unit(
            work_id=credit_id,
            decision_time=decision_time,
            sampler=q_tn_sampler,
        )
        q_dp = self.q_dp.train_work_unit(
            work_id=credit_id,
            decision_time=decision_time,
            sampler=q_dp_sampler,
        )
        self.q_tn.model.eval()
        self.q_dp.model.eval()
        return ESDFMUpdateResult(credit_id=credit_id, q_tn=q_tn, q_dp=q_dp)

    def logits(self, features: FeatureTensorBatch) -> tuple[Tensor, Tensor]:
        return self.q_tn.logits(features), self.q_dp.logits(features)

    def state_dict(self) -> dict[str, object]:
        return {"q_tn": self.q_tn.state_dict(), "q_dp": self.q_dp.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        q_tn = state.get("q_tn")
        q_dp = state.get("q_dp")
        if not isinstance(q_tn, dict) or not isinstance(q_dp, dict):
            raise ConsistencyError("ES-DFM auxiliary pair checkpoint is malformed")
        self.q_tn.load_state_dict(q_tn)
        self.q_dp.load_state_dict(q_dp)
        if self.q_tn.work_units != self.q_dp.work_units:
            raise ConsistencyError("ES-DFM auxiliary pair checkpoint credits do not align")
