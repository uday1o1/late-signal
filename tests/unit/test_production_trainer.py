from __future__ import annotations

import numpy as np
import pytest
import torch

from latesignal.errors import ConfigurationError, ConsistencyError
from latesignal.features.store import FeatureTensorBatch
from latesignal.models.conversion_mlp import CategoricalSpec
from latesignal.training.packed import (
    PackedDeterministicSampler,
    PackedRecordStore,
    RecordKind,
    packed_record_batch,
)
from latesignal.training.production import (
    PackedConversionTrainer,
    ProductionTrainingConfig,
    require_training_device,
)

FIELDS = tuple(f"field_{index}" for index in range(17))


class _Features:
    def __init__(self, count: int) -> None:
        self.values = np.arange(count, dtype=np.int64) % 8

    @property
    def categorical_specs(self) -> dict[str, CategoricalSpec]:
        return {field: CategoricalSpec(8, 2) for field in FIELDS}

    def tensor_batch(self, references: np.ndarray) -> FeatureTensorBatch:
        values = torch.from_numpy(self.values[references])
        numeric = torch.zeros((references.size, 4), dtype=torch.float32)
        numeric[:, 0] = values.float() / 8.0
        return FeatureTensorBatch(
            categorical={field: values.clone() for field in FIELDS},
            numeric=numeric,
        )


def _store() -> PackedRecordStore:
    store = PackedRecordStore(feature_count=64)
    store.append(
        packed_record_batch(
            feature_refs=np.arange(64),
            available_at=np.arange(64, dtype=np.float64),
            targets=np.arange(64) % 2,
            kinds=np.full(64, RecordKind.PROVISIONAL),
        ),
        simulator_time=64.0,
    )
    return store


def _config(loss_mode: str = "bce", *, dropout: float = 0.0) -> ProductionTrainingConfig:
    return ProductionTrainingConfig(
        learning_rate=0.001,
        weight_decay=0.0,
        dropout=dropout,
        gradient_norm_clip=5.0,
        steps_per_credit=2,
        batch_size=8,
        loss_mode=loss_mode,  # type: ignore[arg-type]
    )


def _sampler(store: PackedRecordStore) -> PackedDeterministicSampler:
    return PackedDeterministicSampler(
        store,
        seed=17,
        recent_window_seconds=16.0,
        reservoir_capacity=16,
    )


def test_production_trainer_is_seeded_and_reconciles_packed_exposures() -> None:
    features = _Features(64)
    store = _store()
    trainers = [
        PackedConversionTrainer.create(features, _config(), seed=41, device="cpu") for _ in range(2)
    ]
    results = [
        trainer.spend_credit(
            credit_id=0,
            decision_time=64.0,
            sampler=_sampler(store),
        )
        for trainer in trainers
    ]

    np.testing.assert_array_equal(
        results[0].exposure.record_keys,
        results[1].exposure.record_keys,
    )
    np.testing.assert_array_equal(
        trainers[0].predict(np.arange(8)), trainers[1].predict(np.arange(8))
    )
    assert results[0].mean_loss == results[1].mean_loss
    assert results[0].examples == 16
    assert trainers[0].budget.optimizer_examples == 16
    assert trainers[0].model_version == 2


def test_production_trainer_resume_matches_uninterrupted_next_credit() -> None:
    features = _Features(64)
    store = _store()
    config = _config("fnw", dropout=0.1)
    trainer = PackedConversionTrainer.create(features, config, seed=73, device="cpu")
    sampler = _sampler(store)
    trainer.spend_credit(credit_id=0, decision_time=64.0, sampler=sampler)
    trainer_state = trainer.state_dict()
    sampler_state = sampler.state_dict()
    expected = trainer.spend_credit(credit_id=1, decision_time=64.0, sampler=sampler)

    resumed = PackedConversionTrainer.create(features, config, seed=73, device="cpu")
    resumed_sampler = _sampler(store)
    resumed.load_state_dict(trainer_state)
    resumed_sampler.load_state_dict(sampler_state)
    actual = resumed.spend_credit(
        credit_id=1,
        decision_time=64.0,
        sampler=resumed_sampler,
    )

    np.testing.assert_array_equal(actual.exposure.record_keys, expected.exposure.record_keys)
    np.testing.assert_allclose(actual.exposure.weights, expected.exposure.weights, rtol=0, atol=0)
    np.testing.assert_allclose(
        resumed.predict(np.arange(8)),
        trainer.predict(np.arange(8)),
        rtol=0,
        atol=0,
    )


def test_production_trainer_rejects_noncontiguous_credit_and_cuda_without_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = PackedConversionTrainer.create(_Features(64), _config(), seed=17, device="cpu")
    with pytest.raises(ConsistencyError, match="not contiguous"):
        trainer.spend_credit(
            credit_id=1,
            decision_time=64.0,
            sampler=_sampler(_store()),
        )

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(ConfigurationError, match="CUBLAS_WORKSPACE_CONFIG"):
        require_training_device("cuda")


def test_production_trainers_keep_independent_dropout_rng_streams() -> None:
    features = _Features(64)
    config = _config(dropout=0.1)
    interleaved = PackedConversionTrainer.create(features, config, seed=41, device="cpu")
    auxiliary = PackedConversionTrainer.create(features, config, seed=1041, device="cpu")
    control = PackedConversionTrainer.create(features, config, seed=41, device="cpu")
    interleaved_sampler = _sampler(_store())
    auxiliary_sampler = _sampler(_store())
    control_sampler = _sampler(_store())

    interleaved.spend_credit(
        credit_id=0,
        decision_time=64.0,
        sampler=interleaved_sampler,
    )
    auxiliary.spend_credit(
        credit_id=0,
        decision_time=64.0,
        sampler=auxiliary_sampler,
    )
    interleaved.spend_credit(
        credit_id=1,
        decision_time=64.0,
        sampler=interleaved_sampler,
    )
    control.spend_credit(credit_id=0, decision_time=64.0, sampler=control_sampler)
    control.spend_credit(credit_id=1, decision_time=64.0, sampler=control_sampler)

    np.testing.assert_allclose(
        interleaved.predict(np.arange(8)),
        control.predict(np.arange(8)),
        rtol=0,
        atol=0,
    )
