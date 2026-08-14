from __future__ import annotations

import numpy as np
import torch

from latesignal.features.store import FeatureTensorBatch
from latesignal.methods.production import PackedDelayedMethod
from latesignal.models.conversion_mlp import CategoricalSpec
from latesignal.simulator.production_oracle import TruthEventBatch
from latesignal.training.packed import PackedDeterministicSampler, PackedRecordStore
from latesignal.training.production import ProductionTrainingConfig
from latesignal.training.production_dfm import PackedDFMTrainer

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
        return FeatureTensorBatch(
            categorical={field: values.clone() for field in FIELDS},
            numeric=numeric,
        )


def _fixture() -> tuple[PackedDelayedMethod, PackedRecordStore]:
    click_times = np.arange(64, dtype=np.float64)
    store = PackedRecordStore(feature_count=64)
    method = PackedDelayedMethod(
        "dfm",
        click_times=click_times,
        monitoring_mask=np.zeros(64, dtype=np.bool_),
        main_store=store,
    )
    method.process_boundary(
        boundary=64.0,
        click_refs=np.arange(64, dtype=np.int32),
        truth=TruthEventBatch(
            feature_refs=np.asarray([0, 2, 4], dtype=np.int32),
            available_at=np.asarray([1.0, 3.0, 5.0]),
            labels=np.ones(3, dtype=np.int8),
        ),
    )
    return method, store


def _config() -> ProductionTrainingConfig:
    return ProductionTrainingConfig(
        learning_rate=0.001,
        weight_decay=0.0,
        dropout=0.1,
        gradient_norm_clip=5.0,
        steps_per_credit=2,
        batch_size=8,
        loss_mode="bce",
    )


def _sampler(store: PackedRecordStore) -> PackedDeterministicSampler:
    return PackedDeterministicSampler(
        store,
        seed=17,
        recent_window_seconds=16.0,
        reservoir_capacity=16,
    )


def test_production_dfm_trains_censored_observations_and_resumes() -> None:
    method, store = _fixture()
    features = _Features(64)
    trainer = PackedDFMTrainer.create(features, method, _config(), seed=41, device="cpu")
    sampler = _sampler(store)
    first = trainer.spend_credit(credit_id=0, decision_time=64.0, sampler=sampler)
    trainer_state = trainer.state_dict()
    sampler_state = sampler.state_dict()
    expected = trainer.spend_credit(credit_id=1, decision_time=64.0, sampler=sampler)

    resumed = PackedDFMTrainer.create(features, method, _config(), seed=41, device="cpu")
    resumed_sampler = _sampler(store)
    resumed.load_state_dict(trainer_state)
    resumed_sampler.load_state_dict(sampler_state)
    actual = resumed.spend_credit(
        credit_id=1,
        decision_time=64.0,
        sampler=resumed_sampler,
    )

    assert first.examples == 16
    np.testing.assert_array_equal(actual.exposure.record_keys, expected.exposure.record_keys)
    np.testing.assert_allclose(actual.mean_loss, expected.mean_loss, rtol=0, atol=0)
    np.testing.assert_allclose(
        resumed.predict(np.arange(8)),
        trainer.predict(np.arange(8)),
        rtol=0,
        atol=0,
    )
    assert trainer.parameter_count > trainer.model.conversion.parameter_count
