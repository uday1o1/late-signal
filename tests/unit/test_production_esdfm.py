from __future__ import annotations

import numpy as np
import pytest
import torch

from latesignal.errors import ConsistencyError
from latesignal.features.store import FeatureTensorBatch
from latesignal.models.conversion_mlp import CategoricalSpec
from latesignal.training.packed import (
    PackedDeterministicSampler,
    PackedRecordStore,
    RecordKind,
    packed_record_batch,
)
from latesignal.training.production import PackedConversionTrainer, ProductionTrainingConfig
from latesignal.training.production_esdfm import ESDFMAuxiliaryPair

FIELDS = tuple(f"field_{index}" for index in range(17))


class _Features:
    def __init__(self, count: int) -> None:
        self.values = np.arange(count, dtype=np.int64) % 8

    @property
    def categorical_specs(self) -> dict[str, CategoricalSpec]:
        return {field: CategoricalSpec(8, 2) for field in FIELDS}

    def tensor_batch(self, references: np.ndarray) -> FeatureTensorBatch:
        values = torch.from_numpy(self.values[references])
        return FeatureTensorBatch(
            categorical={field: values.clone() for field in FIELDS},
            numeric=torch.zeros((references.size, 4), dtype=torch.float32),
        )


def _store(target_offset: int = 0) -> PackedRecordStore:
    store = PackedRecordStore(feature_count=32)
    store.append(
        packed_record_batch(
            feature_refs=np.arange(32),
            available_at=np.arange(32, dtype=np.float64),
            targets=(np.arange(32) + target_offset) % 2,
            kinds=np.full(32, RecordKind.FINAL),
        ),
        simulator_time=32.0,
    )
    return store


def _sampler(store: PackedRecordStore, seed: int) -> PackedDeterministicSampler:
    return PackedDeterministicSampler(
        store,
        seed=seed,
        recent_window_seconds=8.0,
        reservoir_capacity=8,
    )


def test_production_esdfm_updates_auxiliaries_before_weighted_main_credit() -> None:
    features = _Features(32)
    q_tn_store = _store()
    q_dp_store = _store(1)
    pair = ESDFMAuxiliaryPair.create(
        features,
        training_seed=17,
        dropout=0.0,
        batch_size=4,
        device="cpu",
    )
    q_tn_sampler = _sampler(q_tn_store, 1017)
    q_dp_sampler = _sampler(q_dp_store, 2017)
    update = pair.update(
        credit_id=0,
        decision_time=32.0,
        q_tn_sampler=q_tn_sampler,
        q_dp_sampler=q_dp_sampler,
    )
    later = pair.update(
        credit_id=1,
        decision_time=33.0,
        q_tn_sampler=q_tn_sampler,
        q_dp_sampler=q_dp_sampler,
    )
    main_store = _store()
    config = ProductionTrainingConfig(
        learning_rate=0.001,
        weight_decay=0.0,
        dropout=0.0,
        gradient_norm_clip=5.0,
        steps_per_credit=2,
        batch_size=4,
        loss_mode="es_dfm",
    )
    main = PackedConversionTrainer.create(features, config, seed=17, device="cpu")
    result = main.spend_credit(
        credit_id=0,
        decision_time=32.0,
        sampler=_sampler(main_store, 17),
        auxiliary_provider=pair,
    )

    assert update.q_tn.steps == update.q_dp.steps == 500
    assert update.auxiliary_examples == 4_000
    assert later.q_tn.steps == later.q_dp.steps == 100
    assert pair.q_tn.optimizer_steps == pair.q_dp.optimizer_steps == 600
    assert result.examples == 8
    assert np.all(result.exposure.weights >= 1e-4)
    assert np.all(result.exposure.weights <= 2.0)
    assert pair.q_tn.seed == 1017
    assert pair.q_dp.seed == 2017


def test_esdfm_pair_checkpoint_round_trip_and_main_requires_pair() -> None:
    features = _Features(32)
    pair = ESDFMAuxiliaryPair.create(
        features,
        training_seed=41,
        dropout=0.0,
        batch_size=4,
        device="cpu",
    )
    pair.update(
        credit_id=0,
        decision_time=32.0,
        q_tn_sampler=_sampler(_store(), 1041),
        q_dp_sampler=_sampler(_store(1), 2041),
    )
    restored = ESDFMAuxiliaryPair.create(
        features,
        training_seed=41,
        dropout=0.0,
        batch_size=4,
        device="cpu",
    )
    restored.load_state_dict(pair.state_dict())
    batch = features.tensor_batch(np.arange(4))
    expected = pair.logits(batch)
    actual = restored.logits(batch)
    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=0)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=0)

    config = ProductionTrainingConfig(
        learning_rate=0.001,
        weight_decay=0.0,
        dropout=0.0,
        gradient_norm_clip=5.0,
        steps_per_credit=1,
        batch_size=4,
        loss_mode="es_dfm",
    )
    main = PackedConversionTrainer.create(features, config, seed=41, device="cpu")
    with pytest.raises(ConsistencyError, match="auxiliary logits"):
        main.spend_credit(
            credit_id=0,
            decision_time=32.0,
            sampler=_sampler(_store(), 41),
        )
