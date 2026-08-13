from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from sklearn.metrics import log_loss

from latesignal.contracts.records import TrainingRecord
from latesignal.errors import ConsistencyError
from latesignal.evaluation.calibration import evaluate_calibration
from latesignal.models.conversion_mlp import CategoricalSpec, ConversionMLP
from latesignal.models.lightgbm import run_lightgbm_reference
from latesignal.models.logistic import run_logistic_reference
from latesignal.models.offline import MatureOfflineSplit
from latesignal.training.reproducibility import configure_determinism
from latesignal.training.sampler import DeterministicSampler
from latesignal.training.trainer import MLPTrainer, ModelBatch

FIELDS = tuple(f"field_{index}" for index in range(17))


def _model(*, dropout: float = 0.0) -> ConversionMLP:
    return ConversionMLP(
        {field: CategoricalSpec(bucket_count=8, embedding_dim=2) for field in FIELDS},
        dropout=dropout,
    )


def _tensor_fixture(
    *, shuffled: bool = False
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    group = torch.tensor([0, 0, 1, 1] * 4, dtype=torch.long)
    categorical = {field: group.clone() for field in FIELDS}
    numeric = torch.zeros((16, 4), dtype=torch.float32)
    numeric[:, 0] = torch.where(group == 1, 2.0, -2.0)
    labels = group.float()
    if shuffled:
        labels = torch.tensor([0, 1, 0, 1] * 4, dtype=torch.float32)
    return categorical, numeric, labels


def _fit_fixture(*, shuffled: bool) -> float:
    configure_determinism(41)
    model = _model()
    categorical, numeric, labels = _tensor_fixture(shuffled=shuffled)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(categorical, numeric), labels
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    with torch.no_grad():
        evaluation_labels = _tensor_fixture(shuffled=False)[2]
        probabilities = torch.sigmoid(model(categorical, numeric))
    return float(torch.nn.functional.binary_cross_entropy(probabilities, evaluation_labels).item())


def _offline_split() -> MatureOfflineSplit:
    rng = np.random.default_rng(17)
    features = rng.normal(size=(240, 4))
    labels = (features[:, 0] + 0.6 * features[:, 1] > 0.0).astype(np.int64)
    return MatureOfflineSplit(
        train_features=features[:180],
        train_labels=labels[:180],
        train_click_times=np.arange(180, dtype=np.float64),
        train_available_at=np.arange(180, dtype=np.float64) + 1.0,
        evaluation_features=features[180:],
        evaluation_labels=labels[180:],
        evaluation_click_times=np.arange(181, 241, dtype=np.float64),
        training_cutoff=180.0,
    )


def _record(index: int, *, available_at: int) -> TrainingRecord:
    return TrainingRecord(
        record_id=f"record-{index}",
        click_id=f"click-{index}",
        available_at=available_at,
        status="final",
        target=float(index % 2),
        weight=1.0,
        correction_group=None,
        source_method="test",
        feature=float(index % 2),
    )


def test_conversion_mlp_has_exact_locked_architecture() -> None:
    model = _model(dropout=0.1)

    assert len(model.embeddings) == 17
    assert [type(layer).__name__ for layer in model.backbone] == [
        "Linear",
        "SiLU",
        "LayerNorm",
        "Dropout",
        "Linear",
        "SiLU",
        "LayerNorm",
        "Dropout",
        "Linear",
        "SiLU",
    ]
    assert model.backbone[0].out_features == 256
    assert model.backbone[4].out_features == 128
    assert model.backbone[8].out_features == 64
    assert model.output.out_features == 1


def test_model_overfits_tiny_fixture_and_shuffle_is_at_chance() -> None:
    fitted_loss = _fit_fixture(shuffled=False)
    shuffled_loss = _fit_fixture(shuffled=True)

    assert fitted_loss < 0.01
    assert 0.68 < shuffled_loss < 0.71


def test_offline_references_use_chronological_mature_labels() -> None:
    split = _offline_split()

    logistic = run_logistic_reference(split, seed=17)
    lightgbm = run_lightgbm_reference(split, seed=17)
    repeated_lightgbm = run_lightgbm_reference(split, seed=17)

    assert log_loss(logistic.labels, logistic.probabilities) < 0.25
    assert log_loss(lightgbm.labels, lightgbm.probabilities) < 0.25
    np.testing.assert_array_equal(lightgbm.probabilities, repeated_lightgbm.probabilities)
    assert not logistic.ranking_eligible
    assert not lightgbm.ranking_eligible


def test_offline_split_rejects_immature_training_truth() -> None:
    split = _offline_split()

    with pytest.raises(ConsistencyError, match="immature"):
        MatureOfflineSplit(
            train_features=split.train_features,
            train_labels=split.train_labels,
            train_click_times=split.train_click_times,
            train_available_at=np.full(split.train_labels.shape, 181.0),
            evaluation_features=split.evaluation_features,
            evaluation_labels=split.evaluation_labels,
            evaluation_click_times=split.evaluation_click_times,
            training_cutoff=180.0,
        )


def test_calibration_has_locked_bins_and_expected_coefficients() -> None:
    centers = np.arange(0.05, 1.0, 0.1)
    probabilities = np.repeat(centers, 100)
    labels = np.concatenate(
        [
            np.concatenate(
                [
                    np.ones(round(center * 100), dtype=np.int64),
                    np.zeros(100 - round(center * 100), dtype=np.int64),
                ]
            )
            for center in centers
        ]
    )
    result = evaluate_calibration(labels, probabilities)

    assert result.count == 1_000
    assert len(result.bins) == 10
    assert sum(item.count for item in result.bins) == 1_000
    assert result.intercept is not None and abs(result.intercept) < 0.01
    assert result.slope is not None and math.isclose(result.slope, 1.0, abs_tol=0.01)
    assert result.expected_calibration_error < 1e-12


def test_sampler_restore_and_training_ledgers_reconcile_exactly() -> None:
    sampler = DeterministicSampler(seed=73, recent_window_seconds=10, reservoir_capacity=3)
    for index in range(8):
        sampler.add(_record(index, available_at=index), simulator_time=index)
    restored = DeterministicSampler(seed=73, recent_window_seconds=10, reservoir_capacity=3)
    restored.load_state_dict(sampler.state_dict())
    first = sampler.sample(simulator_time=20, batch_size=4)
    repeated = restored.sample(simulator_time=20, batch_size=4)
    assert [(item.record.record_id, item.source) for item in first] == [
        (item.record.record_id, item.source) for item in repeated
    ]

    def encode(records: tuple[TrainingRecord, ...]) -> ModelBatch:
        values = torch.tensor([int(record.feature) for record in records], dtype=torch.long)
        return ModelBatch(
            categorical={field: values.clone() for field in FIELDS},
            numeric=torch.zeros((len(records), 4), dtype=torch.float32),
            targets=torch.tensor([record.target for record in records], dtype=torch.float32),
            weights=torch.tensor([record.weight for record in records], dtype=torch.float32),
            record_ids=tuple(record.record_id for record in records),
        )

    trainer = MLPTrainer(
        _model(),
        learning_rate=0.001,
        weight_decay=0.0,
        gradient_norm_clip=5.0,
        steps_per_credit=2,
        batch_size=4,
        encoder=encode,
    )
    credit = trainer.spend_credit(credit_id=0, decision_time=20, sampler=sampler)

    assert credit.steps == 2
    assert credit.examples == 8
    assert trainer.budget.state_dict() == {
        "credits": 1,
        "optimizer_steps": 2,
        "optimizer_examples": 8,
    }
    assert len(trainer.exposures) == 8
