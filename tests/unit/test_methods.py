from __future__ import annotations

import math
import random
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from latesignal.contracts.events import ClickEvent, NegativeMaturity, PositiveReveal, TruthRecord
from latesignal.contracts.study_a import load_study_a_config
from latesignal.experiments.study_a import _model_hash, _new_auxiliary
from latesignal.methods.base import DelayedMethod
from latesignal.methods.complete_wait import CompleteWaitMethod
from latesignal.methods.dfm import DelayedFeedbackMethod
from latesignal.methods.es_dfm import ESDFMMethod
from latesignal.methods.fixed_wait import FixedWaitMethod
from latesignal.methods.fnw import FakeNegativeWeightedMethod
from latesignal.methods.immediate_fake_negative import ImmediateFakeNegativeMethod
from latesignal.methods.losses import dfm_loss, esdfm_loss, fnw_loss
from latesignal.methods.oracle_reference import OracleReferenceMethod


def test_esdfm_auxiliary_models_and_samplers_use_separate_locked_seeds() -> None:
    config = load_study_a_config(Path("configs/experiments/study_a.synthetic.yaml"))

    first = _new_auxiliary(config)
    repeated = _new_auxiliary(config)

    assert _model_hash(first.q_tn) == _model_hash(repeated.q_tn)
    assert _model_hash(first.q_dp) == _model_hash(repeated.q_dp)
    assert _model_hash(first.q_tn) != _model_hash(first.q_dp)
    assert first.q_tn_rng.random() == random.Random(config.seed + 1_000).random()
    assert first.q_dp_rng.random() == random.Random(config.seed + 2_000).random()


def test_complete_wait_emits_only_at_full_cohort_maturity() -> None:
    method = CompleteWaitMethod(attribution_seconds=30)
    method.on_click(ClickEvent("positive", 0, 1.0))
    method.on_click(ClickEvent("negative", 2, -1.0))

    assert method.on_positive_reveal(PositiveReveal("positive", 10)) == []
    assert method.on_boundary(29) == []
    positive = method.on_boundary(30)
    negative = method.on_negative_maturity(NegativeMaturity("negative", 32))

    assert [(record.target, record.available_at) for record in positive] == [(1.0, 30)]
    assert [(record.target, record.available_at) for record in negative] == [(0.0, 32)]
    assert all(record.status == "final" for record in positive + negative)


def test_immediate_fake_negative_and_fnw_preserve_correction_records() -> None:
    for method in (ImmediateFakeNegativeMethod(), FakeNegativeWeightedMethod()):
        provisional = method.on_click(ClickEvent("click", 0, 1.0))
        correction = method.on_positive_reveal(PositiveReveal("click", 9))

        assert [record.target for record in provisional + correction] == [0.0, 1.0]
        assert provisional[0].record_id != correction[0].record_id
        assert provisional[0].correction_group == correction[0].correction_group == "click"


def test_fixed_wait_sequences_cover_early_late_and_final_negative() -> None:
    method = FixedWaitMethod(wait_seconds=10)
    for name in ("early", "late", "negative"):
        method.on_click(ClickEvent(name, 0, 1.0))

    early = method.on_positive_reveal(PositiveReveal("early", 5))
    provisional = method.on_boundary(10)
    late = method.on_positive_reveal(PositiveReveal("late", 20))
    final_negative = method.on_negative_maturity(NegativeMaturity("negative", 30))

    assert [(record.click_id, record.target) for record in early] == [("early", 1.0)]
    assert [(record.click_id, record.target) for record in provisional] == [
        ("late", 0.0),
        ("negative", 0.0),
    ]
    assert [(record.click_id, record.target) for record in late] == [("late", 1.0)]
    assert final_negative == []


def test_dfm_materializes_only_status_legal_at_current_time() -> None:
    method = DelayedFeedbackMethod(attribution_seconds=30 * 86_400)
    method.on_click(ClickEvent("positive", 0, 1.0))
    method.on_click(ClickEvent("unresolved", 0, -1.0))

    before = method.materialize("positive", 86_400)
    method.on_positive_reveal(PositiveReveal("positive", 2 * 86_400))
    after = method.materialize("positive", 2 * 86_400)
    unresolved = method.materialize("unresolved", 40 * 86_400)

    assert (before.target, before.time_days, before.status) == (0.0, 1.0, "right_censored")
    assert (after.target, after.time_days, after.status) == (1.0, 2.0, "revealed_positive")
    assert unresolved.time_days == 30.0


def test_esdfm_auxiliary_labels_use_only_fully_mature_truth() -> None:
    method = ESDFMMethod(wait_seconds=10, attribution_seconds=30)
    for name in ("early", "late", "negative"):
        method.on_click(ClickEvent(name, 0, 1.0))
    method.on_positive_reveal(PositiveReveal("early", 5))
    method.on_boundary(10)
    method.on_positive_reveal(PositiveReveal("late", 20))

    assert method.auxiliary_records(29) == ()
    method.on_negative_maturity(NegativeMaturity("negative", 30))
    auxiliary = {record.click_id: record for record in method.auxiliary_records(30)}

    assert (auxiliary["early"].q_tn_target, auxiliary["early"].q_dp_target) == (None, 0.0)
    assert (auxiliary["late"].q_tn_target, auxiliary["late"].q_dp_target) == (0.0, 1.0)
    assert (auxiliary["negative"].q_tn_target, auxiliary["negative"].q_dp_target) == (
        1.0,
        0.0,
    )


def test_dfm_loss_matches_hand_calculated_likelihood() -> None:
    conversion_logits = torch.tensor([0.0, 0.0], dtype=torch.float64)
    rate_logit = math.log(math.expm1(2.0))
    rate_logits = torch.tensor([rate_logit, rate_logit], dtype=torch.float64)
    targets = torch.tensor([1.0, 0.0], dtype=torch.float64)
    time_days = torch.tensor([0.5, 0.5], dtype=torch.float64)

    actual = dfm_loss(conversion_logits, rate_logits, targets, time_days)
    rate = 2.0 + 1e-6
    positive = math.log(2.0) - math.log(rate) + rate * 0.5
    censored = -math.log(0.5 + 0.5 * math.exp(-rate * 0.5))

    assert actual.item() == pytest.approx((positive + censored) / 2.0, abs=1e-12)


def test_fnw_loss_matches_hand_calculated_weights_and_detaches_probability() -> None:
    logits = torch.tensor([0.0, 0.0], requires_grad=True)
    targets = torch.tensor([1.0, 0.0])

    result = fnw_loss(logits, targets)
    result.loss.backward()

    assert result.weights.tolist() == pytest.approx([1.5, 0.75])
    assert result.loss.item() == pytest.approx(1.125 * math.log(2.0))
    assert logits.grad is not None


def test_esdfm_loss_matches_weights_clips_and_detaches_auxiliary_models() -> None:
    logits = torch.tensor([0.0, 0.0], requires_grad=True)
    targets = torch.tensor([1.0, 0.0])
    q_tn_logits = torch.tensor([math.log(1 / 3), math.log(1 / 3)], requires_grad=True)
    q_dp_logits = torch.tensor([0.0, 0.0], requires_grad=True)

    result = esdfm_loss(logits, targets, q_tn_logits, q_dp_logits)
    result.loss.backward()

    assert result.weights.tolist() == pytest.approx([1.5, 0.375])
    assert result.loss.item() == pytest.approx(0.9375 * math.log(2.0))
    assert logits.grad is not None
    assert q_tn_logits.grad is None
    assert q_dp_logits.grad is None


def test_method_state_round_trips_and_oracle_is_excluded_from_ranking() -> None:
    original = FixedWaitMethod(wait_seconds=10)
    original.on_click(ClickEvent("click", 0, 2.0))
    original.on_boundary(10)
    restored = FixedWaitMethod(wait_seconds=10)
    restored.load_state_dict(original.state_dict())

    assert restored.state_dict() == original.state_dict()

    oracle = OracleReferenceMethod((TruthRecord("click", 1, 20),))
    records = oracle.on_click(ClickEvent("click", 0, 2.0))
    assert records[0].available_at == 0
    assert oracle.deployable is False
    assert oracle.ranking_eligible is False


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CompleteWaitMethod(attribution_seconds=30),
        ImmediateFakeNegativeMethod,
        lambda: FixedWaitMethod(wait_seconds=10),
        lambda: DelayedFeedbackMethod(attribution_seconds=30),
        FakeNegativeWeightedMethod,
        lambda: ESDFMMethod(wait_seconds=10, attribution_seconds=30),
        lambda: OracleReferenceMethod((TruthRecord("click", 1, 20),)),
    ],
)
def test_every_method_state_survives_round_trip(factory: Callable[[], DelayedMethod]) -> None:
    original = factory()
    original.on_click(ClickEvent("click", 0, 2.0))
    original.on_positive_reveal(PositiveReveal("click", 20))
    original.on_boundary(30)
    restored = factory()

    restored.load_state_dict(original.state_dict())

    assert restored.state_dict() == original.state_dict()
    if not isinstance(restored, OracleReferenceMethod):
        assert "_truth" not in vars(restored)
