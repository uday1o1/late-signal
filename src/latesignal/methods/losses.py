"""Numerically stable published-method loss equations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class WeightedLoss:
    loss: Tensor
    weights: Tensor


def dfm_loss(
    conversion_logits: Tensor,
    rate_logits: Tensor,
    targets: Tensor,
    time_days: Tensor,
) -> Tensor:
    """Chapelle joint exponential-delay negative log likelihood."""

    rate = torch.clamp(nn.functional.softplus(rate_logits) + 1e-6, 1e-6, 100.0)
    positive = nn.functional.softplus(-conversion_logits) - torch.log(rate) + rate * time_days
    log_nonconversion = nn.functional.logsigmoid(-conversion_logits)
    log_delayed_conversion = nn.functional.logsigmoid(conversion_logits) - rate * time_days
    censored = -torch.logaddexp(log_nonconversion, log_delayed_conversion)
    return torch.mean(targets * positive + (1.0 - targets) * censored)


def fnw_weights(logits: Tensor, targets: Tensor) -> Tensor:
    """Detached exposure-time weights for Fake Negative Weighted transfer."""

    probability = torch.sigmoid(logits).detach().clamp(1e-6, 1.0 - 1e-6)
    return targets * (1.0 + probability) + (1.0 - targets) * (1.0 - probability.square())


def fnw_loss(logits: Tensor, targets: Tensor) -> WeightedLoss:
    weights = fnw_weights(logits, targets)
    positive = nn.functional.softplus(-logits)
    negative = nn.functional.softplus(logits)
    loss = torch.mean(targets * weights * positive + (1.0 - targets) * weights * negative)
    return WeightedLoss(loss, weights)


def esdfm_weights(targets: Tensor, q_tn_logits: Tensor, q_dp_logits: Tensor) -> Tensor:
    """Detached, probability-clamped ES-DFM constant-wait weights."""

    q_tn = torch.sigmoid(q_tn_logits).detach().clamp(1e-6, 1.0 - 1e-6)
    q_dp = torch.sigmoid(q_dp_logits).detach().clamp(1e-6, 1.0 - 1e-6)
    positive = 1.0 + q_dp
    negative = (1.0 + q_dp) * q_tn
    return torch.where(targets > 0.5, positive, negative).clamp(1e-4, 2.0)


def esdfm_loss(
    logits: Tensor,
    targets: Tensor,
    q_tn_logits: Tensor,
    q_dp_logits: Tensor,
) -> WeightedLoss:
    weights = esdfm_weights(targets, q_tn_logits, q_dp_logits)
    per_example = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    return WeightedLoss(torch.mean(weights * per_example), weights)
