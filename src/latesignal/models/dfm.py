"""Shared conversion backbone with the DFM exponential-rate head."""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn

from latesignal.models.conversion_mlp import ConversionMLP


class DelayedFeedbackMLP(nn.Module):
    def __init__(self, conversion: ConversionMLP) -> None:
        super().__init__()
        self.conversion = conversion
        self.rate_output = nn.Linear(64, 1)

    def forward(self, categorical: dict[str, Tensor], numeric: Tensor) -> tuple[Tensor, Tensor]:
        representation = self.conversion.representation(categorical, numeric)
        conversion_logits = self.conversion.output(representation).squeeze(1)
        rate_logits = self.rate_output(representation).squeeze(1)
        return cast(Tensor, conversion_logits), cast(Tensor, rate_logits)
