"""Shared field-aware conversion MLP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class CategoricalSpec:
    bucket_count: int
    embedding_dim: int


class ConversionMLP(nn.Module):
    """The locked V1 conversion backbone with field-specific embeddings."""

    def __init__(
        self,
        categorical_specs: dict[str, CategoricalSpec],
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(categorical_specs) != 17:
            raise ValueError("ConversionMLP requires exactly 17 categorical fields")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.categorical_fields = tuple(sorted(categorical_specs))
        self.embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(spec.bucket_count, spec.embedding_dim)
                for field, spec in sorted(categorical_specs.items())
            }
        )
        input_width = sum(spec.embedding_dim for spec in categorical_specs.values()) + 4
        self.backbone = nn.Sequential(
            nn.Linear(input_width, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.SiLU(),
        )
        self.output = nn.Linear(64, 1)

    def forward(self, categorical: dict[str, Tensor], numeric: Tensor) -> Tensor:
        if tuple(sorted(categorical)) != self.categorical_fields:
            raise ValueError("Categorical batch does not match the model field contract")
        if numeric.ndim != 2 or numeric.shape[1] != 4:
            raise ValueError("Numeric batch must contain two values and two missing indicators")
        batch_size = numeric.shape[0]
        embedded: list[Tensor] = []
        for field in self.categorical_fields:
            values = categorical[field]
            if values.ndim != 1 or values.shape[0] != batch_size:
                raise ValueError(f"Categorical field {field} has an invalid shape")
            embedded.append(self.embeddings[field](values))
        combined = torch.cat([*embedded, numeric], dim=1)
        return cast(Tensor, self.output(self.backbone(combined)).squeeze(1))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
