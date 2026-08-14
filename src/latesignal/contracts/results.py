"""Sealed final-period evaluation input contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from latesignal.errors import ConsistencyError


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    click_id: str
    click_day: int
    final_label: int
    probability: float
    cold_user: bool
    cold_product: bool
    prior_user_clicks: int
    prior_product_clicks: int
    product_price_bin: str
    device_type: str
    conversion_delay_days: float | None

    def __post_init__(self) -> None:
        if not self.click_id:
            raise ValueError("click_id must be nonempty")
        if self.final_label not in {0, 1}:
            raise ValueError("final_label must be binary")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must lie in [0, 1]")
        if self.prior_user_clicks < 0 or self.prior_product_clicks < 0:
            raise ValueError("Past-only frequency counts must be nonnegative")
        if self.cold_user != (self.prior_user_clicks == 0):
            raise ConsistencyError("cold_user does not match past-only user frequency")
        if self.cold_product != (self.prior_product_clicks == 0):
            raise ConsistencyError("cold_product does not match past-only product frequency")
        if self.final_label == 1:
            if self.conversion_delay_days is None or not 0.0 <= self.conversion_delay_days <= 30.0:
                raise ConsistencyError("Positive evaluation row needs a legal conversion delay")
        elif self.conversion_delay_days is not None:
            raise ConsistencyError("Negative evaluation row cannot have a conversion delay")
        if not self.product_price_bin or not self.device_type:
            raise ValueError("Price-bin and device slice values must be nonempty")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvaluationExample:
        try:
            return cls(
                click_id=str(value["click_id"]),
                click_day=int(value["click_day"]),
                final_label=int(value["final_label"]),
                probability=float(value["probability"]),
                cold_user=_strict_bool(value["cold_user"], "cold_user"),
                cold_product=_strict_bool(value["cold_product"], "cold_product"),
                prior_user_clicks=int(value["prior_user_clicks"]),
                prior_product_clicks=int(value["prior_product_clicks"]),
                product_price_bin=str(value["product_price_bin"]),
                device_type=str(value["device_type"]),
                conversion_delay_days=(
                    None
                    if value["conversion_delay_days"] is None
                    else float(value["conversion_delay_days"])
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ConsistencyError("Evaluation example is malformed") from error


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConsistencyError(f"Evaluation {name} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    method: str
    seed: int
    examples: tuple[EvaluationExample, ...]
    ranking_eligible: bool
    sealed: bool
    period_first_day: int = 65
    period_last_day: int = 89

    def __post_init__(self) -> None:
        if not self.sealed:
            raise ConsistencyError("Final evaluation refuses an unsealed prediction ledger")
        if not self.method or not self.examples:
            raise ConsistencyError("Evaluation dataset must name a method and contain rows")
        identifiers = {example.click_id for example in self.examples}
        if len(identifiers) != len(self.examples):
            raise ConsistencyError("Evaluation dataset contains duplicate click IDs")
        if any(
            not self.period_first_day <= example.click_day <= self.period_last_day
            for example in self.examples
        ):
            raise ConsistencyError("Evaluation dataset contains a row outside the locked period")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvaluationDataset:
        examples = value.get("examples")
        if not isinstance(examples, list):
            raise ConsistencyError("Evaluation input examples must be a list")
        if not all(isinstance(item, dict) for item in examples):
            raise ConsistencyError("Every evaluation input example must be an object")
        method = value.get("method")
        seed = value.get("seed")
        ranking_eligible = value.get("ranking_eligible")
        sealed = value.get("sealed")
        period_first_day = value.get("period_first_day", 65)
        period_last_day = value.get("period_last_day", 89)
        if (
            not isinstance(method, str)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(ranking_eligible, bool)
            or not isinstance(sealed, bool)
            or isinstance(period_first_day, bool)
            or not isinstance(period_first_day, int)
            or isinstance(period_last_day, bool)
            or not isinstance(period_last_day, int)
        ):
            raise ConsistencyError("Evaluation input header is malformed")
        return cls(
            method=method,
            seed=seed,
            examples=tuple(EvaluationExample.from_dict(item) for item in examples),
            ranking_eligible=ranking_eligible,
            sealed=sealed,
            period_first_day=period_first_day,
            period_last_day=period_last_day,
        )
