from __future__ import annotations

from pathlib import Path

import pytest

from latesignal.errors import ConsistencyError
from latesignal.experiments.predictions import (
    PredictionLedgerIdentity,
    PredictionLedgerWriter,
    ordered_click_id_sha256,
)


def _ids(count: int) -> list[str]:
    return [f"{index:064x}" for index in range(count)]


def _identity(
    click_ids: list[str], *, expected_rows: int | None = None
) -> PredictionLedgerIdentity:
    return PredictionLedgerIdentity(
        version=1,
        kind="final_prequential",
        run_id="fixed-wait-seed-17",
        method="fixed_wait",
        seed=17,
        period_first_day=65,
        period_last_day=89,
        protocol_sha256="1" * 64,
        protocol_lock_sha256="2" * 64,
        config_sha256="3" * 64,
        data_manifest_sha256="4" * 64,
        expected_rows=len(click_ids) if expected_rows is None else expected_rows,
        expected_ordered_id_sha256=ordered_click_id_sha256(click_ids),
        ranking_eligible=True,
    )


def _append(writer: PredictionLedgerWriter, click_ids: list[str], *, day: int = 65) -> None:
    writer.append(
        click_ids=click_ids,
        click_days=[day] * len(click_ids),
        probabilities=[0.25 + index / 10.0 for index in range(len(click_ids))],
        model_versions=list(range(len(click_ids))),
    )


def test_prediction_ledger_resumes_then_seals_without_truth(tmp_path: Path) -> None:
    click_ids = _ids(4)
    identity = _identity(click_ids)
    root = tmp_path / "predictions"
    _append(PredictionLedgerWriter(root, identity), click_ids[:2])
    resumed = PredictionLedgerWriter(root, identity)
    _append(resumed, click_ids[2:], day=66)

    sealed = resumed.seal()
    verified = PredictionLedgerWriter(root, identity).verify_seal()

    assert sealed.rows == verified.rows == 4
    assert sealed.seal_sha256 == verified.seal_sha256
    assert sealed.ordered_id_sha256 == ordered_click_id_sha256(click_ids)
    with pytest.raises(ConsistencyError, match="immutable"):
        _append(resumed, _ids(1))


def test_prediction_seal_refuses_partial_or_duplicate_cohort(tmp_path: Path) -> None:
    click_ids = _ids(3)
    partial = PredictionLedgerWriter(tmp_path / "partial", _identity(click_ids))
    _append(partial, click_ids[:2])
    with pytest.raises(ConsistencyError, match="locked evaluation cohort"):
        partial.seal()

    duplicate_ids = [click_ids[0], click_ids[1], click_ids[0]]
    duplicate = PredictionLedgerWriter(
        tmp_path / "duplicate",
        _identity(duplicate_ids),
    )
    _append(duplicate, duplicate_ids)
    with pytest.raises(ConsistencyError, match="duplicate click ID"):
        duplicate.seal()


def test_prediction_seal_detects_file_tampering(tmp_path: Path) -> None:
    click_ids = _ids(2)
    writer = PredictionLedgerWriter(tmp_path / "predictions", _identity(click_ids))
    _append(writer, click_ids)
    writer.seal()
    part = next(writer.root.glob("part-*.parquet"))
    part.write_bytes(b"not parquet")

    with pytest.raises(ConsistencyError, match="could not be verified"):
        writer.verify_seal()


def test_prediction_identity_enforces_selection_and_budget_boundaries() -> None:
    click_ids = _ids(1)
    final = _identity(click_ids)
    selection = final.model_copy(
        update={
            "kind": "selection",
            "period_first_day": 25,
            "period_last_day": 34,
            "protocol_lock_sha256": None,
        }
    )
    assert PredictionLedgerIdentity.model_validate(selection.model_dump()).kind == "selection"

    with pytest.raises(ValueError, match="Final predictions require"):
        PredictionLedgerIdentity.model_validate(
            final.model_copy(update={"protocol_lock_sha256": None}).model_dump()
        )
    with pytest.raises(ValueError, match="budget fraction"):
        PredictionLedgerIdentity.model_validate(
            final.model_copy(
                update={
                    "kind": "intermediate",
                    "budget_fraction": 0.33,
                    "credits_at_snapshot": 20,
                }
            ).model_dump()
        )
