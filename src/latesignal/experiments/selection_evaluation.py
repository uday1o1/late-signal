"""Truth join for already sealed chronological selection predictions."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from numpy.typing import NDArray

from latesignal.data.manifests import canonical_json_bytes, read_json, write_json_atomic
from latesignal.errors import ConsistencyError
from latesignal.evaluation.metrics import classification_metrics
from latesignal.experiments.predictions import PredictionLedgerIdentity, PredictionLedgerWriter
from latesignal.simulator.production_oracle import SECONDS_PER_DAY, ProductionTruthStore


class SelectionEvaluationFeatures(Protocol):
    click_ids: NDArray[np.void]
    click_times: NDArray[np.float64]
    click_days: NDArray[np.int16]

    @property
    def prepared_manifest_sha256(self) -> str: ...

    def references_for_ids(self, click_ids: list[bytes]) -> NDArray[np.int32]: ...


def verify_selection_run_manifest(path: Path) -> dict[str, object]:
    manifest = read_json(path)
    expected = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if expected != actual or manifest.get("status") != "complete":
        raise ConsistencyError("Selection run manifest is incomplete or forged")
    return manifest


def _prediction_columns(root: Path) -> tuple[list[bytes], NDArray[np.int16], NDArray[np.float64]]:
    click_ids: list[bytes] = []
    days: list[NDArray[np.int16]] = []
    probabilities: list[NDArray[np.float64]] = []
    for path in sorted(root.glob("part-*.parquet")):
        try:
            table = pq.ParquetFile(path).read(columns=["click_id", "click_day", "probability"])
        except (OSError, pa.ArrowException) as error:
            raise ConsistencyError("Sealed selection prediction part could not be read") from error
        try:
            click_ids.extend(bytes.fromhex(value) for value in table["click_id"].to_pylist())
        except (TypeError, ValueError) as error:
            raise ConsistencyError("Selection prediction contains an invalid click ID") from error
        days.append(np.asarray(table["click_day"].to_numpy(zero_copy_only=False), dtype=np.int16))
        probabilities.append(
            np.asarray(table["probability"].to_numpy(zero_copy_only=False), dtype=np.float64)
        )
    if not days or not probabilities:
        raise ConsistencyError("Sealed selection ledger contains no predictions")
    return click_ids, np.concatenate(days), np.concatenate(probabilities)


def evaluate_selection_candidate(
    run_root: Path,
    *,
    truth: ProductionTruthStore,
    features: SelectionEvaluationFeatures,
    output_path: Path | None = None,
    require_ranking_eligible: bool = True,
) -> dict[str, object]:
    """Verify the truth-free seal before reading held-out eventual outcomes."""

    root = run_root.resolve()
    manifest = verify_selection_run_manifest(root / "manifest.json")
    identity = PredictionLedgerIdentity.model_validate(
        read_json(root / "predictions/identity.json")
    )
    seal = PredictionLedgerWriter(root / "predictions", identity).verify_seal()
    if identity.kind != "selection" or (
        require_ranking_eligible and identity.ranking_eligible is not True
    ):
        raise ConsistencyError("Selection prediction ledger is not ranking eligible")
    if (
        features.prepared_manifest_sha256 != identity.data_manifest_sha256
        or truth.prepared_manifest_sha256 != identity.data_manifest_sha256
        or manifest.get("config_sha256") != identity.config_sha256
        or manifest.get("prediction_seal_sha256") != seal.seal_sha256
        or manifest.get("prediction_ledger_sha256") != seal.ledger_sha256
        or manifest.get("truth_joined") is not False
        or manifest.get("selection_mode") != "retrospective_chronological"
    ):
        raise ConsistencyError("Selection evidence identities do not align")
    click_ids, click_days, probabilities = _prediction_columns(root / "predictions")
    references = features.references_for_ids(click_ids)
    if (
        references.size != seal.rows
        or not np.array_equal(features.click_days[references], click_days)
        or set(np.unique(click_days).tolist()) != set(range(25, 35))
    ):
        raise ConsistencyError("Selection predictions do not match the held-out cohort")
    labels = truth.final_labels[references]
    available_at = truth.available_at[references]
    maturity_boundary = float(features.click_times[0]) + 65 * SECONDS_PER_DAY
    if (
        not np.isin(labels, (0, 1)).all()
        or not np.isfinite(available_at).all()
        or np.any(available_at > maturity_boundary)
    ):
        raise ConsistencyError("Selection truth is not fully mature at the authored join boundary")
    truth_digest = hashlib.sha256()
    for click_id, label in zip(click_ids, labels, strict=True):
        truth_digest.update(click_id)
        truth_digest.update(bytes((int(label),)))
    metrics = classification_metrics(labels, probabilities)
    payload: dict[str, object] = {
        "version": 1,
        "status": "complete",
        "truth_joined": True,
        "selection_mode": "retrospective_chronological",
        "run_id": identity.run_id,
        "config_sha256": identity.config_sha256,
        "protocol_sha256": identity.protocol_sha256,
        "prediction_seal_sha256": seal.seal_sha256,
        "prediction_ledger_sha256": seal.ledger_sha256,
        "truth_cohort_sha256": truth_digest.hexdigest(),
        "rows": seal.rows,
        "metrics": metrics,
    }
    payload["evaluation_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    destination = output_path if output_path is not None else root / "selection-evaluation.json"
    if destination.exists():
        existing = read_json(destination)
        if existing != payload:
            raise ConsistencyError(
                "Immutable selection evaluation differs from recomputed evidence"
            )
        return existing
    write_json_atomic(destination, payload)
    return payload
