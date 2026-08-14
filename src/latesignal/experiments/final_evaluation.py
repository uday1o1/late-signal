"""Verified truth join, production slices, and compact final-run evidence."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Protocol

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from numpy.typing import NDArray

from latesignal.data.manifests import (
    canonical_json_bytes,
    read_json,
    sha256_file,
    write_json_atomic,
)
from latesignal.errors import ConsistencyError
from latesignal.evaluation.metrics import classification_metrics
from latesignal.experiments.final_snapshots import (
    FinalSnapshotIdentity,
    FinalSnapshotStore,
)
from latesignal.experiments.predictions import PredictionLedgerIdentity, PredictionLedgerWriter
from latesignal.simulator.production_oracle import SECONDS_PER_DAY, ProductionTruthStore


class FinalEvaluationFeatures(Protocol):
    click_ids: NDArray[np.void]
    click_times: NDArray[np.float64]
    click_days: NDArray[np.int16]
    cold_user: NDArray[np.bool_]
    cold_product: NDArray[np.bool_]
    prior_user_clicks: NDArray[np.int64]
    prior_product_clicks: NDArray[np.int64]
    product_price: NDArray[np.float64]
    device_type_codes: NDArray[np.uint16]
    device_types: tuple[str, ...]

    @property
    def prepared_manifest_sha256(self) -> str: ...

    def references_for_day(self, day: int) -> NDArray[np.int32]: ...


def verify_final_run_manifest(path: Path) -> dict[str, object]:
    manifest = read_json(path)
    expected = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        not isinstance(expected, str)
        or hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != expected
        or manifest.get("status") != "complete"
        or manifest.get("truth_joined") is not False
        or manifest.get("primary_evaluation_mode") != "prequential"
        or manifest.get("evaluation_click_days") != [65, 89]
    ):
        raise ConsistencyError("Final run manifest is incomplete or forged")
    return manifest


def _prediction_probabilities(
    root: Path,
    *,
    expected_days: NDArray[np.int16],
) -> NDArray[np.float32]:
    days: list[NDArray[np.int16]] = []
    probabilities: list[NDArray[np.float32]] = []
    for path in sorted(root.glob("part-*.parquet")):
        try:
            table = pq.ParquetFile(path).read(columns=["click_day", "probability"])
        except (OSError, pa.ArrowException) as error:
            raise ConsistencyError("Sealed final prediction part could not be read") from error
        days.append(np.asarray(table["click_day"].to_numpy(zero_copy_only=False), dtype=np.int16))
        probabilities.append(
            np.asarray(table["probability"].to_numpy(zero_copy_only=False), dtype=np.float32)
        )
    if not days:
        raise ConsistencyError("Sealed final prediction ledger contains no rows")
    joined_days = np.concatenate(days)
    joined_probabilities = np.concatenate(probabilities)
    if (
        not np.array_equal(joined_days, expected_days)
        or joined_probabilities.shape != expected_days.shape
        or not np.isfinite(joined_probabilities).all()
        or np.any(joined_probabilities < 0.0)
        or np.any(joined_probabilities > 1.0)
    ):
        raise ConsistencyError("Final prediction rows do not match the locked cohort order")
    return joined_probabilities


def _price_bins(
    features: FinalEvaluationFeatures,
    final_refs: NDArray[np.int32],
) -> tuple[NDArray[np.uint8], list[dict[str, object]]]:
    burn_in = features.product_price[features.click_days <= 14]
    if burn_in.size == 0 or not np.isfinite(burn_in).all() or np.any(burn_in < 0.0):
        raise ConsistencyError("Burn-in product-price values are invalid")
    thresholds = np.quantile(burn_in, [0.25, 0.5, 0.75], method="linear")
    if not np.isfinite(thresholds).all() or np.any(np.diff(thresholds) < 0.0):
        raise ConsistencyError("Burn-in product-price quantiles are invalid")
    codes = np.searchsorted(thresholds, features.product_price[final_refs], side="right").astype(
        np.uint8
    )
    definitions: list[dict[str, object]] = []
    bounds = [-float("inf"), *thresholds.astype(float).tolist(), float("inf")]
    for index in range(4):
        definitions.append(
            {
                "code": index,
                "name": f"burn_in_quartile_{index + 1}",
                "lower": None if index == 0 else bounds[index],
                "upper": None if index == 3 else bounds[index + 1],
                "lower_inclusive": True,
                "upper_inclusive": False,
            }
        )
    return codes, definitions


def _frequency_codes(values: NDArray[np.int64]) -> NDArray[np.uint8]:
    if np.any(values < 0):
        raise ConsistencyError("Past-only frequency slice contains a negative count")
    return np.select(
        [values == 0, values <= 2, values <= 9],
        [0, 1, 2],
        default=3,
    ).astype(np.uint8)


def _slice_row(
    *,
    dimension: str,
    value: str,
    mask: NDArray[np.bool_],
    labels: NDArray[np.int8],
    probabilities: NDArray[np.float32],
    minimum_examples: int = 10_000,
    minimum_positives: int = 100,
) -> dict[str, object]:
    count = int(np.count_nonzero(mask))
    positives = int(labels[mask].sum())
    eligible = count >= minimum_examples and positives >= minimum_positives
    reason = (
        None
        if eligible
        else "empty"
        if count == 0
        else "insufficient_examples"
        if count < minimum_examples
        else "insufficient_positives"
    )
    return {
        "dimension": dimension,
        "value": value,
        "count": count,
        "positives": positives,
        "ranking_eligible": eligible,
        "suppression_reason": reason,
        "metrics": classification_metrics(labels[mask], probabilities[mask]) if eligible else None,
    }


def _production_slices(
    *,
    features: FinalEvaluationFeatures,
    truth: ProductionTruthStore,
    final_refs: NDArray[np.int32],
    labels: NDArray[np.int8],
    probabilities: NDArray[np.float32],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    result: list[dict[str, object]] = []
    for dimension, bool_values in (
        ("cold_user", features.cold_user[final_refs]),
        ("cold_product", features.cold_product[final_refs]),
    ):
        for flag in (False, True):
            result.append(
                _slice_row(
                    dimension=dimension,
                    value=str(flag).lower(),
                    mask=bool_values == flag,
                    labels=labels,
                    probabilities=probabilities,
                )
            )
    frequency_names = ("0", "1-2", "3-9", "10+")
    for dimension, count_values in (
        ("user_frequency", features.prior_user_clicks[final_refs]),
        ("product_frequency", features.prior_product_clicks[final_refs]),
    ):
        codes = _frequency_codes(count_values)
        for code, name in enumerate(frequency_names):
            result.append(
                _slice_row(
                    dimension=dimension,
                    value=name,
                    mask=codes == code,
                    labels=labels,
                    probabilities=probabilities,
                )
            )
    price_codes, price_definitions = _price_bins(features, final_refs)
    for item in price_definitions:
        raw_code = item["code"]
        if isinstance(raw_code, bool) or not isinstance(raw_code, int):
            raise ConsistencyError("Product-price bin code is malformed")
        code = raw_code
        result.append(
            _slice_row(
                dimension="product_price_bin",
                value=str(item["name"]),
                mask=price_codes == code,
                labels=labels,
                probabilities=probabilities,
            )
        )
    device_codes = features.device_type_codes[final_refs]
    for code, name in enumerate(features.device_types):
        result.append(
            _slice_row(
                dimension="device_type",
                value=name,
                mask=device_codes == code,
                labels=labels,
                probabilities=probabilities,
            )
        )
    positive_delays = truth.conversion_delay_days[final_refs]
    delay_masks = (
        ("[0,1)", (positive_delays >= 0.0) & (positive_delays < 1.0)),
        ("[1,3)", (positive_delays >= 1.0) & (positive_delays < 3.0)),
        ("[3,7)", (positive_delays >= 3.0) & (positive_delays < 7.0)),
        ("[7,14)", (positive_delays >= 7.0) & (positive_delays < 14.0)),
        ("[14,30]", (positive_delays >= 14.0) & (positive_delays <= 30.0)),
    )
    for name, mask in delay_masks:
        result.append(
            _slice_row(
                dimension="positive_conversion_delay",
                value=name,
                mask=mask & (labels == 1),
                labels=labels,
                probabilities=probabilities,
            )
        )
    days = features.click_days[final_refs]
    for first_day in range(65, 90, 5):
        last_day = min(first_day + 4, 89)
        result.append(
            _slice_row(
                dimension="click_day_block",
                value=f"{first_day}-{last_day}",
                mask=(days >= first_day) & (days <= last_day),
                labels=labels,
                probabilities=probabilities,
            )
        )
    return result, price_definitions


def _write_compact_probabilities(
    root: Path,
    *,
    probabilities: NDArray[np.float32],
    identity: PredictionLedgerIdentity,
    prediction_seal_sha256: str,
    truth_cohort_sha256: str,
) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "primary-probabilities.npy"
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        expected = manifest.get("manifest_sha256")
        unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        sha256, size = sha256_file(path)
        if (
            not isinstance(expected, str)
            or hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != expected
            or manifest.get("probabilities_sha256") != sha256
            or manifest.get("probabilities_bytes") != size
        ):
            raise ConsistencyError("Compact final probabilities are inconsistent")
        return manifest
    temporary = root / ".primary-probabilities.npy.tmp"
    try:
        with temporary.open("xb") as output:
            np.save(output, probabilities.astype(np.float32, copy=False), allow_pickle=False)
            output.flush()
            os.fsync(output.fileno())
        loaded = np.load(temporary, allow_pickle=False)
        if loaded.dtype != np.float32 or not np.array_equal(loaded, probabilities):
            raise ConsistencyError("Compact final probability round trip changed values")
        os.replace(temporary, path)
        sha256, size = sha256_file(path)
        payload: dict[str, object] = {
            "version": 1,
            "status": "verified_compact_primary",
            "run_id": identity.run_id,
            "method": identity.method,
            "seed": identity.seed,
            "config_sha256": identity.config_sha256,
            "protocol_sha256": identity.protocol_sha256,
            "protocol_lock_sha256": identity.protocol_lock_sha256,
            "data_manifest_sha256": identity.data_manifest_sha256,
            "rows": int(probabilities.size),
            "dtype": "float32",
            "ordered_id_sha256": identity.expected_ordered_id_sha256,
            "prediction_seal_sha256": prediction_seal_sha256,
            "truth_cohort_sha256": truth_cohort_sha256,
            "probabilities_path": path.name,
            "probabilities_sha256": sha256,
            "probabilities_bytes": size,
        }
        payload["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        write_json_atomic(manifest_path, payload)
        return payload
    finally:
        temporary.unlink(missing_ok=True)


def evaluate_final_run(
    run_root: Path,
    *,
    truth: ProductionTruthStore,
    features: FinalEvaluationFeatures,
) -> dict[str, object]:
    """Verify every truth-free seal before joining eventual final-period outcomes."""

    root = run_root.resolve()
    manifest = verify_final_run_manifest(root / "manifest.json")
    primary_root = root / "predictions" / "primary"
    primary_identity = PredictionLedgerIdentity.model_validate(
        read_json(primary_root / "identity.json")
    )
    primary_seal = PredictionLedgerWriter(primary_root, primary_identity).verify_seal()
    final_refs = np.concatenate([features.references_for_day(day) for day in range(65, 90)]).astype(
        np.int32
    )
    expected_ids_sha256 = hashlib.sha256(features.click_ids[final_refs].tobytes()).hexdigest()
    expected_days = features.click_days[final_refs]
    if (
        primary_identity.kind != "final_prequential"
        or primary_identity.expected_ordered_id_sha256 != expected_ids_sha256
        or primary_identity.expected_rows != final_refs.size
        or primary_identity.data_manifest_sha256 != features.prepared_manifest_sha256
        or truth.prepared_manifest_sha256 != features.prepared_manifest_sha256
        or manifest.get("config_sha256") != primary_identity.config_sha256
        or manifest.get("protocol_lock_sha256") != primary_identity.protocol_lock_sha256
        or manifest.get("primary_prediction_seal_sha256") != primary_seal.seal_sha256
        or manifest.get("primary_prediction_ledger_sha256") != primary_seal.ledger_sha256
    ):
        raise ConsistencyError("Final primary evidence identities do not align")
    labels = truth.final_labels[final_refs]
    available_at = truth.available_at[final_refs]
    maturity_boundary = float(features.click_times[0]) + 120 * SECONDS_PER_DAY
    if (
        not np.isin(labels, (0, 1)).all()
        or not np.isfinite(available_at).all()
        or np.any(available_at > maturity_boundary)
    ):
        raise ConsistencyError("Final truth is not mature at the authored join boundary")
    probabilities = _prediction_probabilities(primary_root, expected_days=expected_days)
    truth_digest = hashlib.sha256()
    truth_digest.update(features.click_ids[final_refs].tobytes())
    truth_digest.update(labels.tobytes())
    truth_cohort_sha256 = truth_digest.hexdigest()
    overall = classification_metrics(labels, probabilities)
    slices, price_bins = _production_slices(
        features=features,
        truth=truth,
        final_refs=final_refs,
        labels=labels,
        probabilities=probabilities,
    )
    intermediate_manifest = manifest.get("intermediate_predictions")
    total_credits = manifest.get("credits")
    if (
        not isinstance(intermediate_manifest, list)
        or len(intermediate_manifest) != 4
        or isinstance(total_credits, bool)
        or not isinstance(total_credits, int)
    ):
        raise ConsistencyError("Final manifest has incomplete intermediate evidence")
    intermediate: list[dict[str, object]] = []
    for entry in intermediate_manifest:
        if not isinstance(entry, dict):
            raise ConsistencyError("Final intermediate manifest entry is malformed")
        fraction = entry.get("budget_fraction")
        credits = entry.get("credits_at_snapshot")
        if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
            raise ConsistencyError("Final intermediate fraction is malformed")
        if not isinstance(credits, int) or isinstance(credits, bool):
            raise ConsistencyError("Final intermediate credit boundary is malformed")
        name = f"fraction-{round(float(fraction) * 100):03d}"
        ledger_root = root / "predictions" / "intermediate" / name
        identity = PredictionLedgerIdentity.model_validate(read_json(ledger_root / "identity.json"))
        seal = PredictionLedgerWriter(ledger_root, identity).verify_seal()
        snapshot_identity = FinalSnapshotIdentity(
            version=1,
            run_id=primary_identity.run_id,
            method=primary_identity.method,
            seed=primary_identity.seed,
            config_sha256=primary_identity.config_sha256,
            protocol_sha256=primary_identity.protocol_sha256,
            protocol_lock_sha256=str(primary_identity.protocol_lock_sha256),
            budget_fraction=float(fraction),
            credits_at_snapshot=credits,
            total_credits=total_credits,
        )
        snapshot = FinalSnapshotStore(root / "snapshots").verify(snapshot_identity)
        if (
            identity.kind != "intermediate"
            or identity.expected_ordered_id_sha256 != expected_ids_sha256
            or identity.ranking_eligible is not False
            or entry.get("mode") != "retrospective_inference_only"
            or entry.get("ranking_eligible") is not False
            or entry.get("prediction_seal_sha256") != seal.seal_sha256
            or entry.get("prediction_ledger_sha256") != seal.ledger_sha256
            or entry.get("model_sha256") != snapshot.model_sha256
        ):
            raise ConsistencyError("Final intermediate evidence identities do not align")
        values = _prediction_probabilities(ledger_root, expected_days=expected_days)
        intermediate.append(
            {
                "budget_fraction": float(fraction),
                "credits_at_snapshot": credits,
                "mode": "retrospective_inference_only",
                "ranking_eligible": False,
                "model_sha256": snapshot.model_sha256,
                "prediction_seal_sha256": seal.seal_sha256,
                "prediction_ledger_sha256": seal.ledger_sha256,
                "metrics": classification_metrics(labels, values),
            }
        )
    compact = _write_compact_probabilities(
        root / "compact",
        probabilities=probabilities,
        identity=primary_identity,
        prediction_seal_sha256=primary_seal.seal_sha256,
        truth_cohort_sha256=truth_cohort_sha256,
    )
    payload: dict[str, object] = {
        "version": 1,
        "status": "complete",
        "truth_joined": True,
        "run_id": primary_identity.run_id,
        "method": primary_identity.method,
        "seed": primary_identity.seed,
        "ranking_eligible": primary_identity.ranking_eligible,
        "config_sha256": primary_identity.config_sha256,
        "protocol_sha256": primary_identity.protocol_sha256,
        "protocol_lock_sha256": primary_identity.protocol_lock_sha256,
        "data_manifest_sha256": primary_identity.data_manifest_sha256,
        "prediction_seal_sha256": primary_seal.seal_sha256,
        "prediction_ledger_sha256": primary_seal.ledger_sha256,
        "truth_cohort_sha256": truth_cohort_sha256,
        "rows": int(final_refs.size),
        "period": [65, 89],
        "primary_mode": "prequential",
        "overall": overall,
        "slices": slices,
        "price_bins": {
            "fit_click_days": [0, 14],
            "method": "linear_quartiles",
            "definitions": price_bins,
        },
        "intermediate": intermediate,
        "compact_primary_manifest_sha256": compact["manifest_sha256"],
    }
    payload["evaluation_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    destination = root / "evaluation.json"
    if destination.exists():
        existing = read_json(destination)
        if existing != payload:
            raise ConsistencyError("Immutable final evaluation differs from recomputed evidence")
        return existing
    write_json_atomic(destination, payload)
    return payload
