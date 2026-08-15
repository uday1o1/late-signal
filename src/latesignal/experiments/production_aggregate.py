"""Verified aggregate-only final analysis and static report input generation."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray

from latesignal.data.manifests import (
    canonical_json_bytes,
    read_json,
    sha256_file,
    write_json_atomic,
)
from latesignal.errors import ConsistencyError
from latesignal.evaluation.analysis import (
    BudgetQualityPoint,
    compute_pareto_table,
    validate_intermediate_budget,
)
from latesignal.evaluation.bootstrap import (
    CompactBootstrapMetrics,
    compact_bootstrap_metrics,
    contiguous_day_bootstrap_weights,
    paired_compact_interval,
)
from latesignal.evaluation.compare import scheduler_success
from latesignal.experiments.final_evaluation import verify_final_run_manifest
from latesignal.experiments.production_final import (
    FinalPlanInputs,
    ProductionFinalPlan,
    final_online_plans,
)
from latesignal.experiments.production_offline import offline_reference_plans
from latesignal.features.store import RuntimeFeatureStore
from latesignal.reporting.model import ReportInput
from latesignal.reporting.render import render_report
from latesignal.simulator.production_oracle import ProductionTruthStore

_METRICS = ("log_loss", "brier_score", "pr_auc", "roc_auc")
_BLOCKS = (1, 3, 7)
_LEAKAGE_CONTROLS = frozenset(
    {
        "forbidden_sale_field",
        "forbidden_conversion_delay_field",
        "final_period_normalizer_fit",
        "global_cold_status",
        "reveal_before_prediction",
        "monitoring_training_reuse",
        "early_truth_availability",
    }
)


@dataclass(frozen=True, slots=True)
class _OnlineEvidence:
    plan: ProductionFinalPlan
    manifest: dict[str, Any]
    evaluation: dict[str, Any]
    probability_sha256: str
    probabilities: NDArray[np.float32]

    @property
    def display_name(self) -> str:
        return self.plan.method if self.plan.study == "study_a" else self.plan.scheduler


def _verify_hashed(value: dict[str, Any], digest_name: str, description: str) -> None:
    expected = value.get(digest_name)
    unsigned = {key: item for key, item in value.items() if key != digest_name}
    if (
        not isinstance(expected, str)
        or hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != expected
    ):
        raise ConsistencyError(f"{description} does not match its digest")


def _truth_identity(
    features: RuntimeFeatureStore,
    truth: ProductionTruthStore,
) -> tuple[NDArray[np.int32], NDArray[np.int8], NDArray[np.int16], str, str]:
    references = np.concatenate([features.references_for_day(day) for day in range(65, 90)]).astype(
        np.int32
    )
    labels = truth.final_labels[references]
    days = features.click_days[references]
    if (
        references.size == 0
        or not np.array_equal(np.unique(days), np.arange(65, 90))
        or not np.isin(labels, (0, 1)).all()
        or np.unique(labels).size != 2
    ):
        raise ConsistencyError("Final aggregate truth cohort is malformed")
    identifiers_sha256 = hashlib.sha256(features.click_ids[references].tobytes()).hexdigest()
    digest = hashlib.sha256()
    digest.update(features.click_ids[references].tobytes())
    digest.update(labels.tobytes())
    return references, labels, days, identifiers_sha256, digest.hexdigest()


def _run_root(output_root: Path, plan: ProductionFinalPlan) -> Path:
    study = "study-a" if plan.study == "study_a" else "study-b"
    return output_root / study / "runs" / plan.run_id


def _verify_final_coordinators(
    output_root: Path,
    online_plans: tuple[ProductionFinalPlan, ...],
    offline_run_ids: set[str],
    *,
    protocol_lock_sha256: str,
) -> None:
    combined = read_json(output_root / "final-manifest.json")
    online = read_json(output_root / "online-manifest.json")
    offline = read_json(output_root / "offline-manifest.json")
    for value, description in (
        (combined, "Combined final manifest"),
        (online, "Online final manifest"),
        (offline, "Offline final manifest"),
    ):
        _verify_hashed(value, "manifest_sha256", description)
    online_completed = online.get("completed_runs")
    offline_completed = offline.get("completed_runs")
    if (
        combined.get("status") != "complete"
        or combined.get("completed_count") != 39
        or combined.get("protocol_lock_sha256") != protocol_lock_sha256
        or combined.get("online_manifest_sha256") != online.get("manifest_sha256")
        or combined.get("offline_manifest_sha256") != offline.get("manifest_sha256")
        or online.get("status") != "complete"
        or online.get("completed_count") != 33
        or offline.get("status") != "complete"
        or offline.get("completed_count") != 6
        or not isinstance(online_completed, list)
        or not isinstance(offline_completed, list)
        or {item.get("run_id") for item in online_completed if isinstance(item, dict)}
        != {plan.run_id for plan in online_plans}
        or {item.get("run_id") for item in offline_completed if isinstance(item, dict)}
        != offline_run_ids
    ):
        raise ConsistencyError("Final coordinator manifests do not cover the exact locked matrix")


def _load_online_evidence(
    output_root: Path,
    plan: ProductionFinalPlan,
    *,
    expected_rows: int,
    ordered_id_sha256: str,
    truth_cohort_sha256: str,
) -> _OnlineEvidence:
    root = _run_root(output_root, plan)
    manifest = verify_final_run_manifest(root / "manifest.json")
    evaluation = read_json(root / "evaluation.json")
    retention = read_json(root / "retention.json")
    compact = read_json(root / "compact" / "manifest.json")
    for value, digest_name, description in (
        (evaluation, "evaluation_sha256", "Final evaluation"),
        (retention, "retention_sha256", "Final retention receipt"),
        (compact, "manifest_sha256", "Final compact manifest"),
    ):
        _verify_hashed(value, digest_name, description)
    probability_path = root / "compact" / "primary-probabilities.npy"
    probability_sha256, probability_bytes = sha256_file(probability_path)
    probabilities = np.load(probability_path, allow_pickle=False)
    if (
        manifest.get("config_sha256") != plan.canonical_sha256
        or manifest.get("protocol_lock_sha256") != plan.protocol_lock_sha256
        or evaluation.get("status") != "complete"
        or evaluation.get("truth_joined") is not True
        or evaluation.get("config_sha256") != plan.canonical_sha256
        or retention.get("status") != "verified_and_pruned"
        or retention.get("config_sha256") != plan.canonical_sha256
        or retention.get("manifest_sha256") != manifest.get("manifest_sha256")
        or retention.get("evaluation_sha256") != evaluation.get("evaluation_sha256")
        or retention.get("compact_primary_manifest_sha256") != compact.get("manifest_sha256")
        or compact.get("config_sha256") != plan.canonical_sha256
        or compact.get("rows") != expected_rows
        or compact.get("ordered_id_sha256") != ordered_id_sha256
        or compact.get("truth_cohort_sha256") != truth_cohort_sha256
        or compact.get("probabilities_sha256") != probability_sha256
        or compact.get("probabilities_bytes") != probability_bytes
        or probabilities.dtype != np.float32
        or probabilities.shape != (expected_rows,)
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
    ):
        raise ConsistencyError("Final compact online evidence does not align")
    return _OnlineEvidence(plan, manifest, evaluation, probability_sha256, probabilities)


def _bootstrap_cache(
    aggregate_root: Path,
    evidence: _OnlineEvidence,
    *,
    labels: NDArray[np.int8],
    days: NDArray[np.int16],
    truth_cohort_sha256: str,
    block_days: Literal[1, 3, 7],
    replicates: int,
    bootstrap_seed: int,
    device: Literal["cpu", "cuda"],
    batch_replicates: int | None,
) -> CompactBootstrapMetrics:
    root = aggregate_root / "bootstrap" / evidence.plan.run_id / f"block-{block_days}"
    root.mkdir(parents=True, exist_ok=True)
    array_path = root / "replicates.npz"
    manifest_path = root / "manifest.json"
    expected_weights = contiguous_day_bootstrap_weights(
        days,
        block_days=block_days,
        replicates=replicates,
        bootstrap_seed=bootstrap_seed,
    )
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        _verify_hashed(manifest, "manifest_sha256", "Bootstrap cache manifest")
        sha256, size = sha256_file(array_path)
        try:
            with np.load(array_path, allow_pickle=False) as stored:
                point_values = np.asarray(stored["point"], dtype=np.float64)
                replicate_values = np.asarray(stored["replicates"], dtype=np.float64)
                day_weights = np.asarray(stored["day_weights"], dtype=np.int16)
        except (OSError, KeyError, ValueError) as error:
            raise ConsistencyError("Bootstrap cache arrays could not be read") from error
        if (
            manifest.get("status") != "complete"
            or manifest.get("run_id") != evidence.plan.run_id
            or manifest.get("config_sha256") != evidence.plan.canonical_sha256
            or manifest.get("probability_sha256") != evidence.probability_sha256
            or manifest.get("truth_cohort_sha256") != truth_cohort_sha256
            or manifest.get("block_days") != block_days
            or manifest.get("replicates") != replicates
            or manifest.get("bootstrap_seed") != bootstrap_seed
            or manifest.get("arrays_sha256") != sha256
            or manifest.get("arrays_bytes") != size
            or point_values.shape != (4,)
            or replicate_values.shape != (4, replicates)
            or not np.isfinite(point_values).all()
            or not np.isfinite(replicate_values).all()
            or not np.array_equal(day_weights, expected_weights)
        ):
            raise ConsistencyError("Bootstrap cache identity or arrays changed")
        return CompactBootstrapMetrics(
            block_days=block_days,
            bootstrap_seed=bootstrap_seed,
            day_weights=day_weights,
            point=dict(zip(_METRICS, point_values.tolist(), strict=True)),
            replicates={metric: replicate_values[index] for index, metric in enumerate(_METRICS)},
        )
    result = compact_bootstrap_metrics(
        labels,
        evidence.probabilities,
        days,
        block_days=block_days,
        replicates=replicates,
        bootstrap_seed=bootstrap_seed,
        device=device,
        batch_replicates=batch_replicates,
    )
    temporary = root / ".replicates.npz.tmp"
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("xb") as output:
            np.savez_compressed(
                output,
                point=np.asarray([result.point[name] for name in _METRICS], dtype=np.float64),
                replicates=np.vstack([result.replicates[name] for name in _METRICS]),
                day_weights=result.day_weights,
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, array_path)
    finally:
        temporary.unlink(missing_ok=True)
    sha256, size = sha256_file(array_path)
    payload: dict[str, object] = {
        "version": 1,
        "status": "complete",
        "run_id": evidence.plan.run_id,
        "config_sha256": evidence.plan.canonical_sha256,
        "probability_sha256": evidence.probability_sha256,
        "truth_cohort_sha256": truth_cohort_sha256,
        "block_days": block_days,
        "replicates": replicates,
        "bootstrap_seed": bootstrap_seed,
        "metric_order": list(_METRICS),
        "arrays_sha256": sha256,
        "arrays_bytes": size,
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    write_json_atomic(manifest_path, payload)
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def _quality_gate(path: Path, *, lock_sha256: str, git_commit: str) -> list[dict[str, object]]:
    quality = read_json(path)
    _verify_hashed(quality, "manifest_sha256", "Final quality gate")
    controls = quality.get("leakage_controls")
    if (
        quality.get("status") != "passed"
        or quality.get("protocol_lock_sha256") != lock_sha256
        or quality.get("git_commit") != git_commit
        or not isinstance(controls, list)
        or not all(isinstance(item, dict) for item in controls)
        or {item.get("control") for item in controls} != _LEAKAGE_CONTROLS
        or any(item.get("status") != "passed" or not item.get("evidence") for item in controls)
    ):
        raise ConsistencyError("Final aggregate requires the exact passing quality gate")
    return [dict(item) for item in controls if isinstance(item, dict)]


def _mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise ConsistencyError("Aggregate mean received missing or non-finite values")
    return float(np.mean(values))


def _compute_fields(manifest: dict[str, Any]) -> tuple[int, int, float, float]:
    compute = manifest.get("compute")
    if not isinstance(compute, dict):
        raise ConsistencyError("Final run compute accounting is missing")
    auxiliary = compute.get("auxiliary")
    if not isinstance(auxiliary, list) or not all(isinstance(item, dict) for item in auxiliary):
        raise ConsistencyError("Final auxiliary compute accounting is malformed")
    auxiliary_steps = sum(int(item.get("steps", 0)) for item in auxiliary)
    auxiliary_examples = sum(int(item.get("examples", 0)) for item in auxiliary)
    wall_names = (
        "initialization_seconds",
        "core_training_seconds",
        "auxiliary_training_seconds",
        "primary_prediction_seconds",
        "intermediate_prediction_seconds",
        "monitoring_prediction_seconds",
        "checkpoint_seconds",
        "snapshot_seconds",
    )
    wall = sum(float(compute.get(name, 0.0)) for name in wall_names)
    peak_bytes = max(
        int(compute.get("peak_host_memory_bytes", 0)),
        int(compute.get("peak_accelerator_memory_bytes", 0)),
    )
    if auxiliary_steps < 0 or auxiliary_examples < 0 or wall < 0.0 or peak_bytes < 0:
        raise ConsistencyError("Final compute accounting contains a negative value")
    return auxiliary_steps, auxiliary_examples, wall, peak_bytes / 1024**3


def _aggregate_report(
    *,
    inputs: FinalPlanInputs,
    online: list[_OnlineEvidence],
    offline_evaluations: list[dict[str, Any]],
    bootstrap: dict[tuple[str, int, int], CompactBootstrapMetrics],
    prepared_manifest: dict[str, Any],
    quality_controls: list[dict[str, object]],
) -> tuple[ReportInput, dict[str, object]]:
    lock = inputs.protocol_lock
    evaluations: list[dict[str, object]] = []
    slices: list[dict[str, object]] = []
    for item in online:
        evaluations.append(
            {
                "method": item.display_name,
                "seed": item.plan.seed,
                "ranking_eligible": item.plan.ranking_eligible,
                "metrics": item.evaluation["overall"],
            }
        )
        raw_slices = item.evaluation.get("slices")
        if not isinstance(raw_slices, list):
            raise ConsistencyError("Final evaluation slices are missing")
        for raw in raw_slices:
            if not isinstance(raw, dict):
                raise ConsistencyError("Final evaluation slice is malformed")
            eligible = raw.get("ranking_eligible") is True and item.plan.ranking_eligible
            metrics = raw.get("metrics")
            slices.append(
                {
                    "method": item.display_name,
                    "seed": item.plan.seed,
                    "dimension": raw["dimension"],
                    "value": raw["value"],
                    "count": raw["count"],
                    "positives": raw["positives"],
                    "ranking_eligible": eligible,
                    "suppression_reason": (
                        None
                        if eligible
                        else raw.get("suppression_reason") or "non_ranking_reference"
                    ),
                    "log_loss": metrics.get("log_loss")
                    if eligible and isinstance(metrics, dict)
                    else None,
                }
            )
    for evaluation in offline_evaluations:
        evaluations.append(
            {
                "method": evaluation["name"],
                "seed": evaluation["seed"],
                "ranking_eligible": False,
                "metrics": evaluation["overall"],
            }
        )

    paired_rows: list[dict[str, object]] = []
    comparisons: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    study_a_candidates = (
        "immediate_fake_negative",
        "fixed_wait",
        "dfm",
        "fnw",
        "es_dfm",
    )
    study_b_candidates = ("fixed_early", "fixed_midpoint", "calibration_drift")
    for study, control_name, candidates in (
        ("study_a", "complete_wait", study_a_candidates),
        ("study_b", "fixed_deadline", study_b_candidates),
    ):
        for candidate_name in candidates:
            comparison_metrics: dict[str, dict[str, object]] = {}
            for metric in _METRICS:
                blocks: dict[str, object] = {}
                for block in _BLOCKS:
                    control = {
                        seed: bootstrap[(f"{study}:{control_name}", seed, block)]
                        for seed in (17, 41, 73)
                    }
                    candidate = {
                        seed: bootstrap[(f"{study}:{candidate_name}", seed, block)]
                        for seed in (17, 41, 73)
                    }
                    interval = paired_compact_interval(
                        control,
                        candidate,
                        metric=metric,  # type: ignore[arg-type]
                    )
                    value = interval.as_dict()
                    blocks[str(block)] = value
                    paired_rows.append(
                        {
                            "control": control_name,
                            "candidate": candidate_name,
                            "metric": metric,
                            "block_days": block,
                            "replicates": interval.replicates,
                            "point_difference": interval.point_difference,
                            "lower_95": interval.lower_95,
                            "upper_95": interval.upper_95,
                            "seed_differences": [
                                {"seed": item.seed, "difference": item.difference}
                                for item in interval.seed_differences
                            ],
                        }
                    )
                comparison_metrics[metric] = blocks
            comparisons[(control_name, candidate_name)] = comparison_metrics

    method_rows: list[dict[str, object]] = []
    compute_values: list[tuple[str, float, int, int]] = []
    compute_extra: dict[str, tuple[float, float]] = {}
    intermediate_points: list[BudgetQualityPoint] = []
    for name in (
        "complete_wait",
        "immediate_fake_negative",
        "fixed_wait",
        "dfm",
        "fnw",
        "es_dfm",
        "oracle_reference",
    ):
        runs = [
            item for item in online if item.plan.study == "study_a" and item.plan.method == name
        ]
        if len(runs) != 3:
            raise ConsistencyError("Final Study A aggregate is missing a seed")
        core_steps = {int(item.manifest["core_optimizer_steps"]) for item in runs}
        core_examples = {int(item.manifest["core_optimizer_examples"]) for item in runs}
        credits = {int(item.manifest["credits"]) for item in runs}
        compute = [_compute_fields(item.manifest) for item in runs]
        auxiliary_steps = {item[0] for item in compute}
        auxiliary_examples = {item[1] for item in compute}
        if any(
            len(values) != 1
            for values in (core_steps, core_examples, credits, auxiliary_steps, auxiliary_examples)
        ):
            raise ConsistencyError("Study A seed compute budgets do not match")
        method_rows.append(
            {
                "method": name,
                "deployable": runs[0].plan.deployable,
                "ranking_eligible": runs[0].plan.ranking_eligible,
                "credits": next(iter(credits)),
                "core_optimizer_steps": next(iter(core_steps)),
                "core_optimizer_examples": next(iter(core_examples)),
                "auxiliary_optimizer_steps": next(iter(auxiliary_steps)),
                "auxiliary_optimizer_examples": next(iter(auxiliary_examples)),
                "status": "complete",
            }
        )
        log_loss = _mean([float(item.evaluation["overall"]["log_loss"]) for item in runs])
        core = next(iter(core_examples))
        total = core + next(iter(auxiliary_examples))
        compute_values.append((name, log_loss, core, total))
        compute_extra[name] = (
            _mean([item[2] for item in compute]),
            max(item[3] for item in compute),
        )
        for fraction in (0.25, 0.5, 0.75, 1.0):
            values = []
            for item in runs:
                raw = item.evaluation.get("intermediate")
                if not isinstance(raw, list):
                    raise ConsistencyError("Intermediate evaluation evidence is missing")
                match = next(
                    (
                        entry
                        for entry in raw
                        if isinstance(entry, dict) and entry.get("budget_fraction") == fraction
                    ),
                    None,
                )
                if not isinstance(match, dict) or not isinstance(match.get("metrics"), dict):
                    raise ConsistencyError("Intermediate evaluation fraction is missing")
                values.append(float(match["metrics"]["log_loss"]))
            intermediate_points.append(
                BudgetQualityPoint(
                    name,
                    fraction,
                    math.ceil(fraction * runs[0].plan.credits)
                    * runs[0].plan.steps_per_credit
                    * runs[0].plan.batch_size,
                    _mean(values),
                )
            )

    scheduler_rows: list[dict[str, object]] = []
    for scheduler_name in (
        "fixed_early",
        "fixed_midpoint",
        "fixed_deadline",
        "calibration_drift",
    ):
        runs = [
            item
            for item in online
            if item.plan.study == "study_b" and item.plan.scheduler == scheduler_name
        ]
        if len(runs) != 3:
            raise ConsistencyError("Final Study B aggregate is missing a seed")
        for item in runs:
            scheduler = item.manifest.get("scheduler_audit")
            monitoring = item.manifest.get("monitoring_audit")
            if not isinstance(scheduler, dict) or not isinstance(monitoring, dict):
                raise ConsistencyError("Scheduler audit evidence is missing")
            decisions = scheduler.get("decisions")
            windows = scheduler.get("windows")
            if (
                not isinstance(decisions, list)
                or not isinstance(windows, list)
                or not windows
                or not isinstance(windows[0], dict)
            ):
                raise ConsistencyError("Scheduler decision ledger is malformed")
            origin = int(windows[0]["start_time"]) - 31 * 86_400
            trigger_days = [
                int((int(value["decision_time"]) - origin) // 86_400)
                for value in decisions
                if isinstance(value, dict) and value.get("reason") == "calibration_trigger"
            ]
            scheduler_rows.append(
                {
                    "scheduler": scheduler_name,
                    "seed": item.plan.seed,
                    "credits": int(item.manifest["credits"]),
                    "optimizer_steps": int(item.manifest["core_optimizer_steps"]),
                    "optimizer_examples": int(item.manifest["core_optimizer_examples"]),
                    "monitoring_examples": int(monitoring["inference_examples"]),
                    "monitoring_exposure_overlap": 0,
                    "trigger_days": trigger_days,
                    "status": "complete",
                }
            )
        log_loss = _mean([float(item.evaluation["overall"]["log_loss"]) for item in runs])
        core = int(runs[0].manifest["core_optimizer_examples"])
        compute = [_compute_fields(item.manifest) for item in runs]
        total = core + compute[0][1]
        compute_values.append((scheduler_name, log_loss, core, total))
        compute_extra[scheduler_name] = (
            _mean([item[2] for item in compute]),
            max(item[3] for item in compute),
        )
        for fraction in (0.25, 0.5, 0.75, 1.0):
            values = []
            for item in runs:
                raw = item.evaluation["intermediate"]
                match = next(entry for entry in raw if entry["budget_fraction"] == fraction)
                values.append(float(match["metrics"]["log_loss"]))
            intermediate_points.append(
                BudgetQualityPoint(
                    scheduler_name,
                    fraction,
                    math.ceil(fraction * runs[0].plan.credits)
                    * runs[0].plan.steps_per_credit
                    * runs[0].plan.batch_size,
                    _mean(values),
                )
            )

    pareto = compute_pareto_table(tuple(compute_values))
    compute_rows = [
        {
            **item,
            "wall_seconds": compute_extra[str(item["method"])][0],
            "peak_memory_gb": compute_extra[str(item["method"])][1],
        }
        for item in pareto
    ]
    intermediate_rows = validate_intermediate_budget(tuple(intermediate_points))

    study_b_eval = [
        item for item in evaluations if item["method"] in {"fixed_deadline", "calibration_drift"}
    ]
    primary_scheduler_row = next(
        row
        for row in paired_rows
        if row["control"] == "fixed_deadline"
        and row["candidate"] == "calibration_drift"
        and row["metric"] == "log_loss"
        and row["block_days"] == 3
    )
    primary_seed_differences = primary_scheduler_row.get("seed_differences")
    if not isinstance(primary_seed_differences, list) or not all(
        isinstance(item, dict) for item in primary_seed_differences
    ):
        raise ConsistencyError("Primary scheduler seed differences are malformed")
    success_comparison: dict[str, Any] = {
        "control": "fixed_deadline",
        "candidate": "calibration_drift",
        "paired_intervals": comparisons[("fixed_deadline", "calibration_drift")],
        "seed_metrics": study_b_eval,
        "consistent_favorable_log_loss_sign": all(
            float(item["difference"]) < 0.0 for item in primary_seed_differences
        ),
    }
    fixed_budgets = {
        (row["credits"], row["optimizer_steps"], row["optimizer_examples"])
        for row in scheduler_rows
    }
    outcome = scheduler_success(success_comparison, identical_core_budget=len(fixed_budgets) == 1)
    rows = prepared_manifest.get("rows")
    source = prepared_manifest.get("source")
    git = lock.get("git")
    if not isinstance(rows, dict) or not isinstance(source, dict) or not isinstance(git, dict):
        raise ConsistencyError("Prepared or protocol identity is missing from aggregate inputs")
    statement = (
        "The locked calibration-drift scheduler met every predeclared support condition."
        if outcome["supported"] is True
        else (
            "The locked calibration-drift scheduler result was negative or inconclusive "
            "under the predeclared criterion."
        )
    )
    limitations = [
        "This is one public sponsored-search dataset and may not generalize to other "
        "traffic or attribution policies.",
        "Mature calibration evidence is delayed by the 30-day attribution window and "
        "is not real-time drift detection.",
        "Offline references use a different representation and optimizer and are "
        "excluded from the equal-compute ranking.",
        "Deterministic controls reduce software variance but do not prove bitwise "
        "portability across accelerator stacks.",
    ]
    if (
        isinstance(lock.get("selection_execution"), dict)
        and lock["selection_execution"].get("mode") == "verified_cross_commit_reuse"
    ):
        limitations.append(
            "Selection evidence came from an earlier clean commit; hashed "
            "provenance records that the intervening change fixed dataset-relative scheduler "
            "boundary validation after selection and before final scoring."
        )
    report = ReportInput.model_validate(
        {
            "version": 1,
            "title": "LateSignal final delayed-conversion benchmark",
            "result_kind": "final",
            "dataset": {
                "name": "Criteo Sponsored Search Conversion Log",
                "license_id": "CC-BY-NC-SA-4.0",
                "source_archive_sha256": source["archive_sha256"],
                "preparation_manifest_sha256": lock["data"]["manifest_sha256"],
                "accepted_rows": rows["inspection_accepted"],
                "quarantined_rows": rows["inspection_quarantined"],
            },
            "protocol": {
                "lock_sha256": lock["lock_sha256"],
                "code_commit": git["commit"],
                "environment_sha256": lock["environment_sha256"],
                "seeds": [17, 41, 73],
                "publication_eligible": lock["publication_eligible"],
            },
            "methods": method_rows,
            "schedulers": scheduler_rows,
            "evaluations": evaluations,
            "slices": slices,
            "paired_intervals": paired_rows,
            "intermediate_budget": intermediate_rows,
            "compute": compute_rows,
            "leakage_audit": quality_controls,
            "limitations": limitations,
            "reproduction_commands": [
                "uv sync --frozen --all-groups",
                'GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader '
                "| sed '/^[[:space:]]*$/d')\"",
                'test "$(printf \'%s\\n\' "$GPU_UUID" | wc -l | tr -d \' \')" = "1"',
                'test "${GPU_UUID#GPU-}" != "$GPU_UUID"',
                "export GPU_UUID",
                'export CUDA_VISIBLE_DEVICES="$GPU_UUID"',
                "uv run latesignal protocol validate configs/experiments/final.yaml "
                "--out runs/feasibility/measured.json --json",
                "uv run latesignal protocol bind-feasibility "
                "configs/experiments/final.yaml "
                "--measured runs/feasibility/measured.json "
                "--data-manifest data/processed/manifests/preparation.json "
                '--device-uuid "$GPU_UUID" --out runs/feasibility/final.json --json',
                "SELECTED_STEPS=\"$(uv run --frozen python -c 'import json; "
                'print(json.load(open("runs/feasibility/final.json"))'
                '["selected_steps_per_credit"])\')"',
                "uv run latesignal selection run configs/experiments/final.yaml "
                "--data-manifest data/processed/manifests/preparation.json "
                "--feature-config configs/features.yaml --cache-root data/runtime-features "
                '--out runs/selection --steps-per-credit "$SELECTED_STEPS" '
                '--device-uuid "$GPU_UUID" --json',
                "uv run latesignal protocol lock configs/experiments/final.yaml "
                "--selection runs/selection/selection-results.json "
                "--feasibility runs/feasibility/final.json "
                "--data-manifest data/processed/manifests/preparation.json "
                "--out runs/protocol-lock.json --json",
                "uv run latesignal final qualify configs/experiments/final.yaml "
                "--protocol-lock runs/protocol-lock.json "
                "--data-manifest data/processed/manifests/preparation.json "
                "--feature-config configs/features.yaml --cache-root data/runtime-features "
                '--out runs/quality-gate.json --device-uuid "$GPU_UUID" --json',
                "uv run latesignal final run configs/experiments/final.yaml "
                "--protocol-lock runs/protocol-lock.json "
                "--data-manifest data/processed/manifests/preparation.json "
                "--feature-config configs/features.yaml --cache-root data/runtime-features "
                '--out runs/final --device-uuid "$GPU_UUID" --json',
                "uv run latesignal final aggregate configs/experiments/final.yaml "
                "--protocol-lock runs/protocol-lock.json "
                "--data-manifest data/processed/manifests/preparation.json "
                "--feature-config configs/features.yaml --cache-root data/runtime-features "
                "--out runs/final --quality-gate runs/quality-gate.json "
                '--device-uuid "$GPU_UUID" --json',
            ],
            "claim": {
                "scheduler_outcome": outcome["outcome"],
                "published_number_reproduction": False,
                "statement": statement,
            },
        }
    )
    return report, outcome


def aggregate_production_final(
    inputs: FinalPlanInputs,
    output_root: Path,
    *,
    features: RuntimeFeatureStore,
    truth: ProductionTruthStore,
    prepared_manifest_path: Path,
    quality_gate_path: Path,
    device: Literal["cpu", "cuda"],
    batch_replicates: int | None = None,
) -> dict[str, Any]:
    """Verify row-level final evidence, compute paired uncertainty, and render aggregates."""

    if output_root.is_symlink():
        raise ConsistencyError("Final aggregate output root cannot be a symlink")
    root = output_root.resolve()
    aggregate_root = root / "aggregate"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    plans = final_online_plans(inputs)
    offline_plans = offline_reference_plans(inputs)
    lock = inputs.protocol_lock
    git = lock.get("git")
    if not isinstance(git, dict):
        raise ConsistencyError("Protocol lock Git identity is missing")
    quality_controls = _quality_gate(
        quality_gate_path,
        lock_sha256=str(lock["lock_sha256"]),
        git_commit=str(git["commit"]),
    )
    _verify_final_coordinators(
        root,
        plans,
        {plan.run_id for plan in offline_plans},
        protocol_lock_sha256=str(lock["lock_sha256"]),
    )
    prepared_manifest = read_json(prepared_manifest_path)
    references, labels, days, ordered_ids, truth_sha256 = _truth_identity(features, truth)
    online = [
        _load_online_evidence(
            root,
            plan,
            expected_rows=references.size,
            ordered_id_sha256=ordered_ids,
            truth_cohort_sha256=truth_sha256,
        )
        for plan in plans
    ]
    offline_evaluations: list[dict[str, Any]] = []
    for plan in offline_plans:
        run_root = root / "offline" / "runs" / plan.run_id
        manifest = read_json(run_root / "manifest.json")
        evaluation = read_json(run_root / "evaluation.json")
        _verify_hashed(manifest, "manifest_sha256", "Offline prediction manifest")
        _verify_hashed(evaluation, "evaluation_sha256", "Offline evaluation")
        probability_path = run_root / "probabilities.npy"
        probability_sha256, probability_bytes = sha256_file(probability_path)
        probabilities = np.load(probability_path, allow_pickle=False)
        if (
            manifest.get("config_sha256") != plan.canonical_sha256
            or evaluation.get("config_sha256") != plan.canonical_sha256
            or evaluation.get("prediction_manifest_sha256") != manifest.get("manifest_sha256")
            or evaluation.get("truth_joined") is not True
            or evaluation.get("ranking_eligible") is not False
            or manifest.get("ordered_evaluation_id_sha256") != ordered_ids
            or manifest.get("probabilities_sha256") != probability_sha256
            or manifest.get("probabilities_bytes") != probability_bytes
            or probabilities.dtype != np.float32
            or probabilities.shape != (references.size,)
        ):
            raise ConsistencyError("Offline aggregate evidence does not align")
        offline_evaluations.append(evaluation)
    replicates = inputs.protocol.final_training.bootstrap_replicates
    bootstrap_seed = 20260813
    bootstrap: dict[tuple[str, int, int], CompactBootstrapMetrics] = {}
    for item in online:
        if not item.plan.ranking_eligible:
            continue
        key = (
            f"study_a:{item.plan.method}"
            if item.plan.study == "study_a"
            else f"study_b:{item.plan.scheduler}"
        )
        for raw_block in _BLOCKS:
            block = cast(Literal[1, 3, 7], raw_block)
            result = _bootstrap_cache(
                aggregate_root,
                item,
                labels=labels,
                days=days,
                truth_cohort_sha256=truth_sha256,
                block_days=block,
                replicates=replicates,
                bootstrap_seed=bootstrap_seed,
                device=device,
                batch_replicates=batch_replicates,
            )
            overall = item.evaluation.get("overall")
            if not isinstance(overall, dict) or any(
                not np.isclose(result.point[name], float(overall[name]), rtol=0.0, atol=1e-7)
                for name in _METRICS
            ):
                raise ConsistencyError("Bootstrap point metrics differ from sealed evaluation")
            bootstrap[(key, item.plan.seed, block)] = result
    report, outcome = _aggregate_report(
        inputs=inputs,
        online=online,
        offline_evaluations=offline_evaluations,
        bootstrap=bootstrap,
        prepared_manifest=prepared_manifest,
        quality_controls=quality_controls,
    )
    report_input_path = aggregate_root / "report-input.json"
    report_payload = report.model_dump(mode="json")
    if report_input_path.exists():
        if read_json(report_input_path) != report_payload:
            raise ConsistencyError("Immutable final report input changed")
    else:
        write_json_atomic(report_input_path, report_payload)
    report_root = aggregate_root / "report"
    if not (report_root / "manifest.json").exists():
        render_report(report, report_root, report_format="html", input_path=report_input_path)
    report_manifest = read_json(report_root / "manifest.json")
    _verify_hashed(report_manifest, "manifest_sha256", "Final static report manifest")
    payload: dict[str, object] = {
        "version": 1,
        "status": "complete",
        "aggregate_only": True,
        "protocol_lock_sha256": lock["lock_sha256"],
        "online_runs": 33,
        "offline_runs": 6,
        "bootstrap_replicates": replicates,
        "bootstrap_blocks": list(_BLOCKS),
        "bootstrap_run_blocks": len(bootstrap),
        "report_input_sha256": hashlib.sha256(canonical_json_bytes(report_payload)).hexdigest(),
        "report_manifest_sha256": report_manifest["manifest_sha256"],
        "scheduler_outcome": outcome["outcome"],
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    manifest_path = aggregate_root / "manifest.json"
    if manifest_path.exists():
        if read_json(manifest_path) != payload:
            raise ConsistencyError("Immutable final aggregate manifest changed")
    else:
        write_json_atomic(manifest_path, payload)
    return read_json(manifest_path)
