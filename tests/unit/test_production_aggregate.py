from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from latesignal.data.manifests import (
    canonical_json_bytes,
    sha256_file,
    write_json_atomic,
)
from latesignal.errors import ConsistencyError
from latesignal.evaluation.bootstrap import (
    CompactBootstrapMetrics,
    contiguous_day_bootstrap_weights,
)
from latesignal.evaluation.metrics import classification_metrics
from latesignal.experiments.production_aggregate import (
    _LEAKAGE_CONTROLS,
    _aggregate_report,
    _bootstrap_cache,
    _OnlineEvidence,
    aggregate_production_final,
)
from latesignal.experiments.production_final import FinalPlanInputs, ProductionFinalPlan
from latesignal.experiments.production_offline import ProductionOfflinePlan
from latesignal.features.store import RuntimeFeatureStore
from latesignal.simulator.production_oracle import ProductionTruthStore

SEEDS = (17, 41, 73)
METHODS = (
    "complete_wait",
    "immediate_fake_negative",
    "fixed_wait",
    "dfm",
    "fnw",
    "es_dfm",
    "oracle_reference",
)
SCHEDULERS = ("fixed_early", "fixed_midpoint", "fixed_deadline", "calibration_drift")


def _plan(index: int, *, study: str, name: str, seed: int) -> ProductionFinalPlan:
    method = name if study == "study_a" else "fixed_wait"
    scheduler = "fixed_daily" if study == "study_a" else name
    return ProductionFinalPlan.model_validate(
        {
            "version": 1,
            "phase": "qualification",
            "study": study,
            "run_id": ("study-a-" if study == "study_a" else "study-b-") + f"{index:016x}",
            "method": method,
            "scheduler": scheduler,
            "seed": seed,
            "wait_days": 3 if method in {"fixed_wait", "es_dfm"} else None,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "dropout": 0.0,
            "gradient_norm_clip": 5.0,
            "initialization_steps": 1,
            "steps_per_credit": 1,
            "credits": 59 if study == "study_a" else 12,
            "batch_size": 4,
            "recent_window_days": 3,
            "reservoir_capacity": 1_000_000,
            "feature_policy": "compact",
            "prediction_batch_size": 8,
            "first_decision_day": 31,
            "last_decision_day": 89,
            "evaluation_first_click_day": 65,
            "evaluation_last_click_day": 89,
            "intermediate_budget_fractions": (0.25, 0.5, 0.75, 1.0),
            "deployable": method != "oracle_reference",
            "ranking_eligible": method != "oracle_reference",
            "device": "cpu",
            "protocol_sha256": "1" * 64,
            "protocol_lock_sha256": "2" * 64,
            "selection_decisions_sha256": "3" * 64,
            "data_manifest_sha256": "4" * 64,
            "feature_policy_sha256": "5" * 64,
        }
    )


def _metric(probability: float) -> dict[str, object]:
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), 50)
    probabilities = np.where(labels == 1, probability, 1.0 - probability)
    return classification_metrics(labels, probabilities)


def _slice_rows(metrics: dict[str, object]) -> list[dict[str, object]]:
    dimensions = (
        "cold_user",
        "cold_product",
        "user_frequency",
        "product_frequency",
        "product_price_bin",
        "device_type",
        "positive_conversion_delay",
        "click_day_block",
    )
    return [
        {
            "dimension": dimension,
            "value": "fixture",
            "count": 100,
            "positives": 50,
            "ranking_eligible": True,
            "suppression_reason": None,
            "metrics": metrics,
        }
        for dimension in dimensions
    ]


def _evidence() -> list[_OnlineEvidence]:
    result: list[_OnlineEvidence] = []
    index = 0
    quality = {
        "complete_wait": 0.70,
        "immediate_fake_negative": 0.72,
        "fixed_wait": 0.74,
        "dfm": 0.73,
        "fnw": 0.75,
        "es_dfm": 0.76,
        "oracle_reference": 0.85,
        "fixed_early": 0.74,
        "fixed_midpoint": 0.75,
        "fixed_deadline": 0.70,
        "calibration_drift": 0.80,
    }
    for study, names in (("study_a", METHODS), ("study_b", SCHEDULERS)):
        for name in names:
            for seed in SEEDS:
                plan = _plan(index, study=study, name=name, seed=seed)
                index += 1
                metrics = _metric(quality[name])
                compute: dict[str, Any] = {
                    "initialization_seconds": 1.0,
                    "core_training_seconds": 2.0,
                    "auxiliary_training_seconds": 1.0 if name == "es_dfm" else 0.0,
                    "primary_prediction_seconds": 1.0,
                    "intermediate_prediction_seconds": 1.0,
                    "monitoring_prediction_seconds": 1.0,
                    "checkpoint_seconds": 1.0,
                    "snapshot_seconds": 1.0,
                    "peak_host_memory_bytes": 1024**3,
                    "peak_accelerator_memory_bytes": 2 * 1024**3,
                    "auxiliary": ([{"steps": 200, "examples": 800}] if name == "es_dfm" else []),
                    "credits": [{"decision_time": 31 * 86_400}],
                }
                manifest: dict[str, Any] = {
                    "credits": plan.credits,
                    "core_optimizer_steps": plan.credits * plan.steps_per_credit,
                    "core_optimizer_examples": (
                        plan.credits * plan.steps_per_credit * plan.batch_size
                    ),
                    "compute": compute,
                    "scheduler_audit": {
                        "windows": [{"start_time": 31 * 86_400}],
                        "decisions": (
                            [
                                {
                                    "decision_time": 33 * 86_400,
                                    "reason": "calibration_trigger",
                                }
                            ]
                            if name == "calibration_drift"
                            else []
                        ),
                    },
                    "monitoring_audit": {"inference_examples": 1_000},
                }
                evaluation: dict[str, Any] = {
                    "overall": metrics,
                    "slices": _slice_rows(metrics),
                    "intermediate": [
                        {
                            "budget_fraction": fraction,
                            "metrics": _metric(max(0.55, quality[name] - 0.1 + fraction / 10)),
                        }
                        for fraction in (0.25, 0.5, 0.75, 1.0)
                    ],
                }
                result.append(
                    _OnlineEvidence(
                        plan=plan,
                        manifest=manifest,
                        evaluation=evaluation,
                        probability_sha256=f"{index:064x}",
                        probabilities=np.full(100, 0.5, dtype=np.float32),
                    )
                )
    return result


def _bootstrap(
    online: list[_OnlineEvidence],
) -> dict[tuple[str, int, int], CompactBootstrapMetrics]:
    days = np.arange(65, 90, dtype=np.int16)
    result: dict[tuple[str, int, int], CompactBootstrapMetrics] = {}
    losses = {
        "complete_wait": 0.30,
        "immediate_fake_negative": 0.32,
        "fixed_wait": 0.28,
        "dfm": 0.29,
        "fnw": 0.27,
        "es_dfm": 0.26,
        "fixed_early": 0.29,
        "fixed_midpoint": 0.28,
        "fixed_deadline": 0.30,
        "calibration_drift": 0.20,
    }
    for item in online:
        if not item.plan.ranking_eligible:
            continue
        name = item.display_name
        key = f"{item.plan.study}:{name}"
        for block in (1, 3, 7):
            weights = contiguous_day_bootstrap_weights(days, block_days=block, replicates=2_000)
            point = {
                "log_loss": losses[name],
                "brier_score": losses[name] / 2,
                "pr_auc": 1.0 - losses[name],
                "roc_auc": 1.0 - losses[name] / 2,
            }
            result[(key, item.plan.seed, block)] = CompactBootstrapMetrics(
                block_days=block,
                bootstrap_seed=20260813,
                day_weights=weights,
                point=point,
                replicates={
                    metric: np.full(2_000, value, dtype=np.float64)
                    for metric, value in point.items()
                },
            )
    return result


def test_bootstrap_cache_round_trips_and_rejects_corruption(tmp_path: Path) -> None:
    evidence = _evidence()[0]
    days = np.repeat(np.arange(65, 90, dtype=np.int16), 4)
    labels = np.tile(np.asarray([0, 1, 0, 1], dtype=np.int8), 25)
    probabilities = np.where(labels == 1, 0.7, 0.3).astype(np.float32)
    evidence = _OnlineEvidence(
        evidence.plan,
        evidence.manifest,
        evidence.evaluation,
        "a" * 64,
        probabilities,
    )

    first = _bootstrap_cache(
        tmp_path,
        evidence,
        labels=labels,
        days=days,
        truth_cohort_sha256="b" * 64,
        block_days=3,
        replicates=2_000,
        bootstrap_seed=20260813,
        device="cpu",
        batch_replicates=31,
    )
    repeated = _bootstrap_cache(
        tmp_path,
        evidence,
        labels=labels,
        days=days,
        truth_cohort_sha256="b" * 64,
        block_days=3,
        replicates=2_000,
        bootstrap_seed=20260813,
        device="cpu",
        batch_replicates=31,
    )
    np.testing.assert_array_equal(first.replicates["pr_auc"], repeated.replicates["pr_auc"])

    array_path = tmp_path / "bootstrap" / evidence.plan.run_id / "block-3" / "replicates.npz"
    array_path.write_bytes(b"forged")
    with pytest.raises(ConsistencyError):
        _bootstrap_cache(
            tmp_path,
            evidence,
            labels=labels,
            days=days,
            truth_cohort_sha256="b" * 64,
            block_days=3,
            replicates=2_000,
            bootstrap_seed=20260813,
            device="cpu",
            batch_replicates=31,
        )


def test_final_aggregate_builds_strict_truthful_report_input() -> None:
    online = _evidence()
    offline = [
        {
            "name": name,
            "seed": seed,
            "overall": _metric(0.65),
        }
        for name in ("mature_logistic_regression", "mature_lightgbm")
        for seed in SEEDS
    ]
    lock = {
        "lock_sha256": "2" * 64,
        "environment_sha256": "6" * 64,
        "publication_eligible": True,
        "git": {"commit": "7" * 40},
        "data": {"manifest_sha256": "4" * 64},
    }
    inputs = cast(
        FinalPlanInputs,
        SimpleNamespace(protocol_lock=lock),
    )
    controls = [
        {"control": name, "status": "passed", "evidence": "locked mutation test"}
        for name in sorted(_LEAKAGE_CONTROLS)
    ]

    report, outcome = _aggregate_report(
        inputs=inputs,
        online=online,
        offline_evaluations=offline,
        bootstrap=_bootstrap(online),
        prepared_manifest={
            "source": {"archive_sha256": "8" * 64},
            "rows": {"inspection_accepted": 100, "inspection_quarantined": 2},
        },
        quality_controls=controls,
    )

    assert report.result_kind == "final"
    assert {item.method for item in report.methods} == set(METHODS)
    assert {item.scheduler for item in report.schedulers} == set(SCHEDULERS)
    assert len(report.paired_intervals) == 96
    assert report.claim.scheduler_outcome == "supported"
    commands = report.reproduction_commands
    assert any(
        "protocol validate" in command and "runs/feasibility/measured.json" in command
        for command in commands
    )
    assert any("bind-feasibility" in command for command in report.reproduction_commands)
    assert 'export CUDA_VISIBLE_DEVICES="$GPU_UUID"' in report.reproduction_commands
    assert any("final qualify" in command for command in report.reproduction_commands)
    assert all("ssh" not in command.lower() for command in report.reproduction_commands)
    subprocess.run(
        ["bash", "-n"],
        input="\n".join(commands),
        text=True,
        check=True,
        capture_output=True,
    )
    assert outcome["supported"] is True


def _hashed(value: dict[str, object], digest_name: str) -> dict[str, object]:
    result = dict(value)
    result[digest_name] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return result


class _AggregateFeatures:
    def __init__(self) -> None:
        self.click_days = np.repeat(np.arange(65, 90, dtype=np.int16), 4)
        self.click_ids = np.asarray(
            [hashlib.sha256(f"aggregate-{index}".encode()).digest() for index in range(100)],
            dtype="V32",
        )

    def references_for_day(self, day: int) -> np.ndarray:
        return np.flatnonzero(self.click_days == day).astype(np.int32)


def _aggregate_truth(features: _AggregateFeatures) -> ProductionTruthStore:
    labels = np.tile(np.asarray([0, 1, 0, 1], dtype=np.int8), 25)
    return ProductionTruthStore(
        prepared_manifest_sha256="4" * 64,
        final_labels=labels,
        available_at=np.arange(100, dtype=np.float64),
        conversion_delay_days=np.where(labels == 1, 1.0, np.nan).astype(np.float32),
        event_feature_refs=np.arange(100, dtype=np.int32),
        event_available_at=np.arange(100, dtype=np.float64),
        event_labels=labels,
    )


def _offline_plans() -> tuple[ProductionOfflinePlan, ...]:
    plans: list[ProductionOfflinePlan] = []
    for name in ("mature_logistic_regression", "mature_lightgbm"):
        for seed in SEEDS:
            plans.append(
                ProductionOfflinePlan.model_validate(
                    {
                        "version": 1,
                        "phase": "final",
                        "run_id": f"offline-{len(plans):016x}",
                        "name": name,
                        "seed": seed,
                        "training_first_click_day": 0,
                        "training_last_click_day": 34,
                        "training_cutoff_day": 65,
                        "evaluation_first_click_day": 65,
                        "evaluation_last_click_day": 89,
                        "monitoring_excluded": True,
                        "ranking_eligible": False,
                        "protocol_sha256": "1" * 64,
                        "protocol_lock_sha256": "2" * 64,
                        "data_manifest_sha256": "4" * 64,
                        "feature_policy_sha256": "5" * 64,
                    }
                )
            )
    return tuple(plans)


def test_complete_aggregate_path_verifies_evidence_and_renders_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    online = _evidence()
    plans = tuple(item.plan for item in online)
    offline_plans = _offline_plans()
    features = _AggregateFeatures()
    truth = _aggregate_truth(features)
    ordered_ids = hashlib.sha256(features.click_ids.tobytes()).hexdigest()
    truth_digest = hashlib.sha256()
    truth_digest.update(features.click_ids.tobytes())
    truth_digest.update(truth.final_labels.tobytes())
    truth_sha256 = truth_digest.hexdigest()
    for item in online:
        root = (
            tmp_path
            / ("study-a" if item.plan.study == "study_a" else "study-b")
            / "runs"
            / item.plan.run_id
        )
        root.mkdir(parents=True)
        manifest = _hashed(
            {
                **item.manifest,
                "status": "complete",
                "truth_joined": False,
                "primary_evaluation_mode": "prequential",
                "evaluation_click_days": [65, 89],
                "config_sha256": item.plan.canonical_sha256,
                "protocol_lock_sha256": item.plan.protocol_lock_sha256,
            },
            "manifest_sha256",
        )
        evaluation = _hashed(
            {
                **item.evaluation,
                "status": "complete",
                "truth_joined": True,
                "config_sha256": item.plan.canonical_sha256,
            },
            "evaluation_sha256",
        )
        compact_root = root / "compact"
        compact_root.mkdir()
        probability_path = compact_root / "primary-probabilities.npy"
        with probability_path.open("wb") as output:
            np.save(output, item.probabilities, allow_pickle=False)
        probability_sha256, probability_bytes = sha256_file(probability_path)
        compact = _hashed(
            {
                "status": "verified_compact_primary",
                "config_sha256": item.plan.canonical_sha256,
                "rows": 100,
                "ordered_id_sha256": ordered_ids,
                "truth_cohort_sha256": truth_sha256,
                "probabilities_sha256": probability_sha256,
                "probabilities_bytes": probability_bytes,
            },
            "manifest_sha256",
        )
        retention = _hashed(
            {
                "status": "verified_and_pruned",
                "config_sha256": item.plan.canonical_sha256,
                "manifest_sha256": manifest["manifest_sha256"],
                "evaluation_sha256": evaluation["evaluation_sha256"],
                "compact_primary_manifest_sha256": compact["manifest_sha256"],
            },
            "retention_sha256",
        )
        write_json_atomic(root / "manifest.json", manifest)
        write_json_atomic(root / "evaluation.json", evaluation)
        write_json_atomic(compact_root / "manifest.json", compact)
        write_json_atomic(root / "retention.json", retention)
    offline_completed = []
    for plan in offline_plans:
        root = tmp_path / "offline" / "runs" / plan.run_id
        root.mkdir(parents=True)
        probabilities = np.full(100, 0.5, dtype=np.float32)
        with (root / "probabilities.npy").open("wb") as output:
            np.save(output, probabilities, allow_pickle=False)
        probability_sha256, probability_bytes = sha256_file(root / "probabilities.npy")
        manifest = _hashed(
            {
                "config_sha256": plan.canonical_sha256,
                "ordered_evaluation_id_sha256": ordered_ids,
                "probabilities_sha256": probability_sha256,
                "probabilities_bytes": probability_bytes,
            },
            "manifest_sha256",
        )
        evaluation = _hashed(
            {
                "name": plan.name,
                "seed": plan.seed,
                "config_sha256": plan.canonical_sha256,
                "prediction_manifest_sha256": manifest["manifest_sha256"],
                "truth_joined": True,
                "ranking_eligible": False,
                "overall": _metric(0.65),
            },
            "evaluation_sha256",
        )
        write_json_atomic(root / "manifest.json", manifest)
        write_json_atomic(root / "evaluation.json", evaluation)
        offline_completed.append({"run_id": plan.run_id})
    online_manifest = _hashed(
        {
            "status": "complete",
            "completed_count": 33,
            "completed_runs": [{"run_id": plan.run_id} for plan in plans],
        },
        "manifest_sha256",
    )
    offline_manifest = _hashed(
        {
            "status": "complete",
            "completed_count": 6,
            "completed_runs": offline_completed,
        },
        "manifest_sha256",
    )
    combined = _hashed(
        {
            "status": "complete",
            "completed_count": 39,
            "protocol_lock_sha256": "2" * 64,
            "online_manifest_sha256": online_manifest["manifest_sha256"],
            "offline_manifest_sha256": offline_manifest["manifest_sha256"],
        },
        "manifest_sha256",
    )
    write_json_atomic(tmp_path / "online-manifest.json", online_manifest)
    write_json_atomic(tmp_path / "offline-manifest.json", offline_manifest)
    write_json_atomic(tmp_path / "final-manifest.json", combined)
    prepared = tmp_path / "prepared.json"
    write_json_atomic(
        prepared,
        {
            "source": {"archive_sha256": "8" * 64},
            "rows": {"inspection_accepted": 100, "inspection_quarantined": 2},
        },
    )
    controls = [
        {"control": name, "status": "passed", "evidence": "locked mutation test"}
        for name in sorted(_LEAKAGE_CONTROLS)
    ]
    quality = _hashed(
        {
            "status": "passed",
            "protocol_lock_sha256": "2" * 64,
            "git_commit": "7" * 40,
            "leakage_controls": controls,
        },
        "manifest_sha256",
    )
    quality_path = tmp_path / "quality.json"
    write_json_atomic(quality_path, quality)
    lock = {
        "lock_sha256": "2" * 64,
        "environment_sha256": "6" * 64,
        "publication_eligible": True,
        "git": {"commit": "7" * 40},
        "data": {"manifest_sha256": "4" * 64},
    }
    inputs = cast(
        FinalPlanInputs,
        SimpleNamespace(
            protocol_lock=lock,
            protocol=SimpleNamespace(final_training=SimpleNamespace(bootstrap_replicates=2_000)),
        ),
    )
    monkeypatch.setattr(
        "latesignal.experiments.production_aggregate.final_online_plans", lambda _: plans
    )
    monkeypatch.setattr(
        "latesignal.experiments.production_aggregate.offline_reference_plans",
        lambda _: offline_plans,
    )

    def fake_bootstrap(*args, **kwargs) -> CompactBootstrapMetrics:
        evidence = args[1]
        block = kwargs["block_days"]
        point = {
            name: float(evidence.evaluation["overall"][name])
            for name in ("log_loss", "brier_score", "pr_auc", "roc_auc")
        }
        weights = contiguous_day_bootstrap_weights(
            features.click_days, block_days=block, replicates=2_000
        )
        return CompactBootstrapMetrics(
            block_days=block,
            bootstrap_seed=20260813,
            day_weights=weights,
            point=point,
            replicates={
                name: np.full(2_000, value, dtype=np.float64) for name, value in point.items()
            },
        )

    monkeypatch.setattr(
        "latesignal.experiments.production_aggregate._bootstrap_cache", fake_bootstrap
    )

    result = aggregate_production_final(
        inputs,
        tmp_path,
        features=cast(RuntimeFeatureStore, features),
        truth=truth,
        prepared_manifest_path=prepared,
        quality_gate_path=quality_path,
        device="cpu",
    )

    assert result["status"] == "complete"
    assert result["bootstrap_run_blocks"] == 90
    assert (tmp_path / "aggregate" / "report" / "report.html").is_file()
