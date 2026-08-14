"""Bounded benchmark and exact authored-matrix feasibility projection."""

from __future__ import annotations

import hashlib
import io
import math
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch

from latesignal.contracts.protocol import FinalExperimentConfig, ProtocolDefinition
from latesignal.data.schema import CATEGORICAL_CLICK_FIELDS
from latesignal.features.hashing import categorical_bucket
from latesignal.methods.losses import dfm_loss
from latesignal.models.conversion_mlp import CategoricalSpec, ConversionMLP
from latesignal.models.dfm import DelayedFeedbackMLP

BYTES_PER_GB = 1024**3
CHECKPOINT_WORKING_COPIES = 3
COMPLETED_CHECKPOINTS_RETAINED = 0
REPORT_BYTES_PER_RUN = 250_000
TRAINING_WARMUP_STEPS = 3
PROJECTION_UPPER_MULTIPLIER = 1.5
INTERMEDIATE_BUDGET_FRACTIONS = 4
HIGH_CARDINALITY_FIELDS = frozenset(
    {
        "audience_id",
        "product_brand",
        "product_id",
        "product_title",
        "partner_id",
        "user_id",
    }
)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _prediction_artifact_bytes_per_row(rows: int) -> float:
    click_ids = [hashlib.sha256(index.to_bytes(8, "little")).hexdigest() for index in range(rows)]
    table = pa.table(
        {
            "click_id": click_ids,
            "click_day": np.arange(rows, dtype=np.int16) % 10 + 25,
            "probability": np.linspace(0.001, 0.999, rows, dtype=np.float32),
            "model_version": np.full(rows, 10, dtype=np.int32),
        }
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd", use_dictionary=False)
    return float(sink.getvalue().size) / rows


def _exposure_artifact_bytes_per_row(rows: int) -> float:
    table = pa.table(
        {
            "credit_id": np.zeros(rows, dtype=np.uint16),
            "step": np.arange(rows, dtype=np.uint16) % 500,
            "record_key": np.arange(rows, dtype=np.uint64) * np.uint64(2654435761),
            "source": np.arange(rows, dtype=np.uint8) % 2,
            "weight": np.ones(rows, dtype=np.float32),
        }
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd", use_dictionary=False)
    return float(sink.getvalue().size) / rows


def _benchmark(final: FinalExperimentConfig) -> dict[str, Any]:
    requested = final.target_device
    available = requested == "cpu" or torch.cuda.is_available()
    device = torch.device(requested if available else "cpu")
    fields = tuple(sorted(CATEGORICAL_CLICK_FIELDS))
    if available:
        specs = {
            field: CategoricalSpec(
                bucket_count=2**20 if field in HIGH_CARDINALITY_FIELDS else 2**14,
                embedding_dim=16 if field in HIGH_CARDINALITY_FIELDS else 8,
            )
            for field in fields
        }
        benchmark_scale = "locked-large-candidate"
    else:
        specs = {field: CategoricalSpec(bucket_count=8, embedding_dim=2) for field in fields}
        benchmark_scale = "diagnostic-proxy-only"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = ConversionMLP(specs, dropout=0.0).to(device)
    batch_size = final.pilot.benchmark_batch_size
    categorical = {
        field: torch.arange(batch_size, device=device, dtype=torch.long) % 8 for field in fields
    }
    numeric = torch.zeros((batch_size, 4), device=device)
    targets = torch.arange(batch_size, device=device, dtype=torch.float32) % 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    model.train()

    def train_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        logits = model(categorical, numeric)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    for _ in range(TRAINING_WARMUP_STEPS):
        train_step()
    _synchronize(device)
    training_start = time.perf_counter()
    for _ in range(final.pilot.benchmark_steps):
        train_step()
    _synchronize(device)
    training_seconds = time.perf_counter() - training_start

    q_tn = ConversionMLP(specs, dropout=0.0).to(device).eval()
    q_dp = ConversionMLP(specs, dropout=0.0).to(device).eval()

    def es_train_step() -> None:
        with torch.no_grad():
            q_tn_probability = torch.sigmoid(q_tn(categorical, numeric))
            q_dp_probability = torch.sigmoid(q_dp(categorical, numeric))
            weights = torch.clamp(
                (1.0 + q_dp_probability)
                * torch.where(targets == 1, torch.ones_like(targets), q_tn_probability),
                min=1e-4,
                max=2.0,
            )
        optimizer.zero_grad(set_to_none=True)
        logits = model(categorical, numeric)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        (losses * weights).mean().backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    for _ in range(TRAINING_WARMUP_STEPS):
        es_train_step()
    _synchronize(device)
    es_training_start = time.perf_counter()
    for _ in range(final.pilot.benchmark_steps):
        es_train_step()
    _synchronize(device)
    es_training_seconds = time.perf_counter() - es_training_start

    dfm_model = DelayedFeedbackMLP(ConversionMLP(specs, dropout=0.0)).to(device)
    dfm_optimizer = torch.optim.AdamW(dfm_model.parameters(), lr=3e-4, weight_decay=1e-4)
    time_days = torch.linspace(0.0, 30.0, batch_size, device=device)

    def dfm_train_step() -> None:
        dfm_optimizer.zero_grad(set_to_none=True)
        conversion_logits, rate_logits = dfm_model(categorical, numeric)
        loss = dfm_loss(conversion_logits, rate_logits, targets, time_days)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(dfm_model.parameters(), 5.0)
        dfm_optimizer.step()

    for _ in range(TRAINING_WARMUP_STEPS):
        dfm_train_step()
    _synchronize(device)
    dfm_training_start = time.perf_counter()
    for _ in range(final.pilot.benchmark_steps):
        dfm_train_step()
    _synchronize(device)
    dfm_training_seconds = time.perf_counter() - dfm_training_start
    model.eval()
    prediction_batches = max(1, math.ceil(final.pilot.benchmark_examples / batch_size))
    _synchronize(device)
    prediction_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(prediction_batches):
            model(categorical, numeric)
    _synchronize(device)
    prediction_seconds = time.perf_counter() - prediction_start
    preparation_start = time.perf_counter()
    for index in range(final.pilot.benchmark_examples):
        categorical_bucket("user_id", f"synthetic-{index}", 20260813, 2**18)
    preparation_seconds = time.perf_counter() - preparation_start
    model_state = io.BytesIO()
    torch.save(model.state_dict(), model_state)
    checkpoint = io.BytesIO()
    checkpoint_start = time.perf_counter()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
    checkpoint_seconds = time.perf_counter() - checkpoint_start
    peak_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak_raw if sys.platform == "darwin" else peak_raw * 1024
    return {
        "requested_device": requested,
        "measured_device": str(device),
        "requested_device_available": available,
        "benchmark_scale": benchmark_scale,
        "model_parameters": model.parameter_count,
        "training_batch_size": batch_size,
        "training_warmup_steps": TRAINING_WARMUP_STEPS,
        "training_examples": final.pilot.benchmark_steps * batch_size,
        "training_seconds": training_seconds,
        "training_step_seconds": training_seconds / final.pilot.benchmark_steps,
        "training_examples_per_second": final.pilot.benchmark_steps * batch_size / training_seconds,
        "es_main_training_seconds": es_training_seconds,
        "es_main_training_step_seconds": es_training_seconds / final.pilot.benchmark_steps,
        "dfm_training_seconds": dfm_training_seconds,
        "dfm_training_step_seconds": dfm_training_seconds / final.pilot.benchmark_steps,
        "prediction_examples": prediction_batches * batch_size,
        "prediction_seconds": prediction_seconds,
        "prediction_examples_per_second": prediction_batches * batch_size / prediction_seconds,
        "synthetic_preparation_rows": final.pilot.benchmark_examples,
        "synthetic_preparation_seconds": preparation_seconds,
        "synthetic_preparation_rows_per_second": final.pilot.benchmark_examples
        / preparation_seconds,
        "checkpoint_bytes": len(checkpoint.getvalue()),
        "model_state_bytes": len(model_state.getvalue()),
        "checkpoint_write_seconds": checkpoint_seconds,
        "prediction_artifact_bytes_per_row": _prediction_artifact_bytes_per_row(
            final.pilot.benchmark_examples
        ),
        "exposure_artifact_bytes_per_row": _exposure_artifact_bytes_per_row(
            final.pilot.benchmark_examples
        ),
        "peak_host_memory_gb": peak_bytes / BYTES_PER_GB,
        "peak_device_memory_gb": (
            torch.cuda.max_memory_allocated(device) / BYTES_PER_GB if device.type == "cuda" else 0.0
        ),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }


def _period_rows(feature_root: Path, first_day: int, last_day: int) -> int:
    total = 0
    for day in range(first_day, last_day + 1):
        files = sorted((feature_root / f"click_day={day:03d}").glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"Missing prepared click day {day}")
        total += sum(pq.read_metadata(path).num_rows for path in files)
    return total


def _real_pilot(final: FinalExperimentConfig, config_path: Path) -> dict[str, Any]:
    prepared = (config_path.parent / final.pilot.prepared_root).resolve()
    feature_root = prepared / "features"
    partitions = sorted(feature_root.glob("click_day=*"))[: final.pilot.max_click_days]
    files = [file for partition in partitions for file in sorted(partition.glob("*.parquet"))]
    if not files:
        return {
            "status": "unavailable",
            "prepared_root": str(prepared),
            "reason": "No prepared click-day partitions are available",
            "max_click_days": final.pilot.max_click_days,
        }
    start = time.perf_counter()
    rows = int(
        pl.scan_parquet([str(path) for path in files])
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    seconds = time.perf_counter() - start
    try:
        total_rows = _period_rows(feature_root, 0, 89)
        selection_rows = _period_rows(feature_root, 25, 34)
        final_rows = _period_rows(feature_root, 65, 89)
    except (OSError, pa.ArrowException) as error:
        return {
            "status": "unavailable",
            "prepared_root": str(prepared),
            "reason": f"Real workload inventory failed: {error}",
            "max_click_days": final.pilot.max_click_days,
        }
    return {
        "status": "measured",
        "prepared_root": str(prepared),
        "click_days": len(partitions),
        "files": len(files),
        "rows": rows,
        "seconds": seconds,
        "rows_per_second": rows / seconds,
        "workload_inventory": {
            "total_click_rows_days_0_89": total_rows,
            "selection_rows_days_25_34": selection_rows,
            "final_rows_days_65_89": final_rows,
        },
    }


def enumerate_matrix(protocol: ProtocolDefinition, final: FinalExperimentConfig) -> dict[str, int]:
    model_runs = (
        len(protocol.model_selection.learning_rates)
        * len(protocol.model_selection.weight_decays)
        * len(protocol.model_selection.dropouts)
        * len(protocol.model_selection.feature_policies)
        * len(protocol.model_selection.seeds)
    )
    delayed_runs = (
        len(protocol.delayed_selection.methods)
        * len(protocol.delayed_selection.wait_days)
        * len(protocol.delayed_selection.seeds)
    )
    sampler_runs = (
        len(protocol.sampler_selection.recent_window_days)
        * len(protocol.sampler_selection.reservoir_capacities)
        * len(protocol.sampler_selection.seeds)
    )
    final_a_runs = len(final.methods) * len(protocol.final_training.seeds)
    final_b_runs = len(final.schedulers) * len(protocol.final_training.seeds)
    offline_runs = len(final.offline_references) * len(protocol.final_training.seeds)
    online_runs = model_runs + delayed_runs + sampler_runs + final_a_runs + final_b_runs
    credits = (
        model_runs * protocol.model_selection.credits_per_run
        + delayed_runs * protocol.delayed_selection.credits_per_run
        + sampler_runs * protocol.sampler_selection.credits_per_run
        + final_a_runs * protocol.final_training.study_a_credits
        + final_b_runs * protocol.final_training.study_b_credits
    )
    return {
        "model_selection_runs": model_runs,
        "delayed_selection_runs": delayed_runs,
        "sampler_selection_runs": sampler_runs,
        "final_study_a_runs": final_a_runs,
        "final_study_b_runs": final_b_runs,
        "offline_reference_runs": offline_runs,
        "online_runs": online_runs,
        "total_runs": online_runs + offline_runs,
        "total_online_credits": credits,
    }


def _worst_case_workload(
    protocol: ProtocolDefinition,
    final: FinalExperimentConfig,
    matrix: dict[str, int],
) -> dict[str, int]:
    selection_runs = (
        matrix["model_selection_runs"]
        + matrix["delayed_selection_runs"]
        + matrix["sampler_selection_runs"]
    )
    es_delayed_runs = len(protocol.delayed_selection.wait_days) * len(
        protocol.delayed_selection.seeds
    )
    es_sampler_runs = matrix["sampler_selection_runs"]
    es_final_a_runs = len(protocol.final_training.seeds)
    es_final_b_runs = matrix["final_study_b_runs"]
    es_core_credits = (
        es_delayed_runs * protocol.delayed_selection.credits_per_run
        + es_sampler_runs * protocol.sampler_selection.credits_per_run
        + es_final_a_runs * protocol.final_training.study_a_credits
        + es_final_b_runs * protocol.final_training.study_b_credits
    )
    dfm_core_credits = len(protocol.final_training.seeds) * protocol.final_training.study_a_credits
    base_core_credits = matrix["total_online_credits"] - es_core_credits - dfm_core_credits
    auxiliary_steps = (
        es_delayed_runs * (1_000 + protocol.delayed_selection.credits_per_run * 200)
        + es_sampler_runs * (1_000 + protocol.sampler_selection.credits_per_run * 200)
        + es_final_a_runs * (1_000 + protocol.final_training.study_a_credits * 200)
        + es_final_b_runs * (1_000 + protocol.final_training.study_b_credits * 200)
    )
    one_model_checkpoint_writes = (
        matrix["model_selection_runs"] * (protocol.model_selection.credits_per_run + 2)
        + (matrix["delayed_selection_runs"] - es_delayed_runs)
        * (protocol.delayed_selection.credits_per_run + 2)
        + (matrix["final_study_a_runs"] - es_final_a_runs)
        * (protocol.final_training.study_a_credits + 2)
    )
    three_model_checkpoint_writes = (
        es_delayed_runs * (protocol.delayed_selection.credits_per_run + 2)
        + es_sampler_runs * (protocol.sampler_selection.credits_per_run + 2)
        + es_final_a_runs * (protocol.final_training.study_a_credits + 2)
        + es_final_b_runs * (protocol.final_training.study_b_credits + 2)
    )
    return {
        "selection_runs": selection_runs,
        "final_online_runs": matrix["final_study_a_runs"] + matrix["final_study_b_runs"],
        "initialization_runs": matrix["online_runs"],
        "initialization_steps": (
            matrix["online_runs"] * protocol.final_training.initialization_steps
        ),
        "base_core_credits": base_core_credits,
        "es_core_credits": es_core_credits,
        "dfm_core_credits": dfm_core_credits,
        "auxiliary_steps": auxiliary_steps,
        "one_model_checkpoint_writes": one_model_checkpoint_writes,
        "three_model_checkpoint_writes": three_model_checkpoint_writes,
        "equivalent_single_model_checkpoint_writes": (
            one_model_checkpoint_writes + three_model_checkpoint_writes * 3
        ),
    }


def estimate_protocol(
    final: FinalExperimentConfig,
    protocol: ProtocolDefinition,
    *,
    config_path: Path,
    protocol_sha256: str,
) -> dict[str, Any]:
    benchmark = _benchmark(final)
    real_pilot = _real_pilot(final, config_path)
    matrix = enumerate_matrix(protocol, final)
    workload = _worst_case_workload(protocol, final, matrix)
    inventory = real_pilot.get("workload_inventory")
    inventory_available = (
        isinstance(inventory, dict)
        and isinstance(inventory.get("selection_rows_days_25_34"), int)
        and not isinstance(inventory.get("selection_rows_days_25_34"), bool)
        and isinstance(inventory.get("final_rows_days_65_89"), int)
        and not isinstance(inventory.get("final_rows_days_65_89"), bool)
    )
    selection_rows = 0
    final_rows = 0
    if inventory_available:
        assert isinstance(inventory, dict)
        raw_selection_rows = inventory["selection_rows_days_25_34"]
        raw_final_rows = inventory["final_rows_days_65_89"]
        assert isinstance(raw_selection_rows, int) and not isinstance(raw_selection_rows, bool)
        assert isinstance(raw_final_rows, int) and not isinstance(raw_final_rows, bool)
        selection_rows = raw_selection_rows
        final_rows = raw_final_rows
    source_artifact_gb = (
        final.pilot.assumed_source_archive_gb + final.pilot.assumed_expanded_source_gb
        if final.pilot.execution_host_has_source_artifacts
        else 0.0
    )
    base_working_disk = (
        source_artifact_gb
        + final.pilot.assumed_prepared_data_gb
        + final.pilot.assumed_runtime_feature_cache_gb
        + final.pilot.temporary_margin_gb
    )
    projections: list[dict[str, Any]] = []
    for steps_per_credit in protocol.final_training.steps_per_credit_candidates:
        optimizer_steps = matrix["total_online_credits"] * steps_per_credit
        optimizer_examples = optimizer_steps * protocol.final_training.batch_size
        projection_valid = bool(benchmark["requested_device_available"]) and (
            inventory_available or not final.require_real_pilot
        )
        base_step_seconds = float(benchmark["training_step_seconds"])
        es_step_seconds = float(benchmark["es_main_training_step_seconds"])
        dfm_step_seconds = float(benchmark["dfm_training_step_seconds"])
        training_seconds = (
            workload["initialization_steps"] * base_step_seconds
            + workload["auxiliary_steps"] * base_step_seconds
            + workload["base_core_credits"] * steps_per_credit * base_step_seconds
            + workload["es_core_credits"] * steps_per_credit * es_step_seconds
            + workload["dfm_core_credits"] * steps_per_credit * dfm_step_seconds
        )
        prediction_examples = selection_rows * workload["selection_runs"] + final_rows * workload[
            "final_online_runs"
        ] * (1 + INTERMEDIATE_BUDGET_FRACTIONS)
        prediction_seconds = prediction_examples / float(
            benchmark["prediction_examples_per_second"]
        )
        checkpoint_seconds = workload["equivalent_single_model_checkpoint_writes"] * float(
            benchmark["checkpoint_write_seconds"]
        )
        lower_hours = (
            (training_seconds + prediction_seconds + checkpoint_seconds) / 3600.0
            if projection_valid
            else None
        )
        upper_hours = lower_hours * PROJECTION_UPPER_MULTIPLIER if lower_hours is not None else None
        checkpoint_working_gb = (
            int(benchmark["checkpoint_bytes"]) * 3 * CHECKPOINT_WORKING_COPIES / BYTES_PER_GB
        )
        checkpoint_snapshot_gb = (
            int(benchmark["model_state_bytes"]) * INTERMEDIATE_BUDGET_FRACTIONS / BYTES_PER_GB
        )
        checkpoint_retained_gb = (
            int(benchmark["checkpoint_bytes"]) * COMPLETED_CHECKPOINTS_RETAINED / BYTES_PER_GB
        )
        prediction_bytes_per_row = float(benchmark["prediction_artifact_bytes_per_row"])
        exposure_bytes_per_row = float(benchmark["exposure_artifact_bytes_per_row"])
        selection_prediction_gb = (
            matrix["model_selection_runs"]
            * selection_rows
            * prediction_bytes_per_row
            * PROJECTION_UPPER_MULTIPLIER
            / BYTES_PER_GB
        )
        final_prediction_gb = (
            (1 + INTERMEDIATE_BUDGET_FRACTIONS)
            * final_rows
            * prediction_bytes_per_row
            * PROJECTION_UPPER_MULTIPLIER
            / BYTES_PER_GB
        )
        pending_prediction_gb = max(selection_prediction_gb, final_prediction_gb)
        exposure_working_gb = (
            protocol.final_training.study_a_credits
            * steps_per_credit
            * protocol.final_training.batch_size
            * exposure_bytes_per_row
            * PROJECTION_UPPER_MULTIPLIER
            / BYTES_PER_GB
        )
        report_gb = matrix["total_runs"] * REPORT_BYTES_PER_RUN / BYTES_PER_GB
        working_disk = (
            base_working_disk
            + checkpoint_working_gb
            + checkpoint_snapshot_gb
            + pending_prediction_gb
            + exposure_working_gb
            + report_gb
        )
        retained_disk = checkpoint_retained_gb + report_gb
        caps = final.caps
        cap_checks = {
            "runs": caps.max_runs is not None and matrix["total_runs"] <= caps.max_runs,
            "compute_hours": caps.max_gpu_hours is not None
            and upper_hours is not None
            and upper_hours <= caps.max_gpu_hours,
            "working_disk": caps.max_working_disk_gb is not None
            and working_disk <= caps.max_working_disk_gb,
            "retained_disk": caps.max_retained_disk_gb is not None
            and retained_disk <= caps.max_retained_disk_gb,
        }
        projections.append(
            {
                "steps_per_credit": steps_per_credit,
                "optimizer_steps": optimizer_steps,
                "optimizer_examples": optimizer_examples,
                "workload": {
                    **workload,
                    "prediction_examples": prediction_examples,
                    "training_seconds": training_seconds,
                    "prediction_seconds": prediction_seconds,
                    "checkpoint_seconds": checkpoint_seconds,
                },
                "measured_compute_hours_range": (
                    [lower_hours, upper_hours] if lower_hours is not None else None
                ),
                "projection_valid_for_requested_device": projection_valid,
                "working_disk_gb": working_disk,
                "working_disk_components_gb": {
                    "execution_inputs_and_margin": base_working_disk,
                    "rolling_esdfm_checkpoint": checkpoint_working_gb,
                    "intermediate_model_snapshots": checkpoint_snapshot_gb,
                    "pending_prediction_ledgers": pending_prediction_gb,
                    "current_exposure_ledger": exposure_working_gb,
                    "aggregate_reports": report_gb,
                },
                "retained_disk_gb": retained_disk,
                "host_memory_requirement_gb": max(
                    float(benchmark["peak_host_memory_gb"]),
                    final.pilot.assumed_host_memory_gb,
                ),
                "cap_checks": cap_checks,
                "fits_caps": all(cap_checks.values()),
            }
        )
    eligible = [item["steps_per_credit"] for item in projections if item["fits_caps"]]
    blockers: list[str] = []
    if not final.caps.complete:
        blockers.append("USER_RESOURCE_CAPS_REQUIRED")
    if not bool(benchmark["requested_device_available"]):
        blockers.append("REQUESTED_ACCELERATOR_UNAVAILABLE")
    if final.require_real_pilot and real_pilot["status"] != "measured":
        blockers.append("REAL_DATA_PILOT_REQUIRED")
    if final.require_real_pilot and not inventory_available:
        blockers.append("REAL_WORKLOAD_INVENTORY_REQUIRED")
    if not eligible:
        blockers.append("NO_STEPS_PER_CREDIT_CANDIDATE_FITS_CAPS")
    return {
        "manifest_version": 1,
        "feasibility_model_version": 2,
        "status": "passed" if not blockers else "blocked",
        "protocol_sha256": protocol_sha256,
        "matrix": matrix,
        "worst_case_workload": workload,
        "benchmark": benchmark,
        "real_data_pilot": real_pilot,
        "projections": projections,
        "selected_steps_per_credit": max(eligible) if eligible and not blockers else None,
        "blockers": blockers,
        "assumptions": {
            "compute_upper_multiplier": PROJECTION_UPPER_MULTIPLIER,
            "report_bytes_per_run": REPORT_BYTES_PER_RUN,
            "sequential_run_execution": True,
            "checkpoint_working_copies": CHECKPOINT_WORKING_COPIES,
            "completed_checkpoints_retained": COMPLETED_CHECKPOINTS_RETAINED,
            "completed_row_level_artifacts_pruned_after_aggregate_verification": True,
            "quality_metrics_used_for_steps_choice": False,
            "cross_device_extrapolation": False,
            "cross_batch_extrapolation": False,
            "execution_host_has_source_artifacts": (
                final.pilot.execution_host_has_source_artifacts
            ),
            "initialization_training_included": True,
            "es_dfm_auxiliary_training_included": True,
            "method_specific_core_training_included": True,
            "prediction_and_intermediate_inference_included": True,
            "checkpoint_io_included": True,
            "worst_case_selection_outcome": "large-feature ES-DFM",
        },
    }
