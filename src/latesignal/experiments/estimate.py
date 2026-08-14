"""Bounded benchmark and exact authored-matrix feasibility projection."""

from __future__ import annotations

import io
import math
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl
import torch

from latesignal.contracts.protocol import FinalExperimentConfig, ProtocolDefinition
from latesignal.data.schema import CATEGORICAL_CLICK_FIELDS
from latesignal.features.hashing import categorical_bucket
from latesignal.models.conversion_mlp import CategoricalSpec, ConversionMLP

BYTES_PER_GB = 1024**3
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
    model = ConversionMLP(
        specs,
        dropout=0.0,
    ).to(device)
    batch_size = final.pilot.benchmark_batch_size
    categorical = {
        field: torch.arange(batch_size, device=device, dtype=torch.long) % 8 for field in fields
    }
    numeric = torch.zeros((batch_size, 4), device=device)
    targets = torch.arange(batch_size, device=device, dtype=torch.float32) % 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    model.train()
    _synchronize(device)
    training_start = time.perf_counter()
    for _ in range(final.pilot.benchmark_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(categorical, numeric)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    _synchronize(device)
    training_seconds = time.perf_counter() - training_start
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
        "training_examples": final.pilot.benchmark_steps * batch_size,
        "training_seconds": training_seconds,
        "training_examples_per_second": final.pilot.benchmark_steps * batch_size / training_seconds,
        "prediction_examples": prediction_batches * batch_size,
        "prediction_seconds": prediction_seconds,
        "prediction_examples_per_second": prediction_batches * batch_size / prediction_seconds,
        "synthetic_preparation_rows": final.pilot.benchmark_examples,
        "synthetic_preparation_seconds": preparation_seconds,
        "synthetic_preparation_rows_per_second": final.pilot.benchmark_examples
        / preparation_seconds,
        "checkpoint_bytes": len(checkpoint.getvalue()),
        "checkpoint_write_seconds": checkpoint_seconds,
        "peak_host_memory_gb": peak_bytes / BYTES_PER_GB,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }


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
    return {
        "status": "measured",
        "prepared_root": str(prepared),
        "click_days": len(partitions),
        "files": len(files),
        "rows": rows,
        "seconds": seconds,
        "rows_per_second": rows / seconds,
    }


def _matrix(protocol: ProtocolDefinition, final: FinalExperimentConfig) -> dict[str, int]:
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


def estimate_protocol(
    final: FinalExperimentConfig,
    protocol: ProtocolDefinition,
    *,
    config_path: Path,
    protocol_sha256: str,
) -> dict[str, Any]:
    benchmark = _benchmark(final)
    real_pilot = _real_pilot(final, config_path)
    matrix = _matrix(protocol, final)
    working_disk = (
        final.pilot.assumed_source_archive_gb
        + final.pilot.assumed_expanded_source_gb
        + final.pilot.assumed_prepared_data_gb
        + final.pilot.temporary_margin_gb
    )
    projections: list[dict[str, Any]] = []
    for steps_per_credit in protocol.final_training.steps_per_credit_candidates:
        optimizer_steps = matrix["total_online_credits"] * steps_per_credit
        optimizer_examples = optimizer_steps * protocol.final_training.batch_size
        measured_rate = float(benchmark["training_examples_per_second"])
        projection_valid = bool(benchmark["requested_device_available"])
        lower_hours = optimizer_examples / measured_rate / 3600.0 if projection_valid else None
        upper_hours = lower_hours * 1.5 if lower_hours is not None else None
        checkpoint_gb = (
            int(benchmark["checkpoint_bytes"])
            * (matrix["total_online_credits"] + matrix["online_runs"])
            / BYTES_PER_GB
        )
        report_gb = matrix["total_runs"] * 250_000 / BYTES_PER_GB
        retained_disk = checkpoint_gb + report_gb
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
                "measured_compute_hours_range": (
                    [lower_hours, upper_hours] if lower_hours is not None else None
                ),
                "projection_valid_for_requested_device": projection_valid,
                "working_disk_gb": working_disk,
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
    if not eligible:
        blockers.append("NO_STEPS_PER_CREDIT_CANDIDATE_FITS_CAPS")
    return {
        "manifest_version": 1,
        "status": "passed" if not blockers else "blocked",
        "protocol_sha256": protocol_sha256,
        "matrix": matrix,
        "benchmark": benchmark,
        "real_data_pilot": real_pilot,
        "projections": projections,
        "selected_steps_per_credit": max(eligible) if eligible and not blockers else None,
        "blockers": blockers,
        "assumptions": {
            "compute_upper_multiplier": 1.5,
            "report_bytes_per_run": 250_000,
            "quality_metrics_used_for_steps_choice": False,
            "cross_device_extrapolation": False,
        },
    }
