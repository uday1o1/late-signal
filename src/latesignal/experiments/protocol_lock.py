"""Selection decisions and immutable pre-scoring protocol locks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from latesignal.contracts.protocol import FinalExperimentConfig, ProtocolDefinition
from latesignal.contracts.selection import CandidateResult, SelectionResults
from latesignal.data.manifests import (
    canonical_json_bytes,
    read_json,
    sha256_file,
    write_json_atomic,
)
from latesignal.data.prepared import verify_prepared_inventory
from latesignal.errors import ConfigurationError, ConsistencyError
from latesignal.experiments.estimate import enumerate_matrix

DIRECT_PACKAGES = (
    "blake3",
    "lightgbm",
    "numpy",
    "polars",
    "pyarrow",
    "pydantic",
    "PyYAML",
    "scikit-learn",
    "torch",
    "typer",
    "xxhash",
)


def select_candidate[Candidate: CandidateResult](candidates: list[Candidate]) -> Candidate:
    blockers = [
        item for item in candidates if item.status in {"infrastructure_failed", "incomplete"}
    ]
    if blockers:
        raise ConsistencyError(
            "Selection contains a non-scientific failure and cannot be locked",
            details={"config_sha256": [item.config_sha256 for item in blockers]},
        )
    complete = [item for item in candidates if item.status == "complete"]
    if not complete:
        raise ConsistencyError("Selection stage has no complete candidate")
    for item in complete:
        if (
            item.mean_selection_log_loss is None
            or item.measured_compute_seconds is None
            or item.parameter_count is None
        ):
            raise ConsistencyError("Complete selection candidate has missing measurements")
    minimum_loss = min(
        item.mean_selection_log_loss
        for item in complete
        if item.mean_selection_log_loss is not None
    )
    tied = [
        item
        for item in complete
        if item.mean_selection_log_loss is not None
        and item.mean_selection_log_loss <= minimum_loss + 1e-6
    ]

    def tie_key(item: Candidate) -> tuple[float, int, str]:
        if item.measured_compute_seconds is None or item.parameter_count is None:
            raise ConsistencyError("Complete selection candidate has missing tie measurements")
        return item.measured_compute_seconds, item.parameter_count, item.config_sha256

    return min(tied, key=tie_key)


def selection_decisions(results: SelectionResults) -> dict[str, object]:
    model = select_candidate(results.model_candidates)
    delayed = select_candidate(results.delayed_candidates)
    sampler = select_candidate(results.sampler_candidates)
    return {
        "model": model.model_dump(mode="json"),
        "delayed": delayed.model_dump(mode="json"),
        "sampler": sampler.model_dump(mode="json"),
        "derived": {
            "shared_wait_days": delayed.wait_days,
            "study_b_method": delayed.method,
        },
        "tie_policy": {
            "metric_tolerance": 1e-6,
            "order": [
                "mean_selection_log_loss",
                "measured_compute_seconds",
                "parameter_count",
                "config_sha256",
            ],
        },
    }


def _verify_prepared_data(manifest_path: Path) -> dict[str, object]:
    inventory = verify_prepared_inventory(manifest_path)
    return {
        "manifest_path": str(inventory.manifest_path.relative_to(inventory.root)),
        "manifest_sha256": inventory.manifest_sha256,
        "manifest_bytes": inventory.manifest_bytes,
        "verified_files": len(inventory.files),
        "verified_file_bytes": inventory.total_bytes,
    }


def _git_identity(repository: Path, *, allow_dirty: bool) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status_lines = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ConfigurationError(
            "Could not capture the Git identity for the protocol lock"
        ) from error
    dirty = bool(status_lines)
    if dirty and not allow_dirty:
        raise ConfigurationError(
            "Protocol lock refuses a dirty Git tree without --allow-dirty",
            details={"dirty_paths": status_lines},
        )
    return {
        "commit": commit,
        "dirty": dirty,
        "dirty_paths": status_lines,
        "allow_dirty_override": dirty and allow_dirty,
    }


def _environment(repository: Path) -> dict[str, object]:
    lock_sha256, lock_bytes = sha256_file(repository / "uv.lock")
    packages = {name: importlib.metadata.version(name) for name in DIRECT_PACKAGES}
    gpu: dict[str, object] | None = None
    if torch.cuda.is_available():
        try:
            driver = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        except (OSError, subprocess.CalledProcessError) as error:
            raise ConfigurationError(
                "Could not capture the CUDA driver and GPU identity"
            ) from error
        gpu = {"devices": driver}
    return {
        "python": sys.version.split()[0],
        "python_compiler": platform.python_compiler(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "uv_lock_sha256": lock_sha256,
        "uv_lock_bytes": lock_bytes,
    }


def create_protocol_lock(
    final: FinalExperimentConfig,
    protocol: ProtocolDefinition,
    selection: SelectionResults,
    feasibility: dict[str, Any],
    *,
    protocol_sha256: str,
    final_config_path: Path,
    selection_path: Path,
    data_manifest_path: Path,
    output_path: Path,
    repository: Path,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    if selection.protocol_sha256 != protocol_sha256:
        raise ConsistencyError("Selection results do not match the authored protocol")
    if (
        feasibility.get("feasibility_model_version") != 2
        or feasibility.get("status") != "passed"
        or feasibility.get("protocol_sha256") != protocol_sha256
        or feasibility.get("blockers") != []
        or feasibility.get("matrix") != enumerate_matrix(protocol, final)
    ):
        raise ConsistencyError("A passing matched feasibility result is required before locking")
    if feasibility.get("selected_steps_per_credit") not in (
        protocol.final_training.steps_per_credit_candidates
    ):
        raise ConsistencyError("Feasibility selected an unauthorized steps-per-credit value")
    benchmark = feasibility.get("benchmark")
    real_pilot = feasibility.get("real_data_pilot")
    projections = feasibility.get("projections")
    if (
        not isinstance(benchmark, dict)
        or benchmark.get("requested_device_available") is not True
        or not isinstance(real_pilot, dict)
        or (final.require_real_pilot and real_pilot.get("status") != "measured")
        or not isinstance(real_pilot.get("workload_inventory"), dict)
        or not isinstance(projections, list)
    ):
        raise ConsistencyError("Feasibility evidence does not satisfy its measured prerequisites")
    eligible: list[int] = []
    for item in projections:
        if not isinstance(item, dict):
            continue
        steps = item.get("steps_per_credit")
        checks = item.get("cap_checks")
        if (
            item.get("fits_caps") is True
            and isinstance(checks, dict)
            and all(value is True for value in checks.values())
            and isinstance(steps, int)
            and not isinstance(steps, bool)
        ):
            eligible.append(steps)
    if not eligible or feasibility["selected_steps_per_credit"] != max(eligible):
        raise ConsistencyError("Feasibility did not choose the largest candidate fitting all caps")
    if final.target_device == "cuda" and not torch.cuda.is_available():
        raise ConfigurationError("Protocol lock requires the authored CUDA target to be available")
    git = _git_identity(repository, allow_dirty=allow_dirty)
    environment = _environment(repository)
    data = _verify_prepared_data(data_manifest_path)
    final_config_file_sha256, _ = sha256_file(final_config_path)
    selection_file_sha256, _ = sha256_file(selection_path)
    feasibility_sha256 = hashlib.sha256(canonical_json_bytes(feasibility)).hexdigest()
    environment_sha256 = hashlib.sha256(canonical_json_bytes(environment)).hexdigest()
    payload: dict[str, Any] = {
        "lock_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "locked",
        "locked_before_final_scoring": True,
        "publication_eligible": not bool(git["dirty"]),
        "protocol_sha256": protocol_sha256,
        "final_config_file_sha256": final_config_file_sha256,
        "selection_file_sha256": selection_file_sha256,
        "feasibility_sha256": feasibility_sha256,
        "environment_sha256": environment_sha256,
        "data": data,
        "git": git,
        "environment": environment,
        "selection_window": selection.window.model_dump(mode="json"),
        "selection_decisions": selection_decisions(selection),
        "selected_steps_per_credit": feasibility["selected_steps_per_credit"],
        "final_seeds": protocol.final_training.seeds,
        "final_matrix": feasibility["matrix"],
    }
    lock_sha256 = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    lock = {**payload, "lock_sha256": lock_sha256}
    write_json_atomic(output_path, lock)
    return lock


def verify_protocol_lock(path: Path) -> dict[str, Any]:
    lock = read_json(path)
    expected = lock.get("lock_sha256")
    if not isinstance(expected, str):
        raise ConsistencyError("Protocol lock has no hash")
    unhashed = {key: value for key, value in lock.items() if key != "lock_sha256"}
    actual = hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest()
    if actual != expected:
        raise ConsistencyError("Protocol lock content does not match its hash")
    if lock.get("status") != "locked" or lock.get("locked_before_final_scoring") is not True:
        raise ConsistencyError("Protocol lock is not authorized for final scoring")
    return lock
