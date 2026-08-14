#!/usr/bin/env python3
"""Verify and seal one same-commit final-stage infrastructure recovery."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MAX_GPU_SECONDS = 90_000
_MAX_WORKING_KIB = 26 * 1024 * 1024
_MIN_FREE_KIB = 5 * 1024 * 1024
_ACCOUNTING_RESERVE_SECONDS = 120
_ALLOWED_DIFF_PATHS = {
    "tests/unit/test_remote_gpu_script.py",
    "tools/gpu-study.sh",
    "tools/measure-gpu-study-working-set.sh",
    "tools/prepare-final-resume.py",
    "tools/run-gpu-study-remote.sh",
    "tools/start-gpu-study-remote.sh",
}
_RECOVERY_STAGES = {
    "bootstrap",
    "recovery_verification",
    "final_39",
    "aggregate",
    "recovery_collection_transition",
    "recovery_provenance",
    "collection_manifest",
    "retention",
}
_PRESTART_FAILURE_REASONS = {
    "prestart_gpu_lock_race",
    "prestart_gpu_occupancy_race",
    "prestart_gpu_signal_lost",
}


def _canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"final recovery evidence is missing or redirected: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"final recovery JSON is malformed: {path}")
    return value


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"final recovery evidence is missing or redirected: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_digest(value: dict[str, Any], field: str, description: str) -> None:
    expected = value.get(field)
    unsigned = {key: item for key, item in value.items() if key != field}
    if (
        not isinstance(expected, str)
        or hashlib.sha256(_canonical(unsigned)).hexdigest() != expected
    ):
        raise ValueError(f"{description} digest changed")


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@contextmanager
def _exclusive_job_lock(job: Path) -> Iterator[None]:
    lock_path = job / "job.lock"
    if lock_path.is_symlink() or not lock_path.is_file():
        raise ValueError("final recovery job lock is missing or redirected")
    with lock_path.open("r+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("final recovery job is still active") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def verify_driver_diff(repository: Path, execution_commit: str, driver_commit: str) -> list[str]:
    """Require one to three linear, tooling-only recovery commits."""

    if (
        not re.fullmatch(r"[0-9a-f]{40}", execution_commit)
        or not re.fullmatch(r"[0-9a-f]{40}", driver_commit)
        or execution_commit == driver_commit
        or _git(repository, "rev-parse", "HEAD") != driver_commit
        or _git(repository, "status", "--porcelain", "--untracked-files=all")
    ):
        raise ValueError("final recovery driver repository identity is invalid")
    revisions = _git(
        repository,
        "rev-list",
        "--reverse",
        "--ancestry-path",
        f"{execution_commit}..{driver_commit}",
    ).splitlines()
    previous = execution_commit
    if not 1 <= len(revisions) <= 3:
        raise ValueError("final recovery driver must be a bounded linear child of execution")
    for revision in revisions:
        fields = _git(repository, "rev-list", "--parents", "-n", "1", revision).split()
        if len(fields) != 2 or fields[0] != revision or fields[1] != previous:
            raise ValueError("final recovery driver chain must be linear and merge-free")
        previous = revision
    if previous != driver_commit:
        raise ValueError("final recovery driver chain does not reach the reviewed driver")
    changed = set(
        _git(repository, "diff", "--name-only", execution_commit, driver_commit).splitlines()
    )
    if changed != _ALLOWED_DIFF_PATHS:
        raise ValueError("final recovery driver diff is outside the reviewed exact allowlist")
    return sorted(changed)


def _timestamp(value: object, description: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{description} timestamp is missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{description} timestamp is not timezone-aware")
    return parsed.astimezone(timezone.utc)  # noqa: UP017 - standalone Python 3.9 support


def _verify_failed_receipts(
    job: Path,
    *,
    execution_commit: str,
    gpu_uuid: str,
    allowed_stages: set[str],
    minimum_gpu_seconds: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], int]:
    state = _read(job / "state.json")
    exit_receipt = _read(job / "exit.json")
    failure = _read(job / "resource-failure.json")
    heartbeat = _read(job / "heartbeat.json")
    launch_id = state.get("launch_id")
    common = (execution_commit, gpu_uuid, launch_id)
    if (
        state.get("status") != "failed"
        or state.get("stage") not in allowed_stages
        or state.get("exit_code") != 124
        or (state.get("commit"), state.get("gpu_uuid"), launch_id) != common
        or exit_receipt.get("status") != "failed"
        or exit_receipt.get("stage") != state.get("stage")
        or exit_receipt.get("reason") != "working_disk_measurement_failed"
        or exit_receipt.get("exit_code") != 124
        or (
            exit_receipt.get("commit"),
            exit_receipt.get("gpu_uuid"),
            exit_receipt.get("launch_id"),
        )
        != common
        or failure.get("status") != "failed"
        or failure.get("reason") != "working_disk_measurement_failed"
        or failure.get("detail") != ""
        or (failure.get("commit"), failure.get("gpu_uuid"), failure.get("launch_id")) != common
        or heartbeat.get("stage") != state.get("stage")
        or (heartbeat.get("commit"), heartbeat.get("gpu_uuid"), heartbeat.get("launch_id"))
        != common
        or not isinstance(launch_id, str)
        or not re.fullmatch(r"[A-Za-z0-9-]+", launch_id)
    ):
        raise ValueError("job is not the exact recoverable final disk-measurement failure")
    gpu_seconds = heartbeat.get("gpu_active_seconds")
    working_kib = heartbeat.get("working_kib")
    free_kib = heartbeat.get("free_kib")
    if (
        isinstance(gpu_seconds, bool)
        or not isinstance(gpu_seconds, int)
        or gpu_seconds <= 0
        or (minimum_gpu_seconds is not None and gpu_seconds < minimum_gpu_seconds)
        or isinstance(working_kib, bool)
        or not isinstance(working_kib, int)
        or working_kib < 0
        or working_kib > _MAX_WORKING_KIB
        or isinstance(free_kib, bool)
        or not isinstance(free_kib, int)
        or free_kib < _MIN_FREE_KIB
    ):
        raise ValueError("last successful resource measurement was outside the authored caps")
    gap = int(
        (
            _timestamp(exit_receipt.get("finished_at"), "exit")
            - _timestamp(heartbeat.get("updated_at"), "heartbeat")
        ).total_seconds()
    )
    if gap < 0 or gap > 90:
        raise ValueError("final recovery terminal GPU accounting gap is invalid")
    prior_gpu_seconds = gpu_seconds + gap
    if prior_gpu_seconds + _ACCOUNTING_RESERVE_SECONDS >= _MAX_GPU_SECONDS:
        raise ValueError("final recovery lacks budget after the mandatory accounting reserve")
    return state, exit_receipt, failure, heartbeat, prior_gpu_seconds


def _read_optional(path: Path) -> dict[str, Any] | None:
    if path.exists() or path.is_symlink():
        return _read(path)
    return None


def _receipt_snapshot(job: Path) -> tuple[dict[str, Any], dict[str, str]]:
    receipts: dict[str, Any] = {}
    digests: dict[str, str] = {}
    for name, filename in (
        ("state", "state.json"),
        ("started", "started.json"),
        ("exit", "exit.json"),
        ("resource_failure", "resource-failure.json"),
        ("heartbeat", "heartbeat.json"),
    ):
        path = job / filename
        value = _read_optional(path)
        if value is not None:
            receipts[name] = value
            digests[name] = _sha256(path)
    return receipts, digests


def _bounded_prior_gpu_seconds(value: int) -> int:
    if value + _ACCOUNTING_RESERVE_SECONDS >= _MAX_GPU_SECONDS:
        raise ValueError("final recovery lacks budget after the mandatory accounting reserve")
    return value


def _retry_interruption_evidence(
    job: Path,
    *,
    execution_commit: str,
    gpu_uuid: str,
    recovery_launch_id: str,
    minimum_gpu_seconds: int,
    predecessor_launch_id: str,
) -> tuple[dict[str, object], int]:
    """Verify a sealed recovery interruption not covered by a normal heartbeat."""

    state = _read(job / "state.json")
    started = _read_optional(job / "started.json")
    exit_receipt = _read_optional(job / "exit.json")
    failure = _read_optional(job / "resource-failure.json")
    heartbeat = _read_optional(job / "heartbeat.json")
    common = (execution_commit, gpu_uuid, recovery_launch_id)
    state_common = (state.get("commit"), state.get("gpu_uuid"), state.get("launch_id"))
    started_common = (
        started.get("commit") if started else None,
        started.get("gpu_uuid") if started else None,
        started.get("launch_id") if started else None,
    )
    exit_common = (
        exit_receipt.get("commit") if exit_receipt else None,
        exit_receipt.get("gpu_uuid") if exit_receipt else None,
        exit_receipt.get("launch_id") if exit_receipt else None,
    )
    failure_common = (
        failure.get("commit") if failure else None,
        failure.get("gpu_uuid") if failure else None,
        failure.get("launch_id") if failure else None,
    )
    heartbeat_common = (
        heartbeat.get("commit") if heartbeat else None,
        heartbeat.get("gpu_uuid") if heartbeat else None,
        heartbeat.get("launch_id") if heartbeat else None,
    )
    started_launch = started.get("launch_id") if started else None
    exit_launch = exit_receipt.get("launch_id") if exit_receipt else None
    failure_launch = failure.get("launch_id") if failure else None
    heartbeat_launch = heartbeat.get("launch_id") if heartbeat else None
    predecessor_common = (execution_commit, gpu_uuid, predecessor_launch_id)
    reason: str
    accounting_basis: str
    stage: str
    terminal_gap = 0

    if (
        state_common == common
        and state.get("status") == "failed"
        and state.get("stage") == "bootstrap"
        and state.get("exit_code") == 1
        and exit_receipt is not None
        and exit_common == common
        and exit_receipt.get("status") == "failed"
        and exit_receipt.get("stage") == "bootstrap"
        and exit_receipt.get("exit_code") == 1
        and failure is not None
        and failure_common == common
        and failure.get("status") == "failed"
        and failure.get("reason") in _PRESTART_FAILURE_REASONS
        and failure.get("detail") == ""
        and started_launch != recovery_launch_id
        and heartbeat_launch != recovery_launch_id
    ):
        reason = str(failure["reason"])
        stage = "bootstrap"
        accounting_basis = "prior_import_plus_mandatory_reserve"
        prior_gpu_seconds = minimum_gpu_seconds
    elif (
        state_common == common
        and state.get("status") == "failed"
        and state.get("stage") == "retention"
        and state.get("exit_code") == 4
        and exit_receipt is not None
        and exit_common == common
        and exit_receipt.get("status") == "failed"
        and exit_receipt.get("stage") == "retention"
        and exit_receipt.get("exit_code") == 4
        and failure is not None
        and failure_common == common
        and failure.get("status") == "failed"
        and failure.get("reason") == "retained_disk_measurement_failed"
        and failure.get("detail") == ""
        and started is not None
        and started_common == common
        and started.get("status") == "started"
    ):
        if heartbeat is not None and heartbeat_common == common:
            heartbeat_gpu_seconds = heartbeat.get("gpu_active_seconds")
            working_kib = heartbeat.get("working_kib")
            free_kib = heartbeat.get("free_kib")
            if (
                isinstance(heartbeat_gpu_seconds, bool)
                or not isinstance(heartbeat_gpu_seconds, int)
                or heartbeat_gpu_seconds < minimum_gpu_seconds
                or heartbeat.get("stage") != "retention"
                or isinstance(working_kib, bool)
                or not isinstance(working_kib, int)
                or working_kib < 0
                or working_kib > _MAX_WORKING_KIB
                or isinstance(free_kib, bool)
                or not isinstance(free_kib, int)
                or free_kib < _MIN_FREE_KIB
            ):
                raise ValueError("retention failure heartbeat is outside the authored caps")
            terminal_gap = int(
                (
                    _timestamp(exit_receipt.get("finished_at"), "exit")
                    - _timestamp(heartbeat.get("updated_at"), "heartbeat")
                ).total_seconds()
            )
            prior_gpu_seconds = heartbeat_gpu_seconds + terminal_gap
            accounting_basis = "heartbeat_plus_terminal_exit_gap"
        else:
            terminal_gap = int(
                (
                    _timestamp(exit_receipt.get("finished_at"), "exit")
                    - _timestamp(started.get("started_at"), "started")
                ).total_seconds()
            )
            prior_gpu_seconds = minimum_gpu_seconds + terminal_gap
            accounting_basis = "prior_import_plus_reserve_plus_full_launch_wall_time"
        if terminal_gap < 0 or terminal_gap > 180:
            raise ValueError("retention failure accounting interval is invalid")
        reason = "retained_disk_measurement_failed"
        stage = "retention"
    elif (
        state_common == common
        and state.get("status") == "failed"
        and state.get("stage") in _RECOVERY_STAGES
        and state.get("exit_code") == 124
        and exit_receipt is not None
        and exit_common == common
        and exit_receipt.get("status") == "failed"
        and exit_receipt.get("stage") == state.get("stage")
        and exit_receipt.get("reason") == "working_disk_measurement_failed"
        and exit_receipt.get("exit_code") == 124
        and failure is not None
        and failure_common == common
        and failure.get("status") == "failed"
        and failure.get("reason") == "working_disk_measurement_failed"
        and failure.get("detail") == ""
        and started is not None
        and started_common == common
        and started.get("status") == "started"
        and heartbeat_launch != recovery_launch_id
    ):
        terminal_gap = int(
            (
                _timestamp(exit_receipt.get("finished_at"), "exit")
                - _timestamp(started.get("started_at"), "started")
            ).total_seconds()
        )
        if terminal_gap < 0 or terminal_gap > _ACCOUNTING_RESERVE_SECONDS:
            raise ValueError("early watchdog failure accounting interval is invalid")
        reason = "working_disk_measurement_failed_before_first_heartbeat"
        stage = str(state["stage"])
        accounting_basis = "prior_import_plus_reserve_plus_full_launch_wall_time"
        prior_gpu_seconds = minimum_gpu_seconds + terminal_gap
    else:
        state_is_current = state_common == common
        started_is_current = started is not None and started_common == common
        if (
            not started_is_current
            or started is None
            or started.get("status") != "started"
            or (
                state_is_current
                and (
                    state.get("status") not in {"running", "retrying_infrastructure_failure"}
                    or state.get("stage") not in _RECOVERY_STAGES
                )
            )
            or (not state_is_current and state_common != predecessor_common)
            or exit_launch == recovery_launch_id
            or failure_launch == recovery_launch_id
        ):
            raise ValueError("sealed recovery interruption is not safely classifiable")
        observed_at = datetime.now(
            timezone.utc  # noqa: UP017 - standalone Python 3.9 support
        )
        if heartbeat is not None and heartbeat_common == common:
            heartbeat_gpu_seconds = heartbeat.get("gpu_active_seconds")
            working_kib = heartbeat.get("working_kib")
            free_kib = heartbeat.get("free_kib")
            if (
                isinstance(heartbeat_gpu_seconds, bool)
                or not isinstance(heartbeat_gpu_seconds, int)
                or heartbeat_gpu_seconds < minimum_gpu_seconds
                or heartbeat.get("stage") not in _RECOVERY_STAGES
                or isinstance(working_kib, bool)
                or not isinstance(working_kib, int)
                or working_kib < 0
                or working_kib > _MAX_WORKING_KIB
                or isinstance(free_kib, bool)
                or not isinstance(free_kib, int)
                or free_kib < _MIN_FREE_KIB
            ):
                raise ValueError("interrupted recovery heartbeat undercounts prior GPU time")
            terminal_gap = int(
                (observed_at - _timestamp(heartbeat.get("updated_at"), "heartbeat")).total_seconds()
            )
            prior_gpu_seconds = heartbeat_gpu_seconds + terminal_gap
            accounting_basis = "heartbeat_plus_conservative_wall_time_to_observation"
        else:
            if started is None:
                raise ValueError("interrupted recovery start receipt is missing")
            terminal_gap = int(
                (observed_at - _timestamp(started.get("started_at"), "started")).total_seconds()
            )
            prior_gpu_seconds = minimum_gpu_seconds + terminal_gap
            accounting_basis = "prior_import_plus_reserve_plus_full_launch_wall_time"
        if terminal_gap < 0:
            raise ValueError("interrupted recovery accounting interval is invalid")
        reason = "unclean_remote_interruption"
        stage = str(state.get("stage")) if state_is_current else "prestage_interruption"

    prior_gpu_seconds = _bounded_prior_gpu_seconds(prior_gpu_seconds)
    receipts, receipt_digests = _receipt_snapshot(job)
    return (
        {
            "launch_id": recovery_launch_id,
            "stage": stage,
            "reason": reason,
            "receipts": receipts,
            "receipt_file_sha256": receipt_digests,
            "terminal_gap_seconds": terminal_gap,
            "imported_gpu_seconds": prior_gpu_seconds,
            "accounting_basis": accounting_basis,
        },
        prior_gpu_seconds,
    )


def _verify_scientific_evidence(
    job: Path, execution_commit: str, gpu_uuid: str
) -> dict[str, object]:
    feasibility_path = job / "feasibility.json"
    lock_path = job / "protocol-lock.json"
    quality_path = job / "quality-gate.json"
    selection_path = job / "selection-provenance.json"
    feasibility = _read(feasibility_path)
    lock = _read(lock_path)
    quality = _read(quality_path)
    selection = _read(selection_path)
    selection_results_path = job / "selection" / "selection-results.json"
    selection_manifest_path = job / "selection" / "manifest.json"
    selection_results = _read(selection_results_path)
    selection_manifest = _read(selection_manifest_path)
    _verify_digest(lock, "lock_sha256", "protocol lock")
    _verify_digest(quality, "manifest_sha256", "quality gate")
    _verify_digest(selection, "provenance_sha256", "selection provenance")
    _verify_digest(selection_manifest, "manifest_sha256", "selection manifest")
    matrix = feasibility.get("matrix")
    git_identity = lock.get("git")
    selection_execution = lock.get("selection_execution")
    feasibility_sha256 = hashlib.sha256(_canonical(feasibility)).hexdigest()
    selection_file_sha256 = _sha256(selection_results_path)
    selected_projection = [
        item
        for item in feasibility.get("projections", [])
        if isinstance(item, dict) and item.get("steps_per_credit") == 100
    ]
    if (
        feasibility.get("status") != "passed"
        or feasibility.get("blockers") != []
        or feasibility.get("selected_steps_per_credit") != 100
        or not isinstance(matrix, dict)
        or matrix.get("total_runs") != 89
        or feasibility.get("protocol_sha256") != lock.get("protocol_sha256")
        or feasibility_sha256 != lock.get("feasibility_sha256")
        or matrix != lock.get("final_matrix")
        or len(selected_projection) != 1
        or selected_projection[0].get("fits_caps") is not True
        or not isinstance(selected_projection[0].get("cap_checks"), dict)
        or set(selected_projection[0]["cap_checks"])
        != {"runs", "compute_hours", "working_disk", "retained_disk"}
        or not all(value is True for value in selected_projection[0]["cap_checks"].values())
        or lock.get("status") != "locked"
        or lock.get("publication_eligible") is not True
        or not isinstance(git_identity, dict)
        or git_identity.get("commit") != execution_commit
        or git_identity.get("dirty") is not False
        or not isinstance(lock.get("environment_sha256"), str)
        or quality.get("status") != "passed"
        or quality.get("protocol_lock_sha256") != lock.get("lock_sha256")
        or quality.get("protocol_sha256") != lock.get("protocol_sha256")
        or quality.get("git_commit") != execution_commit
        or selection.get("status") != "verified_cross_commit_reuse"
        or selection.get("target_commit") != execution_commit
        or selection.get("source_gpu_uuid") != gpu_uuid
        or selection.get("target_gpu_uuid") != gpu_uuid
        or selection.get("source_environment_sha256") != lock.get("environment_sha256")
        or selection.get("selection_file_sha256") != selection_file_sha256
        or selection_results.get("protocol_sha256") != lock.get("protocol_sha256")
        or lock.get("selection_file_sha256") != selection_file_sha256
        or not isinstance(selection_execution, dict)
        or selection_execution.get("mode") != "verified_cross_commit_reuse"
        or selection_execution.get("provenance") != selection
        or selection_manifest.get("status") != "complete"
        or selection_manifest.get("candidate_counts")
        != {"model": 36, "delayed": 8, "sampler": 6, "total": 50}
        or selection_manifest.get("selection_results_sha256")
        != hashlib.sha256(_canonical(selection_results)).hexdigest()
    ):
        raise ValueError("final recovery scientific gate evidence is incomplete or changed")
    state_path = job / "final" / "online-state.json"
    online_state: dict[str, Any] | None = None
    completed_count = 0
    if state_path.exists() or state_path.is_symlink():
        online_state = _read(state_path)
        _verify_digest(online_state, "state_sha256", "final online coordinator state")
        completed = online_state.get("completed_runs")
        if (
            online_state.get("status") not in {"running", "complete"}
            or not isinstance(completed, list)
            or not all(isinstance(item, dict) for item in completed)
            or online_state.get("completed_count") != len(completed)
            or online_state.get("next_index") != len(completed)
            or online_state.get("total_runs") != 33
            or len(completed) > 33
        ):
            raise ValueError("final online coordinator state is malformed")
        completed_count = len(completed)
    checkpoint_paths = sorted(
        path
        for pattern in (
            "study-a/runs/*/checkpoints/latest.json",
            "study-b/runs/*/checkpoints/latest.json",
        )
        for path in (job / "final").glob(pattern)
    )
    if len(checkpoint_paths) > 1:
        raise ValueError("final recovery found more than one unpruned online checkpoint")
    incomplete_roots: list[Path] = []
    for pattern in ("study-a/runs/*", "study-b/runs/*"):
        for root in (job / "final").glob(pattern):
            if not root.is_dir() or root.is_symlink() or (root / "retention.json").exists():
                continue
            incomplete_roots.append(root)
    if len(incomplete_roots) > 1:
        raise ValueError("final recovery found more than one incomplete online run")
    if incomplete_roots:
        checkpoint = incomplete_roots[0] / "checkpoints" / "latest.json"
        if checkpoint not in checkpoint_paths:
            raise ValueError("incomplete final run has no durable checkpoint")
    return {
        "feasibility_file_sha256": _sha256(feasibility_path),
        "protocol_lock_sha256": lock["lock_sha256"],
        "environment_sha256": lock["environment_sha256"],
        "gpu_uuid": gpu_uuid,
        "quality_gate_sha256": quality["manifest_sha256"],
        "selection_provenance_sha256": selection["provenance_sha256"],
        "selection_file_sha256": selection_file_sha256,
        "online_state_file_sha256": _sha256(state_path) if online_state is not None else None,
        "completed_online_runs": completed_count,
        "partial_checkpoint_files": [
            {
                "path": path.relative_to(job).as_posix(),
                "sha256": _sha256(path),
            }
            for path in checkpoint_paths
        ],
    }


def _failure_evidence(
    job: Path,
    state: dict[str, Any],
    exit_receipt: dict[str, Any],
    heartbeat: dict[str, Any],
    prior_gpu_seconds: int,
) -> dict[str, object]:
    """Embed full canonical receipts before the live files can be overwritten."""

    receipts, receipt_digests = _receipt_snapshot(job)
    return {
        "launch_id": state["launch_id"],
        "stage": state["stage"],
        "reason": exit_receipt["reason"],
        "receipts": receipts,
        "receipt_file_sha256": receipt_digests,
        "heartbeat_gpu_active_seconds": heartbeat["gpu_active_seconds"],
        "terminal_gap_seconds": prior_gpu_seconds - int(heartbeat["gpu_active_seconds"]),
        "imported_gpu_seconds": prior_gpu_seconds,
        "accounting_basis": "heartbeat_plus_terminal_exit_gap",
    }


def _sealed_collection_before_retry(job: Path, provenance_path: Path) -> dict[str, object] | None:
    published_path = job / "final" / "aggregate" / "infrastructure-recovery.json"
    manifest_path = job / "collection-manifest.json"
    published = _read_optional(published_path)
    manifest = _read_optional(manifest_path)
    current = _read(provenance_path)
    allowed_published_digests = {current.get("provenance_sha256")}
    attempts = current.get("attempts")
    if isinstance(attempts, list):
        allowed_published_digests.update(
            item.get("previous_provenance_sha256") for item in attempts if isinstance(item, dict)
        )
    if published is not None:
        _verify_digest(published, "provenance_sha256", "published recovery provenance")
        if published.get("provenance_sha256") not in allowed_published_digests:
            raise ValueError("published recovery provenance differs before retry")
    if manifest is None:
        return None
    if published is None:
        raise ValueError("sealed collection exists without published recovery provenance")
    _verify_digest(manifest, "manifest_sha256", "collection manifest")
    files = manifest.get("files")
    matches = (
        [
            item
            for item in files
            if isinstance(files, list)
            and isinstance(item, dict)
            and item.get("path") == "final/aggregate/infrastructure-recovery.json"
        ]
        if isinstance(files, list)
        else []
    )
    if (
        manifest.get("status") != "verified_aggregate_only"
        or len(matches) != 1
        or matches[0].get("sha256") != _sha256(published_path)
        or matches[0].get("bytes") != published_path.stat().st_size
    ):
        raise ValueError("sealed collection does not bind published recovery provenance")
    return {
        "manifest": manifest,
        "manifest_file_sha256": _sha256(manifest_path),
    }


def _recovery_attempt(
    *,
    index: int,
    driver_commit: str,
    input_failure: dict[str, object],
    scientific: dict[str, object],
    previous_provenance_sha256: str | None = None,
) -> dict[str, object]:
    prior = input_failure.get("imported_gpu_seconds")
    if isinstance(prior, bool) or not isinstance(prior, int):
        raise ValueError("final recovery attempt accounting is malformed")
    attempt: dict[str, object] = {
        "attempt": index,
        "recovery_launch_id": f"recovery-{driver_commit[:12]}-{index}",
        "input_failure": input_failure,
        "imported_gpu_seconds": prior,
        "driver_accounting_reserve_seconds": _ACCOUNTING_RESERVE_SECONDS,
        "scientific_evidence_before_resume": scientific,
    }
    if previous_provenance_sha256 is not None:
        attempt["previous_provenance_sha256"] = previous_provenance_sha256
    return attempt


def _verify_attempt_chain(attempts: object, driver_commit: str) -> list[dict[str, Any]]:
    if (
        not isinstance(attempts, list)
        or not attempts
        or not all(isinstance(item, dict) for item in attempts)
    ):
        raise ValueError("stored final recovery attempts are malformed")
    checked = list(attempts)
    previous_imported: int | None = None
    previous_reserve: int | None = None
    for index, attempt in enumerate(checked, start=1):
        imported = attempt.get("imported_gpu_seconds")
        reserve = attempt.get("driver_accounting_reserve_seconds")
        input_failure = attempt.get("input_failure")
        if (
            attempt.get("attempt") != index
            or attempt.get("recovery_launch_id") != f"recovery-{driver_commit[:12]}-{index}"
            or isinstance(imported, bool)
            or not isinstance(imported, int)
            or imported <= 0
            or reserve != _ACCOUNTING_RESERVE_SECONDS
            or not isinstance(input_failure, dict)
            or input_failure.get("imported_gpu_seconds") != imported
            or (
                index > 1
                and (
                    not isinstance(attempt.get("previous_provenance_sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", str(attempt["previous_provenance_sha256"]))
                )
            )
            or (
                previous_imported is not None
                and previous_reserve is not None
                and imported < previous_imported + previous_reserve
            )
        ):
            raise ValueError("stored final recovery attempt chain is invalid")
        previous_imported = imported
        previous_reserve = reserve
    return checked


def build_provenance(
    driver_repository: Path,
    execution_repository: Path,
    job: Path,
    *,
    execution_commit: str,
    driver_commit: str,
    gpu_uuid: str,
) -> tuple[dict[str, object], str, int]:
    """Verify the failure and return immutable recovery provenance and accounting."""

    if (
        not re.fullmatch(r"GPU-[A-Za-z0-9-]+", gpu_uuid)
        or job.resolve()
        != execution_repository.resolve() / "runs" / "one-shot" / execution_commit[:12]
        or job.is_symlink()
        or _git(execution_repository, "rev-parse", "HEAD") != execution_commit
        or _git(execution_repository, "status", "--porcelain", "--untracked-files=all")
    ):
        raise ValueError("final recovery execution identity is invalid")
    reviewed = verify_driver_diff(driver_repository, execution_commit, driver_commit)
    state, exit_receipt, _failure, heartbeat, prior_gpu_seconds = _verify_failed_receipts(
        job,
        execution_commit=execution_commit,
        gpu_uuid=gpu_uuid,
        allowed_stages={"final_39"},
    )
    scientific = _verify_scientific_evidence(job, execution_commit, gpu_uuid)
    source_failure = _failure_evidence(job, state, exit_receipt, heartbeat, prior_gpu_seconds)
    first_attempt = _recovery_attempt(
        index=1,
        driver_commit=driver_commit,
        input_failure=source_failure,
        scientific=scientific,
    )
    payload: dict[str, object] = {
        "version": 1,
        "status": "verified_same_commit_final_resume",
        "reason": "transient_working_disk_measurement_failure",
        "execution_commit": execution_commit,
        "driver_commit": driver_commit,
        "reviewed_diff_paths": reviewed,
        "source_launch_id": state["launch_id"],
        "gpu_uuid": gpu_uuid,
        "source_stage": "final_39",
        "source_exit_code": 124,
        "source_failure_reason": "working_disk_measurement_failed",
        "source_failure": source_failure,
        "attempts": [first_attempt],
        "resume_semantics": {
            "completed_runs": "hash-verify-and-skip",
            "interrupted_run": "identity-bound-daily-checkpoint",
            "scientific_commit_changed": False,
        },
    }
    payload["provenance_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload, str(first_attempt["recovery_launch_id"]), prior_gpu_seconds


def _write_prepared(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(_canonical(payload))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prepare_locked(
    driver_repository: Path,
    execution_repository: Path,
    job: Path,
    *,
    execution_commit: str,
    driver_commit: str,
    gpu_uuid: str,
) -> tuple[str, int]:
    path = job / "final-recovery-provenance.json"
    if path.is_symlink():
        raise ValueError("final recovery provenance path is redirected")
    if not path.exists():
        payload, launch_id, prior = build_provenance(
            driver_repository,
            execution_repository,
            job,
            execution_commit=execution_commit,
            driver_commit=driver_commit,
            gpu_uuid=gpu_uuid,
        )
        _write_prepared(path, payload)
        return launch_id, prior

    stored = _read(path)
    _verify_digest(stored, "provenance_sha256", "stored final recovery provenance")
    reviewed = verify_driver_diff(driver_repository, execution_commit, driver_commit)
    if (
        stored.get("status") != "verified_same_commit_final_resume"
        or stored.get("execution_commit") != execution_commit
        or stored.get("driver_commit") != driver_commit
        or stored.get("gpu_uuid") != gpu_uuid
        or stored.get("reviewed_diff_paths") != reviewed
    ):
        raise ValueError("stored final recovery provenance identity changed")
    attempts = _verify_attempt_chain(stored.get("attempts"), driver_commit)
    source_failure = stored.get("source_failure")
    if (
        not isinstance(source_failure, dict)
        or attempts[0].get("input_failure") != source_failure
        or stored.get("source_launch_id") != source_failure.get("launch_id")
    ):
        raise ValueError("stored final recovery source evidence changed")
    latest = attempts[-1]
    latest_launch = latest.get("recovery_launch_id")
    latest_prior = latest.get("imported_gpu_seconds")
    if (
        not isinstance(latest_launch, str)
        or not re.fullmatch(r"[A-Za-z0-9-]+", latest_launch)
        or isinstance(latest_prior, bool)
        or not isinstance(latest_prior, int)
    ):
        raise ValueError("stored final recovery attempt is malformed")
    current_state = _read(job / "state.json")
    input_failure = latest.get("input_failure")
    scientific_before_resume = latest.get("scientific_evidence_before_resume")
    predecessor_launch = input_failure.get("launch_id") if isinstance(input_failure, dict) else None
    if (
        not isinstance(input_failure, dict)
        or not isinstance(scientific_before_resume, dict)
        or not isinstance(predecessor_launch, str)
    ):
        raise ValueError("stored final recovery input evidence is malformed")
    started = _read_optional(job / "started.json")
    started_latest = started is not None and started.get("launch_id") == latest_launch
    if current_state.get("launch_id") == predecessor_launch and not started_latest:
        receipts, receipt_digests = _receipt_snapshot(job)
        scientific = _verify_scientific_evidence(job, execution_commit, gpu_uuid)
        if (
            input_failure.get("receipts") != receipts
            or input_failure.get("receipt_file_sha256") != receipt_digests
            or input_failure.get("imported_gpu_seconds") != latest_prior
            or scientific != scientific_before_resume
        ):
            raise ValueError("prepared recovery predecessor evidence changed before launch")
        return latest_launch, latest_prior
    if current_state.get("launch_id") not in {predecessor_launch, latest_launch}:
        raise ValueError("final recovery receipts do not continue the sealed attempt chain")
    minimum_gpu_seconds = latest_prior + _ACCOUNTING_RESERVE_SECONDS
    try:
        state, exit_receipt, _failure, heartbeat, prior = _verify_failed_receipts(
            job,
            execution_commit=execution_commit,
            gpu_uuid=gpu_uuid,
            allowed_stages=_RECOVERY_STAGES,
            minimum_gpu_seconds=minimum_gpu_seconds,
        )
        next_failure = _failure_evidence(job, state, exit_receipt, heartbeat, prior)
    except ValueError:
        next_failure, prior = _retry_interruption_evidence(
            job,
            execution_commit=execution_commit,
            gpu_uuid=gpu_uuid,
            recovery_launch_id=str(latest_launch),
            minimum_gpu_seconds=minimum_gpu_seconds,
            predecessor_launch_id=predecessor_launch,
        )
    sealed_collection = _sealed_collection_before_retry(job, path)
    if sealed_collection is not None:
        next_failure["sealed_collection_manifest_before_resume"] = sealed_collection
    scientific = _verify_scientific_evidence(job, execution_commit, gpu_uuid)
    previous_provenance_sha256 = stored.get("provenance_sha256")
    if not isinstance(previous_provenance_sha256, str):
        raise ValueError("stored final recovery provenance digest is malformed")
    next_attempt = _recovery_attempt(
        index=len(attempts) + 1,
        driver_commit=driver_commit,
        input_failure=next_failure,
        scientific=scientific,
        previous_provenance_sha256=previous_provenance_sha256,
    )
    updated = {key: value for key, value in stored.items() if key != "provenance_sha256"}
    updated["attempts"] = [*attempts, next_attempt]
    updated["provenance_sha256"] = hashlib.sha256(_canonical(updated)).hexdigest()
    _write_prepared(path, updated)
    return str(next_attempt["recovery_launch_id"]), prior


def prepare(
    driver_repository: Path,
    execution_repository: Path,
    job: Path,
    *,
    execution_commit: str,
    driver_commit: str,
    gpu_uuid: str,
) -> tuple[str, int]:
    """Prepare one recovery while proving no prior driver still owns the job."""

    with _exclusive_job_lock(job):
        return _prepare_locked(
            driver_repository,
            execution_repository,
            job,
            execution_commit=execution_commit,
            driver_commit=driver_commit,
            gpu_uuid=gpu_uuid,
        )


def verify(
    job: Path,
    execution_commit: str,
    driver_commit: str,
    gpu_uuid: str,
    recovery_launch_id: str,
) -> dict[str, Any]:
    """Verify prepared provenance without consulting overwritten failure receipts."""

    value = _read(job / "final-recovery-provenance.json")
    _verify_digest(value, "provenance_sha256", "final recovery provenance")
    attempts = _verify_attempt_chain(value.get("attempts"), driver_commit)
    source_failure = value.get("source_failure")
    if (
        value.get("status") != "verified_same_commit_final_resume"
        or value.get("execution_commit") != execution_commit
        or value.get("driver_commit") != driver_commit
        or value.get("gpu_uuid") != gpu_uuid
        or value.get("reviewed_diff_paths") != sorted(_ALLOWED_DIFF_PATHS)
        or not isinstance(source_failure, dict)
        or attempts[0].get("input_failure") != source_failure
        or value.get("source_launch_id") != source_failure.get("launch_id")
        or attempts[-1].get("recovery_launch_id") != recovery_launch_id
    ):
        raise ValueError("prepared final recovery provenance identity changed")
    return value


def publish(
    job: Path,
    execution_commit: str,
    driver_commit: str,
    gpu_uuid: str,
    recovery_launch_id: str,
) -> None:
    """Publish the verified recovery receipt into the collectable aggregate tree."""

    value = verify(job, execution_commit, driver_commit, gpu_uuid, recovery_launch_id)
    source = job / "final-recovery-provenance.json"
    aggregate = job / "final" / "aggregate"
    if aggregate.is_symlink() or not aggregate.is_dir():
        raise ValueError("final aggregate is missing or redirected")
    target = aggregate / "infrastructure-recovery.json"
    if target.is_symlink():
        raise ValueError("final recovery aggregate path is redirected")
    if target.exists():
        if target.read_bytes() == source.read_bytes():
            return
        target_value = _read(target)
        _verify_digest(target_value, "provenance_sha256", "published recovery provenance")
        allowed_previous = {
            item.get("previous_provenance_sha256")
            for item in value["attempts"]
            if isinstance(item, dict)
        }
        if target_value.get("provenance_sha256") not in allowed_previous:
            raise ValueError("published final recovery provenance changed")
    _write_prepared(target, value)


def prepare_collection(
    job: Path,
    execution_commit: str,
    driver_commit: str,
    gpu_uuid: str,
    recovery_launch_id: str,
) -> None:
    """Remove only a superseded collection seal already embedded in provenance."""

    value = verify(job, execution_commit, driver_commit, gpu_uuid, recovery_launch_id)
    manifest_path = job / "collection-manifest.json"
    if not manifest_path.exists() and not manifest_path.is_symlink():
        return
    manifest = _read(manifest_path)
    latest = value["attempts"][-1]
    input_failure = latest.get("input_failure")
    sealed = (
        input_failure.get("sealed_collection_manifest_before_resume")
        if isinstance(input_failure, dict)
        else None
    )
    if (
        not isinstance(sealed, dict)
        or sealed.get("manifest") != manifest
        or sealed.get("manifest_file_sha256") != _sha256(manifest_path)
    ):
        raise ValueError("collection manifest is not an embedded superseded seal")
    manifest_path.unlink()
    descriptor = os.open(job, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    if len(sys.argv) == 6 and sys.argv[1] == "verify-driver":
        paths = verify_driver_diff(Path(sys.argv[2]), sys.argv[3], sys.argv[4])
        if sys.argv[5] != "--quiet":
            print("\n".join(paths))
        return 0
    if len(sys.argv) == 8 and sys.argv[1] == "prepare":
        launch_id, prior = prepare(
            Path(sys.argv[2]),
            Path(sys.argv[3]),
            Path(sys.argv[4]),
            execution_commit=sys.argv[5],
            driver_commit=sys.argv[6],
            gpu_uuid=sys.argv[7],
        )
        print(f"RECOVERY_LAUNCH_ID={launch_id}")
        print(f"PRIOR_GPU_SECONDS={prior}")
        return 0
    if len(sys.argv) == 7 and sys.argv[1] in {"verify", "publish", "prepare-collection"}:
        arguments = (Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
        if sys.argv[1] == "verify":
            verify(*arguments)
            print("Verified same-commit final recovery provenance")
        elif sys.argv[1] == "publish":
            publish(*arguments)
            print("Published final recovery provenance")
        else:
            prepare_collection(*arguments)
            print("Prepared final recovery collection seal")
        return 0
    print(
        "usage: prepare-final-resume.py prepare DRIVER_REPO EXECUTION_REPO JOB "
        "EXECUTION_COMMIT DRIVER_COMMIT GPU_UUID\n"
        "       prepare-final-resume.py verify JOB EXECUTION_COMMIT DRIVER_COMMIT GPU_UUID "
        "RECOVERY_LAUNCH_ID\n"
        "       prepare-final-resume.py publish JOB EXECUTION_COMMIT DRIVER_COMMIT GPU_UUID "
        "RECOVERY_LAUNCH_ID\n"
        "       prepare-final-resume.py prepare-collection JOB EXECUTION_COMMIT DRIVER_COMMIT "
        "GPU_UUID RECOVERY_LAUNCH_ID\n"
        "       prepare-final-resume.py verify-driver REPO EXECUTION_COMMIT DRIVER_COMMIT --quiet",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
