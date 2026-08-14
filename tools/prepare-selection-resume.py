#!/usr/bin/env python3
"""Verify and seal cross-commit selection reuse after one exact pre-scoring failure."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_EXACT_FAILURE = "Scheduler decisions must occur on daily boundaries"
_REUSED_PATHS = ["selection/manifest.json", "selection/selection-results.json"]
_ALLOWED_DIFF_PATHS = {
    "results/published/synthetic-reproduction.json",
    "src/latesignal/experiments/collection.py",
    "src/latesignal/experiments/production_aggregate.py",
    "src/latesignal/experiments/protocol_lock.py",
    "src/latesignal/scheduling/base.py",
    "tests/unit/test_collection.py",
    "tests/unit/test_production_qualification.py",
    "tests/unit/test_protocol_lock_runtime.py",
    "tests/unit/test_remote_gpu_script.py",
    "tests/unit/test_schedulers.py",
    "tools/gpu-study.sh",
    "tools/prepare-selection-resume.py",
    "tools/run-gpu-study-remote.sh",
    "tools/start-gpu-study-remote.sh",
}
_OLD_BOUNDARY_CHECK = (
    b"    def _window_at(self, simulator_time: int) -> CreditWindow:\n"
    b"        if simulator_time % self.day_seconds:\n"
    b'            raise ConsistencyError("Scheduler decisions must occur on daily boundaries")\n'
)
_NEW_BOUNDARY_CHECK = (
    b"    def _window_at(self, simulator_time: int) -> CreditWindow:\n"
    b"        first_boundary = self.windows[0].start_time\n"
    b"        if (simulator_time - first_boundary) % self.day_seconds:\n"
    b'            raise ConsistencyError("Scheduler decisions must occur on daily boundaries")\n'
)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"resume evidence is missing or redirected: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"resume JSON is malformed: {path}")
    return value


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


def _verify_target_diff(repository: Path, source_commit: str, target_commit: str) -> list[str]:
    if _git(repository, "rev-parse", "HEAD") != target_commit:
        raise ValueError("resume target repository is not at the target commit")
    ancestry = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", source_commit, target_commit],
        check=False,
    )
    if ancestry.returncode != 0:
        raise ValueError("resume source is not an ancestor of the target commit")
    raw_chain = _git(
        repository,
        "rev-list",
        "--reverse",
        "--ancestry-path",
        "--parents",
        f"{source_commit}..{target_commit}",
    ).splitlines()
    if not 1 <= len(raw_chain) <= 2:
        raise ValueError("resume recovery chain is longer than the reviewed bound")
    reviewed_commits: list[str] = []
    previous = source_commit
    for raw in raw_chain:
        fields = raw.split()
        if len(fields) != 2 or fields[1] != previous:
            raise ValueError("resume recovery chain contains a merge or discontinuity")
        commit = fields[0]
        per_commit = set(_git(repository, "diff", "--name-only", previous, commit).splitlines())
        if not per_commit or not per_commit.issubset(_ALLOWED_DIFF_PATHS):
            raise ValueError("resume recovery commit is outside the reviewed allowlist")
        reviewed_commits.append(commit)
        previous = commit
    if previous != target_commit:
        raise ValueError("resume recovery chain does not end at the target commit")
    changed = set(
        _git(repository, "diff", "--name-only", source_commit, target_commit).splitlines()
    )
    if changed != _ALLOWED_DIFF_PATHS:
        raise ValueError("resume source-to-target diff is outside the reviewed allowlist")
    relative = "src/latesignal/scheduling/base.py"
    before = subprocess.run(
        ["git", "-C", str(repository), "show", f"{source_commit}:{relative}"],
        check=True,
        capture_output=True,
    ).stdout
    after = subprocess.run(
        ["git", "-C", str(repository), "show", f"{target_commit}:{relative}"],
        check=True,
        capture_output=True,
    ).stdout
    if before.count(_OLD_BOUNDARY_CHECK) != 1 or after != before.replace(
        _OLD_BOUNDARY_CHECK, _NEW_BOUNDARY_CHECK, 1
    ):
        raise ValueError("scheduler boundary fix differs from the reviewed exact replacement")
    return reviewed_commits


def _selection_identity(root: Path) -> tuple[str, dict[str, Any]]:
    selection_path = root / "selection" / "selection-results.json"
    selection = _read(selection_path)
    selection_file_sha256 = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    selection_manifest = _read(root / "selection" / "manifest.json")
    _verify_digest(selection_manifest, "manifest_sha256", "selection manifest")
    if (
        selection_manifest.get("selection_results_sha256")
        != hashlib.sha256(_canonical(selection)).hexdigest()
        or selection_manifest.get("status") != "complete"
        or selection_manifest.get("candidate_counts")
        != {"model": 36, "delayed": 8, "sampler": 6, "total": 50}
    ):
        raise ValueError("selection evidence changed or is incomplete")
    return selection_file_sha256, selection_manifest


def build_provenance(
    source: Path,
    target: Path,
    *,
    source_commit: str,
    target_commit: str,
    target_gpu_uuid: str,
) -> dict[str, object]:
    """Verify the exact safe failure and describe the only reusable evidence."""

    if (
        not re.fullmatch(r"[0-9a-f]{40}", source_commit)
        or not re.fullmatch(r"[0-9a-f]{40}", target_commit)
        or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", target_gpu_uuid)
        or source_commit == target_commit
    ):
        raise ValueError("resume commit identities are invalid")
    repository = target.parents[2]
    reviewed_commits = _verify_target_diff(repository, source_commit, target_commit)
    state = _read(source / "state.json")
    exit_receipt = _read(source / "exit.json")
    heartbeat = _read(source / "heartbeat.json")
    lock = _read(source / "protocol-lock.json")
    _verify_digest(lock, "lock_sha256", "source protocol lock")
    log_path = source / "job.log"
    if log_path.is_symlink() or not log_path.is_file():
        raise ValueError("resume source job log is missing or redirected")
    job_log = log_path.read_text(encoding="utf-8")
    exact_error = f'"message": "{_EXACT_FAILURE}"'
    compact_error = f'"message":"{_EXACT_FAILURE}"'
    if job_log.count(exact_error) + job_log.count(compact_error) != 1:
        raise ValueError("resume source did not fail for the exact reviewed scheduler reason")
    source_git = lock.get("git")
    source_data = lock.get("data")
    if (
        state.get("status") != "failed"
        or state.get("stage") != "cuda_resume_qualification"
        or state.get("commit") != source_commit
        or state.get("exit_code") != 5
        or exit_receipt.get("status") != "failed"
        or exit_receipt.get("stage") != "cuda_resume_qualification"
        or exit_receipt.get("commit") != source_commit
        or exit_receipt.get("exit_code") != 5
        or state.get("launch_id") != exit_receipt.get("launch_id")
        or heartbeat.get("commit") != source_commit
        or heartbeat.get("launch_id") != state.get("launch_id")
        or heartbeat.get("gpu_uuid") != state.get("gpu_uuid")
        or state.get("gpu_uuid") != target_gpu_uuid
        or lock.get("status") != "locked"
        or lock.get("publication_eligible") is not True
        or not isinstance(source_git, dict)
        or source_git.get("commit") != source_commit
        or source_git.get("dirty") is not False
        or not isinstance(source_data, dict)
        or not isinstance(source_data.get("manifest_sha256"), str)
        or not isinstance(lock.get("protocol_sha256"), str)
        or not isinstance(lock.get("final_config_file_sha256"), str)
        or not isinstance(lock.get("environment_sha256"), str)
        or lock.get("selected_steps_per_credit") not in (100, 250, 500)
    ):
        raise ValueError("resume source is not the exact clean pre-scoring qualification failure")

    selection_file_sha256, _ = _selection_identity(source)
    if lock.get("selection_file_sha256") != selection_file_sha256:
        raise ValueError("source protocol lock does not bind the reused selection")
    heartbeat_seconds = heartbeat.get("gpu_active_seconds")
    heartbeat_updated = heartbeat.get("updated_at")
    finished = exit_receipt.get("finished_at")
    if (
        isinstance(heartbeat_seconds, bool)
        or not isinstance(heartbeat_seconds, int)
        or not isinstance(heartbeat_updated, str)
        or not isinstance(finished, str)
    ):
        raise ValueError("resume source lacks unambiguous GPU accounting evidence")
    updated_at = datetime.fromisoformat(heartbeat_updated.replace("Z", "+00:00"))
    finished_at = datetime.fromisoformat(finished.replace("Z", "+00:00"))
    if updated_at.tzinfo is None or finished_at.tzinfo is None:
        raise ValueError("resume source timestamps are not timezone-aware")
    terminal_gap = int((finished_at.astimezone(UTC) - updated_at.astimezone(UTC)).total_seconds())
    if terminal_gap < 0 or terminal_gap > 120:
        raise ValueError("resume source terminal GPU accounting gap is invalid")
    prior_gpu_seconds = heartbeat_seconds + terminal_gap
    if prior_gpu_seconds <= 0 or prior_gpu_seconds > 90_000:
        raise ValueError("resume source GPU accounting is outside the authored cap")

    payload: dict[str, object] = {
        "version": 1,
        "status": "verified_cross_commit_reuse",
        "reason": "post_selection_scheduler_boundary_fix",
        "source_commit": source_commit,
        "target_commit": target_commit,
        "reviewed_recovery_commits": reviewed_commits,
        "reviewed_diff_paths": sorted(_ALLOWED_DIFF_PATHS),
        "source_exit_stage": "cuda_resume_qualification",
        "source_exit_code": 5,
        "source_error": _EXACT_FAILURE,
        "source_protocol_lock_sha256": lock["lock_sha256"],
        "source_gpu_uuid": state["gpu_uuid"],
        "target_gpu_uuid": target_gpu_uuid,
        "source_environment_sha256": lock["environment_sha256"],
        "source_protocol_sha256": lock["protocol_sha256"],
        "source_data_manifest_sha256": source_data["manifest_sha256"],
        "source_final_config_file_sha256": lock["final_config_file_sha256"],
        "source_steps_per_credit": lock["selected_steps_per_credit"],
        "selection_file_sha256": selection_file_sha256,
        "prior_gpu_seconds": prior_gpu_seconds,
        "reused_paths": _REUSED_PATHS,
    }
    payload["provenance_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def verify_target(target: Path, *, target_commit: str) -> dict[str, Any]:
    """Fail closed unless the target contains only the two sealed selection aggregates."""

    provenance = _read(target / "selection-provenance.json")
    _verify_digest(provenance, "provenance_sha256", "selection provenance")
    selection_root = target / "selection"
    if selection_root.is_symlink() or not selection_root.is_dir():
        raise ValueError("target selection directory is missing or redirected")
    actual: set[str] = set()
    for path in selection_root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError("target selection evidence contains a redirected artifact")
        actual.add(path.relative_to(target).as_posix())
    if (
        actual != set(_REUSED_PATHS)
        or provenance.get("target_commit") != target_commit
        or provenance.get("reused_paths") != _REUSED_PATHS
    ):
        raise ValueError("target selection evidence is outside the aggregate-only reuse set")
    selection_file_sha256, _ = _selection_identity(target)
    if provenance.get("selection_file_sha256") != selection_file_sha256:
        raise ValueError("target selection evidence does not match its provenance")
    return provenance


def _install_selection(source: Path, target: Path) -> None:
    selection_target = target / "selection"
    if selection_target.exists() or selection_target.is_symlink():
        return
    temporary = Path(tempfile.mkdtemp(prefix=".selection-resume-", dir=target))
    try:
        for relative in ("manifest.json", "selection-results.json"):
            source_path = source / "selection" / relative
            if source_path.is_symlink() or not source_path.is_file():
                raise ValueError("source selection aggregate is missing or redirected")
            shutil.copyfile(source_path, temporary / relative, follow_symlinks=False)
        os.replace(temporary, selection_target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _prepare(
    source: Path,
    target: Path,
    source_commit: str,
    target_commit: str,
    target_gpu_uuid: str,
) -> dict[str, Any]:
    provenance = build_provenance(
        source,
        target,
        source_commit=source_commit,
        target_commit=target_commit,
        target_gpu_uuid=target_gpu_uuid,
    )
    _install_selection(source, target)
    output = target / "selection-provenance.json"
    if output.exists() or output.is_symlink():
        if output.is_symlink() or _read(output) != provenance:
            raise ValueError("target selection provenance conflicts with the verified source")
    else:
        temporary = target / f".selection-provenance.{os.getpid()}.tmp"
        temporary.write_bytes(_canonical(provenance) + b"\n")
        os.replace(temporary, output)
    return verify_target(target, target_commit=target_commit)


def main() -> int:
    if len(sys.argv) == 7 and sys.argv[1] == "prepare":
        result = _prepare(
            Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
        )
        print(result["prior_gpu_seconds"])
        return 0
    if len(sys.argv) == 4 and sys.argv[1] == "verify":
        verify_target(Path(sys.argv[2]), target_commit=sys.argv[3])
        print("Verified aggregate-only cross-commit selection evidence")
        return 0
    print(
        "usage: prepare-selection-resume.py prepare SOURCE_JOB TARGET_JOB SOURCE_COMMIT "
        "TARGET_COMMIT TARGET_GPU_UUID\n"
        "       prepare-selection-resume.py verify TARGET_JOB TARGET_COMMIT",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
