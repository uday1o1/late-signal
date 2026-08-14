"""Contract tests for the bounded remote GPU feasibility helper."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

SCRIPT = Path("tools/run-gpu-feasibility.sh").resolve()
STUDY_SCRIPT = Path("tools/gpu-study.sh").resolve()
REMOTE_STARTER = Path("tools/start-gpu-study-remote.sh").resolve()
REMOTE_DRIVER = Path("tools/run-gpu-study-remote.sh").resolve()
RESUME_HELPER = Path("tools/prepare-selection-resume.py").resolve()
FINAL_RESUME_HELPER = Path("tools/prepare-final-resume.py").resolve()
WORKING_SET_MEASURE = Path("tools/measure-gpu-study-working-set.sh").resolve()


def _canonical_fixture(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def test_remote_gpu_script_help_is_safe_and_bounded() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "SSH_HOST [GPU_INDEX]" in result.stdout
    assert "prepared dataset" in result.stdout
    assert "does not start" in result.stdout
    assert "source archive" in result.stdout


def test_remote_gpu_script_uses_existing_https_credentials_without_reading_them() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "credential.helper=store" in source
    assert "GIT_TERMINAL_PROMPT=0" in source
    assert "cat ~/.git-credentials" not in source
    assert "git@github.com:${origin_url#https://github.com/}" not in source


def test_remote_gpu_script_verifies_the_preparation_manifest() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'Path("data/processed/manifests/preparation.json")' in source
    assert '_verify_prepared_data(Path("data/processed"))' not in source


def test_remote_gpu_script_preserves_immutable_results_per_revision() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "gpu${GPU_INDEX}-${LOCAL_HEAD:0:12}.json" in source
    assert 'RESULT_RELATIVE="runs/feasibility/gpu${GPU_INDEX}.json"' not in source


def test_one_shot_gpu_scripts_are_valid_and_explain_detached_operation() -> None:
    for path in (STUDY_SCRIPT, REMOTE_STARTER, REMOTE_DRIVER, WORKING_SET_MEASURE):
        subprocess.run(["bash", "-n", str(path)], check=True)
        result = subprocess.run(
            ["bash", str(path), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "one-shot" in result.stdout
    help_text = subprocess.run(
        ["bash", str(STUDY_SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "detached" in help_text
    assert "Mac can disconnect" in help_text
    assert "bash tools/gpu-study.sh submit cuda-pm 1" in help_text
    assert "bash tools/gpu-study.sh resume SSH_HOST GPU_INDEX SOURCE_COMMIT" in help_text
    assert "bash tools/gpu-study.sh recover-final SSH_HOST GPU_INDEX EXECUTION_COMMIT" in help_text
    assert "from datetime import UTC" not in FINAL_RESUME_HELPER.read_text(encoding="utf-8")


def test_one_shot_remote_driver_orders_every_hard_gate_before_scoring() -> None:
    source = REMOTE_DRIVER.read_text(encoding="utf-8")
    stages = (
        "run_stage bootstrap",
        "run_stage input_preflight",
        "run_stage full_software_preflight",
        "run_stage feasibility",
        "run_stage selection_50",
        "run_stage protocol_freeze",
        "run_stage cuda_resume_qualification",
        "run_stage final_39",
        "run_stage aggregate",
        "run_stage collection_manifest",
        "run_stage retention",
    )

    full_execution = source[source.index("else\n  run_stage bootstrap") :]
    assert [full_execution.index(stage) for stage in stages] == sorted(
        full_execution.index(stage) for stage in stages
    )
    assert "readonly MAX_GPU_SECONDS=14400" in source
    assert "readonly MAX_WORKING_KIB=$((25 * 1024 * 1024))" in source
    assert "readonly MAX_RETAINED_KIB=$((2 * 1024 * 1024))" in source
    assert "if (( exit_code == 4 && attempt < 2 ))" in source
    assert 'export CUDA_VISIBLE_DEVICES="$GPU_UUID"' in source
    assert 'GPU_LOCK_PATH="$GPU_LOCK_ROOT/gpu-$GPU_UUID.lock"' in source
    assert "foreign_gpu_process_detected" in source
    assert 'resource_failure "prestart_gpu_signal_lost"' in source
    assert 'kill -KILL -- "-$MAIN_PGID"' in source
    assert 'gpu_active_seconds="$((PREVIOUS_GPU_SECONDS + elapsed))"' in source
    assert 'gpu_active_seconds="$((gpu_active_seconds + 30))"' not in source
    assert "readonly GPU_ACCOUNTING_RESERVE_SECONDS=120" in source
    assert '"$UV_BIN" sync --frozen --all-groups' in source
    assert '"$UV_BIN" run latesignal final qualify' in source
    assert '"$UV_BIN" run latesignal final run' in source
    assert source.index("final qualify") < source.index("final run")
    assert 'bash "$DRIVER_ROOT/tools/measure-gpu-study-working-set.sh"' in source
    assert "final-recovery" in source
    assert "run_stage recovery_verification" in source
    assert source.index("run_stage recovery_verification") < source.index(
        "run_stage final_39 run_final_matrix"
    )
    assert (
        source.index("run_stage recovery_collection_transition")
        < source.index("run_stage recovery_provenance")
        < source.index("run_stage collection_manifest")
    )
    assert '"$REPO_ROOT" "$JOB_ROOT" retained' in source
    assert 'resource_failure "retained_disk_measurement_failed"' in source
    selection_function = source[source.index("run_selection()") : source.index("lock_protocol()")]
    assert 'verify "$JOB_ROOT" "$EXPECTED_COMMIT"' in selection_function
    assert selection_function.index('verify "$JOB_ROOT"') < selection_function.index("return")
    assert selection_function.index("return") < selection_function.index("selection run")


def test_one_shot_launcher_never_transfers_or_collects_restricted_rows() -> None:
    source = STUDY_SCRIPT.read_text(encoding="utf-8")

    assert "data/processed/" in source
    assert "data/raw" not in source
    assert "source.tar" not in source
    assert "primary-probabilities.npy" not in source
    assert "checkpoints" not in source
    assert "$REMOTE_JOB/collection-manifest.json" in source
    assert '--files-from="$COLLECTION_TEMP/paths.txt"' in source
    assert 'mktemp -d "$REPO_ROOT/runs/collected/.$COMMIT_SHORT-$completed_launch.XXXXXX"' in source
    assert "local collection destination already exists" in source
    assert "local collection temporary already exists" not in source
    assert "credential.helper=store" in source
    assert "cat ~/.git-credentials" not in source
    assert 'worktree add --detach "$remote_root" "$expected_head"' in source
    resume_source = RESUME_HELPER.read_text(encoding="utf-8")
    assert '"status": "verified_cross_commit_reuse"' in resume_source
    assert '"selection/manifest.json"' in resume_source
    assert '"$LAUNCH_ID" "$prior_gpu_seconds"' in source
    assert "prepare-final-resume.py" in source
    assert "measure-gpu-study-working-set.sh" in source


def test_selection_resume_helper_seals_only_the_exact_safe_failure(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    base_path = repository / "src" / "latesignal" / "scheduling" / "base.py"
    base_path.parent.mkdir(parents=True)
    base_path.write_text(
        """class WindowedScheduler:
    def _window_at(self, simulator_time: int) -> CreditWindow:
        if simulator_time % self.day_seconds:
            raise ConsistencyError("Scheduler decisions must occur on daily boundaries")
""",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", "source"], check=True)
    source_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    base_path.write_text(
        """class WindowedScheduler:
    def _window_at(self, simulator_time: int) -> CreditWindow:
        first_boundary = self.windows[0].start_time
        if (simulator_time - first_boundary) % self.day_seconds:
            raise ConsistencyError("Scheduler decisions must occur on daily boundaries")
""",
        encoding="utf-8",
    )
    allowed_additions = {
        "results/published/synthetic-reproduction.json",
        "src/latesignal/experiments/collection.py",
        "src/latesignal/experiments/production_aggregate.py",
        "src/latesignal/experiments/protocol_lock.py",
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
    for relative in allowed_additions:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"target fixture: {relative}\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", "target"], check=True)
    with (repository / "tools" / "prepare-selection-resume.py").open(
        "a", encoding="utf-8"
    ) as helper:
        helper.write("verifier canonicalization fix\n")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", "verifier fix"], check=True)
    target_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source = tmp_path / "source"
    target = repository / "runs" / "one-shot" / target_commit[:12]
    selection_root = source / "selection"
    selection_root.mkdir(parents=True)
    target.mkdir(parents=True)

    selection = {"version": 1, "protocol_sha256": "3" * 64}
    selection_bytes = _canonical_fixture(selection)
    (selection_root / "selection-results.json").write_bytes(selection_bytes)
    selection_manifest: dict[str, object] = {
        "version": 1,
        "status": "complete",
        "candidate_counts": {"model": 36, "delayed": 8, "sampler": 6, "total": 50},
        "selection_results_sha256": hashlib.sha256(selection_bytes).hexdigest(),
    }
    selection_manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_fixture(selection_manifest)
    ).hexdigest()
    (selection_root / "manifest.json").write_bytes(_canonical_fixture(selection_manifest))
    protocol_lock: dict[str, object] = {
        "status": "locked",
        "publication_eligible": True,
        "git": {"commit": source_commit, "dirty": False},
        "protocol_sha256": "3" * 64,
        "data": {"manifest_sha256": "4" * 64},
        "final_config_file_sha256": "5" * 64,
        "environment_sha256": "6" * 64,
        "selected_steps_per_credit": 100,
        "selection_file_sha256": hashlib.sha256(selection_bytes).hexdigest(),
    }
    protocol_lock["lock_sha256"] = hashlib.sha256(_canonical_fixture(protocol_lock)).hexdigest()
    (source / "protocol-lock.json").write_bytes(_canonical_fixture(protocol_lock))
    launch_id = "source-launch"
    common = {
        "status": "failed",
        "stage": "cuda_resume_qualification",
        "commit": source_commit,
        "launch_id": launch_id,
        "gpu_uuid": "GPU-test",
        "exit_code": 5,
    }
    (source / "state.json").write_text(json.dumps(common) + "\n", encoding="utf-8")
    (source / "exit.json").write_text(
        json.dumps({**common, "finished_at": "2026-08-14T00:02:00Z"}) + "\n",
        encoding="utf-8",
    )
    (source / "heartbeat.json").write_text(
        json.dumps(
            {
                "commit": source_commit,
                "launch_id": launch_id,
                "gpu_uuid": "GPU-test",
                "gpu_active_seconds": 100,
                "updated_at": "2026-08-14T00:01:40Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "job.log").write_text(
        '{"message":"Scheduler decisions must occur on daily boundaries"}\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(RESUME_HELPER),
            "prepare",
            str(source),
            str(target),
            source_commit,
            target_commit,
            "GPU-test",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "120"
    provenance = json.loads((target / "selection-provenance.json").read_text(encoding="utf-8"))
    assert provenance["prior_gpu_seconds"] == 120
    assert provenance["source_steps_per_credit"] == 100
    assert provenance["reused_paths"] == [
        "selection/manifest.json",
        "selection/selection-results.json",
    ]
    assert {path.name for path in (target / "selection").iterdir()} == {
        "manifest.json",
        "selection-results.json",
    }

    (selection_root / "selection-results.json").write_text("{}\n", encoding="utf-8")
    changed_target = repository / "runs" / "one-shot" / "changed-target"
    changed_target.mkdir(parents=True)
    changed = subprocess.run(
        [
            sys.executable,
            str(RESUME_HELPER),
            "prepare",
            str(source),
            str(changed_target),
            source_commit,
            target_commit,
            "GPU-test",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert changed.returncode != 0
    assert "selection evidence changed" in changed.stderr


def test_final_resume_helper_preserves_same_commit_checkpoint_evidence(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    (repository / ".gitignore").write_text("runs/\n", encoding="utf-8")
    allowed = {
        "tests/unit/test_remote_gpu_script.py",
        "tools/gpu-study.sh",
        "tools/measure-gpu-study-working-set.sh",
        "tools/prepare-final-resume.py",
        "tools/run-gpu-study-remote.sh",
        "tools/start-gpu-study-remote.sh",
    }
    for relative in allowed:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"execution fixture: {relative}\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", "execution"], check=True)
    execution_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    execution_root = tmp_path / "execution"
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "worktree",
            "add",
            "--detach",
            str(execution_root),
            execution_commit,
        ],
        check=True,
        capture_output=True,
    )
    for relative in allowed:
        (repository / relative).write_text(f"driver fixture: {relative}\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", "driver"], check=True)
    (repository / "tools" / "prepare-final-resume.py").write_text(
        "driver fixture with compatibility correction\n", encoding="utf-8"
    )
    subprocess.run(
        ["git", "-C", str(repository), "add", "tools/prepare-final-resume.py"], check=True
    )
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", "correction"], check=True)
    driver_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    job = execution_root / "runs" / "one-shot" / execution_commit[:12]
    job.mkdir(parents=True)

    def write_hashed(relative: str, value: dict[str, object], digest: str) -> None:
        value[digest] = hashlib.sha256(_canonical_fixture(value)).hexdigest()
        path = job / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_canonical_fixture(value))

    protocol_sha256 = "3" * 64
    feasibility: dict[str, object] = {
        "status": "passed",
        "ok": True,
        "blockers": [],
        "selected_steps_per_credit": 100,
        "matrix": {"total_runs": 89},
        "protocol_sha256": protocol_sha256,
        "projections": [
            {
                "steps_per_credit": 100,
                "fits_caps": True,
                "cap_checks": {
                    "runs": True,
                    "compute_hours": True,
                    "working_disk": True,
                    "retained_disk": True,
                },
            }
        ],
    }
    (job / "feasibility.json").write_bytes(_canonical_fixture(feasibility))
    selection_results = {"protocol_sha256": protocol_sha256}
    selection_results_path = job / "selection" / "selection-results.json"
    selection_results_path.parent.mkdir(parents=True)
    selection_results_path.write_bytes(_canonical_fixture(selection_results))
    selection_file_sha256 = hashlib.sha256(selection_results_path.read_bytes()).hexdigest()
    write_hashed(
        "selection/manifest.json",
        {
            "status": "complete",
            "candidate_counts": {"model": 36, "delayed": 8, "sampler": 6, "total": 50},
            "selection_results_sha256": hashlib.sha256(
                _canonical_fixture(selection_results)
            ).hexdigest(),
        },
        "manifest_sha256",
    )
    selection_provenance: dict[str, object] = {
        "status": "verified_cross_commit_reuse",
        "target_commit": execution_commit,
        "source_gpu_uuid": "GPU-test",
        "target_gpu_uuid": "GPU-test",
        "source_environment_sha256": "4" * 64,
        "selection_file_sha256": selection_file_sha256,
    }
    selection_provenance["provenance_sha256"] = hashlib.sha256(
        _canonical_fixture(selection_provenance)
    ).hexdigest()
    (job / "selection-provenance.json").write_bytes(_canonical_fixture(selection_provenance))
    feasibility_sha256 = hashlib.sha256(_canonical_fixture(feasibility)).hexdigest()
    write_hashed(
        "protocol-lock.json",
        {
            "status": "locked",
            "publication_eligible": True,
            "git": {"commit": execution_commit, "dirty": False},
            "protocol_sha256": protocol_sha256,
            "environment_sha256": "4" * 64,
            "feasibility_sha256": feasibility_sha256,
            "final_matrix": feasibility["matrix"],
            "selection_file_sha256": selection_file_sha256,
            "selection_execution": {
                "mode": "verified_cross_commit_reuse",
                "provenance": selection_provenance,
            },
        },
        "lock_sha256",
    )
    lock = json.loads((job / "protocol-lock.json").read_text(encoding="utf-8"))
    write_hashed(
        "quality-gate.json",
        {
            "status": "passed",
            "protocol_lock_sha256": lock["lock_sha256"],
            "protocol_sha256": protocol_sha256,
            "git_commit": execution_commit,
        },
        "manifest_sha256",
    )
    write_hashed(
        "final/online-state.json",
        {
            "version": 1,
            "status": "running",
            "completed_runs": [],
            "completed_count": 0,
            "next_index": 0,
            "total_runs": 33,
        },
        "state_sha256",
    )
    checkpoint = job / "final" / "study-a" / "runs" / "study-a-test" / "checkpoints"
    checkpoint.mkdir(parents=True)
    (checkpoint / "latest.json").write_text("{}\n", encoding="utf-8")
    (job / "job.lock").touch()

    common = {
        "status": "failed",
        "stage": "final_39",
        "commit": execution_commit,
        "gpu_uuid": "GPU-test",
        "launch_id": "source-launch",
        "exit_code": 124,
    }
    (job / "state.json").write_bytes(_canonical_fixture(common))
    (job / "heartbeat.json").write_bytes(
        _canonical_fixture(
            {
                "stage": "final_39",
                "commit": execution_commit,
                "gpu_uuid": "GPU-test",
                "launch_id": "source-launch",
                "gpu_active_seconds": 100,
                "working_kib": 1_000,
                "free_kib": 6 * 1024 * 1024,
                "updated_at": "2026-08-14T00:00:00Z",
            }
        )
    )
    exit_receipt = {
        **common,
        "reason": "wrong_reason",
        "finished_at": "2026-08-14T00:00:30Z",
    }
    (job / "exit.json").write_bytes(_canonical_fixture(exit_receipt))
    failure = {
        "status": "failed",
        "reason": "working_disk_measurement_failed",
        "detail": "",
        "commit": execution_commit,
        "gpu_uuid": "GPU-test",
        "launch_id": "source-launch",
    }
    (job / "resource-failure.json").write_bytes(_canonical_fixture(failure))
    arguments = [
        sys.executable,
        str(FINAL_RESUME_HELPER),
        "prepare",
        str(repository),
        str(execution_root),
        str(job),
        execution_commit,
        driver_commit,
        "GPU-test",
    ]

    rejected = subprocess.run(arguments, check=False, capture_output=True, text=True)

    assert rejected.returncode != 0
    assert "exact recoverable final disk-measurement failure" in rejected.stderr
    exit_receipt["reason"] = "working_disk_measurement_failed"
    (job / "exit.json").write_bytes(_canonical_fixture(exit_receipt))

    prepared = subprocess.run(arguments, check=False, capture_output=True, text=True)

    assert prepared.returncode == 0, prepared.stderr
    first_launch = f"recovery-{driver_commit[:12]}-1"
    assert prepared.stdout.splitlines() == [
        f"RECOVERY_LAUNCH_ID={first_launch}",
        "PRIOR_GPU_SECONDS=130",
    ]
    provenance = json.loads((job / "final-recovery-provenance.json").read_text(encoding="utf-8"))
    assert provenance["execution_commit"] == execution_commit
    assert provenance["driver_commit"] == driver_commit
    assert provenance["source_failure"]["receipts"]["exit"] == exit_receipt
    first_attempt = provenance["attempts"][0]
    assert first_attempt["scientific_evidence_before_resume"]["completed_online_runs"] == 0
    assert first_attempt["scientific_evidence_before_resume"]["partial_checkpoint_files"] == [
        {
            "path": "final/study-a/runs/study-a-test/checkpoints/latest.json",
            "sha256": hashlib.sha256(b"{}\n").hexdigest(),
        }
    ]
    wrong_gpu = subprocess.run(
        [
            sys.executable,
            str(FINAL_RESUME_HELPER),
            "verify",
            str(job),
            execution_commit,
            driver_commit,
            "GPU-other",
            first_launch,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert wrong_gpu.returncode != 0
    idempotent = subprocess.run(arguments, check=False, capture_output=True, text=True)
    assert idempotent.stdout == prepared.stdout

    second_common = {**common, "launch_id": first_launch}
    (job / "state.json").write_bytes(_canonical_fixture(second_common))
    (job / "heartbeat.json").write_bytes(
        _canonical_fixture(
            {
                "stage": "final_39",
                "commit": execution_commit,
                "gpu_uuid": "GPU-test",
                "launch_id": first_launch,
                "gpu_active_seconds": 200,
                "working_kib": 1_100,
                "free_kib": 6 * 1024 * 1024,
                "updated_at": "2026-08-14T00:01:00Z",
            }
        )
    )
    (job / "exit.json").write_bytes(
        _canonical_fixture(
            {
                **second_common,
                "reason": "working_disk_measurement_failed",
                "finished_at": "2026-08-14T00:01:20Z",
            }
        )
    )
    (job / "resource-failure.json").write_bytes(
        _canonical_fixture(
            {
                **failure,
                "launch_id": first_launch,
            }
        )
    )
    undercounted = subprocess.run(arguments, check=False, capture_output=True, text=True)
    assert undercounted.returncode != 0
    second_heartbeat = json.loads((job / "heartbeat.json").read_text(encoding="utf-8"))
    second_heartbeat["gpu_active_seconds"] = 260
    (job / "heartbeat.json").write_bytes(_canonical_fixture(second_heartbeat))
    retried = subprocess.run(arguments, check=False, capture_output=True, text=True)
    second_launch = f"recovery-{driver_commit[:12]}-2"
    assert retried.returncode == 0, retried.stderr
    assert retried.stdout.splitlines() == [
        f"RECOVERY_LAUNCH_ID={second_launch}",
        "PRIOR_GPU_SECONDS=280",
    ]
    repeated_retry = subprocess.run(arguments, check=False, capture_output=True, text=True)
    assert repeated_retry.returncode == 0, repeated_retry.stderr
    assert repeated_retry.stdout == retried.stdout
    provenance = json.loads((job / "final-recovery-provenance.json").read_text(encoding="utf-8"))
    assert len(provenance["attempts"]) == 2
    assert provenance["attempts"][1]["input_failure"]["receipts"]["state"] == second_common
    aggregate = job / "final" / "aggregate"
    aggregate.mkdir()
    published = subprocess.run(
        [
            sys.executable,
            str(FINAL_RESUME_HELPER),
            "publish",
            str(job),
            execution_commit,
            driver_commit,
            "GPU-test",
            second_launch,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert published.returncode == 0, published.stderr
    assert (aggregate / "infrastructure-recovery.json").read_bytes() == (
        job / "final-recovery-provenance.json"
    ).read_bytes()

    published_path = aggregate / "infrastructure-recovery.json"
    collection_manifest: dict[str, object] = {
        "version": 1,
        "status": "verified_aggregate_only",
        "files": [
            {
                "path": "final/aggregate/infrastructure-recovery.json",
                "sha256": hashlib.sha256(published_path.read_bytes()).hexdigest(),
                "bytes": published_path.stat().st_size,
            }
        ],
        "file_count": 1,
        "total_bytes": published_path.stat().st_size,
    }
    collection_manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_fixture(collection_manifest)
    ).hexdigest()
    (job / "collection-manifest.json").write_bytes(_canonical_fixture(collection_manifest))

    third_common = {**common, "stage": "retention", "launch_id": second_launch}
    (job / "state.json").write_bytes(_canonical_fixture(third_common))
    (job / "heartbeat.json").write_bytes(
        _canonical_fixture(
            {
                "stage": "retention",
                "commit": execution_commit,
                "gpu_uuid": "GPU-test",
                "launch_id": second_launch,
                "gpu_active_seconds": 420,
                "working_kib": 1_100,
                "free_kib": 6 * 1024 * 1024,
                "updated_at": "2026-08-14T00:02:00Z",
            }
        )
    )
    (job / "exit.json").write_bytes(
        _canonical_fixture(
            {
                **third_common,
                "reason": "working_disk_measurement_failed",
                "finished_at": "2026-08-14T00:02:10Z",
            }
        )
    )
    (job / "resource-failure.json").write_bytes(
        _canonical_fixture({**failure, "launch_id": second_launch})
    )
    third = subprocess.run(arguments, check=False, capture_output=True, text=True)
    third_launch = f"recovery-{driver_commit[:12]}-3"
    assert third.returncode == 0, third.stderr
    assert third.stdout.splitlines() == [
        f"RECOVERY_LAUNCH_ID={third_launch}",
        "PRIOR_GPU_SECONDS=430",
    ]
    prepared_collection = subprocess.run(
        [
            sys.executable,
            str(FINAL_RESUME_HELPER),
            "prepare-collection",
            str(job),
            execution_commit,
            driver_commit,
            "GPU-test",
            third_launch,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert prepared_collection.returncode == 0, prepared_collection.stderr
    assert not (job / "collection-manifest.json").exists()
    repeated_transition = subprocess.run(arguments, check=False, capture_output=True, text=True)
    assert repeated_transition.returncode == 0, repeated_transition.stderr
    assert repeated_transition.stdout == third.stdout
    republished = subprocess.run(
        [
            sys.executable,
            str(FINAL_RESUME_HELPER),
            "publish",
            str(job),
            execution_commit,
            driver_commit,
            "GPU-test",
            third_launch,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert republished.returncode == 0, republished.stderr

    prestart_common = {**common, "stage": "bootstrap", "launch_id": third_launch}
    (job / "state.json").write_bytes(_canonical_fixture({**prestart_common, "exit_code": 1}))
    (job / "exit.json").write_bytes(_canonical_fixture({**prestart_common, "exit_code": 1}))
    (job / "resource-failure.json").write_bytes(
        _canonical_fixture(
            {
                **failure,
                "reason": "prestart_gpu_signal_lost",
                "launch_id": third_launch,
            }
        )
    )
    fourth = subprocess.run(arguments, check=False, capture_output=True, text=True)
    fourth_launch = f"recovery-{driver_commit[:12]}-4"
    assert fourth.returncode == 0, fourth.stderr
    assert fourth.stdout.splitlines() == [
        f"RECOVERY_LAUNCH_ID={fourth_launch}",
        "PRIOR_GPU_SECONDS=550",
    ]

    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    (job / "started.json").write_bytes(
        _canonical_fixture(
            {
                "status": "started",
                "commit": execution_commit,
                "gpu_uuid": "GPU-test",
                "launch_id": fourth_launch,
                "started_at": now,
            }
        )
    )
    fifth = subprocess.run(arguments, check=False, capture_output=True, text=True)
    fifth_launch = f"recovery-{driver_commit[:12]}-5"
    assert fifth.returncode == 0, fifth.stderr
    assert fifth.stdout.splitlines() == [
        f"RECOVERY_LAUNCH_ID={fifth_launch}",
        "PRIOR_GPU_SECONDS=670",
    ]

    (job / "started.json").write_bytes(
        _canonical_fixture(
            {
                "status": "started",
                "commit": execution_commit,
                "gpu_uuid": "GPU-test",
                "launch_id": fifth_launch,
                "started_at": now,
            }
        )
    )
    (job / "state.json").write_bytes(
        _canonical_fixture(
            {
                **common,
                "status": "running",
                "stage": "aggregate",
                "launch_id": fifth_launch,
                "exit_code": None,
            }
        )
    )
    (job / "heartbeat.json").write_bytes(
        _canonical_fixture(
            {
                "stage": "aggregate",
                "commit": execution_commit,
                "gpu_uuid": "GPU-test",
                "launch_id": fifth_launch,
                "gpu_active_seconds": 800,
                "working_kib": 1_100,
                "free_kib": 6 * 1024 * 1024,
                "updated_at": now,
            }
        )
    )
    sixth = subprocess.run(arguments, check=False, capture_output=True, text=True)
    sixth_launch = f"recovery-{driver_commit[:12]}-6"
    assert sixth.returncode == 0, sixth.stderr
    sixth_lines = sixth.stdout.splitlines()
    assert sixth_lines[0] == f"RECOVERY_LAUNCH_ID={sixth_launch}"
    sixth_prior = int(sixth_lines[1].removeprefix("PRIOR_GPU_SECONDS="))
    assert 800 <= sixth_prior <= 805

    (job / "started.json").write_bytes(
        _canonical_fixture(
            {
                "status": "started",
                "commit": execution_commit,
                "gpu_uuid": "GPU-test",
                "launch_id": sixth_launch,
                "started_at": now,
            }
        )
    )
    retained_common = {
        **common,
        "stage": "retention",
        "launch_id": sixth_launch,
        "exit_code": 4,
    }
    (job / "state.json").write_bytes(_canonical_fixture(retained_common))
    (job / "exit.json").write_bytes(_canonical_fixture({**retained_common, "finished_at": now}))
    (job / "resource-failure.json").write_bytes(
        _canonical_fixture(
            {
                **failure,
                "reason": "retained_disk_measurement_failed",
                "launch_id": sixth_launch,
            }
        )
    )
    (job / "heartbeat.json").write_bytes(
        _canonical_fixture(
            {
                "stage": "retention",
                "commit": execution_commit,
                "gpu_uuid": "GPU-test",
                "launch_id": sixth_launch,
                "gpu_active_seconds": 930,
                "working_kib": 1_100,
                "free_kib": 6 * 1024 * 1024,
                "updated_at": now,
            }
        )
    )
    seventh = subprocess.run(arguments, check=False, capture_output=True, text=True)
    seventh_launch = f"recovery-{driver_commit[:12]}-7"
    assert seventh.returncode == 0, seventh.stderr
    assert seventh.stdout.splitlines() == [
        f"RECOVERY_LAUNCH_ID={seventh_launch}",
        "PRIOR_GPU_SECONDS=930",
    ]


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_recover_final_reaches_old_worktree_through_remote_tmux(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "local" / "late-signal"
    repository.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "branch", "-M", "main"], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    (repository / ".gitignore").write_text("runs/\n.venv/\n", encoding="utf-8")
    (repository / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (repository / "uv.lock").write_text("fixture\n", encoding="utf-8")
    (repository / "configs" / "experiments").mkdir(parents=True)
    (repository / "configs" / "experiments" / "final.yaml").write_text("{}\n", encoding="utf-8")
    (repository / "configs" / "features.yaml").write_text("{}\n", encoding="utf-8")
    (repository / "data" / "processed" / "manifests").mkdir(parents=True)
    (repository / "data" / "processed" / "manifests" / "preparation.json").write_text(
        "{}\n", encoding="utf-8"
    )
    allowed = {
        "tests/unit/test_remote_gpu_script.py",
        "tools/gpu-study.sh",
        "tools/measure-gpu-study-working-set.sh",
        "tools/prepare-final-resume.py",
        "tools/run-gpu-study-remote.sh",
        "tools/start-gpu-study-remote.sh",
    }
    for relative in allowed:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"execution fixture: {relative}\n", encoding="utf-8")
    selection_resume = repository / "tools" / "prepare-selection-resume.py"
    selection_resume.write_text("raise SystemExit(0)\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", "execution"], check=True)
    execution_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_paths = {
        "tools/gpu-study.sh": STUDY_SCRIPT,
        "tools/measure-gpu-study-working-set.sh": WORKING_SET_MEASURE,
        "tools/prepare-final-resume.py": FINAL_RESUME_HELPER,
        "tools/run-gpu-study-remote.sh": REMOTE_DRIVER,
        "tools/start-gpu-study-remote.sh": REMOTE_STARTER,
    }
    for relative, source in source_paths.items():
        shutil.copyfile(source, repository / relative)
        (repository / relative).chmod(0o755)
    (repository / "tests" / "unit" / "test_remote_gpu_script.py").write_text(
        "driver fixture\n", encoding="utf-8"
    )
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", "driver"], check=True)
    driver_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    remote_home = tmp_path / "remote-home"
    remote_git = remote_home / "late-signal"
    remote_home.mkdir()
    subprocess.run(["git", "clone", "-q", str(repository), str(remote_git)], check=True)
    execution_root = remote_home / "late-signal-worktrees" / execution_commit[:12]
    execution_root.parent.mkdir()
    subprocess.run(
        [
            "git",
            "-C",
            str(remote_git),
            "worktree",
            "add",
            "--detach",
            str(execution_root),
            execution_commit,
        ],
        check=True,
        capture_output=True,
    )
    (execution_root / ".venv").mkdir()
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "remote",
            "add",
            "origin",
            "https://github.com/example/late-signal.git",
        ],
        check=True,
    )

    job = execution_root / "runs" / "one-shot" / execution_commit[:12]
    job.mkdir(parents=True)

    def write_hashed(relative: str, value: dict[str, object], digest: str) -> None:
        value[digest] = hashlib.sha256(_canonical_fixture(value)).hexdigest()
        path = job / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_canonical_fixture(value))

    protocol_sha256 = "3" * 64
    feasibility: dict[str, object] = {
        "status": "passed",
        "ok": True,
        "blockers": [],
        "selected_steps_per_credit": 100,
        "matrix": {"total_runs": 89},
        "protocol_sha256": protocol_sha256,
        "projections": [
            {
                "steps_per_credit": 100,
                "fits_caps": True,
                "cap_checks": {
                    "runs": True,
                    "compute_hours": True,
                    "working_disk": True,
                    "retained_disk": True,
                },
            }
        ],
    }
    (job / "feasibility.json").write_bytes(_canonical_fixture(feasibility))
    selection_results = {"protocol_sha256": protocol_sha256}
    selection_results_path = job / "selection" / "selection-results.json"
    selection_results_path.parent.mkdir()
    selection_results_path.write_bytes(_canonical_fixture(selection_results))
    selection_file_sha256 = hashlib.sha256(selection_results_path.read_bytes()).hexdigest()
    write_hashed(
        "selection/manifest.json",
        {
            "status": "complete",
            "candidate_counts": {"model": 36, "delayed": 8, "sampler": 6, "total": 50},
            "selection_results_sha256": selection_file_sha256,
        },
        "manifest_sha256",
    )
    selection_provenance: dict[str, object] = {
        "status": "verified_cross_commit_reuse",
        "target_commit": execution_commit,
        "source_gpu_uuid": "GPU-test",
        "target_gpu_uuid": "GPU-test",
        "source_environment_sha256": "4" * 64,
        "selection_file_sha256": selection_file_sha256,
    }
    selection_provenance["provenance_sha256"] = hashlib.sha256(
        _canonical_fixture(selection_provenance)
    ).hexdigest()
    (job / "selection-provenance.json").write_bytes(_canonical_fixture(selection_provenance))
    write_hashed(
        "protocol-lock.json",
        {
            "status": "locked",
            "publication_eligible": True,
            "git": {"commit": execution_commit, "dirty": False},
            "protocol_sha256": protocol_sha256,
            "environment_sha256": "4" * 64,
            "feasibility_sha256": hashlib.sha256(_canonical_fixture(feasibility)).hexdigest(),
            "final_matrix": feasibility["matrix"],
            "selection_file_sha256": selection_file_sha256,
            "selection_execution": {
                "mode": "verified_cross_commit_reuse",
                "provenance": selection_provenance,
            },
        },
        "lock_sha256",
    )
    lock = json.loads((job / "protocol-lock.json").read_text(encoding="utf-8"))
    write_hashed(
        "quality-gate.json",
        {
            "status": "passed",
            "protocol_lock_sha256": lock["lock_sha256"],
            "protocol_sha256": protocol_sha256,
            "git_commit": execution_commit,
        },
        "manifest_sha256",
    )
    write_hashed(
        "final/online-state.json",
        {
            "version": 1,
            "status": "running",
            "completed_runs": [],
            "completed_count": 0,
            "next_index": 0,
            "total_runs": 33,
        },
        "state_sha256",
    )
    (job / "final" / "aggregate").mkdir()
    (job / "job.lock").touch()
    source_common = {
        "status": "failed",
        "stage": "final_39",
        "commit": execution_commit,
        "gpu_uuid": "GPU-test",
        "launch_id": "source-launch",
        "exit_code": 124,
    }
    (job / "state.json").write_bytes(_canonical_fixture(source_common))
    (job / "heartbeat.json").write_bytes(
        _canonical_fixture(
            {
                "stage": "final_39",
                "commit": execution_commit,
                "gpu_uuid": "GPU-test",
                "launch_id": "source-launch",
                "gpu_active_seconds": 100,
                "working_kib": 1_000,
                "free_kib": 6 * 1024 * 1024,
                "updated_at": "2026-08-14T00:00:00Z",
            }
        )
    )
    (job / "exit.json").write_bytes(
        _canonical_fixture(
            {
                **source_common,
                "reason": "working_disk_measurement_failed",
                "finished_at": "2026-08-14T00:00:10Z",
            }
        )
    )
    resource_failure = {
        "status": "failed",
        "reason": "working_disk_measurement_failed",
        "detail": "",
        "commit": execution_commit,
        "gpu_uuid": "GPU-test",
        "launch_id": "source-launch",
    }
    (job / "resource-failure.json").write_bytes(_canonical_fixture(resource_failure))
    first_prepare = subprocess.run(
        [
            sys.executable,
            str(remote_git / "tools" / "prepare-final-resume.py"),
            "prepare",
            str(remote_git),
            str(execution_root),
            str(job),
            execution_commit,
            driver_commit,
            "GPU-test",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert first_prepare.returncode == 0, first_prepare.stderr
    first_launch = f"recovery-{driver_commit[:12]}-1"
    retry_common = {**source_common, "launch_id": first_launch}
    (job / "state.json").write_bytes(_canonical_fixture(retry_common))
    retry_heartbeat = {
        "stage": "final_39",
        "commit": execution_commit,
        "gpu_uuid": "GPU-test",
        "launch_id": first_launch,
        "gpu_active_seconds": 240,
        "working_kib": 1_000,
        "free_kib": 6 * 1024 * 1024,
        "updated_at": "2026-08-14T00:01:00Z",
    }
    (job / "heartbeat.json").write_text(
        json.dumps(retry_heartbeat, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (job / "exit.json").write_bytes(
        _canonical_fixture(
            {
                **retry_common,
                "reason": "working_disk_measurement_failed",
                "finished_at": "2026-08-14T00:01:10Z",
            }
        )
    )
    (job / "resource-failure.json").write_bytes(
        _canonical_fixture({**resource_failure, "launch_id": first_launch})
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "git",
        """#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "$*" == "ls-remote origin refs/heads/main" ]]; then
  printf '%s\\trefs/heads/main\\n' "$FAKE_DRIVER_COMMIT"
elif [[ "$*" == *"fetch --prune origin main"* ]]; then
  exit 0
else
  exec "$REAL_GIT" "$@"
fi
""",
    )
    _write_executable(
        fake_bin / "ssh",
        """#!/usr/bin/env bash
set -Eeuo pipefail
while [[ "${1:-}" == "-o" ]]; do shift 2; done
host="$1"
shift
[[ "$host" == "fake-host" ]] || exit 9
if [[ "$#" == 1 && "$1" == *'printf %s "$HOME"'* ]]; then
  printf '%s' "$FAKE_REMOTE_HOME"
  exit 0
fi
HOME="$FAKE_REMOTE_HOME" "$@"
""",
    )
    _write_executable(
        fake_bin / "tmux",
        """#!/usr/bin/env bash
set -Eeuo pipefail
alive() {
  [[ -f "$FAKE_TMUX_STATE" ]] || return 1
  pid="$(sed -n '2p' "$FAKE_TMUX_STATE")"
  kill -0 "$pid" 2>/dev/null
}
case "$1" in
  has-session) alive ;;
  list-sessions)
    if alive; then sed -n '1p' "$FAKE_TMUX_STATE"; fi
    ;;
  new-session)
    session="$4"
    command_line="$5"
    nohup bash -c "$command_line" >/dev/null 2>&1 &
    printf '%s\\n%s\\n' "$session" "$!" >"$FAKE_TMUX_STATE"
    ;;
  *) exit 9 ;;
esac
""",
    )
    _write_executable(
        fake_bin / "setsid",
        """#!/usr/bin/env python3
import os
import sys
arguments = [item for item in sys.argv[1:] if item not in {'--fork', '--wait'}]
os.setsid()
os.execvp(arguments[0], arguments)
""",
    )
    _write_executable(fake_bin / "flock", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "ps",
        """#!/usr/bin/env bash
if [[ "$*" == *"-o pgid="* ]]; then
  printf '%s\\n' "${*: -1}"
else
  exit 1
fi
""",
    )
    _write_executable(
        fake_bin / "nvidia-smi",
        """#!/usr/bin/env bash
case "$*" in
  *query-gpu=uuid*) printf 'GPU-test\\n' ;;
  *query-gpu=memory.total*) printf '98000\\n' ;;
  *query-gpu=utilization.gpu*) printf '0\\n' ;;
  *query-compute-apps=pid*) : ;;
  *) : ;;
esac
""",
    )
    _write_executable(
        fake_bin / "awk",
        """#!/usr/bin/env bash
if [[ "${*: -1}" == "/proc/meminfo" ]]; then
  printf '134217728\\n'
else
  exec /usr/bin/awk "$@"
fi
""",
    )
    _write_executable(
        fake_bin / "df",
        """#!/usr/bin/env bash
printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'
printf 'fixture 100000000 1 50000000 1%% /\\n'
""",
    )
    _write_executable(
        fake_bin / "du",
        """#!/usr/bin/env bash
printf '1000\\t%s\\n' "${*: -1}"
""",
    )
    _write_executable(fake_bin / "timeout", '#!/usr/bin/env bash\nshift\nexec "$@"\n')
    _write_executable(fake_bin / "rsync", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "python3",
        f'#!/usr/bin/env bash\nexec "{sys.executable}" "$@"\n',
    )
    tmux_state = tmp_path / "tmux-state"
    uv_bin = remote_home / ".local" / "share" / "latesignal" / "uv-0.11.23" / "bin" / "uv"
    uv_bin.parent.mkdir(parents=True)
    _write_executable(
        uv_bin,
        """#!/usr/bin/env python3
import hashlib
import json
import sys
from pathlib import Path

arguments = sys.argv[1:]
if arguments[:3] == ['run', 'python', '-']:
    sys.stdin.read()
    if len(arguments) == 4:
        job = Path(arguments[3])
        artifact = job / 'final' / 'aggregate' / 'infrastructure-recovery.json'
        entry = {
            'path': 'final/aggregate/infrastructure-recovery.json',
            'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
            'bytes': artifact.stat().st_size,
        }
        value = {
            'version': 1,
            'status': 'verified_aggregate_only',
            'files': [entry],
            'file_count': 1,
            'total_bytes': entry['bytes'],
        }
        canonical = lambda item: (json.dumps(item, indent=2, sort_keys=True) + '\\n').encode()
        value['manifest_sha256'] = hashlib.sha256(canonical(value)).hexdigest()
        (job / 'collection-manifest.json').write_bytes(canonical(value))
    raise SystemExit(0)
if arguments[:3] == ['run', 'latesignal', 'final']:
    raise SystemExit(0)
raise SystemExit(9)
""",
    )
    monkeypatch.setenv("FAKE_DRIVER_COMMIT", driver_commit)
    monkeypatch.setenv("FAKE_REMOTE_HOME", str(remote_home))
    monkeypatch.setenv("FAKE_TMUX_STATE", str(tmux_state))
    monkeypatch.setenv("REAL_GIT", shutil.which("git") or "/usr/bin/git")
    monkeypatch.setenv("PATH", f"{fake_bin}:{Path('/usr/bin')}:{Path('/bin')}")

    result = subprocess.run(
        [
            "bash",
            str(repository / "tools" / "gpu-study.sh"),
            "recover-final",
            "fake-host",
            "1",
            execution_commit,
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    second_launch = f"recovery-{driver_commit[:12]}-2"
    assert f"RECOVERY_LAUNCH_ID={second_launch}" in result.stdout
    assert "PRIOR_GPU_SECONDS=250" in result.stdout
    assert "Remote same-commit final recovery started successfully" in result.stdout
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        state = json.loads((job / "state.json").read_text(encoding="utf-8"))
        if state.get("status") == "complete":
            break
        time.sleep(0.05)
    assert state["status"] == "complete"
    provenance = json.loads((job / "final-recovery-provenance.json").read_text(encoding="utf-8"))
    assert len(provenance["attempts"]) == 2
    assert provenance["attempts"][1]["recovery_launch_id"] == second_launch
    assert (job / "final" / "aggregate" / "infrastructure-recovery.json").read_bytes() == (
        job / "final-recovery-provenance.json"
    ).read_bytes()
    collection = json.loads((job / "collection-manifest.json").read_text(encoding="utf-8"))
    assert collection["files"][0]["path"] == ("final/aggregate/infrastructure-recovery.json")


def test_working_set_measurement_retries_one_transient_race(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    job = repository / "runs" / "one-shot" / ("a" * 12)
    (repository / "data" / "processed").mkdir(parents=True)
    job.mkdir(parents=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    state = tmp_path / "du-state"
    _write_executable(fake_bin / "timeout", '#!/usr/bin/env bash\nshift\nexec "$@"\n')
    _write_executable(
        fake_bin / "du",
        """#!/usr/bin/env bash
set -Eeuo pipefail
if [[ ! -e "$FAKE_DU_STATE" ]]; then
  touch "$FAKE_DU_STATE"
  exit 1
fi
printf '7\t%s\n' "${@: -1}"
""",
    )
    environment = os.environ | {
        "FAKE_DU_STATE": str(state),
        "PATH": f"{fake_bin}:{Path('/usr/bin')}:{Path('/bin')}",
    }

    result = subprocess.run(
        ["bash", str(WORKING_SET_MEASURE), str(repository), str(job)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "7"
    assert state.exists()


def test_working_set_measurement_fails_after_bounded_persistent_error(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    job = repository / "runs" / "one-shot" / ("a" * 12)
    (repository / "data" / "processed").mkdir(parents=True)
    job.mkdir(parents=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "timeout", '#!/usr/bin/env bash\nshift\nexec "$@"\n')
    _write_executable(fake_bin / "du", "#!/usr/bin/env bash\nexit 1\n")
    environment = os.environ | {
        "PATH": f"{fake_bin}:{Path('/usr/bin')}:{Path('/bin')}",
    }

    started = time.monotonic()
    result = subprocess.run(
        ["bash", str(WORKING_SET_MEASURE), str(repository), str(job)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=10,
    )

    assert result.returncode != 0
    assert 1.5 <= time.monotonic() - started < 8


def test_retained_measurement_is_bounded_to_the_job_evidence(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    job = repository / "runs" / "one-shot" / ("a" * 12)
    (repository / ".venv").mkdir(parents=True)
    (repository / "data" / "processed").mkdir(parents=True)
    job.mkdir(parents=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    arguments_log = tmp_path / "du-arguments"
    _write_executable(fake_bin / "timeout", '#!/usr/bin/env bash\nshift\nexec "$@"\n')
    _write_executable(
        fake_bin / "du",
        """#!/usr/bin/env bash
printf '%s\n' "$*" >"$FAKE_DU_ARGUMENTS"
printf '9\t%s\n' "${*: -1}"
""",
    )
    environment = os.environ | {
        "FAKE_DU_ARGUMENTS": str(arguments_log),
        "PATH": f"{fake_bin}:{Path('/usr/bin')}:{Path('/bin')}",
    }

    result = subprocess.run(
        ["bash", str(WORKING_SET_MEASURE), str(repository), str(job), "retained"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "9"
    measured_arguments = arguments_log.read_text(encoding="utf-8")
    assert str(job) in measured_arguments
    assert str(repository / ".venv") not in measured_arguments
    assert str(repository / "data" / "processed") not in measured_arguments


def test_remote_starter_ignores_stale_receipt_and_detaches_exact_launch(
    tmp_path: Path,
) -> None:
    commit = "c" * 40
    repository = tmp_path / "late-signal"
    job = repository / "runs" / "one-shot" / commit[:12]
    tools = repository / "tools"
    tools.mkdir(parents=True)
    job.mkdir(parents=True)
    (job / "started.json").write_text('{"launch_id":"stale-launch"}\n', encoding="utf-8")
    _write_executable(
        tools / "run-gpu-study-remote.sh",
        """#!/usr/bin/env bash
set -Eeuo pipefail
job="$2"
launch_id="$5"
sleep 0.2
printf '{"launch_id":"%s"}\n' "$launch_id" >"$job/started.json"
sleep 1.5
printf '%s\n' "$launch_id" >"$job/detached-survivor.txt"
""",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    tmux_state = tmp_path / "tmux-state"
    tmux_log = tmp_path / "tmux-log"
    _write_executable(
        fake_bin / "tmux",
        """#!/usr/bin/env bash
set -Eeuo pipefail
printf '%s\n' "$*" >>"$FAKE_TMUX_LOG"
case "$1" in
  new-session)
    [[ ! -e "$FAKE_TMUX_STATE" ]] || exit 1
    command_line="${5#exec setsid --fork --wait }"
    nohup bash -c "$command_line" >/dev/null 2>&1 &
    printf '%s\n' "$!" >"$FAKE_TMUX_STATE"
    ;;
  has-session)
    [[ -f "$FAKE_TMUX_STATE" ]] || exit 1
    kill -0 "$(sed -n '1p' "$FAKE_TMUX_STATE")" 2>/dev/null
    ;;
  *) exit 9 ;;
esac
""",
    )
    environment = os.environ | {
        "FAKE_TMUX_LOG": str(tmux_log),
        "FAKE_TMUX_STATE": str(tmux_state),
        "PATH": f"{fake_bin}:{Path('/usr/bin')}:{Path('/bin')}",
    }
    arguments = [
        "bash",
        str(REMOTE_STARTER),
        str(repository),
        str(job),
        "GPU-fake-stable",
        commit,
        f"latesignal-{commit[:12]}",
        "fresh-launch",
    ]

    started_at = time.monotonic()
    result = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert time.monotonic() - started_at >= 0.8
    assert '"launch_id":"fresh-launch"' in (job / "started.json").read_text(encoding="utf-8")
    duplicate = subprocess.run(
        [*arguments[:-1], "duplicate-launch"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=10,
    )
    assert duplicate.returncode != 0
    deadline = time.monotonic() + 3
    while not (job / "detached-survivor.txt").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert (job / "detached-survivor.txt").read_text(encoding="utf-8").strip() == "fresh-launch"
    tmux_calls = tmux_log.read_text(encoding="utf-8")
    assert f"new-session -d -s latesignal-{commit[:12]}" in tmux_calls
    assert "exec setsid --fork --wait bash" in tmux_calls


def test_one_shot_submit_reaches_a_confirmed_detached_started_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    commit = "a" * 40
    counter = tmp_path / "ssh-counter"
    counter.write_text("0\n", encoding="utf-8")
    _write_executable(
        fake_bin / "git",
        f"""#!/usr/bin/env bash
set -Eeuo pipefail
case "$*" in
  "rev-parse HEAD") printf '%s\\n' "{commit}" ;;
  "status --porcelain --untracked-files=all") exit 0 ;;
  "ls-remote origin refs/heads/main") printf '%s\\trefs/heads/main\\n' "{commit}" ;;
  "remote get-url origin") printf '%s\\n' "https://github.com/example/late-signal.git" ;;
  *) printf 'unexpected fake git call: %s\\n' "$*" >&2; exit 9 ;;
esac
""",
    )
    _write_executable(
        fake_bin / "ssh",
        """#!/usr/bin/env bash
set -Eeuo pipefail
counter="$FAKE_SSH_COUNTER"
value="$(sed -n '1p' "$counter")"
value="$((value + 1))"
printf '%s\\n' "$value" >"$counter"
case "$value" in
  1) printf '/home/test' ;;
  2) printf 'GPU_UUID=GPU-fake-stable\\nPRIOR_GPU_SECONDS=0\\n' ;;
  3) exit 0 ;;
  *) printf 'unexpected fake ssh call %s\\n' "$value" >&2; exit 9 ;;
esac
""",
    )
    _write_executable(fake_bin / "rsync", "#!/usr/bin/env bash\nexit 0\n")
    monkeypatch.setenv("FAKE_SSH_COUNTER", str(counter))
    monkeypatch.setenv("PATH", f"{fake_bin}:{Path('/usr/bin')}:{Path('/bin')}")

    result = subprocess.run(
        ["bash", str(STUDY_SCRIPT), "submit", "fake-host", "1"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Remote one-shot study started successfully" in result.stdout
    assert "Mac may now disconnect" in result.stdout
    assert "bash tools/gpu-study.sh status fake-host" in result.stdout
    assert counter.read_text(encoding="utf-8").strip() == "3"


def test_remote_one_shot_driver_reaches_a_durable_terminal_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "b" * 40
    repository = tmp_path / "late-signal"
    job = repository / "runs" / "one-shot" / commit[:12]
    (repository / "data" / "processed" / "manifests").mkdir(parents=True)
    (repository / "configs" / "experiments").mkdir(parents=True)
    (repository / "data" / "processed" / "manifests" / "preparation.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (repository / "configs" / "experiments" / "final.yaml").write_text("{}\n", encoding="utf-8")
    (repository / "configs" / "features.yaml").write_text("{}\n", encoding="utf-8")
    (repository / "tools").mkdir()
    _write_executable(
        repository / "tools" / "measure-gpu-study-working-set.sh",
        WORKING_SET_MEASURE.read_text(encoding="utf-8"),
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "git",
        f"""#!/usr/bin/env bash
case "$*" in
  "rev-parse HEAD") printf '%s\\n' "{commit}" ;;
  "status --porcelain --untracked-files=all") exit 0 ;;
  *) exit 9 ;;
esac
""",
    )
    _write_executable(
        fake_bin / "ps",
        """#!/usr/bin/env bash
printf '%s\\n' "${@: -1}"
""",
    )
    _write_executable(fake_bin / "flock", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "make", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "timeout", '#!/usr/bin/env bash\nshift\nexec "$@"\n')
    _write_executable(
        fake_bin / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$*" == *"utilization.gpu"* ]]; then
  printf '0\\n'
fi
exit 0
""",
    )
    fake_home = tmp_path / "home"
    uv_bin = fake_home / ".local" / "share" / "latesignal" / "uv-0.11.23" / "bin" / "uv"
    uv_bin.parent.mkdir(parents=True)
    _write_executable(
        uv_bin,
        """#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "$*" == *"protocol estimate"* ]]; then
  while (( $# )); do
    if [[ "$1" == "--out" ]]; then
      shift
      printf '%s%s%s%s%s\\n' \\
        '{"feasibility_model_version":2,"status":"passed","blockers":[],' \\
        '"matrix":{"total_runs":89},"selected_steps_per_credit":500,' \\
        '"projections":[{"steps_per_credit":500,"fits_caps":true,' \\
        '"cap_checks":{"runs":true,"compute_hours":true,' \\
        '"working_disk":true,"retained_disk":true}}]}' \\
        >"$1"
      exit 0
    fi
    shift
  done
fi
if (( $# >= 5 )) && [[ "$4" == */.feasibility-* && "$5" == */feasibility.json ]]; then
  cp -- "$4" "$5"
  exit 0
fi
if [[ "$*" == *"feasibility.json"* ]]; then
  printf '500\\n'
fi
exit 0
""",
    )
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("PATH", f"{fake_bin}:{Path('/usr/bin')}:{Path('/bin')}")
    job.mkdir(parents=True)
    (job / "heartbeat.json").write_text(
        '{"gpu_active_seconds":30,"launch_id":"crashed-launch"}\n',
        encoding="utf-8",
    )
    (job / "exit.json").write_text(
        '{"status":"failed","launch_id":"crashed-launch"}\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(REMOTE_DRIVER),
            str(repository),
            str(job),
            "GPU-fake-stable",
            commit,
            "test-launch-1",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    state = (job / "state.json").read_text(encoding="utf-8")
    exit_receipt = (job / "exit.json").read_text(encoding="utf-8")
    log = (job / "job.log").read_text(encoding="utf-8")
    assert '"status":"complete"' in state
    assert '"status":"complete"' in exit_receipt
    assert '"launch_id":"test-launch-1"' in exit_receipt
    assert "stage=selection_50" in log
    assert "stage=cuda_resume_qualification" in log
    assert "stage=final_39" in log
    assert "stage=aggregate" in log
    assert "stage=collection_manifest" in log
    assert "crashed-launch" not in exit_receipt

    (job / "heartbeat.json").write_text('{"gpu_active_seconds":"invalid"}\n', encoding="utf-8")
    repeated = subprocess.run(
        [
            "bash",
            str(REMOTE_DRIVER),
            str(repository),
            str(job),
            "GPU-fake-stable",
            commit,
            "test-launch-2",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert repeated.returncode != 0
    assert "existing heartbeat has invalid accumulated GPU time" in (job / "job.log").read_text(
        encoding="utf-8"
    )


def _watchdog_fixture(
    tmp_path: Path,
    *,
    nvidia_source: str,
) -> tuple[Path, Path, dict[str, str]]:
    commit = "d" * 40
    repository = tmp_path / "late-signal"
    job = repository / "runs" / "one-shot" / commit[:12]
    (repository / "data" / "processed" / "manifests").mkdir(parents=True)
    (repository / "configs" / "experiments").mkdir(parents=True)
    (repository / "data" / "processed" / "manifests" / "preparation.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (repository / "configs" / "experiments" / "final.yaml").write_text("{}\n", encoding="utf-8")
    (repository / "configs" / "features.yaml").write_text("{}\n", encoding="utf-8")
    (repository / "tools").mkdir()
    _write_executable(
        repository / "tools" / "measure-gpu-study-working-set.sh",
        WORKING_SET_MEASURE.read_text(encoding="utf-8"),
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "git",
        f"""#!/usr/bin/env bash
case "$*" in
  "rev-parse HEAD") printf '%s\\n' "{commit}" ;;
  "status --porcelain --untracked-files=all") exit 0 ;;
  *) exit 9 ;;
esac
""",
    )
    _write_executable(fake_bin / "ps", "#!/usr/bin/env bash\nprintf '%s\\n' \"${@: -1}\"\n")
    _write_executable(fake_bin / "flock", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "make", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "timeout", '#!/usr/bin/env bash\nshift\nexec "$@"\n')
    _write_executable(fake_bin / "nvidia-smi", nvidia_source)
    fake_home = tmp_path / "home"
    uv_bin = fake_home / ".local" / "share" / "latesignal" / "uv-0.11.23" / "bin" / "uv"
    uv_bin.parent.mkdir(parents=True)
    _write_executable(
        uv_bin,
        """#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "$*" == *"sync --frozen"* ]]; then
  sleep 2
fi
exit 0
""",
    )
    environment = os.environ | {
        "HOME": str(fake_home),
        "PATH": f"{fake_bin}:{Path('/usr/bin')}:{Path('/bin')}",
    }
    return repository, job, environment


def _run_fast_watchdog(
    tmp_path: Path,
    repository: Path,
    job: Path,
    environment: dict[str, str],
    *,
    working_limit: int | None = None,
    gpu_limit: int | None = None,
    accounting_reserve: int | None = None,
) -> subprocess.CompletedProcess[str]:
    source = REMOTE_DRIVER.read_text(encoding="utf-8").replace(
        "readonly WATCHDOG_INTERVAL_SECONDS=30",
        "readonly WATCHDOG_INTERVAL_SECONDS=0.05",
    )
    source = source.replace('kill -KILL -- "-$MAIN_PGID"', 'kill -KILL "$MAIN_PID"')
    if working_limit is not None:
        source = source.replace(
            "readonly MAX_WORKING_KIB=$((25 * 1024 * 1024))",
            f"readonly MAX_WORKING_KIB={working_limit}",
        )
    if gpu_limit is not None:
        source = source.replace(
            "readonly MAX_GPU_SECONDS=14400", f"readonly MAX_GPU_SECONDS={gpu_limit}"
        )
    if accounting_reserve is not None:
        source = source.replace(
            "readonly GPU_ACCOUNTING_RESERVE_SECONDS=120",
            f"readonly GPU_ACCOUNTING_RESERVE_SECONDS={accounting_reserve}",
        )
    driver = tmp_path / "run-gpu-study-remote-fast.sh"
    driver.write_text(source, encoding="utf-8")
    driver.chmod(0o755)
    commit = "d" * 40
    return subprocess.run(
        [
            "bash",
            str(driver),
            str(repository),
            str(job),
            "GPU-fake-stable",
            commit,
            "watchdog-launch",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=10,
        start_new_session=True,
    )


def test_remote_watchdog_terminates_the_job_at_working_disk_cap(tmp_path: Path) -> None:
    repository, job, environment = _watchdog_fixture(
        tmp_path,
        nvidia_source="""#!/usr/bin/env bash
if [[ "$*" == *"utilization.gpu"* ]]; then
  printf '0\\n'
fi
exit 0
""",
    )

    result = _run_fast_watchdog(
        tmp_path,
        repository,
        job,
        environment,
        working_limit=0,
    )

    assert result.returncode != 0
    failure = (job / "resource-failure.json").read_text(encoding="utf-8")
    assert '"reason":"hard_resource_cap_exceeded"' in failure
    assert '"launch_id":"watchdog-launch"' in failure


def test_remote_watchdog_fails_closed_after_persistent_measurement_error(
    tmp_path: Path,
) -> None:
    repository, job, environment = _watchdog_fixture(
        tmp_path,
        nvidia_source="""#!/usr/bin/env bash
if [[ "$*" == *"utilization.gpu"* ]]; then
  printf '0\n'
fi
exit 0
""",
    )
    fake_bin = Path(environment["PATH"].split(":", maxsplit=1)[0])
    _write_executable(fake_bin / "du", "#!/usr/bin/env bash\nexit 1\n")

    result = _run_fast_watchdog(tmp_path, repository, job, environment)

    assert result.returncode != 0
    failure = (job / "resource-failure.json").read_text(encoding="utf-8")
    assert '"reason":"working_disk_measurement_failed"' in failure


def test_remote_watchdog_terminates_for_a_foreign_gpu_process(tmp_path: Path) -> None:
    gpu_state = tmp_path / "gpu-query-state"
    foreign = subprocess.Popen(
        ["sleep", "20"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    repository, job, environment = _watchdog_fixture(
        tmp_path,
        nvidia_source="""#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "$*" == *"utilization.gpu"* ]]; then
  printf '0\\n'
elif [[ "$*" == *"query-compute-apps=pid"* ]]; then
  if [[ -e "$FAKE_GPU_STATE" ]]; then
    printf '%s\\n' "$FAKE_FOREIGN_PID"
  else
    touch "$FAKE_GPU_STATE"
  fi
fi
exit 0
""",
    )
    environment["FAKE_GPU_STATE"] = str(gpu_state)
    environment["FAKE_FOREIGN_PID"] = str(foreign.pid)

    try:
        result = _run_fast_watchdog(tmp_path, repository, job, environment)
    finally:
        foreign.terminate()
        foreign.wait(timeout=5)

    assert result.returncode != 0
    failure = (job / "resource-failure.json").read_text(encoding="utf-8")
    assert '"reason":"foreign_gpu_process_detected"' in failure
    assert f'"detail":"pid={foreign.pid}"' in failure


def test_remote_watchdog_charges_measured_delay_to_gpu_budget(tmp_path: Path) -> None:
    repository, job, environment = _watchdog_fixture(
        tmp_path,
        nvidia_source="""#!/usr/bin/env bash
if [[ "$*" == *"utilization.gpu"* ]]; then
  printf '0\\n'
fi
exit 0
""",
    )
    fake_bin = Path(environment["PATH"].split(":", maxsplit=1)[0])
    date_counter = tmp_path / "date-counter"
    _write_executable(
        fake_bin / "date",
        """#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "${1:-}" == "+%s" ]]; then
  count="$(sed -n '1p' "$FAKE_DATE_COUNTER" 2>/dev/null || printf '0')"
  count="$((count + 1))"
  printf '%s\\n' "$count" >"$FAKE_DATE_COUNTER"
  if (( count <= 2 )); then
    printf '1000\\n'
  else
    printf '1005\\n'
  fi
  exit 0
fi
exec /bin/date "$@"
""",
    )
    environment["FAKE_DATE_COUNTER"] = str(date_counter)

    result = _run_fast_watchdog(
        tmp_path,
        repository,
        job,
        environment,
        gpu_limit=2,
        accounting_reserve=0,
    )

    assert result.returncode != 0
    failure = (job / "resource-failure.json").read_text(encoding="utf-8")
    assert '"reason":"hard_resource_cap_exceeded"' in failure
    assert "gpu_active=5" in failure
