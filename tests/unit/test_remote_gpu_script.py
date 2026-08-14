"""Contract tests for the bounded remote GPU feasibility helper."""

import os
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path("tools/run-gpu-feasibility.sh").resolve()
STUDY_SCRIPT = Path("tools/gpu-study.sh").resolve()
REMOTE_STARTER = Path("tools/start-gpu-study-remote.sh").resolve()
REMOTE_DRIVER = Path("tools/run-gpu-study-remote.sh").resolve()


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
    for path in (STUDY_SCRIPT, REMOTE_STARTER, REMOTE_DRIVER):
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

    assert [source.index(stage) for stage in stages] == sorted(
        source.index(stage) for stage in stages
    )
    assert "readonly MAX_GPU_SECONDS=14400" in source
    assert "readonly MAX_WORKING_KIB=$((25 * 1024 * 1024))" in source
    assert "readonly MAX_RETAINED_KIB=$((2 * 1024 * 1024))" in source
    assert "if (( exit_code == 4 && attempt < 2 ))" in source
    assert 'export CUDA_VISIBLE_DEVICES="$GPU_UUID"' in source
    assert 'GPU_LOCK_PATH="$GPU_LOCK_ROOT/gpu-$GPU_UUID.lock"' in source
    assert "foreign_gpu_process_detected" in source
    assert 'kill -KILL -- "-$MAIN_PGID"' in source
    assert 'gpu_active_seconds="$((PREVIOUS_GPU_SECONDS + elapsed))"' in source
    assert 'gpu_active_seconds="$((gpu_active_seconds + 30))"' not in source
    assert "readonly GPU_ACCOUNTING_RESERVE_SECONDS=120" in source
    assert '"$UV_BIN" sync --frozen --all-groups' in source
    assert '"$UV_BIN" run latesignal final qualify' in source
    assert '"$UV_BIN" run latesignal final run' in source
    assert source.index("final qualify") < source.index("final run")


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


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


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
  2) printf 'GPU_UUID=GPU-fake-stable\\n' ;;
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
