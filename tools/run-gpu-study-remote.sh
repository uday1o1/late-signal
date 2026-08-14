#!/usr/bin/env bash

set -Eeuo pipefail

readonly UV_VERSION="0.11.23"
readonly MAX_GPU_SECONDS=14400
readonly MAX_WALL_SECONDS=43200
readonly MAX_WORKING_KIB=$((25 * 1024 * 1024))
readonly MAX_RETAINED_KIB=$((2 * 1024 * 1024))
readonly MIN_FREE_KIB=$((5 * 1024 * 1024))
readonly WATCHDOG_INTERVAL_SECONDS=30
readonly GPU_ACCOUNTING_RESERVE_SECONDS=120

usage() {
  cat <<'EOF'
Usage: tools/run-gpu-study-remote.sh REPO_ROOT JOB_ROOT GPU_UUID EXPECTED_COMMIT LAUNCH_ID [PRIOR_GPU_SECONDS]

Internal commit-pinned driver for LateSignal's one-shot GPU study.
Use tools/gpu-study.sh from the submitting Mac instead of invoking this
driver directly.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

(( $# == 5 || $# == 6 )) || {
  usage >&2
  exit 2
}

readonly REPO_ROOT="$1"
readonly JOB_ROOT="$2"
readonly GPU_UUID="$3"
readonly EXPECTED_COMMIT="$4"
readonly LAUNCH_ID="$5"
readonly IMPORTED_GPU_SECONDS="${6:-0}"
readonly COMMIT_SHORT="${EXPECTED_COMMIT:0:12}"
readonly CACHE_ROOT="$JOB_ROOT/runtime-cache"
readonly DATA_MANIFEST="$REPO_ROOT/data/processed/manifests/preparation.json"
readonly FINAL_CONFIG="$REPO_ROOT/configs/experiments/final.yaml"
readonly FEATURE_CONFIG="$REPO_ROOT/configs/features.yaml"
readonly FEASIBILITY="$JOB_ROOT/feasibility.json"
readonly FEASIBILITY_MEASURED="$JOB_ROOT/.feasibility-$LAUNCH_ID.json"
readonly SELECTION_ROOT="$JOB_ROOT/selection"
readonly SELECTION_RESULTS="$SELECTION_ROOT/selection-results.json"
readonly PROTOCOL_LOCK="$JOB_ROOT/protocol-lock.json"
readonly QUALITY_GATE="$JOB_ROOT/quality-gate.json"
readonly FINAL_ROOT="$JOB_ROOT/final"
readonly LOG_PATH="$JOB_ROOT/job.log"
readonly UV_ENVIRONMENT="$HOME/.local/share/latesignal/uv-$UV_VERSION"
readonly UV_BIN="$UV_ENVIRONMENT/bin/uv"

[[ "$REPO_ROOT" =~ ^/[A-Za-z0-9._/-]+$ && "$REPO_ROOT" != *..* ]] || \
  die "repository path is unsafe"
[[ "$JOB_ROOT" == "$REPO_ROOT/runs/one-shot/$COMMIT_SHORT" ]] || \
  die "job root is not the exact commit-scoped directory"
[[ "$GPU_UUID" =~ ^GPU-[A-Za-z0-9-]+$ ]] || die "GPU UUID is malformed"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "expected commit is malformed"
[[ "$LAUNCH_ID" =~ ^[A-Za-z0-9-]+$ ]] || die "launch ID is malformed"
[[ "$IMPORTED_GPU_SECONDS" =~ ^[0-9]+$ ]] || die "prior GPU accounting is malformed"
[[ -f "$DATA_MANIFEST" && -f "$FINAL_CONFIG" && -f "$FEATURE_CONFIG" ]] || \
  die "required repository inputs are missing"
[[ ! -L "$JOB_ROOT" ]] || die "job root cannot be a symbolic link"

mkdir -p "$JOB_ROOT"
for protected_path in \
  "$LOG_PATH" \
  "$JOB_ROOT/state.json" \
  "$JOB_ROOT/heartbeat.json" \
  "$JOB_ROOT/started.json" \
  "$JOB_ROOT/exit.json" \
  "$JOB_ROOT/resource-failure.json" \
  "$JOB_ROOT/job.lock" \
  "$FEASIBILITY" \
  "$SELECTION_ROOT" \
  "$PROTOCOL_LOCK" \
  "$QUALITY_GATE" \
  "$FINAL_ROOT" \
  "$JOB_ROOT/collection-manifest.json"; do
  [[ ! -L "$protected_path" ]] || die "a protected job path is a symbolic link"
done
touch "$LOG_PATH"
exec >>"$LOG_PATH" 2>&1

cd "$REPO_ROOT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || die "remote commit changed"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "remote tree is dirty"

export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PATH="$UV_ENVIRONMENT/bin:$PATH"
export UV_CACHE_DIR="$JOB_ROOT/uv-cache"
export PYTHONUNBUFFERED=1

stage="bootstrap"
attempt=0
watchdog_pid=""
readonly START_EPOCH="$(date +%s)"
readonly MAIN_PID="$$"
previous_gpu_seconds="$IMPORTED_GPU_SECONDS"
if [[ -f "$JOB_ROOT/heartbeat.json" ]]; then
  parsed_gpu_seconds="$(
    sed -n 's/.*"gpu_active_seconds":\([0-9][0-9]*\).*/\1/p' \
      "$JOB_ROOT/heartbeat.json" | tail -n 1
  )"
  if [[ "$parsed_gpu_seconds" =~ ^[0-9]+$ ]]; then
    if (( parsed_gpu_seconds > previous_gpu_seconds )); then
      previous_gpu_seconds="$parsed_gpu_seconds"
    fi
  else
    die "existing heartbeat has invalid accumulated GPU time"
  fi
fi
if (( previous_gpu_seconds > 0 )); then
  previous_gpu_seconds="$((previous_gpu_seconds + GPU_ACCOUNTING_RESERVE_SECONDS))"
fi
readonly PREVIOUS_GPU_SECONDS="${previous_gpu_seconds:-0}"
main_pgid="$(ps -o pgid= -p "$MAIN_PID" | tr -d '[:space:]')"
[[ "$main_pgid" == "$MAIN_PID" ]] || die "remote driver must run in its own process group"
readonly MAIN_PGID="$main_pgid"

atomic_state() {
  local status="$1"
  local exit_code="${2:-null}"
  local temporary="$JOB_ROOT/.state.$MAIN_PID.tmp"
  printf '%s\n' \
    "{\"version\":1,\"status\":\"$status\",\"stage\":\"$stage\",\"attempt\":$attempt,\"commit\":\"$EXPECTED_COMMIT\",\"gpu_uuid\":\"$GPU_UUID\",\"launch_id\":\"$LAUNCH_ID\",\"updated_at\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"exit_code\":$exit_code}" \
    >"$temporary"
  mv -f -- "$temporary" "$JOB_ROOT/state.json"
}

resource_failure() {
  local reason="$1"
  local detail="${2:-}"
  local temporary="$JOB_ROOT/.resource-failure.$MAIN_PID.tmp"
  printf '%s\n' \
    "{\"version\":1,\"status\":\"failed\",\"reason\":\"$reason\",\"detail\":\"$detail\",\"commit\":\"$EXPECTED_COMMIT\",\"gpu_uuid\":\"$GPU_UUID\",\"launch_id\":\"$LAUNCH_ID\",\"updated_at\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
    >"$temporary"
  mv -f -- "$temporary" "$JOB_ROOT/resource-failure.json"
}

watchdog_stop() {
  local reason="$1"
  local detail="${2:-}"
  local failure_stage failure_attempt stopped_at temporary
  resource_failure "$reason" "$detail"
  failure_stage="$(
    sed -n 's/.*"stage":"\([a-z0-9_]*\)".*/\1/p' "$JOB_ROOT/state.json"
  )"
  failure_attempt="$(
    sed -n 's/.*"attempt":\([0-9][0-9]*\).*/\1/p' "$JOB_ROOT/state.json"
  )"
  [[ "$failure_stage" =~ ^[a-z0-9_]+$ ]] || failure_stage="watchdog"
  [[ "$failure_attempt" =~ ^[0-9]+$ ]] || failure_attempt=0
  stopped_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  temporary="$JOB_ROOT/.state.$MAIN_PID.watchdog.tmp"
  printf '%s\n' \
    "{\"version\":1,\"status\":\"failed\",\"stage\":\"$failure_stage\",\"attempt\":$failure_attempt,\"commit\":\"$EXPECTED_COMMIT\",\"gpu_uuid\":\"$GPU_UUID\",\"launch_id\":\"$LAUNCH_ID\",\"updated_at\":\"$stopped_at\",\"exit_code\":124}" \
    >"$temporary"
  mv -f -- "$temporary" "$JOB_ROOT/state.json"
  temporary="$JOB_ROOT/.exit.$MAIN_PID.watchdog.tmp"
  printf '%s\n' \
    "{\"version\":1,\"status\":\"failed\",\"stage\":\"$failure_stage\",\"reason\":\"$reason\",\"commit\":\"$EXPECTED_COMMIT\",\"gpu_uuid\":\"$GPU_UUID\",\"launch_id\":\"$LAUNCH_ID\",\"exit_code\":124,\"finished_at\":\"$stopped_at\"}" \
    >"$temporary"
  mv -f -- "$temporary" "$JOB_ROOT/exit.json"
  kill -KILL -- "-$MAIN_PGID"
  exit 124
}

working_kib() {
  local paths=()
  local candidate
  for candidate in "$REPO_ROOT/.venv" "$REPO_ROOT/data/processed" "$JOB_ROOT"; do
    if [[ -e "$candidate" ]]; then
      paths+=("$candidate")
    fi
  done
  (( ${#paths[@]} > 0 )) || {
    printf '0\n'
    return
  }
  timeout 60s du -sk -- "${paths[@]}" | awk '{total += $1} END {print total + 0}'
}

watchdog() {
  local watchdog_sleep_pid=""
  trap - EXIT INT TERM
  trap 'if [[ -n "$watchdog_sleep_pid" ]]; then kill "$watchdog_sleep_pid" 2>/dev/null || true; wait "$watchdog_sleep_pid" 2>/dev/null || true; fi; exit 0' TERM
  local gpu_active_seconds="$PREVIOUS_GPU_SECONDS"
  while kill -0 "$MAIN_PID" 2>/dev/null; do
    local now elapsed used free utilization compute_pids compute_processes
    local heartbeat_stage heartbeat_attempt foreign_pid foreign_pgid temporary
    now="$(date +%s)"
    elapsed="$((now - START_EPOCH))"
    gpu_active_seconds="$((PREVIOUS_GPU_SECONDS + elapsed))"
    if (( gpu_active_seconds + GPU_ACCOUNTING_RESERVE_SECONDS >= MAX_GPU_SECONDS )); then
      watchdog_stop \
        "gpu_budget_reserve_reached" \
        "elapsed=$elapsed gpu_active=$gpu_active_seconds reserve=$GPU_ACCOUNTING_RESERVE_SECONDS"
    fi
    heartbeat_stage="$(sed -n 's/.*"stage":"\([a-z0-9_]*\)".*/\1/p' "$JOB_ROOT/state.json")"
    heartbeat_attempt="$(sed -n 's/.*"attempt":\([0-9][0-9]*\).*/\1/p' "$JOB_ROOT/state.json")"
    if [[ ! "$heartbeat_stage" =~ ^[a-z0-9_]+$ || ! "$heartbeat_attempt" =~ ^[0-9]+$ ]]; then
      watchdog_stop "stage_state_invalid"
    fi
    if ! used="$(working_kib)" || [[ ! "$used" =~ ^[0-9]+$ ]]; then
      watchdog_stop "working_disk_measurement_failed"
    fi
    if ! free="$(df -Pk "$JOB_ROOT" | awk 'NR == 2 {print $4}')" || \
      [[ ! "$free" =~ ^[0-9]+$ ]]; then
      watchdog_stop "free_disk_measurement_failed"
    fi
    if ! utilization="$(
      timeout 10s nvidia-smi --id="$GPU_UUID" --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]'
    )" || [[ ! "$utilization" =~ ^[0-9]+$ ]]; then
      watchdog_stop "gpu_utilization_signal_lost"
    fi
    if ! compute_pids="$(
      timeout 10s nvidia-smi --id="$GPU_UUID" --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
    )"; then
      watchdog_stop "gpu_process_signal_lost"
    fi
    compute_processes="$(printf '%s\n' "$compute_pids" | sed '/^[[:space:]]*$/d' | wc -l | tr -d '[:space:]')"
    while IFS= read -r foreign_pid; do
      [[ -n "$foreign_pid" ]] || continue
      if [[ ! "$foreign_pid" =~ ^[0-9]+$ ]]; then
        watchdog_stop "gpu_process_identity_malformed"
      fi
      foreign_pgid="$(
        ps -o pgid= -p "$foreign_pid" 2>/dev/null | tr -d '[:space:]' || true
      )"
      if [[ -z "$foreign_pgid" ]]; then
        if ps -p "$foreign_pid" >/dev/null 2>&1; then
          watchdog_stop "gpu_process_identity_unavailable" "pid=$foreign_pid"
        fi
        continue
      fi
      if [[ "$foreign_pgid" != "$MAIN_PGID" ]]; then
        watchdog_stop "foreign_gpu_process_detected" "pid=$foreign_pid"
      fi
    done <<<"$compute_pids"
    now="$(date +%s)"
    elapsed="$((now - START_EPOCH))"
    gpu_active_seconds="$((PREVIOUS_GPU_SECONDS + elapsed))"
    temporary="$JOB_ROOT/.heartbeat.$MAIN_PID.tmp"
    printf '%s\n' \
      "{\"version\":1,\"stage\":\"$heartbeat_stage\",\"attempt\":$heartbeat_attempt,\"elapsed_seconds\":$elapsed,\"gpu_active_seconds\":$gpu_active_seconds,\"gpu_compute_processes\":$compute_processes,\"gpu_utilization_percent\":${utilization:-null},\"working_kib\":$used,\"free_kib\":$free,\"commit\":\"$EXPECTED_COMMIT\",\"gpu_uuid\":\"$GPU_UUID\",\"launch_id\":\"$LAUNCH_ID\",\"updated_at\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
      >"$temporary"
    mv -f -- "$temporary" "$JOB_ROOT/heartbeat.json"
    if ((
      elapsed > MAX_WALL_SECONDS
      || gpu_active_seconds > MAX_GPU_SECONDS
      || used > MAX_WORKING_KIB
      || free < MIN_FREE_KIB
    )); then
      watchdog_stop \
        "hard_resource_cap_exceeded" \
        "elapsed=$elapsed gpu_active=$gpu_active_seconds used_kib=$used free_kib=$free"
    fi
    sleep "$WATCHDOG_INTERVAL_SECONDS" &
    watchdog_sleep_pid="$!"
    wait "$watchdog_sleep_pid" || true
    watchdog_sleep_pid=""
  done
}

finish() {
  local exit_code="$?"
  trap - EXIT INT TERM
  if [[ -n "$watchdog_pid" ]]; then
    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true
  fi
  if (( exit_code == 0 )); then
    stage="complete"
    atomic_state "complete" 0
  else
    atomic_state "failed" "$exit_code"
  fi
  local temporary="$JOB_ROOT/.exit.$MAIN_PID.tmp"
  printf '%s\n' \
    "{\"version\":1,\"status\":\"$([[ $exit_code == 0 ]] && printf complete || printf failed)\",\"stage\":\"$stage\",\"commit\":\"$EXPECTED_COMMIT\",\"gpu_uuid\":\"$GPU_UUID\",\"launch_id\":\"$LAUNCH_ID\",\"exit_code\":$exit_code,\"finished_at\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
    >"$temporary"
  mv -f -- "$temporary" "$JOB_ROOT/exit.json"
  printf 'LateSignal remote job finished with exit code %s at stage %s\n' \
    "$exit_code" "$stage"
  exit "$exit_code"
}

trap finish EXIT
trap 'exit 124' INT TERM

readonly GPU_LOCK_ROOT="$HOME/.local/state/latesignal"
[[ ! -L "$GPU_LOCK_ROOT" ]] || die "GPU lock root cannot be a symbolic link"
mkdir -p "$GPU_LOCK_ROOT"
readonly GPU_LOCK_PATH="$GPU_LOCK_ROOT/gpu-$GPU_UUID.lock"
[[ ! -L "$GPU_LOCK_PATH" ]] || die "GPU lock file cannot be a symbolic link"
exec 8>"$GPU_LOCK_PATH"
flock -n 8 || die "another LateSignal process owns the selected GPU lock"
exec 9>"$JOB_ROOT/job.lock"
flock -n 9 || die "another process owns this exact one-shot job"

initial_compute_pids="$(
  nvidia-smi --id="$GPU_UUID" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
)"
[[ -z "$initial_compute_pids" ]] || die "selected GPU became occupied before launch"

started_temporary="$JOB_ROOT/.started.$MAIN_PID.tmp"
printf '%s\n' \
  "{\"version\":1,\"status\":\"started\",\"commit\":\"$EXPECTED_COMMIT\",\"gpu_uuid\":\"$GPU_UUID\",\"launch_id\":\"$LAUNCH_ID\",\"started_at\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
  >"$started_temporary"
mv -f -- "$started_temporary" "$JOB_ROOT/started.json"
atomic_state "running"
watchdog &
watchdog_pid="$!"

run_stage() {
  local selected_stage="$1"
  shift
  stage="$selected_stage"
  attempt=1
  if [[ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]] || \
    [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
    die "detached execution worktree changed before stage $stage"
  fi
  atomic_state "running"
  while true; do
    printf '\n[%s] stage=%s attempt=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$stage" "$attempt"
    set +e
    "$@"
    local exit_code="$?"
    set -e
    if (( exit_code == 0 )); then
      return 0
    fi
    if (( exit_code == 4 && attempt < 2 )); then
      attempt="$((attempt + 1))"
      atomic_state "retrying_infrastructure_failure"
      sleep 10
      continue
    fi
    printf 'stage %s failed with exit code %s\n' "$stage" "$exit_code" >&2
    return "$exit_code"
  done
}

bootstrap_environment() {
  local bootstrap_attempt
  for bootstrap_attempt in 1 2; do
    set +e
    if [[ ! -x "$UV_BIN" ]]; then
      python3 -m venv "$UV_ENVIRONMENT" && \
        "$UV_ENVIRONMENT/bin/pip" install "uv==$UV_VERSION"
    fi
    local install_status="$?"
    if (( install_status == 0 )); then
      "$UV_BIN" sync --frozen --all-groups
      install_status="$?"
    fi
    set -e
    if (( install_status == 0 )); then
      return 0
    fi
    (( bootstrap_attempt < 2 )) || return "$install_status"
    sleep 10
  done
}

verify_remote_inputs() {
  nvidia-smi --id="$GPU_UUID" \
    --query-gpu=uuid,name,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader
  "$UV_BIN" run python - <<'PY'
import os
from pathlib import Path

import torch

from latesignal.experiments.protocol_lock import _verify_prepared_data
from latesignal.features.cache import build_feature_cache
from latesignal.features.policy import load_feature_policy

uuid = os.environ["CUDA_VISIBLE_DEVICES"]
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("exactly one UUID-selected CUDA device is required")
verification = _verify_prepared_data(Path("data/processed/manifests/preparation.json"))
print(
    "Verified prepared data: "
    f"{verification['verified_files']} files, "
    f"{verification['verified_file_bytes']} bytes"
)
authored = load_feature_policy(Path("configs/features.yaml"))
for name in ("compact", "large"):
    cache = build_feature_cache(
        Path("data/processed/manifests/preparation.json"),
        authored_policy=authored,
        policy_name=name,
        storage_root=Path(os.environ["LATESIGNAL_CACHE_ROOT"]),
    )
    print(f"Verified truth-free {name} feature cache: {cache.rows} rows")
print(f"CUDA preflight passed for {uuid}")
PY
}

run_feasibility() {
  if [[ -e "$FEASIBILITY" ]]; then
    "$UV_BIN" run python - \
      "$FEASIBILITY" "$REPO_ROOT" "$DATA_MANIFEST" "$FINAL_CONFIG" "$GPU_UUID" <<'PY'
import json
import sys
from pathlib import Path

from latesignal.experiments.feasibility_context import verify_feasibility_context

value = verify_feasibility_context(
    Path(sys.argv[1]),
    repository=Path(sys.argv[2]),
    data_manifest_path=Path(sys.argv[3]),
    final_config_path=Path(sys.argv[4]),
    device_uuid=sys.argv[5],
)
eligible = [
    item["steps_per_credit"]
    for item in value.get("projections", [])
    if item.get("fits_caps") is True
    and all(item.get("cap_checks", {}).values())
]
if (
    value.get("feasibility_model_version") != 2
    or value.get("status") != "passed"
    or value.get("blockers") != []
    or value.get("matrix", {}).get("total_runs") != 89
    or not eligible
    or value.get("selected_steps_per_credit") != max(eligible)
):
    raise SystemExit("stored feasibility result is not a passing exact 89-run gate")
print(f"Reusing passing feasibility result with {max(eligible)} steps per credit")
PY
    return
  fi
  [[ ! -e "$FEASIBILITY_MEASURED" ]] || \
    die "launch-specific feasibility temporary already exists"
  "$UV_BIN" run latesignal protocol estimate \
    "$FINAL_CONFIG" --out "$FEASIBILITY_MEASURED" --json
  "$UV_BIN" run python - \
    "$FEASIBILITY_MEASURED" "$FEASIBILITY" "$REPO_ROOT" \
    "$DATA_MANIFEST" "$FINAL_CONFIG" "$GPU_UUID" <<'PY'
import sys
from pathlib import Path

from latesignal.experiments.feasibility_context import bind_feasibility_context

bind_feasibility_context(
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    repository=Path(sys.argv[3]),
    data_manifest_path=Path(sys.argv[4]),
    final_config_path=Path(sys.argv[5]),
    device_uuid=sys.argv[6],
)
PY
  rm -f -- "$FEASIBILITY_MEASURED"
}

selected_steps() {
  "$UV_BIN" run python - "$FEASIBILITY" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
steps = value.get("selected_steps_per_credit")
if value.get("status") != "passed" or steps not in (100, 250, 500):
    raise SystemExit("feasibility did not select an authorized training budget")
print(steps)
PY
}

run_selection() {
  if [[ -f "$JOB_ROOT/selection-provenance.json" ]]; then
    python3 "$REPO_ROOT/tools/prepare-selection-resume.py" \
      verify "$JOB_ROOT" "$EXPECTED_COMMIT"
    return
  fi
  "$UV_BIN" run latesignal selection run "$FINAL_CONFIG" \
    --data-manifest "$DATA_MANIFEST" \
    --feature-config "$FEATURE_CONFIG" \
    --cache-root "$CACHE_ROOT" \
    --out "$SELECTION_ROOT" \
    --steps-per-credit "$STEPS_PER_CREDIT" \
    --device-uuid "$GPU_UUID" \
    --json
}

lock_protocol() {
  if [[ -e "$PROTOCOL_LOCK" ]]; then
    "$UV_BIN" run python - "$PROTOCOL_LOCK" <<'PY'
import sys
from pathlib import Path

from latesignal.experiments.protocol_lock import verify_protocol_lock

lock = verify_protocol_lock(Path(sys.argv[1]))
if lock.get("locked_before_final_scoring") is not True:
    raise SystemExit("stored protocol lock is not pre-scoring")
print(f"Reusing protocol lock {lock['lock_sha256']}")
PY
    return
  fi
  "$UV_BIN" run latesignal protocol lock "$FINAL_CONFIG" \
    --selection "$SELECTION_RESULTS" \
    --feasibility "$FEASIBILITY" \
    --data-manifest "$DATA_MANIFEST" \
    --out "$PROTOCOL_LOCK" \
    --json
}

run_qualification() {
  "$UV_BIN" run latesignal final qualify "$FINAL_CONFIG" \
    --protocol-lock "$PROTOCOL_LOCK" \
    --data-manifest "$DATA_MANIFEST" \
    --feature-config "$FEATURE_CONFIG" \
    --cache-root "$CACHE_ROOT" \
    --out "$QUALITY_GATE" \
    --device-uuid "$GPU_UUID" \
    --json
}

run_final_matrix() {
  "$UV_BIN" run latesignal final run "$FINAL_CONFIG" \
    --protocol-lock "$PROTOCOL_LOCK" \
    --data-manifest "$DATA_MANIFEST" \
    --feature-config "$FEATURE_CONFIG" \
    --cache-root "$CACHE_ROOT" \
    --out "$FINAL_ROOT" \
    --device-uuid "$GPU_UUID" \
    --json
}

run_aggregate() {
  "$UV_BIN" run latesignal final aggregate "$FINAL_CONFIG" \
    --protocol-lock "$PROTOCOL_LOCK" \
    --data-manifest "$DATA_MANIFEST" \
    --feature-config "$FEATURE_CONFIG" \
    --cache-root "$CACHE_ROOT" \
    --out "$FINAL_ROOT" \
    --quality-gate "$QUALITY_GATE" \
    --device-uuid "$GPU_UUID" \
    --json
}

prepare_collection_manifest() {
  "$UV_BIN" run python - "$JOB_ROOT" <<'PY'
import sys
from pathlib import Path

from latesignal.experiments.collection import build_collection_manifest

manifest = build_collection_manifest(Path(sys.argv[1]))
print(
    "Sealed aggregate-only collection: "
    f"{manifest['file_count']} files, {manifest['total_bytes']} bytes"
)
PY
}

prune_rebuildable_cache() {
  local target
  for target in "$CACHE_ROOT" "$UV_CACHE_DIR"; do
    [[ "$target" == "$JOB_ROOT/runtime-cache" || "$target" == "$JOB_ROOT/uv-cache" ]] || \
      die "generated cache cleanup target is unsafe"
    [[ ! -L "$target" ]] || die "generated cache cleanup target is a symbolic link"
    if [[ -d "$target" ]]; then
      rm -rf -- "$target"
    fi
  done
  local retained
  retained="$(du -sk -- "$JOB_ROOT" | awk '{print $1}')"
  (( retained <= MAX_RETAINED_KIB )) || \
    die "verified retained evidence exceeds the authored 2 GB cap"
  printf 'Retained evidence: %s KiB\n' "$retained"
}

export LATESIGNAL_CACHE_ROOT="$CACHE_ROOT"
run_stage bootstrap bootstrap_environment
run_stage input_preflight verify_remote_inputs
run_stage full_software_preflight make check
run_stage feasibility run_feasibility
readonly STEPS_PER_CREDIT="$(selected_steps)"
run_stage selection_50 run_selection
run_stage protocol_freeze lock_protocol
run_stage cuda_resume_qualification run_qualification
run_stage final_39 run_final_matrix
run_stage aggregate run_aggregate
run_stage collection_manifest prepare_collection_manifest
run_stage retention prune_rebuildable_cache

exit 0
