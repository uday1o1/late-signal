#!/usr/bin/env bash

set -Eeuo pipefail

readonly MIN_DISK_GB=25
readonly MIN_MEMORY_GB=16
readonly MIN_VRAM_MIB=8192

usage() {
  cat <<'EOF'
Usage:
  bash tools/gpu-study.sh submit SSH_HOST [GPU_INDEX]
  bash tools/gpu-study.sh resume SSH_HOST GPU_INDEX SOURCE_COMMIT
  bash tools/gpu-study.sh recover-final SSH_HOST GPU_INDEX EXECUTION_COMMIT
  bash tools/gpu-study.sh status SSH_HOST [EXECUTION_COMMIT]
  bash tools/gpu-study.sh logs SSH_HOST [EXECUTION_COMMIT]
  bash tools/gpu-study.sh follow SSH_HOST [EXECUTION_COMMIT]
  bash tools/gpu-study.sh attach SSH_HOST [EXECUTION_COMMIT]
  bash tools/gpu-study.sh collect SSH_HOST [EXECUTION_COMMIT]

Submit the exact current origin/main revision as one detached, resumable tmux
one-shot job on a trusted SSH-accessible NVIDIA GPU host. After submit confirms the
started receipt, the Mac can disconnect, sleep, or shut down without stopping
the remote job.

Resume reuses only sealed selection evidence from a prior clean commit whose
job stopped at the pre-scoring CUDA qualification gate. It binds explicit
cross-commit provenance into the new protocol lock and carries prior GPU time.

Recover-final resumes one exact final-stage checkpoint after the reviewed
transient working-disk measurement failure. It keeps the scientific execution
commit unchanged and uses only an allowlisted, commit-pinned watchdog fix.

The same submit command resumes incomplete commit-scoped evidence. It refuses a
duplicate active job, a completed job, a dirty checkout, a revision that is not
on origin/main, an occupied GPU, or a host below the authored resource caps.

Example:
  bash tools/gpu-study.sh submit cuda-pm 1
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

(( $# >= 1 )) || {
  usage >&2
  exit 2
}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  usage
  exit 0
fi

readonly ACTION="$1"
shift
case "$ACTION" in
  submit)
    (( $# == 1 || $# == 2 )) || {
      usage >&2
      exit 2
    }
    ;;
  resume)
    (( $# == 3 )) || {
      usage >&2
      exit 2
    }
    ;;
  recover-final)
    (( $# == 3 )) || {
      usage >&2
      exit 2
    }
    ;;
  status|logs|follow|attach|collect)
    (( $# == 1 || $# == 2 )) || {
      usage >&2
      exit 2
    }
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

readonly SSH_HOST="$1"
readonly GPU_INDEX="$([[ "$ACTION" =~ ^(submit|resume|recover-final)$ ]] && printf '%s' "${2:-0}" || printf '0')"
readonly RESUME_COMMIT="$([[ "$ACTION" == "resume" ]] && printf '%s' "${3:--}" || printf '-')"
readonly REQUESTED_EXECUTION_COMMIT="$(
  if [[ "$ACTION" == "recover-final" ]]; then
    printf '%s' "$3"
  elif [[ "$ACTION" =~ ^(status|logs|follow|attach|collect)$ && $# == 2 ]]; then
    printf '%s' "$2"
  else
    printf '-'
  fi
)"
[[ "$SSH_HOST" =~ ^[A-Za-z0-9._@-]+$ ]] || die "SSH_HOST contains unsupported characters"
[[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || die "GPU_INDEX must be a non-negative integer"
if [[ "$ACTION" == "resume" ]]; then
  [[ "$RESUME_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "SOURCE_COMMIT must be a full Git hash"
else
  [[ "$RESUME_COMMIT" == "-" ]] || die "unexpected resume commit"
fi
if [[ "$REQUESTED_EXECUTION_COMMIT" != "-" ]]; then
  [[ "$REQUESTED_EXECUTION_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
    die "EXECUTION_COMMIT must be a full Git hash"
fi

for command_name in git rsync ssh; do
  require_command "$command_name"
done
if [[ "$ACTION" == "recover-final" ]]; then
  require_command python3
fi
if [[ "$ACTION" == "collect" ]]; then
  require_command uv
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd "$REPO_ROOT"
[[ -f pyproject.toml && -f uv.lock ]] || die "could not resolve the LateSignal repository root"

readonly LOCAL_HEAD="$(git rev-parse HEAD)"
readonly EXECUTION_COMMIT="$([[ "$REQUESTED_EXECUTION_COMMIT" == "-" ]] && printf '%s' "$LOCAL_HEAD" || printf '%s' "$REQUESTED_EXECUTION_COMMIT")"
readonly COMMIT_SHORT="${EXECUTION_COMMIT:0:12}"
readonly SESSION_NAME="latesignal-$COMMIT_SHORT"
readonly LAUNCH_ID="$([[ "$ACTION" == "recover-final" ]] && printf 'recovery-%s' "${LOCAL_HEAD:0:12}" || printf '%s-%s' "$(date -u +%Y%m%dT%H%M%SZ)" "$$")"
remote_home="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$SSH_HOST" 'printf %s "$HOME"')"
[[ "$remote_home" =~ ^/[A-Za-z0-9._/-]+$ && "$remote_home" != *..* ]] || \
  die "remote home path contains unsupported characters"
readonly REMOTE_GIT_ROOT="$remote_home/late-signal"
readonly REMOTE_ROOT="$remote_home/late-signal-worktrees/$COMMIT_SHORT"
readonly REMOTE_JOB="$REMOTE_ROOT/runs/one-shot/$COMMIT_SHORT"
readonly RESUME_SHORT="${RESUME_COMMIT:0:12}"
readonly RESUME_ROOT="$remote_home/late-signal-worktrees/$RESUME_SHORT"
readonly RESUME_JOB="$RESUME_ROOT/runs/one-shot/$RESUME_SHORT"

case "$ACTION" in
  status)
    ssh "$SSH_HOST" bash -s -- "$SESSION_NAME" "$REMOTE_JOB" <<'REMOTE_STATUS'
set -Eeuo pipefail
session="$1"
job="$2"
if tmux has-session -t "=$session" 2>/dev/null; then
  printf 'tmux: running (%s)\n' "$session"
else
  printf 'tmux: not running (%s)\n' "$session"
fi
state_launch="$(sed -n 's/.*"launch_id":"\([^"]*\)".*/\1/p' "$job/state.json" 2>/dev/null)"
gpu_uuid="$(sed -n 's/.*"gpu_uuid":"\([^"]*\)".*/\1/p' "$job/state.json" 2>/dev/null)"
for name in state.json heartbeat.json exit.json resource-failure.json; do
  [[ -f "$job/$name" ]] || continue
  item_launch="$(sed -n 's/.*"launch_id":"\([^"]*\)".*/\1/p' "$job/$name")"
  if [[ "$name" == "state.json" || "$item_launch" == "$state_launch" ]]; then
    printf '%s:\n' "$name"
    sed -n '1,8p' "$job/$name"
  fi
done
[[ "$gpu_uuid" =~ ^GPU-[A-Za-z0-9-]+$ ]] || {
  printf 'GPU identity is not available yet.\n'
  exit 0
}
nvidia-smi --id="$gpu_uuid" \
  --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader
REMOTE_STATUS
    exit 0
    ;;
  logs)
    ssh "$SSH_HOST" "tail -n 120 '$REMOTE_JOB/job.log'"
    exit 0
    ;;
  follow)
    ssh -t "$SSH_HOST" "tail -n 120 -f '$REMOTE_JOB/job.log'"
    exit 0
    ;;
  attach)
    ssh -t "$SSH_HOST" "tmux attach-session -t '=$SESSION_NAME'"
    exit 0
    ;;
  collect)
    completed_launch="$(ssh "$SSH_HOST" bash -s -- "$SESSION_NAME" "$REMOTE_JOB" <<'REMOTE_COMPLETE'
set -Eeuo pipefail
session="$1"
job="$2"
tmux has-session -t "=$session" 2>/dev/null && exit 1
state_status="$(sed -n 's/.*"status":"\([^"]*\)".*/\1/p' "$job/state.json")"
state_launch="$(sed -n 's/.*"launch_id":"\([^"]*\)".*/\1/p' "$job/state.json")"
exit_status="$(sed -n 's/.*"status":"\([^"]*\)".*/\1/p' "$job/exit.json")"
exit_launch="$(sed -n 's/.*"launch_id":"\([^"]*\)".*/\1/p' "$job/exit.json")"
[[ "$state_status" == "complete" && "$exit_status" == "complete" ]] || exit 1
[[ -n "$state_launch" && "$state_launch" == "$exit_launch" ]] || exit 1
[[ -f "$job/collection-manifest.json" && ! -L "$job/collection-manifest.json" ]] || exit 1
printf '%s\n' "$state_launch"
REMOTE_COMPLETE
)" || die "remote job is not a matching completed launch; use status or logs"
    [[ "$completed_launch" =~ ^[A-Za-z0-9-]+$ ]] || die "remote completion identity is invalid"
    readonly COLLECTION_ROOT="$REPO_ROOT/runs/collected/$COMMIT_SHORT"
    [[ ! -e "$COLLECTION_ROOT" ]] || die "local collection destination already exists"
    mkdir -p "$REPO_ROOT/runs/collected"
    collection_temp="$(
      mktemp -d "$REPO_ROOT/runs/collected/.$COMMIT_SHORT-$completed_launch.XXXXXX"
    )"
    readonly COLLECTION_TEMP="$collection_temp"
    mkdir -p "$COLLECTION_TEMP/evidence"
    rsync --archive "$SSH_HOST:$REMOTE_JOB/collection-manifest.json" \
      "$COLLECTION_TEMP/evidence/collection-manifest.json"
    UV_CACHE_DIR="${UV_CACHE_DIR:-/private/tmp/uv-cache}" uv run python - \
      "$COLLECTION_TEMP/evidence/collection-manifest.json" <<'PY' \
      >"$COLLECTION_TEMP/paths.txt"
import hashlib
import sys
from pathlib import Path, PurePosixPath

from latesignal.data.manifests import canonical_json_bytes, read_json

manifest = read_json(Path(sys.argv[1]))
expected = manifest.get("manifest_sha256")
unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
files = manifest.get("files")
if (
    manifest.get("status") != "verified_aggregate_only"
    or not isinstance(expected, str)
    or hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != expected
    or not isinstance(files, list)
):
    raise SystemExit("remote collection manifest is invalid")
required = {
    "feasibility.json",
    "selection/selection-results.json",
    "protocol-lock.json",
    "quality-gate.json",
    "final/final-manifest.json",
}
allowed_suffixes = {".json", ".csv", ".html", ".npz"}
forbidden = ("checkpoint", "prediction", "probabilities", "model-weight")
seen = set()
total_bytes = 0
for item in files:
    if (
        not isinstance(item, dict)
        or not isinstance(item.get("path"), str)
        or not isinstance(item.get("bytes"), int)
        or isinstance(item.get("bytes"), bool)
        or item["bytes"] < 0
    ):
        raise SystemExit("remote collection entry is invalid")
    path = PurePosixPath(item["path"])
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise SystemExit("remote collection path is unsafe")
    value = path.as_posix()
    if value not in required and (
        value != "selection-provenance.json"
        and (
            not value.startswith("final/aggregate/")
            or path.suffix not in allowed_suffixes
            or any(token in value.lower() for token in forbidden)
        )
    ):
        raise SystemExit("remote collection path is outside the aggregate-only allowlist")
    if value in seen:
        raise SystemExit("remote collection path is duplicated")
    seen.add(value)
    total_bytes += item["bytes"]
    print(value)
if not required.issubset(seen):
    raise SystemExit("remote collection is missing a required receipt")
if (
    manifest.get("file_count") != len(seen)
    or manifest.get("total_bytes") != total_bytes
    or total_bytes > 2 * 1024**3
):
    raise SystemExit("remote collection size or count is invalid")
PY
    rsync --archive --relative --files-from="$COLLECTION_TEMP/paths.txt" \
      "$SSH_HOST:$REMOTE_JOB/" "$COLLECTION_TEMP/evidence/"
    UV_CACHE_DIR="${UV_CACHE_DIR:-/private/tmp/uv-cache}" uv run python - \
      "$COLLECTION_TEMP/evidence" <<'PY'
import sys
from pathlib import Path

from latesignal.experiments.collection import verify_collection_manifest

root = Path(sys.argv[1])
manifest = verify_collection_manifest(root, root / "collection-manifest.json")
print(
    "Verified collected evidence: "
    f"{manifest['file_count']} files, {manifest['total_bytes']} bytes"
)
PY
    mv -- "$COLLECTION_TEMP/evidence" "$COLLECTION_ROOT"
    rm -f -- "$COLLECTION_TEMP/paths.txt"
    rmdir -- "$COLLECTION_TEMP"
    printf 'Collected aggregate-safe evidence at %s\n' "$COLLECTION_ROOT"
    exit 0
    ;;
esac

[[ -f data/processed/manifests/preparation.json ]] || \
  die "prepared data is missing; expected data/processed/manifests/preparation.json"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || \
  die "local repository has uncommitted or untracked source changes"
remote_main="$(git ls-remote origin refs/heads/main | awk '{print $1}')"
[[ "$LOCAL_HEAD" == "$remote_main" ]] || die "local HEAD is not the current origin/main revision"

origin_url="$(git remote get-url origin)"
case "$origin_url" in
  https://github.com/*)
    clone_url="$origin_url"
    ;;
  git@github.com:*)
    clone_url="https://github.com/${origin_url#git@github.com:}"
    ;;
  *)
    die "origin must be a GitHub HTTPS or SSH URL"
    ;;
esac

if [[ "$ACTION" == "recover-final" ]]; then
  python3 tools/prepare-final-resume.py \
    verify-driver "$REPO_ROOT" "$EXECUTION_COMMIT" "$LOCAL_HEAD" --quiet
  printf 'Checking same-commit final recovery on %s, GPU %s, execution %s, driver %s...\n' \
    "$SSH_HOST" "$GPU_INDEX" "$EXECUTION_COMMIT" "$LOCAL_HEAD"
  setup_output="$(ssh "$SSH_HOST" bash -s -- \
    "$REMOTE_GIT_ROOT" \
    "$REMOTE_ROOT" \
    "$REMOTE_JOB" \
    "$clone_url" \
    "$LOCAL_HEAD" \
    "$EXECUTION_COMMIT" \
    "$SESSION_NAME" \
    "$GPU_INDEX" \
    "$MIN_DISK_GB" \
    "$MIN_MEMORY_GB" \
    "$MIN_VRAM_MIB" <<'REMOTE_FINAL_RECOVERY'
set -Eeuo pipefail
remote_git_root="$1"
execution_root="$2"
job="$3"
clone_url="$4"
driver_commit="$5"
execution_commit="$6"
session="$7"
gpu_index="$8"
min_disk_gb="$9"
min_memory_gb="${10}"
min_vram_mib="${11}"

for command_name in git tmux flock setsid nvidia-smi python3 df awk sed timeout du; do
  command -v "$command_name" >/dev/null 2>&1 || {
    printf 'error: required remote command not found: %s\n' "$command_name" >&2
    exit 1
  }
done
tmux has-session -t "=$session" 2>/dev/null && {
  printf 'error: the exact final recovery is already active\n' >&2
  exit 1
}
active="$(tmux list-sessions -F '#{session_name}' 2>/dev/null | sed -n '/^latesignal-/p' || true)"
[[ -z "$active" ]] || {
  printf 'error: another LateSignal tmux job is active: %s\n' "$active" >&2
  exit 1
}

[[ -d "$remote_git_root/.git" ]] || {
  printf 'error: remote source repository is missing\n' >&2
  exit 1
}
[[ -z "$(git -C "$remote_git_root" status --porcelain --untracked-files=all)" ]] || {
  printf 'error: remote driver repository has source changes\n' >&2
  exit 1
}
git -C "$remote_git_root" remote set-url origin "$clone_url"
GIT_TERMINAL_PROMPT=0 git \
  -c credential.helper= \
  -c credential.helper=store \
  -C "$remote_git_root" fetch --prune origin main
git -C "$remote_git_root" checkout main
git -C "$remote_git_root" merge --ff-only origin/main
[[ "$(git -C "$remote_git_root" rev-parse HEAD)" == "$driver_commit" ]] || {
  printf 'error: remote recovery driver does not match confirmed origin/main\n' >&2
  exit 1
}
[[ -e "$execution_root/.git" && ! -L "$execution_root/.git" ]] || {
  printf 'error: scientific execution worktree is missing\n' >&2
  exit 1
}
[[ "$(git -C "$execution_root" rev-parse HEAD)" == "$execution_commit" ]] || {
  printf 'error: scientific execution commit changed\n' >&2
  exit 1
}
[[ -z "$(git -C "$execution_root" status --porcelain --untracked-files=all)" ]] || {
  printf 'error: scientific execution worktree has source changes\n' >&2
  exit 1
}
[[ "$job" == "$execution_root/runs/one-shot/${execution_commit:0:12}" && \
   -d "$job" && ! -L "$job" ]] || {
  printf 'error: exact failed final job root is missing or redirected\n' >&2
  exit 1
}

available_memory_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
(( available_memory_kib >= min_memory_gb * 1024 * 1024 )) || {
  printf 'error: remote host has less than %s GB available memory\n' "$min_memory_gb" >&2
  exit 1
}
gpu_uuid="$(timeout 10s nvidia-smi --id="$gpu_index" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')"
vram_mib="$(timeout 10s nvidia-smi --id="$gpu_index" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d '[:space:]')"
[[ "$gpu_uuid" =~ ^GPU-[A-Za-z0-9-]+$ && "$vram_mib" =~ ^[0-9]+$ ]] || {
  printf 'error: selected GPU is unavailable\n' >&2
  exit 1
}
(( vram_mib >= min_vram_mib )) || {
  printf 'error: selected GPU has less than %s MiB VRAM\n' "$min_vram_mib" >&2
  exit 1
}
compute_pids="$(timeout 10s nvidia-smi --id="$gpu_index" --query-compute-apps=pid \
  --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
[[ -z "$compute_pids" ]] || {
  printf 'error: selected GPU already has active compute processes\n' >&2
  exit 1
}

measured_working_kib="$(
  bash "$remote_git_root/tools/measure-gpu-study-working-set.sh" "$execution_root" "$job"
)" || {
  printf 'error: patched working-set measurement did not pass on the live job tree\n' >&2
  exit 1
}
[[ "$measured_working_kib" =~ ^[0-9]+$ ]] || {
  printf 'error: patched working-set measurement is malformed\n' >&2
  exit 1
}
(( measured_working_kib <= 25 * 1024 * 1024 )) || {
  printf 'error: live job tree exceeds the 25 GB working cap\n' >&2
  exit 1
}
free_kib="$(df -Pk "$job" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ ]] || {
  printf 'error: live free-disk measurement is malformed\n' >&2
  exit 1
}
(( free_kib >= min_disk_gb * 1024 * 1024 )) || {
  printf 'error: remote host has less than %s GB free disk\n' "$min_disk_gb" >&2
  exit 1
}

recovery_output="$(
  python3 "$remote_git_root/tools/prepare-final-resume.py" prepare \
    "$remote_git_root" "$execution_root" "$job" \
    "$execution_commit" "$driver_commit" "$gpu_uuid"
)"
recovery_launch_id="$(
  printf '%s\n' "$recovery_output" | sed -n 's/^RECOVERY_LAUNCH_ID=//p' | tail -n 1
)"
prior_gpu_seconds="$(
  printf '%s\n' "$recovery_output" | sed -n 's/^PRIOR_GPU_SECONDS=//p' | tail -n 1
)"
[[ "$recovery_launch_id" =~ ^[A-Za-z0-9-]+$ ]] || {
  printf 'error: final recovery launch identity is malformed\n' >&2
  exit 1
}
[[ "$prior_gpu_seconds" =~ ^[0-9]+$ ]] || {
  printf 'error: final recovery GPU accounting is malformed\n' >&2
  exit 1
}
printf 'GPU_UUID=%s\n' "$gpu_uuid"
printf 'RECOVERY_LAUNCH_ID=%s\n' "$recovery_launch_id"
printf 'PRIOR_GPU_SECONDS=%s\n' "$prior_gpu_seconds"
printf 'WORKING_KIB=%s\n' "$measured_working_kib"
printf 'FREE_KIB=%s\n' "$free_kib"
REMOTE_FINAL_RECOVERY
)"
  printf '%s\n' "$setup_output"
  gpu_uuid="$(printf '%s\n' "$setup_output" | sed -n 's/^GPU_UUID=//p' | tail -n 1)"
  recovery_launch_id="$(
    printf '%s\n' "$setup_output" | sed -n 's/^RECOVERY_LAUNCH_ID=//p' | tail -n 1
  )"
  prior_gpu_seconds="$(
    printf '%s\n' "$setup_output" | sed -n 's/^PRIOR_GPU_SECONDS=//p' | tail -n 1
  )"
  [[ "$gpu_uuid" =~ ^GPU-[A-Za-z0-9-]+$ ]] || \
    die "remote final recovery did not return a stable GPU UUID"
  [[ "$recovery_launch_id" =~ ^[A-Za-z0-9-]+$ ]] || \
    die "remote final recovery did not return a launch identity"
  [[ "$prior_gpu_seconds" =~ ^[0-9]+$ ]] || \
    die "remote final recovery did not return GPU accounting"
  printf 'Submitting detached tmux session %s...\n' "$SESSION_NAME"
  ssh "$SSH_HOST" bash "$REMOTE_GIT_ROOT/tools/start-gpu-study-remote.sh" \
    "$REMOTE_ROOT" "$REMOTE_JOB" "$gpu_uuid" "$EXECUTION_COMMIT" "$SESSION_NAME" \
    "$recovery_launch_id" "$prior_gpu_seconds" final-recovery "$LOCAL_HEAD"
  printf '\nRemote same-commit final recovery started successfully.\n'
  printf 'The Mac may now disconnect, sleep, or shut down.\n'
  printf 'Status:  bash tools/gpu-study.sh status %s %s\n' "$SSH_HOST" "$EXECUTION_COMMIT"
  printf 'Logs:    bash tools/gpu-study.sh logs %s %s\n' "$SSH_HOST" "$EXECUTION_COMMIT"
  printf 'Collect: bash tools/gpu-study.sh collect %s %s\n' "$SSH_HOST" "$EXECUTION_COMMIT"
  exit 0
fi

printf 'Checking %s, GPU %s, and exact commit %s...\n' \
  "$SSH_HOST" "$GPU_INDEX" "$LOCAL_HEAD"
setup_output="$(ssh "$SSH_HOST" bash -s -- \
  "$REMOTE_GIT_ROOT" \
  "$REMOTE_ROOT" \
  "$REMOTE_JOB" \
  "$clone_url" \
  "$LOCAL_HEAD" \
  "$SESSION_NAME" \
  "$GPU_INDEX" \
  "$MIN_DISK_GB" \
  "$MIN_MEMORY_GB" \
  "$MIN_VRAM_MIB" \
  "$RESUME_COMMIT" \
  "$RESUME_JOB" <<'REMOTE_SETUP'
set -Eeuo pipefail
remote_git_root="$1"
remote_root="$2"
remote_job="$3"
clone_url="$4"
expected_head="$5"
session="$6"
gpu_index="$7"
min_disk_gb="$8"
min_memory_gb="$9"
min_vram_mib="${10}"
resume_commit="${11}"
resume_job="${12}"

for command_name in git tmux flock setsid nvidia-smi python3 df awk sed timeout; do
  command -v "$command_name" >/dev/null 2>&1 || {
    printf 'error: required remote command not found: %s\n' "$command_name" >&2
    exit 1
  }
done

if tmux has-session -t "=$session" 2>/dev/null; then
  printf 'error: the exact one-shot job is already active\n' >&2
  exit 1
fi
if [[ -f "$remote_job/exit.json" ]] && \
  grep -q '"status":"complete"' "$remote_job/exit.json"; then
  printf 'error: this exact commit already completed; use collect\n' >&2
  exit 1
fi
active="$(tmux list-sessions -F '#{session_name}' 2>/dev/null | sed -n '/^latesignal-/p' || true)"
[[ -z "$active" ]] || {
  printf 'error: another LateSignal tmux job is active: %s\n' "$active" >&2
  exit 1
}

if [[ ! -d "$remote_git_root/.git" ]]; then
  GIT_TERMINAL_PROMPT=0 git \
    -c credential.helper= \
    -c credential.helper=store \
    clone --branch main --single-branch "$clone_url" "$remote_git_root"
else
  [[ -z "$(git -C "$remote_git_root" status --porcelain --untracked-files=all)" ]] || {
    printf 'error: remote source repository has source changes\n' >&2
    exit 1
  }
  git -C "$remote_git_root" remote set-url origin "$clone_url"
  GIT_TERMINAL_PROMPT=0 git \
    -c credential.helper= \
    -c credential.helper=store \
    -C "$remote_git_root" fetch --prune origin main
  git -C "$remote_git_root" checkout main
  git -C "$remote_git_root" merge --ff-only origin/main
fi

[[ "$(git -C "$remote_git_root" rev-parse HEAD)" == "$expected_head" ]] || {
  printf 'error: remote source revision does not match confirmed origin/main\n' >&2
  exit 1
}

git -C "$remote_git_root" worktree prune
if [[ ! -e "$remote_root/.git" ]]; then
  mkdir -p "$(dirname -- "$remote_root")"
  git -C "$remote_git_root" worktree add --detach "$remote_root" "$expected_head"
fi
[[ "$(git -C "$remote_root" rev-parse HEAD)" == "$expected_head" ]] || {
  printf 'error: detached execution worktree has the wrong revision\n' >&2
  exit 1
}
[[ -z "$(git -C "$remote_root" status --porcelain --untracked-files=all)" ]] || {
  printf 'error: detached execution worktree has source changes\n' >&2
  exit 1
}

available_disk_kib="$(df -Pk "$remote_root" | awk 'NR == 2 {print $4}')"
(( available_disk_kib >= min_disk_gb * 1024 * 1024 )) || {
  printf 'error: remote host has less than %s GB free disk\n' "$min_disk_gb" >&2
  exit 1
}
available_memory_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
(( available_memory_kib >= min_memory_gb * 1024 * 1024 )) || {
  printf 'error: remote host has less than %s GB available memory\n' "$min_memory_gb" >&2
  exit 1
}

gpu_uuid="$(timeout 10s nvidia-smi --id="$gpu_index" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')"
vram_mib="$(timeout 10s nvidia-smi --id="$gpu_index" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d '[:space:]')"
[[ "$gpu_uuid" =~ ^GPU-[A-Za-z0-9-]+$ && "$vram_mib" =~ ^[0-9]+$ ]] || {
  printf 'error: selected GPU is unavailable\n' >&2
  exit 1
}
(( vram_mib >= min_vram_mib )) || {
  printf 'error: selected GPU has less than %s MiB VRAM\n' "$min_vram_mib" >&2
  exit 1
}
compute_pids="$(timeout 10s nvidia-smi --id="$gpu_index" --query-compute-apps=pid \
  --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
[[ -z "$compute_pids" ]] || {
  printf 'error: selected GPU already has active compute processes\n' >&2
  exit 1
}

[[ ! -L "$remote_job" ]] || {
  printf 'error: remote job root is a symbolic link\n' >&2
  exit 1
}
mkdir -p "$remote_root/data/processed" "$remote_job"
prior_gpu_seconds=0
if [[ "$resume_commit" != "-" ]]; then
  [[ "$resume_commit" =~ ^[0-9a-f]{40}$ ]] || {
    printf 'error: resume source commit is malformed\n' >&2
    exit 1
  }
  [[ "$resume_job" =~ ^/[A-Za-z0-9._/-]+$ && "$resume_job" != *..* ]] || {
    printf 'error: resume source job path is unsafe\n' >&2
    exit 1
  }
  [[ "$resume_commit" != "$expected_head" ]] || {
    printf 'error: cross-commit resume source equals the target commit\n' >&2
    exit 1
  }
  for source_path in \
    "$resume_job" \
    "$resume_job/state.json" \
    "$resume_job/exit.json" \
    "$resume_job/heartbeat.json" \
    "$resume_job/job.log" \
    "$resume_job/selection" \
    "$resume_job/selection/manifest.json" \
    "$resume_job/selection/selection-results.json" \
    "$resume_job/protocol-lock.json"; do
    [[ -e "$source_path" && ! -L "$source_path" ]] || {
      printf 'error: resume source evidence is missing or redirected: %s\n' "$source_path" >&2
      exit 1
    }
  done
  prior_gpu_seconds="$(
    python3 "$remote_root/tools/prepare-selection-resume.py" \
      prepare "$resume_job" "$remote_job" "$resume_commit" "$expected_head" "$gpu_uuid"
  )"
  [[ "$prior_gpu_seconds" =~ ^[0-9]+$ ]] || {
    printf 'error: resume GPU accounting is malformed\n' >&2
    exit 1
  }
fi
printf 'GPU_UUID=%s\n' "$gpu_uuid"
printf 'PRIOR_GPU_SECONDS=%s\n' "$prior_gpu_seconds"
REMOTE_SETUP
)"
printf '%s\n' "$setup_output"
gpu_uuid="$(printf '%s\n' "$setup_output" | sed -n 's/^GPU_UUID=//p' | tail -n 1)"
prior_gpu_seconds="$(
  printf '%s\n' "$setup_output" | sed -n 's/^PRIOR_GPU_SECONDS=//p' | tail -n 1
)"
[[ "$gpu_uuid" =~ ^GPU-[A-Za-z0-9-]+$ ]] || die "remote setup did not return a stable GPU UUID"
[[ "$prior_gpu_seconds" =~ ^[0-9]+$ ]] || die "remote setup did not return GPU accounting"

printf 'Synchronizing only the verified prepared dataset...\n'
rsync --archive --checksum --human-readable --partial --progress \
  data/processed/ "$SSH_HOST:$REMOTE_ROOT/data/processed/"

printf 'Submitting detached tmux session %s...\n' "$SESSION_NAME"
ssh "$SSH_HOST" bash "$REMOTE_ROOT/tools/start-gpu-study-remote.sh" \
  "$REMOTE_ROOT" "$REMOTE_JOB" "$gpu_uuid" "$LOCAL_HEAD" "$SESSION_NAME" \
  "$LAUNCH_ID" "$prior_gpu_seconds"

printf '\nRemote one-shot study started successfully.\n'
printf 'The Mac may now disconnect, sleep, or shut down.\n'
printf 'Status:  bash tools/gpu-study.sh status %s\n' "$SSH_HOST"
printf 'Logs:    bash tools/gpu-study.sh logs %s\n' "$SSH_HOST"
printf 'Follow:  bash tools/gpu-study.sh follow %s\n' "$SSH_HOST"
printf 'Attach:  bash tools/gpu-study.sh attach %s\n' "$SSH_HOST"
printf 'Collect: bash tools/gpu-study.sh collect %s\n' "$SSH_HOST"
