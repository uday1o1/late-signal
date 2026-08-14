#!/usr/bin/env bash

set -Eeuo pipefail

readonly MIN_DISK_GB=25
readonly MIN_MEMORY_GB=16
readonly MIN_VRAM_MIB=8192

usage() {
  cat <<'EOF'
Usage:
  bash tools/gpu-study.sh submit SSH_HOST [GPU_INDEX]
  bash tools/gpu-study.sh status SSH_HOST
  bash tools/gpu-study.sh logs SSH_HOST
  bash tools/gpu-study.sh follow SSH_HOST
  bash tools/gpu-study.sh attach SSH_HOST
  bash tools/gpu-study.sh collect SSH_HOST

Submit the exact current origin/main revision as one detached, resumable tmux
one-shot job on a trusted SSH-accessible NVIDIA GPU host. After submit confirms the
started receipt, the Mac can disconnect, sleep, or shut down without stopping
the remote job.

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
  status|logs|follow|attach|collect)
    (( $# == 1 )) || {
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
readonly GPU_INDEX="${2:-0}"
[[ "$SSH_HOST" =~ ^[A-Za-z0-9._@-]+$ ]] || die "SSH_HOST contains unsupported characters"
[[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || die "GPU_INDEX must be a non-negative integer"

for command_name in git rsync ssh; do
  require_command "$command_name"
done
if [[ "$ACTION" == "collect" ]]; then
  require_command uv
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd "$REPO_ROOT"
[[ -f pyproject.toml && -f uv.lock ]] || die "could not resolve the LateSignal repository root"

readonly LOCAL_HEAD="$(git rev-parse HEAD)"
readonly COMMIT_SHORT="${LOCAL_HEAD:0:12}"
readonly SESSION_NAME="latesignal-$COMMIT_SHORT"
readonly LAUNCH_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
remote_home="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$SSH_HOST" 'printf %s "$HOME"')"
[[ "$remote_home" =~ ^/[A-Za-z0-9._/-]+$ && "$remote_home" != *..* ]] || \
  die "remote home path contains unsupported characters"
readonly REMOTE_GIT_ROOT="$remote_home/late-signal"
readonly REMOTE_ROOT="$remote_home/late-signal-worktrees/$COMMIT_SHORT"
readonly REMOTE_JOB="$REMOTE_ROOT/runs/one-shot/$COMMIT_SHORT"

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
        not value.startswith("final/aggregate/")
        or path.suffix not in allowed_suffixes
        or any(token in value.lower() for token in forbidden)
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
  "$MIN_VRAM_MIB" <<'REMOTE_SETUP'
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

gpu_uuid="$(nvidia-smi --id="$gpu_index" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')"
vram_mib="$(nvidia-smi --id="$gpu_index" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d '[:space:]')"
[[ "$gpu_uuid" =~ ^GPU-[A-Za-z0-9-]+$ && "$vram_mib" =~ ^[0-9]+$ ]] || {
  printf 'error: selected GPU is unavailable\n' >&2
  exit 1
}
(( vram_mib >= min_vram_mib )) || {
  printf 'error: selected GPU has less than %s MiB VRAM\n' "$min_vram_mib" >&2
  exit 1
}
compute_pids="$(nvidia-smi --id="$gpu_index" --query-compute-apps=pid \
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
printf 'GPU_UUID=%s\n' "$gpu_uuid"
REMOTE_SETUP
)"
printf '%s\n' "$setup_output"
gpu_uuid="$(printf '%s\n' "$setup_output" | sed -n 's/^GPU_UUID=//p' | tail -n 1)"
[[ "$gpu_uuid" =~ ^GPU-[A-Za-z0-9-]+$ ]] || die "remote setup did not return a stable GPU UUID"

printf 'Synchronizing only the verified prepared dataset...\n'
rsync --archive --checksum --human-readable --partial --progress \
  data/processed/ "$SSH_HOST:$REMOTE_ROOT/data/processed/"

printf 'Submitting detached tmux session %s...\n' "$SESSION_NAME"
ssh "$SSH_HOST" bash "$REMOTE_ROOT/tools/start-gpu-study-remote.sh" \
  "$REMOTE_ROOT" "$REMOTE_JOB" "$gpu_uuid" "$LOCAL_HEAD" "$SESSION_NAME" \
  "$LAUNCH_ID"

printf '\nRemote one-shot study started successfully.\n'
printf 'The Mac may now disconnect, sleep, or shut down.\n'
printf 'Status:  bash tools/gpu-study.sh status %s\n' "$SSH_HOST"
printf 'Logs:    bash tools/gpu-study.sh logs %s\n' "$SSH_HOST"
printf 'Follow:  bash tools/gpu-study.sh follow %s\n' "$SSH_HOST"
printf 'Attach:  bash tools/gpu-study.sh attach %s\n' "$SSH_HOST"
printf 'Collect: bash tools/gpu-study.sh collect %s\n' "$SSH_HOST"
