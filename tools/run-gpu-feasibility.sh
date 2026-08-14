#!/usr/bin/env bash

set -Eeuo pipefail

readonly UV_VERSION="0.11.23"
readonly MIN_DISK_GB=25
readonly MIN_MEMORY_GB=16
readonly MIN_VRAM_MIB=8192

usage() {
  cat <<'EOF'
Usage: bash tools/run-gpu-feasibility.sh SSH_HOST [GPU_INDEX]

Prepare a trusted SSH-accessible NVIDIA GPU host and run LateSignal's bounded
feasibility benchmark. The script transfers only the verified prepared dataset,
not the licensed source archive or extracted raw file. It does not start the
selection matrix or final experiments.

Arguments:
  SSH_HOST   SSH configuration alias or user@host target.
  GPU_INDEX  Physical GPU index to expose to the benchmark. Defaults to 0.

Example:
  bash tools/run-gpu-feasibility.sh gpu-workstation 1
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if (( $# < 1 || $# > 2 )); then
  usage >&2
  exit 2
fi

readonly SSH_HOST="$1"
readonly GPU_INDEX="${2:-0}"

[[ "$SSH_HOST" =~ ^[A-Za-z0-9._@-]+$ ]] || die "SSH_HOST contains unsupported characters"
[[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || die "GPU_INDEX must be a non-negative integer"

for command_name in git rsync scp ssh; do
  require_command "$command_name"
done

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd "$REPO_ROOT"

[[ -f pyproject.toml && -f uv.lock ]] || die "could not resolve the LateSignal repository root"
[[ -f data/processed/manifests/preparation.json ]] || die \
  "prepared data is missing; expected data/processed/manifests/preparation.json"
[[ -z "$(git status --porcelain)" ]] || die \
  "local repository has uncommitted or untracked source changes"

readonly LOCAL_HEAD="$(git rev-parse HEAD)"
readonly ORIGIN_HEAD="$(git rev-parse origin/main)"
[[ "$LOCAL_HEAD" == "$ORIGIN_HEAD" ]] || die \
  "local HEAD is not the confirmed origin/main revision"

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

printf 'Checking remote host %s and GPU %s...\n' "$SSH_HOST" "$GPU_INDEX"
remote_home="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$SSH_HOST" 'printf %s "$HOME"')"
[[ "$remote_home" =~ ^/[A-Za-z0-9._/-]+$ && "$remote_home" != *..* ]] || die \
  "remote home path contains unsupported characters"
readonly REMOTE_ROOT="$remote_home/late-signal"
readonly RESULT_RELATIVE="runs/feasibility/gpu${GPU_INDEX}.json"

ssh "$SSH_HOST" bash -s -- \
  "$REMOTE_ROOT" \
  "$clone_url" \
  "$LOCAL_HEAD" \
  "$GPU_INDEX" \
  "$MIN_DISK_GB" \
  "$MIN_MEMORY_GB" \
  "$MIN_VRAM_MIB" <<'REMOTE_SETUP'
set -Eeuo pipefail

remote_root="$1"
clone_url="$2"
expected_head="$3"
gpu_index="$4"
min_disk_gb="$5"
min_memory_gb="$6"
min_vram_mib="$7"

if [[ ! -d "$remote_root/.git" ]]; then
  GIT_TERMINAL_PROMPT=0 git \
    -c credential.helper= \
    -c credential.helper=store \
    clone \
    --branch main \
    --single-branch \
    "$clone_url" \
    "$remote_root"
else
  [[ -z "$(git -C "$remote_root" status --porcelain)" ]] || {
    printf 'error: remote repository has uncommitted or untracked source changes\n' >&2
    exit 1
  }
  git -C "$remote_root" remote set-url origin "$clone_url"
  GIT_TERMINAL_PROMPT=0 git \
    -c credential.helper= \
    -c credential.helper=store \
    -C "$remote_root" \
    fetch --prune origin main
  git -C "$remote_root" checkout main
  git -C "$remote_root" merge --ff-only origin/main
fi

actual_head="$(git -C "$remote_root" rev-parse HEAD)"
[[ "$actual_head" == "$expected_head" ]] || {
  printf 'error: remote revision does not match the confirmed local revision\n' >&2
  exit 1
}

command -v nvidia-smi >/dev/null 2>&1 || {
  printf 'error: nvidia-smi is unavailable on the remote host\n' >&2
  exit 1
}

available_disk_kib="$(df -Pk "$remote_root" | awk 'NR == 2 {print $4}')"
required_disk_kib="$((min_disk_gb * 1024 * 1024))"
(( available_disk_kib >= required_disk_kib )) || {
  printf 'error: remote host has less than %s GB free disk\n' "$min_disk_gb" >&2
  exit 1
}

available_memory_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
required_memory_kib="$((min_memory_gb * 1024 * 1024))"
(( available_memory_kib >= required_memory_kib )) || {
  printf 'error: remote host has less than %s GB available memory\n' "$min_memory_gb" >&2
  exit 1
}

vram_mib="$(
  nvidia-smi \
    --id="$gpu_index" \
    --query-gpu=memory.total \
    --format=csv,noheader,nounits | tr -d '[:space:]'
)"
[[ "$vram_mib" =~ ^[0-9]+$ ]] || {
  printf 'error: GPU %s is unavailable\n' "$gpu_index" >&2
  exit 1
}
(( vram_mib >= min_vram_mib )) || {
  printf 'error: GPU %s has less than %s MiB VRAM\n' "$gpu_index" "$min_vram_mib" >&2
  exit 1
}

compute_pids="$(
  nvidia-smi \
    --id="$gpu_index" \
    --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true
)"
[[ -z "$compute_pids" ]] || {
  printf 'error: GPU %s already has active compute processes\n' "$gpu_index" >&2
  exit 1
}

mkdir -p "$remote_root/data/processed" "$remote_root/runs/feasibility"
nvidia-smi \
  --id="$gpu_index" \
  --query-gpu=index,name,memory.total,memory.free,utilization.gpu \
  --format=csv,noheader
REMOTE_SETUP

printf 'Synchronizing the prepared dataset to %s:%s...\n' "$SSH_HOST" "$REMOTE_ROOT"
rsync \
  --archive \
  --checksum \
  --human-readable \
  --partial \
  --progress \
  data/processed/ \
  "$SSH_HOST:$REMOTE_ROOT/data/processed/"

printf 'Installing the locked environment and running the bounded benchmark...\n'
ssh "$SSH_HOST" bash -s -- \
  "$REMOTE_ROOT" \
  "$GPU_INDEX" \
  "$UV_VERSION" \
  "$RESULT_RELATIVE" <<'REMOTE_BENCHMARK'
set -Eeuo pipefail

remote_root="$1"
gpu_index="$2"
uv_version="$3"
result_relative="$4"
uv_environment="$HOME/.local/share/latesignal/uv-$uv_version"
uv_bin="$uv_environment/bin/uv"

cd "$remote_root"

if [[ ! -x "$uv_bin" ]]; then
  python3 -m venv "$uv_environment"
  "$uv_environment/bin/pip" install "uv==$uv_version"
fi

"$uv_bin" sync --frozen --all-groups

"$uv_bin" run python - <<'PY'
from pathlib import Path

from latesignal.experiments.protocol_lock import _verify_prepared_data

manifest_path = Path("data/processed/manifests/preparation.json")
verification = _verify_prepared_data(manifest_path)
print(
    "Verified prepared data: "
    f"{verification['verified_files']} files, "
    f"{verification['verified_file_bytes']} bytes"
)
PY

set +e
CUDA_VISIBLE_DEVICES="$gpu_index" "$uv_bin" run latesignal protocol estimate \
  configs/experiments/final.yaml \
  --out "$result_relative" \
  --json
estimate_status=$?
set -e

if (( estimate_status > 1 )); then
  printf 'error: feasibility benchmark failed with exit code %s\n' "$estimate_status" >&2
  exit "$estimate_status"
fi
[[ -s "$result_relative" ]] || {
  printf 'error: feasibility benchmark did not produce its result\n' >&2
  exit 1
}
REMOTE_BENCHMARK

mkdir -p "$(dirname -- "$REPO_ROOT/$RESULT_RELATIVE")"
scp "$SSH_HOST:$REMOTE_ROOT/$RESULT_RELATIVE" "$REPO_ROOT/$RESULT_RELATIVE"

printf '\nBenchmark complete.\n'
printf 'Result: %s\n' "$REPO_ROOT/$RESULT_RELATIVE"
