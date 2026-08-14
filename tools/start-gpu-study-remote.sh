#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: tools/start-gpu-study-remote.sh REPO_ROOT JOB_ROOT GPU_UUID EXPECTED_COMMIT SESSION LAUNCH_ID [PRIOR_GPU_SECONDS [MODE [DRIVER_COMMIT]]]

Internal detached tmux starter for LateSignal's one-shot GPU study.
Use tools/gpu-study.sh from the submitting Mac instead of invoking this
starter directly.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

(( $# >= 6 && $# <= 9 )) || {
  usage >&2
  exit 2
}

readonly REPO_ROOT="$1"
readonly JOB_ROOT="$2"
readonly GPU_UUID="$3"
readonly EXPECTED_COMMIT="$4"
readonly SESSION="$5"
readonly LAUNCH_ID="$6"
readonly PRIOR_GPU_SECONDS="${7:-0}"
readonly EXECUTION_MODE="${8:-full}"
readonly DRIVER_COMMIT="${9:-$EXPECTED_COMMIT}"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly DRIVER_PATH="$([[ "$EXECUTION_MODE" == "full" ]] && printf '%s' "$REPO_ROOT/tools/run-gpu-study-remote.sh" || printf '%s' "$SCRIPT_DIR/run-gpu-study-remote.sh")"

[[ "$REPO_ROOT" =~ ^/[A-Za-z0-9._/-]+$ && "$REPO_ROOT" != *..* ]] || exit 2
[[ "$JOB_ROOT" =~ ^/[A-Za-z0-9._/-]+$ && "$JOB_ROOT" != *..* ]] || exit 2
[[ "$GPU_UUID" =~ ^GPU-[A-Za-z0-9-]+$ ]] || exit 2
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$SESSION" =~ ^latesignal-[0-9a-f]{12}$ ]] || exit 2
[[ "$LAUNCH_ID" =~ ^[A-Za-z0-9-]+$ ]] || exit 2
[[ "$PRIOR_GPU_SECONDS" =~ ^[0-9]+$ ]] || exit 2
[[ "$EXECUTION_MODE" == "full" || "$EXECUTION_MODE" == "final-recovery" ]] || exit 2
[[ "$DRIVER_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ -x "$DRIVER_PATH" && ! -L "$DRIVER_PATH" ]] || exit 2

command_line="exec setsid --fork --wait bash '$DRIVER_PATH' '$REPO_ROOT' '$JOB_ROOT' '$GPU_UUID' '$EXPECTED_COMMIT' '$LAUNCH_ID' '$PRIOR_GPU_SECONDS' '$EXECUTION_MODE' '$DRIVER_COMMIT'"
tmux new-session -d -s "$SESSION" "$command_line"
for _ in $(seq 1 30); do
  if grep -q "\"launch_id\":\"$LAUNCH_ID\"" "$JOB_ROOT/started.json" 2>/dev/null; then
    exit 0
  fi
  if ! tmux has-session -t "=$SESSION" 2>/dev/null; then
    printf 'error: remote tmux job exited before its started receipt\n' >&2
    tail -n 80 "$JOB_ROOT/job.log" >&2 || true
    exit 1
  fi
  sleep 1
done
printf 'error: timed out waiting for the remote started receipt\n' >&2
exit 1
