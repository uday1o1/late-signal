#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: tools/start-gpu-study-remote.sh REPO_ROOT JOB_ROOT GPU_UUID EXPECTED_COMMIT SESSION LAUNCH_ID

Internal detached tmux starter for LateSignal's one-shot GPU study.
Use tools/gpu-study.sh from the submitting Mac instead of invoking this
starter directly.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

(( $# == 6 )) || {
  usage >&2
  exit 2
}

readonly REPO_ROOT="$1"
readonly JOB_ROOT="$2"
readonly GPU_UUID="$3"
readonly EXPECTED_COMMIT="$4"
readonly SESSION="$5"
readonly LAUNCH_ID="$6"

[[ "$REPO_ROOT" =~ ^/[A-Za-z0-9._/-]+$ && "$REPO_ROOT" != *..* ]] || exit 2
[[ "$JOB_ROOT" =~ ^/[A-Za-z0-9._/-]+$ && "$JOB_ROOT" != *..* ]] || exit 2
[[ "$GPU_UUID" =~ ^GPU-[A-Za-z0-9-]+$ ]] || exit 2
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$SESSION" =~ ^latesignal-[0-9a-f]{12}$ ]] || exit 2
[[ "$LAUNCH_ID" =~ ^[A-Za-z0-9-]+$ ]] || exit 2

command_line="exec setsid --fork --wait bash '$REPO_ROOT/tools/run-gpu-study-remote.sh' '$REPO_ROOT' '$JOB_ROOT' '$GPU_UUID' '$EXPECTED_COMMIT' '$LAUNCH_ID'"
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
