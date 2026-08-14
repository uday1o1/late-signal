#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: tools/measure-gpu-study-working-set.sh REPO_ROOT JOB_ROOT [working|retained]

Measure the LateSignal one-shot GPU study working set with bounded retries under one
45-second deadline. Retained mode measures only the final job evidence. A persistent
timeout or filesystem race fails closed.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

(( $# == 2 || $# == 3 )) || {
  usage >&2
  exit 2
}

readonly REPO_ROOT="$1"
readonly JOB_ROOT="$2"
readonly MODE="${3:-working}"
readonly DEADLINE_SECONDS=45
readonly MAX_ATTEMPTS=3

[[ "$REPO_ROOT" =~ ^/[A-Za-z0-9._/-]+$ && "$REPO_ROOT" != *..* ]] || exit 2
[[ "$JOB_ROOT" == "$REPO_ROOT"/runs/one-shot/* && "$JOB_ROOT" != *..* ]] || exit 2
[[ ! -L "$REPO_ROOT" && ! -L "$JOB_ROOT" ]] || exit 2
[[ "$MODE" == "working" || "$MODE" == "retained" ]] || exit 2

paths=()
if [[ "$MODE" == "retained" ]]; then
  candidates=("$JOB_ROOT")
else
  candidates=("$REPO_ROOT/.venv" "$REPO_ROOT/data/processed" "$JOB_ROOT")
fi
for candidate in "${candidates[@]}"; do
  if [[ -e "$candidate" ]]; then
    [[ ! -L "$candidate" ]] || exit 2
    paths+=("$candidate")
  fi
done
(( ${#paths[@]} > 0 )) || {
  printf '0\n'
  exit 0
}

readonly STARTED_AT="$(date +%s)"
readonly DEADLINE_AT="$((STARTED_AT + DEADLINE_SECONDS))"
attempt=1
while (( attempt <= MAX_ATTEMPTS )); do
  now="$(date +%s)"
  remaining="$((DEADLINE_AT - now))"
  (( remaining > 0 )) || exit 1
  if measured="$(
    timeout "${remaining}s" du -sk -- "${paths[@]}" 2>/dev/null \
      | awk '{total += $1} END {print total + 0}'
  )" && [[ "$measured" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$measured"
    exit 0
  fi
  attempt="$((attempt + 1))"
  (( attempt <= MAX_ATTEMPTS )) || exit 1
  now="$(date +%s)"
  (( now + 1 < DEADLINE_AT )) || exit 1
  sleep 1
done

exit 1
