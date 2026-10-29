#!/usr/bin/env bash
# Run a command only once a GPU is idle: 0% utilisation for QUIET seconds in a row. Processes that are resident
# but not computing (holding memory between phases) do not count; any active work does.
# Timings taken next to other jobs are not meaningful; this makes that rule automatic.
#
# usage: scripts/wait_idle_gpu.sh [--any] [--gpu N] [--quiet SECONDS] [--timeout SECONDS] -- command args...
#   --any       accept whichever GPU becomes idle first (sets CUDA_VISIBLE_DEVICES accordingly)
#   --gpu N     wait for GPU N specifically (default 0)
#   --no-wait   do not wait (pipeline testing only: results are NOT valid timings)
set -euo pipefail
GPUS=(0); QUIET=60; TIMEOUT=0; NOWAIT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --any) mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader); shift ;;
    --gpu) GPUS=("$2"); shift 2 ;;
    --quiet) QUIET="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --no-wait) NOWAIT=1; shift ;;
    --) shift; break ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[[ $# -gt 0 ]] || { echo "no command given" >&2; exit 2; }
if [[ $NOWAIT == 1 ]]; then
  echo "[wait_idle_gpu] GPU ${GPUS[0]} NOT checked (--no-wait); running: $*" >&2
  export CUDA_VISIBLE_DEVICES=${GPUS[0]}
  exec "$@"
fi

busy() {  # true if GPU $1 shows any utilisation
  local util
  util=$(nvidia-smi -i "$1" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  [[ "$util" -ne 0 ]]
}

start=$(date +%s)
declare -A quiet_since
while true; do
  now=$(date +%s)
  for g in "${GPUS[@]}"; do
    if busy "$g"; then
      quiet_since[$g]=""
    else
      [[ -z "${quiet_since[$g]:-}" ]] && quiet_since[$g]=$now
      if (( now - quiet_since[$g] >= QUIET )); then
        echo "[wait_idle_gpu] GPU $g idle for ${QUIET}s at $(date '+%F %T') epoch=$(date +%s); running: $*" >&2
        export CUDA_VISIBLE_DEVICES=$g
        exec "$@"
      fi
    fi
  done
  if (( TIMEOUT > 0 && now - start >= TIMEOUT )); then
    echo "[wait_idle_gpu] timed out after ${TIMEOUT}s" >&2; exit 3
  fi
  sleep 2
done
