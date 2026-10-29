#!/usr/bin/env bash
# Re-run specific benchmark commands (one per line in CMDFILE) under the same rules as run_all.sh:
# idle-GPU gate before each, per-process GPU monitor, timestamped log for scripts/check_contention.py.
# Use it for measurements check_contention.py flagged. CSV rows are appended; summaries keep the last row.
#
# usage: scripts/rerun_commands.sh CMDFILE OUTDIR      (then: scripts/check_contention.py OUTDIR)
set -uo pipefail
cd "$(dirname "$0")/.."
CMDFILE="$1"; OUT="$2"
mkdir -p "$OUT"
LOG="$OUT/run_all.log"
GPU="scripts/wait_idle_gpu.sh --any --quiet ${QUIET:-60} --"

( while true; do
    echo "$(date +%s) $(nvidia-smi pmon -c 1 -s u 2>/dev/null | grep -v '^#' | awk '{print $1, $2, $4, $NF}' | tr '\n' ';')"
    sleep 2
  done ) >> "$OUT/gpu_monitor.log" 2>/dev/null &
MONITOR=$!
trap 'kill $MONITOR 2>/dev/null' EXIT

echo "=== $(date '+%F %T') rerun of $(grep -c . "$CMDFILE") commands" >> "$LOG"
while IFS= read -r cmd; do
  [[ -z "$cmd" ]] && continue
  echo "+ start=$(date +%s) $cmd" >> "$LOG"
  # shellcheck disable=SC2086
  $GPU $cmd >> "$LOG" 2>&1
  echo "+ end=$(date +%s)" >> "$LOG"
done < "$CMDFILE"
echo "=== $(date '+%F %T') done" >> "$LOG"
