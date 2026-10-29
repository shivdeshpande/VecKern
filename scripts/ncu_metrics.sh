#!/usr/bin/env bash
# Collect the Nsight Compute metrics behind the analysis in docs/RESULTS.md, one variant at a time.
# Output: one CSV row per (variant, metric) in $OUT (default results/ncu_metrics.csv).
#
# usage: scripts/ncu_metrics.sh [--n N] [--nq NQ] [--k K] [--m M] [--out FILE] [--variants "v0 v1 ..."]
set -euo pipefail
cd "$(dirname "$0")/.."
N=1000000; NQ=1000; K=10; M=16; OUT=results/ncu_metrics.csv; VARIANTS="v0 v1 v2 v3 v4 v5 v6 v7 v8"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --n) N="$2"; shift 2 ;; --nq) NQ="$2"; shift 2 ;; --k) K="$2"; shift 2 ;; --m) M="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;; --variants) VARIANTS="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

METRICS=(
  gpu__time_duration.sum                                   # kernel time
  launch__grid_size launch__block_size launch__registers_per_thread launch__shared_mem_per_block_dynamic
  launch__occupancy_limit_registers launch__occupancy_limit_shared_mem launch__occupancy_limit_warps
  launch__waves_per_multiprocessor
  sm__warps_active.avg.pct_of_peak_sustained_active        # achieved occupancy
  dram__bytes_read.sum dram__throughput.avg.pct_of_peak_sustained_elapsed
  lts__t_sector_hit_rate.pct                               # L2 hit rate
  l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum              # local-memory (spill) traffic
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum     # shared-memory load wavefronts
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum # ... of which bank conflicts
  smsp__inst_executed.sum
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  l1tex__throughput.avg.pct_of_peak_sustained_active        # L1/shared-memory pipe utilisation
  smsp__issue_active.avg.pct_of_peak_sustained_active      # issue-slot utilisation
)
IFS=,; METRIC_LIST="${METRICS[*]}"; unset IFS

mkdir -p "$(dirname "$OUT")"
[[ -f "$OUT" ]] || echo "variant,n,m,nq,k,kernel,metric,value" > "$OUT"
TMP=$(mktemp)
for v in $VARIANTS; do
  # --reps 0: only the warm-up search runs, so each kernel appears once (plus the tiny LUT kernel, skipped).
  ncu --metrics "$METRIC_LIST" --kernel-name regex:"scan|fused|select" --csv --print-units base \
      ./build/veckern_bench --n "$N" --m "$M" --nq "$NQ" --k "$K" --variants "$v" \
      --cpu-queries 0 --eval-queries 0 --reps 0 --iters 8 --csv /dev/null 2>/dev/null | grep -v '^==' > "$TMP" || true
  python3 - "$TMP" "$v" "$N" "$M" "$NQ" "$K" >> "$OUT" <<'EOF'
import csv, sys
path, variant, n, m, nq, k = sys.argv[1:]
seen = {}
for r in csv.DictReader(open(path)):
    kernel = r["Kernel Name"].split("(")[0].split("<")[0].replace("void ", "").split("::")[-1]
    key = (kernel, r["Metric Name"])
    if key in seen:  # unfused variants launch once per query batch: keep the first batch
        continue
    seen[key] = True
    print(f'{variant},{n},{m},{nq},{k},{kernel},{r["Metric Name"]},{r["Metric Value"].replace(",", "")}')
EOF
  echo "[ncu] $v done" >&2
done
rm -f "$TMP"
echo "wrote $OUT" >&2
