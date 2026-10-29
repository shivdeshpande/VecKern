#!/usr/bin/env bash
# The full measurement campaign behind docs/RESULTS.md. Every GPU step waits for an idle GPU first
# (scripts/wait_idle_gpu.sh), because timings taken next to other jobs are not meaningful.
#
# usage: scripts/run_all.sh [--sift DIR] [--cuvs-python PATH] [--out DIR] [--quick]
#   --quick  small sizes and fewer repetitions, to check the pipeline end to end (minutes, not hours)
# Results: $OUT/*.csv and $OUT/run_all.log. Safe to re-run: CSVs are appended, each row carries a label.
set -uo pipefail
cd "$(dirname "$0")/.."
SIFT=/mnt/hdd_volume2/cs24mtech12017_500GB/data/sift-1M
CUVS_PY=python3
OUT=results/final
QUICK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sift) SIFT="$2"; shift 2 ;; --cuvs-python) CUVS_PY="$2"; shift 2 ;; --out) OUT="$2"; shift 2 ;;
    --quick) QUICK=1; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$OUT"
LOG="$OUT/run_all.log"
B=./build/veckern_bench
GPU="scripts/wait_idle_gpu.sh --any --quiet ${QUIET:-60} --"
# NO_WAIT=1 skips the idle check (to test the pipeline on a busy GPU; such numbers are not valid results).
[[ "${NO_WAIT:-0}" == 1 ]] && GPU="scripts/wait_idle_gpu.sh --gpu ${GPU_ID:-0} --no-wait --"
REPS=5; NQ=1000; SCALES="1000000 10000000 100000000"; SIFT_NQ=10000
if [[ $QUICK == 1 ]]; then REPS=2; SCALES="200000 1000000"; SIFT_NQ=1000; fi

step() { echo; echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
# Each command is bracketed by epoch timestamps so scripts/check_contention.py can match it against the
# GPU monitor log and flag any measurement that overlapped another job.
run() {
  echo "+ start=$(date +%s) $*" >> "$LOG"
  "$@" 2>&1 | tee -a "$LOG" | grep -E '^(v[0-9]|cpu|cuvs|appended|error|\[)' || true
  echo "+ end=$(date +%s)" >> "$LOG"
}

# GPU monitor: every 2 s, per-process SM utilisation on each GPU ("<epoch> <gpu> <pid> <sm%> <name>;...").
( while true; do
    echo "$(date +%s) $(nvidia-smi pmon -c 1 -s u 2>/dev/null | grep -v '^#' | awk '{print $1, $2, $4, $NF}' | tr '\n' ';')"
    sleep 2
  done ) >> "$OUT/gpu_monitor.log" 2>/dev/null &
MONITOR=$!
trap 'kill $MONITOR 2>/dev/null' EXIT

step "environment"
{ nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; nvcc --version | tail -1; git rev-parse --short HEAD 2>/dev/null; } | tee -a "$LOG"

step "1. correctness: CPU and GPU test suites"
run ./build/test_cpu
run $GPU ./build/test_gpu

step "2. SIFT1M: all variants, m = 16/32/64, official ground truth"
for m in 16 32 64; do
  run $GPU $B --base "$SIFT/sift_base.fvecs" --query "$SIFT/sift_query.fvecs" --gt "$SIFT/sift_groundtruth.ivecs" \
      --m $m --nq $SIFT_NQ --k 10 --eval-queries 1000 --cpu-queries 1000 --reps $REPS \
      --label "sift1m_m$m" --csv "$OUT/sift1m.csv"
done

step "3. SIFT1M with exact re-ranking of the top 100 (fused GPU variants)"
for m in 16 32 64; do
  run $GPU $B --base "$SIFT/sift_base.fvecs" --query "$SIFT/sift_query.fvecs" --gt "$SIFT/sift_groundtruth.ivecs" \
      --m $m --nq $SIFT_NQ --k 10 --rerank 100 --variants v4,v7,v8 --eval-queries 1000 --cpu-queries 0 --reps $REPS \
      --label "sift1m_m${m}_rerank100" --csv "$OUT/sift1m_rerank.csv"
done

step "4. scaling on synthetic data (latent dim 16): 1M -> 100M vectors"
for n in $SCALES; do
  vars="all"; cpu=100
  # Unfused variants need an n-wide distance row per query; at 100M that is 400 MB per query, so only 1
  # query fits a batch and the select pass runs one block at a time (impractical). The fused variants are run.
  (( n >= 100000000 )) && { vars="v4,v5,v6,v7,v8"; cpu=10; }
  (( n >= 10000000 && n < 100000000 )) && vars="v2,v4,v5,v6,v7,v8"
  run $GPU $B --n $n --m 16 --nq $NQ --k 10 --variants $vars --eval-queries 100 --cpu-queries $cpu --reps $REPS \
      --label "synthetic_n$n" --csv "$OUT/scaling.csv"
done

step "5. online latency regime: few queries at 10M"
n_lat=10000000; [[ $QUICK == 1 ]] && n_lat=1000000
for nq in 1 10 100; do
  run $GPU $B --n $n_lat --m 16 --nq $nq --k 10 --variants v4,v6,v7,v8 --eval-queries 0 --cpu-queries 0 \
      --reps $((REPS * 4)) --label "latency_nq$nq" --csv "$OUT/latency.csv"
done

step "6. parameter sweeps (1M and 10M, nq = 1000)"
for n in 1000000 $n_lat; do
  for ilp in 1 2 4; do
    for q in 1 2 4 8; do
      run $GPU $B --n $n --m 16 --nq $NQ --k 10 --variants v7,v8 --queries-per-block $q --ilp $ilp \
          --eval-queries 0 --cpu-queries 0 --reps $REPS --label "sweep_q${q}_ilp${ilp}_n$n" --csv "$OUT/sweep_q_ilp.csv"
    done
  done
  for bpq in 0 1 2 4 8 16 32 64; do
    run $GPU $B --n $n --m 16 --nq $NQ --k 10 --variants v4,v6,v8 --blocks-per-query $bpq \
        --eval-queries 0 --cpu-queries 0 --reps $REPS --label "sweep_bpq${bpq}_n$n" --csv "$OUT/sweep_blocks.csv"
  done
done
for t in 64 128 256 512 1024; do
  run $GPU $B --n 1000000 --m 16 --nq $NQ --k 10 --variants v2 --threads $t \
      --eval-queries 0 --cpu-queries 0 --reps $REPS --label "sweep_threads$t" --csv "$OUT/sweep_threads.csv"
done

step "7. cuVS baseline (IVF-PQ as a flat PQ scan)"
run $GPU "$CUVS_PY" scripts/cuvs_baseline.py --base "$SIFT/sift_base.fvecs" --query "$SIFT/sift_query.fvecs" \
    --gt "$SIFT/sift_groundtruth.ivecs" --m 16 --nq $SIFT_NQ --reps $REPS --label sift1m_m16 --csv "$OUT/cuvs.csv"
for n in 1000000 $n_lat; do
  run $GPU "$CUVS_PY" scripts/cuvs_baseline.py --n $n --m 16 --nq $NQ --reps $REPS --label "synthetic_n$n" \
      --csv "$OUT/cuvs.csv"
done

step "8. Nsight Compute metrics"
run $GPU scripts/ncu_metrics.sh --n 1000000 --nq $NQ --out "$OUT/ncu_1m.csv"
run $GPU scripts/ncu_metrics.sh --n $n_lat --nq $NQ --variants "v2 v4 v5 v6 v7 v8" --out "$OUT/ncu_10m.csv"

step "done"
