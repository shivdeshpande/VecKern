#!/usr/bin/env bash
# Profile the scan kernels with Nsight Systems (timeline) and Nsight Compute (roofline, memory, stalls).
# Usage: scripts/profile.sh [bench args...]     e.g. scripts/profile.sh --n 10000000 --m 16 --nq 256
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p profiles
ARGS=("$@")
[ ${#ARGS[@]} -eq 0 ] && ARGS=(--n 2000000 --d 128 --m 16 --nq 256 --reps 1 --cpu-queries 1 --eval-queries 10)

# Timeline: where does end-to-end time go (transfers, LUT, scan, select)?
nsys profile -o profiles/timeline --force-overwrite true ./build/veckern_bench "${ARGS[@]}" --csv profiles/nsys.csv

# Kernel deep dive: roofline + memory workload + warp-stall reasons for every scan/select kernel.
# Limit the capture count; ncu replays each kernel many times.
ncu --set full --kernel-name regex:"scan|fused|select" --launch-count 12 \
    -o profiles/kernels --force-overwrite ./build/veckern_bench "${ARGS[@]}" --csv profiles/ncu.csv

echo "Open profiles/timeline.nsys-rep in Nsight Systems and profiles/kernels.ncu-rep in Nsight Compute."
echo "In Nsight Compute, check: GPU Speed Of Light Roofline, Memory Workload Analysis (L2 vs DRAM),"
echo "Shared memory bank conflicts, and Warp State Statistics."
