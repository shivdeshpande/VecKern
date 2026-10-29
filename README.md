# VecKern: GPU kernels for compressed vector search

VecKern is a CUDA study of the inner loop of GPU vector search: the **product-quantization (PQ) distance scan
and top-k selection** behind IVF-PQ (FAISS, cuVS) and the compressed in-memory traversal of billion-scale
systems such as BANG and DiskANN. It builds that loop up in **nine kernel versions (v0-v8)**, each changing one
thing, validates every version against a CPU reference, profiles each one with Nsight Compute, and compares the
result with NVIDIA's cuVS.

**Main results** (A100 80GB PCIe; all from one idle-GPU, contention-checked campaign; see [docs/RESULTS.md](docs/RESULTS.md))

| | |
| --- | --- |
| 1M vectors, 1,000 queries, k = 10 | **86,216 queries/s** (11.6 ms): 8.6x the baseline kernel, 3.5x from fusion alone |
| vs NVIDIA cuVS (IVF-PQ as a flat scan, same FP32 lookup tables) | **5-18% faster** (SIFT1M, 1M, 10M); within 4% of cuVS's FP16 tables on SIFT1M |
| k = 100 (candidates for re-ranking) | warp-cooperative top-k is **8-12x faster** than per-thread lists |
| Scale and latency | 904 queries/s at 100M vectors; 0.44 ms for one query over 10M |
| Bottleneck | shared-memory bank conflicts on random lookups: the final kernel runs at **92-94% of that floor** (L1/shared pipe 99.7%, DRAM < 1% busy at 1M) |
| Accuracy | recall identical to the CPU reference for all 9 versions; SIFT1M recall@10 0.543 / 0.700 / 0.875 for m = 16 / 32 / 64, 0.968-0.999 with exact re-ranking |

![Time per stage for each kernel version](docs/figures/variants.png)

**Start here: [docs/GUIDE.md](docs/GUIDE.md)** explains every kernel, block size and design decision in detail.
Full measurements, figures and the profiler evidence: [docs/RESULTS.md](docs/RESULTS.md). Milestone status:
[docs/PLAN.md](docs/PLAN.md).

## Contents

- [Background](#background)
- [Kernel versions](#kernel-versions)
- [How it works](#how-it-works)
- [Build](#build)
- [Run](#run)
- [Reproduce all results](#reproduce-all-results)
- [Testing and correctness](#testing-and-correctness)
- [Repository layout](#repository-layout)
- [Limitations](#limitations)
- [References](#references)

## Background

Product quantization compresses each d-dimensional float vector into m one-byte codes. Each byte is the id of
the nearest of 256 centroids in one sub-space, so a 128-dimension vector (512 bytes) becomes 16 bytes. Search has
three steps:

```
query ─► LUT kernel ─────────► scan ───────────────► k-selection ─► top-k ids
         m x 256 table of       distance(i) =         keep the k
         sub-distances          Σ_s LUT[s][code(i,s)]  smallest
                                └──────────── v4-v8: fused into one kernel ─────────┘
```

The scan does one table lookup and one add per code byte. That makes it limited by memory and latency rather
than arithmetic, which is what makes it a good subject for HPC-style optimisation.

## Kernel versions

| Version | What changes | Why |
| --- | --- | --- |
| `v0_global_lut` | baseline: LUT in global memory, row-major codes | starting point |
| `v1_shared_lut` | LUT in shared memory | lookups from on-chip memory |
| `v2_shared_lut_vec` | + 16-byte code loads | fewer load instructions |
| `v3_interleaved` | warp-interleaved code layout (coalesced byte loads) | coalescing |
| `v4_fused` | scan + per-thread top-k in one kernel | no `nq x n` distance matrix, no low-occupancy select pass |
| `v5_fused_reg` | top-k list kept in registers (fully unrolled insertion) | v4's list spilled to local memory (21 GB per launch) |
| `v6_fused_vec` | 16-byte-word interleaved layout + ILP (4 vectors per thread per step) | coalesced *and* wide loads; overlap latency |
| `v7_fused_vec_mq` | + Q queries per block sharing every code load | each fetched code byte serves Q queries (+17% at 100M vectors) |
| `v8_fused_warp` | one warp-cooperative top-k list per warp (ballot + shuffles) | lists cost K/32 registers per lane instead of K per thread; 8-12x faster at k = 100 |

## How it works

All kernels are in [`src/cuda/kernels.cuh`](src/cuda/kernels.cuh); launch logic, batching and timing are in
[`src/cuda/searcher.cu`](src/cuda/searcher.cu).

| Kernel | Block | Grid | Shared memory | Each thread / warp |
| --- | --- | --- | --- | --- |
| `lut_kernel` | 256 | (m, nq) | dsub floats | one thread per centroid |
| `scan_*` (v0-v3) | 256 (tunable) | (blocks_x, queries per batch) | m x 1 KB LUT (v1-v3) | grid-stride loop, one distance at a time |
| `select_kernel` | 32-128 | (queries) | THREADS x K x 8 B | per-thread sorted list, tree merge in shared memory |
| `fused_scan_select` (v4) | 32-128 | (blocks_x, queries) | max(LUT, merge buffer), reused | scan + per-thread top-K, block merge |
| `fused_v2<K,T,Q,VEC,ILP>` (v5-v7) | 32-128 | (blocks_x, queries / Q) | max(Q LUTs, merge buffer) | ILP vectors per step, Q queries per block, register top-K |
| `fused_warp<K,T,Q,ILP>` (v8) | 256 | (blocks_x, queries / Q) | max(Q LUTs, (T/32) x K x 8 B) | one sorted top-K per warp spread over the lanes |

Key design points:

- **Layouts.** Row-major codes (v0-v2), byte-interleaved per warp (v3-v5) and 16-byte-word-interleaved per warp
  (v6-v8). The interleaved copies are built on the GPU the first time a variant needs them.
- **Register-resident lists (v5).** GPU registers cannot be indexed with a run-time value, so insertion is fully
  unrolled with compile-time indices only.
- **Warp-cooperative top-k (v8).** Each lane holds K/32 entries; a candidate below the warp's K-th best is
  broadcast with `__shfl_sync` and inserted by all lanes at once (keep / take / shift from the left neighbour).
  Loop bounds are warp-uniform so every lane joins every ballot and shuffle.
- **Grid sizing (v5-v8).** `cudaOccupancyMaxActiveBlocksPerMultiprocessor` gives the real number of resident
  blocks per wave for each kernel configuration; `blocks_x` minimises wasted capacity in the last wave (best or
  within 0.1% of the best fixed value in the sweeps), and small batches may use more blocks per query so even one
  query fills the GPU.
- **K as a template parameter.** k ∈ [1, 128] maps to K ∈ {8, 16, 32, 64, 128}.
- **Bounded memory.** Unfused variants batch queries to keep the distance matrix under 512 MB; fused variants need
  no matrix.

The host API ([`include/veckern/gpu.hpp`](include/veckern/gpu.hpp)) contains no CUDA types:

```cpp
veckern::ProductQuantizer pq(d, m);
pq.train(base, n_train);
pq.encode(base, n, codes);

veckern::gpu::PQSearcher searcher(pq, codes, n);         // codes copied to the GPU once
auto t = searcher.search(queries, nq, k, veckern::gpu::ScanVariant::FusedWarp, results);
auto shape = searcher.last_launch();                     // threads, blocks_x, Q, ILP, waves actually used
```

## Build

Requirements: CMake ≥ 3.22 and a C++17 compiler. With CUDA ≥ 11.8 the GPU kernels and tests are built too;
without it, only the CPU reference builds (enough to develop on a laptop). OpenMP is used when found.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The CUDA architecture defaults to `native` on CMake ≥ 3.24 and to `80` (A100) otherwise; override it with
`-DCMAKE_CUDA_ARCHITECTURES="80;90"`. Disable CUDA with `-DVECKERN_ENABLE_CUDA=OFF`. The GPU library takes a few
minutes to compile because every (K, Q, ILP) combination is a separate template instance.

## Run

```bash
# synthetic data (clustered, low intrinsic dimension), all variants
./build/veckern_bench --n 1000000 --d 128 --m 16 --nq 1000 --k 10

# SIFT1M with official ground truth, plus exact re-ranking of the top 100
./build/veckern_bench --base sift_base.fvecs --query sift_query.fvecs --gt sift_groundtruth.ivecs \
                      --m 16 --nq 10000 --k 10 --rerank 100 --csv results/sift1m.csv

# selected variants with explicit tuning
./build/veckern_bench --n 10000000 --variants v4,v8 --queries-per-block 4 --ilp 2 --label my_test
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--n`, `--d` | 1000000, 128 | synthetic size (with `--base`, `--n` caps the vectors loaded) |
| `--clusters`, `--latent-dim`, `--seed` | 100, 16, 1 | synthetic data: x = center + A z + noise, z in `latent-dim` dimensions |
| `--m` | 16 | PQ sub-spaces = bytes per vector; must divide d (v2, v6-v8 need m % 16 == 0) |
| `--nq`, `--k` | 1000, 10 | queries and neighbours (k ≤ 128) |
| `--rerank R` | 0 | fetch R candidates and re-score them exactly on the CPU |
| `--train`, `--iters` | 100000, 15 | PQ training sample and k-means iterations |
| `--reps` | 5 | timed repetitions (median reported) after one warm-up |
| `--eval-queries`, `--cpu-queries` | 100, 100 | queries used for recall / run on the CPU reference (0 = skip) |
| `--variants` | all | comma list of prefixes, e.g. `v2,v8` |
| `--threads`, `--blocks-per-query` | 256, 0 (auto) | v0-v3 block size; grid x-dimension override |
| `--queries-per-block`, `--ilp` | 2, 4 | Q and vectors-per-step for v6-v8 |
| `--base`, `--query`, `--gt` | | `.fvecs` / `.bvecs` data and `.ivecs` ground truth |
| `--csv`, `--label` | results.csv | output file (rows are appended) and a label for the rows |

The table printed by the benchmark includes the launch shape actually used (threads, blocks_x, grid_y, Q, ILP,
waves) next to the per-stage times, so every number can be traced to its configuration.

## Reproduce all results

```bash
scripts/run_all.sh --sift /path/to/sift1M --cuvs-python /path/to/cuvs_env/bin/python   # hours; waits for an idle GPU
python3 scripts/check_contention.py results/final       # flags any measurement that overlapped another job
python3 scripts/summarize_results.py results/final > docs/results_tables.md
python3 scripts/plot_results.py results/final --out docs/figures
```

`run_all.sh` runs the tests, SIFT1M at m = 16/32/64 (with and without re-ranking), scaling from 1M to 100M
vectors, latency at small batch sizes, the parameter sweeps, the cuVS baseline and the Nsight Compute metrics.
Each GPU step first waits for 60 s of 0% utilisation (`scripts/wait_idle_gpu.sh`), and a monitor records
per-process GPU use throughout so contaminated steps can be identified afterwards. `--quick` runs the same
pipeline at small sizes in minutes.

| Script | Purpose |
| --- | --- |
| `scripts/run_all.sh` | the full measurement campaign |
| `scripts/wait_idle_gpu.sh` | run a command only once a GPU is idle |
| `scripts/check_contention.py` | flag measurements that overlapped other GPU work |
| `scripts/ncu_metrics.sh` | Nsight Compute metrics per variant (occupancy, spills, bank conflicts, L1/L2/DRAM) |
| `scripts/cuvs_baseline.py` | cuVS IVF-PQ configured as a flat PQ scan, same data |
| `scripts/summarize_results.py` | markdown tables straight from the CSVs |
| `scripts/plot_results.py` | figures |
| `scripts/profile.sh` | interactive Nsight Systems / Nsight Compute reports |

## Testing and correctness

```bash
ctest --test-dir build --output-on-failure
```

- **`test_cpu`** checks the reference: top-k ordering and padding, ADC equals the exact distance to the decoded
  vector, PQ reconstruction error, staged vs fused CPU search, exact search, reranking, file I/O and the
  synthetic data generator.
- **`test_gpu`** checks every GPU version against the CPU: distances within a relative 1e-3, ids at least 99%
  identical (float rounding can reorder near-ties), ascending order, every K template (k = 1 to 128), every
  queries-per-block and ILP setting with a query count and database size that do not divide evenly, tuning
  overrides, and rejection of invalid inputs.

Ties are broken by (distance, id) on both CPU and GPU so results are comparable.

## Repository layout

| Path | Contents |
| --- | --- |
| `include/veckern/` | public headers: PQ, CPU reference, I/O and synthetic data, GPU API |
| `src/cpu/` | PQ training (k-means per sub-space), encoding, ADC, top-k, exact search, reranking, recall |
| `src/cuda/kernels.cuh` | all CUDA kernels (v0-v8, LUT, selection) |
| `src/cuda/searcher.cu` | device memory, layouts, launch configuration, batching, CUDA-event timing |
| `bench/bench_main.cpp` | benchmark driver (table + CSV) |
| `tests/` | CPU reference tests and GPU-vs-CPU tests |
| `scripts/` | measurement campaign, profiling, baseline, tables and figures |
| `results/` | benchmark CSVs (`results/final` = the published campaign) |
| `docs/` | RESULTS.md (measurements), PLAN.md (milestones and decisions), figures |

## Limitations

- Timed on one GPU model (A100 80GB PCIe); no AMD/HIP port (no hardware available to test it).
- Large-scale runs (10M, 100M) use synthetic data with realistic structure; real-data recall is measured on SIFT1M.
- 8-bit PQ with FP32 lookup tables only; FP16/4-bit tables and IVF partitioning are listed as next steps.
- The grid model ignores L2 locality (it makes the intermediate v5 slower than v4 at 10M+); the final versions are unaffected.
- CPU-reference timings are indicative: the server's CPUs were shared with other jobs.
- Interleaved layouts are extra copies of the codes on the GPU (each variant builds only the one it needs).

## References

- H. Jégou, M. Douze, C. Schmid. *Product Quantization for Nearest Neighbor Search.* IEEE TPAMI, 2011.
- J. Johnson, M. Douze, H. Jégou. *Billion-scale similarity search with GPUs.* IEEE Transactions on Big Data, 2019
  (FAISS GPU, WarpSelect).
- S. J. Subramanya et al. *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node.* NeurIPS 2019.
- K. Ootomo et al. *CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs.* ICDE 2024.
- NVIDIA cuVS: GPU vector search library (IVF-PQ), used as the baseline.
# VecKern
