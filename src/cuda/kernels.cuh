#pragma once
// CUDA kernels for PQ asymmetric distance computation and k-selection.
//
// Distances are summed in sub-space order s = 0..m-1, the same order as the CPU reference.
// The vectorised variant groups the additions differently, so results agree to rounding, not bit-exactly.
#include <cfloat>
#include <cstdint>
#include <cuda_runtime.h>

namespace veckern::gpu::kernels {

constexpr int KSUB = 256;
constexpr int WARP = 32;

// ---------------------------------------------------------------- LUT
// grid (m, nq), block KSUB, dynamic smem = dsub floats. Thread c owns centroid c.
__global__ void lut_kernel(const float* __restrict__ queries, const float* __restrict__ centroids, int d, int m,
                           int dsub, float* __restrict__ lut) {
    extern __shared__ float qsub[];
    const int s = blockIdx.x, q = blockIdx.y, c = threadIdx.x;
    for (int j = threadIdx.x; j < dsub; j += blockDim.x) qsub[j] = queries[(size_t)q * d + s * dsub + j];
    __syncthreads();
    const float* cent = centroids + ((size_t)s * KSUB + c) * dsub;
    float acc = 0.f;
    for (int j = 0; j < dsub; ++j) {
        const float diff = qsub[j] - cent[j];
        acc += diff * diff;
    }
    lut[((size_t)q * m + s) * KSUB + c] = acc;
}

__device__ inline void load_lut_to_smem(const float* __restrict__ lut, int m, float* sLut) {
    const float* src = lut + (size_t)blockIdx.y * m * KSUB;
    for (int j = threadIdx.x; j < m * KSUB; j += blockDim.x) sLut[j] = src[j];
    __syncthreads();
}

// ---------------------------------------------------------------- scans
// All scans: grid (blocks_x, nq), grid-stride over the n database vectors, one thread per vector.

// v0: every lookup goes to global memory (cached in L1/L2 at best).
__global__ void scan_global_lut(const float* __restrict__ lut, const uint8_t* __restrict__ codes, size_t n, int m,
                                float* __restrict__ dists) {
    const float* table = lut + (size_t)blockIdx.y * m * KSUB;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint8_t* code = codes + i * m;
        float acc = 0.f;
        for (int s = 0; s < m; ++s) acc += table[s * KSUB + code[s]];
        dists[(size_t)blockIdx.y * n + i] = acc;
    }
}

// v1: LUT in shared memory; code loads are still byte-wide and strided by m across a warp.
__global__ void scan_shared_lut(const float* __restrict__ lut, const uint8_t* __restrict__ codes, size_t n, int m,
                                float* __restrict__ dists) {
    extern __shared__ float sLut[];
    load_lut_to_smem(lut, m, sLut);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint8_t* code = codes + i * m;
        float acc = 0.f;
        for (int s = 0; s < m; ++s) acc += sLut[s * KSUB + code[s]];
        dists[(size_t)blockIdx.y * n + i] = acc;
    }
}

__device__ inline float sum4(uint32_t word, const float* sLut) {
    return sLut[word & 0xff] + sLut[KSUB + ((word >> 8) & 0xff)] + sLut[2 * KSUB + ((word >> 16) & 0xff)] +
           sLut[3 * KSUB + (word >> 24)];
}

// v2: each thread reads its code row 16 bytes at a time. Requires m % 16 == 0 (rows stay 16-byte aligned).
__global__ void scan_shared_lut_vec(const float* __restrict__ lut, const uint8_t* __restrict__ codes, size_t n,
                                    int m, float* __restrict__ dists) {
    extern __shared__ float sLut[];
    load_lut_to_smem(lut, m, sLut);
    const int words = m / 16;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint4* row = reinterpret_cast<const uint4*>(codes + i * m);
        float acc = 0.f;
        for (int w = 0; w < words; ++w) {
            const uint4 v = row[w];
            const float* t = sLut + w * 16 * KSUB;
            acc += sum4(v.x, t);
            acc += sum4(v.y, t + 4 * KSUB);
            acc += sum4(v.z, t + 8 * KSUB);
            acc += sum4(v.w, t + 12 * KSUB);
        }
        dists[(size_t)blockIdx.y * n + i] = acc;
    }
}

// Interleaved layout: for each group of 32 vectors, sub-code s of all 32 is contiguous:
//   codes_il[((i / 32) * m + s) * 32 + i % 32]
// A warp reading sub-space s touches one 32-byte segment: fully coalesced.
__device__ inline float adc_interleaved(const float* sLut, const uint8_t* __restrict__ codes_il, size_t i, int m) {
    const uint8_t* group = codes_il + (i / WARP) * (size_t)m * WARP + (i % WARP);
    float acc = 0.f;
    for (int s = 0; s < m; ++s) acc += sLut[s * KSUB + group[(size_t)s * WARP]];
    return acc;
}

// v3
__global__ void scan_interleaved(const float* __restrict__ lut, const uint8_t* __restrict__ codes_il, size_t n,
                                 int m, float* __restrict__ dists) {
    extern __shared__ float sLut[];
    load_lut_to_smem(lut, m, sLut);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        dists[(size_t)blockIdx.y * n + i] = adc_interleaved(sLut, codes_il, i, m);
}

// ---------------------------------------------------------------- k-selection
// Baseline design: each thread keeps a sorted top-K list in registers/local memory, then the block
// merges lists pairwise in shared memory (log2(THREADS) rounds). Padding is (FLT_MAX, -1).
// Replacing this with a warp-select (FAISS-style) is a planned optimisation.

__device__ inline bool nb_less(float da, int ia, float db, int ib) { return da < db || (da == db && ia < ib); }

template <int K>
__device__ inline void list_init(float (&kd)[K], int (&ki)[K]) {
#pragma unroll
    for (int j = 0; j < K; ++j) {
        kd[j] = FLT_MAX;
        ki[j] = -1;
    }
}

template <int K>
__device__ inline void list_insert(float (&kd)[K], int (&ki)[K], float d, int id) {
    if (!nb_less(d, id, kd[K - 1], ki[K - 1])) return;
    int j = K - 1;
    while (j > 0 && nb_less(d, id, kd[j - 1], ki[j - 1])) {
        kd[j] = kd[j - 1];
        ki[j] = ki[j - 1];
        --j;
    }
    kd[j] = d;
    ki[j] = id;
}

// Merge sorted list b into sorted list a (both length K in shared memory), keeping the K smallest.
template <int K>
__device__ inline void merge_into(float* ad, int* ai, const float* bd, const int* bi) {
    float od[K];
    int oi[K];
    int a = 0, b = 0;
    for (int j = 0; j < K; ++j) {
        const bool take_a = b >= K || (a < K && nb_less(ad[a], ai[a], bd[b], bi[b]));
        if (take_a) {
            od[j] = ad[a];
            oi[j] = ai[a++];
        } else {
            od[j] = bd[b];
            oi[j] = bi[b++];
        }
    }
    for (int j = 0; j < K; ++j) {
        ad[j] = od[j];
        ai[j] = oi[j];
    }
}

// After this, md[0..K) / mi[0..K) hold the block's best K. Must be called by all THREADS threads.
template <int K, int THREADS>
__device__ inline void block_merge(const float (&kd)[K], const int (&ki)[K], float* md, int* mi) {
    const int t = threadIdx.x;
#pragma unroll  // constant indices keep register-resident lists (v5+) in registers
    for (int j = 0; j < K; ++j) {
        md[t * K + j] = kd[j];
        mi[t * K + j] = ki[j];
    }
    __syncthreads();
    for (int active = THREADS / 2; active > 0; active >>= 1) {
        if (t < active) merge_into<K>(md + t * K, mi + t * K, md + (t + active) * K, mi + (t + active) * K);
        __syncthreads();
    }
}

// grid (nq), block THREADS. Selects k from rows of length len; ids == nullptr means ids are column indices.
template <int K, int THREADS>
__global__ void __launch_bounds__(THREADS)
    select_kernel(const float* __restrict__ dists, const int* __restrict__ ids, size_t len, int k,
                  float* __restrict__ out_d, int* __restrict__ out_i) {
    extern __shared__ __align__(16) unsigned char smem[];
    float* md = reinterpret_cast<float*>(smem);
    int* mi = reinterpret_cast<int*>(md + THREADS * K);
    const size_t q = blockIdx.x;
    const float* row = dists + q * len;
    const int* row_ids = ids ? ids + q * len : nullptr;

    float kd[K];
    int ki[K];
    list_init<K>(kd, ki);
    for (size_t j = threadIdx.x; j < len; j += THREADS) {
        const int id = row_ids ? row_ids[j] : static_cast<int>(j);
        if (id >= 0) list_insert<K>(kd, ki, row[j], id);
    }
    block_merge<K, THREADS>(kd, ki, md, mi);
    for (int j = threadIdx.x; j < k; j += THREADS) {
        out_d[q * k + j] = md[j];
        out_i[q * k + j] = mi[j];
    }
}

// v4: grid (blocks_x, nq), block THREADS. Scans with the interleaved layout and keeps a per-thread top-K,
// so the nq x n distance matrix is never written. Each block emits K partial candidates; select_kernel
// then reduces the blocks_x * K partials per query. Shared memory holds the LUT, then is reused for merging.
template <int K, int THREADS>
__global__ void __launch_bounds__(THREADS)
    fused_scan_select(const float* __restrict__ lut, const uint8_t* __restrict__ codes_il, size_t n, int m,
                      float* __restrict__ part_d, int* __restrict__ part_i) {
    extern __shared__ __align__(16) unsigned char smem[];
    float* sLut = reinterpret_cast<float*>(smem);
    load_lut_to_smem(lut, m, sLut);

    float kd[K];
    int ki[K];
    list_init<K>(kd, ki);
    for (size_t i = (size_t)blockIdx.x * THREADS + threadIdx.x; i < n; i += (size_t)gridDim.x * THREADS)
        list_insert<K>(kd, ki, adc_interleaved(sLut, codes_il, i, m), static_cast<int>(i));
    __syncthreads();  // everyone is done with the LUT before smem is reused

    float* md = reinterpret_cast<float*>(smem);
    int* mi = reinterpret_cast<int*>(md + THREADS * K);
    block_merge<K, THREADS>(kd, ki, md, mi);
    const size_t base = ((size_t)blockIdx.y * gridDim.x + blockIdx.x) * K;
    for (int j = threadIdx.x; j < K; j += THREADS) {
        part_d[base + j] = md[j];
        part_i[base + j] = mi[j];
    }
}

// ================================================================ v5 - v7
// One kernel template with three switches, so each optimisation can be measured on its own:
//   v5 = <K, T, 1, false>: v4 + register-resident top-K (M2)
//   v6 = <K, T, 1, true> : v5 + 16-byte-word interleaved codes (M1)
//   v7 = <K, T, Q, true> : v6 + Q queries per block sharing every code load (M3)

// 16-byte-word interleaved layout (m % 16 == 0), as uint4 words:
//   codes_v[((i / 32) * (m / 16) + w) * 32 + i % 32] holds sub-codes 16w .. 16w+15 of vector i.
// A warp loading word w reads 512 contiguous bytes, 16 per thread: coalesced AND wide.

// Insertion with only compile-time indices (fully unrolled), so the list can live in registers.
// v4's `while` loop indexes with a run-time value, which forces the list into local memory.
template <int K>
__device__ __forceinline__ void reg_insert(float (&kd)[K], int (&ki)[K], float d, int id) {
    if (!nb_less(d, id, kd[K - 1], ki[K - 1])) return;  // the common case: one compare
    kd[K - 1] = d;
    ki[K - 1] = id;
#pragma unroll
    for (int j = K - 1; j > 0; --j) {  // bubble the new entry towards the front
        if (nb_less(kd[j], ki[j], kd[j - 1], ki[j - 1])) {
            const float td = kd[j];
            kd[j] = kd[j - 1];
            kd[j - 1] = td;
            const int ti = ki[j];
            ki[j] = ki[j - 1];
            ki[j - 1] = ti;
        }
    }
}

// ADC distances of vector i to Q queries whose LUTs sit back to back in shared memory.
template <int Q, bool VEC>
__device__ __forceinline__ void adc_multi(const float* sLut, int m, const uint8_t* __restrict__ codes_il,
                                          const uint4* __restrict__ codes_v, size_t i, float (&acc)[Q]) {
#pragma unroll
    for (int q = 0; q < Q; ++q) acc[q] = 0.f;
    if constexpr (VEC) {
        const int words = m / 16;
        const uint4* p = codes_v + (i / WARP) * (size_t)words * WARP + (i % WARP);
        for (int w = 0; w < words; ++w) {
            const uint4 v = p[(size_t)w * WARP];  // one 16-byte load serves all Q queries
#pragma unroll
            for (int q = 0; q < Q; ++q) {
                const float* t = sLut + (size_t)q * m * KSUB + w * 16 * KSUB;
                acc[q] += sum4(v.x, t);
                acc[q] += sum4(v.y, t + 4 * KSUB);
                acc[q] += sum4(v.z, t + 8 * KSUB);
                acc[q] += sum4(v.w, t + 12 * KSUB);
            }
        }
    } else {
        const uint8_t* g = codes_il + (i / WARP) * (size_t)m * WARP + (i % WARP);
        for (int s = 0; s < m; ++s) {
            const int c = g[(size_t)s * WARP];
#pragma unroll
            for (int q = 0; q < Q; ++q) acc[q] += sLut[(size_t)q * m * KSUB + s * KSUB + c];
        }
    }
}

// ILP version of the word-interleaved scan: each thread handles ILP vectors (i, i + stride, ...) per step and
// issues all their 16-byte loads before any lookup, so independent work hides shared-memory and L2 latency.
template <int Q, int ILP>
__device__ __forceinline__ void adc_vec_ilp(const float* sLut, int m, const uint4* __restrict__ codes_v, size_t i0,
                                            size_t stride, size_t n, float (&acc)[ILP][Q]) {
    const int words = m / 16;
#pragma unroll
    for (int u = 0; u < ILP; ++u)
#pragma unroll
        for (int q = 0; q < Q; ++q) acc[u][q] = 0.f;
    for (int w = 0; w < words; ++w) {
        uint4 v[ILP];
#pragma unroll
        for (int u = 0; u < ILP; ++u) {
            const size_t i = i0 + u * stride;  // vectors past n read padding (zeros) and are discarded later
            v[u] = i < n ? codes_v[((i / WARP) * (size_t)words + w) * WARP + (i % WARP)] : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int u = 0; u < ILP; ++u)
#pragma unroll
            for (int q = 0; q < Q; ++q) {
                const float* t = sLut + (size_t)q * m * KSUB + w * 16 * KSUB;
                acc[u][q] += sum4(v[u].x, t);
                acc[u][q] += sum4(v[u].y, t + 4 * KSUB);
                acc[u][q] += sum4(v[u].z, t + 8 * KSUB);
                acc[u][q] += sum4(v[u].w, t + 12 * KSUB);
            }
    }
}

// grid (blocks_x, ceil(nq / Q)), block T. Block row y serves queries yQ .. yQ+Q-1 (the LUT buffer is padded
// to a multiple of Q; padded queries are computed but never written). Partials as in v4.
// ILP > 1 is only used with VEC (the word-interleaved layout).
template <int K, int T, int Q, bool VEC, int ILP = 1>
__global__ void __launch_bounds__(T)
    fused_v2(const float* __restrict__ lut, const uint8_t* __restrict__ codes_il, const uint4* __restrict__ codes_v,
             size_t n, int m, int nq, float* __restrict__ part_d, int* __restrict__ part_i) {
    extern __shared__ __align__(16) unsigned char smem[];
    float* sLut = reinterpret_cast<float*>(smem);
    const size_t q0 = (size_t)blockIdx.y * Q;
    {
        const float* src = lut + q0 * m * KSUB;  // Q consecutive LUTs
        for (int j = threadIdx.x; j < Q * m * KSUB; j += T) sLut[j] = src[j];
        __syncthreads();
    }

    float kd[Q][K];
    int ki[Q][K];
#pragma unroll
    for (int q = 0; q < Q; ++q) list_init<K>(kd[q], ki[q]);

    if constexpr (ILP == 1) {
        for (size_t i = (size_t)blockIdx.x * T + threadIdx.x; i < n; i += (size_t)gridDim.x * T) {
            float acc[Q];
            adc_multi<Q, VEC>(sLut, m, codes_il, codes_v, i, acc);
#pragma unroll
            for (int q = 0; q < Q; ++q) reg_insert<K>(kd[q], ki[q], acc[q], static_cast<int>(i));
        }
    } else {
        static_assert(VEC, "ILP > 1 requires the word-interleaved layout");
        const size_t stride = (size_t)gridDim.x * T;
        for (size_t i0 = (size_t)blockIdx.x * T + threadIdx.x; i0 < n; i0 += stride * ILP) {
            float acc[ILP][Q];
            adc_vec_ilp<Q, ILP>(sLut, m, codes_v, i0, stride, n, acc);
#pragma unroll
            for (int u = 0; u < ILP; ++u) {
                const size_t i = i0 + u * stride;
                if (i < n) {
#pragma unroll
                    for (int q = 0; q < Q; ++q) reg_insert<K>(kd[q], ki[q], acc[u][q], static_cast<int>(i));
                }
            }
        }
    }
    __syncthreads();  // all threads are done with the LUTs before shared memory is reused

    float* md = reinterpret_cast<float*>(smem);
    int* mi = reinterpret_cast<int*>(md + T * K);
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        block_merge<K, T>(kd[q], ki[q], md, mi);
        if (q0 + q < (size_t)nq) {
            const size_t base = ((q0 + q) * gridDim.x + blockIdx.x) * K;
            for (int j = threadIdx.x; j < K; j += T) {
                part_d[base + j] = md[j];
                part_i[base + j] = mi[j];
            }
        }
        __syncthreads();  // the merge buffer is reused by the next query
    }
}

// ================================================================ v8: warp-cooperative top-k
// Each warp owns ONE sorted top-K list distributed over its 32 lanes: segment s, lane l holds entry 32s + l
// (S = ceil(K / 32) registers per lane per list, instead of K per thread in v5-v7). Every lane keeps the
// warp's K-th best (the admission threshold) in a register. A new candidate below the threshold is broadcast
// with a shuffle and inserted by all lanes at once: each lane takes its own entry, its left neighbour's, or
// the candidate, and the entry pushed out of lane 31 carries into the next segment. All calls are warp-uniform.
constexpr unsigned kFullMask = 0xffffffffu;

template <int K>
struct WarpTopK {
    static constexpr int S = (K + WARP - 1) / WARP;
    float d[S];
    int id[S];
    float thr_d;  // entry K-1: the admission threshold
    int thr_i;

    __device__ __forceinline__ void init() {
#pragma unroll
        for (int s = 0; s < S; ++s) {
            d[s] = FLT_MAX;
            id[s] = -1;
        }
        thr_d = FLT_MAX;
        thr_i = -1;
    }

    __device__ __forceinline__ bool admits(float x, int xi) const { return nb_less(x, xi, thr_d, thr_i); }

    // Must be called by all 32 lanes with the same (x, xi).
    __device__ __forceinline__ void insert(float x, int xi) {
        const int lane = threadIdx.x % WARP;
        float cd = x;
        int ci = xi;
#pragma unroll
        for (int s = 0; s < S; ++s) {
            const float pd = __shfl_up_sync(kFullMask, d[s], 1);
            const int pi = __shfl_up_sync(kFullMask, id[s], 1);
            const float last_d = __shfl_sync(kFullMask, d[s], WARP - 1);
            const int last_i = __shfl_sync(kFullMask, id[s], WARP - 1);
            const bool keep = nb_less(d[s], id[s], cd, ci);               // my entry stays where it is
            const bool take = lane == 0 || nb_less(pd, pi, cd, ci);       // the candidate lands here
            const float nd = keep ? d[s] : (take ? cd : pd);
            const int ni = keep ? id[s] : (take ? ci : pi);
            const bool inserted = nb_less(cd, ci, last_d, last_i);        // uniform: same values in every lane
            d[s] = nd;
            id[s] = ni;
            if (inserted) {  // the old last entry of this segment moves on to the next one
                cd = last_d;
                ci = last_i;
            }
        }
        if constexpr (K < WARP) {  // lanes past K are not part of the list
            if (lane >= K) {
                d[0] = FLT_MAX;
                id[0] = -1;
            }
        }
        thr_d = __shfl_sync(kFullMask, d[(K - 1) / WARP], (K - 1) % WARP);
        thr_i = __shfl_sync(kFullMask, id[(K - 1) / WARP], (K - 1) % WARP);
    }

    // Offer one value per lane; admitted values are inserted one at a time (rare after the list fills up).
    __device__ __forceinline__ void offer(float x, int xi, bool valid) {
        unsigned mask = __ballot_sync(kFullMask, valid && admits(x, xi));
        while (mask) {
            const int src = __ffs(mask) - 1;
            mask &= mask - 1;
            const float vx = __shfl_sync(kFullMask, x, src);
            const int vi = __shfl_sync(kFullMask, xi, src);
            if (admits(vx, vi)) insert(vx, vi);  // re-check: the threshold may have dropped meanwhile
        }
    }

    __device__ __forceinline__ void store(float* md, int* mi) const {  // K entries, in order
        const int lane = threadIdx.x % WARP;
#pragma unroll
        for (int s = 0; s < S; ++s)
            if (s * WARP + lane < K) {
                md[s * WARP + lane] = d[s];
                mi[s * WARP + lane] = id[s];
            }
    }
};

// grid (blocks_x, ceil(nq / Q)), block T (multiple of 32). Word-interleaved codes, ILP vectors per lane per
// step, Q queries per block, one WarpTopK per (warp, query). The warp lists are then merged in shared memory
// (NW = T / 32 lists of K: a few hundred bytes instead of T * K * 8).
template <int K, int T, int Q, int ILP>
__global__ void __launch_bounds__(T)
    fused_warp(const float* __restrict__ lut, const uint4* __restrict__ codes_v, size_t n, int m, int nq,
               float* __restrict__ part_d, int* __restrict__ part_i) {
    static_assert(T % WARP == 0, "T must be a multiple of the warp size");
    constexpr int NW = T / WARP;
    extern __shared__ __align__(16) unsigned char smem[];
    float* sLut = reinterpret_cast<float*>(smem);
    const size_t q0 = (size_t)blockIdx.y * Q;
    {
        const float* src = lut + q0 * m * KSUB;
        for (int j = threadIdx.x; j < Q * m * KSUB; j += T) sLut[j] = src[j];
        __syncthreads();
    }
    WarpTopK<K> top[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) top[q].init();

    const int lane = threadIdx.x % WARP, warp = threadIdx.x / WARP;
    const size_t stride = (size_t)gridDim.x * T;
    // Warp-uniform loop bound (base is the warp's first vector), so every lane joins every ballot and shuffle.
    for (size_t base = (size_t)blockIdx.x * T + warp * WARP; base < n; base += stride * ILP) {
        const size_t i0 = base + lane;
        float acc[ILP][Q];
        adc_vec_ilp<Q, ILP>(sLut, m, codes_v, i0, stride, n, acc);
#pragma unroll
        for (int u = 0; u < ILP; ++u) {
            const size_t i = i0 + u * stride;
#pragma unroll
            for (int q = 0; q < Q; ++q) top[q].offer(acc[u][q], static_cast<int>(i), i < n);
        }
    }
    __syncthreads();  // all warps are done with the LUTs before shared memory is reused

    float* md = reinterpret_cast<float*>(smem);
    int* mi = reinterpret_cast<int*>(md + NW * K);
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        top[q].store(md + warp * K, mi + warp * K);
        __syncthreads();
        for (int active = NW / 2; active > 0; active >>= 1) {  // tree merge of the NW warp lists
            if (threadIdx.x < active)
                merge_into<K>(md + threadIdx.x * K, mi + threadIdx.x * K, md + (threadIdx.x + active) * K,
                              mi + (threadIdx.x + active) * K);
            __syncthreads();
        }
        if (q0 + q < (size_t)nq) {
            const size_t out = ((q0 + q) * gridDim.x + blockIdx.x) * K;
            for (int j = threadIdx.x; j < K; j += T) {
                part_d[out + j] = md[j];
                part_i[out + j] = mi[j];
            }
        }
        __syncthreads();
    }
}

}  // namespace veckern::gpu::kernels
