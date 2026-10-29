// Host orchestration: device buffers, launch configuration, query batching and per-stage timing.
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "kernels.cuh"
#include "veckern/gpu.hpp"

#define VK_CUDA_CHECK(call)                                                                            \
    do {                                                                                               \
        const cudaError_t err_ = (call);                                                               \
        if (err_ != cudaSuccess)                                                                       \
            throw std::runtime_error(std::string(#call) + " failed: " + cudaGetErrorString(err_));     \
    } while (0)

namespace veckern::gpu {
namespace {

namespace k = kernels;

constexpr std::size_t kVectorsPerThread = 16;             // amortises the per-block LUT load
constexpr std::size_t kDistanceBudgetBytes = 512u << 20;  // cap on the unfused nq x n matrix
constexpr std::size_t kMaxGridY = 65535;

template <typename T>
class DeviceBuffer {
public:
    ~DeviceBuffer() { cudaFree(ptr_); }
    void ensure(std::size_t count) {
        if (count <= capacity_) return;
        cudaFree(ptr_);
        ptr_ = nullptr;
        VK_CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        capacity_ = count;
    }
    T* get() const { return ptr_; }
    bool empty() const { return ptr_ == nullptr; }

private:
    T* ptr_ = nullptr;
    std::size_t capacity_ = 0;
};

// Records an event pair around a stage and accumulates its elapsed time.
class StageTimer {
public:
    explicit StageTimer(cudaStream_t stream) : stream_(stream) {
        VK_CUDA_CHECK(cudaEventCreate(&start_));
        VK_CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~StageTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    template <typename F>
    void time(double& acc, F&& work) {
        VK_CUDA_CHECK(cudaEventRecord(start_, stream_));
        work();
        VK_CUDA_CHECK(cudaGetLastError());
        VK_CUDA_CHECK(cudaEventRecord(stop_, stream_));
        VK_CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        VK_CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        acc += ms;
    }

private:
    cudaStream_t stream_;
    cudaEvent_t start_{}, stop_{};
};

// Runs CALL with constexpr K (smallest supported >= k) and T (threads per block) in scope.
// Merge buffers stay at 8-32 KB of shared memory for every case.
#define VK_DISPATCH_K(kk, CALL)                                       \
    do {                                                              \
        if ((kk) <= 8) { constexpr int K = 8, T = 128; CALL; }        \
        else if ((kk) <= 16) { constexpr int K = 16, T = 128; CALL; } \
        else if ((kk) <= 32) { constexpr int K = 32, T = 128; CALL; } \
        else if ((kk) <= 64) { constexpr int K = 64, T = 64; CALL; }  \
        else if ((kk) <= 128) { constexpr int K = 128, T = 32; CALL; } \
        else throw std::invalid_argument("k must be <= 128");         \
    } while (0)

int dispatched_k(std::size_t kk) {
    for (int K : {8, 16, 32, 64, 128})
        if (kk <= static_cast<std::size_t>(K)) return K;
    throw std::invalid_argument("k must be <= 128");
}

template <typename Kernel>
void allow_smem(Kernel kernel, std::size_t bytes, std::size_t limit) {
    check(bytes <= limit, "kernel needs " + std::to_string(bytes) + " B of shared memory; device allows " +
                              std::to_string(limit) + " B (reduce m or queries_per_block)");
    if (bytes > 48 * 1024)
        VK_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
}

std::size_t ceil_div(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

// One-time layout transforms, run on the GPU. Writes are coalesced; reads are strided (fine for a one-off).
__global__ void build_interleaved_bytes(const uint8_t* __restrict__ codes, size_t n, int m, uint8_t* __restrict__ out,
                                        size_t total) {
    for (size_t o = (size_t)blockIdx.x * blockDim.x + threadIdx.x; o < total; o += (size_t)gridDim.x * blockDim.x) {
        const size_t lane = o % k::WARP, s = (o / k::WARP) % m, group = o / (k::WARP * (size_t)m);
        const size_t i = group * k::WARP + lane;
        out[o] = i < n ? codes[i * m + s] : 0;
    }
}

__global__ void build_interleaved_words(const uint4* __restrict__ codes, size_t n, int words, uint4* __restrict__ out,
                                        size_t total) {
    for (size_t o = (size_t)blockIdx.x * blockDim.x + threadIdx.x; o < total; o += (size_t)gridDim.x * blockDim.x) {
        const size_t lane = o % k::WARP, w = (o / k::WARP) % words, group = o / (k::WARP * (size_t)words);
        const size_t i = group * k::WARP + lane;
        out[o] = i < n ? codes[i * words + w] : make_uint4(0, 0, 0, 0);
    }
}

}  // namespace

const char* variant_name(ScanVariant v) {
    switch (v) {
        case ScanVariant::GlobalLut: return "v0_global_lut";
        case ScanVariant::SharedLut: return "v1_shared_lut";
        case ScanVariant::SharedLutVec: return "v2_shared_lut_vec";
        case ScanVariant::Interleaved: return "v3_interleaved";
        case ScanVariant::Fused: return "v4_fused";
        case ScanVariant::FusedReg: return "v5_fused_reg";
        case ScanVariant::FusedVec: return "v6_fused_vec";
        case ScanVariant::FusedVecMQ: return "v7_fused_vec_mq";
        case ScanVariant::FusedWarp: return "v8_fused_warp";
    }
    return "unknown";
}

bool is_fused(ScanVariant v) {
    return v == ScanVariant::Fused || v == ScanVariant::FusedReg || v == ScanVariant::FusedVec ||
           v == ScanVariant::FusedVecMQ || v == ScanVariant::FusedWarp;
}

DeviceInfo device_info(int device) {
    DeviceInfo info;
    cudaDeviceProp prop{};
    VK_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    info.name = prop.name;
    int mem_clock_khz = 0, bus_bits = 0, smem = 0;
    VK_CUDA_CHECK(cudaDeviceGetAttribute(&info.sm_count, cudaDevAttrMultiProcessorCount, device));
    VK_CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device));
    VK_CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, device));
    VK_CUDA_CHECK(cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    info.peak_bandwidth_gbs = 2.0 * mem_clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;  // DDR: two transfers per clock
    info.max_smem_per_block = static_cast<std::size_t>(smem);
    return info;
}

struct PQSearcher::Impl {
    std::size_t n = 0, d = 0, m = 0, dsub = 0;
    DeviceInfo info;
    Tuning tuning;
    LaunchShape last;
    cudaStream_t stream{};
    DeviceBuffer<float> centroids, queries, lut, dists, part_d, out_d;
    DeviceBuffer<int> part_i, out_i;
    DeviceBuffer<uint8_t> codes, codes_il;
    DeviceBuffer<uint4> codes_v;

    // v0-v4 heuristic (kept unchanged so their published numbers stay reproducible): ~8 blocks per SM in
    // total; with many queries the y-dimension already fills the GPU, so each block scans more vectors and loads
    // its LUT fewer times; never split below kVectorsPerThread vectors per thread.
    std::size_t blocks_x_legacy(std::size_t grid_y, int threads) const {
        if (tuning.blocks_per_query > 0) return static_cast<std::size_t>(tuning.blocks_per_query);
        const std::size_t by_work = ceil_div(n, threads * kVectorsPerThread);
        const std::size_t to_fill = ceil_div(8 * static_cast<std::size_t>(info.sm_count), grid_y);
        return std::max<std::size_t>(1, std::min(by_work, to_fill));
    }

    // v5-v7 heuristic: wave-quantisation model. A wave = blocks resident at once, from the occupancy API for
    // this exact kernel, block size and shared memory (so register and shared-memory limits count, not just
    // threads). Every block does equal work, so run time ~ waves / blocks_x; pick the blocks_x that minimises it,
    // taking a larger blocks_x only for a >= 2% gain (more blocks = more LUT loads and partial lists).
    template <typename Kernel>
    std::size_t blocks_x_waves(Kernel kernel, int threads, std::size_t smem, std::size_t grid_y) {
        int per_sm = 0;
        VK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, threads, smem));
        const std::size_t wave = static_cast<std::size_t>(std::max(per_sm, 1)) * info.sm_count;
        last.blocks_per_sm = per_sm;
        std::size_t best = 1;
        if (tuning.blocks_per_query > 0) {
            best = static_cast<std::size_t>(tuning.blocks_per_query);
        } else {
            // Cap: 64 blocks per query row, raised for small batches so that even one query can fill the GPU
            // (up to ~4 waves' worth). A fixed cap of 64 left a single query with 63 blocks on 108 SMs.
            const std::size_t cap = std::max<std::size_t>(64, 4 * ceil_div(wave, grid_y));
            const std::size_t max_bx = std::max<std::size_t>(1, std::min(cap, ceil_div(n, threads * kVectorsPerThread)));
            double best_cost = static_cast<double>(ceil_div(grid_y, wave));
            for (std::size_t bx = 2; bx <= max_bx; ++bx) {
                const double cost = static_cast<double>(ceil_div(bx * grid_y, wave)) / bx;
                if (cost < best_cost * 0.98) {
                    best_cost = cost;
                    best = bx;
                }
            }
        }
        last.waves = static_cast<double>(best * grid_y) / wave;
        return best;
    }

    // Queries per block for v7: the requested Q, halved until the per-thread lists (Q * K entries) stay
    // register-sized and the Q lookup tables fit in shared memory.
    // v8 keeps lists in a few registers per lane, so only shared memory limits its Q.
    int effective_q(std::size_t kk, ScanVariant v) const {
        const int K = dispatched_k(kk);
        int q = tuning.queries_per_block;
        const bool per_thread_lists = v != ScanVariant::FusedWarp;
        while (q > 1 && ((per_thread_lists && q * K > 64) || q * m * k::KSUB * sizeof(float) > info.max_smem_per_block))
            q /= 2;
        return q;
    }

    void ensure_layout(ScanVariant v) {
        const unsigned blocks = static_cast<unsigned>(4 * info.sm_count);
        if ((v == ScanVariant::Interleaved || v == ScanVariant::Fused || v == ScanVariant::FusedReg) &&
            codes_il.empty()) {
            const std::size_t total = ceil_div(n, k::WARP) * k::WARP * m;
            codes_il.ensure(total);
            build_interleaved_bytes<<<blocks, 256, 0, stream>>>(codes.get(), n, static_cast<int>(m), codes_il.get(),
                                                                total);
            VK_CUDA_CHECK(cudaGetLastError());
        }
        if ((v == ScanVariant::FusedVec || v == ScanVariant::FusedVecMQ || v == ScanVariant::FusedWarp) &&
            codes_v.empty()) {
            const std::size_t words = m / 16, total = ceil_div(n, k::WARP) * k::WARP * words;
            codes_v.ensure(total);
            build_interleaved_words<<<blocks, 256, 0, stream>>>(reinterpret_cast<const uint4*>(codes.get()), n,
                                                                static_cast<int>(words), codes_v.get(), total);
            VK_CUDA_CHECK(cudaGetLastError());
        }
        VK_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void validate(ScanVariant v) const {
        if (v == ScanVariant::SharedLutVec || v == ScanVariant::FusedVec || v == ScanVariant::FusedVecMQ ||
            v == ScanVariant::FusedWarp)
            check(m % 16 == 0, std::string(variant_name(v)) + " requires m % 16 == 0");
        if (v != ScanVariant::GlobalLut)
            check(m * k::KSUB * sizeof(float) <= info.max_smem_per_block, "LUT does not fit in shared memory");
    }

    void launch_lut(std::size_t nq_batch) {
        const dim3 grid(static_cast<unsigned>(m), static_cast<unsigned>(nq_batch));
        k::lut_kernel<<<grid, k::KSUB, dsub * sizeof(float), stream>>>(
            queries.get(), centroids.get(), static_cast<int>(d), static_cast<int>(m), static_cast<int>(dsub), lut.get());
    }

    void launch_scan(ScanVariant v, std::size_t nq_batch) {
        const int threads = tuning.scan_threads;
        const std::size_t bx = blocks_x_legacy(nq_batch, threads);
        last = {threads, static_cast<int>(bx), static_cast<int>(nq_batch), 1};
        const dim3 grid(static_cast<unsigned>(bx), static_cast<unsigned>(nq_batch));
        const std::size_t smem = m * k::KSUB * sizeof(float);
        const int mi = static_cast<int>(m);
        switch (v) {
            case ScanVariant::GlobalLut:
                k::scan_global_lut<<<grid, threads, 0, stream>>>(lut.get(), codes.get(), n, mi, dists.get());
                break;
            case ScanVariant::SharedLut:
                allow_smem(k::scan_shared_lut, smem, info.max_smem_per_block);
                k::scan_shared_lut<<<grid, threads, smem, stream>>>(lut.get(), codes.get(), n, mi, dists.get());
                break;
            case ScanVariant::SharedLutVec:
                allow_smem(k::scan_shared_lut_vec, smem, info.max_smem_per_block);
                k::scan_shared_lut_vec<<<grid, threads, smem, stream>>>(lut.get(), codes.get(), n, mi, dists.get());
                break;
            case ScanVariant::Interleaved:
                allow_smem(k::scan_interleaved, smem, info.max_smem_per_block);
                k::scan_interleaved<<<grid, threads, smem, stream>>>(lut.get(), codes_il.get(), n, mi, dists.get());
                break;
            default:
                throw std::logic_error("fused variants have no standalone scan");
        }
    }

    template <int K, int T>
    void select_impl(const float* rows, const int* ids, std::size_t len, std::size_t kk, std::size_t nq_batch) {
        const std::size_t smem = static_cast<std::size_t>(T) * K * (sizeof(float) + sizeof(int));
        allow_smem(k::select_kernel<K, T>, smem, info.max_smem_per_block);
        k::select_kernel<K, T><<<static_cast<unsigned>(nq_batch), T, smem, stream>>>(
            rows, ids, len, static_cast<int>(kk), out_d.get(), out_i.get());
    }

    // Selects k per query from rows of `len` candidates into out_d / out_i.
    void launch_select(const float* rows, const int* ids, std::size_t len, std::size_t kk, std::size_t nq_batch) {
        VK_DISPATCH_K(kk, (select_impl<K, T>(rows, ids, len, kk, nq_batch)));
    }

    // ---- v4
    template <int K, int T>
    std::size_t fused_v4_impl(std::size_t nq_batch) {
        const std::size_t bx = std::max<std::size_t>(1, std::min(blocks_x_legacy(nq_batch, 256), ceil_div(n, T)));
        last = {T, static_cast<int>(bx), static_cast<int>(nq_batch), 1};
        const std::size_t per_query = bx * K;
        part_d.ensure(nq_batch * per_query);
        part_i.ensure(nq_batch * per_query);
        const std::size_t smem = std::max(m * k::KSUB * sizeof(float), static_cast<std::size_t>(T) * K * 8);
        allow_smem(k::fused_scan_select<K, T>, smem, info.max_smem_per_block);
        const dim3 grid(static_cast<unsigned>(bx), static_cast<unsigned>(nq_batch));
        k::fused_scan_select<K, T><<<grid, T, smem, stream>>>(lut.get(), codes_il.get(), n, static_cast<int>(m),
                                                            part_d.get(), part_i.get());
        return per_query;
    }

    // ---- v5-v7
    template <int K, int T, int Q, bool VEC, int ILP = 1>
    std::size_t fused_v2_launch(std::size_t nq_batch) {
        const std::size_t grid_y = ceil_div(nq_batch, Q);
        const std::size_t smem = std::max(Q * m * k::KSUB * sizeof(float), static_cast<std::size_t>(T) * K * 8);
        allow_smem(k::fused_v2<K, T, Q, VEC, ILP>, smem, info.max_smem_per_block);
        last = LaunchShape{};
        const std::size_t bx = std::max<std::size_t>(
            1, std::min(blocks_x_waves(k::fused_v2<K, T, Q, VEC, ILP>, T, smem, grid_y), ceil_div(n, T)));
        last.threads = T;
        last.blocks_x = static_cast<int>(bx);
        last.grid_y = static_cast<int>(grid_y);
        last.queries_per_block = Q;
        last.vectors_per_step = ILP;
        const std::size_t per_query = bx * K;
        part_d.ensure(nq_batch * per_query);
        part_i.ensure(nq_batch * per_query);
        const dim3 grid(static_cast<unsigned>(bx), static_cast<unsigned>(grid_y));
        k::fused_v2<K, T, Q, VEC, ILP><<<grid, T, smem, stream>>>(lut.get(), codes_il.get(), codes_v.get(), n,
                                                           static_cast<int>(m), static_cast<int>(nq_batch),
                                                           part_d.get(), part_i.get());
        return per_query;
    }

    // Word-interleaved launch with the tuned ILP (vectors per thread per loop step).
    template <int K, int T, int Q>
    std::size_t fused_vec_launch(std::size_t nq_batch) {
        switch (tuning.vectors_per_step) {
            case 4: return fused_v2_launch<K, T, Q, true, 4>(nq_batch);
            case 2: return fused_v2_launch<K, T, Q, true, 2>(nq_batch);
            default: return fused_v2_launch<K, T, Q, true, 1>(nq_batch);
        }
    }

    template <int K, int T>
    std::size_t fused_v2_impl(ScanVariant v, int q, std::size_t nq_batch) {
        if (v == ScanVariant::FusedReg) return fused_v2_launch<K, T, 1, false>(nq_batch);
        if (v == ScanVariant::FusedVec || q <= 1) return fused_vec_launch<K, T, 1>(nq_batch);
        if constexpr (K <= 8) {
            if (q >= 8) return fused_vec_launch<K, T, 8>(nq_batch);
        }
        if constexpr (K <= 16) {
            if (q >= 4) return fused_vec_launch<K, T, 4>(nq_batch);
        }
        if constexpr (K <= 32) {
            return fused_vec_launch<K, T, 2>(nq_batch);
        }
        return fused_vec_launch<K, T, 1>(nq_batch);
    }

    // ---- v8
    template <int K, int Q, int ILP>
    std::size_t fused_warp_launch(std::size_t nq_batch) {
        constexpr int T = 256;  // the merge buffer is (T / 32) * K * 8 bytes, so wide blocks are cheap here
        const std::size_t grid_y = ceil_div(nq_batch, Q);
        const std::size_t smem = std::max(Q * m * k::KSUB * sizeof(float), static_cast<std::size_t>(T / 32) * K * 8);
        allow_smem(k::fused_warp<K, T, Q, ILP>, smem, info.max_smem_per_block);
        last = LaunchShape{};
        const std::size_t bx =
            std::max<std::size_t>(1, std::min(blocks_x_waves(k::fused_warp<K, T, Q, ILP>, T, smem, grid_y), ceil_div(n, T)));
        last.threads = T;
        last.blocks_x = static_cast<int>(bx);
        last.grid_y = static_cast<int>(grid_y);
        last.queries_per_block = Q;
        last.vectors_per_step = ILP;
        const std::size_t per_query = bx * K;
        part_d.ensure(nq_batch * per_query);
        part_i.ensure(nq_batch * per_query);
        const dim3 grid(static_cast<unsigned>(bx), static_cast<unsigned>(grid_y));
        k::fused_warp<K, T, Q, ILP><<<grid, T, smem, stream>>>(lut.get(), codes_v.get(), n, static_cast<int>(m),
                                                            static_cast<int>(nq_batch), part_d.get(), part_i.get());
        return per_query;
    }

    template <int K, int Q>
    std::size_t fused_warp_ilp(std::size_t nq_batch) {
        switch (tuning.vectors_per_step) {
            case 4: return fused_warp_launch<K, Q, 4>(nq_batch);
            case 2: return fused_warp_launch<K, Q, 2>(nq_batch);
            default: return fused_warp_launch<K, Q, 1>(nq_batch);
        }
    }

    template <int K>
    std::size_t fused_warp_q(int q, std::size_t nq_batch) {
        switch (q) {
            case 8: return fused_warp_ilp<K, 8>(nq_batch);
            case 4: return fused_warp_ilp<K, 4>(nq_batch);
            case 2: return fused_warp_ilp<K, 2>(nq_batch);
            default: return fused_warp_ilp<K, 1>(nq_batch);
        }
    }

    std::size_t launch_fused_warp(std::size_t kk, int q, std::size_t nq_batch) {
        switch (dispatched_k(kk)) {
            case 8: return fused_warp_q<8>(q, nq_batch);
            case 16: return fused_warp_q<16>(q, nq_batch);
            case 32: return fused_warp_q<32>(q, nq_batch);
            case 64: return fused_warp_q<64>(q, nq_batch);
            default: return fused_warp_q<128>(q, nq_batch);
        }
    }

    // Returns the number of partial candidates per query written to part_d / part_i.
    std::size_t launch_fused(ScanVariant v, std::size_t kk, int q, std::size_t nq_batch) {
        if (v == ScanVariant::FusedWarp) return launch_fused_warp(kk, q, nq_batch);
        std::size_t per_query = 0;
        if (v == ScanVariant::Fused) VK_DISPATCH_K(kk, (per_query = fused_v4_impl<K, T>(nq_batch)));
        else VK_DISPATCH_K(kk, (per_query = fused_v2_impl<K, T>(v, q, nq_batch)));
        return per_query;
    }
};

PQSearcher::PQSearcher(const ProductQuantizer& pq, const uint8_t* host_codes, std::size_t n)
    : impl_(std::make_unique<Impl>()) {
    auto& s = *impl_;
    check(n > 0 && n < static_cast<std::size_t>(std::numeric_limits<int>::max()), "n must fit in int32 ids");
    s.n = n;
    s.d = pq.d();
    s.m = pq.m();
    s.dsub = pq.dsub();
    s.info = device_info();
    VK_CUDA_CHECK(cudaStreamCreate(&s.stream));

    s.centroids.ensure(pq.centroids().size());
    VK_CUDA_CHECK(cudaMemcpy(s.centroids.get(), pq.centroids().data(), pq.centroids().size() * sizeof(float),
                             cudaMemcpyHostToDevice));
    s.codes.ensure(n * s.m);
    VK_CUDA_CHECK(cudaMemcpy(s.codes.get(), host_codes, n * s.m, cudaMemcpyHostToDevice));
    // The interleaved layouts are built on the GPU the first time a variant needs them (ensure_layout).
}

PQSearcher::~PQSearcher() {
    if (impl_) cudaStreamDestroy(impl_->stream);
}

void PQSearcher::set_tuning(const Tuning& t) {
    check(t.scan_threads >= 32 && t.scan_threads <= 1024 && t.scan_threads % 32 == 0,
          "scan_threads must be a multiple of 32 in [32, 1024]");
    check(t.blocks_per_query >= 0, "blocks_per_query must be >= 0");
    check(t.queries_per_block == 1 || t.queries_per_block == 2 || t.queries_per_block == 4 ||
              t.queries_per_block == 8,
          "queries_per_block must be 1, 2, 4 or 8");
    check(t.vectors_per_step == 1 || t.vectors_per_step == 2 || t.vectors_per_step == 4,
          "vectors_per_step must be 1, 2 or 4");
    impl_->tuning = t;
}

const Tuning& PQSearcher::tuning() const { return impl_->tuning; }

PQSearcher::LaunchShape PQSearcher::last_launch() const { return impl_->last; }

Timings PQSearcher::search(const float* host_queries, std::size_t nq, std::size_t kk, ScanVariant v, Neighbor* out) {
    auto& s = *impl_;
    check(kk >= 1 && kk <= 128 && kk <= s.n, "k must be in [1, min(128, n)]");
    s.validate(v);
    s.ensure_layout(v);
    Timings t;
    StageTimer timer(s.stream);

    const bool fused = is_fused(v);
    const int q = (v == ScanVariant::FusedVecMQ || v == ScanVariant::FusedWarp) ? s.effective_q(kk, v) : 1;
    const std::size_t batch =
        fused ? kMaxGridY : std::max<std::size_t>(1, std::min(kMaxGridY, kDistanceBudgetBytes / (s.n * 4)));
    std::vector<float> host_d;
    std::vector<int> host_i;
    for (std::size_t q0 = 0; q0 < nq; q0 += batch) {
        const std::size_t b = std::min(batch, nq - q0);
        const std::size_t b_pad = ceil_div(b, q) * q;  // v7 reads LUTs in groups of q
        s.queries.ensure(b * s.d);
        s.lut.ensure(b_pad * s.m * k::KSUB);
        s.out_d.ensure(b * kk);
        s.out_i.ensure(b * kk);
        if (b_pad > b)  // padded queries are computed and discarded; give them defined input
            VK_CUDA_CHECK(cudaMemsetAsync(s.lut.get() + b * s.m * k::KSUB, 0,
                                          (b_pad - b) * s.m * k::KSUB * sizeof(float), s.stream));

        timer.time(t.transfer_ms, [&] {
            VK_CUDA_CHECK(cudaMemcpyAsync(s.queries.get(), host_queries + q0 * s.d, b * s.d * sizeof(float),
                                          cudaMemcpyHostToDevice, s.stream));
        });
        timer.time(t.lut_ms, [&] { s.launch_lut(b); });
        if (fused) {
            std::size_t per_query = 0;
            timer.time(t.scan_ms, [&] { per_query = s.launch_fused(v, kk, q, b); });
            timer.time(t.select_ms, [&] { s.launch_select(s.part_d.get(), s.part_i.get(), per_query, kk, b); });
        } else {
            s.dists.ensure(b * s.n);
            timer.time(t.scan_ms, [&] { s.launch_scan(v, b); });
            timer.time(t.select_ms, [&] { s.launch_select(s.dists.get(), nullptr, s.n, kk, b); });
        }

        host_d.resize(b * kk);
        host_i.resize(b * kk);
        timer.time(t.transfer_ms, [&] {
            VK_CUDA_CHECK(cudaMemcpyAsync(host_d.data(), s.out_d.get(), b * kk * sizeof(float), cudaMemcpyDeviceToHost,
                                          s.stream));
            VK_CUDA_CHECK(cudaMemcpyAsync(host_i.data(), s.out_i.get(), b * kk * sizeof(int), cudaMemcpyDeviceToHost,
                                          s.stream));
        });
        for (std::size_t j = 0; j < b * kk; ++j) {
            const bool pad = host_i[j] < 0;
            out[q0 * kk + j] = {pad ? std::numeric_limits<float>::infinity() : host_d[j], pad ? -1 : host_i[j]};
        }
    }
    return t;
}

Timings PQSearcher::distances(const float* host_queries, std::size_t nq, ScanVariant v, float* out) {
    auto& s = *impl_;
    check(!is_fused(v), "fused variants do not materialise distances");
    s.validate(v);
    s.ensure_layout(v);
    Timings t;
    StageTimer timer(s.stream);
    const std::size_t batch = std::max<std::size_t>(1, std::min(kMaxGridY, kDistanceBudgetBytes / (s.n * 4)));
    for (std::size_t q0 = 0; q0 < nq; q0 += batch) {
        const std::size_t b = std::min(batch, nq - q0);
        s.queries.ensure(b * s.d);
        s.lut.ensure(b * s.m * k::KSUB);
        s.dists.ensure(b * s.n);
        timer.time(t.transfer_ms, [&] {
            VK_CUDA_CHECK(cudaMemcpyAsync(s.queries.get(), host_queries + q0 * s.d, b * s.d * sizeof(float),
                                          cudaMemcpyHostToDevice, s.stream));
        });
        timer.time(t.lut_ms, [&] { s.launch_lut(b); });
        timer.time(t.scan_ms, [&] { s.launch_scan(v, b); });
        timer.time(t.transfer_ms, [&] {
            VK_CUDA_CHECK(cudaMemcpyAsync(out + q0 * s.n, s.dists.get(), b * s.n * sizeof(float),
                                          cudaMemcpyDeviceToHost, s.stream));
        });
    }
    return t;
}

}  // namespace veckern::gpu
