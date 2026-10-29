#pragma once
// Host-side API for the CUDA kernels. Contains no CUDA types, so plain C++ code can include it.
#include <cstdint>
#include <memory>
#include <string>

#include "veckern/common.hpp"
#include "veckern/pq.hpp"

namespace veckern::gpu {

// Each variant is one step of the optimisation story (see docs/PLAN.md).
enum class ScanVariant {
    GlobalLut,     // v0: LUT read from global memory, row-major codes
    SharedLut,     // v1: LUT staged in shared memory
    SharedLutVec,  // v2: + 16-byte vectorised code loads (m % 16 == 0)
    Interleaved,   // v3: + warp-interleaved code layout for coalesced loads
    Fused,         // v4: interleaved scan fused with per-thread top-k; no n-wide distance matrix
    FusedReg,      // v5: v4 with the top-k list held in registers (fully unrolled insertion)
    FusedVec,      // v6: v5 + 16-byte-word interleaved layout (m % 16 == 0)
    FusedVecMQ,    // v7: v6 + several queries per block sharing every code load (m % 16 == 0)
    FusedWarp,     // v8: v7 with one warp-cooperative top-k list per warp instead of one per thread
};

const char* variant_name(ScanVariant v);
bool is_fused(ScanVariant v);

// Launch-shape overrides for parameter sweeps. 0 = use the built-in heuristic.
struct Tuning {
    int scan_threads = 256;      // block size of the v0-v3 scan kernels (multiple of 32, <= 1024)
    int blocks_per_query = 0;    // blocks_x for scans and fused kernels; 0 = auto
    int queries_per_block = 2;   // Q for v7/v8 (1, 2, 4 or 8); v7 also caps Q * K <= 64, both cap Q by shared memory
    int vectors_per_step = 4;    // ILP for v6-v8: vectors each thread processes per loop step (1, 2 or 4)
};

struct DeviceInfo {
    std::string name;
    int sm_count = 0;
    double peak_bandwidth_gbs = 0.0;  // theoretical DRAM bandwidth
    std::size_t max_smem_per_block = 0;  // opt-in limit
};

DeviceInfo device_info(int device = 0);

// Milliseconds per stage, measured with CUDA events and summed over query batches.
struct Timings {
    double transfer_ms = 0, lut_ms = 0, scan_ms = 0, select_ms = 0;
    double kernel_ms() const { return lut_ms + scan_ms + select_ms; }
    double total_ms() const { return transfer_ms + kernel_ms(); }
};

class PQSearcher {
public:
    // Copies centroids and codes (row-major n x m) to the device once.
    PQSearcher(const ProductQuantizer& pq, const uint8_t* codes, std::size_t n);
    ~PQSearcher();
    PQSearcher(const PQSearcher&) = delete;
    PQSearcher& operator=(const PQSearcher&) = delete;

    // out: nq x k neighbors, ascending distance. Supports 1 <= k <= 128.
    Timings search(const float* queries, std::size_t nq, std::size_t k, ScanVariant variant, Neighbor* out);
    // out: nq x n ADC distances. Not available for the fused variants. Used by the tests.
    Timings distances(const float* queries, std::size_t nq, ScanVariant variant, float* out);

    void set_tuning(const Tuning& tuning);
    const Tuning& tuning() const;
    // The launch shape actually used by the last search (after heuristics and clamping).
    struct LaunchShape {
        int threads = 0, blocks_x = 0, grid_y = 0, queries_per_block = 1, vectors_per_step = 1;
        int blocks_per_sm = 0;  // occupancy-API residency (v5-v7 only)
        double waves = 0.0;     // grid blocks / resident blocks (v5-v7 only)
    };
    LaunchShape last_launch() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace veckern::gpu
