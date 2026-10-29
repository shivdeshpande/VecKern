// GPU kernels vs the CPU reference. Built only when CUDA is available.
#include <cmath>
#include <cstdio>

#include "test_util.hpp"
#include "veckern/cpu_ref.hpp"
#include "veckern/gpu.hpp"
#include "veckern/io.hpp"

using namespace veckern;
using gpu::ScanVariant;

namespace {

struct Fixture {
    // n deliberately not a multiple of 32 or of the block size, to exercise tails and padding.
    std::size_t n = 50'003, d = 64, m = 16, nq = 37;
    std::vector<float> base = synthetic_clustered(n, d, 50, 11);
    std::vector<float> queries = synthetic_clustered(nq, d, 50, 12);
    ProductQuantizer pq{d, m};
    std::vector<uint8_t> codes = std::vector<uint8_t>(n * m);
    Fixture() {
        pq.train(base.data(), 20'000, 8);
        pq.encode(base.data(), n, codes.data());
    }
};

Fixture& fixture() {
    static Fixture f;
    return f;
}

const ScanVariant kScanVariants[] = {ScanVariant::GlobalLut, ScanVariant::SharedLut, ScanVariant::SharedLutVec,
                                     ScanVariant::Interleaved};

const ScanVariant kFusedVariants[] = {ScanVariant::Fused, ScanVariant::FusedReg, ScanVariant::FusedVec,
                                      ScanVariant::FusedVecMQ, ScanVariant::FusedWarp};

void check_search(std::size_t k, ScanVariant v, const gpu::Tuning& tuning = {}) {
    auto& f = fixture();
    std::vector<Neighbor> expected(f.nq * k), got(f.nq * k);
    adc_search(f.pq, f.codes.data(), f.n, f.queries.data(), f.nq, k, expected.data());
    gpu::PQSearcher searcher(f.pq, f.codes.data(), f.n);
    searcher.set_tuning(tuning);
    searcher.search(f.queries.data(), f.nq, k, v, got.data());

    std::vector<int32_t> expected_ids(expected.size());
    for (std::size_t j = 0; j < expected.size(); ++j) expected_ids[j] = expected[j].id;
    // Ids may differ only where float rounding reorders near-ties, so require near-total agreement.
    const double agreement = recall_at_k(got.data(), k, expected_ids.data(), k, f.nq, k);
    REQUIRE(agreement > 0.99);
    for (std::size_t q = 0; q < f.nq; ++q) {
        for (std::size_t j = 0; j < k; ++j) {
            REQUIRE_NEAR(got[q * k + j].dist, expected[q * k + j].dist, 1e-3 * (1 + expected[q * k + j].dist));
            if (j > 0) REQUIRE(!(got[q * k + j] < got[q * k + j - 1]));  // ascending
        }
    }
}

}  // namespace

TEST_CASE("device info is sane") {
    const auto info = gpu::device_info();
    std::printf("    %s, %d SMs, %.0f GB/s peak, %zu B smem/block\n", info.name.c_str(), info.sm_count,
                info.peak_bandwidth_gbs, info.max_smem_per_block);
    REQUIRE(info.sm_count > 0 && info.peak_bandwidth_gbs > 0);
}

TEST_CASE("every scan variant reproduces CPU ADC distances") {
    auto& f = fixture();
    std::vector<float> lut(f.nq * f.m * kSubCentroids), expected(f.nq * f.n), got(f.nq * f.n);
    compute_lut(f.pq, f.queries.data(), f.nq, lut.data());
    adc_scan(lut.data(), f.nq, f.codes.data(), f.n, f.m, expected.data());
    gpu::PQSearcher searcher(f.pq, f.codes.data(), f.n);
    for (auto v : kScanVariants) {
        searcher.distances(f.queries.data(), f.nq, v, got.data());
        for (std::size_t j = 0; j < got.size(); ++j) REQUIRE_NEAR(got[j], expected[j], 1e-3 * (1 + expected[j]));
    }
}

TEST_CASE("search matches CPU for all variants, k = 10") {
    for (auto v : kScanVariants) check_search(10, v);
    for (auto v : kFusedVariants) check_search(10, v);
}

TEST_CASE("search covers every K template (k = 1, 16, 33, 100)") {
    for (std::size_t k : {1, 16, 33, 100}) {
        check_search(k, ScanVariant::SharedLut);
        for (auto v : kFusedVariants) check_search(k, v);
    }
}

TEST_CASE("v7 and v8 are correct for every queries-per-block, including nq not divisible by Q") {
    // nq = 37 is prime, so every Q > 1 leaves a partial last group of padded queries.
    for (int q : {1, 2, 4, 8}) {
        gpu::Tuning t;
        t.queries_per_block = q;
        for (std::size_t k : {1, 8, 10, 32}) check_search(k, ScanVariant::FusedVecMQ, t);
        for (std::size_t k : {1, 8, 10, 32, 64, 128}) check_search(k, ScanVariant::FusedWarp, t);
    }
}

TEST_CASE("tuning overrides keep results correct") {
    for (int threads : {64, 128, 512}) {
        gpu::Tuning t;
        t.scan_threads = threads;
        check_search(10, ScanVariant::SharedLutVec, t);
    }
    for (int bpq : {1, 3, 16}) {
        gpu::Tuning t;
        t.blocks_per_query = bpq;
        check_search(10, ScanVariant::Fused, t);
        check_search(10, ScanVariant::FusedVec, t);
        check_search(10, ScanVariant::FusedVecMQ, t);
    }
    // ILP: with 50,003 vectors and odd block counts, the last step of many threads is partly past n.
    for (int ilp : {2, 4}) {
        for (int bpq : {0, 7}) {
            gpu::Tuning t;
            t.vectors_per_step = ilp;
            t.blocks_per_query = bpq;
            for (std::size_t k : {1, 10, 33}) {
                check_search(k, ScanVariant::FusedVec, t);
                check_search(k, ScanVariant::FusedVecMQ, t);
                check_search(k, ScanVariant::FusedWarp, t);
            }
        }
    }
}

TEST_CASE("launch shape is reported and the wave model is used") {
    auto& f = fixture();
    gpu::PQSearcher searcher(f.pq, f.codes.data(), f.n);
    std::vector<Neighbor> out(f.nq * 10);
    searcher.search(f.queries.data(), f.nq, 10, ScanVariant::FusedVecMQ, out.data());
    const auto shape = searcher.last_launch();
    std::printf("    v7: T=%d blocks_x=%d grid_y=%d Q=%d blocks/SM=%d waves=%.2f\n", shape.threads, shape.blocks_x,
                shape.grid_y, shape.queries_per_block, shape.blocks_per_sm, shape.waves);
    REQUIRE(shape.queries_per_block == 2 && shape.grid_y == 19);  // ceil(37 / 2)
    REQUIRE(shape.blocks_per_sm > 0 && shape.waves > 0);

    // A single query must still be spread over a full wave of blocks (when n has enough work for that).
    std::vector<Neighbor> one(10);
    searcher.search(f.queries.data(), 1, 10, ScanVariant::FusedWarp, one.data());
    const auto single = searcher.last_launch();
    std::printf("    v8, 1 query: blocks_x=%d waves=%.2f\n", single.blocks_x, single.waves);
    const double wave = single.blocks_x / single.waves;  // resident blocks per wave
    const double max_useful = std::ceil(static_cast<double>(f.n) / (single.threads * 16.0));
    REQUIRE(single.blocks_x >= std::min(wave, max_useful) - 0.5);
}

TEST_CASE("word-interleaved variants reject m not divisible by 16; v5 accepts it") {
    auto& f = fixture();
    ProductQuantizer pq8(f.d, 8);
    pq8.train(f.base.data(), 5000, 4);
    std::vector<uint8_t> codes8(f.n * 8);
    pq8.encode(f.base.data(), f.n, codes8.data());
    gpu::PQSearcher searcher(pq8, codes8.data(), f.n);
    std::vector<Neighbor> got(f.nq * 10), expected(f.nq * 10);
    for (auto v : {ScanVariant::SharedLutVec, ScanVariant::FusedVec, ScanVariant::FusedVecMQ, ScanVariant::FusedWarp}) {
        bool threw = false;
        try {
            searcher.search(f.queries.data(), f.nq, 10, v, got.data());
        } catch (const std::exception&) {
            threw = true;
        }
        REQUIRE(threw);
    }
    searcher.search(f.queries.data(), f.nq, 10, ScanVariant::FusedReg, got.data());
    adc_search(pq8, codes8.data(), f.n, f.queries.data(), f.nq, 10, expected.data());
    std::vector<int32_t> ids(expected.size());
    for (std::size_t j = 0; j < ids.size(); ++j) ids[j] = expected[j].id;
    REQUIRE(recall_at_k(got.data(), 10, ids.data(), 10, f.nq, 10) > 0.99);
}

TEST_CASE("invalid tuning is rejected") {
    auto& f = fixture();
    gpu::PQSearcher searcher(f.pq, f.codes.data(), f.n);
    for (auto bad : {gpu::Tuning{100, 0, 2, 1}, gpu::Tuning{256, -1, 2, 1}, gpu::Tuning{256, 0, 3, 1},
                     gpu::Tuning{256, 0, 2, 3}}) {
        bool threw = false;
        try {
            searcher.set_tuning(bad);
        } catch (const std::exception&) {
            threw = true;
        }
        REQUIRE(threw);
    }
}

TEST_CASE("invalid configurations are rejected") {
    auto& f = fixture();
    gpu::PQSearcher searcher(f.pq, f.codes.data(), f.n);
    std::vector<Neighbor> out(129);
    bool threw = false;
    try {
        searcher.search(f.queries.data(), 1, 129, ScanVariant::Fused, out.data());
    } catch (const std::exception&) {
        threw = true;
    }
    REQUIRE(threw);
}

int main() { return veckern::test::run_all(); }
