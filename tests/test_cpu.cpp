// CPU reference tests. These define "correct" for the GPU kernels, so they must pass first.
#include <cstdio>
#include <fstream>
#include <numeric>

#include "test_util.hpp"
#include "veckern/cpu_ref.hpp"
#include "veckern/io.hpp"

using namespace veckern;

namespace {

struct Fixture {
    std::size_t n = 4000, d = 32, m = 8, nq = 12;
    std::vector<float> base = synthetic_clustered(n, d, 20, 7);
    std::vector<float> queries = synthetic_clustered(nq, d, 20, 8);
    ProductQuantizer pq{d, m};
    std::vector<uint8_t> codes = std::vector<uint8_t>(n * m);
    Fixture() {
        pq.train(base.data(), n, 10);
        pq.encode(base.data(), n, codes.data());
    }
};

Fixture& fixture() {
    static Fixture f;
    return f;
}

}  // namespace

TEST_CASE("topk keeps the k smallest in ascending order and pads") {
    TopK top(3);
    const float values[] = {5, 1, 4, 1, 3};
    for (int i = 0; i < 5; ++i) top.push(values[i], i);
    Neighbor out[3];
    top.write(out);
    REQUIRE(out[0].id == 1 && out[1].id == 3 && out[2].id == 4);  // tie on 1.0 broken by id

    TopK small(4);
    small.push(2.f, 9);
    Neighbor padded[4];
    small.write(padded);
    REQUIRE(padded[0].id == 9 && padded[1].id == -1 && std::isinf(padded[3].dist));
}

TEST_CASE("PQ rejects d not divisible by m") {
    bool threw = false;
    try {
        ProductQuantizer bad(30, 8);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST_CASE("ADC distance equals exact distance to the decoded vector") {
    auto& f = fixture();
    std::vector<float> lut(f.nq * f.m * kSubCentroids), dists(f.nq * f.n), decoded(f.n * f.d);
    compute_lut(f.pq, f.queries.data(), f.nq, lut.data());
    adc_scan(lut.data(), f.nq, f.codes.data(), f.n, f.m, dists.data());
    f.pq.decode(f.codes.data(), f.n, decoded.data());
    for (std::size_t q = 0; q < f.nq; ++q)
        for (std::size_t i = 0; i < f.n; i += 97)
            REQUIRE_NEAR(dists[q * f.n + i], l2sq(f.queries.data() + q * f.d, decoded.data() + i * f.d, f.d), 1e-3);
}

TEST_CASE("PQ reconstruction error is far below the data spread") {
    auto& f = fixture();
    std::vector<float> decoded(f.n * f.d);
    f.pq.decode(f.codes.data(), f.n, decoded.data());
    double err = 0, spread = 0;
    std::vector<float> mean(f.d, 0.f);
    for (std::size_t i = 0; i < f.n; ++i)
        for (std::size_t j = 0; j < f.d; ++j) mean[j] += f.base[i * f.d + j] / f.n;
    for (std::size_t i = 0; i < f.n; ++i) {
        err += l2sq(f.base.data() + i * f.d, decoded.data() + i * f.d, f.d);
        spread += l2sq(f.base.data() + i * f.d, mean.data(), f.d);
    }
    REQUIRE(err < 0.2 * spread);
}

TEST_CASE("fused adc_search matches lut + scan + select") {
    auto& f = fixture();
    const std::size_t k = 10;
    std::vector<float> lut(f.nq * f.m * kSubCentroids), dists(f.nq * f.n);
    compute_lut(f.pq, f.queries.data(), f.nq, lut.data());
    adc_scan(lut.data(), f.nq, f.codes.data(), f.n, f.m, dists.data());
    std::vector<Neighbor> staged(f.nq * k), fused(f.nq * k);
    select_topk(dists.data(), f.nq, f.n, k, staged.data());
    adc_search(f.pq, f.codes.data(), f.n, f.queries.data(), f.nq, k, fused.data());
    for (std::size_t j = 0; j < staged.size(); ++j) REQUIRE(staged[j].id == fused[j].id);
}

TEST_CASE("exact search finds itself and reranking improves PQ recall") {
    auto& f = fixture();
    const std::size_t k = 10, r = 100;
    std::vector<Neighbor> self(5 * k);
    exact_knn(f.base.data(), f.n, f.base.data(), 5, f.d, k, self.data());
    for (std::size_t q = 0; q < 5; ++q) REQUIRE(self[q * k].id == static_cast<int32_t>(q));

    std::vector<Neighbor> truth(f.nq * k), approx(f.nq * r), reranked(f.nq * k);
    exact_knn(f.base.data(), f.n, f.queries.data(), f.nq, f.d, k, truth.data());
    std::vector<int32_t> truth_ids(truth.size());
    for (std::size_t j = 0; j < truth.size(); ++j) truth_ids[j] = truth[j].id;

    adc_search(f.pq, f.codes.data(), f.n, f.queries.data(), f.nq, r, approx.data());
    rerank_exact(f.base.data(), f.d, f.queries.data(), f.nq, approx.data(), r, k, reranked.data());
    const double pq_recall = recall_at_k(approx.data(), r, truth_ids.data(), k, f.nq, k);
    const double rr_recall = recall_at_k(reranked.data(), k, truth_ids.data(), k, f.nq, k);
    std::printf("    recall@10: pq=%.3f reranked(R=100)=%.3f\n", pq_recall, rr_recall);
    REQUIRE(rr_recall >= pq_recall);
    REQUIRE(rr_recall > 0.8);
}

TEST_CASE("fvecs round trip") {
    const std::string path = "veckern_test_tmp.fvecs";
    {
        std::ofstream out(path, std::ios::binary);
        for (int row = 0; row < 3; ++row) {
            const int32_t dim = 4;
            out.write(reinterpret_cast<const char*>(&dim), 4);
            for (int j = 0; j < 4; ++j) {
                const float v = row * 10.f + j;
                out.write(reinterpret_cast<const char*>(&v), 4);
            }
        }
    }
    std::size_t n = 0, d = 0;
    const auto data = read_fvecs(path, n, d, 2);
    std::remove(path.c_str());
    REQUIRE(n == 2 && d == 4);
    REQUIRE(data[4] == 10.f && data[7] == 13.f);
}

TEST_CASE("SyntheticSource rows depend only on (seed, index)") {
    SyntheticSource src(16, 10, 3);
    std::vector<float> all(100 * 16), tail(50 * 16);
    src.fill(0, 100, all.data());
    src.fill(50, 50, tail.data());
    for (std::size_t j = 0; j < tail.size(); ++j) REQUIRE(all[50 * 16 + j] == tail[j]);
    REQUIRE(all[0] != all[16]);  // different rows differ
}

int main() { return veckern::test::run_all(); }
