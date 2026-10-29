#include "veckern/cpu_ref.hpp"

#include <unordered_set>

namespace veckern {

void compute_lut(const ProductQuantizer& pq, const float* queries, std::size_t nq, float* lut) {
    const std::size_t m = pq.m(), dsub = pq.dsub(), d = pq.d();
#pragma omp parallel for schedule(static)
    for (long long q = 0; q < static_cast<long long>(nq); ++q)
        for (std::size_t s = 0; s < m; ++s)
            for (std::size_t c = 0; c < kSubCentroids; ++c)
                lut[(q * m + s) * kSubCentroids + c] = l2sq(queries + q * d + s * dsub, pq.centroid(s, c), dsub);
}

void adc_scan(const float* lut, std::size_t nq, const uint8_t* codes, std::size_t n, std::size_t m, float* dists) {
#pragma omp parallel for schedule(static)
    for (long long q = 0; q < static_cast<long long>(nq); ++q) {
        const float* table = lut + q * m * kSubCentroids;
        for (std::size_t i = 0; i < n; ++i) {
            const uint8_t* code = codes + i * m;
            float acc = 0.f;
            for (std::size_t s = 0; s < m; ++s) acc += table[s * kSubCentroids + code[s]];
            dists[q * n + i] = acc;
        }
    }
}

void select_topk(const float* dists, std::size_t nq, std::size_t n, std::size_t k, Neighbor* out) {
#pragma omp parallel for schedule(static)
    for (long long q = 0; q < static_cast<long long>(nq); ++q) {
        TopK top(k);
        for (std::size_t i = 0; i < n; ++i) top.push(dists[q * n + i], static_cast<int32_t>(i));
        top.write(out + q * k);
    }
}

void adc_search(const ProductQuantizer& pq, const uint8_t* codes, std::size_t n, const float* queries,
                std::size_t nq, std::size_t k, Neighbor* out) {
    const std::size_t m = pq.m();
#pragma omp parallel for schedule(dynamic)
    for (long long q = 0; q < static_cast<long long>(nq); ++q) {
        std::vector<float> lut(m * kSubCentroids);
        compute_lut(pq, queries + q * pq.d(), 1, lut.data());
        TopK top(k);
        for (std::size_t i = 0; i < n; ++i) {
            const uint8_t* code = codes + i * m;
            float acc = 0.f;
            for (std::size_t s = 0; s < m; ++s) acc += lut[s * kSubCentroids + code[s]];
            top.push(acc, static_cast<int32_t>(i));
        }
        top.write(out + q * k);
    }
}

void exact_knn(const float* base, std::size_t n, const float* queries, std::size_t nq, std::size_t d,
               std::size_t k, Neighbor* out) {
#pragma omp parallel for schedule(dynamic)
    for (long long q = 0; q < static_cast<long long>(nq); ++q) {
        TopK top(k);
        for (std::size_t i = 0; i < n; ++i) top.push(l2sq(queries + q * d, base + i * d, d), static_cast<int32_t>(i));
        top.write(out + q * k);
    }
}

void rerank_exact(const float* base, std::size_t d, const float* queries, std::size_t nq, const Neighbor* candidates,
                  std::size_t r, std::size_t k, Neighbor* out) {
#pragma omp parallel for schedule(static)
    for (long long q = 0; q < static_cast<long long>(nq); ++q) {
        TopK top(k);
        for (std::size_t j = 0; j < r; ++j) {
            const int32_t id = candidates[q * r + j].id;
            if (id >= 0) top.push(l2sq(queries + q * d, base + static_cast<std::size_t>(id) * d, d), id);
        }
        top.write(out + q * k);
    }
}

double recall_at_k(const Neighbor* found, std::size_t found_stride, const int32_t* truth, std::size_t truth_stride,
                   std::size_t nq, std::size_t k) {
    check(k <= found_stride && k <= truth_stride, "k exceeds the result width");
    if (nq == 0 || k == 0) return 0.0;
    double total = 0.0;
    for (std::size_t q = 0; q < nq; ++q) {
        std::unordered_set<int32_t> expected(truth + q * truth_stride, truth + q * truth_stride + k);
        std::size_t hits = 0;
        for (std::size_t j = 0; j < k; ++j) hits += expected.count(found[q * found_stride + j].id);
        total += static_cast<double>(hits) / k;
    }
    return total / nq;
}

}  // namespace veckern
