#include "veckern/pq.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>

namespace veckern {
namespace {

std::size_t nearest_centroid(const float* x, const float* centroids, std::size_t dsub) {
    std::size_t best = 0;
    float best_dist = std::numeric_limits<float>::max();
    for (std::size_t c = 0; c < kSubCentroids; ++c) {
        const float dist = l2sq(x, centroids + c * dsub, dsub);
        if (dist < best_dist) {
            best_dist = dist;
            best = c;
        }
    }
    return best;
}

}  // namespace

ProductQuantizer::ProductQuantizer(std::size_t d, std::size_t m) : d_(d), m_(m), dsub_(m ? d / m : 0) {
    check(m > 0 && d > 0 && d % m == 0, "d must be a positive multiple of m");
    centroids_.assign(m_ * kSubCentroids * dsub_, 0.f);
}

void ProductQuantizer::train(const float* x, std::size_t n, std::size_t iters, uint64_t seed) {
    check(n >= kSubCentroids, "PQ training needs at least 256 vectors");
    std::mt19937_64 rng(seed);
    std::vector<float> sub(n * dsub_);
    std::vector<std::size_t> assign(n), counts(kSubCentroids), perm(n);
    std::vector<float> sums(kSubCentroids * dsub_);

    for (std::size_t s = 0; s < m_; ++s) {
        for (std::size_t i = 0; i < n; ++i)
            std::copy_n(x + i * d_ + s * dsub_, dsub_, sub.data() + i * dsub_);

        // Initialise with 256 distinct training points.
        float* cent = centroids_.data() + s * kSubCentroids * dsub_;
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (std::size_t c = 0; c < kSubCentroids; ++c)
            std::copy_n(sub.data() + perm[c] * dsub_, dsub_, cent + c * dsub_);

        for (std::size_t it = 0; it < iters; ++it) {
#pragma omp parallel for schedule(static)
            for (long long i = 0; i < static_cast<long long>(n); ++i)
                assign[i] = nearest_centroid(sub.data() + i * dsub_, cent, dsub_);

            std::fill(sums.begin(), sums.end(), 0.f);
            std::fill(counts.begin(), counts.end(), 0);
            for (std::size_t i = 0; i < n; ++i) {
                ++counts[assign[i]];
                for (std::size_t j = 0; j < dsub_; ++j) sums[assign[i] * dsub_ + j] += sub[i * dsub_ + j];
            }
            for (std::size_t c = 0; c < kSubCentroids; ++c) {
                if (counts[c] == 0) {  // re-seed an empty cluster with a random point
                    std::copy_n(sub.data() + (rng() % n) * dsub_, dsub_, cent + c * dsub_);
                    continue;
                }
                for (std::size_t j = 0; j < dsub_; ++j) cent[c * dsub_ + j] = sums[c * dsub_ + j] / counts[c];
            }
        }
    }
}

void ProductQuantizer::encode(const float* x, std::size_t n, uint8_t* codes) const {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i)
        for (std::size_t s = 0; s < m_; ++s)
            codes[i * m_ + s] = static_cast<uint8_t>(nearest_centroid(x + i * d_ + s * dsub_, centroid(s, 0), dsub_));
}

void ProductQuantizer::decode(const uint8_t* codes, std::size_t n, float* x) const {
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t s = 0; s < m_; ++s) std::copy_n(centroid(s, codes[i * m_ + s]), dsub_, x + i * d_ + s * dsub_);
}

}  // namespace veckern
