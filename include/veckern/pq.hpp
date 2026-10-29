#pragma once
// Product quantizer: d dimensions split into m sub-spaces, 256 centroids each, one byte per sub-code.
#include <cstdint>
#include <vector>

#include "veckern/common.hpp"

namespace veckern {

class ProductQuantizer {
public:
    ProductQuantizer(std::size_t d, std::size_t m);

    // Independent k-means per sub-space on (a sample of) x, row-major n x d.
    void train(const float* x, std::size_t n, std::size_t iters = 20, uint64_t seed = 1);
    // codes: row-major n x m bytes.
    void encode(const float* x, std::size_t n, uint8_t* codes) const;
    void decode(const uint8_t* codes, std::size_t n, float* x) const;

    const float* centroid(std::size_t sub, std::size_t c) const {
        return centroids_.data() + (sub * kSubCentroids + c) * dsub_;
    }
    // Layout: [m][256][dsub] floats, the layout the GPU LUT kernel expects.
    const std::vector<float>& centroids() const { return centroids_; }

    std::size_t d() const { return d_; }
    std::size_t m() const { return m_; }
    std::size_t dsub() const { return dsub_; }

private:
    std::size_t d_, m_, dsub_;
    std::vector<float> centroids_;
};

}  // namespace veckern
