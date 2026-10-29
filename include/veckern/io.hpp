#pragma once
// Readers for the TEXMEX / big-ann-benchmarks formats and a synthetic data generator.
#include <cstdint>
#include <string>
#include <vector>

namespace veckern {

// .fvecs: per row an int32 dimension followed by d float32 values.
std::vector<float> read_fvecs(const std::string& path, std::size_t& n, std::size_t& d, std::size_t max_n = 0);
// .bvecs: per row an int32 dimension followed by d uint8 values (converted to float).
std::vector<float> read_bvecs(const std::string& path, std::size_t& n, std::size_t& d, std::size_t max_n = 0);
// .ivecs: per row an int32 count followed by that many int32 ids (ground-truth files).
std::vector<int32_t> read_ivecs(const std::string& path, std::size_t& n, std::size_t& d, std::size_t max_n = 0);

// Gaussian blobs around random centers; clustered data is closer to real embeddings than uniform noise.
std::vector<float> synthetic_clustered(std::size_t n, std::size_t d, std::size_t clusters, uint64_t seed);

// Benchmark data with realistic structure: x = center_c + A z + noise, where z has `latent_dim` dimensions and
// A is a fixed random d x latent_dim matrix. Real embeddings have low intrinsic dimension; isotropic Gaussian
// blobs do not, and on them every compressed index looks useless (distances concentrate).
// Vector i depends only on (seed, i): generation is parallel (OpenMP) and any range can be produced on its own.
class SyntheticSource {
public:
    SyntheticSource(std::size_t d, std::size_t clusters, uint64_t seed, std::size_t latent_dim = 16,
                    float latent_scale = 0.5f, float noise = 0.05f);
    void fill(std::size_t first, std::size_t count, float* out) const;  // rows first .. first+count-1
    std::size_t d() const { return d_; }

private:
    std::size_t d_, clusters_, latent_;
    uint64_t seed_;
    float latent_scale_, noise_;
    std::vector<float> centers_, basis_;  // basis_: d x latent_dim, row-major
};

}  // namespace veckern
