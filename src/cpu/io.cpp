#include "veckern/io.hpp"

#include <cmath>
#include <fstream>
#include <random>

#include "veckern/common.hpp"

namespace veckern {
namespace {

template <typename T>
std::vector<T> read_vecs(const std::string& path, std::size_t& n, std::size_t& d, std::size_t max_n) {
    std::ifstream in(path, std::ios::binary);
    check(in.good(), "cannot open " + path);
    int32_t dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    check(in.good() && dim > 0, "bad header in " + path);
    in.seekg(0, std::ios::end);
    const auto bytes = static_cast<std::size_t>(in.tellg());
    const std::size_t row = sizeof(int32_t) + sizeof(T) * static_cast<std::size_t>(dim);
    check(bytes % row == 0, "file size is not a multiple of the row size: " + path);
    n = bytes / row;
    if (max_n > 0 && max_n < n) n = max_n;
    d = static_cast<std::size_t>(dim);

    std::vector<T> out(n * d);
    in.seekg(0);
    for (std::size_t i = 0; i < n; ++i) {
        int32_t row_dim = 0;
        in.read(reinterpret_cast<char*>(&row_dim), sizeof(row_dim));
        check(row_dim == dim, "inconsistent dimension in " + path);
        in.read(reinterpret_cast<char*>(out.data() + i * d), static_cast<std::streamsize>(sizeof(T) * d));
    }
    check(in.good(), "truncated file " + path);
    return out;
}

}  // namespace

std::vector<float> read_fvecs(const std::string& path, std::size_t& n, std::size_t& d, std::size_t max_n) {
    return read_vecs<float>(path, n, d, max_n);
}

std::vector<float> read_bvecs(const std::string& path, std::size_t& n, std::size_t& d, std::size_t max_n) {
    const auto bytes = read_vecs<uint8_t>(path, n, d, max_n);
    return std::vector<float>(bytes.begin(), bytes.end());
}

std::vector<int32_t> read_ivecs(const std::string& path, std::size_t& n, std::size_t& d, std::size_t max_n) {
    return read_vecs<int32_t>(path, n, d, max_n);
}

std::vector<float> synthetic_clustered(std::size_t n, std::size_t d, std::size_t clusters, uint64_t seed) {
    check(clusters > 0, "clusters must be positive");
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> center_dist(-1.f, 1.f);
    std::normal_distribution<float> noise(0.f, 0.15f);
    std::uniform_int_distribution<std::size_t> pick(0, clusters - 1);

    std::vector<float> centers(clusters * d);
    for (auto& v : centers) v = center_dist(rng);
    std::vector<float> out(n * d);
    for (std::size_t i = 0; i < n; ++i) {
        const float* c = centers.data() + pick(rng) * d;
        for (std::size_t j = 0; j < d; ++j) out[i * d + j] = c[j] + noise(rng);
    }
    return out;
}

namespace {

// SplitMix64: tiny, fast, statistically good enough for synthetic data, and seedable per row.
struct SplitMix {
    uint64_t state;
    uint64_t next() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    float uniform() { return ((next() >> 40) + 0.5f) * (1.0f / 16777216.0f); }  // (0, 1)
    float normal() {  // Box-Muller (one value per call; simplicity over speed)
        const float u1 = uniform(), u2 = uniform();
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2);
    }
};

}  // namespace

SyntheticSource::SyntheticSource(std::size_t d, std::size_t clusters, uint64_t seed, std::size_t latent_dim,
                                 float latent_scale, float noise)
    : d_(d), clusters_(clusters), latent_(latent_dim), seed_(seed), latent_scale_(latent_scale), noise_(noise),
      centers_(clusters * d), basis_(d * latent_dim) {
    check(d > 0 && clusters > 0 && latent_dim > 0, "d, clusters and latent_dim must be positive");
    SplitMix rng{seed ^ 0xC0FFEEull};
    for (auto& v : centers_) v = 2.0f * rng.uniform() - 1.0f;
    const float norm = 1.0f / std::sqrt(static_cast<float>(latent_dim));
    for (auto& v : basis_) v = norm * rng.normal();
}

void SyntheticSource::fill(std::size_t first, std::size_t count, float* out) const {
#pragma omp parallel for schedule(static)
    for (long long r = 0; r < static_cast<long long>(count); ++r) {
        SplitMix rng{seed_ * 0x2545F4914F6CDD1Dull + first + static_cast<std::size_t>(r)};
        const float* c = centers_.data() + (rng.next() % clusters_) * d_;
        float z[256];
        const std::size_t L = latent_ < 256 ? latent_ : 256;
        for (std::size_t l = 0; l < L; ++l) z[l] = latent_scale_ * rng.normal();
        float* x = out + r * d_;
        for (std::size_t j = 0; j < d_; ++j) {
            float acc = c[j] + noise_ * rng.normal();
            for (std::size_t l = 0; l < L; ++l) acc += basis_[j * latent_ + l] * z[l];
            x[j] = acc;
        }
    }
}

}  // namespace veckern
