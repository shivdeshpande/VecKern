#pragma once
// Shared types and helpers used by the CPU reference and the GPU kernels.
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace veckern {

constexpr std::size_t kSubCentroids = 256;  // one byte per PQ sub-code

struct Neighbor {
    float dist;
    int32_t id;  // -1 marks padding when fewer than k results exist
};

// Strict ordering by (distance, id) so CPU and GPU break ties identically.
inline bool operator<(const Neighbor& a, const Neighbor& b) {
    return a.dist < b.dist || (a.dist == b.dist && a.id < b.id);
}

inline void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

inline float l2sq(const float* a, const float* b, std::size_t d) {
    float acc = 0.f;
    for (std::size_t j = 0; j < d; ++j) {
        const float diff = a[j] - b[j];
        acc += diff * diff;
    }
    return acc;
}

class Timer {
public:
    Timer() : start_(std::chrono::steady_clock::now()) {}
    double ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
    }

private:
    std::chrono::steady_clock::time_point start_;
};

}  // namespace veckern
