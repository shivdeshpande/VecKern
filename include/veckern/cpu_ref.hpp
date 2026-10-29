#pragma once
// CPU reference implementations. Every GPU kernel is validated against these.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "veckern/common.hpp"
#include "veckern/pq.hpp"

namespace veckern {

// Bounded max-heap that keeps the k smallest (dist, id) pairs.
class TopK {
public:
    explicit TopK(std::size_t k) : k_(k) { heap_.reserve(k); }

    void push(float dist, int32_t id) {
        const Neighbor cand{dist, id};
        if (heap_.size() < k_) {
            heap_.push_back(cand);
            std::push_heap(heap_.begin(), heap_.end());
        } else if (k_ > 0 && cand < heap_.front()) {
            std::pop_heap(heap_.begin(), heap_.end());
            heap_.back() = cand;
            std::push_heap(heap_.begin(), heap_.end());
        }
    }

    // Writes k results in ascending order, padding with {inf, -1}.
    void write(Neighbor* out) {
        std::sort_heap(heap_.begin(), heap_.end());
        std::copy(heap_.begin(), heap_.end(), out);
        for (std::size_t j = heap_.size(); j < k_; ++j) out[j] = {std::numeric_limits<float>::infinity(), -1};
    }

private:
    std::size_t k_;
    std::vector<Neighbor> heap_;
};

// lut: nq x m x 256 squared sub-distances.
void compute_lut(const ProductQuantizer& pq, const float* queries, std::size_t nq, float* lut);
// Asymmetric distance computation: dists is nq x n.
void adc_scan(const float* lut, std::size_t nq, const uint8_t* codes, std::size_t n, std::size_t m, float* dists);
// Top-k of each row of an nq x n distance matrix; ids are column indices.
void select_topk(const float* dists, std::size_t nq, std::size_t n, std::size_t k, Neighbor* out);
// LUT + scan + selection per query without materialising the nq x n matrix.
void adc_search(const ProductQuantizer& pq, const uint8_t* codes, std::size_t n, const float* queries,
                std::size_t nq, std::size_t k, Neighbor* out);
// Exact brute-force L2 search: the ground truth for recall.
void exact_knn(const float* base, std::size_t n, const float* queries, std::size_t nq, std::size_t d,
               std::size_t k, Neighbor* out);
// Re-scores r candidates per query with full-precision vectors and keeps the best k.
void rerank_exact(const float* base, std::size_t d, const float* queries, std::size_t nq, const Neighbor* candidates,
                  std::size_t r, std::size_t k, Neighbor* out);
// Mean |found[:k] ∩ truth[:k]| / k; truth rows have truth_stride ids.
double recall_at_k(const Neighbor* found, std::size_t found_stride, const int32_t* truth, std::size_t truth_stride,
                   std::size_t nq, std::size_t k);

}  // namespace veckern
