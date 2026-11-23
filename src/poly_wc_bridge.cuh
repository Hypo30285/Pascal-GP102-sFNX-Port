// Write-Combined zero-copy GPU<->CPU bridge for cold MoE experts on Dual-P6000 (sm_61) + Ivy Bridge Xeon.
// GPU card 0 writes cold expert routing pairs and mixed activations into one UVA-mapped WC slot, signals req_seq
// via __threadfence_system(), and runs hot VRAM experts + shared expert concurrently while the 12-core physical-core
// CpuExpertPool computes both FF halves [0..FF/2) and [FF/2..FF) in DDR3 RAM and writes the weighted sum to out.
#pragma once

#include <cstdint>

#include "model.hpp"

namespace fnx {

// Ring::err bit set when GPU 0's bounded wait on PolyWcSlot::done_seq times out.
constexpr uint32_t POLY_WC_ERR_BIT = 128u;
constexpr int POLY_WC_WAIT_S = 10;

#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
uint32_t poly_wc_seq(unsigned tag, int layer) {
    return (tag << 8) | (uint32_t) (layer + 1);
}

struct alignas(64) PolyWcSlot {
    volatile uint32_t req_seq;                  // GPU card 0 writes poly_wc_seq(ctl[1], layer) (nonzero) after __threadfence_system()
    alignas(64) volatile uint32_t done_seq;     // CPU writes == req_seq after _mm_sfence(), followed by _mm_sfence()
    alignas(64) int32_t layer;                  // 0..NL-1
    int32_t T;                                  // 1..MAXT
    int32_t n_cold_pairs;                       // 0 -> FAST PATH (CPU not signaled, GPU skips done_seq poll)
    alignas(16) int32_t cold_tok[MAXT * TOPK];  // token index t in [0, T)
    alignas(16) int32_t cold_exp[MAXT * TOPK];  // expert id e in [0, NEXP)
    alignas(16) float   cold_w[MAXT * TOPK];    // routing weight topw[t * TOPK + k]
    alignas(64) float   mixed[MAXT * N];        // T rows of N f32 (written by GPU 0 ONLY when n_cold_pairs > 0)
    alignas(64) float   out[MAXT * N];          // T rows of N f32 weighted cold expert sum (written by CPU)
};

}  // namespace fnx
