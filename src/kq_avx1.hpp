// Ivy Bridge (AVX1 + SSE4.2 + F16C, zero AVX2/FMA3/BMI2) CPU cold-expert pool and K-quant / Q8_0 kernels.
// Compiled exclusively with -O3 -march=ivybridge -mtune=ivybridge -mavx -msse4.2 -mf16c -mno-avx2 -mno-fma.
#pragma once

#include <cstddef>
#include <cstdint>

#include "model.hpp"
#include "poly_wc_bridge.cuh"

namespace fnx {

// Host-side bridge allocator & 12-core physical-core-pinned Ivy Bridge worker pool.
class CpuExpertPool {
public:
    // n_threads <= 0 reads FNX_CPU_THREADS (default 12 physical cores 0..11, excluding SMT siblings).
    explicit CpuExpertPool(const Model& m, int n_threads = -1);
    ~CpuExpertPool();

    CpuExpertPool(const CpuExpertPool&) = delete;
    CpuExpertPool& operator=(const CpuExpertPool&) = delete;

    // Mapped write-combined UVA slot shared by GPU card 0 and the CPU worker pool.
    PolyWcSlot* device_slot() const;

    void reset_stats();
    struct Stats {
        uint64_t calls = 0;
        uint64_t fast_skips = 0;
        uint64_t cold_pairs = 0;
        uint64_t busy_ns = 0;
    };
    Stats stats() const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

// Direct Ivy Bridge AVX1+F16C math primitives (also exercised by tools/test_kq_avx1.cpp).
namespace avx1 {

// Quantize k (multiple of 256) floats into ggml block_q8_K blocks (292 B per 256 floats).
void quantize_row_q8_K(const float* x, void* dst, int k);

// Quantize k (multiple of 32) floats into ggml block_q8_0 blocks (34 B per 32 floats).
void quantize_row_q8_0(const float* x, void* dst, int k);

// Dot products: k must be a multiple of 256 for Q4_K/Q6_K and a multiple of 32 for Q8_0.
float vec_dot_q4_K_q8_K(const void* vx, const void* vy, int k);
float vec_dot_q6_K_q8_K(const void* vx, const void* vy, int k);
float vec_dot_q8_0_q8_0(const void* vx, const void* vy, int k);

// Single-expert 2-half SwiGLU forward pass (for unit testing exact parity against FP32 reference):
//   gu_half[h] stores stacked [gate (ff_half rows); up (ff_half rows)] of width n_in (gu_type = T_Q4_K/T_Q6_K/T_Q8_0)
//   dn_half[h] stores [n_in][ff_half] column-sliced down rows (dn_type = T_Q8_0/T_Q6_K/T_Q4_K)
//   Accumulates weight * (dn_half[0](silu(g0)*u0) + dn_half[1](silu(g1)*u1)) into out[0..n_in).
void cold_expert_2half_accum(const uint8_t* gu_half0, const uint8_t* gu_half1,
                             const uint8_t* dn_half0, const uint8_t* dn_half1,
                             uint32_t gu_type, uint32_t dn_type,
                             size_t gu_row_bytes, size_t dn_row_bytes,
                             int n_in, int ff_half,
                             const float* x, float weight, float* out);

}  // namespace avx1

}  // namespace fnx
