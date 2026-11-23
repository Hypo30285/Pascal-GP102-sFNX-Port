// Ivy Bridge (AVX1 + SSE4.2 + F16C, zero AVX2/FMA3/BMI2) CPU cold-expert pool and K-quant / Q8_0 kernels.
// Compiled exclusively with -O3 -march=ivybridge -mtune=ivybridge -mavx -msse4.2 -mf16c -mno-avx2 -mno-fma.
#include "kq_avx1.hpp"
#include "env.hpp"

#include <cuda_runtime.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include <smmintrin.h>
#include <xmmintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

#if !defined(__AVX__) || !defined(__F16C__)
#error "src/kq_avx1.cpp must be compiled with -mavx -msse4.2 -mf16c"
#endif
#if defined(__AVX2__) || defined(__FMA__) || defined(__BMI2__)
#error "src/kq_avx1.cpp must NOT be compiled with AVX2, FMA3, or BMI2 (Xeon E5-2697 v2 Ivy Bridge)"
#endif

#define FNX_MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

namespace fnx {
namespace {

// Exact GGML quantization block layouts (byte-identical to llama.cpp/ggml/src/ggml-common.h).
constexpr int QK8_0 = 32;
constexpr int QK_K = 256;

struct BlockQ8_0 {
    uint16_t d;
    int8_t   qs[QK8_0];
};
static_assert(sizeof(BlockQ8_0) == 34, "BlockQ8_0 size mismatch");

struct BlockQ4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];
};
static_assert(sizeof(BlockQ4_K) == 144, "BlockQ4_K size mismatch");

struct BlockQ6_K {
    uint8_t  ql[QK_K / 2];
    uint8_t  qh[QK_K / 4];
    int8_t   scales[QK_K / 16];
    uint16_t d;
};
static_assert(sizeof(BlockQ6_K) == 210, "BlockQ6_K size mismatch");

struct BlockQ8_K {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
};
static_assert(sizeof(BlockQ8_K) == 292, "BlockQ8_K size mismatch");

constexpr int kTileT = 8;
constexpr int kMaxBlocksQ8 = 256;          // up to 8192 columns per Q8_0 row
constexpr int kSpinsBeforeSleep = 2000000; // ~7 ms on Ivy Bridge: stays hot across GPU layers, sleeps when idle
constexpr int kGuChunkRows = 32;           // matches Q8_0 block=32 for instant in-L1 down activation quantization
constexpr int kDnChunkRows = 32;           // 32 floats = 128 B = 2 whole 64B cache lines (zero false sharing on WC out)

inline float h2f_avx1(uint16_t u) noexcept {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) u)));
}

inline uint16_t f2h_avx1(float f) noexcept {
    return (uint16_t) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT), 0);
}

inline float hsum_float_8_avx1(__m256 v) noexcept {
    __m128 res = _mm256_extractf128_ps(v, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(v));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

inline float eval_swiglu(float g, float u) noexcept {
    return (g / (1.0f + std::exp(-g))) * u;
}

// SSE4.1 MOVNTDQA (_mm_stream_load_si128) burst readers for Write-Combined mapped GPU->CPU memory.
inline void stream_load_wc_floats(float* dst, const float* wc_src, size_t count) noexcept {
    size_t i = 0;
    if ((reinterpret_cast<std::uintptr_t>(wc_src) & 15u) == 0u) {
        for (; i + 16 <= count; i += 16) {
            const __m128i v0 = _mm_stream_load_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(wc_src + i + 0)));
            const __m128i v1 = _mm_stream_load_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(wc_src + i + 4)));
            const __m128i v2 = _mm_stream_load_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(wc_src + i + 8)));
            const __m128i v3 = _mm_stream_load_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(wc_src + i + 12)));
            _mm_storeu_ps(dst + i + 0, _mm_castsi128_ps(v0));
            _mm_storeu_ps(dst + i + 4, _mm_castsi128_ps(v1));
            _mm_storeu_ps(dst + i + 8, _mm_castsi128_ps(v2));
            _mm_storeu_ps(dst + i + 12, _mm_castsi128_ps(v3));
        }
        for (; i + 4 <= count; i += 4) {
            const __m128i v0 = _mm_stream_load_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(wc_src + i)));
            _mm_storeu_ps(dst + i, _mm_castsi128_ps(v0));
        }
    }
    for (; i < count; ++i) dst[i] = wc_src[i];
}

inline void stream_load_wc_i32(int32_t* dst, const int32_t* wc_src, size_t count) noexcept {
    size_t i = 0;
    if ((reinterpret_cast<std::uintptr_t>(wc_src) & 15u) == 0u) {
        for (; i + 4 <= count; i += 4) {
            const __m128i v0 = _mm_stream_load_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(wc_src + i)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), v0);
        }
    }
    for (; i < count; ++i) dst[i] = wc_src[i];
}

// ---------------------------------------------------------------- Q6_K x Q8_K (AVX1 + SSE4.2 + F16C)
alignas(16) constexpr uint8_t k_scale_shuffle_128[128] = {
     0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
     2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
     4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
     6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
     8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
    10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,
    12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,
    14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15
};

inline __m128i get_scale_shuffle_avx1(int i) noexcept {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(k_scale_shuffle_128) + i);
}

inline void q6k_half_1tok_avx1(const uint8_t* __restrict__ q4, const uint8_t* __restrict__ qh,
                               const int8_t* __restrict__ q8, const __m128i sc_16,
                               __m128i& sumi) noexcept {
    const __m128i m15 = _mm_set1_epi8(15);
    const __m128i m48 = _mm_set1_epi8(48);
    const __m128i q4bitsH_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qh + 0));
    const __m128i q4bitsH_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qh + 16));
    const __m128i q4bits1_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4 + 0));
    const __m128i q4bits1_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4 + 16));
    const __m128i q4bits2_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4 + 32));
    const __m128i q4bits2_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4 + 48));

    const __m128i sc_lo = _mm_unpacklo_epi16(sc_16, sc_16);
    const __m128i sc_hi = _mm_unpackhi_epi16(sc_16, sc_16);

    {
        const __m128i q4_0 = _mm_or_si128(_mm_and_si128(q4bits1_0, m15), _mm_and_si128(_mm_slli_epi16(q4bitsH_0, 4), m48));
        const __m128i q4_1 = _mm_or_si128(_mm_and_si128(q4bits1_1, m15), _mm_and_si128(_mm_slli_epi16(q4bitsH_1, 4), m48));
        sumi = _mm_add_epi32(sumi, _mm_madd_epi16(_mm_shuffle_epi32(sc_lo, 0x00), _mm_maddubs_epi16(q4_0, _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 0)))));
        sumi = _mm_add_epi32(sumi, _mm_madd_epi16(_mm_shuffle_epi32(sc_lo, 0x55), _mm_maddubs_epi16(q4_1, _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 16)))));
    }
    {
        const __m128i q4_2 = _mm_or_si128(_mm_and_si128(q4bits2_0, m15), _mm_and_si128(_mm_slli_epi16(q4bitsH_0, 2), m48));
        const __m128i q4_3 = _mm_or_si128(_mm_and_si128(q4bits2_1, m15), _mm_and_si128(_mm_slli_epi16(q4bitsH_1, 2), m48));
        sumi = _mm_add_epi32(sumi, _mm_madd_epi16(_mm_shuffle_epi32(sc_lo, 0xAA), _mm_maddubs_epi16(q4_2, _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 32)))));
        sumi = _mm_add_epi32(sumi, _mm_madd_epi16(_mm_shuffle_epi32(sc_lo, 0xFF), _mm_maddubs_epi16(q4_3, _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 48)))));
    }
    {
        const __m128i q4_4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_0, 4), m15), _mm_and_si128(q4bitsH_0, m48));
        const __m128i q4_5 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_1, 4), m15), _mm_and_si128(q4bitsH_1, m48));
        sumi = _mm_add_epi32(sumi, _mm_madd_epi16(_mm_shuffle_epi32(sc_hi, 0x00), _mm_maddubs_epi16(q4_4, _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 64)))));
        sumi = _mm_add_epi32(sumi, _mm_madd_epi16(_mm_shuffle_epi32(sc_hi, 0x55), _mm_maddubs_epi16(q4_5, _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 80)))));
    }
    {
        const __m128i q4_6 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_0, 4), m15), _mm_and_si128(_mm_srli_epi16(q4bitsH_0, 2), m48));
        const __m128i q4_7 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_1, 4), m15), _mm_and_si128(_mm_srli_epi16(q4bitsH_1, 2), m48));
        sumi = _mm_add_epi32(sumi, _mm_madd_epi16(_mm_shuffle_epi32(sc_hi, 0xAA), _mm_maddubs_epi16(q4_6, _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 96)))));
        sumi = _mm_add_epi32(sumi, _mm_madd_epi16(_mm_shuffle_epi32(sc_hi, 0xFF), _mm_maddubs_epi16(q4_7, _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 112)))));
    }
}

inline float q6k_dot_1tok_avx1(const BlockQ6_K* x, const BlockQ8_K* y, int nb) noexcept {
    __m128 acc = _mm_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        if (i + 3 < nb) {
            _mm_prefetch(reinterpret_cast<const char*>(&x[i + 3]), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(&x[i + 3]) + 64, _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(&x[i + 3]) + 128, _MM_HINT_T0);
        }
        const float d = y[i].d * h2f_avx1(x[i].d);
        const __m128i scales = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[i].scales));
        const __m128i sc_lo = _mm_cvtepi8_epi16(scales);
        const __m128i sc_hi = _mm_cvtepi8_epi16(_mm_bsrli_si128(scales, 8));
        const __m128i q8sums_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y[i].bsums));
        const __m128i q8sums_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y[i].bsums) + 1);
        __m128i sumi = _mm_sub_epi32(_mm_setzero_si128(),
            _mm_slli_epi32(_mm_add_epi32(_mm_madd_epi16(q8sums_0, sc_lo), _mm_madd_epi16(q8sums_1, sc_hi)), 5));

        q6k_half_1tok_avx1(x[i].ql + 0,  x[i].qh + 0,  y[i].qs + 0,   sc_lo, sumi);
        q6k_half_1tok_avx1(x[i].ql + 64, x[i].qh + 32, y[i].qs + 128, sc_hi, sumi);

        acc = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(d), _mm_cvtepi32_ps(sumi)), acc);
    }
    acc = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
    acc = _mm_add_ss(acc, _mm_movehdup_ps(acc));
    return _mm_cvtss_f32(acc);
}

template <int NT>
inline void q6k_dot_nt_avx1(const BlockQ6_K* x, int nb, const void* const* act, float* res) noexcept {
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float xd = h2f_avx1(x[i].d);
        const __m128i scales = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[i].scales));

        __m128i sumi_0[NT], sumi_1[NT];
        for (int t = 0; t < NT; ++t) {
            sumi_0[t] = _mm_setzero_si128();
            sumi_1[t] = _mm_setzero_si128();
        }

        const uint8_t* q4 = x[i].ql;
        const uint8_t* qh = x[i].qh;
        int is = 0;
        for (int j = 0; j < QK_K / 128; ++j) {
            const __m128i q4bitsH_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qh)); qh += 16;
            const __m128i q4bitsH_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qh)); qh += 16;
            const __m128i q4bits1_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4)); q4 += 16;
            const __m128i q4bits1_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4)); q4 += 16;

            {
                const __m128i m15 = _mm_set1_epi8(15);
                const __m128i m3 = _mm_set1_epi8(3);
                const __m128i q4_0 = _mm_or_si128(_mm_and_si128(q4bits1_0, m15), _mm_slli_epi16(_mm_and_si128(q4bitsH_0, m3), 4));
                const __m128i q4_1 = _mm_or_si128(_mm_and_si128(q4bits1_1, m15), _mm_slli_epi16(_mm_and_si128(q4bitsH_1, m3), 4));
                const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle_avx1(is + 0));
                const __m128i sc16_0 = _mm_cvtepi8_epi16(scale_0);
                const __m128i sc16_1 = _mm_cvtepi8_epi16(_mm_bsrli_si128(scale_0, 8));
                for (int t = 0; t < NT; ++t) {
                    const int8_t* q8 = (static_cast<const BlockQ8_K*>(act[t]) + i)->qs + j * 128;
                    sumi_0[t] = _mm_add_epi32(sumi_0[t], _mm_madd_epi16(sc16_0, _mm_maddubs_epi16(q4_0, _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 0)))));
                    sumi_1[t] = _mm_add_epi32(sumi_1[t], _mm_madd_epi16(sc16_1, _mm_maddubs_epi16(q4_1, _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 16)))));
                }
            }
            {
                const __m128i m15 = _mm_set1_epi8(15);
                const __m128i m48 = _mm_set1_epi8(48);
                const __m128i q4_4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_0, 4), m15), _mm_and_si128(q4bitsH_0, m48));
                const __m128i q4_5 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_1, 4), m15), _mm_and_si128(q4bitsH_1, m48));
                const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle_avx1(is + 2));
                const __m128i sc16_4 = _mm_cvtepi8_epi16(scale_2);
                const __m128i sc16_5 = _mm_cvtepi8_epi16(_mm_bsrli_si128(scale_2, 8));
                for (int t = 0; t < NT; ++t) {
                    const int8_t* q8 = (static_cast<const BlockQ8_K*>(act[t]) + i)->qs + j * 128;
                    sumi_0[t] = _mm_add_epi32(sumi_0[t], _mm_madd_epi16(sc16_4, _mm_maddubs_epi16(q4_4, _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 64)))));
                    sumi_1[t] = _mm_add_epi32(sumi_1[t], _mm_madd_epi16(sc16_5, _mm_maddubs_epi16(q4_5, _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 80)))));
                }
            }

            const __m128i q4bits2_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4)); q4 += 16;
            const __m128i q4bits2_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4)); q4 += 16;
            {
                const __m128i m15 = _mm_set1_epi8(15);
                const __m128i m12 = _mm_set1_epi8(12);
                const __m128i q4_2 = _mm_or_si128(_mm_and_si128(q4bits2_0, m15), _mm_slli_epi16(_mm_and_si128(q4bitsH_0, m12), 2));
                const __m128i q4_3 = _mm_or_si128(_mm_and_si128(q4bits2_1, m15), _mm_slli_epi16(_mm_and_si128(q4bitsH_1, m12), 2));
                const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle_avx1(is + 1));
                const __m128i sc16_2 = _mm_cvtepi8_epi16(scale_1);
                const __m128i sc16_3 = _mm_cvtepi8_epi16(_mm_bsrli_si128(scale_1, 8));
                for (int t = 0; t < NT; ++t) {
                    const int8_t* q8 = (static_cast<const BlockQ8_K*>(act[t]) + i)->qs + j * 128;
                    sumi_0[t] = _mm_add_epi32(sumi_0[t], _mm_madd_epi16(sc16_2, _mm_maddubs_epi16(q4_2, _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 32)))));
                    sumi_1[t] = _mm_add_epi32(sumi_1[t], _mm_madd_epi16(sc16_3, _mm_maddubs_epi16(q4_3, _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 48)))));
                }
            }
            {
                const __m128i m15 = _mm_set1_epi8(15);
                const __m128i mn64 = _mm_set1_epi8(-64);
                const __m128i q4_6 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_0, 4), m15), _mm_srli_epi16(_mm_and_si128(q4bitsH_0, mn64), 2));
                const __m128i q4_7 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_1, 4), m15), _mm_srli_epi16(_mm_and_si128(q4bitsH_1, mn64), 2));
                const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle_avx1(is + 3));
                const __m128i sc16_6 = _mm_cvtepi8_epi16(scale_3);
                const __m128i sc16_7 = _mm_cvtepi8_epi16(_mm_bsrli_si128(scale_3, 8));
                for (int t = 0; t < NT; ++t) {
                    const int8_t* q8 = (static_cast<const BlockQ8_K*>(act[t]) + i)->qs + j * 128;
                    sumi_0[t] = _mm_add_epi32(sumi_0[t], _mm_madd_epi16(sc16_6, _mm_maddubs_epi16(q4_6, _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 96)))));
                    sumi_1[t] = _mm_add_epi32(sumi_1[t], _mm_madd_epi16(sc16_7, _mm_maddubs_epi16(q4_7, _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 112)))));
                }
            }
            is += 4;
        }

        const __m128i scales_16_0 = _mm_cvtepi8_epi16(scales);
        const __m128i scales_16_1 = _mm_cvtepi8_epi16(_mm_bsrli_si128(scales, 8));
        for (int t = 0; t < NT; ++t) {
            const BlockQ8_K* y = static_cast<const BlockQ8_K*>(act[t]) + i;
            const float d = y->d * xd;
            const __m128i q8sums_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y->bsums));
            const __m128i q8sums_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y->bsums) + 1);
            const __m128i q8sclsub_0 = _mm_slli_epi32(_mm_madd_epi16(q8sums_0, scales_16_0), 5);
            const __m128i q8sclsub_1 = _mm_slli_epi32(_mm_madd_epi16(q8sums_1, scales_16_1), 5);
            const __m128i s0 = _mm_sub_epi32(sumi_0[t], q8sclsub_0);
            const __m128i s1 = _mm_sub_epi32(sumi_1[t], q8sclsub_1);
            const __m256i sumi = FNX_MM256_SET_M128I(s1, s0);
            acc[t] = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi)), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_float_8_avx1(acc[t]);
}

inline void q6k_dot_multi_avx1(const BlockQ6_K* x, int nb, const void* const* act, int nt, float* res) noexcept {
    int t0 = 0;
    for (; t0 + 3 <= nt; t0 += 3) q6k_dot_nt_avx1<3>(x, nb, act + t0, res + t0);
    const int rem = nt - t0;
    if (rem == 2) q6k_dot_nt_avx1<2>(x, nb, act + t0, res + t0);
    else if (rem == 1) res[t0] = q6k_dot_1tok_avx1(x, static_cast<const BlockQ8_K*>(act[t0]), nb);
}

// ---------------------------------------------------------------- Q4_K x Q8_K (AVX1 + SSE4.2 + F16C)
inline void q4k_block_1tok_avx1(const BlockQ4_K& __restrict__ xb, const int8_t* __restrict__ q8,
                                const __m128i& q8s, float yd, const __m128i& m4,
                                __m128& acc, __m128& acc_m) noexcept {
    static constexpr uint32_t kmask1 = 0x3f3f3f3fu;
    static constexpr uint32_t kmask2 = 0x0f0f0f0fu;
    static constexpr uint32_t kmask3 = 0x03030303u;

    const float d = yd * h2f_avx1(xb.d);
    const float dmin = -yd * h2f_avx1(xb.dmin);
    const uint8_t* q4 = xb.qs;

    uint32_t utmp[4];
    std::memcpy(utmp, xb.scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    const __m128i utmps = _mm_set_epi32((int) utmp[3], (int) utmp[2], (int) utmp[1], (int) utmp[0]);
    const __m128i scales = _mm_cvtepu8_epi16(utmps);
    const __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));
    acc_m = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(_mm_madd_epi16(mins, q8s))), acc_m);

    const __m128i sc_lo = _mm_unpacklo_epi16(scales, scales);
    const __m128i sc_hi = _mm_unpackhi_epi16(scales, scales);
    __m128i sumi = _mm_setzero_si128();

    {
        const __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4 + 0));
        const __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4 + 16));
        const __m128i dl = _mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(b0, m4), _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 0))),
                                         _mm_maddubs_epi16(_mm_and_si128(b1, m4), _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 16))));
        const __m128i dh = _mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(b0, 4), m4), _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 32))),
                                         _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(b1, 4), m4), _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 48))));
        sumi = _mm_add_epi32(sumi, _mm_madd_epi16(_mm_shuffle_epi32(sc_lo, 0x00), dl));
        sumi = _mm_add_epi32(sumi, _mm_madd_epi16(_mm_shuffle_epi32(sc_lo, 0x55), dh));
    }
    {
        const __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4 + 32));
        const __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4 + 48));
        const __m128i dl = _mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(b0, m4), _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 64))),
                                         _mm_maddubs_epi16(_mm_and_si128(b1, m4), _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 80))));
        const __m128i dh = _mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(b0, 4), m4), _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 96))),
                                         _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(b1, 4), m4), _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 112))));
        sumi = _mm_add_epi32(sumi, _mm_madd_epi16(_mm_shuffle_epi32(sc_lo, 0xAA), dl));
        sumi = _mm_add_epi32(sumi, _mm_madd_epi16(_mm_shuffle_epi32(sc_lo, 0xFF), dh));
    }
    {
        const __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4 + 64));
        const __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4 + 80));
        const __m128i dl = _mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(b0, m4), _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 128))),
                                         _mm_maddubs_epi16(_mm_and_si128(b1, m4), _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 144))));
        const __m128i dh = _mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(b0, 4), m4), _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 160))),
                                         _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(b1, 4), m4), _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 176))));
        sumi = _mm_add_epi32(sumi, _mm_madd_epi16(_mm_shuffle_epi32(sc_hi, 0x00), dl));
        sumi = _mm_add_epi32(sumi, _mm_madd_epi16(_mm_shuffle_epi32(sc_hi, 0x55), dh));
    }
    {
        const __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4 + 96));
        const __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4 + 112));
        const __m128i dl = _mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(b0, m4), _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 192))),
                                         _mm_maddubs_epi16(_mm_and_si128(b1, m4), _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 208))));
        const __m128i dh = _mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(b0, 4), m4), _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 224))),
                                         _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(b1, 4), m4), _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 240))));
        sumi = _mm_add_epi32(sumi, _mm_madd_epi16(_mm_shuffle_epi32(sc_hi, 0xAA), dl));
        sumi = _mm_add_epi32(sumi, _mm_madd_epi16(_mm_shuffle_epi32(sc_hi, 0xFF), dh));
    }
    acc = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(d), _mm_cvtepi32_ps(sumi)), acc);
}

inline float q4k_dot_1tok_avx1(const BlockQ4_K* x, const BlockQ8_K* y, int nb) noexcept {
    const __m128i m4 = _mm_set1_epi8(0xF);
    __m128 acc = _mm_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        if (i + 4 < nb) {
            _mm_prefetch(reinterpret_cast<const char*>(&x[i + 4]), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(&x[i + 4]) + 64, _MM_HINT_T0);
        }
        const __m128i q8sums_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&y[i].bsums[0]));
        const __m128i q8sums_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&y[i].bsums[8]));
        const __m128i q8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
        q4k_block_1tok_avx1(x[i], y[i].qs, q8s, y[i].d, m4, acc, acc_m);
    }
    acc = _mm_add_ps(acc, acc_m);
    acc = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
    acc = _mm_add_ss(acc, _mm_movehdup_ps(acc));
    return _mm_cvtss_f32(acc);
}

inline float q4k_dot_1tok_pre_avx1(const BlockQ4_K* x, const BlockQ8_K* y, const __m128i* q8s, int nb) noexcept {
    const __m128i m4 = _mm_set1_epi8(0xF);
    __m128 acc = _mm_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        if (i + 4 < nb) {
            _mm_prefetch(reinterpret_cast<const char*>(&x[i + 4]), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(&x[i + 4]) + 64, _MM_HINT_T0);
        }
        q4k_block_1tok_avx1(x[i], y[i].qs, q8s[i], y[i].d, m4, acc, acc_m);
    }
    acc = _mm_add_ps(acc, acc_m);
    acc = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
    acc = _mm_add_ss(acc, _mm_movehdup_ps(acc));
    return _mm_cvtss_f32(acc);
}

template <int NT>
inline void q4k_dot_nt_avx1(const BlockQ4_K* x, int nb, const void* const* act, float* res) noexcept {
    static constexpr uint32_t kmask1 = 0x3f3f3f3fu;
    static constexpr uint32_t kmask2 = 0x0f0f0f0fu;
    static constexpr uint32_t kmask3 = 0x03030303u;
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i m2 = _mm_set1_epi8(0x2);

    __m256 acc[NT];
    __m128 acc_m[NT];
    for (int t = 0; t < NT; ++t) {
        acc[t] = _mm256_setzero_ps();
        acc_m[t] = _mm_setzero_ps();
    }
    uint32_t utmp[4];

    for (int i = 0; i < nb; ++i) {
        const float xd = h2f_avx1(x[i].d);
        const float xdmin = -h2f_avx1(x[i].dmin);

        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m128i utmps = _mm_set_epi32((int) utmp[3], (int) utmp[2], (int) utmp[1], (int) utmp[0]);
        const __m128i scales = _mm_cvtepu8_epi16(utmps);
        const __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));

        for (int t = 0; t < NT; ++t) {
            const BlockQ8_K* y = static_cast<const BlockQ8_K*>(act[t]) + i;
            const __m128i q8sums_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&y->bsums[0]));
            const __m128i q8sums_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&y->bsums[8]));
            const __m128i q8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            acc_m[t] = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(y->d * xdmin), _mm_cvtepi32_ps(prod)), acc_m[t]);
        }

        __m128i sumi_0[NT], sumi_1[NT];
        for (int t = 0; t < NT; ++t) {
            sumi_0[t] = _mm_setzero_si128();
            sumi_1[t] = _mm_setzero_si128();
        }

        const uint8_t* q4 = x[i].qs;
        __m128i shuffle = _mm_set1_epi16(0x0100);
        for (int j = 0; j < QK_K / 64; ++j) {
            const __m128i scale_l = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);
            const __m128i scale_h = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);

            __m128i q4bits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4)); q4 += 16;
            const __m128i q4l_0 = _mm_and_si128(q4bits, m4);
            const __m128i q4h_0 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4);
            q4bits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4)); q4 += 16;
            const __m128i q4l_1 = _mm_and_si128(q4bits, m4);
            const __m128i q4h_1 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4);

            for (int t = 0; t < NT; ++t) {
                const int8_t* q8 = (static_cast<const BlockQ8_K*>(act[t]) + i)->qs + j * 64;
                sumi_0[t] = _mm_add_epi32(sumi_0[t], _mm_madd_epi16(scale_l, _mm_maddubs_epi16(q4l_0, _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 0)))));
                sumi_1[t] = _mm_add_epi32(sumi_1[t], _mm_madd_epi16(scale_l, _mm_maddubs_epi16(q4l_1, _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 16)))));
                sumi_0[t] = _mm_add_epi32(sumi_0[t], _mm_madd_epi16(scale_h, _mm_maddubs_epi16(q4h_0, _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 32)))));
                sumi_1[t] = _mm_add_epi32(sumi_1[t], _mm_madd_epi16(scale_h, _mm_maddubs_epi16(q4h_1, _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8 + 48)))));
            }
        }

        for (int t = 0; t < NT; ++t) {
            const BlockQ8_K* y = static_cast<const BlockQ8_K*>(act[t]) + i;
            const __m256 vd = _mm256_set1_ps(y->d * xd);
            const __m256i sumi = FNX_MM256_SET_M128I(sumi_1[t], sumi_0[t]);
            acc[t] = _mm256_add_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(sumi)), acc[t]);
        }
    }

    for (int t = 0; t < NT; ++t) {
        __m128 am = acc_m[t];
        am = _mm_add_ps(am, _mm_movehl_ps(am, am));
        am = _mm_add_ss(am, _mm_movehdup_ps(am));
        res[t] = hsum_float_8_avx1(acc[t]) + _mm_cvtss_f32(am);
    }
}

inline void q4k_dot_multi_avx1(const BlockQ4_K* x, int nb, const void* const* act, int nt, float* res) noexcept {
    int t0 = 0;
    for (; t0 + 3 <= nt; t0 += 3) q4k_dot_nt_avx1<3>(x, nb, act + t0, res + t0);
    const int rem = nt - t0;
    if (rem == 2) q4k_dot_nt_avx1<2>(x, nb, act + t0, res + t0);
    else if (rem == 1) res[t0] = q4k_dot_1tok_avx1(x, static_cast<const BlockQ8_K*>(act[t0]), nb);
}

// ---------------------------------------------------------------- Q8_0 x Q8_0 (AVX1 + SSE4.2 + F16C)
inline __m128i mul_add_epi8_sse_avx1(const __m128i x, const __m128i y) noexcept {
    const __m128i ax = _mm_sign_epi8(x, x);
    const __m128i sy = _mm_sign_epi8(y, x);
    return _mm_maddubs_epi16(ax, sy);
}

inline __m256 mul_sum_i8_quad_float_avx1(const __m128i x_1_0, const __m128i x_1_1, const __m128i x_2_0, const __m128i x_2_1,
                                         const __m128i y_1_0, const __m128i y_1_1, const __m128i y_2_0, const __m128i y_2_1) noexcept {
    const __m128i mone = _mm_set1_epi16(1);
    const __m128i p_1 = _mm_add_epi32(_mm_madd_epi16(mul_add_epi8_sse_avx1(x_1_0, y_1_0), mone),
                                      _mm_madd_epi16(mul_add_epi8_sse_avx1(x_1_1, y_1_1), mone));
    const __m128i p_2 = _mm_add_epi32(_mm_madd_epi16(mul_add_epi8_sse_avx1(x_2_0, y_2_0), mone),
                                      _mm_madd_epi16(mul_add_epi8_sse_avx1(x_2_1, y_2_1), mone));
    return _mm256_cvtepi32_ps(FNX_MM256_SET_M128I(p_2, p_1));
}

inline float q8_0_dot_1tok_avx1(const BlockQ8_0* x, const BlockQ8_0* y, int nb) noexcept {
    __m256 acc = _mm256_setzero_ps();
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const __m128i qx_1_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[ib].qs));
        const __m128i qx_1_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[ib].qs) + 1);
        const __m128i qx_2_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[ib + 1].qs));
        const __m128i qx_2_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[ib + 1].qs) + 1);
        const __m128i qy_1_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y[ib].qs));
        const __m128i qy_1_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y[ib].qs) + 1);
        const __m128i qy_2_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y[ib + 1].qs));
        const __m128i qy_2_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y[ib + 1].qs) + 1);
        const __m256 p = mul_sum_i8_quad_float_avx1(qx_1_0, qx_1_1, qx_2_0, qx_2_1, qy_1_0, qy_1_1, qy_2_0, qy_2_1);
        const __m256 deltas = _mm256_set_m128(_mm_set1_ps(h2f_avx1(x[ib + 1].d) * h2f_avx1(y[ib + 1].d)),
                                              _mm_set1_ps(h2f_avx1(x[ib].d) * h2f_avx1(y[ib].d)));
        acc = _mm256_add_ps(_mm256_mul_ps(deltas, p), acc);
    }
    float s = hsum_float_8_avx1(acc);
    for (; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < QK8_0; ++j) sumi += (int) x[ib].qs[j] * (int) y[ib].qs[j];
        s += (float) sumi * (h2f_avx1(x[ib].d) * h2f_avx1(y[ib].d));
    }
    return s;
}

inline void q8_0_2rows_1tok_avx1(const BlockQ8_0* x0, const BlockQ8_0* x1, const BlockQ8_0* y, const float* yd,
                                 int nb, float& out0, float& out1) noexcept {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const __m128i qy_1_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y[ib].qs));
        const __m128i qy_1_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y[ib].qs) + 1);
        const __m128i qy_2_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y[ib + 1].qs));
        const __m128i qy_2_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y[ib + 1].qs) + 1);
        const float yd0 = yd[ib], yd1 = yd[ib + 1];
        {
            const __m128i qx_1_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x0[ib].qs));
            const __m128i qx_1_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x0[ib].qs) + 1);
            const __m128i qx_2_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x0[ib + 1].qs));
            const __m128i qx_2_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x0[ib + 1].qs) + 1);
            const __m256 p = mul_sum_i8_quad_float_avx1(qx_1_0, qx_1_1, qx_2_0, qx_2_1, qy_1_0, qy_1_1, qy_2_0, qy_2_1);
            const __m256 deltas = _mm256_set_m128(_mm_set1_ps(h2f_avx1(x0[ib + 1].d) * yd1),
                                                  _mm_set1_ps(h2f_avx1(x0[ib].d) * yd0));
            acc0 = _mm256_add_ps(_mm256_mul_ps(deltas, p), acc0);
        }
        {
            const __m128i qx_1_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x1[ib].qs));
            const __m128i qx_1_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x1[ib].qs) + 1);
            const __m128i qx_2_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x1[ib + 1].qs));
            const __m128i qx_2_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x1[ib + 1].qs) + 1);
            const __m256 p = mul_sum_i8_quad_float_avx1(qx_1_0, qx_1_1, qx_2_0, qx_2_1, qy_1_0, qy_1_1, qy_2_0, qy_2_1);
            const __m256 deltas = _mm256_set_m128(_mm_set1_ps(h2f_avx1(x1[ib + 1].d) * yd1),
                                                  _mm_set1_ps(h2f_avx1(x1[ib].d) * yd0));
            acc1 = _mm256_add_ps(_mm256_mul_ps(deltas, p), acc1);
        }
    }
    float s0 = hsum_float_8_avx1(acc0);
    float s1 = hsum_float_8_avx1(acc1);
    for (; ib < nb; ++ib) {
        int sumi0 = 0, sumi1 = 0;
        for (int j = 0; j < QK8_0; ++j) {
            sumi0 += (int) x0[ib].qs[j] * (int) y[ib].qs[j];
            sumi1 += (int) x1[ib].qs[j] * (int) y[ib].qs[j];
        }
        s0 += (float) sumi0 * (h2f_avx1(x0[ib].d) * yd[ib]);
        s1 += (float) sumi1 * (h2f_avx1(x1[ib].d) * yd[ib]);
    }
    out0 = s0;
    out1 = s1;
}

inline void q8_0_2rows_2tok_avx1(const BlockQ8_0* x0, const BlockQ8_0* x1, const void* const* hq,
                                 const float (*yd)[kMaxBlocksQ8], int nb, float* out_r0, float* out_r1) noexcept {
    const BlockQ8_0* y0 = static_cast<const BlockQ8_0*>(hq[0]);
    const BlockQ8_0* y1 = static_cast<const BlockQ8_0*>(hq[1]);
    __m256 acc00 = _mm256_setzero_ps(), acc01 = _mm256_setzero_ps();
    __m256 acc10 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps();
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const __m128i qy0_1_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y0[ib].qs));
        const __m128i qy0_1_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y0[ib].qs) + 1);
        const __m128i qy0_2_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y0[ib + 1].qs));
        const __m128i qy0_2_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y0[ib + 1].qs) + 1);
        const __m128i qy1_1_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y1[ib].qs));
        const __m128i qy1_1_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y1[ib].qs) + 1);
        const __m128i qy1_2_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y1[ib + 1].qs));
        const __m128i qy1_2_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y1[ib + 1].qs) + 1);
        const float yd00 = yd[0][ib], yd01 = yd[0][ib + 1];
        const float yd10 = yd[1][ib], yd11 = yd[1][ib + 1];
        {
            const __m128i qx_1_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x0[ib].qs));
            const __m128i qx_1_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x0[ib].qs) + 1);
            const __m128i qx_2_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x0[ib + 1].qs));
            const __m128i qx_2_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x0[ib + 1].qs) + 1);
            const float xd0 = h2f_avx1(x0[ib].d), xd1 = h2f_avx1(x0[ib + 1].d);
            acc00 = _mm256_add_ps(_mm256_mul_ps(_mm256_set_m128(_mm_set1_ps(xd1 * yd01), _mm_set1_ps(xd0 * yd00)),
                                                mul_sum_i8_quad_float_avx1(qx_1_0, qx_1_1, qx_2_0, qx_2_1, qy0_1_0, qy0_1_1, qy0_2_0, qy0_2_1)), acc00);
            acc01 = _mm256_add_ps(_mm256_mul_ps(_mm256_set_m128(_mm_set1_ps(xd1 * yd11), _mm_set1_ps(xd0 * yd10)),
                                                mul_sum_i8_quad_float_avx1(qx_1_0, qx_1_1, qx_2_0, qx_2_1, qy1_1_0, qy1_1_1, qy1_2_0, qy1_2_1)), acc01);
        }
        {
            const __m128i qx_1_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x1[ib].qs));
            const __m128i qx_1_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x1[ib].qs) + 1);
            const __m128i qx_2_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x1[ib + 1].qs));
            const __m128i qx_2_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x1[ib + 1].qs) + 1);
            const float xd0 = h2f_avx1(x1[ib].d), xd1 = h2f_avx1(x1[ib + 1].d);
            acc10 = _mm256_add_ps(_mm256_mul_ps(_mm256_set_m128(_mm_set1_ps(xd1 * yd01), _mm_set1_ps(xd0 * yd00)),
                                                mul_sum_i8_quad_float_avx1(qx_1_0, qx_1_1, qx_2_0, qx_2_1, qy0_1_0, qy0_1_1, qy0_2_0, qy0_2_1)), acc10);
            acc11 = _mm256_add_ps(_mm256_mul_ps(_mm256_set_m128(_mm_set1_ps(xd1 * yd11), _mm_set1_ps(xd0 * yd10)),
                                                mul_sum_i8_quad_float_avx1(qx_1_0, qx_1_1, qx_2_0, qx_2_1, qy1_1_0, qy1_1_1, qy1_2_0, qy1_2_1)), acc11);
        }
    }
    float s00 = hsum_float_8_avx1(acc00), s01 = hsum_float_8_avx1(acc01);
    float s10 = hsum_float_8_avx1(acc10), s11 = hsum_float_8_avx1(acc11);
    for (; ib < nb; ++ib) {
        int sum00 = 0, sum01 = 0, sum10 = 0, sum11 = 0;
        for (int j = 0; j < QK8_0; ++j) {
            sum00 += (int) x0[ib].qs[j] * (int) y0[ib].qs[j];
            sum01 += (int) x0[ib].qs[j] * (int) y1[ib].qs[j];
            sum10 += (int) x1[ib].qs[j] * (int) y0[ib].qs[j];
            sum11 += (int) x1[ib].qs[j] * (int) y1[ib].qs[j];
        }
        const float xd0 = h2f_avx1(x0[ib].d), xd1 = h2f_avx1(x1[ib].d);
        s00 += (float) sum00 * (xd0 * yd[0][ib]);
        s01 += (float) sum01 * (xd0 * yd[1][ib]);
        s10 += (float) sum10 * (xd1 * yd[0][ib]);
        s11 += (float) sum11 * (xd1 * yd[1][ib]);
    }
    out_r0[0] = s00; out_r0[1] = s01;
    out_r1[0] = s10; out_r1[1] = s11;
}

template <int NT>
inline void q8_0_dot_nt_avx1(const BlockQ8_0* x, int nb, const void* const* hq,
                             const float (*yd)[kMaxBlocksQ8], float* res) noexcept {
    const __m128i mone = _mm_set1_epi16(1);
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();

    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const __m128i qx_1_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[ib].qs));
        const __m128i qx_1_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[ib].qs) + 1);
        const __m128i qx_2_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[ib + 1].qs));
        const __m128i qx_2_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x[ib + 1].qs) + 1);

        const __m128i ax_1_0 = _mm_sign_epi8(qx_1_0, qx_1_0);
        const __m128i ax_1_1 = _mm_sign_epi8(qx_1_1, qx_1_1);
        const __m128i ax_2_0 = _mm_sign_epi8(qx_2_0, qx_2_0);
        const __m128i ax_2_1 = _mm_sign_epi8(qx_2_1, qx_2_1);

        const float xd0 = h2f_avx1(x[ib].d);
        const float xd1 = h2f_avx1(x[ib + 1].d);

        for (int t = 0; t < NT; ++t) {
            const BlockQ8_0* y = static_cast<const BlockQ8_0*>(hq[t]);
            const __m128i qy_1_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y[ib].qs));
            const __m128i qy_1_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y[ib].qs) + 1);
            const __m128i qy_2_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y[ib + 1].qs));
            const __m128i qy_2_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y[ib + 1].qs) + 1);

            const __m128i p_1 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(ax_1_0, _mm_sign_epi8(qy_1_0, qx_1_0)), mone),
                                              _mm_madd_epi16(_mm_maddubs_epi16(ax_1_1, _mm_sign_epi8(qy_1_1, qx_1_1)), mone));
            const __m128i p_2 = _mm_add_epi32(_mm_madd_epi16(_mm_maddubs_epi16(ax_2_0, _mm_sign_epi8(qy_2_0, qx_2_0)), mone),
                                              _mm_madd_epi16(_mm_maddubs_epi16(ax_2_1, _mm_sign_epi8(qy_2_1, qx_2_1)), mone));
            const __m256 p = _mm256_cvtepi32_ps(FNX_MM256_SET_M128I(p_2, p_1));
            const __m256 deltas = _mm256_set_m128(_mm_set1_ps(xd1 * yd[t][ib + 1]), _mm_set1_ps(xd0 * yd[t][ib]));
            acc[t] = _mm256_add_ps(_mm256_mul_ps(deltas, p), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        float s = hsum_float_8_avx1(acc[t]);
        const BlockQ8_0* y = static_cast<const BlockQ8_0*>(hq[t]);
        for (int b = ib; b < nb; ++b) {
            int sumi = 0;
            for (int j = 0; j < QK8_0; ++j) sumi += (int) x[b].qs[j] * (int) y[b].qs[j];
            s += (float) sumi * (h2f_avx1(x[b].d) * yd[t][b]);
        }
        res[t] = s;
    }
}

inline void q8_0_dot_multi_avx1(const BlockQ8_0* x, int nb, const void* const* hq,
                                const float (*yd)[kMaxBlocksQ8], int nt, float* res) noexcept {
    int t0 = 0;
    for (; t0 + 4 <= nt; t0 += 4) q8_0_dot_nt_avx1<4>(x, nb, hq + t0, yd + t0, res + t0);
    const int rem = nt - t0;
    if (rem == 3) q8_0_dot_nt_avx1<3>(x, nb, hq + t0, yd + t0, res + t0);
    else if (rem == 2) q8_0_dot_nt_avx1<2>(x, nb, hq + t0, yd + t0, res + t0);
    else if (rem == 1) q8_0_dot_nt_avx1<1>(x, nb, hq + t0, yd + t0, res + t0);
}

// ---------------------------------------------------------------- Sequential gate-then-up chunk + SwiGLU
void gu_rows_chunk_avx1(uint32_t gu_type, const uint8_t* gate_base, const uint8_t* up_base, size_t gu_row,
                        int n_in, const void* const* act, int nt, float* const* ff_out, int r0, int r1) noexcept {
    const int nr = r1 - r0;
    if (nr <= 0 || nt <= 0) return;

    if (gu_type == T_Q4_K) {
        const int nb = n_in / QK_K;
        if (nt == 1) {
            const BlockQ8_K* y = static_cast<const BlockQ8_K*>(act[0]);
            alignas(16) __m128i q8s[64];
            const bool use_pre = (nb <= 64);
            if (use_pre) {
                for (int ib = 0; ib < nb; ++ib) {
                    const __m128i s0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&y[ib].bsums[0]));
                    const __m128i s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&y[ib].bsums[8]));
                    q8s[ib] = _mm_hadd_epi16(s0, s1);
                }
            }
            float g_buf[kGuChunkRows];
            for (int base = r0; base < r1; base += kGuChunkRows) {
                const int end = std::min(r1, base + kGuChunkRows);
                for (int r = base; r < end; ++r) {
                    const auto* gr = reinterpret_cast<const BlockQ4_K*>(gate_base + (size_t) r * gu_row);
                    g_buf[r - base] = use_pre ? q4k_dot_1tok_pre_avx1(gr, y, q8s, nb) : q4k_dot_1tok_avx1(gr, y, nb);
                }
                for (int r = base; r < end; ++r) {
                    const auto* ur = reinterpret_cast<const BlockQ4_K*>(up_base + (size_t) r * gu_row);
                    const float u = use_pre ? q4k_dot_1tok_pre_avx1(ur, y, q8s, nb) : q4k_dot_1tok_avx1(ur, y, nb);
                    ff_out[0][r] = eval_swiglu(g_buf[r - base], u);
                }
            }
            return;
        }
        for (int t0 = 0; t0 < nt; t0 += kTileT) {
            const int m = std::min(kTileT, nt - t0);
            float g_buf[kGuChunkRows][kTileT];
            float u[kTileT];
            for (int base = r0; base < r1; base += kGuChunkRows) {
                const int end = std::min(r1, base + kGuChunkRows);
                for (int r = base; r < end; ++r) {
                    const auto* gr = reinterpret_cast<const BlockQ4_K*>(gate_base + (size_t) r * gu_row);
                    q4k_dot_multi_avx1(gr, nb, act + t0, m, g_buf[r - base]);
                }
                for (int r = base; r < end; ++r) {
                    const auto* ur = reinterpret_cast<const BlockQ4_K*>(up_base + (size_t) r * gu_row);
                    q4k_dot_multi_avx1(ur, nb, act + t0, m, u);
                    for (int t = 0; t < m; ++t) ff_out[t0 + t][r] = eval_swiglu(g_buf[r - base][t], u[t]);
                }
            }
        }
        return;
    }

    if (gu_type == T_Q6_K) {
        const int nb = n_in / QK_K;
        if (nt == 1) {
            const BlockQ8_K* y = static_cast<const BlockQ8_K*>(act[0]);
            float g_buf[kGuChunkRows];
            for (int base = r0; base < r1; base += kGuChunkRows) {
                const int end = std::min(r1, base + kGuChunkRows);
                for (int r = base; r < end; ++r) {
                    const auto* gr = reinterpret_cast<const BlockQ6_K*>(gate_base + (size_t) r * gu_row);
                    g_buf[r - base] = q6k_dot_1tok_avx1(gr, y, nb);
                }
                for (int r = base; r < end; ++r) {
                    const auto* ur = reinterpret_cast<const BlockQ6_K*>(up_base + (size_t) r * gu_row);
                    const float u = q6k_dot_1tok_avx1(ur, y, nb);
                    ff_out[0][r] = eval_swiglu(g_buf[r - base], u);
                }
            }
            return;
        }
        for (int t0 = 0; t0 < nt; t0 += kTileT) {
            const int m = std::min(kTileT, nt - t0);
            float g_buf[kGuChunkRows][kTileT];
            float u[kTileT];
            for (int base = r0; base < r1; base += kGuChunkRows) {
                const int end = std::min(r1, base + kGuChunkRows);
                for (int r = base; r < end; ++r) {
                    const auto* gr = reinterpret_cast<const BlockQ6_K*>(gate_base + (size_t) r * gu_row);
                    q6k_dot_multi_avx1(gr, nb, act + t0, m, g_buf[r - base]);
                }
                for (int r = base; r < end; ++r) {
                    const auto* ur = reinterpret_cast<const BlockQ6_K*>(up_base + (size_t) r * gu_row);
                    q6k_dot_multi_avx1(ur, nb, act + t0, m, u);
                    for (int t = 0; t < m; ++t) ff_out[t0 + t][r] = eval_swiglu(g_buf[r - base][t], u[t]);
                }
            }
        }
        return;
    }

    if (gu_type == T_Q8_0) {
        const int nb = n_in / QK8_0;
        if (nb > kMaxBlocksQ8) return;
        if (nt == 1) {
            const BlockQ8_0* y = static_cast<const BlockQ8_0*>(act[0]);
            float yd[kMaxBlocksQ8];
            for (int ib = 0; ib < nb; ++ib) yd[ib] = h2f_avx1(y[ib].d);
            float g_buf[kGuChunkRows];
            for (int base = r0; base < r1; base += kGuChunkRows) {
                const int end = std::min(r1, base + kGuChunkRows);
                int r = base;
                for (; r + 1 < end; r += 2) {
                    q8_0_2rows_1tok_avx1(reinterpret_cast<const BlockQ8_0*>(gate_base + (size_t) r * gu_row),
                                         reinterpret_cast<const BlockQ8_0*>(gate_base + (size_t) (r + 1) * gu_row),
                                         y, yd, nb, g_buf[r - base], g_buf[r + 1 - base]);
                }
                if (r < end) {
                    g_buf[r - base] = q8_0_dot_1tok_avx1(reinterpret_cast<const BlockQ8_0*>(gate_base + (size_t) r * gu_row), y, nb);
                }
                r = base;
                for (; r + 1 < end; r += 2) {
                    float u0 = 0.f, u1 = 0.f;
                    q8_0_2rows_1tok_avx1(reinterpret_cast<const BlockQ8_0*>(up_base + (size_t) r * gu_row),
                                         reinterpret_cast<const BlockQ8_0*>(up_base + (size_t) (r + 1) * gu_row),
                                         y, yd, nb, u0, u1);
                    ff_out[0][r]     = eval_swiglu(g_buf[r - base], u0);
                    ff_out[0][r + 1] = eval_swiglu(g_buf[r + 1 - base], u1);
                }
                if (r < end) {
                    const float u = q8_0_dot_1tok_avx1(reinterpret_cast<const BlockQ8_0*>(up_base + (size_t) r * gu_row), y, nb);
                    ff_out[0][r] = eval_swiglu(g_buf[r - base], u);
                }
            }
            return;
        }
        for (int t0 = 0; t0 < nt; t0 += kTileT) {
            const int m = std::min(kTileT, nt - t0);
            float yd[kTileT][kMaxBlocksQ8];
            for (int t = 0; t < m; ++t) {
                const BlockQ8_0* y = static_cast<const BlockQ8_0*>(act[t0 + t]);
                for (int ib = 0; ib < nb; ++ib) yd[t][ib] = h2f_avx1(y[ib].d);
            }
            float g_buf[kGuChunkRows][kTileT];
            float u[kTileT];
            for (int base = r0; base < r1; base += kGuChunkRows) {
                const int end = std::min(r1, base + kGuChunkRows);
                for (int r = base; r < end; ++r) {
                    const auto* gr = reinterpret_cast<const BlockQ8_0*>(gate_base + (size_t) r * gu_row);
                    q8_0_dot_multi_avx1(gr, nb, act + t0, yd, m, g_buf[r - base]);
                }
                for (int r = base; r < end; ++r) {
                    const auto* ur = reinterpret_cast<const BlockQ8_0*>(up_base + (size_t) r * gu_row);
                    q8_0_dot_multi_avx1(ur, nb, act + t0, yd, m, u);
                    for (int t = 0; t < m; ++t) ff_out[t0 + t][r] = eval_swiglu(g_buf[r - base][t], u[t]);
                }
            }
        }
    }
}

void dn_rows_chunk_avx1(uint32_t dn_type, const uint8_t* w, size_t row_bytes, int ff_half,
                        const void* const* hq, int nt, float* const* out, int r0, int r1,
                        bool accumulate) noexcept {
    if (r0 >= r1 || nt <= 0) return;

    if (dn_type == T_Q8_0) {
        const int nb = ff_half / QK8_0;
        if (nb > kMaxBlocksQ8) return;
        if (nt == 1) {
            const BlockQ8_0* y = static_cast<const BlockQ8_0*>(hq[0]);
            float yd[kMaxBlocksQ8];
            for (int ib = 0; ib < nb; ++ib) yd[ib] = h2f_avx1(y[ib].d);
            int r = r0;
            for (; r + 1 < r1; r += 2) {
                float s0 = 0.f, s1 = 0.f;
                q8_0_2rows_1tok_avx1(reinterpret_cast<const BlockQ8_0*>(w + (size_t) r * row_bytes),
                                     reinterpret_cast<const BlockQ8_0*>(w + (size_t) (r + 1) * row_bytes),
                                     y, yd, nb, s0, s1);
                const int o = r - r0;
                if (accumulate) {
                    out[0][o]     += s0;
                    out[0][o + 1] += s1;
                } else {
                    out[0][o]     = s0;
                    out[0][o + 1] = s1;
                }
            }
            if (r < r1) {
                const float s = q8_0_dot_1tok_avx1(reinterpret_cast<const BlockQ8_0*>(w + (size_t) r * row_bytes), y, nb);
                const int o = r - r0;
                if (accumulate) out[0][o] += s;
                else out[0][o] = s;
            }
            return;
        }
        for (int t0 = 0; t0 < nt; t0 += kTileT) {
            const int m = std::min(kTileT, nt - t0);
            float yd[kTileT][kMaxBlocksQ8];
            for (int t = 0; t < m; ++t) {
                const BlockQ8_0* y = static_cast<const BlockQ8_0*>(hq[t0 + t]);
                for (int ib = 0; ib < nb; ++ib) yd[t][ib] = h2f_avx1(y[ib].d);
            }
            if (m == 2) {
                int r = r0;
                float s0[2], s1[2];
                for (; r + 1 < r1; r += 2) {
                    q8_0_2rows_2tok_avx1(reinterpret_cast<const BlockQ8_0*>(w + (size_t) r * row_bytes),
                                         reinterpret_cast<const BlockQ8_0*>(w + (size_t) (r + 1) * row_bytes),
                                         hq + t0, yd, nb, s0, s1);
                    const int o = r - r0;
                    if (accumulate) {
                        out[t0 + 0][o]     += s0[0];
                        out[t0 + 1][o]     += s0[1];
                        out[t0 + 0][o + 1] += s1[0];
                        out[t0 + 1][o + 1] += s1[1];
                    } else {
                        out[t0 + 0][o]     = s0[0];
                        out[t0 + 1][o]     = s0[1];
                        out[t0 + 0][o + 1] = s1[0];
                        out[t0 + 1][o + 1] = s1[1];
                    }
                }
                if (r < r1) {
                    q8_0_dot_multi_avx1(reinterpret_cast<const BlockQ8_0*>(w + (size_t) r * row_bytes), nb, hq + t0, yd, 2, s0);
                    const int o = r - r0;
                    if (accumulate) {
                        out[t0 + 0][o] += s0[0];
                        out[t0 + 1][o] += s0[1];
                    } else {
                        out[t0 + 0][o] = s0[0];
                        out[t0 + 1][o] = s0[1];
                    }
                }
                continue;
            }
            float s[kTileT];
            for (int r = r0; r < r1; ++r) {
                q8_0_dot_multi_avx1(reinterpret_cast<const BlockQ8_0*>(w + (size_t) r * row_bytes), nb, hq + t0, yd, m, s);
                const int o = r - r0;
                if (accumulate) {
                    for (int t = 0; t < m; ++t) out[t0 + t][o] += s[t];
                } else {
                    for (int t = 0; t < m; ++t) out[t0 + t][o] = s[t];
                }
            }
        }
        return;
    }

    if (dn_type == T_Q6_K) {
        const int nb = ff_half / QK_K;
        for (int t0 = 0; t0 < nt; t0 += kTileT) {
            const int m = std::min(kTileT, nt - t0);
            float s[kTileT];
            for (int r = r0; r < r1; ++r) {
                q6k_dot_multi_avx1(reinterpret_cast<const BlockQ6_K*>(w + (size_t) r * row_bytes), nb, hq + t0, m, s);
                const int o = r - r0;
                if (accumulate) {
                    for (int t = 0; t < m; ++t) out[t0 + t][o] += s[t];
                } else {
                    for (int t = 0; t < m; ++t) out[t0 + t][o] = s[t];
                }
            }
        }
        return;
    }

    if (dn_type == T_Q4_K) {
        const int nb = ff_half / QK_K;
        for (int t0 = 0; t0 < nt; t0 += kTileT) {
            const int m = std::min(kTileT, nt - t0);
            float s[kTileT];
            for (int r = r0; r < r1; ++r) {
                q4k_dot_multi_avx1(reinterpret_cast<const BlockQ4_K*>(w + (size_t) r * row_bytes), nb, hq + t0, m, s);
                const int o = r - r0;
                if (accumulate) {
                    for (int t = 0; t < m; ++t) out[t0 + t][o] += s[t];
                } else {
                    for (int t = 0; t < m; ++t) out[t0 + t][o] = s[t];
                }
            }
        }
    }
}

void pin_physical_core(int phys_idx) noexcept {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#ifdef __linux__
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0 && CPU_ISSET(phys_idx, &allowed)) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(phys_idx, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
#else
    (void) phys_idx;
#endif
}

int detect_physical_cores() noexcept {
    const int n_hw = (int) std::thread::hardware_concurrency();
    if (n_hw <= 1) return 1;
    return (n_hw >= 4 && (n_hw % 2) == 0) ? (n_hw / 2) : n_hw;
}

}  // namespace

namespace avx1 {

void quantize_row_q8_K(const float* x, void* dst, int k) {
    BlockQ8_K* y = static_cast<BlockQ8_K*>(dst);
    const int nb = k / QK_K;
    const __m128 absm = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    const __m128 magic = _mm_set1_ps(12582912.f);
    const __m128i mant = _mm_set1_epi32(0x007fffff), off = _mm_set1_epi32(0x00400000);
    const __m128i c127 = _mm_set1_epi32(127);

    for (int i = 0; i < nb; ++i, x += QK_K) {
        __m128 m = _mm_setzero_ps();
        for (int j = 0; j < QK_K; j += 4) m = _mm_max_ps(_mm_and_ps(_mm_loadu_ps(x + j), absm), m);
        __m128 h = _mm_max_ps(m, _mm_movehl_ps(m, m));
        h = _mm_max_ss(h, _mm_movehdup_ps(h));
        const float amax = _mm_cvtss_f32(h);
        if (!amax) {
            y[i].d = 0.f;
            std::memset(y[i].qs, 0, QK_K);
            std::memset(y[i].bsums, 0, sizeof(y[i].bsums));
            continue;
        }
        float mx = 0.f;
        const __m128 va = _mm_set1_ps(amax);
        for (int j = 0; j < QK_K; j += 4) {
            const int msk = _mm_movemask_ps(_mm_cmpeq_ps(_mm_and_ps(_mm_loadu_ps(x + j), absm), va));
            if (msk) {
                int b = 0;
                while (!((msk >> b) & 1)) ++b;
                mx = x[j + b];
                break;
            }
        }
        const float iscale = -127.f / mx;
        const __m128 vis = _mm_set1_ps(iscale);
        for (int j = 0; j < QK_K; j += 16) {
            __m128i v[4];
            for (int q = 0; q < 4; ++q) {
                __m128 p = _mm_mul_ps(vis, _mm_loadu_ps(x + j + 4 * q));
#if defined(__GNUC__) && !defined(__clang__)
                __asm__("" : "+x"(p));
#endif
                const __m128 t = _mm_add_ps(p, magic);
                v[q] = _mm_min_epi32(_mm_sub_epi32(_mm_and_si128(_mm_castps_si128(t), mant), off), c127);
                v[q] = _mm_srai_epi32(_mm_slli_epi32(v[q], 24), 24);
            }
            __m128i s = _mm_add_epi32(_mm_add_epi32(v[0], v[1]), _mm_add_epi32(v[2], v[3]));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
            y[i].bsums[j / 16] = (int16_t) _mm_cvtsi128_si32(s);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(y[i].qs + j),
                             _mm_packs_epi16(_mm_packs_epi32(v[0], v[1]), _mm_packs_epi32(v[2], v[3])));
        }
        y[i].d = 1.f / iscale;
    }
}

void quantize_row_q8_0(const float* x, void* dst, int k) {
    BlockQ8_0* y = static_cast<BlockQ8_0*>(dst);
    const int nb = k / QK8_0;
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m128i zero128 = _mm_setzero_si128();

    for (int i = 0; i < nb; ++i, x += QK8_0) {
        const __m256 v0 = _mm256_loadu_ps(x + 0);
        const __m256 v1 = _mm256_loadu_ps(x + 8);
        const __m256 v2 = _mm256_loadu_ps(x + 16);
        const __m256 v3 = _mm256_loadu_ps(x + 24);

        __m256 m01 = _mm256_max_ps(_mm256_and_ps(v0, absm), _mm256_and_ps(v1, absm));
        __m256 m23 = _mm256_max_ps(_mm256_and_ps(v2, absm), _mm256_and_ps(v3, absm));
        __m256 m = _mm256_max_ps(m01, m23);
        __m128 mh = _mm_max_ps(_mm256_extractf128_ps(m, 1), _mm256_castps256_ps128(m));
        mh = _mm_max_ps(mh, _mm_movehl_ps(mh, mh));
        mh = _mm_max_ss(mh, _mm_movehdup_ps(mh));
        const float amax = _mm_cvtss_f32(mh);

        if (amax == 0.0f) {
            y[i].d = 0;
            _mm_storeu_si128(reinterpret_cast<__m128i*>(y[i].qs + 0), zero128);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(y[i].qs + 16), zero128);
            continue;
        }
        const float d = amax / 127.0f;
        const float id = 1.0f / d;
        y[i].d = f2h_avx1(d);
        const __m256 vid = _mm256_set1_ps(id);

        const __m256i q0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, vid));
        const __m256i q1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, vid));
        const __m256i q2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, vid));
        const __m256i q3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, vid));

        const __m128i p0 = _mm_packs_epi32(_mm256_castsi256_si128(q0), _mm256_extractf128_si256(q0, 1));
        const __m128i p1 = _mm_packs_epi32(_mm256_castsi256_si128(q1), _mm256_extractf128_si256(q1, 1));
        const __m128i p2 = _mm_packs_epi32(_mm256_castsi256_si128(q2), _mm256_extractf128_si256(q2, 1));
        const __m128i p3 = _mm_packs_epi32(_mm256_castsi256_si128(q3), _mm256_extractf128_si256(q3, 1));

        _mm_storeu_si128(reinterpret_cast<__m128i*>(y[i].qs + 0),  _mm_packs_epi16(p0, p1));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(y[i].qs + 16), _mm_packs_epi16(p2, p3));
    }
}

float vec_dot_q4_K_q8_K(const void* vx, const void* vy, int k) {
    return q4k_dot_1tok_avx1(static_cast<const BlockQ4_K*>(vx), static_cast<const BlockQ8_K*>(vy), k / QK_K);
}

float vec_dot_q6_K_q8_K(const void* vx, const void* vy, int k) {
    return q6k_dot_1tok_avx1(static_cast<const BlockQ6_K*>(vx), static_cast<const BlockQ8_K*>(vy), k / QK_K);
}

float vec_dot_q8_0_q8_0(const void* vx, const void* vy, int k) {
    return q8_0_dot_1tok_avx1(static_cast<const BlockQ8_0*>(vx), static_cast<const BlockQ8_0*>(vy), k / QK8_0);
}

void cold_expert_2half_accum(const uint8_t* gu_half0, const uint8_t* gu_half1,
                             const uint8_t* dn_half0, const uint8_t* dn_half1,
                             uint32_t gu_type, uint32_t dn_type,
                             size_t gu_row_bytes, size_t dn_row_bytes,
                             int n_in, int ff_half,
                             const float* x, float weight, float* out) {
    std::vector<uint8_t> act_buf((gu_type == T_Q8_0) ? (size_t) (n_in / QK8_0) * sizeof(BlockQ8_0)
                                                     : (size_t) (n_in / QK_K) * sizeof(BlockQ8_K));
    if (gu_type == T_Q8_0) quantize_row_q8_0(x, act_buf.data(), n_in);
    else quantize_row_q8_K(x, act_buf.data(), n_in);

    const void* act_ptr = act_buf.data();
    std::vector<float> h0((size_t) ff_half), h1((size_t) ff_half);
    float* h0_p = h0.data();
    float* h1_p = h1.data();
    gu_rows_chunk_avx1(gu_type, gu_half0, gu_half0 + (size_t) ff_half * gu_row_bytes, gu_row_bytes,
                       n_in, &act_ptr, 1, &h0_p, 0, ff_half);
    gu_rows_chunk_avx1(gu_type, gu_half1, gu_half1 + (size_t) ff_half * gu_row_bytes, gu_row_bytes,
                       n_in, &act_ptr, 1, &h1_p, 0, ff_half);

    const size_t hq_bytes = (dn_type == T_Q8_0) ? (size_t) (ff_half / QK8_0) * sizeof(BlockQ8_0)
                                                : (size_t) (ff_half / QK_K) * sizeof(BlockQ8_K);
    std::vector<uint8_t> hq0(hq_bytes), hq1(hq_bytes);
    if (dn_type == T_Q8_0) {
        quantize_row_q8_0(h0.data(), hq0.data(), ff_half);
        quantize_row_q8_0(h1.data(), hq1.data(), ff_half);
    } else {
        quantize_row_q8_K(h0.data(), hq0.data(), ff_half);
        quantize_row_q8_K(h1.data(), hq1.data(), ff_half);
    }

    std::vector<float> tmp((size_t) n_in, 0.0f);
    float* tmp_p = tmp.data();
    const void* hq0_p = hq0.data();
    const void* hq1_p = hq1.data();
    dn_rows_chunk_avx1(dn_type, dn_half0, dn_row_bytes, ff_half, &hq0_p, 1, &tmp_p, 0, n_in, false);
    dn_rows_chunk_avx1(dn_type, dn_half1, dn_row_bytes, ff_half, &hq1_p, 1, &tmp_p, 0, n_in, true);
    for (int i = 0; i < n_in; ++i) out[i] += weight * tmp[(size_t) i];
}

}  // namespace avx1

struct CpuExpertPool::Impl {
    struct Batch {
        const std::function<void(int)>* fn = nullptr;
        int n = 0;
        alignas(64) std::atomic<int> next{0};
        alignas(64) std::atomic<int> left{0};
    };

    const Model& m_;
    int n_threads_ = 12;
    PolyWcSlot* slot_ = nullptr;

    alignas(64) std::atomic<bool> quit_{false};
    alignas(64) std::atomic<uint32_t> posted_{0};
    alignas(64) std::atomic<Batch*> cur_{nullptr};
    alignas(64) std::atomic<int> users_{0};
    alignas(64) std::atomic<int> sleepers_{0};
    std::mutex sleep_mu_;
    std::condition_variable sleep_cv_;

    std::thread poller_;
    std::vector<std::thread> workers_;

    alignas(64) std::atomic<uint64_t> st_calls_{0};
    alignas(64) std::atomic<uint64_t> st_fast_skips_{0};
    alignas(64) std::atomic<uint64_t> st_cold_pairs_{0};
    alignas(64) std::atomic<uint64_t> st_busy_ns_{0};

    alignas(64) float x_l1_[N] = {};
    alignas(64) int32_t cold_tok_[MAXT * TOPK] = {};
    alignas(64) int32_t cold_exp_[MAXT * TOPK] = {};
    alignas(64) float   cold_w_[MAXT * TOPK] = {};
    std::vector<uint8_t> act_q8k_;
    std::vector<uint8_t> act_q80_;
    std::vector<float>   h_half_[2];
    std::vector<uint8_t> hq_half_[2];

    Impl(const Model& m, int req_threads) : m_(m) {
        const int n_phys = detect_physical_cores();
        int nth = req_threads;
        if (nth <= 0) {
            nth = (int) env_int("FNX_CPU_THREADS", std::min(12, n_phys));
        }
        n_threads_ = std::max(1, std::min(nth, n_phys));

        void* p = nullptr;
        const cudaError_t e = cudaHostAlloc(&p, sizeof(PolyWcSlot),
                                            cudaHostAllocPortable | cudaHostAllocMapped | cudaHostAllocWriteCombined);
        if (e != cudaSuccess) {
            cudaGetLastError();
            throw std::runtime_error(std::string("CpuExpertPool: cudaHostAlloc(PolyWcSlot) failed: ") + cudaGetErrorString(e));
        }
        void* dp = nullptr;
        const cudaError_t pe = cudaHostGetDevicePointer(&dp, p, 0);
        if (pe != cudaSuccess || dp != p) {
            cudaFreeHost(p);
            throw std::runtime_error("CpuExpertPool: mapped WC PolyWcSlot requires UVA (dp == p)");
        }
        slot_ = static_cast<PolyWcSlot*>(p);
        std::memset(slot_, 0, sizeof(PolyWcSlot));
        _mm_sfence();

        const size_t q8k_row = (size_t) (N / QK_K) * sizeof(BlockQ8_K);
        const size_t q80_row = (size_t) (N / QK8_0) * sizeof(BlockQ8_0);
        act_q8k_.resize((size_t) MAXT * q8k_row);
        act_q80_.resize((size_t) MAXT * q80_row);
        const size_t hq_row = std::max((size_t) (FF_CARD_MAX / QK8_0) * sizeof(BlockQ8_0),
                                       (size_t) ((FF_CARD_MAX + QK_K - 1) / QK_K) * sizeof(BlockQ8_K));
        for (int h = 0; h < 2; ++h) {
            h_half_[h].resize((size_t) MAXT * TOPK * FF_CARD_MAX);
            hq_half_[h].resize((size_t) MAXT * TOPK * hq_row);
        }

        workers_.reserve((size_t) std::max(0, n_threads_ - 1));
        for (int i = 1; i < n_threads_; ++i) {
            workers_.emplace_back([this, i] {
                pin_physical_core(i);
                worker_loop();
            });
        }
        poller_ = std::thread([this] {
            pin_physical_core(0);
            poller_loop();
        });
        std::fprintf(stderr, "fnx: CpuExpertPool ready: %d physical cores (0..%d, zero SMT), PolyWcSlot=%zu KiB (WC+UVA)\n",
                     n_threads_, n_threads_ - 1, sizeof(PolyWcSlot) >> 10);
    }

    ~Impl() {
        quit_.store(true, std::memory_order_seq_cst);
        posted_.fetch_add(1, std::memory_order_seq_cst);
        {
            std::lock_guard<std::mutex> lk(sleep_mu_);
            sleep_cv_.notify_all();
        }
        if (poller_.joinable()) poller_.join();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
        if (slot_) {
            cudaFreeHost(slot_);
            slot_ = nullptr;
        }
    }

    bool drain(Batch& b) noexcept {
        bool ran = false;
        while (b.next.load(std::memory_order_relaxed) < b.n) {
            const int idx = b.next.fetch_add(1, std::memory_order_relaxed);
            if (idx >= b.n) break;
            (*b.fn)(idx);
            b.left.fetch_sub(1, std::memory_order_release);
            ran = true;
        }
        return ran;
    }

    void worker_loop() noexcept {
        int idle = 0;
        uint32_t seen = posted_.load(std::memory_order_acquire);
        while (!quit_.load(std::memory_order_relaxed)) {
            const uint32_t now = posted_.load(std::memory_order_acquire);
            if (now != seen) {
                bool ran = false;
                if (cur_.load(std::memory_order_acquire) != nullptr) {
                    users_.fetch_add(1, std::memory_order_acq_rel);
                    Batch* b = cur_.load(std::memory_order_acquire);
                    if (b) ran = drain(*b);
                    users_.fetch_sub(1, std::memory_order_acq_rel);
                }
                if (ran) {
                    idle = 0;
                    continue;
                }
                seen = now;
            }
            if (++idle < kSpinsBeforeSleep) {
                _mm_pause();
                continue;
            }
            sleepers_.fetch_add(1, std::memory_order_seq_cst);
            {
                std::unique_lock<std::mutex> lk(sleep_mu_);
                sleep_cv_.wait(lk, [&] {
                    return quit_.load(std::memory_order_acquire) ||
                           posted_.load(std::memory_order_acquire) != seen;
                });
            }
            sleepers_.fetch_sub(1, std::memory_order_seq_cst);
            idle = 0;
        }
    }

    void run_batch(int n, const std::function<void(int)>& fn) noexcept {
        if (n <= 0) return;
        if (n == 1 || workers_.empty()) {
            for (int i = 0; i < n; ++i) fn(i);
            return;
        }
        Batch b;
        b.fn = &fn;
        b.n = n;
        b.left.store(n, std::memory_order_relaxed);
        cur_.store(&b, std::memory_order_release);
        posted_.fetch_add(1, std::memory_order_seq_cst);
        if (sleepers_.load(std::memory_order_seq_cst) > 0) {
            std::lock_guard<std::mutex> lk(sleep_mu_);
            sleep_cv_.notify_all();
        }
        drain(b);
        cur_.store(nullptr, std::memory_order_release);
        while (b.left.load(std::memory_order_acquire) != 0) _mm_pause();
        while (users_.load(std::memory_order_acquire) != 0) _mm_pause();
    }

    void execute_request(uint32_t seq) noexcept {
        const auto t_start = std::chrono::steady_clock::now();
        _mm_lfence();

        const int32_t layer = *reinterpret_cast<const volatile int32_t*>(&slot_->layer);
        const int32_t T = *reinterpret_cast<const volatile int32_t*>(&slot_->T);
        const int32_t n_cold = *reinterpret_cast<const volatile int32_t*>(&slot_->n_cold_pairs);

        if (n_cold <= 0 || T <= 0 || T > MAXT || layer < 0 || layer >= NL || m_.cold_experts.empty()) {
            st_fast_skips_.fetch_add(1, std::memory_order_relaxed);
            _mm_sfence();
            __atomic_store_n(&slot_->done_seq, seq, __ATOMIC_RELEASE);
            _mm_sfence();
            return;
        }

        const int nc = std::min<int>(n_cold, MAXT * TOPK);
        stream_load_wc_i32(cold_tok_, slot_->cold_tok, (size_t) nc);
        stream_load_wc_i32(cold_exp_, slot_->cold_exp, (size_t) nc);
        stream_load_wc_floats(cold_w_, slot_->cold_w, (size_t) nc);

        uint32_t tok_mask = 0u;
        bool need_q8k = false, need_q80 = false;
        const size_t layer_base = (size_t) layer * NEXP;
        for (int j = 0; j < nc; ++j) {
            const int tok = cold_tok_[j];
            const int exp = cold_exp_[j];
            if (tok < 0 || tok >= T || exp < 0 || exp >= NEXP) continue;
            tok_mask |= (1u << tok);
            const ColdExpertInfo& ce = m_.cold_experts[layer_base + (size_t) exp];
            if (ce.gu_type == T_Q8_0) need_q80 = true;
            else need_q8k = true;
        }

        const size_t q8k_row = (size_t) (N / QK_K) * sizeof(BlockQ8_K);
        const size_t q80_row = (size_t) (N / QK8_0) * sizeof(BlockQ8_0);
        for (int t = 0; t < T; ++t) {
            if (!(tok_mask & (1u << t))) continue;
            stream_load_wc_floats(x_l1_, slot_->mixed + (size_t) t * N, (size_t) N);
            if (need_q8k) avx1::quantize_row_q8_K(x_l1_, act_q8k_.data() + (size_t) t * q8k_row, N);
            if (need_q80) avx1::quantize_row_q8_0(x_l1_, act_q80_.data() + (size_t) t * q80_row, N);
        }

        int by[MAXT * TOPK];
        std::iota(by, by + nc, 0);
        std::stable_sort(by, by + nc, [&](int a, int b) { return cold_exp_[a] < cold_exp_[b]; });

        int run[MAXT * TOPK + 1];
        int n_run = 0;
        for (int i = 0; i < nc; ++i) {
            if (i == 0 || cold_exp_[by[i]] != cold_exp_[by[i - 1]]) run[n_run++] = i;
        }
        run[n_run] = nc;

        const ColdExpertInfo& ce0 = m_.cold_experts[layer_base + (size_t) cold_exp_[by[0]]];
        const int ff_half = (ce0.gu_row_bytes > 0) ? (int) (ce0.gu_half_bytes / (2 * ce0.gu_row_bytes)) : (FF / 2);
        const bool inline_q80 = (ce0.dn_type == T_Q8_0) && ((ff_half % kGuChunkRows) == 0);
        const size_t hq_row = (ce0.dn_type == T_Q8_0) ? (size_t) (ff_half / QK8_0) * sizeof(BlockQ8_0)
                                                      : (size_t) (ff_half / QK_K) * sizeof(BlockQ8_K);

        const int chunks_per_half = (ff_half + kGuChunkRows - 1) / kGuChunkRows;
        const int total_gu_tasks = n_run * 2 * chunks_per_half;
        run_batch(total_gu_tasks, [&](int task_id) {
            const int u = task_id / (2 * chunks_per_half);
            const int rem = task_id % (2 * chunks_per_half);
            const int h = rem / chunks_per_half;
            const int sub = rem % chunks_per_half;
            const int r0 = sub * kGuChunkRows;
            const int r1 = std::min(ff_half, r0 + kGuChunkRows);
            if (r0 >= r1) return;

            const int i0 = run[u], i1 = run[u + 1];
            const int nt = i1 - i0;
            const ColdExpertInfo& ce = m_.cold_experts[layer_base + (size_t) cold_exp_[by[i0]]];
            if (!ce.gu_half[h]) return;

            const void* act_ptrs[MAXT * TOPK];
            float* ff_ptrs[MAXT * TOPK];
            const bool is_q80 = (ce.gu_type == T_Q8_0);
            for (int k = 0; k < nt; ++k) {
                const int j = by[i0 + k];
                const int tok = cold_tok_[j];
                act_ptrs[k] = is_q80 ? (act_q80_.data() + (size_t) tok * q80_row)
                                     : (act_q8k_.data() + (size_t) tok * q8k_row);
                ff_ptrs[k] = h_half_[h].data() + (size_t) j * (size_t) ff_half;
            }
            const uint8_t* gate_base = ce.gu_half[h];
            const uint8_t* up_base   = ce.gu_half[h] + (size_t) ff_half * ce.gu_row_bytes;
            gu_rows_chunk_avx1(ce.gu_type, gate_base, up_base, ce.gu_row_bytes, N, act_ptrs, nt, ff_ptrs, r0, r1);

            if (inline_q80 && ce.dn_type == T_Q8_0) {
                const size_t blk_off = (size_t) (r0 / QK8_0) * sizeof(BlockQ8_0);
                for (int k = 0; k < nt; ++k) {
                    const int j = by[i0 + k];
                    avx1::quantize_row_q8_0(ff_ptrs[k] + r0,
                                            hq_half_[h].data() + (size_t) j * hq_row + blk_off,
                                            r1 - r0);
                }
            }
        });

        if (!inline_q80) {
            run_batch(nc * 2, [&](int idx) {
                const int j = idx >> 1;
                const int h = idx & 1;
                const ColdExpertInfo& ce = m_.cold_experts[layer_base + (size_t) cold_exp_[j]];
                const float* src = h_half_[h].data() + (size_t) j * (size_t) ff_half;
                uint8_t* dst = hq_half_[h].data() + (size_t) j * hq_row;
                if (ce.dn_type == T_Q8_0) avx1::quantize_row_q8_0(src, dst, ff_half);
                else avx1::quantize_row_q8_K(src, dst, ff_half);
            });
        }

        const int d_tasks = (N + kDnChunkRows - 1) / kDnChunkRows;
        run_batch(d_tasks, [&](int task_id) {
            const int r0 = task_id * kDnChunkRows;
            const int r1 = std::min(N, r0 + kDnChunkRows);
            const int nr = r1 - r0;
            if (nr <= 0) return;

            alignas(32) float s_buf[MAXT * TOPK * kDnChunkRows];
            const void* hq_ptrs0[MAXT * TOPK];
            const void* hq_ptrs1[MAXT * TOPK];
            float* out_ptrs[MAXT * TOPK];

            for (int u = 0; u < n_run; ++u) {
                const int i0 = run[u], i1 = run[u + 1];
                const int nt = i1 - i0;
                const ColdExpertInfo& ce = m_.cold_experts[layer_base + (size_t) cold_exp_[by[i0]]];
                for (int k = 0; k < nt; ++k) {
                    const int j = by[i0 + k];
                    hq_ptrs0[k] = hq_half_[0].data() + (size_t) j * hq_row;
                    hq_ptrs1[k] = hq_half_[1].data() + (size_t) j * hq_row;
                    out_ptrs[k] = s_buf + (size_t) j * (size_t) nr;
                }
                if (ce.dn_half[0]) {
                    dn_rows_chunk_avx1(ce.dn_type, ce.dn_half[0], ce.dn_row_bytes, ff_half,
                                       hq_ptrs0, nt, out_ptrs, r0, r1, false);
                } else {
                    for (int k = 0; k < nt; ++k) {
                        std::memset(s_buf + (size_t) by[i0 + k] * (size_t) nr, 0, (size_t) nr * sizeof(float));
                    }
                }
                if (ce.dn_half[1]) {
                    dn_rows_chunk_avx1(ce.dn_type, ce.dn_half[1], ce.dn_row_bytes, ff_half,
                                       hq_ptrs1, nt, out_ptrs, r0, r1, true);
                }
            }

            for (int t = 0; t < T; ++t) {
                float* dst = slot_->out + (size_t) t * N + r0;
                if (!(tok_mask & (1u << t))) {
                    for (int r = 0; r < nr; ++r) dst[r] = 0.0f;
                    continue;
                }
                alignas(32) float acc[kDnChunkRows] = {};
                for (int j = 0; j < nc; ++j) {
                    if (cold_tok_[j] != t) continue;
                    const float w = cold_w_[j];
                    const float* sj = s_buf + (size_t) j * (size_t) nr;
                    for (int r = 0; r < nr; ++r) acc[r] += w * sj[r];
                }
                for (int r = 0; r < nr; ++r) dst[r] = acc[r];
            }
            _mm_sfence();
        });

        _mm_sfence();
        __atomic_store_n(&slot_->done_seq, seq, __ATOMIC_RELEASE);
        _mm_sfence();

        const auto dt_ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t_start).count();
        st_calls_.fetch_add(1, std::memory_order_relaxed);
        st_cold_pairs_.fetch_add((uint64_t) nc, std::memory_order_relaxed);
        st_busy_ns_.fetch_add(dt_ns, std::memory_order_relaxed);
    }

    void poller_loop() noexcept {
        uint32_t last_seq = 0;
        int idle = 0;
        while (!quit_.load(std::memory_order_relaxed)) {
            const uint32_t seq = __atomic_load_n(&slot_->req_seq, __ATOMIC_ACQUIRE);
            if (seq != 0 && seq != last_seq) {
                last_seq = seq;
                execute_request(seq);
                idle = 0;
                continue;
            }
            if (++idle < kSpinsBeforeSleep) {
                _mm_pause();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(25));
            }
        }
    }
};

CpuExpertPool::CpuExpertPool(const Model& m, int n_threads) : impl_(new Impl(m, n_threads)) {}

CpuExpertPool::~CpuExpertPool() {
    delete impl_;
    impl_ = nullptr;
}

PolyWcSlot* CpuExpertPool::device_slot() const { return impl_->slot_; }

void CpuExpertPool::reset_stats() {
    impl_->st_calls_.store(0, std::memory_order_relaxed);
    impl_->st_fast_skips_.store(0, std::memory_order_relaxed);
    impl_->st_cold_pairs_.store(0, std::memory_order_relaxed);
    impl_->st_busy_ns_.store(0, std::memory_order_relaxed);
}

CpuExpertPool::Stats CpuExpertPool::stats() const {
    Stats s;
    s.calls = impl_->st_calls_.load(std::memory_order_relaxed);
    s.fast_skips = impl_->st_fast_skips_.load(std::memory_order_relaxed);
    s.cold_pairs = impl_->st_cold_pairs_.load(std::memory_order_relaxed);
    s.busy_ns = impl_->st_busy_ns_.load(std::memory_order_relaxed);
    return s;
}

}  // namespace fnx
