// Standalone parity unit test for Ivy Bridge AVX1 + F16C K-quant / Q8_0 kernels
// and full 2-half SwiGLU cold-expert pass (src/kq_avx1.cpp).
// Does NOT require a GGUF file or GPU; verifies against exact scalar FP64/FP32 GGML reference.
#include "kq_avx1.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int QK8_0 = 32;
constexpr int QK_K = 256;

struct BlockQ8_0 {
    uint16_t d;
    int8_t   qs[QK8_0];
};
static_assert(sizeof(BlockQ8_0) == 34, "BlockQ8_0 size");

struct BlockQ4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];
};
static_assert(sizeof(BlockQ4_K) == 144, "BlockQ4_K size");

struct BlockQ6_K {
    uint8_t  ql[QK_K / 2];
    uint8_t  qh[QK_K / 4];
    int8_t   scales[QK_K / 16];
    uint16_t d;
};
static_assert(sizeof(BlockQ6_K) == 210, "BlockQ6_K size");

struct BlockQ8_K {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
};
static_assert(sizeof(BlockQ8_K) == 292, "BlockQ8_K size");

// Verify PolyWcSlot cache-line and MOVNTDQA 16-byte alignment invariants.
static_assert(alignof(fnx::PolyWcSlot) == 64, "PolyWcSlot must be 64B aligned");
static_assert(offsetof(fnx::PolyWcSlot, done_seq) - offsetof(fnx::PolyWcSlot, req_seq) >= 64,
              "req_seq and done_seq must sit on distinct 64B cache lines");
static_assert((offsetof(fnx::PolyWcSlot, cold_tok) % 16) == 0, "cold_tok must be 16B aligned for MOVNTDQA");
static_assert((offsetof(fnx::PolyWcSlot, cold_exp) % 16) == 0, "cold_exp must be 16B aligned for MOVNTDQA");
static_assert((offsetof(fnx::PolyWcSlot, cold_w)   % 16) == 0, "cold_w must be 16B aligned for MOVNTDQA");
static_assert((offsetof(fnx::PolyWcSlot, mixed)    % 64) == 0, "mixed must be 64B aligned");
static_assert((offsetof(fnx::PolyWcSlot, out)      % 64) == 0, "out must be 64B aligned");

// Pure scalar IEEE-754 binary16 <-> binary32 (independent of F16C intrinsics).
float fp16_to_fp32(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t bits = 0;
    if (e == 0) {
        if (m == 0) {
            bits = s;
        } else {
            int shift = 0;
            while ((m & 0x400u) == 0) { m <<= 1; ++shift; }
            m &= 0x3FFu;
            bits = s | ((uint32_t) (127 - 14 - shift) << 23) | (m << 13);
        }
    } else if (e == 31) {
        bits = s | (0xFFu << 23) | (m << 13);
    } else {
        bits = s | ((e + 112u) << 23) | (m << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

uint16_t fp32_to_fp16(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint16_t s = (uint16_t) ((bits >> 16) & 0x8000u);
    int32_t e = (int32_t) ((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t m = bits & 0x7FFFFFu;
    if (e <= 0) {
        if (e < -10) return s;
        m |= 0x800000u;
        const uint32_t shift = (uint32_t) (14 - e);
        uint32_t r = m >> shift;
        if ((m >> (shift - 1)) & 1u) ++r;
        return (uint16_t) (s | r);
    }
    if (e >= 31) return (uint16_t) (s | 0x7C00u);
    const uint32_t round_bit = (m >> 12) & 1u;
    const uint32_t sticky = (m & 0xFFFu) != 0 ? 1u : 0u;
    uint32_t mant = m >> 13;
    if (round_bit && (sticky || (mant & 1u))) {
        ++mant;
        if (mant == 0x400u) { mant = 0; ++e; }
    }
    if (e >= 31) return (uint16_t) (s | 0x7C00u);
    return (uint16_t) (s | ((uint16_t) e << 10) | (uint16_t) mant);
}

// Pure scalar GGML reference quantizers (independent of AVX1/SSE/F16C intrinsics).
void quantize_row_q8_K_ref(const float* x, BlockQ8_K* y, int k) {
    const int nb = k / QK_K;
    for (int i = 0; i < nb; ++i, x += QK_K) {
        float max_val = 0.0f;
        float amax = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            const float ax = std::fabs(x[j]);
            if (ax > amax) { amax = ax; max_val = x[j]; }
        }
        if (amax == 0.0f) {
            y[i].d = 0.0f;
            std::memset(y[i].qs, 0, QK_K);
            std::memset(y[i].bsums, 0, sizeof(y[i].bsums));
            continue;
        }
        const float iscale = -127.0f / max_val;
        for (int j = 0; j < QK_K; ++j) {
            const int v = (int) std::nearbyintf(iscale * x[j]);
            y[i].qs[j] = (int8_t) std::min(127, v);
        }
        for (int j = 0; j < QK_K / 16; ++j) {
            int sum = 0;
            for (int l = 0; l < 16; ++l) sum += y[i].qs[j * 16 + l];
            y[i].bsums[j] = (int16_t) sum;
        }
        y[i].d = 1.0f / iscale;
    }
}

void quantize_row_q8_0_ref(const float* x, BlockQ8_0* y, int k) {
    const int nb = k / QK8_0;
    for (int i = 0; i < nb; ++i, x += QK8_0) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) amax = std::max(amax, std::fabs(x[j]));
        if (amax == 0.0f) {
            y[i].d = 0;
            std::memset(y[i].qs, 0, QK8_0);
            continue;
        }
        const float d = amax / 127.0f;
        const float id = 1.0f / d;
        y[i].d = fp32_to_fp16(d);
        for (int j = 0; j < QK8_0; ++j) {
            const int v = (int) std::nearbyintf(x[j] * id);
            y[i].qs[j] = (int8_t) std::max(-128, std::min(127, v));
        }
    }
}

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (uint8_t) ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = (uint8_t) ((q[j + 4] >> 4)  | ((q[j]     >> 6) << 4));
    }
}

void dequant_row_q4_K(const BlockQ4_K* x, float* y, int k) {
    const int nb = k / QK_K;
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        const float dmin = fp16_to_fp32(x[i].dmin);
        const uint8_t* q = x[i].qs;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc1 = 0, m1 = 0, sc2 = 0, m2 = 0;
            get_scale_min_k4(is + 0, x[i].scales, sc1, m1);
            get_scale_min_k4(is + 1, x[i].scales, sc2, m2);
            const float d1 = d * sc1, min1 = dmin * m1;
            const float d2 = d * sc2, min2 = dmin * m2;
            for (int l = 0; l < 32; ++l) y[i * QK_K + j + l]      = d1 * (float) (q[l] & 0xF) - min1;
            for (int l = 0; l < 32; ++l) y[i * QK_K + j + 32 + l] = d2 * (float) (q[l] >> 4)  - min2;
            q += 32;
            is += 2;
        }
    }
}

void dequant_row_q6_K(const BlockQ6_K* x, float* y, int k) {
    const int nb = k / QK_K;
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        for (int n = 0; n < QK_K; n += 128) {
            const uint8_t* ql = x[i].ql + n / 2;
            const uint8_t* qh = x[i].qh + n / 4;
            const int8_t*  sc = x[i].scales + n / 16;
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t) (((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                const int8_t q2 = (int8_t) (((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                const int8_t q3 = (int8_t) (((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                const int8_t q4 = (int8_t) (((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32);
                y[i * QK_K + n + l +  0] = d * (float) sc[is + 0] * (float) q1;
                y[i * QK_K + n + l + 32] = d * (float) sc[is + 2] * (float) q2;
                y[i * QK_K + n + l + 64] = d * (float) sc[is + 4] * (float) q3;
                y[i * QK_K + n + l + 96] = d * (float) sc[is + 6] * (float) q4;
            }
        }
    }
}

void dequant_row_q8_0(const BlockQ8_0* x, float* y, int k) {
    const int nb = k / QK8_0;
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        for (int l = 0; l < QK8_0; ++l) y[i * QK8_0 + l] = d * (float) x[i].qs[l];
    }
}

void dequant_row_q8_K(const BlockQ8_K* x, float* y, int k) {
    const int nb = k / QK_K;
    for (int i = 0; i < nb; ++i) {
        for (int l = 0; l < QK_K; ++l) y[i * QK_K + l] = x[i].d * (float) x[i].qs[l];
    }
}

void fill_random_q4_K(std::mt19937& rng, std::vector<BlockQ4_K>& blocks, float scale) {
    std::uniform_real_distribution<float> dist_d(0.25f * scale, 1.0f * scale);
    std::uniform_int_distribution<int> dist_byte(0, 255);
    for (auto& b : blocks) {
        b.d = fp32_to_fp16(dist_d(rng));
        b.dmin = fp32_to_fp16(dist_d(rng) * 7.5f);
        for (int i = 0; i < 12; ++i) b.scales[i] = (uint8_t) dist_byte(rng);
        for (int i = 0; i < QK_K / 2; ++i) b.qs[i] = (uint8_t) dist_byte(rng);
    }
}

void fill_random_q6_K(std::mt19937& rng, std::vector<BlockQ6_K>& blocks, float scale) {
    std::uniform_real_distribution<float> dist_d(0.25f * scale, 1.0f * scale);
    std::uniform_int_distribution<int> dist_byte(0, 255);
    std::uniform_int_distribution<int> dist_sc(-32, 32);
    for (auto& b : blocks) {
        b.d = fp32_to_fp16(dist_d(rng));
        for (int i = 0; i < QK_K / 2; ++i) b.ql[i] = (uint8_t) dist_byte(rng);
        for (int i = 0; i < QK_K / 4; ++i) b.qh[i] = (uint8_t) dist_byte(rng);
        for (int i = 0; i < QK_K / 16; ++i) b.scales[i] = (int8_t) dist_sc(rng);
    }
}

void fill_random_q8_0(std::mt19937& rng, std::vector<BlockQ8_0>& blocks, float scale) {
    std::uniform_real_distribution<float> dist_d(0.25f * scale, 1.0f * scale);
    std::uniform_int_distribution<int> dist_q(-127, 127);
    for (auto& b : blocks) {
        b.d = fp32_to_fp16(dist_d(rng));
        for (int i = 0; i < QK8_0; ++i) b.qs[i] = (int8_t) dist_q(rng);
    }
}

struct Metrics {
    double max_abs_err = 0.0;
    double cos_sim = 1.0;
};

Metrics compare_vecs(const std::vector<float>& got, const std::vector<float>& ref) {
    double max_err = 0.0;
    double dot = 0.0, ng = 0.0, nr = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double g = got[i], r = ref[i];
        max_err = std::max(max_err, std::abs(g - r));
        dot += g * r;
        ng += g * g;
        nr += r * r;
    }
    Metrics m;
    m.max_abs_err = max_err;
    m.cos_sim = (ng > 0.0 && nr > 0.0) ? (dot / (std::sqrt(ng) * std::sqrt(nr))) : 1.0;
    return m;
}

double dot_fp64(const float* a, const float* b, int k) {
    double s = 0.0;
    for (int i = 0; i < k; ++i) s += (double) a[i] * (double) b[i];
    return s;
}

bool test_dot_kernels(std::mt19937& rng) {
    bool ok = true;
    constexpr int kRows = 64;
    const int dims[] = {2560, 5120};

    for (int k : dims) {
        std::uniform_real_distribution<float> dist_x(-0.25f, 0.25f);
        std::vector<float> x((size_t) k);
        for (int i = 0; i < k; ++i) x[(size_t) i] = dist_x(rng);

        // Verify AVX1 quantizers bit-exact against scalar GGML reference quantizers
        std::vector<BlockQ8_K> x_q8k((size_t) (k / QK_K)), x_q8k_ref((size_t) (k / QK_K));
        std::vector<BlockQ8_0> x_q80((size_t) (k / QK8_0)), x_q80_ref((size_t) (k / QK8_0));
        fnx::avx1::quantize_row_q8_K(x.data(), x_q8k.data(), k);
        quantize_row_q8_K_ref(x.data(), x_q8k_ref.data(), k);
        fnx::avx1::quantize_row_q8_0(x.data(), x_q80.data(), k);
        quantize_row_q8_0_ref(x.data(), x_q80_ref.data(), k);

        const bool q8k_exact = (std::memcmp(x_q8k.data(), x_q8k_ref.data(), x_q8k.size() * sizeof(BlockQ8_K)) == 0);
        const bool q80_exact = (std::memcmp(x_q80.data(), x_q80_ref.data(), x_q80.size() * sizeof(BlockQ8_0)) == 0);
        std::printf("Quantizers  (k=%4d): Q8_K_bit_exact=%d  Q8_0_bit_exact=%d  [%s]\n",
                    k, (int) q8k_exact, (int) q80_exact, (q8k_exact && q80_exact) ? "PASS" : "FAIL");
        ok &= (q8k_exact && q80_exact);

        std::vector<float> x_q8k_f((size_t) k), x_q80_f((size_t) k);
        dequant_row_q8_K(x_q8k.data(), x_q8k_f.data(), k);
        dequant_row_q8_0(x_q80.data(), x_q80_f.data(), k);

        // 1. Q4_K x Q8_K
        {
            const int nb = k / QK_K;
            std::vector<BlockQ4_K> w((size_t) kRows * (size_t) nb);
            fill_random_q4_K(rng, w, 4.0e-5f);
            std::vector<float> got(kRows), ref_exact(kRows), ref_fp32(kRows), wf((size_t) k);
            for (int r = 0; r < kRows; ++r) {
                const BlockQ4_K* row = w.data() + (size_t) r * (size_t) nb;
                dequant_row_q4_K(row, wf.data(), k);
                got[(size_t) r]       = fnx::avx1::vec_dot_q4_K_q8_K(row, x_q8k.data(), k);
                ref_exact[(size_t) r] = (float) dot_fp64(wf.data(), x_q8k_f.data(), k);
                ref_fp32[(size_t) r]  = (float) dot_fp64(wf.data(), x.data(), k);
            }
            const Metrics me = compare_vecs(got, ref_exact);
            const Metrics mf = compare_vecs(got, ref_fp32);
            const bool pass = (me.max_abs_err < 1e-5) && (me.cos_sim > 0.99999999) &&
                              (mf.max_abs_err < 1e-3) && (mf.cos_sim > 0.99999);
            std::printf("Q4_K x Q8_K (k=%4d): exact_err=%.3e (cos=%.8f) | e2e_fp32_err=%.3e (cos=%.8f) [%s]\n",
                        k, me.max_abs_err, me.cos_sim, mf.max_abs_err, mf.cos_sim, pass ? "PASS" : "FAIL");
            ok &= pass;
        }

        // 2. Q6_K x Q8_K
        {
            const int nb = k / QK_K;
            std::vector<BlockQ6_K> w((size_t) kRows * (size_t) nb);
            fill_random_q6_K(rng, w, 2.0e-5f);
            std::vector<float> got(kRows), ref_exact(kRows), ref_fp32(kRows), wf((size_t) k);
            for (int r = 0; r < kRows; ++r) {
                const BlockQ6_K* row = w.data() + (size_t) r * (size_t) nb;
                dequant_row_q6_K(row, wf.data(), k);
                got[(size_t) r]       = fnx::avx1::vec_dot_q6_K_q8_K(row, x_q8k.data(), k);
                ref_exact[(size_t) r] = (float) dot_fp64(wf.data(), x_q8k_f.data(), k);
                ref_fp32[(size_t) r]  = (float) dot_fp64(wf.data(), x.data(), k);
            }
            const Metrics me = compare_vecs(got, ref_exact);
            const Metrics mf = compare_vecs(got, ref_fp32);
            const bool pass = (me.max_abs_err < 1e-5) && (me.cos_sim > 0.99999999) &&
                              (mf.max_abs_err < 1e-3) && (mf.cos_sim > 0.99999);
            std::printf("Q6_K x Q8_K (k=%4d): exact_err=%.3e (cos=%.8f) | e2e_fp32_err=%.3e (cos=%.8f) [%s]\n",
                        k, me.max_abs_err, me.cos_sim, mf.max_abs_err, mf.cos_sim, pass ? "PASS" : "FAIL");
            ok &= pass;
        }

        // 3. Q8_0 x Q8_0
        {
            const int nb = k / QK8_0;
            std::vector<BlockQ8_0> w((size_t) kRows * (size_t) nb);
            fill_random_q8_0(rng, w, 1.0e-4f);
            std::vector<float> got(kRows), ref_exact(kRows), ref_fp32(kRows), wf((size_t) k);
            for (int r = 0; r < kRows; ++r) {
                const BlockQ8_0* row = w.data() + (size_t) r * (size_t) nb;
                dequant_row_q8_0(row, wf.data(), k);
                got[(size_t) r]       = fnx::avx1::vec_dot_q8_0_q8_0(row, x_q80.data(), k);
                ref_exact[(size_t) r] = (float) dot_fp64(wf.data(), x_q80_f.data(), k);
                ref_fp32[(size_t) r]  = (float) dot_fp64(wf.data(), x.data(), k);
            }
            const Metrics me = compare_vecs(got, ref_exact);
            const Metrics mf = compare_vecs(got, ref_fp32);
            const bool pass = (me.max_abs_err < 1e-5) && (me.cos_sim > 0.99999999) &&
                              (mf.max_abs_err < 1e-3) && (mf.cos_sim > 0.99999);
            std::printf("Q8_0 x Q8_0 (k=%4d): exact_err=%.3e (cos=%.8f) | e2e_fp32_err=%.3e (cos=%.8f) [%s]\n",
                        k, me.max_abs_err, me.cos_sim, mf.max_abs_err, mf.cos_sim, pass ? "PASS" : "FAIL");
            ok &= pass;
        }
    }

    // 4. All-zero input edge case (must yield exact 0.0f without NaN/Inf)
    {
        constexpr int k = 2560;
        std::vector<float> zeros(k, 0.0f);
        std::vector<BlockQ8_K> z_q8k((size_t) (k / QK_K));
        std::vector<BlockQ8_0> z_q80((size_t) (k / QK8_0));
        fnx::avx1::quantize_row_q8_K(zeros.data(), z_q8k.data(), k);
        fnx::avx1::quantize_row_q8_0(zeros.data(), z_q80.data(), k);
        std::vector<BlockQ4_K> w4((size_t) (k / QK_K));
        std::vector<BlockQ6_K> w6((size_t) (k / QK_K));
        std::vector<BlockQ8_0> w8((size_t) (k / QK8_0));
        fill_random_q4_K(rng, w4, 1e-3f);
        fill_random_q6_K(rng, w6, 1e-3f);
        fill_random_q8_0(rng, w8, 1e-3f);
        const float d4 = fnx::avx1::vec_dot_q4_K_q8_K(w4.data(), z_q8k.data(), k);
        const float d6 = fnx::avx1::vec_dot_q6_K_q8_K(w6.data(), z_q8k.data(), k);
        const float d8 = fnx::avx1::vec_dot_q8_0_q8_0(w8.data(), z_q80.data(), k);
        const bool pass = (d4 == 0.0f && d6 == 0.0f && d8 == 0.0f);
        std::printf("Zero-input edge case: q4k=%.1f q6k=%.1f q80=%.1f  [%s]\n",
                    d4, d6, d8, pass ? "PASS" : "FAIL");
        ok &= pass;
    }

    return ok;
}

bool test_2half_swiglu(std::mt19937& rng, const char* label, uint32_t gu_type) {
    constexpr int kN = 2560;
    constexpr int kFFHalf = 320;
    constexpr float kWeight = 0.37f;

    std::uniform_real_distribution<float> dist_x(-0.25f, 0.25f);
    std::vector<float> x(kN), init_out(kN);
    for (int i = 0; i < kN; ++i) {
        x[(size_t) i] = dist_x(rng);
        init_out[(size_t) i] = 0.05f * dist_x(rng);
    }

    const size_t gu_row_bytes =
        (gu_type == fnx::T_Q4_K) ? (size_t) (kN / QK_K) * sizeof(BlockQ4_K) :
        (gu_type == fnx::T_Q6_K) ? (size_t) (kN / QK_K) * sizeof(BlockQ6_K) :
                                   (size_t) (kN / QK8_0) * sizeof(BlockQ8_0);
    const size_t dn_row_bytes = (size_t) (kFFHalf / QK8_0) * sizeof(BlockQ8_0);

    std::vector<uint8_t> gu_h0((size_t) (2 * kFFHalf) * gu_row_bytes);
    std::vector<uint8_t> gu_h1((size_t) (2 * kFFHalf) * gu_row_bytes);
    std::vector<BlockQ8_0> dn_h0((size_t) kN * (size_t) (kFFHalf / QK8_0));
    std::vector<BlockQ8_0> dn_h1((size_t) kN * (size_t) (kFFHalf / QK8_0));

    auto init_gu = [&](std::vector<uint8_t>& buf) {
        if (gu_type == fnx::T_Q4_K) {
            std::vector<BlockQ4_K> tmp(buf.size() / sizeof(BlockQ4_K));
            fill_random_q4_K(rng, tmp, 4.0e-5f);
            std::memcpy(buf.data(), tmp.data(), buf.size());
        } else if (gu_type == fnx::T_Q6_K) {
            std::vector<BlockQ6_K> tmp(buf.size() / sizeof(BlockQ6_K));
            fill_random_q6_K(rng, tmp, 2.0e-5f);
            std::memcpy(buf.data(), tmp.data(), buf.size());
        } else {
            std::vector<BlockQ8_0> tmp(buf.size() / sizeof(BlockQ8_0));
            fill_random_q8_0(rng, tmp, 1.0e-4f);
            std::memcpy(buf.data(), tmp.data(), buf.size());
        }
    };
    init_gu(gu_h0);
    init_gu(gu_h1);
    fill_random_q8_0(rng, dn_h0, 2.5e-4f);
    fill_random_q8_0(rng, dn_h1, 2.5e-4f);

    std::vector<float> got_acc = init_out;
    fnx::avx1::cold_expert_2half_accum(
        gu_h0.data(), gu_h1.data(),
        reinterpret_cast<const uint8_t*>(dn_h0.data()),
        reinterpret_cast<const uint8_t*>(dn_h1.data()),
        gu_type, fnx::T_Q8_0,
        gu_row_bytes, dn_row_bytes,
        kN, kFFHalf,
        x.data(), kWeight, got_acc.data());

    // Isolate the pure expert contribution (delta = got_acc - init_out) so init_out cannot mask cosine!
    std::vector<float> got_delta(kN);
    for (int i = 0; i < kN; ++i) got_delta[(size_t) i] = got_acc[(size_t) i] - init_out[(size_t) i];

    auto dequant_gu_row = [&](const uint8_t* row, float* out) {
        if (gu_type == fnx::T_Q4_K) dequant_row_q4_K(reinterpret_cast<const BlockQ4_K*>(row), out, kN);
        else if (gu_type == fnx::T_Q6_K) dequant_row_q6_K(reinterpret_cast<const BlockQ6_K*>(row), out, kN);
        else dequant_row_q8_0(reinterpret_cast<const BlockQ8_0*>(row), out, kN);
    };

    // Reference A: exact scalar GGML quantized-activation pipeline (verifies zero SIMD drift)
    std::vector<float> x_q_f(kN);
    if (gu_type == fnx::T_Q8_0) {
        std::vector<BlockQ8_0> xq((size_t) (kN / QK8_0));
        quantize_row_q8_0_ref(x.data(), xq.data(), kN);
        dequant_row_q8_0(xq.data(), x_q_f.data(), kN);
    } else {
        std::vector<BlockQ8_K> xq((size_t) (kN / QK_K));
        quantize_row_q8_K_ref(x.data(), xq.data(), kN);
        dequant_row_q8_K(xq.data(), x_q_f.data(), kN);
    }

    std::vector<float> wf(kN), df(kFFHalf);
    std::vector<float> h_q_f[2];
    std::vector<double> h_fp64[2];
    const uint8_t* gu_halves[2] = {gu_h0.data(), gu_h1.data()};
    for (int h = 0; h < 2; ++h) {
        std::vector<float> h_scalar(kFFHalf);
        h_fp64[h].resize(kFFHalf);
        for (int r = 0; r < kFFHalf; ++r) {
            dequant_gu_row(gu_halves[h] + (size_t) r * gu_row_bytes, wf.data());
            const double g_q = dot_fp64(wf.data(), x_q_f.data(), kN);
            const double g_f = dot_fp64(wf.data(), x.data(), kN);
            dequant_gu_row(gu_halves[h] + (size_t) (kFFHalf + r) * gu_row_bytes, wf.data());
            const double u_q = dot_fp64(wf.data(), x_q_f.data(), kN);
            const double u_f = dot_fp64(wf.data(), x.data(), kN);
            h_scalar[(size_t) r] = (float) ((g_q / (1.0 + std::exp(-g_q))) * u_q);
            h_fp64[h][(size_t) r] = (g_f / (1.0 + std::exp(-g_f))) * u_f;
        }
        std::vector<BlockQ8_0> hq((size_t) (kFFHalf / QK8_0));
        quantize_row_q8_0_ref(h_scalar.data(), hq.data(), kFFHalf);
        h_q_f[h].resize(kFFHalf);
        dequant_row_q8_0(hq.data(), h_q_f[h].data(), kFFHalf);
    }

    std::vector<float> ref_exact_delta(kN), ref_fp32_delta(kN);
    const BlockQ8_0* dn_halves[2] = {dn_h0.data(), dn_h1.data()};
    const int nb_dn = kFFHalf / QK8_0;
    for (int i = 0; i < kN; ++i) {
        double sum_q = 0.0, sum_f = 0.0;
        for (int h = 0; h < 2; ++h) {
            dequant_row_q8_0(dn_halves[h] + (size_t) i * (size_t) nb_dn, df.data(), kFFHalf);
            for (int r = 0; r < kFFHalf; ++r) {
                sum_q += (double) df[(size_t) r] * (double) h_q_f[h][(size_t) r];
                sum_f += (double) df[(size_t) r] * h_fp64[h][(size_t) r];
            }
        }
        ref_exact_delta[(size_t) i] = (float) ((double) kWeight * sum_q);
        ref_fp32_delta[(size_t) i]  = (float) ((double) kWeight * sum_f);
    }

    const Metrics me = compare_vecs(got_delta, ref_exact_delta);
    const Metrics mf = compare_vecs(got_delta, ref_fp32_delta);
    // Exact quantized-activation SIMD parity must match FP64 reference to <1e-5 (cos > 0.9999999);
    // end-to-end vs unquantized FP32 activations reflects 2-stage (input Q8 + heavy-tailed SwiGLU Q8_0)
    // quantization floor (~0.999955).
    const bool pass = (me.max_abs_err < 1e-5) && (me.cos_sim > 0.9999999) &&
                      (mf.max_abs_err < 1e-3) && (mf.cos_sim > 0.9999);
    std::printf("2-half SwiGLU %-10s (N=%d, FF/2=%d): exact_err=%.3e (cos=%.8f) | e2e_fp32_err=%.3e (cos=%.8f) [%s]\n",
                label, kN, kFFHalf, me.max_abs_err, me.cos_sim, mf.max_abs_err, mf.cos_sim, pass ? "PASS" : "FAIL");
    return pass;
}

}  // namespace

int main() {
    std::mt19937 rng(0x5F6E7831u);
    bool ok = true;
    ok &= test_dot_kernels(rng);
    ok &= test_2half_swiglu(rng, "Q4_K+Q8_0", fnx::T_Q4_K);
    ok &= test_2half_swiglu(rng, "Q6_K+Q8_0", fnx::T_Q6_K);
    ok &= test_2half_swiglu(rng, "Q8_0+Q8_0", fnx::T_Q8_0);
    if (!ok) {
        std::fprintf(stderr, "test_kq_avx1: FAILED\n");
        return 1;
    }
    std::printf("test_kq_avx1: ALL TESTS PASSED\n");
    return 0;
}
