// Lossless 6+2 Interval Split-Plane LM Head + Clustered MTP Fast-Head for Pascal (sm_61)
// Included by src/kernels.cu under -DFNX_KPART=9. Self-contained to isolate SASS/stack inspection.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <sm_61_intrinsics.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "kernels.hpp"

namespace fnx {

// Defined in build/kernels_p0.o (src/kernels.cu)
struct Q8Twin {
    const char* q8;
    size_t bytes;
    int8_t* tq;
    float* td;
};
std::vector<Q8Twin>& q8_twins();

namespace {

constexpr int VH_WARPS = 4;
constexpr int VH_THREADS = VH_WARPS * 32;
constexpr size_t VH_Q8_1_BYTES = 36;  // ggml block_q8_1: half2 ds (4B) + int8_t qs[32]

inline void vh_ck(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

inline void vh_twin_of(const void* y, int n, int8_t*& tq, float*& td) {
    tq = nullptr;
    td = nullptr;
    for (const Q8Twin& tw : q8_twins()) {
        if (static_cast<const char*>(y) >= tw.q8 && static_cast<const char*>(y) < tw.q8 + tw.bytes) {
            const size_t off = static_cast<size_t>(static_cast<const char*>(y) - tw.q8);
            const size_t rb = static_cast<size_t>(n / 32) * VH_Q8_1_BYTES;
            if (off % rb) throw std::runtime_error("vhead_6p2 twin: q8_1 pointer between rows");
            const size_t r = off / rb;
            tq = tw.tq + r * n;
            td = tw.td + r * (n / 32);
            return;
        }
    }
}

inline float vh_f16_to_f32_host(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t frac = h & 0x3FFu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (frac == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((frac & 0x400u) == 0) {
                frac <<= 1;
                --e;
            }
            frac &= 0x3FFu;
            bits = sign | (static_cast<uint32_t>(e + 127) << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (frac << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (frac << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

__device__ __forceinline__ float vh_half_bits_to_f32(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

__device__ __forceinline__ void vh_unpack_hi(uint32_t hi0, uint32_t hi1, uint32_t hi2,
                                             int32_t& b0, int32_t& b1, int32_t& b2, int32_t& b3) {
    b0 = static_cast<int32_t>((hi0 & 0x0F0F0F0Fu) | ((hi2 & 0x03030303u) << 4));
    b1 = static_cast<int32_t>(((hi0 >> 4) & 0x0F0F0F0Fu) | ((hi2 & 0x0C0C0C0Cu) << 2));
    b2 = static_cast<int32_t>((hi1 & 0x0F0F0F0Fu) | (hi2 & 0x30303030u));
    b3 = static_cast<int32_t>(((hi1 >> 4) & 0x0F0F0F0Fu) | ((hi2 & 0xC0C0C0C0u) >> 2));
}

__device__ __forceinline__ int32_t vh_recon_i8x4(int32_t b, uint32_t r2) {
    // Biased u = ((uint8_t)w ^ 0x80) >> 2 has 0..63 per byte, so (b << 2) has zero cross-byte carry.
    return static_cast<int32_t>(((static_cast<uint32_t>(b) << 2) | r2) ^ 0x80808080u);
}

// Monotonic float <-> uint32 mapping matching src/kernels.cu::fkey and src/topk.cuh::tk_key
__device__ __forceinline__ uint32_t vh_float_to_ord(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float vh_ord_to_float(uint32_t k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k);
}

__device__ __forceinline__ unsigned long long vh_fkey(float v, int idx) {
    uint32_t ord = vh_float_to_ord(v);
    uint32_t inv_id = ~static_cast<uint32_t>(idx);
    return (static_cast<unsigned long long>(ord) << 32) | static_cast<unsigned long long>(inv_id);
}

// Matches src/topk.cuh:34 and src/kernels.cu:1463
__device__ __forceinline__ float vh_apply_pen_scalar(float l, int cnt, float rpt, float freq, float pres) {
    if (!isfinite(l)) return l;
    float lp = (l <= 0.0f) ? (l * rpt) : (l / rpt);
    return lp - static_cast<float>(cnt) * freq - pres;
}

__device__ __forceinline__ int vh_abs_i8(int32_t w, int shift) {
    int32_t v = static_cast<int8_t>((w >> shift) & 0xFF);
    return v < 0 ? -v : v;
}

// Unpacks q8_1 rows into 16B-aligned xq / fp32 xd + precomputes Stage-1 sx253, xd_half, and act_meta ({act_l2, act_l1}).
template <bool SPLIT>
__global__ void k_vh_stage0(const void* __restrict__ y_q8_1, const int8_t* __restrict__ tq,
                            const float* __restrict__ td, int n_in,
                            int8_t* __restrict__ out_xq, float* __restrict__ out_xd,
                            float* __restrict__ xd_half, int32_t* __restrict__ sx253,
                            float2* __restrict__ act_meta) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int nb = n_in >> 5;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nhb = n_in >> 4;
    const char* y_t = SPLIT ? nullptr : static_cast<const char*>(y_q8_1) + static_cast<size_t>(t) * nb * VH_Q8_1_BYTES;
    const int8_t* tq_t = SPLIT ? (tq + static_cast<size_t>(t) * n_in) : nullptr;
    const float* td_t = SPLIT ? (td + static_cast<size_t>(t) * nb) : nullptr;
    int8_t* xq_t = SPLIT ? nullptr : (out_xq + static_cast<size_t>(t) * n_in);
    float* xd_t = SPLIT ? nullptr : (out_xd + static_cast<size_t>(t) * nb);
    float* xdh_t = xd_half + static_cast<size_t>(t) * nb;
    int32_t* s253_t = sx253 + static_cast<size_t>(t) * nhb;

    float l2_sum = 0.0f;
    float blk_l1_sq = 0.0f;
    for (int b = tid; b < nb; b += blockDim.x) {
        int32_t x[8];
        float da;
        if constexpr (SPLIT) {
            const int4 v0 = *reinterpret_cast<const int4*>(tq_t + b * 32);
            const int4 v1 = *reinterpret_cast<const int4*>(tq_t + b * 32 + 16);
            x[0] = v0.x; x[1] = v0.y; x[2] = v0.z; x[3] = v0.w;
            x[4] = v1.x; x[5] = v1.y; x[6] = v1.z; x[7] = v1.w;
            da = td_t[b];
        } else {
            const char* bp = y_t + static_cast<size_t>(b) * VH_Q8_1_BYTES;
            da = vh_half_bits_to_f32(*reinterpret_cast<const uint16_t*>(bp));
            xd_t[b] = da;
            const int32_t* qp = reinterpret_cast<const int32_t*>(bp + 4);
            #pragma unroll
            for (int i = 0; i < 8; ++i) x[i] = qp[i];
            *reinterpret_cast<int4*>(xq_t + b * 32) = make_int4(x[0], x[1], x[2], x[3]);
            *reinterpret_cast<int4*>(xq_t + b * 32 + 16) = make_int4(x[4], x[5], x[6], x[7]);
        }
        xdh_t[b] = 0.5f * da;

        int sx0 = 0, sx1 = 0, ax_blk = 0, xx_blk = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            sx0 = __dp4a(x[i], 0x01010101, sx0);
            sx1 = __dp4a(x[4 + i], 0x01010101, sx1);
            xx_blk = __dp4a(x[i], x[i], xx_blk);
            xx_blk = __dp4a(x[4 + i], x[4 + i], xx_blk);
            #pragma unroll
            for (int sh = 0; sh < 32; sh += 8) {
                ax_blk += vh_abs_i8(x[i], sh) + vh_abs_i8(x[4 + i], sh);
            }
        }
        s253_t[2 * b] = 253 * sx0;
        s253_t[2 * b + 1] = 253 * sx1;
        l2_sum = fmaf(da * da, static_cast<float>(xx_blk), l2_sum);
        float bl1 = 1.5f * fabsf(da) * static_cast<float>(ax_blk);
        blk_l1_sq = fmaf(bl1, bl1, blk_l1_sq);
    }

    for (int o = 16; o >= 1; o >>= 1) {
        l2_sum += __shfl_xor_sync(0xffffffffu, l2_sum, o);
        blk_l1_sq += __shfl_xor_sync(0xffffffffu, blk_l1_sq, o);
    }
    __shared__ float sm2[8];
    __shared__ float sm1[8];
    if (lane == 0) {
        sm2[warp] = l2_sum;
        sm1[warp] = blk_l1_sq;
    }
    __syncthreads();
    if (warp == 0) {
        float v2 = (lane < 8) ? sm2[lane] : 0.0f;
        float v1 = (lane < 8) ? sm1[lane] : 0.0f;
        for (int o = 4; o >= 1; o >>= 1) {
            v2 += __shfl_xor_sync(0xffffffffu, v2, o);
            v1 += __shfl_xor_sync(0xffffffffu, v1, o);
        }
        if (lane == 0) act_meta[t] = make_float2(sqrtf(v2), sqrtf(v1));
    }
}

constexpr int VH_S1_WARPS = 8;
constexpr int VH_S1_THREADS = VH_S1_WARPS * 32;
constexpr int VH_S1_R = 4;
constexpr int VH_S1_RPC = VH_S1_WARPS * VH_S1_R;  // 32 rows/CTA

template <int TN, int R_ROWS, bool WRITE_MAX>
__device__ __forceinline__ void
k_vh_stage1_impl(const uint32_t* __restrict__ hi, const uint16_t* __restrict__ wd,
                 const float2* __restrict__ row_meta, int n_in, int n_out,
                 const int8_t* __restrict__ xq, const float* __restrict__ xd_half,
                 const int32_t* __restrict__ sx253, const float2* __restrict__ act_meta,
                 float* __restrict__ lb_out, float* __restrict__ ub_out,
                 float* __restrict__ cta_max, int n_cta_x, int T_total) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int r0 = (blockIdx.x * VH_S1_WARPS + warp) * R_ROWS;
    const int t_base = blockIdx.y * 8;
    const int S = n_in >> 9;
    const int nb = n_in >> 5;
    const int nhb = n_in >> 4;

    float mid[R_ROWS][TN];
    #pragma unroll
    for (int ri = 0; ri < R_ROWS; ++ri) {
        #pragma unroll
        for (int ti = 0; ti < TN; ++ti) mid[ri][ti] = 0.0f;
    }

    #pragma unroll 1
    for (int s = 0; s < S; ++s) {
        const int blk = (s << 4) + (lane >> 1);
        const int hb = (s << 5) + lane;
        const int col_off = (s << 9) + (lane << 4);
        constexpr int TC = (TN <= 4) ? TN : 4;
        #pragma unroll
        for (int t0 = 0; t0 < TN; t0 += TC) {
            int4 xv[TC];
            int32_t s253[TC];
            float dah[TC];
            #pragma unroll
            for (int k = 0; k < TC; ++k) {
                const int t = t_base + t0 + k;
                if (t0 + k < TN && t < T_total) {
                    xv[k] = __ldg(reinterpret_cast<const int4*>(xq + static_cast<size_t>(t) * n_in + col_off));
                    s253[k] = __ldg(sx253 + static_cast<size_t>(t) * nhb + hb);
                    dah[k] = __ldg(xd_half + static_cast<size_t>(t) * nb + blk);
                } else {
                    xv[k] = make_int4(0, 0, 0, 0);
                    s253[k] = 0;
                    dah[k] = 0.0f;
                }
            }
            #pragma unroll
            for (int ri = 0; ri < R_ROWS; ++ri) {
                const int r = r0 + ri;
                if (r >= n_out) break;
                const size_t hbase = (static_cast<size_t>(r) * S + s) * 96 + lane;
                uint32_t hi0 = __ldg(hi + hbase);
                uint32_t hi1 = __ldg(hi + hbase + 32);
                uint32_t hi2 = __ldg(hi + hbase + 64);
                int32_t b0, b1, b2, b3;
                vh_unpack_hi(hi0, hi1, hi2, b0, b1, b2, b3);
                float dw = vh_half_bits_to_f32(__ldg(wd + static_cast<size_t>(r) * nb + blk));
                #pragma unroll
                for (int k = 0; k < TC; ++k) {
                    if (t0 + k < TN && t_base + t0 + k < T_total) {
                        int du = __dp4a(b0, xv[k].x, 0);
                        du = __dp4a(b1, xv[k].y, du);
                        du = __dp4a(b2, xv[k].z, du);
                        du = __dp4a(b3, xv[k].w, du);
                        int q2 = (du << 3) - s253[k];
                        mid[ri][t0 + k] = fmaf(dw * dah[k], static_cast<float>(q2), mid[ri][t0 + k]);
                    }
                }
            }
        }
    }

    __shared__ float sm_max[VH_S1_WARPS][8];
    if (WRITE_MAX && lane == 0) {
        #pragma unroll
        for (int ti = 0; ti < 8; ++ti) sm_max[warp][ti] = -1.0e30f;
    }

    #pragma unroll
    for (int ri = 0; ri < R_ROWS; ++ri) {
        const int r = r0 + ri;
        if (r < n_out) {
            const float2 rm = row_meta ? __ldg(row_meta + r) : make_float2(1.0e30f, 1.0e30f);
            #pragma unroll
            for (int ti = 0; ti < TN; ++ti) {
                const int t = t_base + ti;
                if (t >= T_total) continue;
                float m = mid[ri][ti];
                #pragma unroll
                for (int o = 16; o >= 1; o >>= 1) m += __shfl_xor_sync(0xffffffffu, m, o);
                if (lane == 0) {
                    const float2 am = act_meta[t];
                    const float e1_ub = rm.y * am.y;
                    const float e2_ub = rm.x * am.x;
                    const float hw = fminf(e1_ub, e2_ub) + fmaf(1.0e-3f, e1_ub, 1.0e-5f);
                    const float lb = m - hw;
                    const float ub = m + hw;
                    const size_t idx = static_cast<size_t>(t) * n_out + r;
                    if constexpr (WRITE_MAX) {
                        const bool ok = isfinite(lb) && isfinite(ub);
                        ub_out[idx] = ok ? ub : NAN;
                        const float lf = ok ? lb : -1.0e30f;
                        if (lf > sm_max[warp][ti]) sm_max[warp][ti] = lf;
                    } else {
                        lb_out[idx] = lb;
                        ub_out[idx] = ub;
                    }
                }
            }
        }
    }

    if (WRITE_MAX) {
        __syncthreads();
        if (tid < TN && t_base + tid < T_total) {
            float mv = -1.0e30f;
            #pragma unroll
            for (int w = 0; w < VH_S1_WARPS; ++w) {
                if (sm_max[w][tid] > mv) mv = sm_max[w][tid];
            }
            cta_max[static_cast<size_t>(t_base + tid) * n_cta_x + blockIdx.x] = mv;
        }
    }
}

template <int TN, bool WRITE_MAX>
__global__ void __launch_bounds__(VH_S1_THREADS, 4)
k_vh_stage1_sm(const uint32_t* __restrict__ hi, const uint16_t* __restrict__ wd,
               const float2* __restrict__ row_meta, int n_in, int n_out,
               const int8_t* __restrict__ xq, const float* __restrict__ xd_half,
               const int32_t* __restrict__ sx253, const float2* __restrict__ act_meta,
               float* __restrict__ lb_out, float* __restrict__ ub_out,
               float* __restrict__ cta_max, int n_cta_x, int T_total) {
    k_vh_stage1_impl<TN, VH_S1_R, WRITE_MAX>(hi, wd, row_meta, n_in, n_out, xq, xd_half, sx253, act_meta, lb_out, ub_out, cta_max, n_cta_x, T_total);
}

template <int TN, bool WRITE_MAX>
__global__ void __launch_bounds__(VH_S1_THREADS, 2)
k_vh_stage1_lg(const uint32_t* __restrict__ hi, const uint16_t* __restrict__ wd,
               const float2* __restrict__ row_meta, int n_in, int n_out,
               const int8_t* __restrict__ xq, const float* __restrict__ xd_half,
               const int32_t* __restrict__ sx253, const float2* __restrict__ act_meta,
               float* __restrict__ lb_out, float* __restrict__ ub_out,
               float* __restrict__ cta_max, int n_cta_x, int T_total) {
    k_vh_stage1_impl<TN, VH_S1_R, WRITE_MAX>(hi, wd, row_meta, n_in, n_out, xq, xd_half, sx253, act_meta, lb_out, ub_out, cta_max, n_cta_x, T_total);
}

__global__ void k_vh_max_reduce(const float* __restrict__ cta_max, int n_cta_x,
                                float* __restrict__ lcut, uint32_t* __restrict__ cand_cnt) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row = cta_max + static_cast<size_t>(t) * n_cta_x;
    float best = -1.0e30f;
    for (int i = tid; i < n_cta_x; i += blockDim.x) {
        float v = row[i];
        if (v > best) best = v;
    }
    for (int o = 16; o >= 1; o >>= 1) {
        float other = __shfl_xor_sync(0xffffffffu, best, o);
        if (other > best) best = other;
    }
    __shared__ float sm[8];
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) sm[warp] = best;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 8) ? sm[lane] : -1.0e30f;
        for (int o = 4; o >= 1; o >>= 1) {
            float other = __shfl_xor_sync(0xffffffffu, v, o);
            if (other > v) v = other;
        }
        if (lane == 0) {
            lcut[t] = v;
            cand_cnt[t] = 0u;
        }
    }
}

// pen_tok holds GLOBAL token IDs; subtract v0 to get card-local offset u in [0, n_out), matching topk.cuh:31
__global__ void k_vh_apply_pen_bounds(float* __restrict__ lb, float* __restrict__ ub, int n_out, int v0,
                                      const int32_t* __restrict__ vocab_to_local,
                                      const int* __restrict__ pen_tok, const int* __restrict__ pen_cnt,
                                      const int* __restrict__ pen_n, int pen_stride,
                                      const float* __restrict__ pen_par) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int pn = pen_n ? pen_n[t] : 0;
    if (pn <= 0) return;

    const float rpt = pen_par[t * 3 + 0];
    const float freq = pen_par[t * 3 + 1];
    const float pres = pen_par[t * 3 + 2];
    const int* pt = pen_tok + static_cast<size_t>(t) * pen_stride;
    const int* pc = pen_cnt + static_cast<size_t>(t) * pen_stride;
    float* lb_t = lb + static_cast<size_t>(t) * n_out;
    float* ub_t = ub + static_cast<size_t>(t) * n_out;

    for (int i = tid; i < pn; i += blockDim.x) {
        const int u = pt[i] - v0;
        const int cnt = pc[i];
        if (u < 0 || u >= n_out || cnt <= 0) continue;
        const int r = vocab_to_local ? vocab_to_local[u] : u;
        if (rpt > 0.0f) {
            lb_t[r] = vh_apply_pen_scalar(lb_t[r], cnt, rpt, freq, pres);
            ub_t[r] = vh_apply_pen_scalar(ub_t[r], cnt, rpt, freq, pres);
        } else {
            lb_t[r] = -1.0e30f;
            ub_t[r] = 1.0e30f;
        }
    }
}

__global__ void k_vh_select_lcut64(const float* __restrict__ lb, const float* __restrict__ ub,
                                   int n_out, float* __restrict__ lcut, uint32_t* __restrict__ cand_cnt) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const float* lb_t = lb + static_cast<size_t>(t) * n_out;
    const float* ub_t = ub + static_cast<size_t>(t) * n_out;
    const int K = TOPK_N < n_out ? TOPK_N : n_out;

    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_prefix;
    __shared__ int s_rem;

    if (tid == 0) {
        s_prefix = 0u;
        s_rem = K;
    }
    __syncthreads();

    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - pass * 8;
        hist[tid] = 0u;
        __syncthreads();

        const uint32_t pref = s_prefix;
        const uint32_t pref_mask = (pass == 0) ? 0u : (~0u << (shift + 8));
        for (int r = tid; r < n_out; r += blockDim.x) {
            float lf = lb_t[r];
            float uf = ub_t[r];
            float val = (isfinite(lf) && isfinite(uf)) ? lf : -1.0e30f;
            uint32_t ord = vh_float_to_ord(val);
            if ((ord & pref_mask) == (pref & pref_mask)) {
                atomicAdd(&hist[(ord >> shift) & 0xFFu], 1u);
            }
        }
        __syncthreads();

        if (tid == 0) {
            int rem = s_rem;
            uint32_t sum = 0;
            for (int b = 255; b >= 0; --b) {
                uint32_t c = hist[b];
                if (static_cast<int>(sum + c) >= rem) {
                    s_rem = rem - static_cast<int>(sum);
                    s_prefix = pref | (static_cast<uint32_t>(b) << shift);
                    break;
                }
                sum += c;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        lcut[t] = (n_out <= TOPK_N) ? -1.0e30f : vh_ord_to_float(s_prefix);
        cand_cnt[t] = 0u;
    }
}

template <bool CHECK_LB>
__global__ void k_vh_filter(const float* __restrict__ lb, const float* __restrict__ ub,
                            const float* __restrict__ lcut, int n_out,
                            int32_t* __restrict__ cand_idx, uint32_t* __restrict__ cand_cnt) {
    const int t = blockIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const float cut = lcut[t];
    bool pass = false;
    if (r < n_out) {
        const size_t idx = static_cast<size_t>(t) * n_out + r;
        float uf = ub[idx];
        if constexpr (CHECK_LB) {
            float lf = lb[idx];
            pass = (!isfinite(lf) || !isfinite(uf) || uf >= cut);
        } else {
            pass = (!isfinite(uf) || uf >= cut);
        }
    }

    const unsigned mask = __ballot_sync(0xffffffffu, pass);
    if (!mask) return;
    const int lane = threadIdx.x & 31;
    const int leader = __ffs(mask) - 1;
    uint32_t base = 0;
    if (lane == leader) {
        base = atomicAdd(&cand_cnt[t], static_cast<uint32_t>(__popc(mask)));
    }
    base = __shfl_sync(0xffffffffu, base, leader);
    if (pass) {
        uint32_t rank = static_cast<uint32_t>(__popc(mask & ((1u << lane) - 1u)));
        cand_idx[static_cast<size_t>(t) * n_out + base + rank] = r;
    }
}

__device__ __forceinline__ float vh_stage2_row(const uint32_t* __restrict__ hi, const uint32_t* __restrict__ lo,
                                               const uint16_t* __restrict__ wd, int n_in, int r,
                                               const int8_t* __restrict__ xq_t, const float* __restrict__ xd_t) {
    const int lane = threadIdx.x & 31;
    const int S = n_in >> 9;
    const int nb = n_in >> 5;
    float acc = 0.0f;

    for (int s = 0; s < S; ++s) {
        const size_t hbase = (static_cast<size_t>(r) * S + s) * 96 + lane;
        const uint32_t hi0 = hi[hbase];
        const uint32_t hi1 = hi[hbase + 32];
        const uint32_t hi2 = hi[hbase + 64];
        const uint32_t lo0 = lo[(static_cast<size_t>(r) * S + s) * 32 + lane];
        int32_t b0, b1, b2, b3;
        vh_unpack_hi(hi0, hi1, hi2, b0, b1, b2, b3);
        int32_t w0 = vh_recon_i8x4(b0, lo0 & 0x03030303u);
        int32_t w1 = vh_recon_i8x4(b1, (lo0 >> 2) & 0x03030303u);
        int32_t w2 = vh_recon_i8x4(b2, (lo0 >> 4) & 0x03030303u);
        int32_t w3 = vh_recon_i8x4(b3, (lo0 >> 6) & 0x03030303u);

        const int32_t* xp = reinterpret_cast<const int32_t*>(xq_t + (s << 9) + (lane << 4));
        int4 xv = __ldg(reinterpret_cast<const int4*>(xp));
        const int blk = (s << 4) + (lane >> 1);
        float dw = vh_half_bits_to_f32(__ldg(wd + static_cast<size_t>(r) * nb + blk));
        float da = __ldg(xd_t + blk);

        int isum = __dp4a(w0, xv.x, 0);
        isum = __dp4a(w1, xv.y, isum);
        isum = __dp4a(w2, xv.z, isum);
        isum = __dp4a(w3, xv.w, isum);
        acc = fmaf(dw * da, static_cast<float>(isum), acc);
    }

    for (int o = 16; o >= 1; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}

__global__ void k_vh_stage2_verify(const uint32_t* __restrict__ hi, const uint32_t* __restrict__ lo,
                                   const uint16_t* __restrict__ wd, const int32_t* __restrict__ perm_to_vocab,
                                   int n_in, int n_out, int v0,
                                   const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                   const int32_t* __restrict__ cand_idx, const uint32_t* __restrict__ cand_cnt,
                                   unsigned long long* __restrict__ keys, unsigned* __restrict__ err) {
    const int t = blockIdx.y;
    const int cnt = static_cast<int>(cand_cnt[t]);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nb = n_in >> 5;
    const int8_t* xq_t = xq + static_cast<size_t>(t) * n_in;
    const float* xd_t = xd + static_cast<size_t>(t) * nb;
    const int32_t* cands = cand_idx + static_cast<size_t>(t) * n_out;

    unsigned long long best_k = 0ull;
    const int stride = gridDim.x * VH_WARPS;
    for (int k = blockIdx.x * VH_WARPS + warp; k < cnt; k += stride) {
        int r = cands[k];
        float acc = vh_stage2_row(hi, lo, wd, n_in, r, xq_t, xd_t);
        if (lane == 0) {
            int vr = perm_to_vocab ? perm_to_vocab[r] : r;
            if (!isfinite(acc) && err) atomicOr(err, 16u);
            // Match k_argmax: key uses raw vr index, fkey_inv adds v0 in argmax_fin
            unsigned long long cand_k = vh_fkey(acc, vr);
            if (cand_k > best_k) best_k = cand_k;
        }
    }

    if (lane == 0 && best_k > 0ull) {
        atomicMax(&keys[t], best_k);
    }
}

__global__ void k_vh_argmax_fin(int T, int v0, unsigned long long* __restrict__ keys,
                                float* __restrict__ best_v, int* __restrict__ best_i) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    unsigned long long k = keys[t];
    best_v[t] = vh_ord_to_float(static_cast<uint32_t>(k >> 32));
    best_i[t] = static_cast<int>(~static_cast<uint32_t>(k & 0xFFFFFFFFu)) + v0;
    keys[t] = 0ULL;
}

__global__ void k_vh_stage2_top(const uint32_t* __restrict__ hi, const uint32_t* __restrict__ lo,
                                const uint16_t* __restrict__ wd, int n_in, int n_out,
                                const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                const int32_t* __restrict__ cand_idx, const uint32_t* __restrict__ cand_cnt,
                                float* __restrict__ ub_logits, unsigned* __restrict__ err) {
    const int t = blockIdx.y;
    const int cnt = static_cast<int>(cand_cnt[t]);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nb = n_in >> 5;
    const int8_t* xq_t = xq + static_cast<size_t>(t) * n_in;
    const float* xd_t = xd + static_cast<size_t>(t) * nb;
    const int32_t* cands = cand_idx + static_cast<size_t>(t) * n_out;
    float* dst_t = ub_logits + static_cast<size_t>(t) * n_out;

    const int stride = gridDim.x * VH_WARPS;
    for (int k = blockIdx.x * VH_WARPS + warp; k < cnt; k += stride) {
        int r = cands[k];
        float acc = vh_stage2_row(hi, lo, wd, n_in, r, xq_t, xd_t);
        if (lane == 0) {
            if (!isfinite(acc) && err) atomicOr(err, 16u);
            dst_t[r] = acc;
        }
    }
}

__global__ void k_vh_repen_logits(float* __restrict__ ub_logits, int n_out, int v0,
                                  const int32_t* __restrict__ vocab_to_local,
                                  const int* __restrict__ pen_tok, const int* __restrict__ pen_cnt,
                                  const int* __restrict__ pen_n, int pen_stride,
                                  const float* __restrict__ pen_par) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int pn = pen_n ? pen_n[t] : 0;
    if (pn <= 0) return;

    const float rpt = pen_par[t * 3 + 0];
    const float freq = pen_par[t * 3 + 1];
    const float pres = pen_par[t * 3 + 2];
    const int* pt = pen_tok + static_cast<size_t>(t) * pen_stride;
    const int* pc = pen_cnt + static_cast<size_t>(t) * pen_stride;
    float* dst_t = ub_logits + static_cast<size_t>(t) * n_out;

    for (int i = tid; i < pn; i += blockDim.x) {
        const int u = pt[i] - v0;
        const int cnt = pc[i];
        if (u < 0 || u >= n_out || cnt <= 0) continue;
        const int r = vocab_to_local ? vocab_to_local[u] : u;
        dst_t[r] = vh_apply_pen_scalar(dst_t[r], cnt, rpt, freq, pres);
    }
}

// penalize_row format: pen = {count k, k card-local indices, k counts}; par = {repeat, freq, present}
__global__ void k_vh_penalize_row_6p2(float* __restrict__ row_logits, int n_out,
                                      const int32_t* __restrict__ vocab_to_local,
                                      const int* __restrict__ pen, const float* __restrict__ par) {
    const int k = pen[0];
    if (k <= 0) return;
    const float rpt = par[0], freq = par[1], pres = par[2];
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        const int u = pen[1 + i];
        const int cnt = pen[1 + k + i];
        if (u < 0 || u >= n_out || cnt <= 0) continue;
        const int r = vocab_to_local ? vocab_to_local[u] : u;
        row_logits[r] = vh_apply_pen_scalar(row_logits[r], cnt, rpt, freq, pres);
    }
}

__global__ void k_vh_cand_argmax(const float* __restrict__ ub_logits, const int32_t* __restrict__ perm_to_vocab,
                                 int n_out, const int32_t* __restrict__ cand_idx,
                                 const uint32_t* __restrict__ cand_cnt, unsigned long long* __restrict__ keys) {
    const int t = blockIdx.y;
    const int cnt = static_cast<int>(cand_cnt[t]);
    const int32_t* cands = cand_idx + static_cast<size_t>(t) * n_out;
    const float* row = ub_logits + static_cast<size_t>(t) * n_out;
    unsigned long long best = 0ull;
    const int stride = gridDim.x * blockDim.x;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < cnt; k += stride) {
        int r = cands[k];
        int vr = perm_to_vocab ? perm_to_vocab[r] : r;
        unsigned long long ck_val = vh_fkey(row[r], vr);
        if (ck_val > best) best = ck_val;
    }
    for (int o = 16; o > 0; o >>= 1) {
        unsigned long long ok = __shfl_xor_sync(0xffffffffu, best, o);
        if (ok > best) best = ok;
    }
    if ((threadIdx.x & 31) == 0 && best > 0ull) atomicMax(&keys[t], best);
}

__global__ void k_vh_topk_cand(const float* __restrict__ ub_logits, const int32_t* __restrict__ perm_to_vocab,
                               int n_out, int v0, int row_off,
                               const int32_t* __restrict__ cand_idx, const uint32_t* __restrict__ cand_cnt,
                               float* __restrict__ out_v, int* __restrict__ out_i) {
    const int t = row_off >= 0 ? row_off : blockIdx.x;
    const int out_row = row_off >= 0 ? 0 : blockIdx.x;
    const int tid = threadIdx.x;
    const int M = static_cast<int>(cand_cnt[t]);
    const int32_t* cands = cand_idx + static_cast<size_t>(t) * n_out;
    const float* vals_t = ub_logits + static_cast<size_t>(t) * n_out;
    const int K = TOPK_N;
    const int K_eff = M < K ? M : K;

    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_prefix;
    __shared__ int s_rem;
    __shared__ uint32_t s_id_cut;
    __shared__ int s_out_cnt;
    __shared__ float sh_v[TOPK_N];
    __shared__ int sh_i[TOPK_N];

    if (M <= 0) {
        if (tid < K) {
            out_v[out_row * K + tid] = -1.0e30f;
            out_i[out_row * K + tid] = v0;
        }
        return;
    }

    if (tid == 0) {
        s_prefix = 0u;
        s_rem = K_eff;
    }
    __syncthreads();

    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - pass * 8;
        hist[tid] = 0u;
        __syncthreads();
        const uint32_t pref = s_prefix;
        const uint32_t pref_mask = (pass == 0) ? 0u : (~0u << (shift + 8));
        for (int k = tid; k < M; k += blockDim.x) {
            uint32_t ord = vh_float_to_ord(vals_t[cands[k]]);
            if ((ord & pref_mask) == (pref & pref_mask)) {
                atomicAdd(&hist[(ord >> shift) & 0xFFu], 1u);
            }
        }
        __syncthreads();
        if (tid == 0) {
            int rem = s_rem;
            uint32_t sum = 0;
            for (int b = 255; b >= 0; --b) {
                uint32_t c = hist[b];
                if (static_cast<int>(sum + c) >= rem || b == 0) {
                    s_rem = rem - static_cast<int>(sum);
                    s_prefix = pref | (static_cast<uint32_t>(b) << shift);
                    break;
                }
                sum += c;
            }
        }
        __syncthreads();
    }

    const uint32_t thresh_ord = s_prefix;
    if (tid == 0) {
        s_out_cnt = 0;
        s_id_cut = 0xFFFFFFFFu;
    }
    __syncthreads();

    for (int k = tid; k < M; k += blockDim.x) {
        int r = cands[k];
        float f = vals_t[r];
        uint32_t ord = vh_float_to_ord(f);
        if (ord > thresh_ord) {
            int slot = atomicAdd(&s_out_cnt, 1);
            if (slot < K_eff) {
                int gid = v0 + (perm_to_vocab ? perm_to_vocab[r] : r);
                sh_v[slot] = f;
                sh_i[slot] = gid;
            }
        }
    }
    __syncthreads();

    const int strict_cnt = s_out_cnt < K_eff ? s_out_cnt : K_eff;
    const int need_ties = K_eff - strict_cnt;

    if (need_ties > 0) {
        if (tid == 0) {
            s_prefix = 0u;
            s_rem = need_ties;
        }
        __syncthreads();

        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - pass * 8;
            hist[tid] = 0u;
            __syncthreads();
            const uint32_t pref = s_prefix;
            const uint32_t pref_mask = (pass == 0) ? 0u : (~0u << (shift + 8));
            for (int k = tid; k < M; k += blockDim.x) {
                int r = cands[k];
                uint32_t ord = vh_float_to_ord(vals_t[r]);
                if (ord == thresh_ord) {
                    uint32_t gid = static_cast<uint32_t>(v0 + (perm_to_vocab ? perm_to_vocab[r] : r));
                    if ((gid & pref_mask) == (pref & pref_mask)) {
                        atomicAdd(&hist[(gid >> shift) & 0xFFu], 1u);
                    }
                }
            }
            __syncthreads();
            if (tid == 0) {
                int rem = s_rem;
                uint32_t sum = 0;
                for (int b = 0; b < 256; ++b) {
                    uint32_t c = hist[b];
                    if (static_cast<int>(sum + c) >= rem || b == 255) {
                        s_rem = rem - static_cast<int>(sum);
                        s_prefix = pref | (static_cast<uint32_t>(b) << shift);
                        break;
                    }
                    sum += c;
                }
            }
            __syncthreads();
        }
        if (tid == 0) s_id_cut = s_prefix;
        __syncthreads();

        const uint32_t id_cut = s_id_cut;
        for (int k = tid; k < M; k += blockDim.x) {
            int r = cands[k];
            float f = vals_t[r];
            if (vh_float_to_ord(f) == thresh_ord) {
                int gid = v0 + (perm_to_vocab ? perm_to_vocab[r] : r);
                if (static_cast<uint32_t>(gid) <= id_cut) {
                    int slot = atomicAdd(&s_out_cnt, 1);
                    if (slot < K_eff) {
                        sh_v[slot] = f;
                        sh_i[slot] = gid;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (tid < K && tid >= K_eff) {
        sh_v[tid] = -1.0e30f;
        sh_i[tid] = v0;
    }
    __syncthreads();

    for (int size = 2; size <= K; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (tid < K) {
                int ixj = tid ^ stride;
                if (ixj > tid) {
                    bool asc_dir = ((tid & size) == 0);
                    float va = sh_v[tid];
                    float vb = sh_v[ixj];
                    int ia = sh_i[tid];
                    int ib = sh_i[ixj];
                    uint32_t oka = vh_float_to_ord(va);
                    uint32_t okb = vh_float_to_ord(vb);
                    bool a_better = (oka > okb) || (oka == okb && ia < ib);
                    if (asc_dir ? (!a_better) : a_better) {
                        sh_v[tid] = vb;
                        sh_i[tid] = ib;
                        sh_v[ixj] = va;
                        sh_i[ixj] = ia;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (tid < K) {
        out_v[out_row * K + tid] = sh_v[tid];
        out_i[out_row * K + tid] = sh_i[tid];
    }
}

template <int TN, int R, bool SPLIT>
__global__ void __launch_bounds__(VH_S1_THREADS)
k_vh_all_logits(const uint32_t* __restrict__ hi, const uint32_t* __restrict__ lo,
                const uint16_t* __restrict__ wd, const int32_t* __restrict__ perm_to_vocab,
                int n_in, int n_out, const void* __restrict__ y_q8_1,
                const int8_t* __restrict__ tq, const float* __restrict__ td,
                float* __restrict__ logits, int T_total) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int r0 = (blockIdx.x * VH_S1_WARPS + warp) * R;
    const int t0_base = blockIdx.y * 8;
    const int S = n_in >> 9;
    const int nb = n_in >> 5;
    const int half = lane & 1;

    float acc[R][TN];
    #pragma unroll
    for (int ri = 0; ri < R; ++ri) {
        #pragma unroll
        for (int ti = 0; ti < TN; ++ti) acc[ri][ti] = 0.0f;
    }

    for (int s = 0; s < S; ++s) {
        int32_t w0[R], w1[R], w2[R], w3[R];
        float dw[R];
        #pragma unroll
        for (int ri = 0; ri < R; ++ri) {
            const int r = r0 + ri;
            if (r < n_out) {
                const size_t hbase = (static_cast<size_t>(r) * S + s) * 96 + lane;
                uint32_t hi0 = __ldg(hi + hbase);
                uint32_t hi1 = __ldg(hi + hbase + 32);
                uint32_t hi2 = __ldg(hi + hbase + 64);
                uint32_t lo0 = __ldg(lo + (static_cast<size_t>(r) * S + s) * 32 + lane);
                int32_t b0, b1, b2, b3;
                vh_unpack_hi(hi0, hi1, hi2, b0, b1, b2, b3);
                w0[ri] = vh_recon_i8x4(b0, lo0 & 0x03030303u);
                w1[ri] = vh_recon_i8x4(b1, (lo0 >> 2) & 0x03030303u);
                w2[ri] = vh_recon_i8x4(b2, (lo0 >> 4) & 0x03030303u);
                w3[ri] = vh_recon_i8x4(b3, (lo0 >> 6) & 0x03030303u);
                dw[ri] = vh_half_bits_to_f32(__ldg(wd + static_cast<size_t>(r) * nb + (s << 4) + (lane >> 1)));
            } else {
                w0[ri] = w1[ri] = w2[ri] = w3[ri] = 0;
                dw[ri] = 0.0f;
            }
        }

        const int blk = (s << 4) + (lane >> 1);
        const int col_off = (s << 9) + (lane << 4);
        constexpr int TC = (TN <= 4) ? TN : 4;
        #pragma unroll
        for (int t0 = 0; t0 < TN; t0 += TC) {
            int32_t x0[TC], x1[TC], x2[TC], x3[TC];
            float da[TC];
            #pragma unroll
            for (int k = 0; k < TC; ++k) {
                const int t = t0_base + t0 + k;
                if (t0 + k < TN && t < T_total) {
                    if constexpr (SPLIT) {
                        int4 xv = __ldg(reinterpret_cast<const int4*>(tq + static_cast<size_t>(t) * n_in + col_off));
                        x0[k] = xv.x; x1[k] = xv.y; x2[k] = xv.z; x3[k] = xv.w;
                        da[k] = __ldg(td + static_cast<size_t>(t) * nb + blk);
                    } else {
                        const char* bp = static_cast<const char*>(y_q8_1)
                                       + (static_cast<size_t>(t) * nb + blk) * VH_Q8_1_BYTES;
                        da[k] = vh_half_bits_to_f32(__ldg(reinterpret_cast<const uint16_t*>(bp)));
                        const int32_t* qp = reinterpret_cast<const int32_t*>(bp + 4 + half * 16);
                        x0[k] = __ldg(qp); x1[k] = __ldg(qp + 1); x2[k] = __ldg(qp + 2); x3[k] = __ldg(qp + 3);
                    }
                } else {
                    x0[k] = x1[k] = x2[k] = x3[k] = 0;
                    da[k] = 0.0f;
                }
            }
            #pragma unroll
            for (int ri = 0; ri < R; ++ri) {
                if (r0 + ri < n_out) {
                    #pragma unroll
                    for (int k = 0; k < TC; ++k) {
                        if (t0 + k < TN && t0_base + t0 + k < T_total) {
                            int isum = __dp4a(w0[ri], x0[k], 0);
                            isum = __dp4a(w1[ri], x1[k], isum);
                            isum = __dp4a(w2[ri], x2[k], isum);
                            isum = __dp4a(w3[ri], x3[k], isum);
                            acc[ri][t0 + k] = fmaf(dw[ri] * da[k], static_cast<float>(isum), acc[ri][t0 + k]);
                        }
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int ri = 0; ri < R; ++ri) {
        const int r = r0 + ri;
        if (r < n_out) {
            int vr = perm_to_vocab ? perm_to_vocab[r] : r;
            #pragma unroll
            for (int ti = 0; ti < TN; ++ti) {
                const int t = t0_base + ti;
                if (t >= T_total) continue;
                float a = acc[ri][ti];
                for (int o = 16; o >= 1; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
                if (lane == 0) logits[static_cast<size_t>(t) * n_out + vr] = a;
            }
        }
    }
}

__global__ void k_vh_score_centroids(const uint16_t* __restrict__ cent_f16, int n_clusters, int n_in,
                                     const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                     float* __restrict__ scores, int T_total) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int c = blockIdx.x * VH_WARPS + warp;
    if (c >= n_clusters) return;

    const int t = blockIdx.y;
    if (t >= T_total) return;

    const int S = n_in >> 9;
    const int nb = n_in >> 5;
    const uint16_t* c_ptr = cent_f16 + static_cast<size_t>(c) * n_in;
    const int8_t* xq_t = xq + static_cast<size_t>(t) * n_in;
    const float* xd_t = xd + static_cast<size_t>(t) * nb;

    float acc = 0.0f;
    for (int s = 0; s < S; ++s) {
        const int off = (s << 9) + (lane << 4);
        const int blk = (s << 4) + (lane >> 1);
        float da = __ldg(xd_t + blk);
        const int32_t* xp = reinterpret_cast<const int32_t*>(xq_t + off);
        int4 xv = __ldg(reinterpret_cast<const int4*>(xp));
        const uint4 cv0 = __ldg(reinterpret_cast<const uint4*>(c_ptr + off));
        const uint4 cv1 = __ldg(reinterpret_cast<const uint4*>(c_ptr + off + 8));
        const uint32_t cv[8] = {cv0.x, cv0.y, cv0.z, cv0.w, cv1.x, cv1.y, cv1.z, cv1.w};
        const int32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};

        float bsum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint32_t c_lo = cv[2 * k];
            uint32_t c_hi = cv[2 * k + 1];
            int32_t xi = xw[k];
            float f0 = vh_half_bits_to_f32(static_cast<uint16_t>(c_lo & 0xFFFFu));
            float f1 = vh_half_bits_to_f32(static_cast<uint16_t>(c_lo >> 16));
            float f2 = vh_half_bits_to_f32(static_cast<uint16_t>(c_hi & 0xFFFFu));
            float f3 = vh_half_bits_to_f32(static_cast<uint16_t>(c_hi >> 16));
            bsum = fmaf(f0, static_cast<float>(static_cast<int8_t>(xi & 0xFF)), bsum);
            bsum = fmaf(f1, static_cast<float>(static_cast<int8_t>((xi >> 8) & 0xFF)), bsum);
            bsum = fmaf(f2, static_cast<float>(static_cast<int8_t>((xi >> 16) & 0xFF)), bsum);
            bsum = fmaf(f3, static_cast<float>(static_cast<int8_t>((xi >> 24) & 0xFF)), bsum);
        }
        acc = fmaf(da, bsum, acc);
    }

    for (int o = 16; o >= 1; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) scores[static_cast<size_t>(t) * n_clusters + c] = acc;
}

__global__ void k_vh_expand_clusters(const float* __restrict__ scores, const int32_t* __restrict__ cluster_off,
                                     int n_clusters, int top_clusters, int n_out,
                                     int32_t* __restrict__ cand_idx, uint32_t* __restrict__ cand_cnt) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const float* sc = scores + static_cast<size_t>(t) * n_clusters;
    const int K = (top_clusters < n_clusters ? top_clusters : n_clusters);
    int32_t* dst = cand_idx + static_cast<size_t>(t) * n_out;

    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_prefix;
    __shared__ int s_rem;
    __shared__ int s_tie_budget;
    __shared__ uint32_t s_write_pos;

    if (tid == 0) {
        s_prefix = 0u;
        s_rem = K;
        s_tie_budget = 0;
        s_write_pos = 0u;
    }
    __syncthreads();

    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - pass * 8;
        hist[tid] = 0u;
        __syncthreads();
        const uint32_t pref = s_prefix;
        const uint32_t pref_mask = (pass == 0) ? 0u : (~0u << (shift + 8));
        for (int c = tid; c < n_clusters; c += blockDim.x) {
            float f = sc[c];
            uint32_t ord = vh_float_to_ord(isfinite(f) ? f : -1.0e30f);
            if ((ord & pref_mask) == (pref & pref_mask)) atomicAdd(&hist[(ord >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            int rem = s_rem;
            uint32_t sum = 0;
            for (int b = 255; b >= 0; --b) {
                uint32_t cnt = hist[b];
                if (static_cast<int>(sum + cnt) >= rem || b == 0) {
                    s_rem = rem - static_cast<int>(sum);
                    s_prefix = pref | (static_cast<uint32_t>(b) << shift);
                    break;
                }
                sum += cnt;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        s_tie_budget = s_rem;
        s_write_pos = 0u;
    }
    __syncthreads();

    __shared__ int sh_eq_w[8], sh_tk_w[8], sh_nsel, sh_chunk_tie;
    __shared__ int sh_rb[256], sh_re[256];
    const uint32_t thresh_ord = s_prefix;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    for (int c_base = 0; c_base < n_clusters; c_base += 256) {
        const int c = c_base + tid;
        bool gt = false, eq = false;
        if (c < n_clusters) {
            float f = sc[c];
            uint32_t ord = vh_float_to_ord(isfinite(f) ? f : -1.0e30f);
            gt = (ord > thresh_ord);
            eq = (ord == thresh_ord);
        }
        uint32_t eq_mask = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) sh_eq_w[warp] = __popc(eq_mask);
        __syncthreads();

        if (tid == 0) {
            int acc = 0;
            for (int w = 0; w < 8; ++w) {
                int cnt = sh_eq_w[w];
                sh_eq_w[w] = acc;
                acc += cnt;
            }
            int bud = s_tie_budget;
            sh_chunk_tie = bud;
            s_tie_budget = bud > acc ? (bud - acc) : 0;
        }
        __syncthreads();

        const int eq_rank = sh_eq_w[warp] + __popc(eq_mask & ((1u << lane) - 1u));
        const bool take = gt || (eq && eq_rank < sh_chunk_tie);
        uint32_t tk_mask = __ballot_sync(0xffffffffu, take);
        if (lane == 0) sh_tk_w[warp] = __popc(tk_mask);
        __syncthreads();

        if (tid == 0) {
            int acc = 0;
            for (int w = 0; w < 8; ++w) {
                int cnt = sh_tk_w[w];
                sh_tk_w[w] = acc;
                acc += cnt;
            }
            sh_nsel = acc;
        }
        __syncthreads();

        if (take) {
            int slot = sh_tk_w[warp] + __popc(tk_mask & ((1u << lane) - 1u));
            sh_rb[slot] = cluster_off[c];
            sh_re[slot] = cluster_off[c + 1];
        }
        __syncthreads();

        const int nsel = sh_nsel;
        uint32_t base = s_write_pos;
        for (int i = 0; i < nsel; ++i) {
            const int rb = sh_rb[i], len = sh_re[i] - rb;
            for (int j = tid; j < len; j += blockDim.x) {
                const uint32_t pos = base + static_cast<uint32_t>(j);
                if (pos < static_cast<uint32_t>(n_out)) dst[pos] = rb + j;
            }
            base += static_cast<uint32_t>(len > 0 ? len : 0);
        }
        if (tid == 0) s_write_pos = base;
        __syncthreads();
    }

    if (tid == 0) {
        cand_cnt[t] = (s_write_pos < static_cast<uint32_t>(n_out)) ? s_write_pos : static_cast<uint32_t>(n_out);
    }
}

inline void launch_stage0_and_get_xy(const void* y, int n_in, int T, const VHeadWork& work,
                                     cudaStream_t stream, const int8_t*& xq_out, const float*& xd_out) {
    int8_t* tq = nullptr;
    float* td = nullptr;
    vh_twin_of(y, n_in, tq, td);
    if (tq) {
        k_vh_stage0<true><<<T, 256, 0, stream>>>(nullptr, tq, td, n_in, nullptr, nullptr, work.xd_half, work.sx253, work.act_meta);
        xq_out = tq;
        xd_out = td;
    } else {
        k_vh_stage0<false><<<T, 256, 0, stream>>>(y, nullptr, nullptr, n_in, work.xq, work.xd, work.xd_half, work.sx253, work.act_meta);
        xq_out = work.xq;
        xd_out = work.xd;
    }
}

inline void launch_stage1(const VHead6p2& w, const int8_t* xq, int T, bool write_max,
                          const VHeadWork& work, cudaStream_t stream) {
    const int n_cta_x = vhead_max_cta(w.n_out, T);
    const float2* row_meta = static_cast<const float2*>(w.row_meta);
    auto go_sm = [&](auto tn_c) {
        constexpr int TN = decltype(tn_c)::value;
        dim3 grid(n_cta_x, 1);
        if (write_max) k_vh_stage1_sm<TN, true><<<grid, VH_S1_THREADS, 0, stream>>>(w.hi, w.d, row_meta, w.n_in, w.n_out, xq, work.xd_half, work.sx253, work.act_meta, work.lb, work.ub, work.cta_max, n_cta_x, T);
        else           k_vh_stage1_sm<TN, false><<<grid, VH_S1_THREADS, 0, stream>>>(w.hi, w.d, row_meta, w.n_in, w.n_out, xq, work.xd_half, work.sx253, work.act_meta, work.lb, work.ub, work.cta_max, n_cta_x, T);
    };
    auto go_lg = [&](auto tn_c, int ny) {
        constexpr int TN = decltype(tn_c)::value;
        dim3 grid(n_cta_x, ny);
        if (write_max) k_vh_stage1_lg<TN, true><<<grid, VH_S1_THREADS, 0, stream>>>(w.hi, w.d, row_meta, w.n_in, w.n_out, xq, work.xd_half, work.sx253, work.act_meta, work.lb, work.ub, work.cta_max, n_cta_x, T);
        else           k_vh_stage1_lg<TN, false><<<grid, VH_S1_THREADS, 0, stream>>>(w.hi, w.d, row_meta, w.n_in, w.n_out, xq, work.xd_half, work.sx253, work.act_meta, work.lb, work.ub, work.cta_max, n_cta_x, T);
    };
    switch (T) {
        case 1: go_sm(std::integral_constant<int, 1>{}); break;
        case 2: go_sm(std::integral_constant<int, 2>{}); break;
        case 3: go_sm(std::integral_constant<int, 3>{}); break;
        case 4: go_sm(std::integral_constant<int, 4>{}); break;
        case 5: go_lg(std::integral_constant<int, 5>{}, 1); break;
        case 6: go_lg(std::integral_constant<int, 6>{}, 1); break;
        case 7: go_lg(std::integral_constant<int, 7>{}, 1); break;
        case 8: go_lg(std::integral_constant<int, 8>{}, 1); break;
        default: go_lg(std::integral_constant<int, 8>{}, (T + 7) / 8); break;
    }
}

}  // namespace

void pack_q80_to_6p2_host(const void* q8_src, size_t row_bytes, int rows, int cols,
                           const int32_t* perm_to_vocab,
                           uint32_t* hi_out, uint32_t* lo_out, uint16_t* d_out, float2* row_meta_out) {
    if (rows <= 0 || cols <= 0 || (cols & 511) != 0) {
        throw std::invalid_argument("pack_q80_to_6p2_host: rows must be >0 and cols a positive multiple of 512");
    }
    const int S = cols >> 9;
    const int nb = cols >> 5;
    if (row_bytes < static_cast<size_t>(nb) * 34) {
        throw std::invalid_argument("pack_q80_to_6p2_host: row_bytes smaller than packed Q8_0 row");
    }
    const uint8_t* src_base = static_cast<const uint8_t*>(q8_src);

    auto pack4 = [](uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3) -> uint32_t {
        return a0 | (a1 << 8) | (a2 << 16) | (a3 << 24);
    };

    auto worker = [&](int r_begin, int r_end) {
        for (int r = r_begin; r < r_end; ++r) {
            int src_r = perm_to_vocab ? perm_to_vocab[r] : r;
            const uint8_t* src_row = src_base + static_cast<size_t>(src_r) * row_bytes;
            double sum_sq_res = 0.0;
            double sum_sq_dw = 0.0;

            for (int s = 0; s < S; ++s) {
                for (int b_local = 0; b_local < 16; ++b_local) {
                    const uint8_t* blk = src_row + static_cast<size_t>(s * 16 + b_local) * 34;
                    uint16_t d_bits;
                    std::memcpy(&d_bits, blk, 2);
                    d_out[static_cast<size_t>(r) * nb + s * 16 + b_local] = d_bits;
                    float dw = vh_f16_to_f32_host(d_bits);
                    sum_sq_dw += static_cast<double>(dw) * static_cast<double>(dw);

                    for (int half = 0; half < 2; ++half) {
                        const int lane = (b_local << 1) | half;
                        const int8_t* w = reinterpret_cast<const int8_t*>(blk + 2 + half * 16);
                        uint32_t u[16], r2[16];
                        for (int j = 0; j < 16; ++j) {
                            uint32_t wb = static_cast<uint32_t>(static_cast<uint8_t>(w[j])) ^ 0x80u;
                            u[j] = wb >> 2;
                            r2[j] = wb & 3u;
                            double delta = static_cast<double>(r2[j]) - 1.5;
                            double scaled = static_cast<double>(dw) * delta;
                            sum_sq_res += scaled * scaled;
                        }
                        uint32_t b0_lo = pack4(u[0] & 0xFu, u[1] & 0xFu, u[2] & 0xFu, u[3] & 0xFu);
                        uint32_t b1_lo = pack4(u[4] & 0xFu, u[5] & 0xFu, u[6] & 0xFu, u[7] & 0xFu);
                        uint32_t b2_lo = pack4(u[8] & 0xFu, u[9] & 0xFu, u[10] & 0xFu, u[11] & 0xFu);
                        uint32_t b3_lo = pack4(u[12] & 0xFu, u[13] & 0xFu, u[14] & 0xFu, u[15] & 0xFu);
                        uint32_t b0_hi = pack4(u[0] >> 4, u[1] >> 4, u[2] >> 4, u[3] >> 4);
                        uint32_t b1_hi = pack4(u[4] >> 4, u[5] >> 4, u[6] >> 4, u[7] >> 4);
                        uint32_t b2_hi = pack4(u[8] >> 4, u[9] >> 4, u[10] >> 4, u[11] >> 4);
                        uint32_t b3_hi = pack4(u[12] >> 4, u[13] >> 4, u[14] >> 4, u[15] >> 4);

                        size_t hbase = (static_cast<size_t>(r) * S + s) * 96 + lane;
                        hi_out[hbase]      = b0_lo | (b1_lo << 4);
                        hi_out[hbase + 32] = b2_lo | (b3_lo << 4);
                        hi_out[hbase + 64] = b0_hi | (b1_hi << 2) | (b2_hi << 4) | (b3_hi << 6);

                        lo_out[(static_cast<size_t>(r) * S + s) * 32 + lane] =
                            pack4(r2[0], r2[1], r2[2], r2[3])
                            | (pack4(r2[4], r2[5], r2[6], r2[7]) << 2)
                            | (pack4(r2[8], r2[9], r2[10], r2[11]) << 4)
                            | (pack4(r2[12], r2[13], r2[14], r2[15]) << 6);
                    }
                }
            }
            if (row_meta_out) {
                row_meta_out[r] = make_float2(
                    static_cast<float>(std::sqrt(sum_sq_res) * 1.0001 + 1.0e-6),
                    static_cast<float>(std::sqrt(sum_sq_dw) * 1.0001 + 1.0e-6));
            }
        }
    };

    int n_threads = rows >= 2048 ? static_cast<int>(std::min<unsigned>(12u, std::max<unsigned>(1u, std::thread::hardware_concurrency()))) : 1;
    if (n_threads <= 1) {
        worker(0, rows);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(n_threads);
    int chunk = (rows + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; ++t) {
        int b = t * chunk;
        int e = std::min(rows, b + chunk);
        if (b < e) pool.emplace_back(worker, b, e);
    }
    for (auto& th : pool) th.join();
}

int vhead_max_cta(int rows, int T) {
    (void) T;
    return (rows + VH_S1_RPC - 1) / VH_S1_RPC;
}

size_t vhead_cta_max_floats(int rows, int T) {
    return static_cast<size_t>(vhead_max_cta(rows, T)) * static_cast<size_t>(T);
}

template <bool SPLIT>
static void launch_all_logits(const VHead6p2& w, const void* y, const int8_t* tq, const float* td,
                              float* logits, int T, cudaStream_t stream) {
    const int n_cta_x = (w.n_out + VH_S1_RPC - 1) / VH_S1_RPC;
    auto go = [&](auto tn_c, int ny) {
        constexpr int TN = decltype(tn_c)::value;
        dim3 grid(n_cta_x, ny);
        k_vh_all_logits<TN, VH_S1_R, SPLIT><<<grid, VH_S1_THREADS, 0, stream>>>(w.hi, w.lo, w.d, w.perm_to_vocab, w.n_in, w.n_out, y, tq, td, logits, T);
    };
    switch (T) {
        case 1: go(std::integral_constant<int, 1>{}, 1); break;
        case 2: go(std::integral_constant<int, 2>{}, 1); break;
        case 3: go(std::integral_constant<int, 3>{}, 1); break;
        case 4: go(std::integral_constant<int, 4>{}, 1); break;
        case 5: go(std::integral_constant<int, 5>{}, 1); break;
        case 6: go(std::integral_constant<int, 6>{}, 1); break;
        case 7: go(std::integral_constant<int, 7>{}, 1); break;
        case 8: go(std::integral_constant<int, 8>{}, 1); break;
        default: go(std::integral_constant<int, 8>{}, (T + 7) / 8); break;
    }
}

void vhead_all_logits_6p2(const VHead6p2& w, const void* y, float* logits, int T, void* st) {
    if (T <= 0 || T > MAXT) throw std::invalid_argument("vhead_all_logits_6p2: bad T");
    cudaStream_t stream = static_cast<cudaStream_t>(st);
    int8_t* tq = nullptr;
    float* td = nullptr;
    vh_twin_of(y, w.n_in, tq, td);
    if (tq) launch_all_logits<true>(w, y, tq, td, logits, T, stream);
    else    launch_all_logits<false>(w, y, nullptr, nullptr, logits, T, stream);
    vh_ck("vhead_all_logits_6p2");
}

void vhead_verify_6p2(const VHead6p2& w, const void* y, int T, int v0, float* best_v, int* best_i,
                      unsigned long long* keys, unsigned* err, const VHeadWork& work, void* st) {
    if (T <= 0 || T > MAXT) throw std::invalid_argument("vhead_verify_6p2: bad T");
    if (w.n_out > work.max_rows || w.n_in > work.max_in) {
        throw std::invalid_argument("vhead_verify_6p2: work buffer too small");
    }
    cudaStream_t stream = static_cast<cudaStream_t>(st);
    const int n_cta_x = vhead_max_cta(w.n_out, T);
    const int8_t* xq = nullptr;
    const float* xd = nullptr;
    launch_stage0_and_get_xy(y, w.n_in, T, work, stream, xq, xd);

    launch_stage1(w, xq, T, true, work, stream);
    k_vh_max_reduce<<<T, 256, 0, stream>>>(work.cta_max, n_cta_x, work.lcut, work.cand_cnt);

    dim3 fgrid((w.n_out + 255) / 256, T);
    k_vh_filter<false><<<fgrid, 256, 0, stream>>>(work.lb, work.ub, work.lcut, w.n_out, work.cand_idx, work.cand_cnt);

    dim3 s2grid(240, T);
    k_vh_stage2_verify<<<s2grid, VH_THREADS, 0, stream>>>(
        w.hi, w.lo, w.d, w.perm_to_vocab, w.n_in, w.n_out, v0, xq, xd,
        work.cand_idx, work.cand_cnt, keys, err);
    k_vh_argmax_fin<<<1, T, 0, stream>>>(T, v0, keys, best_v, best_i);
    vh_ck("vhead_verify_6p2");
}

void vhead_top_tokens_6p2(const VHead6p2& w, const void* y, int T, int v0,
                          const int* pen_tok, const int* pen_cnt, const int* pen_n, int pen_stride,
                          const float* pen_par, float* out_v, int* out_i, unsigned* err,
                          const VHeadWork& work, void* st) {
    if (T <= 0 || T > MAXT) throw std::invalid_argument("vhead_top_tokens_6p2: bad T");
    if (w.n_out > work.max_rows || w.n_in > work.max_in) {
        throw std::invalid_argument("vhead_top_tokens_6p2: work buffer too small");
    }
    cudaStream_t stream = static_cast<cudaStream_t>(st);
    const int8_t* xq = nullptr;
    const float* xd = nullptr;
    launch_stage0_and_get_xy(y, w.n_in, T, work, stream, xq, xd);

    launch_stage1(w, xq, T, false, work, stream);

    if (pen_tok && pen_cnt && pen_n && pen_par) {
        k_vh_apply_pen_bounds<<<T, 256, 0, stream>>>(
            work.lb, work.ub, w.n_out, v0, w.vocab_to_local, pen_tok, pen_cnt, pen_n, pen_stride, pen_par);
    }

    k_vh_select_lcut64<<<T, 256, 0, stream>>>(work.lb, work.ub, w.n_out, work.lcut, work.cand_cnt);

    dim3 fgrid((w.n_out + 255) / 256, T);
    k_vh_filter<true><<<fgrid, 256, 0, stream>>>(work.lb, work.ub, work.lcut, w.n_out, work.cand_idx, work.cand_cnt);

    dim3 s2grid(240, T);
    k_vh_stage2_top<<<s2grid, VH_THREADS, 0, stream>>>(
        w.hi, w.lo, w.d, w.n_in, w.n_out, xq, xd, work.cand_idx, work.cand_cnt, work.ub, err);

    if (pen_tok && pen_cnt && pen_n && pen_par) {
        k_vh_repen_logits<<<T, 256, 0, stream>>>(
            work.ub, w.n_out, v0, w.vocab_to_local, pen_tok, pen_cnt, pen_n, pen_stride, pen_par);
    }

    k_vh_topk_cand<<<T, 256, 0, stream>>>(
        work.ub, w.perm_to_vocab, w.n_out, v0, -1, work.cand_idx, work.cand_cnt, out_v, out_i);
    vh_ck("vhead_top_tokens_6p2");
}

void vhead_mtp_fast_6p2(const VHead6p2& w, const uint16_t* cent_f16, const int32_t* cluster_off, int n_clusters,
                        int top_clusters, const void* y, int T, int v0, const int* pen, const float* pen_par,
                        float* best_v, int* best_i, unsigned long long* keys, unsigned* err,
                        const VHeadWork& work, void* st) {
    if (T <= 0 || T > MAXT) throw std::invalid_argument("vhead_mtp_fast_6p2: bad T");
    if (n_clusters <= 0 || n_clusters > w.n_out) throw std::invalid_argument("vhead_mtp_fast_6p2: bad n_clusters");
    cudaStream_t stream = static_cast<cudaStream_t>(st);
    const int8_t* xq = nullptr;
    const float* xd = nullptr;
    launch_stage0_and_get_xy(y, w.n_in, T, work, stream, xq, xd);

    dim3 cgrid((n_clusters + VH_WARPS - 1) / VH_WARPS, T);
    k_vh_score_centroids<<<cgrid, VH_THREADS, 0, stream>>>(
        cent_f16, n_clusters, w.n_in, xq, xd, work.lb, T);

    k_vh_expand_clusters<<<T, 256, 0, stream>>>(
        work.lb, cluster_off, n_clusters, top_clusters, w.n_out, work.cand_idx, work.cand_cnt);

    dim3 s2grid(240, T);
    k_vh_stage2_top<<<s2grid, VH_THREADS, 0, stream>>>(
        w.hi, w.lo, w.d, w.n_in, w.n_out, xq, xd, work.cand_idx, work.cand_cnt, work.ub, err);

    if (pen && pen_par) {
        k_vh_penalize_row_6p2<<<1, 256, 0, stream>>>(
            work.ub + static_cast<size_t>(T - 1) * w.n_out, w.n_out, w.vocab_to_local, pen, pen_par);
    }

    k_vh_cand_argmax<<<dim3(16, T), 256, 0, stream>>>(
        work.ub, w.perm_to_vocab, w.n_out, work.cand_idx, work.cand_cnt, keys);
    k_vh_argmax_fin<<<1, T, 0, stream>>>(T, v0, keys, best_v, best_i);
    vh_ck("vhead_mtp_fast_6p2");
}

void vhead_mtp_fast_topk_6p2(const VHead6p2& w, int row, int v0, float* out_v, int* out_i, unsigned* err,
                             const VHeadWork& work, void* st) {
    if (row < 0 || row >= MAXT) throw std::invalid_argument("vhead_mtp_fast_topk_6p2: bad row");
    (void) err;  // NaN bit already set in vhead_mtp_fast_6p2
    cudaStream_t stream = static_cast<cudaStream_t>(st);
    k_vh_topk_cand<<<1, 256, 0, stream>>>(
        work.ub, w.perm_to_vocab, w.n_out, v0, row, work.cand_idx, work.cand_cnt, out_v, out_i);
    vh_ck("vhead_mtp_fast_topk_6p2");
}

}  // namespace fnx
