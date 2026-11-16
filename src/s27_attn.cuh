// The 27B's attention split by heads: card c holds query heads [S27_NQ c, S27_NQ (c + 1)) and the S27_NKV KV heads
// they read (query head h reads KV head h / 6), K/V for every position (f16): 12 query heads and two KV heads a card on
// two cards, 8 and two on three (a KV head straddling two cards' heads is kept on both), 6 and one on four. No exchange
// between the cards until wo's sum. Rows of the projection: [q | gate] for each of the S27_NQ heads, then the S27_NKV
// K heads, then the S27_NKV V heads (S27_QKV values).
// K/V live in a paged pool: a page holds S27_PP positions of every attention layer and the draft layer (a slot each:
// its K rows [S27_PP][S27_NKV][HD], then its V rows); a sequence's page table maps its logical pages to physical ones.
// A window's rows each have a position and a page table: one sequence's rows at *pos0 + t on pt0, or a batched
// window's, each from its lane (LaneRows: the row's position word and table; tabs: its lane's rotary table, null for a
// lane without an image)
#pragma once

#include <cuda_fp16.h>

#include "kernels.hpp"
#include "rope.hpp"

namespace fnx {

// the KV heads card c's query heads read
constexpr int s27_nkv_of(int c) { return (c * (NQH / NCARD) + NQH / NCARD - 1) / (NQH / NKVH) - c * (NQH / NCARD) / (NQH / NKVH) + 1; }
constexpr bool s27_nkv_even(int c) { return c == NCARD || (s27_nkv_of(c) == s27_nkv_of(0) && s27_nkv_even(c + 1)); }
constexpr int S27_NQ = NQH / NCARD, S27_NKV = s27_nkv_of(0);
constexpr int S27_QKV = S27_NQ * 2 * HD + 2 * S27_NKV * HD;
constexpr int S27_ASPLIT = 32;   // position splits of a row's attention

constexpr int S27_SLOTS = NL / 4 + 1;   // the attention layers, then the draft layer
constexpr size_t S27_ROWH = (size_t) S27_NKV * HD;              // halves a position takes in K (or V)
constexpr size_t S27_SLOT = (size_t) 2 * S27_PP * S27_ROWH;     // halves of a layer's slot in a page
constexpr size_t S27_PAGE = (size_t) S27_SLOTS * S27_SLOT;      // halves a page
// the attention and the prompt path read K/V rows as 16-byte uint4s (8 halves) from the pool's base
static_assert(S27_ROWH % 8 == 0 && HD % 8 == 0 && S27_SLOT % 8 == 0 && S27_PAGE % 8 == 0, "K/V rows start on 16 bytes");
static_assert(NQH % NCARD == 0, "query heads split evenly");
// every card's K/V the same size (the pool's page layout is one for all); six cards would put one KV head on some, two
// on others
static_assert(NQH % NCARD == 0 && s27_nkv_even(0), "the 27B's query heads split evenly, every card reading as many KV heads");

// the slot an attention layer's K/V take (the draft layer: S27_SLOTS - 1)
inline int s27_slot(int l) { return l / 4; }

__device__ __forceinline__ size_t s27_k(const int* pt, int slot, int p) {
    return (size_t) pt[p / S27_PP] * S27_PAGE + (size_t) slot * S27_SLOT + (size_t) (p % S27_PP) * S27_ROWH;
}
__device__ __forceinline__ int s27_row_pos(const LaneRows& lr, const int* pos0, int t) { return lr.pos ? *lr.pos[t] : *pos0 + t; }
__device__ __forceinline__ const int* s27_row_pt(const LaneRows& lr, const int* pt0, int t) { return lr.pt ? lr.pt[t] : pt0; }

__device__ __forceinline__ float s27_sum256(float v, float* sh) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0.f;
    for (int i = 0; i < HD / 32; ++i) s += sh[i];
    __syncthreads();
    return s;
}

// per (head, row): q heads normed and rotated in place; K heads normed, rotated and stored at the row's position; V
// heads stored. HD threads a block, one per value. qkv rows at row_stride
static __global__ void __launch_bounds__(HD) k_s27_prep(float* qkv, int row_stride, const float* qn, const float* kn, __half* pool, int slot,
                                                 const int* pos0p, const int* pt0, LaneRows lr, const int* const* tabs, float eps,
                                                 Rope rope) {
    __shared__ float sh[HD / 32];
    __shared__ float x[HD];
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x, pos = s27_row_pos(lr, pos0p, t);
    if (tabs) rope.tab = tabs[t];
    const int* pt = s27_row_pt(lr, pt0, t);
    float* row = qkv + (size_t) t * row_stride;
    float* p;
    const float* w;
    if (h < S27_NQ) p = row + h * 2 * HD, w = qn;
    else if (h < S27_NQ + S27_NKV) p = row + S27_NQ * 2 * HD + (h - S27_NQ) * HD, w = kn;
    else {
        const int vh = h - S27_NQ - S27_NKV;
        pool[s27_k(pt, slot, pos) + (size_t) S27_PP * S27_ROWH + vh * HD + i] = __float2half_rn(row[S27_NQ * 2 * HD + (S27_NKV + vh) * HD + i]);
        return;
    }
    const float v = p[i];
    const float ms = s27_sum256(v * v, sh) / (float) HD;
    x[i] = v * rsqrtf(ms + eps) * w[i];
    __syncthreads();
    if (i < NROT / 2) {
        const float theta = (float) rope_pos(rope, pos, i) * rope_unit_freq(rope, i, NROT);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        sn *= rope.mscale, cs *= rope.mscale;
        const float x0 = x[i], x1 = x[i + NROT / 2];
        x[i] = x0 * cs - x1 * sn;
        x[i + NROT / 2] = x0 * sn + x1 * cs;
    }
    __syncthreads();
    if (h < S27_NQ) p[i] = x[i];
    else pool[s27_k(pt, slot, pos) + (h - S27_NQ) * HD + i] = __float2half_rn(x[i]);
}

// per (split, row, local KV head): the split's share of the row's positions [0, pos], four warps over them, each lane 8
// values of the head; K and V loaded once from the paged pool and shared across all GQA query heads of that KV head on
// this card (up to S27_QPK = 6 heads); each head's max, sum and weighted values into part ([T][NQ][S27_ASPLIT] of HD + 2)
constexpr int S27_QPK = NQH / NKVH;
static __global__ void __launch_bounds__(128, 3) k_s27_attn(const float* qkv, int row_stride, const __half* pool, int slot, const int* pos0p,
                                                     const int* pt0, LaneRows lr, int kvh0, int q0, float* part) {
    const int s = blockIdx.x, t = blockIdx.y, kvh = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int len = s27_row_pos(lr, pos0p, t) + 1, chunk = (len + S27_ASPLIT - 1) / S27_ASPLIT;
    const int* pt = s27_row_pt(lr, pt0, t);
    const int p0 = s * chunk, p1 = min(len, p0 + chunk);
    const int h0 = max(0, (kvh0 + kvh) * S27_QPK - q0);
    const int h1 = min(S27_NQ, (kvh0 + kvh + 1) * S27_QPK - q0);
    const int nh = h1 - h0;

    __shared__ union {
        float sq[S27_QPK][8][32];
        struct {
            float wm[4], wl[4], wa[4][HD];
        } red;
    } smem;
    const float* qrow = qkv + (size_t) t * row_stride;
    for (int hh = 0; hh < nh; ++hh) {
        for (int idx = threadIdx.x; idx < HD; idx += 128)
            smem.sq[hh][idx & 7][idx >> 3] = qrow[(h0 + hh) * 2 * HD + idx] * 0.0625f;   // 1 / sqrt(HD)
    }
    __syncthreads();

    float m[S27_QPK], l[S27_QPK], acc[S27_QPK][8];
#pragma unroll
    for (int hh = 0; hh < S27_QPK; ++hh) {
        m[hh] = -INFINITY;
        l[hh] = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[hh][j] = 0.f;
    }

    for (int p = p0 + warp; p < p1; p += 4) {
        const size_t at = s27_k(pt, slot, p) + kvh * HD + lane * 8;
        const uint4 kr = *reinterpret_cast<const uint4*>(pool + at);
        const __half2* k2 = reinterpret_cast<const __half2*>(&kr);
        float kf[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __half22float2(k2[j]);
            kf[2 * j] = f.x;
            kf[2 * j + 1] = f.y;
        }
        const uint4 vr = *reinterpret_cast<const uint4*>(pool + at + (size_t) S27_PP * S27_ROWH);
        const __half2* v2 = reinterpret_cast<const __half2*>(&vr);
        float vf[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __half22float2(v2[j]);
            vf[2 * j] = f.x;
            vf[2 * j + 1] = f.y;
        }
#pragma unroll
        for (int hh = 0; hh < S27_QPK; ++hh) {
            if (hh >= nh) break;
            float d = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) d = fmaf(smem.sq[hh][j][lane], kf[j], d);
#pragma unroll
            for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
            const float mn = fmaxf(m[hh], d), c = __expf(m[hh] - mn), e = __expf(d - mn);
            l[hh] = l[hh] * c + e;
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[hh][j] = fmaf(acc[hh][j], c, e * vf[j]);
            m[hh] = mn;
        }
    }

#pragma unroll
    for (int hh = 0; hh < S27_QPK; ++hh) {
        if (hh >= nh) break;
        __syncthreads();
        if (lane == 0) smem.red.wm[warp] = m[hh], smem.red.wl[warp] = l[hh];
#pragma unroll
        for (int j = 0; j < 8; ++j) smem.red.wa[warp][lane * 8 + j] = acc[hh][j];
        __syncthreads();
        float M = -INFINITY;
        for (int w = 0; w < 4; ++w) M = fmaxf(M, smem.red.wm[w]);
        const int h = h0 + hh;
        float* out = part + (((size_t) t * S27_NQ + h) * S27_ASPLIT + s) * (HD + 2);
        for (int d = threadIdx.x; d < HD; d += blockDim.x) {
            float a = 0.f;
            if (M != -INFINITY)
                for (int w = 0; w < 4; ++w) a += smem.red.wa[w][d] * __expf(smem.red.wm[w] - M);
            out[d] = a;
        }
        if (threadIdx.x == 0) {
            float L = 0.f;
            if (M != -INFINITY)
                for (int w = 0; w < 4; ++w) L += smem.red.wl[w] * __expf(smem.red.wm[w] - M);
            out[HD] = M;
            out[HD + 1] = L;
        }
    }
}

// per (row, head): the splits merged, gated by sigmoid(gate), and quantized directly into the split twin (tq, td)
// of a q8_1 buffer of S27_NQ * HD-wide rows so wo's q8s_gemv needs no intermediate fp32 buffer or quantize_q8_1 launch
static __global__ void __launch_bounds__(HD) k_s27_merge(const float* qkv, int row_stride, const float* part,
                                                         int8_t* tq, float* td) {
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const float* ps = part + ((size_t) t * S27_NQ + h) * S27_ASPLIT * (HD + 2);
    float M = -INFINITY;
    for (int s = 0; s < S27_ASPLIT; ++s) M = fmaxf(M, ps[(size_t) s * (HD + 2) + HD]);
    float a = 0.f, L = 0.f;
    for (int s = 0; s < S27_ASPLIT; ++s) {
        const float* p = ps + (size_t) s * (HD + 2);
        if (p[HD] == -INFINITY) continue;
        const float e = __expf(p[HD] - M);
        a = fmaf(p[d], e, a);
        L = fmaf(p[HD + 1], e, L);
    }
    const float g = qkv[(size_t) t * row_stride + h * 2 * HD + HD + d];
    const float v = a / L / (1.f + __expf(-g));
    const size_t idx = ((size_t) t * S27_NQ + h) * HD + d;
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float sc = amax / 127.0f;
    tq[idx] = amax == 0.0f ? 0 : (int8_t) roundf(v / sc);
    if ((d & 31) == 0)
        td[(size_t) t * (S27_NQ * HD / 32) + h * (HD / 32) + (d >> 5)] = __half2float(__float2half(sc));
}

}  // namespace fnx
