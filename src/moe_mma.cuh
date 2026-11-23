// The prompt path's routed experts on the tensor cores, as llama.cpp's CUDA backend runs a mul_mat_id of more than
// MMQ_DP4A_MAX_BATCH_SIZE rows on a V100: each expert's weights dequantized to f16 (the f32 product d * q - m rounded
// once), the activations as f16, the products f16 x f16 with fp32 accumulation (llama.cpp's cuBLAS call accumulates
// in f16). A block of 4 warps takes one BigJob (one expert, up to PF_JOB rows) and MM_BM of the expert's rows: per
// MM_KC columns the weight tile is dequantized into shared memory next to the job's activation rows, and each warp runs
// a 32-row x 32-token corner of the output on HMMA.884 (mma_884: each quad-pair 2 x 2 tiles of 8 x 8, its operands
// 16-byte loads laid out without bank conflicts; shared-memory bandwidth is what limits these kernels on a V100). The
// next step's raw weights and
// activations are fetched into registers while the tensor cores run the current one. On random experts of the real
// shapes (tools/moe_mma_bench, 4,096 tokens) this takes gate|up 3.4 ms and down 2.2 ms per layer and card, against
// 4.8 and 4.0 ms for 8 warps of 16-row tiles at 128 columns.
#if FNX_ARCH >= 70
#include <mma.h>
#endif

namespace {

#if FNX_ARCH >= 70
// a k-step of MM_KC columns; a K-quant row takes MM_TPR threads (16 quant bytes, 32 weights each), so one pass of the
// block's threads fills MM_PASS rows; tile rows are padded to MM_LD halves (8: the fewest bank conflicts)
constexpr int MM_BM = 64, MM_KC = 64, MM_TPR = MM_KC / 32, MM_THREADS = 128, MM_PASS = MM_THREADS / MM_TPR, MM_LD = MM_KC + 8;
constexpr int MM_XV = PF_JOB * MM_KC / 8 / MM_THREADS;   // activation uint4s per thread and step, at most
// a K-quant tile's thread: its row and which half of the step's 64 columns (MM_TPR = 2). Eight threads in a row take 8
// rows at the same columns, so a quarter-warp's 16-byte stores hit 8 different bank groups (a 144-byte row stride); both
// halves of a row are still fetched by one warp (threads t and t + 8)
static_assert(MM_TPR == 2 && MM_THREADS == 128, "the K-quant thread layout");
__device__ __forceinline__ int mm_kq_row() { return (threadIdx.x & 7) | ((threadIdx.x >> 4) << 3); }
__device__ __forceinline__ int mm_kq_part() { return (threadIdx.x >> 3) & 1; }

// get_scale_min_k4 on a K-quant block's first 16 bytes (scales[0..11] are hd.y, hd.z, hd.w), by shifts: a byte index
// into a register would put it on the stack
__device__ __forceinline__ void mm_scale_min_k4(int j, const uint4& hd, int& d, int& m) {
    if (j < 4) {
        d = (hd.y >> (8 * j)) & 63;
        m = (hd.z >> (8 * j)) & 63;
    } else {
        const int s = 8 * (j - 4);
        d = ((hd.w >> s) & 0xF) | (((hd.y >> (s + 6)) & 3) << 4);
        m = ((hd.w >> (s + 4)) & 0xF) | (((hd.z >> (s + 6)) & 3) << 4);
    }
}
// the 4 bytes of w (each 0..255) as two half2 {b0, b1}, {b2, b3}: each byte under 0x64 is 1024 + b exactly; minus `bias`
__device__ __forceinline__ void mm_bytes_h2(uint32_t w, __half2 bias, __half2& a, __half2& b) {
    uint32_t x = __byte_perm(w, 0x64646464u, 0x4140), y = __byte_perm(w, 0x64646464u, 0x4342);
    a = __hsub2(*reinterpret_cast<__half2*>(&x), bias);
    b = __hsub2(*reinterpret_cast<__half2*>(&y), bias);
}
__device__ __forceinline__ __half2 mm_k1024() { return __half2half2(__ushort_as_half(0x6400)); }
// bit k of the low 4 bits of t to bit 4 of byte k
__device__ __forceinline__ uint32_t mm_spread4(uint32_t t) {
    return ((t & 1u) << 4) | ((t & 2u) << 11) | ((t & 4u) << 18) | ((t & 8u) << 25);
}
// 16 values (8 half2) to shared memory
__device__ __forceinline__ void mm_store8(__half* dst, const __half2* h) {
    reinterpret_cast<uint4*>(dst)[0] = *reinterpret_cast<const uint4*>(h);
    reinterpret_cast<uint4*>(dst)[1] = *reinterpret_cast<const uint4*>(h + 4);
}

// One k-step of the weight tile, MM_PASS rows x kc columns from column k0: fetch() reads this thread's raw bytes,
// put() writes their f16 values into shared memory, row stride MM_LD. The quants become f16 exactly (the 0x6400
// trick) and one half fma applies the scale: for Q5_1 and Q8_0 that is llama.cpp's dequantize_block_* value (its f32
// d * q + m is exact, rounded once to f16); for the K-quants the sub-block scale d * sc and min dmin * m are f16 here,
// rounded where llama.cpp keeps them f32 (at most an ulp of f16 per weight).
// The K-quants: 4 threads per row, each 16 bytes of the step's 64 quant bytes (one byte = a weight of sub-block 2j and
// one of 2j + 1). The 32-weight blocks: a block per thread (MM_TPR per row; those past a short last step idle).
template <int TYPE> struct MmTile;

// the K-quants' two sub-blocks 2j and 2j + 1 of one thread: lo / hi nibbles of the 16 quant bytes (plus, Q5_K, the
// high bits), scaled, into columns (i / 2) * 64 + l0 and + 32
template <bool Q5>
__device__ __forceinline__ void mm_put_k(const uint4& hd, const uint4& qv, const uint4& hv, bool live, int k0, __half* sw) {
    const int r = mm_kq_row(), i = mm_kq_part();
    const int j = (k0 % 256) / 64 + i / 2, l0 = 16 * (i % 2);
    __half2 lo[8], hi[8];
    if (live) {
        const float d = __half2float(__ushort_as_half((uint16_t) (hd.x & 0xFFFF))), dmin = __half2float(__ushort_as_half((uint16_t) (hd.x >> 16)));
        int sc, m;
        mm_scale_min_k4(2 * j, hd, sc, m);
        const __half2 d1 = __float2half2_rn(d * sc), nm1 = __float2half2_rn(-(dmin * m));
        mm_scale_min_k4(2 * j + 1, hd, sc, m);
        const __half2 d2 = __float2half2_rn(d * sc), nm2 = __float2half2_rn(-(dmin * m));
        const uint32_t q[4] = {qv.x, qv.y, qv.z, qv.w};
        const uint32_t h[4] = {hv.x, hv.y, hv.z, hv.w};
        const __half2 k = mm_k1024();
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            uint32_t a = q[w] & 0x0F0F0F0Fu, b = (q[w] >> 4) & 0x0F0F0F0Fu;
            if (Q5) {
                a |= ((h[w] >> (2 * j)) & 0x01010101u) << 4;
                b |= ((h[w] >> (2 * j + 1)) & 0x01010101u) << 4;
            }
            __half2 x0, x1;
            mm_bytes_h2(a, k, x0, x1);
            lo[2 * w] = __hfma2(x0, d1, nm1);
            lo[2 * w + 1] = __hfma2(x1, d1, nm1);
            mm_bytes_h2(b, k, x0, x1);
            hi[2 * w] = __hfma2(x0, d2, nm2);
            hi[2 * w + 1] = __hfma2(x1, d2, nm2);
        }
    } else {
#pragma unroll
        for (int l = 0; l < 8; ++l) lo[l] = hi[l] = __float2half2_rn(0.f);
    }
    __half* dst = sw + r * MM_LD + (i / 2) * 64 + l0;
    mm_store8(dst, lo);
    mm_store8(dst + 32, hi);
}

template <> struct MmTile<GGML_TYPE_Q4_K> {
    uint4 hd, qv;
    bool live;
    __device__ __forceinline__ void fetch(const char* we, size_t row_a, int nrows, int row0, int k0, int) {
        const int r = mm_kq_row(), i = mm_kq_part(), row = row0 + r;
        const int j = (k0 % 256) / 64 + i / 2, l0 = 16 * (i % 2);
        live = row < nrows;
        if (live) {
            const uint8_t* b = (const uint8_t*) we + (size_t) row * row_a + (size_t) (k0 / 256) * 144;
            hd = *reinterpret_cast<const uint4*>(b);
            qv = *reinterpret_cast<const uint4*>(b + 16 + 32 * j + l0);
        }
    }
    __device__ __forceinline__ void put(int k0, int, __half* sw) const { mm_put_k<false>(hd, qv, qv, live, k0, sw); }
};
template <> struct MmTile<GGML_TYPE_Q5_K> {
    uint4 hd, hv, qv;
    bool live;
    __device__ __forceinline__ void fetch(const char* we, size_t row_a, int nrows, int row0, int k0, int) {
        const int r = mm_kq_row(), i = mm_kq_part(), row = row0 + r;
        const int j = (k0 % 256) / 64 + i / 2, l0 = 16 * (i % 2);
        live = row < nrows;
        if (live) {
            const uint8_t* b = (const uint8_t*) we + (size_t) row * row_a + (size_t) (k0 / 256) * 176;
            hd = *reinterpret_cast<const uint4*>(b);
            hv = *reinterpret_cast<const uint4*>(b + 16 + l0);
            qv = *reinterpret_cast<const uint4*>(b + 48 + 32 * j + l0);
        }
    }
    __device__ __forceinline__ void put(int k0, int, __half* sw) const { mm_put_k<true>(hd, qv, hv, live, k0, sw); }
};
template <> struct MmTile<GGML_TYPE_Q5_1> {
    uint2 v[3];
    bool live, item;
    __device__ __forceinline__ void fetch(const char* we, size_t row_a, int nrows, int row0, int k0, int kc) {
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR, row = row0 + r;
        item = u * 32 < kc;
        live = item && row < nrows;
        if (live) {
            const uint2* b = reinterpret_cast<const uint2*>((const uint8_t*) we + (size_t) row * row_a + (size_t) (k0 / 32 + u) * 24);
            v[0] = b[0];
            v[1] = b[1];
            v[2] = b[2];
        }
    }
    // byte l of the 16 quant bytes: value l (low nibble, high bit qh bit l) and value l + 16 (high nibble, qh bit l + 16)
    __device__ __forceinline__ void put(int, int kc, __half* sw) const {
        if (!item) return;
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        __half2 lo[8], hi[8];
        if (live) {
            const __half2 d = __half2half2(__ushort_as_half((uint16_t) (v[0].x & 0xFFFF))), m = __half2half2(__ushort_as_half((uint16_t) (v[0].x >> 16)));
            const uint32_t qh = v[0].y, q[4] = {v[1].x, v[1].y, v[2].x, v[2].y};
            const __half2 k = mm_k1024();
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint32_t a = (q[w] & 0x0F0F0F0Fu) | mm_spread4(qh >> (4 * w));
                const uint32_t b = ((q[w] >> 4) & 0x0F0F0F0Fu) | mm_spread4(qh >> (16 + 4 * w));
                __half2 x0, x1;
                mm_bytes_h2(a, k, x0, x1);
                lo[2 * w] = __hfma2(x0, d, m);
                lo[2 * w + 1] = __hfma2(x1, d, m);
                mm_bytes_h2(b, k, x0, x1);
                hi[2 * w] = __hfma2(x0, d, m);
                hi[2 * w + 1] = __hfma2(x1, d, m);
            }
        } else {
#pragma unroll
            for (int l = 0; l < 8; ++l) lo[l] = hi[l] = __float2half2_rn(0.f);
        }
        mm_store8(sw + r * MM_LD + u * 32, lo);
        mm_store8(sw + r * MM_LD + u * 32 + 16, hi);
    }
};
template <> struct MmTile<GGML_TYPE_Q8_0> {
    uint16_t v[17];
    bool live, item;
    __device__ __forceinline__ void fetch(const char* we, size_t row_a, int nrows, int row0, int k0, int kc) {
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR, row = row0 + r;
        item = u * 32 < kc;
        live = item && row < nrows;
        if (live) {
            const uint16_t* b = reinterpret_cast<const uint16_t*>((const uint8_t*) we + (size_t) row * row_a + (size_t) (k0 / 32 + u) * 34);
#pragma unroll
            for (int l = 0; l < 17; ++l) v[l] = b[l];
        }
    }
    // the signed quants offset by 128 (bytes 0..255), so the f16 is 1152 + q: minus 1152, times d
    __device__ __forceinline__ void put(int, int kc, __half* sw) const {
        if (!item) return;
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        __half2 y[16];
        if (live) {
            const __half2 d = __half2half2(__ushort_as_half(v[0]));
            const __half2 k = __half2half2(__ushort_as_half(0x6480));   // 1152
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                const uint32_t a = ((uint32_t) v[1 + 2 * w] | ((uint32_t) v[2 + 2 * w] << 16)) ^ 0x80808080u;
                __half2 x0, x1;
                mm_bytes_h2(a, k, x0, x1);
                y[2 * w] = __hmul2(x0, d);
                y[2 * w + 1] = __hmul2(x1, d);
            }
        } else {
#pragma unroll
            for (int l = 0; l < 16; ++l) y[l] = __float2half2_rn(0.f);
        }
        mm_store8(sw + r * MM_LD + u * 32, y);
        mm_store8(sw + r * MM_LD + u * 32 + 16, y + 8);
    }
};
template <> struct MmTile<GGML_TYPE_Q6_K> {
    uint16_t ql[16], qh[16], dh;
    int8_t sc0, sc1;
    bool live, item;
    __device__ __forceinline__ void fetch(const char* we, size_t row_a, int nrows, int row0, int k0, int kc) {
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR, row = row0 + r;
        item = u * 32 < kc;
        live = item && row < nrows;
        if (live) {
            const int b32 = k0 / 32 + u, kb = b32 >> 3, sb = b32 & 7, h128 = sb >> 2, q32 = sb & 3;
            const uint8_t* blk = (const uint8_t*) we + (size_t) row * row_a + (size_t) kb * 210;
            const uint16_t* qlp = reinterpret_cast<const uint16_t*>(blk + h128 * 64 + (q32 & 1) * 32);
            const uint16_t* qhp = reinterpret_cast<const uint16_t*>(blk + 128 + h128 * 32);
#pragma unroll
            for (int l = 0; l < 16; ++l) {
                ql[l] = qlp[l];
                qh[l] = qhp[l];
            }
            const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);
            sc0 = sc[2 * sb];
            sc1 = sc[2 * sb + 1];
            dh = *reinterpret_cast<const uint16_t*>(blk + 208);
        }
    }
    __device__ __forceinline__ void put(int k0, int, __half* sw) const {
        if (!item) return;
        const int r = threadIdx.x / MM_TPR, u = threadIdx.x % MM_TPR;
        __half2 y[16];
        if (live) {
            const int sb = (k0 / 32 + u) & 7, q32 = sb & 3;
            const int ql_shift = (q32 >> 1) * 4, qh_shift = q32 * 2;
            const float d = __half2float(__ushort_as_half(dh));
            const __half2 d0 = __float2half2_rn(d * (float) sc0), d1 = __float2half2_rn(d * (float) sc1);
            const __half2 k = __half2half2(__ushort_as_half(0x6420));   // 1024 + 32 = 1056
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                const uint32_t qlw = (uint32_t) ql[2 * w] | ((uint32_t) ql[2 * w + 1] << 16);
                const uint32_t qhw = (uint32_t) qh[2 * w] | ((uint32_t) qh[2 * w + 1] << 16);
                const uint32_t a = ((qlw >> ql_shift) & 0x0F0F0F0Fu) | (((qhw >> qh_shift) & 0x03030303u) << 4);
                const __half2 ds = w < 4 ? d0 : d1;
                __half2 x0, x1;
                mm_bytes_h2(a, k, x0, x1);
                y[2 * w] = __hmul2(x0, ds);
                y[2 * w + 1] = __hmul2(x1, ds);
            }
        } else {
#pragma unroll
            for (int l = 0; l < 16; ++l) y[l] = __float2half2_rn(0.f);
        }
        mm_store8(sw + r * MM_LD + u * 32, y);
        mm_store8(sw + r * MM_LD + u * 32 + 16, y + 8);
    }
};

// the job's activation rows for one k-step: rows past n (up to the token tiles' 16 * nt) are zeros. A thread's rows and
// columns are the same at every step (element e = threadIdx.x + i * MM_THREADS is row e / MM_V8, 8 columns e % MM_V8)
constexpr int MM_V8 = MM_KC / 8;
struct MmX {
    uint4 v[MM_XV];
    int y[MM_XV];   // the row's token, -1 past the job's rows, -2 past its token tiles
    __device__ __forceinline__ void rows(const BigJob& jb, int nt) {
#pragma unroll
        for (int i = 0; i < MM_XV; ++i) {
            const int j = (threadIdx.x + i * MM_THREADS) / MM_V8;
            y[i] = j < jb.n ? jb.y[j] : j < nt * 16 ? -1 : -2;
        }
    }
    __device__ __forceinline__ void fetch(const __half* x, int ldx, int k0, int kc) {
        const int c = threadIdx.x % MM_V8 * 8;   // MM_THREADS is a multiple of MM_V8
#pragma unroll
        for (int i = 0; i < MM_XV; ++i) {
            v[i] = make_uint4(0, 0, 0, 0);
            if (y[i] >= 0 && c < kc) v[i] = *reinterpret_cast<const uint4*>(x + (size_t) y[i] * ldx + k0 + c);
        }
    }
    __device__ __forceinline__ void put(int kc, __half* sx) const {
        const int c = threadIdx.x % MM_V8 * 8;
#pragma unroll
        for (int i = 0; i < MM_XV; ++i)
            if (y[i] >= -1 && c < kc) *reinterpret_cast<uint4*>(sx + (threadIdx.x + i * MM_THREADS) / MM_V8 * MM_LD + c) = v[i];
    }
};

// d (each quad-pair's 8 x 8) += a (8 x 4, row-major: this lane's row, 4 halves in a0 a1) b (4 x 8, col-major: this
// lane's column), f16 products with f32 accumulation: Volta's HMMA.884
__device__ __forceinline__ void mma_884(float (&d)[8], unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
#if FNX_TC
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#endif
}

// the output staging's row stride in floats (8 past the tile: fewer conflicts than none or 16)
template <int BM>
constexpr int MM_SLD = BM + 8;
// the block's shared memory: the weight and activation tiles, later the output staging over them
template <int BM>
constexpr size_t mm_smem() { return std::max(sizeof(__half) * (BM + PF_JOB) * MM_LD, PF_JOB * MM_SLD<BM> * sizeof(float)); }

// out[jb.out[j] * nrows + row] = sum_k W_e[row][k] * x[jb.y[j] * ldx + k] for the job's rows j, BM rows of the expert
// from row BM * blockIdx.x. Each warp holds a 32-row x (64 / WT)-token output (2 x TT WMMA tiles, so a fragment loaded
// from shared memory feeds TT or 2 products): warp w takes rows 32 (w % WR) and tokens (64 / WT) (w / WR). A job of at
// most TT token tiles runs on every warp anyway: the WT warps of a row group then take every WT-th 16 columns of each
// k-step, summed at the end.
template <int TYPE, int BM>
__global__ void __launch_bounds__(MM_THREADS) k_moe_mma(MoeW w, const BigJob* __restrict__ jobs, const int* __restrict__ njobs,
                                                        const __half* __restrict__ x, int ldx, float* __restrict__ out) {
#if FNX_TC
    using namespace nvcuda;
    constexpr int WR = BM / 32, WT = MM_THREADS / 32 / WR, TT = 4 / WT, NP = BM / MM_PASS, LO = 16 * TT;
    static_assert(WR * WT == MM_THREADS / 32 && TT * WT == 4 && NP * MM_PASS == BM, "warp layout");
    const int ji = blockIdx.y;
    if (ji >= *njobs) return;
    __shared__ BigJob jb;
    extern __shared__ __align__(32) __half smem[];   // the weight tile (BM rows), then the activation rows (PF_JOB)
    __half* sw = smem;
    __half* sx = smem + BM * MM_LD;
    load_bigjob(jb, jobs + ji);
    const int n = jb.n, nt = (n + 15) / 16, warp = threadIdx.x >> 5, row0 = blockIdx.x * BM;
    const int rg = warp % WR, tg = warp / WR;
    const bool split = WT > 1 && nt <= TT;
    const int tb = split ? 0 : tg * TT;   // the warp's first token tile
    // the warp's 32 rows x 16 TT tokens on HMMA.884 (mma_884): quad-pair q (lanes 4q..4q+3, 4q+16..4q+19) takes rows
    // 16 (q >> 1) and tokens 8 TT (q & 1) of them, as 2 x TT tiles of 8 x 8; this lane loads row / token `rl` of a tile
    const int lane = threadIdx.x & 31, q = (lane >> 2) & 3, rl = (lane & 3) + ((lane >> 4) & 1) * 4;
    const __half* arow = sw + (rg * 32 + (q >> 1) * 16 + rl) * MM_LD;            // + 8 MM_LD for tile row 1
    // quad-pair 1's lanes take token rl ^ 4 of a tile: their loads share wavefronts with quad-pair 0's, whose rows are rl,
    // and rows 8 apart share banks (a 144-byte stride); its output columns are swizzled back below
    const int sw4 = (q & 1) << 2;
    const __half* bcol = sx + (tb * 16 + (q & 1) * 8 * TT + (rl ^ sw4)) * MM_LD;   // + 8 MM_LD per token tile
    float acc[2][TT][8];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < TT; ++b)
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[a][b][j] = 0.f;
    const char* we = moe_expert(w, jb.w);
    MmTile<TYPE> wt[NP];
    MmX xt;
    xt.rows(jb, nt);
#pragma unroll
    for (int p = 0; p < NP; ++p) wt[p].fetch(we, w.row_a, w.nrows, row0 + p * MM_PASS, 0, min(MM_KC, w.ncols));
    xt.fetch(x, ldx, 0, min(MM_KC, w.ncols));
    for (int k0 = 0; k0 < w.ncols; k0 += MM_KC) {
        const int kc = min(MM_KC, w.ncols - k0);
#pragma unroll
        for (int p = 0; p < NP; ++p) wt[p].put(k0, kc, sw + p * MM_PASS * MM_LD);
        xt.put(kc, sx);
        __syncthreads();
        if (k0 + MM_KC < w.ncols) {
            const int kn = min(MM_KC, w.ncols - k0 - MM_KC);
#pragma unroll
            for (int p = 0; p < NP; ++p) wt[p].fetch(we, w.row_a, w.nrows, row0 + p * MM_PASS, k0 + MM_KC, kn);
            xt.fetch(x, ldx, k0 + MM_KC, kn);
        }
        if (tb < nt) {
            const int kt0 = split ? tg : 0, kts = split ? WT : 1;
#pragma unroll
            for (int i = 0; i < MM_KC / 16; ++i) {
                const int kt = kt0 + i * kts, kk = kt * 16;
                if (kt >= kc / 16) break;
                // 16 k: a 16-byte load per tile row (8 k) twice; each 4 k of it is one HMMA's operand
                uint4 a[2][2], b[TT][2];
#pragma unroll
                for (int r = 0; r < 2; ++r)
#pragma unroll
                    for (int h = 0; h < 2; ++h) a[r][h] = *reinterpret_cast<const uint4*>(arow + r * 8 * MM_LD + kk + h * 8);
#pragma unroll
                for (int t = 0; t < TT; ++t)
#pragma unroll
                    for (int h = 0; h < 2; ++h) b[t][h] = *reinterpret_cast<const uint4*>(bcol + t * 8 * MM_LD + kk + h * 8);
#pragma unroll
                for (int k4 = 0; k4 < 4; ++k4)
#pragma unroll
                    for (int r = 0; r < 2; ++r)
#pragma unroll
                        for (int t = 0; t < TT; ++t) {
                            const uint4& A = a[r][k4 >> 1];
                            const uint4& B = b[t][k4 >> 1];
                            mma_884(acc[r][t], k4 & 1 ? A.z : A.x, k4 & 1 ? A.w : A.y, k4 & 1 ? B.z : B.x, k4 & 1 ? B.w : B.y);
                        }
            }
        }
        __syncthreads();
    }
    // the output through shared memory over the tiles, token-major [token][row] with rows padded to MM_SLD<BM> floats,
    // so the read back below takes consecutive rows; a split's partial sums are one such [LO][row] block per tg
    float* so = reinterpret_cast<float*>(smem);
    constexpr int SLD = MM_SLD<BM>;
    static_assert(MM_LD % 8 == 0, "16-byte operand loads");
    // a lane's accumulators in an 8 x 8 tile (measured, tools/m884_probe.cu): rows r0 and r0 + 2, columns cb, cb + 1,
    // cb + 4, cb + 5, in the order (r0,cb) (r0,cb+1) (r1,cb) (r1,cb+1) (r0,cb+4) (r0,cb+5) (r1,cb+4) (r1,cb+5)
    const int r0 = (lane & 1) + ((lane >> 4) & 1) * 4, cb = lane & 2;
    float* sob = so + (split ? tg * LO * SLD : tb * 16 * SLD) + rg * 32 + (q >> 1) * 16;   // the warp's rows, its first token
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int t = 0; t < TT; ++t)
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int row = r * 8 + r0 + ((j >> 1) & 1) * 2;
                const int tok = (q & 1) * 8 * TT + t * 8 + ((cb + (j & 1) + ((j >> 2) & 1) * 4) ^ sw4);
                sob[tok * SLD + row] = acc[r][t][j];
            }
    __syncthreads();
    for (int e = threadIdx.x; e < n * BM; e += MM_THREADS) {
        const int j = e / BM, r = e % BM, row = row0 + r;
        if (row >= w.nrows) continue;
        float v = so[j * SLD + r];
        if (split) {
#pragma unroll
            for (int g = 1; g < WT; ++g) v += so[g * LO * SLD + j * SLD + r];
        }
        out[(size_t) jb.out[j] * w.nrows + row] = v;
    }
#endif
}
#else
// Pascal sm_61 (GP102) INT8 __dp4a tiled MoE kernel (47 TOPS/card, zero FP16 arithmetic, zero HMMA, <= 6 KB SMEM).
// Processes a BigJob (up to PF_JOB=64 tokens) x DP_BM=64 weight rows in 32-column steps: on-the-fly f16 -> q8_1
// activation quantization in SMEM, bank-conflict-free weight unpacking for Q4_K / Q5_K / Q6_K / Q5_1 / Q8_0, and
// 4x4 register-tiled __dp4a inner products with 16-lane coalesced output stores.
constexpr int DP_BM = 64, DP_BN = PF_JOB, DP_THREADS = 256, DP_SW_LD = DP_BM + 1, DP_SX_LD = DP_BN + 1;
static_assert(DP_BM == 64 && DP_BN == 64 && DP_THREADS == 256, "k_pf_moe_dp4a tile layout");

__device__ __forceinline__ float dp_wsum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float dp_wmax(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ void dp_scale_min_k4(int j, const uint4& hd, int& d, int& m) {
    if (j < 4) {
        d = (hd.y >> (8 * j)) & 63;
        m = (hd.z >> (8 * j)) & 63;
    } else {
        const int s = 8 * (j - 4);
        d = ((hd.w >> s) & 0xF) | (((hd.y >> (s + 6)) & 3) << 4);
        m = ((hd.w >> (s + 4)) & 0xF) | (((hd.z >> (s + 6)) & 3) << 4);
    }
}
__device__ __forceinline__ uint32_t dp_spread4(uint32_t t) {
    return ((t & 1u) << 4) | ((t & 2u) << 11) | ((t & 4u) << 18) | ((t & 8u) << 25);
}
__device__ __forceinline__ int dp_q6_sign(uint32_t u) {
    const uint32_t nm = (~u & 0x20202020u);
    return (int) ((u & 0x1F1F1F1Fu) | nm | (nm << 1) | (nm << 2));
}

// Warp-cooperative 128 B coalesced global/UVA load of nbytes (even, <= 272) strictly within [src, src + nbytes).
template <bool ALWAYS_AL4>
__device__ __forceinline__ void dp_load_row_coop(uint8_t* dst, const uint8_t* src, int nbytes, int lane) {
    if (ALWAYS_AL4 || (((uintptr_t) src & 3u) == 0u)) {
        const int n32 = nbytes >> 2;
        const uint32_t* s32 = reinterpret_cast<const uint32_t*>(src);
        uint32_t* d32 = reinterpret_cast<uint32_t*>(dst);
        for (int i = lane; i < n32; i += 32) d32[i] = __ldg(s32 + i);
        if (!ALWAYS_AL4 && (nbytes & 2) && lane == 0) {
            *reinterpret_cast<uint16_t*>(dst + nbytes - 2) = __ldg(reinterpret_cast<const uint16_t*>(src + nbytes - 2));
        }
    } else {
        if (lane == 0) *reinterpret_cast<uint16_t*>(dst) = __ldg(reinterpret_cast<const uint16_t*>(src));
        const int rem = nbytes - 2;
        const int n32 = rem >> 2;
        const uint32_t* s32 = reinterpret_cast<const uint32_t*>(src + 2);
        uint16_t* d16 = reinterpret_cast<uint16_t*>(dst + 2);
        for (int i = lane; i < n32; i += 32) {
            const uint32_t v = __ldg(s32 + i);
            d16[i * 2 + 0] = (uint16_t) (v & 0xFFFFu);
            d16[i * 2 + 1] = (uint16_t) (v >> 16);
        }
        if ((rem & 2) && lane == 0) {
            *reinterpret_cast<uint16_t*>(dst + nbytes - 2) = __ldg(reinterpret_cast<const uint16_t*>(src + nbytes - 2));
        }
    }
    __syncwarp();
}

template <bool HAS_MIN>
__device__ __forceinline__ void dp_quant_x(const BigJob& jb, const __half* __restrict__ x, int ldx, int b, int n,
                                           int warp, int lane, int (&sx)[8][DP_SX_LD], float* sxd, float* sxs) {
    for (int t = warp; t < n; t += 8) {
        const float v = __half2float(__ldg(x + (size_t) jb.y[t] * ldx + b * 32 + lane));
        const float amax = dp_wmax(fabsf(v));
        const float scale = amax / 127.0f;
        const float iscale = amax > 0.0f ? 127.0f / amax : 0.0f;
        const uint32_t b0 = (uint8_t) (int8_t) __float2int_rn(v * iscale);
        const uint32_t b1 = __shfl_down_sync(0xffffffffu, b0, 1);
        const uint32_t b2 = __shfl_down_sync(0xffffffffu, b0, 2);
        const uint32_t b3 = __shfl_down_sync(0xffffffffu, b0, 3);
        if ((lane & 3) == 0) sx[lane >> 2][t] = (int) (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
        if constexpr (HAS_MIN) {
            const float xsum = dp_wsum(v);
            if (lane == 0) {
                sxd[t] = scale;
                sxs[t] = xsum;
            }
        } else {
            if (lane == 0) sxd[t] = scale;
        }
    }
}

template <bool HAS_MIN>
__device__ __forceinline__ void dp_mma_32(const int (&sw8)[8][8][DP_SW_LD], int sb, const float* sdw, const float* smw,
                                          const int (&sx)[8][DP_SX_LD], const float* sxd, const float* sxs,
                                          int r_lane, int tc, int n, float (&acc)[4][4]) {
    if (tc >= n) return;
    int dp[4][4] = {};
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        int wv[4], xv[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) wv[i] = sw8[sb][k][r_lane + i * 16];
#pragma unroll
        for (int j = 0; j < 4; ++j) xv[j] = sx[k][tc + j];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) dp[i][j] = __dp4a(wv[i], xv[j], dp[i][j]);
    }
    float dw[4], dx[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) dw[i] = sdw[r_lane + i * 16];
#pragma unroll
    for (int j = 0; j < 4; ++j) dx[j] = sxd[tc + j];
    if constexpr (HAS_MIN) {
        float mw[4], xs[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) mw[i] = smw[r_lane + i * 16];
#pragma unroll
        for (int j = 0; j < 4; ++j) xs[j] = sxs[tc + j];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                acc[i][j] = fmaf(mw[i], xs[j], fmaf((float) dp[i][j], dw[i] * dx[j], acc[i][j]));
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                acc[i][j] = fmaf((float) dp[i][j], dw[i] * dx[j], acc[i][j]);
    }
}

__device__ __forceinline__ void dp_mma_q6k_32(const int (&sw8)[8][8][DP_SW_LD], int sb,
                                              const float (&sdw16)[16][DP_BM], int s_base,
                                              const int (&sx)[8][DP_SX_LD], const float* sxd,
                                              int r_lane, int tc, int n, float (&acc)[4][4]) {
    if (tc >= n) return;
#pragma unroll 1
    for (int half = 0; half < 2; ++half) {
        const int k0 = half * 4;
#pragma unroll
        for (int ig = 0; ig < 2; ++ig) {
            const int i0 = ig * 2;
            int dp[2][4] = {};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                int xv[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) xv[j] = sx[k0 + k][tc + j];
#pragma unroll
                for (int ii = 0; ii < 2; ++ii) {
                    const int wi = sw8[sb][k0 + k][r_lane + (i0 + ii) * 16];
#pragma unroll
                    for (int j = 0; j < 4; ++j) dp[ii][j] = __dp4a(wi, xv[j], dp[ii][j]);
                }
            }
            float dx[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) dx[j] = sxd[tc + j];
#pragma unroll
            for (int ii = 0; ii < 2; ++ii) {
                const float dwi = sdw16[s_base + half][r_lane + (i0 + ii) * 16];
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i0 + ii][j] = fmaf((float) dp[ii][j], dwi * dx[j], acc[i0 + ii][j]);
            }
        }
    }
}

template <int TYPE>
__global__ void __launch_bounds__(DP_THREADS, 4) k_pf_moe_dp4a(MoeW w, const BigJob* __restrict__ jobs,
                                                               const int* __restrict__ njobs, const __half* __restrict__ x,
                                                               int ldx, float* __restrict__ out) {
    constexpr bool HAS_MIN = (TYPE == GGML_TYPE_Q4_K || TYPE == GGML_TYPE_Q5_K || TYPE == GGML_TYPE_Q5_1);
    constexpr int SDW_SLOTS = (TYPE == GGML_TYPE_Q6_K) ? 16 : 8;
    const int ji = blockIdx.y;
    if (ji >= *njobs) return;
    __shared__ BigJob jb;
    // Union aliases per-warp raw PCIe/UVA load buffer (272 B/warp = 2,176 B) with sx[8][65] (2,080 B) across phases.
    __shared__ union {
        int sx[8][DP_SX_LD];
        uint32_t raw32[8][68];
        uint16_t raw16[8][136];
        uint8_t raw8[8][272];
    } sm_x;
    __shared__ int sw8[8][8][DP_SW_LD];
    __shared__ float sdw[SDW_SLOTS][DP_BM];
    __shared__ float smw[HAS_MIN ? 8 : 1][HAS_MIN ? DP_BM : 1];
    __shared__ float sxd[DP_BN];
    __shared__ float sxs[HAS_MIN ? DP_BN : 1];
    load_bigjob(jb, jobs + ji);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n = jb.n, row0 = blockIdx.x * DP_BM, nb = w.ncols >> 5;
    if (tid < DP_BN) {
        sxd[tid] = 0.f;
        if constexpr (HAS_MIN) sxs[tid] = 0.f;
    }
    const char* we = moe_expert(w, jb.w);
    const int r_lane = tid & 15, c_grp = tid >> 4, tc = c_grp * 4;
    float acc[4][4] = {};

    // Unified 256-weight tile loop (8 x 32-weight sub-blocks):
    // Exact 1.0x global/UVA reads via warp-cooperative 128 B contiguous row loads (8 rows/warp, <= 23.9 KB SMEM -> 4 CTAs/SM).
    for (int k8 = 0; k8 < nb; k8 += 8) {
        const int chunk = (nb - k8 < 8) ? (nb - k8) : 8;
        uint8_t* raw8 = sm_x.raw8[warp];
        uint16_t* raw16 = sm_x.raw16[warp];
        uint32_t* raw32 = sm_x.raw32[warp];
#pragma unroll 1
        for (int rr = 0; rr < 8; ++rr) {
            const int r = warp * 8 + rr;
            const int gr = row0 + r;
            if (gr >= w.nrows) {
                if (lane < SDW_SLOTS) sdw[lane][r] = 0.f;
                if constexpr (HAS_MIN) {
                    if (lane < 8) smw[lane][r] = 0.f;
                }
                sw8[lane >> 3][lane & 7][r] = 0;
                sw8[4 + (lane >> 3)][lane & 7][r] = 0;
                continue;
            }
            const uint8_t* row_p = reinterpret_cast<const uint8_t*>(we) + (size_t) gr * w.row_a;
            if constexpr (TYPE == GGML_TYPE_Q4_K) {
                dp_load_row_coop<true>(raw8, row_p + (size_t) (k8 >> 3) * 144, 144, lane);
                if (lane < 8) {
                    const uint4 hd = make_uint4(raw32[0], raw32[1], raw32[2], raw32[3]);
                    const float d = __half2float(__ushort_as_half((uint16_t) (hd.x & 0xFFFFu)));
                    const float dmin = __half2float(__ushort_as_half((uint16_t) (hd.x >> 16)));
                    int sc, m;
                    dp_scale_min_k4(lane, hd, sc, m);
                    sdw[lane][r] = d * (float) sc;
                    smw[lane][r] = -(dmin * (float) m);
                }
                const uint32_t q = raw32[4 + lane];
                const int jg = lane >> 3, kw = lane & 7;
                sw8[jg * 2 + 0][kw][r] = (int) (q & 0x0F0F0F0Fu);
                sw8[jg * 2 + 1][kw][r] = (int) ((q >> 4) & 0x0F0F0F0Fu);
            } else if constexpr (TYPE == GGML_TYPE_Q5_K) {
                dp_load_row_coop<true>(raw8, row_p + (size_t) (k8 >> 3) * 176, 176, lane);
                if (lane < 8) {
                    const uint4 hd = make_uint4(raw32[0], raw32[1], raw32[2], raw32[3]);
                    const float d = __half2float(__ushort_as_half((uint16_t) (hd.x & 0xFFFFu)));
                    const float dmin = __half2float(__ushort_as_half((uint16_t) (hd.x >> 16)));
                    int sc, m;
                    dp_scale_min_k4(lane, hd, sc, m);
                    sdw[lane][r] = d * (float) sc;
                    smw[lane][r] = -(dmin * (float) m);
                }
                const uint32_t q = raw32[12 + lane];
                const int jg = lane >> 3, kw = lane & 7;
                const uint32_t h = raw32[4 + kw];
                const int sb0 = jg * 2, sb1 = sb0 + 1;
                sw8[sb0][kw][r] = (int) ((q & 0x0F0F0F0Fu) | (((h >> sb0) & 0x01010101u) << 4));
                sw8[sb1][kw][r] = (int) (((q >> 4) & 0x0F0F0F0Fu) | (((h >> sb1) & 0x01010101u) << 4));
            } else if constexpr (TYPE == GGML_TYPE_Q6_K) {
                dp_load_row_coop<false>(raw8, row_p + (size_t) (k8 >> 3) * 210, 210, lane);
                if (lane < 16) {
                    const float d = __half2float(__ushort_as_half(raw16[104]));
                    sdw[lane][r] = d * (float) (int8_t) raw8[192 + lane];
                }
                const int h128 = lane >> 4, pair = (lane >> 3) & 1, kw = lane & 7;
                const uint32_t ql = raw32[lane];
                const uint32_t qh = raw32[32 + h128 * 8 + kw];
                const int sh_lo = pair * 2, sh_hi = 4 + pair * 2;
                sw8[h128 * 4 + pair][kw][r] = dp_q6_sign((ql & 0x0F0F0F0Fu) | (((qh >> sh_lo) & 0x03030303u) << 4));
                sw8[h128 * 4 + 2 + pair][kw][r] = dp_q6_sign(((ql >> 4) & 0x0F0F0F0Fu) | (((qh >> sh_hi) & 0x03030303u) << 4));
            } else if constexpr (TYPE == GGML_TYPE_Q8_0) {
                dp_load_row_coop<false>(raw8, row_p + (size_t) k8 * 34, chunk * 34, lane);
                if (lane < chunk) sdw[lane][r] = __half2float(__ushort_as_half(raw16[lane * 17]));
#pragma unroll
                for (int s2 = 0; s2 < 2; ++s2) {
                    const int sb = s2 * 4 + (lane >> 3);
                    if (sb < chunk) {
                        const int kw = lane & 7, h0 = sb * 17 + 1 + kw * 2;
                        sw8[sb][kw][r] = (int) ((uint32_t) raw16[h0] | ((uint32_t) raw16[h0 + 1] << 16));
                    }
                }
            } else if constexpr (TYPE == GGML_TYPE_Q5_1) {
                dp_load_row_coop<true>(raw8, row_p + (size_t) k8 * 24, chunk * 24, lane);
                if (lane < chunk) {
                    const uint32_t dm = raw32[lane * 6];
                    sdw[lane][r] = __half2float(__ushort_as_half((uint16_t) (dm & 0xFFFFu)));
                    smw[lane][r] = __half2float(__ushort_as_half((uint16_t) (dm >> 16)));
                }
                const int sb = lane >> 2, kw4 = lane & 3;
                if (sb < chunk) {
                    const uint32_t qh = raw32[sb * 6 + 1];
                    const uint32_t q = raw32[sb * 6 + 2 + kw4];
                    sw8[sb][kw4][r] = (int) ((q & 0x0F0F0F0Fu) | dp_spread4(qh >> (kw4 * 4)));
                    sw8[sb][4 + kw4][r] = (int) (((q >> 4) & 0x0F0F0F0Fu) | dp_spread4(qh >> (16 + kw4 * 4)));
                }
            }
        }
        __syncthreads();  // all warps done reading sm_x.raw before dp_quant_x writes sm_x.sx

#pragma unroll 1
        for (int sb = 0; sb < chunk; ++sb) {
            dp_quant_x<HAS_MIN>(jb, x, ldx, k8 + sb, n, warp, lane, sm_x.sx, sxd, sxs);
            __syncthreads();
            if constexpr (TYPE == GGML_TYPE_Q6_K) {
                dp_mma_q6k_32(sw8, sb, sdw, sb * 2, sm_x.sx, sxd, r_lane, tc, n, acc);
            } else {
                dp_mma_32<HAS_MIN>(sw8, sb, sdw[sb], smw[HAS_MIN ? sb : 0], sm_x.sx, sxd, sxs, r_lane, tc, n, acc);
            }
            __syncthreads();
        }
    }

    if (tc < n) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int tj = tc + j;
            if (tj < n) {
                const size_t base = (size_t) jb.out[tj] * w.nrows + row0 + r_lane;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    if (row0 + r_lane + i * 16 < w.nrows) out[base + i * 16] = acc[i][j];
            }
        }
    }
}
#endif

// h = silu(gate) * up of the stacked gate|up rows, rounded to f16 (llama.cpp's swiglu output, converted for cuBLAS)
__global__ void k_swiglu_h16(const float* __restrict__ gu, int ffs, int P, __half* __restrict__ h) {
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < (size_t) P * ffs; e += (size_t) gridDim.x * blockDim.x) {
        const size_t r = e / ffs, i = e % ffs;
        const float g = gu[r * 2 * ffs + i];
        h[e] = __float2half(silu_(g) * gu[r * 2 * ffs + ffs + i]);
    }
}

}  // namespace
