// Expert gate|up for Q6_K weights (included by kernels.cu after moe_q4k.cuh).
// A lane takes whole 64-weight (b, half, is) groups of a row (4 groups per 256-weight block_q6_K = 210 B).
// Uses alignment-safe 16-bit __ldg loads (block_q6_K is 2-byte aligned on odd blocks) and borrow-free
// 4-byte SIMD signed subtraction ((u + 0x60606060u) ^ 0x80808080u) so each 4 weights take ONE __dp4a.
namespace {

constexpr int Q6K_THREADS = 256, Q6K_LPR = 8, Q6K_ROWS = Q6K_THREADS / Q6K_LPR;
constexpr int Q6K_GPL = N / 64 / Q6K_LPR;
static_assert(N % (64 * Q6K_LPR) == 0, "Q6_K groups per lane");

// Alignment-safe 32-bit load from a 2-byte aligned block_q6_K address via read-only texture cache.
__device__ __forceinline__ unsigned q6k_u32(const char* p) {
    const uint16_t* p16 = reinterpret_cast<const uint16_t*>(p);
    return (unsigned) __ldg(p16) | ((unsigned) __ldg(p16 + 1) << 16);
}

// Subtract 32 from each byte of 4 packed 6-bit values (each byte in 0..63) -> 4 signed int8 in [-32, +31].
// Max byte is 0x3F; 0x3F + 0x60 = 0x9F <= 0xFF (zero inter-byte carry); XOR 0x80 maps [0x60..0x9F] -> [0xE0..0x1F].
__device__ __forceinline__ int q6k_signed4(unsigned u) {
    return (int) ((u + 0x60606060u) ^ 0x80808080u);
}

template <int TT>
__device__ __forceinline__ void q6k_row(const char* wrow, int lane8, int n, const block_q8_1* y, const int* ys, int y_stride,
                                        float* tmp) {
#pragma unroll 1
    for (int i = 0; i < Q6K_GPL; ++i) {
        const int g = lane8 + i * Q6K_LPR;
        const int b = g >> 2, h = (g >> 1) & 1, is = g & 1;
        const char* blk = wrow + (size_t) b * 210;
        const char* ql0_p = blk + 64 * h + 16 * is;
        const char* ql1_p = blk + 64 * h + 32 + 16 * is;
        const char* qh_p  = blk + 128 + 32 * h + 16 * is;
        const int8_t* sc_p = reinterpret_cast<const int8_t*>(blk + 192 + 8 * h + is);

        const float d = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const uint16_t*>(blk + 208))));
        const float sc0 = (float) __ldg(sc_p + 0);
        const float sc1 = (float) __ldg(sc_p + 2);
        const float sc2 = (float) __ldg(sc_p + 4);
        const float sc3 = (float) __ldg(sc_p + 6);

        int q0[4], q1[4], q2[4], q3[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const unsigned vl0 = q6k_u32(ql0_p + 4 * w);
            const unsigned vl1 = q6k_u32(ql1_p + 4 * w);
            const unsigned vh  = q6k_u32(qh_p  + 4 * w);
            q0[w] = q6k_signed4((vl0 & 0x0F0F0F0Fu) | ((vh << 4) & 0x30303030u));
            q1[w] = q6k_signed4((vl1 & 0x0F0F0F0Fu) | ((vh << 2) & 0x30303030u));
            q2[w] = q6k_signed4(((vl0 >> 4) & 0x0F0F0F0Fu) | (vh & 0x30303030u));
            q3[w] = q6k_signed4(((vl1 >> 4) & 0x0F0F0F0Fu) | ((vh >> 2) & 0x30303030u));
        }

        const int aoff = is * 4;
#pragma unroll 1
        for (int j = 0; j < n; ++j) {
            const block_q8_1* yb = y + (size_t) ys[j] * y_stride + b * 8 + h * 4;
            const int* a0 = reinterpret_cast<const int*>(yb[0].qs) + aoff;
            const int* a1 = reinterpret_cast<const int*>(yb[1].qs) + aoff;
            const int* a2 = reinterpret_cast<const int*>(yb[2].qs) + aoff;
            const int* a3 = reinterpret_cast<const int*>(yb[3].qs) + aoff;
            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                d0 = __dp4a(q0[w], a0[w], d0);
                d1 = __dp4a(q1[w], a1[w], d1);
                d2 = __dp4a(q2[w], a2[w], d2);
                d3 = __dp4a(q3[w], a3[w], d3);
            }
            const float d8_0 = __low2float(yb[0].ds);
            const float d8_1 = __low2float(yb[1].ds);
            const float d8_2 = __low2float(yb[2].ds);
            const float d8_3 = __low2float(yb[3].ds);
            tmp[j] += d * (d8_0 * (sc0 * (float) d0) + d8_1 * (sc1 * (float) d1) +
                           d8_2 * (sc2 * (float) d2) + d8_3 * (sc3 * (float) d3));
        }
    }
}

template <int TT>
__global__ void __launch_bounds__(Q6K_THREADS) k_moe_gu_q6k(MoeW w, const Job* __restrict__ jobs, const int* __restrict__ njobs,
                                                          const block_q8_1* __restrict__ y, float* __restrict__ out, const ShexpDown sd,
                                                          int sd_rows) {
    static_assert(Q6K_THREADS == Q8S_THREADS, "the shared expert's q8s blocks in the Q6_K gate|up launch");
    if ((int) blockIdx.y < sd_rows) {
        shexp_down_blocks<TT>(sd, blockIdx.y * gridDim.x + blockIdx.x, sd_rows * gridDim.x);
        return;
    }
    const int ji = blockIdx.y - sd_rows;
    if (ji >= *njobs) return;
    __shared__ Job jb;
    if (threadIdx.x == 0) jb = jobs[ji];
    __syncthreads();
    const int row = blockIdx.x * Q6K_ROWS + threadIdx.x / Q6K_LPR, lane8 = threadIdx.x % Q6K_LPR;
    const bool live = row < w.nrows;
    float tmp[TT] = {};
    if (live) q6k_row<TT>(moe_expert(w, jb.w) + (size_t) row * w.row_a, lane8, jb.n, y, jb.y, N / 32, tmp);
    job_store<Q6K_LPR, TT>(tmp, jb.n, live, row, jb.out, out, w.nrows);
}

__global__ void __launch_bounds__(Q6K_THREADS) k_moe_gu_q6k_big(MoeW w, const BigJob* __restrict__ jobs, const int* __restrict__ njobs,
                                                              const block_q8_1* __restrict__ y, float* __restrict__ out) {
    const int ji = blockIdx.y;
    if (ji >= *njobs) return;
    __shared__ BigJob jb;
    load_bigjob(jb, jobs + ji);
    const int row = blockIdx.x * Q6K_ROWS + threadIdx.x / Q6K_LPR, lane8 = threadIdx.x % Q6K_LPR;
    const bool live = row < w.nrows;
#pragma unroll 1
    for (int b0 = 0; b0 < jb.n; b0 += MAXT) {
        const int nb = min(MAXT, jb.n - b0);
        float tmp[MAXT] = {};
        if (live) q6k_row<MAXT>(moe_expert(w, jb.w) + (size_t) row * w.row_a, lane8, nb, y, jb.y + b0, N / 32, tmp);
        job_store<Q6K_LPR>(tmp, nb, live, row, jb.out + b0, out, w.nrows);
    }
}

inline void gu_q6k_launch(int T, const MoeW& w, const Job* jobs, const int* njobs, int njobs_max, const block_q8_1* y, float* out,
                          cudaStream_t s, const ShexpDown& sd = ShexpDown{}) {
    if (w.ncols != N || w.row_a != (size_t) N / 256 * 210) throw std::runtime_error("moe gate|up q6_K: row layout");
    const int gx = (w.nrows + Q6K_ROWS - 1) / Q6K_ROWS, sd_blocks = (sd.rows + Q8S_THREADS / 32 - 1) / (Q8S_THREADS / 32);
    const int sd_rows = sd.rows ? (sd_blocks + gx - 1) / gx : 0;
    const dim3 grid(gx, njobs_max + sd_rows);
    switch (T) {
        case 1: k_moe_gu_q6k<1><<<grid, Q6K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 2: k_moe_gu_q6k<2><<<grid, Q6K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 3: k_moe_gu_q6k<3><<<grid, Q6K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 4: k_moe_gu_q6k<4><<<grid, Q6K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 5: k_moe_gu_q6k<5><<<grid, Q6K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 6: k_moe_gu_q6k<6><<<grid, Q6K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 7: k_moe_gu_q6k<7><<<grid, Q6K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 8: k_moe_gu_q6k<8><<<grid, Q6K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 9: k_moe_gu_q6k<9><<<grid, Q6K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 10: k_moe_gu_q6k<10><<<grid, Q6K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 11: k_moe_gu_q6k<11><<<grid, Q6K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 12: k_moe_gu_q6k<12><<<grid, Q6K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 13: k_moe_gu_q6k<13><<<grid, Q6K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 14: k_moe_gu_q6k<14><<<grid, Q6K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 15: k_moe_gu_q6k<15><<<grid, Q6K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        case 16: k_moe_gu_q6k<16><<<grid, Q6K_THREADS, 0, s>>>(w, jobs, njobs, y, out, sd, sd_rows); break;
        default: throw std::runtime_error("moe gate|up q6_K: T out of range");
    }
}

}  // namespace
