// s27_probe: the dense 27B's decode window floor on the build's cards: every Q8_0 matrix one card reads in a window (64
// layers at card 0's slice, then its third of the output head), through q8s_gemv at T rows, each card at once in a
// CUDA graph. No sums between the cards, no recurrent or attention core: the weight reading alone.
//   s27_probe [T...]
#include <cuda_runtime.h>

#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "../src/env.hpp"
#include "../src/kernels.hpp"
#include "../src/s27_attn.cuh"

using namespace fnx;

static uint32_t splitmix32(uint32_t x) {
    x += 0x9e3779b9u;
    x = (x ^ (x >> 16)) * 0x85ebca6bu;
    x = (x ^ (x >> 13)) * 0xc2b2ae35u;
    return x ^ (x >> 16);
}

static uint16_t f32_to_f16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    int exp = (int) ((u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = u & 0x7fffffu;
    if (exp <= 0) return (uint16_t) sign;
    if (exp >= 31) return (uint16_t) (sign | 0x7c00u);
    return (uint16_t) (sign | ((uint32_t) exp << 10) | (mant >> 13));
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// card 0's recurrent slice (card_split.hpp): its key heads, its share of the value heads
constexpr int KH0 = khead_count(NCARD, 0), VH0 = NVH / NCARD;
constexpr int VOCAB = 248320;   // Qwen3.8-27B's output head rows
struct Mat {
    int rows, cols;
};

int main(int argc, char** argv) {
    std::vector<int> Ts;
    for (int i = 1; i < argc; ++i) {
        long long T = 0;
        if (!parse_whole(argv[i], T) || T < 1 || T > MAXT) {
            std::fprintf(stderr, "s27_probe: T is 1..%d, not '%s'\n", MAXT, argv[i]);
            return 2;
        }
        Ts.push_back((int) T);
    }
    if (Ts.empty()) Ts = {1, 2, 3, 4, 6, 8, 12, 16};
    // S27_PF_MB=m: beside each matrix, the first m MB of the next pulled into L2 on a second stream
    double pf_mb = 0;
    if (const char* v = std::getenv("S27_PF_MB")) {
        if (!parse_number(v, pf_mb) || pf_mb < 0) {
            std::fprintf(stderr, "s27_probe: S27_PF_MB: want MB 0 or more, not '%s'\n", v);
            return 2;
        }
    }
    // card 0's slice: KH0 of the key heads, VH0 of the value heads, its FF blocks, S27_NQ query heads and the S27_NKV KV
    // heads they read
    std::vector<Mat> mats;
    for (int l = 0; l < NL; ++l) {
        if ((l + 1) % 4) {
            mats.push_back({2 * KH0 * HS + 2 * VH0 * HS, N});   // qkv + z
            mats.push_back({N, VH0 * HS});                       // ssm_out
        } else {
            mats.push_back({S27_QKV, N});                        // q + gate, k, v
            mats.push_back({N, S27_NQ * HD});                    // wo
        }
        mats.push_back({2 * FFD_CARD_MAX, N});                   // gate | up
        mats.push_back({N, FFD_CARD_MAX});                       // down
    }
    mats.push_back({VOCAB / NCARD + 1, N});                      // output head
    // S27_PROBE_ONE=1: the same bytes as a few 2 GB matrices (the kernel's rate without the boundaries between the
    // window's 257; one matrix would pass q8s's 32-bit offsets)
    long long one = 0;
    if (const char* v = std::getenv("S27_PROBE_ONE")) {
        if (!parse_whole(v, one)) {
            std::fprintf(stderr, "s27_probe: S27_PROBE_ONE: on/off as a whole number, not '%s'\n", v);
            return 2;
        }
    }
    if (one) {
        double b0 = 0;
        for (auto& m : mats) b0 += (double) m.rows * m.cols;
        const int rows = (int) (b0 / N), step = 400000;
        mats.clear();
        for (int r = 0; r < rows; r += step) mats.push_back({std::min(step, rows - r), N});
    }
    const bool use_6p2 = !one && env_int("FNX_LMHEAD_6P2", 1) != 0;
    const int head_rows = mats.back().rows, head_cols = mats.back().cols;
    double bytes = 0;
    for (auto& m : mats) bytes += (double) m.rows * m.cols * (1 + 2.0 / 32);
    std::printf("per card: %zu matrices, %.2f GB (%s LM head)\n", mats.size(), bytes / 1e9, use_6p2 ? "6+2" : "q8s");

    std::vector<float> h_xf((size_t) MAXT * 17408);
    for (size_t i = 0; i < h_xf.size(); ++i) {
        uint32_t h = splitmix32((uint32_t) i + 1337u);
        float s = ((h & 255u) + ((h >> 8) & 255u) + ((h >> 16) & 255u) + ((h >> 24) & 255u)) * (1.0f / 510.0f) - 1.0f;
        h_xf[i] = s * 1.3f;
    }

    std::vector<uint32_t> h_hi, h_lo;
    std::vector<uint16_t> h_d;
    std::vector<float2> h_rmeta;
    if (use_6p2) {
        const size_t rb = (size_t) (head_cols / 32) * 34;
        std::vector<uint8_t> h_head((size_t) head_rows * rb);
        int nth = std::min<unsigned>(12u, std::max<unsigned>(1u, std::thread::hardware_concurrency()));
        std::vector<std::thread> th_fill;
        for (int ti = 0; ti < nth; ++ti) {
            int r0 = (int) ((size_t) head_rows * ti / nth), r1 = (int) ((size_t) head_rows * (ti + 1) / nth);
            th_fill.emplace_back([&, r0, r1] {
                for (int r = r0; r < r1; ++r) {
                    uint8_t* rp = h_head.data() + (size_t) r * rb;
                    for (int b = 0; b < head_cols / 32; ++b) {
                        uint8_t* bp = rp + (size_t) b * 34;
                        float d_val = 0.014f + 0.004f * ((splitmix32((uint32_t) (r * 257 + b)) & 15u) / 15.0f);
                        uint16_t dh = f32_to_f16_bits(d_val);
                        std::memcpy(bp, &dh, 2);
                        for (int j = 0; j < 32; ++j) {
                            uint32_t h = splitmix32((uint32_t) ((r * head_cols + b * 32 + j) ^ 0xa5a5a5a5u));
                            int s = (int) (h & 63u) + (int) ((h >> 8) & 63u) + (int) ((h >> 16) & 63u) + (int) ((h >> 24) & 63u) - 126;
                            bp[2 + j] = (uint8_t) (int8_t) std::max(-127, std::min(127, s));
                        }
                    }
                }
            });
        }
        for (auto& t : th_fill) t.join();
        const int S = head_cols / 512, nb = head_cols / 32;
        h_hi.resize((size_t) head_rows * S * 96);
        h_lo.resize((size_t) head_rows * S * 32);
        h_d.resize((size_t) head_rows * nb);
        h_rmeta.resize(head_rows);
        pack_q80_to_6p2_host(h_head.data(), rb, head_rows, head_cols, nullptr, h_hi.data(), h_lo.data(), h_d.data(), h_rmeta.data());
    }

    constexpr int NC = NCARD;
    std::vector<std::vector<Q8S>> w(NC);
    std::vector<VHead6p2> vh(NC);
    std::vector<VHeadWork> vh_work(NC);
    std::vector<float*> best_v(NC);
    std::vector<int*> best_i(NC);
    std::vector<unsigned long long*> am_keys(NC);
    std::vector<unsigned*> err_dev(NC);
    std::vector<void*> xq(NC);
    std::vector<float*> out(NC);
    std::vector<cudaStream_t> st(NC);
    for (int c = 0; c < NC; ++c) {
        ck(cudaSetDevice(c), "set");
        ck(cudaStreamCreateWithFlags(&st[c], cudaStreamNonBlocking), "stream");
        ck(cudaMalloc(&xq[c], (size_t) MAXT * q8_1_bytes(17408)), "xq");
        {
            int8_t* tq;
            float* td;
            ck(cudaMalloc(&tq, (size_t) MAXT * 17408), "tq");
            ck(cudaMalloc(&td, (size_t) MAXT * 17408 / 32 * 4), "td");
            q8_twin(xq[c], (size_t) MAXT * q8_1_bytes(17408), tq, td);
        }
        float* d_xf = nullptr;
        ck(cudaMalloc(&d_xf, h_xf.size() * 4), "d_xf");
        ck(cudaMemcpy(d_xf, h_xf.data(), h_xf.size() * 4, cudaMemcpyHostToDevice), "xf");
        quantize_q8_1(d_xf, xq[c], 17408, MAXT, st[c]);
        ck(cudaStreamSynchronize(st[c]), "q8 sync");
        ck(cudaFree(d_xf), "d_xf free");

        ck(cudaMalloc(&out[c], (size_t) MAXT * 400000 * 4), "out");
        const size_t n_q8 = use_6p2 ? mats.size() - 1 : mats.size();
        for (size_t i = 0; i < n_q8; ++i) {
            const auto& m = mats[i];
            Q8S s;
            ck(cudaMalloc(&s.q, (size_t) m.rows * m.cols), "q");
            ck(cudaMalloc(&s.d, (size_t) m.rows * m.cols / 32 * 2), "d");
            ck(cudaMemset(s.q, 1, (size_t) m.rows * m.cols), "q0");
            ck(cudaMemset(s.d, 0x2c, (size_t) m.rows * m.cols / 32 * 2), "d0");
            w[c].push_back(s);
        }
        if (use_6p2) {
            vh[c].n_in = head_cols;
            vh[c].n_out = head_rows;
            ck(cudaMalloc(&vh[c].hi, h_hi.size() * 4), "vh.hi");
            ck(cudaMalloc(&vh[c].lo, h_lo.size() * 4), "vh.lo");
            ck(cudaMalloc(&vh[c].d, h_d.size() * 2), "vh.d");
            ck(cudaMalloc(&vh[c].row_meta, h_rmeta.size() * sizeof(float2)), "vh.rmeta");
            ck(cudaMemcpy(vh[c].hi, h_hi.data(), h_hi.size() * 4, cudaMemcpyHostToDevice), "H2D hi");
            ck(cudaMemcpy(vh[c].lo, h_lo.data(), h_lo.size() * 4, cudaMemcpyHostToDevice), "H2D lo");
            ck(cudaMemcpy(vh[c].d, h_d.data(), h_d.size() * 2, cudaMemcpyHostToDevice), "H2D d");
            ck(cudaMemcpy(vh[c].row_meta, h_rmeta.data(), h_rmeta.size() * sizeof(float2), cudaMemcpyHostToDevice), "H2D rmeta");
            VHeadWork& wk = vh_work[c];
            wk.max_rows = head_rows;
            wk.max_in = head_cols;
            ck(cudaMalloc(&wk.xq, (size_t) MAXT * head_cols), "wk.xq");
            ck(cudaMalloc(&wk.xd, (size_t) MAXT * (head_cols / 32) * 4), "wk.xd");
            ck(cudaMalloc(&wk.xd_half, (size_t) MAXT * (head_cols / 32) * 4), "wk.xd_half");
            ck(cudaMalloc(&wk.sx253, (size_t) MAXT * (head_cols / 16) * 4), "wk.sx253");
            ck(cudaMalloc(&wk.act_meta, MAXT * sizeof(float2)), "wk.act_meta");
            ck(cudaMalloc(&wk.lb, (size_t) MAXT * head_rows * 4), "wk.lb");
            wk.ub = out[c];
            ck(cudaMalloc(&wk.cta_max, vhead_cta_max_floats(head_rows, MAXT) * 4), "wk.cta_max");
            ck(cudaMalloc(&wk.lcut, MAXT * 4), "wk.lcut");
            ck(cudaMalloc(&wk.cand_idx, (size_t) MAXT * head_rows * 4), "wk.cand_idx");
            ck(cudaMalloc(&wk.cand_cnt, MAXT * 4), "wk.cand_cnt");
            ck(cudaMalloc(&best_v[c], MAXT * 4), "bv");
            ck(cudaMalloc(&best_i[c], MAXT * 4), "bi");
            ck(cudaMalloc(&am_keys[c], MAXT * sizeof(unsigned long long)), "keys");
            ck(cudaMemset(am_keys[c], 0, MAXT * sizeof(unsigned long long)), "keys0");
            ck(cudaMalloc(&err_dev[c], 4), "err");
            ck(cudaMemset(err_dev[c], 0, 4), "err0");
        }
    }
    ck(cudaSetDevice(0), "set");
    l2_prefetch(w[0][0].q, 32, st[0]);   // its sink allocated now, not inside a capture
    ck(cudaStreamSynchronize(st[0]), "prefetch warm");
    for (int T : Ts) {
        std::vector<cudaGraphExec_t> ex(NC);
        for (int c = 0; c < NC; ++c) {
            ck(cudaSetDevice(c), "set");
            cudaGraph_t g;
            ck(cudaStreamBeginCapture(st[c], cudaStreamCaptureModeThreadLocal), "capture");
            static cudaStream_t st2[NCARD];
            static cudaEvent_t fork[NCARD], join[NCARD];
            if (pf_mb > 0 && !st2[c]) {
                ck(cudaStreamCreateWithFlags(&st2[c], cudaStreamNonBlocking), "st2");
                ck(cudaEventCreateWithFlags(&fork[c], cudaEventDisableTiming), "fork");
                ck(cudaEventCreateWithFlags(&join[c], cudaEventDisableTiming), "join");
            }
            for (size_t i = 0; i < mats.size(); ++i) {
                const bool pf = pf_mb > 0 && i + 1 < mats.size();
                if (pf) {
                    ck(cudaEventRecord(fork[c], st[c]), "fork");
                    ck(cudaStreamWaitEvent(st2[c], fork[c], 0), "fork wait");
                    // clamp in double before the cast: S27_PF_MB is unbounded and a huge MB must not wrap size_t to a small count
                    const bool next_is_vh = use_6p2 && i + 1 == mats.size() - 1;
                    const size_t next_b = next_is_vh ? h_hi.size() * 4 : (size_t) mats[i + 1].rows * mats[i + 1].cols;
                    const size_t nb = (size_t) std::min(pf_mb * 1048576.0, (double) next_b);
                    l2_prefetch(next_is_vh ? (const void*) vh[c].hi : w[c][i + 1].q, nb, st2[c]);
                    ck(cudaEventRecord(join[c], st2[c]), "join");
                }
                if (use_6p2 && i == mats.size() - 1) {
                    if (T <= 5) {
                        vhead_verify_6p2(vh[c], xq[c], T, 0, best_v[c], best_i[c], am_keys[c], err_dev[c], vh_work[c], st[c]);
                    } else {
                        vhead_all_logits_6p2(vh[c], xq[c], out[c], T, st[c]);
                        argmax(out[c], head_rows, T, 0, best_v[c], best_i[c], am_keys[c], err_dev[c], st[c]);
                    }
                } else {
                    q8s_gemv(w[c][i], mats[i].rows, mats[i].cols, xq[c], out[c], mats[i].rows, T, st[c]);
                }
                if (pf) ck(cudaStreamWaitEvent(st[c], join[c], 0), "join wait");
            }
            ck(cudaStreamEndCapture(st[c], &g), "end");
            ck(cudaGraphInstantiate(&ex[c], g, 0), "inst");
            ck(cudaGraphLaunch(ex[c], st[c]), "warm");
            ck(cudaStreamSynchronize(st[c]), "sync");
        }
        constexpr int REPS = 20;
        std::vector<float> ms(NC);
        std::vector<std::thread> th;
        for (int c = 0; c < NC; ++c)
            th.emplace_back([&, c] {
                ck(cudaSetDevice(c), "set");
                cudaEvent_t a, b;
                cudaEventCreate(&a);
                cudaEventCreate(&b);
                cudaEventRecord(a, st[c]);
                for (int r = 0; r < REPS; ++r) cudaGraphLaunch(ex[c], st[c]);
                cudaEventRecord(b, st[c]);
                ck(cudaEventSynchronize(b), "time");
                cudaEventElapsedTime(&ms[c], a, b);
                ms[c] /= REPS;
            });
        for (auto& t : th) t.join();
        std::printf("T=%2d  window", T);
        for (int c = 0; c < NC; ++c) std::printf("%s%6.2f", c ? " / " : " ", ms[c]);
        std::printf(" ms  (%4.0f GB/s on card 0)\n", bytes / ms[0] / 1e6);
    }
}
