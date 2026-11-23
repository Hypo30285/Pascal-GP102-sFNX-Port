#include "../src/gguf.hpp"
#include "../src/kernels.hpp"
#include "../src/model.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

uint16_t f32_to_f16_bits(float x) {
    __half h = __float2half_rn(x);
    uint16_t b;
    std::memcpy(&b, &h, sizeof(b));
    return b;
}

float f16_bits_to_f32(uint16_t h) {
    __half hv;
    std::memcpy(&hv, &h, sizeof(hv));
    return __half2float(hv);
}

void extract_q8s_from_q80(const uint8_t* src, size_t row_bytes, int rows, int cols,
                          std::vector<int8_t>& q_out, std::vector<uint16_t>& d_out) {
    const int nb = cols / 32;
    q_out.resize(static_cast<size_t>(rows) * cols);
    d_out.resize(static_cast<size_t>(rows) * nb);
    for (int r = 0; r < rows; ++r) {
        const uint8_t* rp = src + static_cast<size_t>(r) * row_bytes;
        for (int b = 0; b < nb; ++b) {
            const uint8_t* blk = rp + static_cast<size_t>(b) * 34;
            uint16_t db;
            std::memcpy(&db, blk, 2);
            d_out[static_cast<size_t>(r) * nb + b] = db;
            std::memcpy(&q_out[static_cast<size_t>(r) * cols + b * 32], blk + 2, 32);
        }
    }
}

}  // namespace

int main() {
    const std::string model_path = "/mnt/storage_8tb_linux/models/gguf/Qwen3.8-27B-Q8_0/Qwen3.8-27B-Q8_0.gguf";
    std::printf("vhead_6p2_test: opening %s...\n", model_path.c_str());
    fnx::Gguf g(model_path);
    const fnx::Tensor& tw = g.t("output.weight");
    const int cols = static_cast<int>(tw.ne[0]);
    const int total_vocab = static_cast<int>(tw.ne[1]);
    const int rows = total_vocab / 2;  // 124,160 (Card 0 slice)
    const size_t row_bytes = tw.row_bytes();
    if (tw.type != fnx::T_Q8_0 || cols != 5120 || rows != 124160) {
        std::fprintf(stderr, "Unexpected output.weight shape (%d x %d) type=%u\n", cols, total_vocab, tw.type);
        return 1;
    }
    std::printf("vhead_6p2_test: output.weight slice rows=%d cols=%d row_bytes=%zu (%.2f MiB)\n",
                rows, cols, row_bytes, static_cast<double>(rows) * row_bytes / (1024.0 * 1024.0));

    ck(cudaSetDevice(0), "cudaSetDevice(0)");
    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st), "cudaStreamCreate");

    // 1. Pack Q8S ref + identity VHead6p2 + permuted VHead6p2
    std::vector<int8_t> h_ref_q;
    std::vector<uint16_t> h_ref_d;
    extract_q8s_from_q80(tw.data, row_bytes, rows, cols, h_ref_q, h_ref_d);

    const int S = cols / 512;
    const int nb = cols / 32;
    std::vector<uint32_t> h_hi(static_cast<size_t>(rows) * S * 96);
    std::vector<uint32_t> h_lo(static_cast<size_t>(rows) * S * 32);
    std::vector<uint16_t> h_d(static_cast<size_t>(rows) * nb);
    std::vector<float2> h_row_meta(rows);
    fnx::pack_q80_to_6p2_host(tw.data, row_bytes, rows, cols, nullptr,
                               h_hi.data(), h_lo.data(), h_d.data(), h_row_meta.data());

    std::vector<int32_t> h_perm(rows), h_inv(rows);
    std::iota(h_perm.begin(), h_perm.end(), 0);
    {
        std::mt19937 rng(42);
        std::shuffle(h_perm.begin(), h_perm.end(), rng);
        for (int r = 0; r < rows; ++r) h_inv[h_perm[r]] = r;
    }
    std::vector<uint32_t> h_phi(static_cast<size_t>(rows) * S * 96);
    std::vector<uint32_t> h_plo(static_cast<size_t>(rows) * S * 32);
    std::vector<uint16_t> h_pd(static_cast<size_t>(rows) * nb);
    std::vector<float2> h_prow_meta(rows);
    fnx::pack_q80_to_6p2_host(tw.data, row_bytes, rows, cols, h_perm.data(),
                               h_phi.data(), h_plo.data(), h_pd.data(), h_prow_meta.data());

    fnx::Q8S d_ref{};
    ck(cudaMalloc(&d_ref.q, h_ref_q.size()), "d_ref.q");
    ck(cudaMalloc(&d_ref.d, h_ref_d.size() * 2), "d_ref.d");
    ck(cudaMemcpy(d_ref.q, h_ref_q.data(), h_ref_q.size(), cudaMemcpyHostToDevice), "H2D ref.q");
    ck(cudaMemcpy(d_ref.d, h_ref_d.data(), h_ref_d.size() * 2, cudaMemcpyHostToDevice), "H2D ref.d");

    fnx::VHead6p2 d_vh{}, d_pvh{};
    d_vh.n_in = cols;
    d_vh.n_out = rows;
    ck(cudaMalloc(&d_vh.hi, h_hi.size() * 4), "d_vh.hi");
    ck(cudaMalloc(&d_vh.lo, h_lo.size() * 4), "d_vh.lo");
    ck(cudaMalloc(&d_vh.d, h_d.size() * 2), "d_vh.d");
    ck(cudaMalloc(&d_vh.row_meta, rows * sizeof(float2)), "d_vh.row_meta");
    ck(cudaMemcpy(d_vh.hi, h_hi.data(), h_hi.size() * 4, cudaMemcpyHostToDevice), "H2D vh.hi");
    ck(cudaMemcpy(d_vh.lo, h_lo.data(), h_lo.size() * 4, cudaMemcpyHostToDevice), "H2D vh.lo");
    ck(cudaMemcpy(d_vh.d, h_d.data(), h_d.size() * 2, cudaMemcpyHostToDevice), "H2D vh.d");
    ck(cudaMemcpy(d_vh.row_meta, h_row_meta.data(), rows * sizeof(float2), cudaMemcpyHostToDevice), "H2D vh.row_meta");

    d_pvh.n_in = cols;
    d_pvh.n_out = rows;
    ck(cudaMalloc(&d_pvh.hi, h_phi.size() * 4), "d_pvh.hi");
    ck(cudaMalloc(&d_pvh.lo, h_plo.size() * 4), "d_pvh.lo");
    ck(cudaMalloc(&d_pvh.d, h_pd.size() * 2), "d_pvh.d");
    ck(cudaMalloc(&d_pvh.row_meta, rows * sizeof(float2)), "d_pvh.row_meta");
    ck(cudaMalloc(&d_pvh.perm_to_vocab, rows * 4), "d_pvh.perm");
    ck(cudaMalloc(&d_pvh.vocab_to_local, rows * 4), "d_pvh.inv");
    ck(cudaMemcpy(d_pvh.hi, h_phi.data(), h_phi.size() * 4, cudaMemcpyHostToDevice), "H2D pvh.hi");
    ck(cudaMemcpy(d_pvh.lo, h_plo.data(), h_plo.size() * 4, cudaMemcpyHostToDevice), "H2D pvh.lo");
    ck(cudaMemcpy(d_pvh.d, h_pd.data(), h_pd.size() * 2, cudaMemcpyHostToDevice), "H2D pvh.d");
    ck(cudaMemcpy(d_pvh.row_meta, h_prow_meta.data(), rows * sizeof(float2), cudaMemcpyHostToDevice), "H2D pvh.row_meta");
    ck(cudaMemcpy(d_pvh.perm_to_vocab, h_perm.data(), rows * 4, cudaMemcpyHostToDevice), "H2D pvh.perm");
    ck(cudaMemcpy(d_pvh.vocab_to_local, h_inv.data(), rows * 4, cudaMemcpyHostToDevice), "H2D pvh.inv");

    // 2. Allocate VHeadWork + activations + outputs
    const int MAXT = fnx::MAXT;
    fnx::VHeadWork work{};
    work.max_rows = rows;
    work.max_in = cols;
    ck(cudaMalloc(&work.xq, static_cast<size_t>(MAXT) * cols), "work.xq");
    ck(cudaMalloc(&work.xd, static_cast<size_t>(MAXT) * nb * 4), "work.xd");
    ck(cudaMalloc(&work.xd_half, static_cast<size_t>(MAXT) * nb * 4), "work.xd_half");
    ck(cudaMalloc(&work.sx253, static_cast<size_t>(MAXT) * (cols / 16) * 4), "work.sx253");
    ck(cudaMalloc(&work.act_meta, static_cast<size_t>(MAXT) * sizeof(float2)), "work.act_meta");
    ck(cudaMalloc(&work.lb, static_cast<size_t>(MAXT) * rows * 4), "work.lb");
    ck(cudaMalloc(&work.ub, static_cast<size_t>(MAXT) * rows * 4), "work.ub");
    ck(cudaMalloc(&work.cta_max, fnx::vhead_cta_max_floats(rows, MAXT) * 4), "work.cta_max");
    ck(cudaMalloc(&work.lcut, static_cast<size_t>(MAXT) * 4), "work.lcut");
    ck(cudaMalloc(&work.cand_idx, static_cast<size_t>(MAXT) * rows * 4), "work.cand_idx");
    ck(cudaMalloc(&work.cand_cnt, static_cast<size_t>(MAXT) * 4), "work.cand_cnt");

    float* d_xf = nullptr;
    void* d_y_q81 = nullptr;
    void* d_y_twin = nullptr;
    int8_t* d_tq = nullptr;
    float* d_td = nullptr;
    const size_t y_bytes = static_cast<size_t>(MAXT) * fnx::q8_1_bytes(cols);
    ck(cudaMalloc(&d_xf, static_cast<size_t>(MAXT) * cols * 4), "d_xf");
    ck(cudaMalloc(&d_y_q81, y_bytes), "d_y_q81");
    ck(cudaMalloc(&d_y_twin, y_bytes), "d_y_twin");
    ck(cudaMalloc(&d_tq, static_cast<size_t>(MAXT) * cols), "d_tq");
    ck(cudaMalloc(&d_td, static_cast<size_t>(MAXT) * nb * 4), "d_td");
    fnx::q8_twin(d_y_twin, y_bytes, d_tq, d_td);

    float *d_logits_ref = nullptr, *d_logits_6p2 = nullptr;
    ck(cudaMalloc(&d_logits_ref, static_cast<size_t>(MAXT) * rows * 4), "d_logits_ref");
    ck(cudaMalloc(&d_logits_6p2, static_cast<size_t>(MAXT) * rows * 4), "d_logits_6p2");

    float *d_bv_ref = nullptr, *d_bv_6p2 = nullptr;
    int *d_bi_ref = nullptr, *d_bi_6p2 = nullptr;
    unsigned long long *d_keys_ref = nullptr, *d_keys_6p2 = nullptr;
    unsigned *d_err_ref = nullptr, *d_err_6p2 = nullptr;
    ck(cudaMalloc(&d_bv_ref, MAXT * 4), "d_bv_ref");
    ck(cudaMalloc(&d_bv_6p2, MAXT * 4), "d_bv_6p2");
    ck(cudaMalloc(&d_bi_ref, MAXT * 4), "d_bi_ref");
    ck(cudaMalloc(&d_bi_6p2, MAXT * 4), "d_bi_6p2");
    ck(cudaMalloc(&d_keys_ref, MAXT * 8), "d_keys_ref");
    ck(cudaMalloc(&d_keys_6p2, MAXT * 8), "d_keys_6p2");
    ck(cudaMalloc(&d_err_ref, 4), "d_err_ref");
    ck(cudaMalloc(&d_err_6p2, 4), "d_err_6p2");
    ck(cudaMemset(d_keys_ref, 0, MAXT * 8), "zero keys_ref");
    ck(cudaMemset(d_keys_6p2, 0, MAXT * 8), "zero keys_6p2");

    constexpr int PEN_STRIDE = 128;
    int *d_pen_tok = nullptr, *d_pen_cnt = nullptr, *d_pen_n = nullptr;
    float *d_pen_par = nullptr, *d_topv_ref = nullptr, *d_topv_6p2 = nullptr;
    int *d_topi_ref = nullptr, *d_topi_6p2 = nullptr;
    ck(cudaMalloc(&d_pen_tok, MAXT * PEN_STRIDE * 4), "d_pen_tok");
    ck(cudaMalloc(&d_pen_cnt, MAXT * PEN_STRIDE * 4), "d_pen_cnt");
    ck(cudaMalloc(&d_pen_n, MAXT * 4), "d_pen_n");
    ck(cudaMalloc(&d_pen_par, MAXT * 3 * 4), "d_pen_par");
    ck(cudaMalloc(&d_topv_ref, MAXT * fnx::TOPK_N * 4), "d_topv_ref");
    ck(cudaMalloc(&d_topv_6p2, MAXT * fnx::TOPK_N * 4), "d_topv_6p2");
    ck(cudaMalloc(&d_topi_ref, MAXT * fnx::TOPK_N * 4), "d_topi_ref");
    ck(cudaMalloc(&d_topi_6p2, MAXT * fnx::TOPK_N * 4), "d_topi_6p2");

    std::mt19937 rng(12345);
    std::normal_distribution<float> norm(0.0f, 1.0f);
    std::vector<float> h_xf(static_cast<size_t>(MAXT) * cols);

    auto make_xf = [&](int T, int mode) {
        for (int t = 0; t < T; ++t) {
            for (int c = 0; c < cols; ++c) {
                float v = norm(rng) * 0.4f;
                if (mode == 1 && (c % 257 == (t % 17))) v *= 25.0f;  // RMSNorm outlier channels
                h_xf[static_cast<size_t>(t) * cols + c] = v;
            }
        }
        ck(cudaMemcpy(d_xf, h_xf.data(), static_cast<size_t>(T) * cols * 4, cudaMemcpyHostToDevice), "H2D xf");
    };

    // Prepare host buffers for checking
    std::vector<float> h_ref(static_cast<size_t>(MAXT) * rows);
    std::vector<float> h_6p2(static_cast<size_t>(MAXT) * rows);
    std::vector<float> h_lb(static_cast<size_t>(MAXT) * rows);
    std::vector<float> h_ub(static_cast<size_t>(MAXT) * rows);

    // 3. Test 0-ULP + Soundness + Verify + TopK across T = 1, 4, 5, 8, 16 on both SPLIT=false and SPLIT=true, identity & permuted
    const int Ts[] = {1, 4, 5, 8, 16};
    for (int mode = 0; mode < 2; ++mode) {  // mode 0 = SPLIT false (raw q8_1), mode 1 = SPLIT true (q8_twin)
        for (int perm_mode = 0; perm_mode < 2; ++perm_mode) {
            const fnx::VHead6p2& vh = perm_mode ? d_pvh : d_vh;
            void* y_buf = mode ? d_y_twin : d_y_q81;
            const int v0 = perm_mode ? 124160 : 0;

            for (int T : Ts) {
                make_xf(T, (T + perm_mode) % 2);
                fnx::quantize_q8_1(d_xf, y_buf, cols, T, st);

                fnx::q8s_gemv(d_ref, rows, cols, y_buf, d_logits_ref, rows, T, st);
                fnx::vhead_all_logits_6p2(vh, y_buf, d_logits_6p2, T, st);
                ck(cudaStreamSynchronize(st), "sync all_logits");

                ck(cudaMemcpy(h_ref.data(), d_logits_ref, static_cast<size_t>(T) * rows * 4, cudaMemcpyDeviceToHost), "D2H ref");
                ck(cudaMemcpy(h_6p2.data(), d_logits_6p2, static_cast<size_t>(T) * rows * 4, cudaMemcpyDeviceToHost), "D2H 6p2");
                for (int t = 0; t < T; ++t) {
                    for (int r = 0; r < rows; ++r) {
                        uint32_t br, b6;
                        std::memcpy(&br, &h_ref[static_cast<size_t>(t) * rows + r], 4);
                        std::memcpy(&b6, &h_6p2[static_cast<size_t>(t) * rows + r], 4);
                        if (br != b6) {
                            std::fprintf(stderr, "FAIL 0-ULP split=%d perm=%d T=%d t=%d r=%d ref=0x%08x (%g) got=0x%08x (%g)\n",
                                         mode, perm_mode, T, t, r, br, h_ref[static_cast<size_t>(t) * rows + r],
                                         b6, h_6p2[static_cast<size_t>(t) * rows + r]);
                            return 1;
                        }
                    }
                }

                // Exact verify
                ck(cudaMemset(d_err_ref, 0, 4), "zero err_ref");
                ck(cudaMemset(d_err_6p2, 0, 4), "zero err_6p2");
                fnx::argmax(d_logits_ref, rows, T, v0, d_bv_ref, d_bi_ref, d_keys_ref, d_err_ref, st);
                fnx::vhead_verify_6p2(vh, y_buf, T, v0, d_bv_6p2, d_bi_6p2, d_keys_6p2, d_err_6p2, work, st);
                ck(cudaStreamSynchronize(st), "sync verify");

                // Soundness check on work.ub from verify (lb <= exact <= ub checked in unpenalized top_tokens pass)
                ck(cudaMemcpy(h_ub.data(), work.ub, static_cast<size_t>(T) * rows * 4, cudaMemcpyDeviceToHost), "D2H ub");
                std::vector<uint32_t> h_cnt(T);
                ck(cudaMemcpy(h_cnt.data(), work.cand_cnt, static_cast<size_t>(T) * 4, cudaMemcpyDeviceToHost), "D2H cnt");

                for (int t = 0; t < T; ++t) {
                    for (int r = 0; r < rows; ++r) {
                        int vr = perm_mode ? h_perm[r] : r;
                        float ex = h_ref[static_cast<size_t>(t) * rows + vr];
                        float ub = h_ub[static_cast<size_t>(t) * rows + r];
                        if (!(ex <= ub)) {
                            std::fprintf(stderr, "FAIL verify ub miss split=%d perm=%d T=%d t=%d r=%d ex=%.6f ub=%.6f\n",
                                         mode, perm_mode, T, t, r, ex, ub);
                            return 1;
                        }
                    }
                }

                // Run unpenalized top_tokens first (pen_n=0) to verify full [lb, ub] soundness
                ck(cudaMemset(d_pen_n, 0, T * 4), "zero pn");
                {
                    std::vector<float> unit_par(T * 3, 0.0f);
                    for (int t = 0; t < T; ++t) unit_par[t * 3] = 1.0f;
                    ck(cudaMemcpy(d_pen_par, unit_par.data(), T * 3 * 4, cudaMemcpyHostToDevice), "unit ppar");
                }
                ck(cudaMemset(d_err_6p2, 0, 4), "zero err_6p2");
                fnx::vhead_top_tokens_6p2(vh, y_buf, T, v0, d_pen_tok, d_pen_cnt, d_pen_n, PEN_STRIDE, d_pen_par,
                                          d_topv_6p2, d_topi_6p2, d_err_6p2, work, st);
                ck(cudaStreamSynchronize(st), "sync unpen topk");
                ck(cudaMemcpy(h_lb.data(), work.lb, static_cast<size_t>(T) * rows * 4, cudaMemcpyDeviceToHost), "D2H lb");
                for (int t = 0; t < T; ++t) {
                    for (int r = 0; r < rows; ++r) {
                        int vr = perm_mode ? h_perm[r] : r;
                        float ex = h_ref[static_cast<size_t>(t) * rows + vr];
                        float lb = h_lb[static_cast<size_t>(t) * rows + r];
                        if (!(lb <= ex)) {
                            std::fprintf(stderr, "FAIL interval lb miss split=%d perm=%d T=%d t=%d r=%d lb=%.6f ex=%.6f\n",
                                         mode, perm_mode, T, t, r, lb, ex);
                            return 1;
                        }
                    }
                }

                std::vector<float> bv_r(T), bv_6(T);
                std::vector<int> bi_r(T), bi_6(T);
                unsigned er_r = 0, er_6 = 0;
                ck(cudaMemcpy(bv_r.data(), d_bv_ref, T * 4, cudaMemcpyDeviceToHost), "D2H bv_r");
                ck(cudaMemcpy(bv_6.data(), d_bv_6p2, T * 4, cudaMemcpyDeviceToHost), "D2H bv_6");
                ck(cudaMemcpy(bi_r.data(), d_bi_ref, T * 4, cudaMemcpyDeviceToHost), "D2H bi_r");
                ck(cudaMemcpy(bi_6.data(), d_bi_6p2, T * 4, cudaMemcpyDeviceToHost), "D2H bi_6");
                ck(cudaMemcpy(&er_r, d_err_ref, 4, cudaMemcpyDeviceToHost), "D2H er_r");
                ck(cudaMemcpy(&er_6, d_err_6p2, 4, cudaMemcpyDeviceToHost), "D2H er_6");
                for (int t = 0; t < T; ++t) {
                    uint32_t ur, u6;
                    std::memcpy(&ur, &bv_r[t], 4);
                    std::memcpy(&u6, &bv_6[t], 4);
                    if (ur != u6 || bi_r[t] != bi_6[t] || er_r != er_6) {
                        std::fprintf(stderr, "FAIL verify mismatch split=%d perm=%d T=%d t=%d: ref=(%g,%d,e=%u) got=(%g,%d,e=%u)\n",
                                     mode, perm_mode, T, t, bv_r[t], bi_r[t], er_r, bv_6[t], bi_6[t], er_6);
                        return 1;
                    }
                }

                // Exact top_tokens (with heavy global-ID penalties covering both inside [v0, v0+rows) and outside!)
                std::vector<int> h_ptok(T * PEN_STRIDE, 0), h_pcnt(T * PEN_STRIDE, 0), h_pn(T);
                std::vector<float> h_ppar(T * 3);
                for (int t = 0; t < T; ++t) {
                    h_ppar[t * 3 + 0] = 1.15f + 0.05f * (t % 3);
                    h_ppar[t * 3 + 1] = 0.35f;
                    h_ppar[t * 3 + 2] = 0.25f;
                    // Guarantee we penalize the true argmax AND top rows to stress boundary re-ordering!
                    std::vector<std::pair<float, int>> tmp;
                    tmp.reserve(rows);
                    for (int vr = 0; vr < rows; ++vr) tmp.emplace_back(h_ref[static_cast<size_t>(t) * rows + vr], v0 + vr);
                    std::partial_sort(tmp.begin(), tmp.begin() + 32, tmp.end(),
                                      [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
                    int pn = 48;
                    h_pn[t] = pn;
                    for (int k = 0; k < 32; ++k) {
                        h_ptok[t * PEN_STRIDE + k] = tmp[k].second;
                        h_pcnt[t * PEN_STRIDE + k] = (k % 5) + 1;
                    }
                    // Include out-of-slice tokens and cnt==0 cases
                    for (int k = 32; k < pn; ++k) {
                        h_ptok[t * PEN_STRIDE + k] = (k < 40) ? (v0 ? 10 : 124200) : (v0 + k * 7);
                        h_pcnt[t * PEN_STRIDE + k] = (k % 2 == 0) ? 0 : 3;
                    }
                }
                ck(cudaMemcpy(d_pen_tok, h_ptok.data(), T * PEN_STRIDE * 4, cudaMemcpyHostToDevice), "H2D ptok");
                ck(cudaMemcpy(d_pen_cnt, h_pcnt.data(), T * PEN_STRIDE * 4, cudaMemcpyHostToDevice), "H2D pcnt");
                ck(cudaMemcpy(d_pen_n, h_pn.data(), T * 4, cudaMemcpyHostToDevice), "H2D pn");
                ck(cudaMemcpy(d_pen_par, h_ppar.data(), T * 3 * 4, cudaMemcpyHostToDevice), "H2D ppar");
                ck(cudaMemset(d_err_ref, 0, 4), "zero err_ref");
                ck(cudaMemset(d_err_6p2, 0, 4), "zero err_6p2");

                fnx::top_tokens(d_logits_ref, rows, T, v0, d_pen_tok, d_pen_cnt, d_pen_n, PEN_STRIDE, d_pen_par,
                                d_topv_ref, d_topi_ref, d_err_ref, st);
                fnx::vhead_top_tokens_6p2(vh, y_buf, T, v0, d_pen_tok, d_pen_cnt, d_pen_n, PEN_STRIDE, d_pen_par,
                                          d_topv_6p2, d_topi_6p2, d_err_6p2, work, st);
                ck(cudaStreamSynchronize(st), "sync topk");

                std::vector<float> tv_r(T * fnx::TOPK_N), tv_6(T * fnx::TOPK_N);
                std::vector<int> ti_r(T * fnx::TOPK_N), ti_6(T * fnx::TOPK_N);
                ck(cudaMemcpy(tv_r.data(), d_topv_ref, T * fnx::TOPK_N * 4, cudaMemcpyDeviceToHost), "D2H tv_r");
                ck(cudaMemcpy(tv_6.data(), d_topv_6p2, T * fnx::TOPK_N * 4, cudaMemcpyDeviceToHost), "D2H tv_6");
                ck(cudaMemcpy(ti_r.data(), d_topi_ref, T * fnx::TOPK_N * 4, cudaMemcpyDeviceToHost), "D2H ti_r");
                ck(cudaMemcpy(ti_6.data(), d_topi_6p2, T * fnx::TOPK_N * 4, cudaMemcpyDeviceToHost), "D2H ti_6");
                ck(cudaMemcpy(&er_r, d_err_ref, 4, cudaMemcpyDeviceToHost), "D2H er_r");
                ck(cudaMemcpy(&er_6, d_err_6p2, 4, cudaMemcpyDeviceToHost), "D2H er_6");
                for (int i = 0; i < T * fnx::TOPK_N; ++i) {
                    uint32_t ur, u6;
                    std::memcpy(&ur, &tv_r[i], 4);
                    std::memcpy(&u6, &tv_6[i], 4);
                    if (ur != u6 || ti_r[i] != ti_6[i] || er_r != er_6) {
                        std::fprintf(stderr, "FAIL topk mismatch split=%d perm=%d T=%d slot=%d: ref=(%g,%d) got=(%g,%d) err=(%u,%u)\n",
                                     mode, perm_mode, T, i, tv_r[i], ti_r[i], tv_6[i], ti_6[i], er_r, er_6);
                        return 1;
                    }
                }
                if (perm_mode == 0 && mode == 1) {
                    double mean_c = 0;
                    uint32_t max_c = 0;
                    for (uint32_t c : h_cnt) { mean_c += c; max_c = std::max(max_c, c); }
                    std::printf("  [OK] T=%2d (split=%d): 0-ULP + 0-miss verify & top64 OK | surv mean=%.0f (%.2f%%), max=%u\n",
                                T, mode, mean_c / T, 100.0 * (mean_c / T) / rows, max_c);
                }
            }
        }
    }

    // 4. Wide-tie equal-row test (all zeros -> 124,160 rows tie at 0.0f under random permutation)
    {
        ck(cudaMemset(d_xf, 0, cols * 4), "zero xf");
        fnx::quantize_q8_1(d_xf, d_y_twin, cols, 1, st);
        ck(cudaMemset(d_pen_n, 0, 4), "zero pn");
        float dummy_par[3] = {1.0f, 0.0f, 0.0f};
        ck(cudaMemcpy(d_pen_par, dummy_par, 12, cudaMemcpyHostToDevice), "dummy par");
        ck(cudaMemset(d_err_6p2, 0, 4), "zero err");
        fnx::vhead_verify_6p2(d_pvh, d_y_twin, 1, 500, d_bv_6p2, d_bi_6p2, d_keys_6p2, d_err_6p2, work, st);
        fnx::vhead_top_tokens_6p2(d_pvh, d_y_twin, 1, 500, d_pen_tok, d_pen_cnt, d_pen_n, PEN_STRIDE, d_pen_par,
                                  d_topv_6p2, d_topi_6p2, d_err_6p2, work, st);
        ck(cudaStreamSynchronize(st), "sync tie");
        int bi = -1;
        float bv = -1;
        ck(cudaMemcpy(&bi, d_bi_6p2, 4, cudaMemcpyDeviceToHost), "D2H tie bi");
        ck(cudaMemcpy(&bv, d_bv_6p2, 4, cudaMemcpyDeviceToHost), "D2H tie bv");
        std::vector<int> ti(fnx::TOPK_N);
        std::vector<float> tv(fnx::TOPK_N);
        ck(cudaMemcpy(ti.data(), d_topi_6p2, fnx::TOPK_N * 4, cudaMemcpyDeviceToHost), "D2H tie topi");
        ck(cudaMemcpy(tv.data(), d_topv_6p2, fnx::TOPK_N * 4, cudaMemcpyDeviceToHost), "D2H tie topv");
        if (bv != 0.0f || bi != 500) {
            std::fprintf(stderr, "FAIL wide-tie verify: bv=%g bi=%d (expected 0, 500)\n", bv, bi);
            return 1;
        }
        for (int k = 0; k < fnx::TOPK_N; ++k) {
            if (tv[k] != 0.0f || ti[k] != 500 + k) {
                std::fprintf(stderr, "FAIL wide-tie top64 k=%d: got (%g, %d) exp (0, %d)\n", k, tv[k], ti[k], 500 + k);
                return 1;
            }
        }
        std::printf("  [OK] Wide-tie (all 124,160 zero logits under shuffle): smallest IDs 500..563 exact.\n");
    }

    // 5. NaN / Inf error bit 16 check (whole 32-elem Q8 block so wmax/fmaxf in k_quantize preserves NaN scale)
    {
        make_xf(1, 0);
        std::fill(h_xf.begin(), h_xf.begin() + 32, NAN);
        ck(cudaMemcpy(d_xf, h_xf.data(), cols * 4, cudaMemcpyHostToDevice), "H2D NaN");
        fnx::quantize_q8_1(d_xf, d_y_twin, cols, 1, st);
        ck(cudaMemset(d_err_ref, 0, 4), "zero err_ref");
        ck(cudaMemset(d_err_6p2, 0, 4), "zero err_6p2");
        fnx::q8s_gemv(d_ref, rows, cols, d_y_twin, d_logits_ref, rows, 1, st);
        fnx::argmax(d_logits_ref, rows, 1, 0, d_bv_ref, d_bi_ref, d_keys_ref, d_err_ref, st);
        fnx::vhead_verify_6p2(d_vh, d_y_twin, 1, 0, d_bv_6p2, d_bi_6p2, d_keys_6p2, d_err_6p2, work, st);
        ck(cudaStreamSynchronize(st), "sync nan");
        unsigned er = 0, e1 = 0;
        ck(cudaMemcpy(&er, d_err_ref, 4, cudaMemcpyDeviceToHost), "D2H nan ref err");
        ck(cudaMemcpy(&e1, d_err_6p2, 4, cudaMemcpyDeviceToHost), "D2H nan verify err");
        if ((e1 & 16u) == 0u || er != e1) {
            std::fprintf(stderr, "FAIL NaN verify err mismatch ref=%u got=%u\n", er, e1);
            return 1;
        }
        ck(cudaMemset(d_err_ref, 0, 4), "zero err_ref");
        ck(cudaMemset(d_err_6p2, 0, 4), "zero err");
        ck(cudaMemset(d_pen_n, 0, 4), "zero pn");
        fnx::top_tokens(d_logits_ref, rows, 1, 0, d_pen_tok, d_pen_cnt, d_pen_n, PEN_STRIDE, d_pen_par,
                        d_topv_ref, d_topi_ref, d_err_ref, st);
        fnx::vhead_top_tokens_6p2(d_vh, d_y_twin, 1, 0, d_pen_tok, d_pen_cnt, d_pen_n, PEN_STRIDE, d_pen_par,
                                  d_topv_6p2, d_topi_6p2, d_err_6p2, work, st);
        ck(cudaStreamSynchronize(st), "sync nan top");
        unsigned et = 0, e2 = 0;
        ck(cudaMemcpy(&et, d_err_ref, 4, cudaMemcpyDeviceToHost), "D2H nan ref top err");
        ck(cudaMemcpy(&e2, d_err_6p2, 4, cudaMemcpyDeviceToHost), "D2H nan top err");
        if ((e2 & 16u) == 0u || et != e2) {
            std::fprintf(stderr, "FAIL NaN top_tokens err mismatch ref=%u got=%u\n", et, e2);
            return 1;
        }
        std::printf("  [OK] NaN error bit 16 parity check passed (ref/6p2 verify err=%u/%u, top_tokens err=%u/%u).\n", er, e1, et, e2);
    }

    // 6. Clustered MTP fast-head + penalty + topk test (512 contiguous blocks)
    {
        const int NC = 512;
        std::vector<int32_t> h_coff(NC + 1, 0);
        for (int c = 0; c <= NC; ++c) h_coff[c] = (c * rows) / NC;
        std::vector<uint16_t> h_cent(static_cast<size_t>(NC) * cols, f32_to_f16_bits(0.0f));
        for (int c = 0; c < NC; ++c) {
            std::vector<double> mean(cols, 0.0);
            for (int r = h_coff[c]; r < h_coff[c + 1]; ++r) {
                for (int b = 0; b < nb; ++b) {
                    float dw = f16_bits_to_f32(h_ref_d[static_cast<size_t>(r) * nb + b]);
                    for (int k = 0; k < 32; ++k) mean[b * 32 + k] += dw * h_ref_q[static_cast<size_t>(r) * cols + b * 32 + k];
                }
            }
            double n2 = 0.0;
            for (double v : mean) n2 += v * v;
            double inv = (n2 > 0.0) ? (1.0 / std::sqrt(n2)) : 0.0;
            for (int j = 0; j < cols; ++j) h_cent[static_cast<size_t>(c) * cols + j] = f32_to_f16_bits(static_cast<float>(mean[j] * inv));
        }
        uint16_t* d_cent = nullptr;
        int32_t* d_coff = nullptr;
        ck(cudaMalloc(&d_cent, h_cent.size() * 2), "d_cent");
        ck(cudaMalloc(&d_coff, h_coff.size() * 4), "d_coff");
        ck(cudaMemcpy(d_cent, h_cent.data(), h_cent.size() * 2, cudaMemcpyHostToDevice), "H2D cent");
        ck(cudaMemcpy(d_coff, h_coff.data(), h_coff.size() * 4, cudaMemcpyHostToDevice), "H2D coff");

        make_xf(1, 0);
        fnx::quantize_q8_1(d_xf, d_y_twin, cols, 1, st);
        fnx::vhead_all_logits_6p2(d_vh, d_y_twin, d_logits_ref, 1, st);

        int h_mtp_pen[5] = {1, 0, 0, 2, 0};
        // Penalize row 0 by putting index 0 in cluster 0 (force nonzero check regardless of winning token)
        float h_mtp_par[3] = {1.2f, 0.5f, 0.3f};
        int* d_mtp_pen = nullptr;
        float* d_mtp_par = nullptr;
        ck(cudaMalloc(&d_mtp_pen, 5 * 4), "d_mtp_pen");
        ck(cudaMalloc(&d_mtp_par, 3 * 4), "d_mtp_par");
        ck(cudaMemcpy(d_mtp_pen, h_mtp_pen, 5 * 4, cudaMemcpyHostToDevice), "H2D mtp_pen");
        ck(cudaMemcpy(d_mtp_par, h_mtp_par, 3 * 4, cudaMemcpyHostToDevice), "H2D mtp_par");
        ck(cudaMemset(d_err_6p2, 0, 4), "zero err");

        fnx::vhead_mtp_fast_6p2(d_vh, d_cent, d_coff, NC, 64, d_y_twin, 1, 0, d_mtp_pen, d_mtp_par,
                                d_bv_6p2, d_bi_6p2, d_keys_6p2, d_err_6p2, work, st);
        fnx::vhead_mtp_fast_topk_6p2(d_vh, 0, 0, d_topv_6p2, d_topi_6p2, d_err_6p2, work, st);
        ck(cudaStreamSynchronize(st), "sync mtp");

        float mbv = 0.0f;
        int mbi = -1;
        ck(cudaMemcpy(&mbv, d_bv_6p2, 4, cudaMemcpyDeviceToHost), "D2H mbv");
        ck(cudaMemcpy(&mbi, d_bi_6p2, 4, cudaMemcpyDeviceToHost), "D2H mbi");
        std::vector<float> mtv(fnx::TOPK_N);
        std::vector<int> mti(fnx::TOPK_N);
        ck(cudaMemcpy(mtv.data(), d_topv_6p2, fnx::TOPK_N * 4, cudaMemcpyDeviceToHost), "D2H mtv");
        ck(cudaMemcpy(mti.data(), d_topi_6p2, fnx::TOPK_N * 4, cudaMemcpyDeviceToHost), "D2H mti");
        if (mbi != mti[0] || mbv != mtv[0] || !std::isfinite(mbv)) {
            std::fprintf(stderr, "FAIL MTP fasthead argmax (%g,%d) != top0 (%g,%d)\n", mbv, mbi, mtv[0], mti[0]);
            return 1;
        }
        std::printf("  [OK] MTP fast-head + penalize_row + top64 passed: top=(%.4f, id=%d).\n", mbv, mbi);
        cudaFree(d_cent);
        cudaFree(d_coff);
        cudaFree(d_mtp_pen);
        cudaFree(d_mtp_par);
    }

    // 7. Soundness over 64 diverse synthetic hidden states + GPU CUDA-event benchmark vs q8s_gemv + argmax
    {
        uint64_t total_surv = 0;
        for (int s = 0; s < 32; ++s) {
            make_xf(1, s % 2);
            fnx::quantize_q8_1(d_xf, d_y_twin, cols, 1, st);
            fnx::vhead_verify_6p2(d_vh, d_y_twin, 1, 0, d_bv_6p2, d_bi_6p2, d_keys_6p2, d_err_6p2, work, st);
            uint32_t c = 0;
            ck(cudaMemcpyAsync(&c, work.cand_cnt, 4, cudaMemcpyDeviceToHost, st), "D2H cnt");
            ck(cudaStreamSynchronize(st), "sync");
            total_surv += c;
        }
        std::printf("  [INFO] 32-state T=1 Gaussian+outlier mean survivors: %.1f / %d (%.2f%%)\n",
                    static_cast<double>(total_surv) / 32.0, rows,
                    100.0 * (static_cast<double>(total_surv) / 32.0) / rows);

        // Timed comparison (total + stage breakdown)
        cudaEvent_t ev0, ev1, es0, es1, es2, es3;
        ck(cudaEventCreate(&ev0), "ev0");
        ck(cudaEventCreate(&ev1), "ev1");
        ck(cudaEventCreate(&es0), "es0");
        ck(cudaEventCreate(&es1), "es1");
        ck(cudaEventCreate(&es2), "es2");
        ck(cudaEventCreate(&es3), "es3");
        const int bench_T[] = {1, 4, 5, 8};
        for (int T : bench_T) {
            make_xf(T, 1);
            fnx::quantize_q8_1(d_xf, d_y_twin, cols, T, st);
            for (int w = 0; w < 3; ++w) {
                fnx::q8s_gemv(d_ref, rows, cols, d_y_twin, d_logits_ref, rows, T, st);
                fnx::argmax(d_logits_ref, rows, T, 0, d_bv_ref, d_bi_ref, d_keys_ref, d_err_ref, st);
                fnx::vhead_verify_6p2(d_vh, d_y_twin, T, 0, d_bv_6p2, d_bi_6p2, d_keys_6p2, d_err_6p2, work, st);
            }
            ck(cudaStreamSynchronize(st), "warmup");
            uint32_t surv[8] = {};
            ck(cudaMemcpy(surv, work.cand_cnt, T * 4, cudaMemcpyDeviceToHost), "D2H surv");
            uint64_t surv_sum = 0;
            for (int t = 0; t < T; ++t) surv_sum += surv[t];

            const int NITER = 15;
            ck(cudaEventRecord(ev0, st), "ev0");
            for (int i = 0; i < NITER; ++i) {
                fnx::q8s_gemv(d_ref, rows, cols, d_y_twin, d_logits_ref, rows, T, st);
                fnx::argmax(d_logits_ref, rows, T, 0, d_bv_ref, d_bi_ref, d_keys_ref, d_err_ref, st);
            }
            ck(cudaEventRecord(ev1, st), "ev1");
            ck(cudaEventSynchronize(ev1), "sync ref bench");
            float ms_ref = 0.0f;
            ck(cudaEventElapsedTime(&ms_ref, ev0, ev1), "elapsed ref");

            ck(cudaEventRecord(ev0, st), "ev0");
            for (int i = 0; i < NITER; ++i) {
                fnx::vhead_verify_6p2(d_vh, d_y_twin, T, 0, d_bv_6p2, d_bi_6p2, d_keys_6p2, d_err_6p2, work, st);
            }
            ck(cudaEventRecord(ev1, st), "ev1");
            ck(cudaEventSynchronize(ev1), "sync 6p2 bench");
            float ms_6p2 = 0.0f;
            ck(cudaEventElapsedTime(&ms_6p2, ev0, ev1), "elapsed 6p2");
            ms_ref /= NITER;
            ms_6p2 /= NITER;
            std::printf("  [BENCH] T=%d verify (avg_surv=%.0f, %.2f%%): q8s_gemv+argmax = %.3f ms | vhead_verify_6p2 = %.3f ms (%.2fx)\n",
                        T, static_cast<double>(surv_sum) / T, 100.0 * (static_cast<double>(surv_sum) / T) / rows,
                        ms_ref, ms_6p2, ms_ref / ms_6p2);
            // Measure all-logits as Stage-1+2 proxy vs vhead_verify to pinpoint bottleneck
            ck(cudaEventRecord(es0, st), "es0");
            for (int i = 0; i < NITER; ++i) fnx::vhead_all_logits_6p2(d_vh, d_y_twin, d_logits_6p2, T, st);
            ck(cudaEventRecord(es1, st), "es1");
            ck(cudaEventSynchronize(es1), "sync es1");
            float ms_all = 0.0f;
            ck(cudaEventElapsedTime(&ms_all, es0, es1), "elapsed all");
            std::printf("          T=%d vhead_all_logits_6p2 = %.3f ms\n", T, ms_all / NITER);
        }
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        cudaEventDestroy(es0);
        cudaEventDestroy(es1);
        cudaEventDestroy(es2);
        cudaEventDestroy(es3);
    }

    fnx::q8_untwin(d_y_twin);
    cudaFree(d_ref.q); cudaFree(d_ref.d);
    cudaFree(d_vh.hi); cudaFree(d_vh.lo); cudaFree(d_vh.d); cudaFree(d_vh.row_meta);
    cudaFree(d_pvh.hi); cudaFree(d_pvh.lo); cudaFree(d_pvh.d); cudaFree(d_pvh.row_meta);
    cudaFree(d_pvh.perm_to_vocab); cudaFree(d_pvh.vocab_to_local);
    cudaFree(work.xq); cudaFree(work.xd); cudaFree(work.xd_half); cudaFree(work.sx253); cudaFree(work.act_meta);
    cudaFree(work.lb); cudaFree(work.ub); cudaFree(work.cta_max); cudaFree(work.lcut);
    cudaFree(work.cand_idx); cudaFree(work.cand_cnt);
    cudaFree(d_xf); cudaFree(d_y_q81); cudaFree(d_y_twin); cudaFree(d_tq); cudaFree(d_td);
    cudaFree(d_logits_ref); cudaFree(d_logits_6p2);
    cudaFree(d_bv_ref); cudaFree(d_bv_6p2); cudaFree(d_bi_ref); cudaFree(d_bi_6p2);
    cudaFree(d_keys_ref); cudaFree(d_keys_6p2); cudaFree(d_err_ref); cudaFree(d_err_6p2);
    cudaFree(d_pen_tok); cudaFree(d_pen_cnt); cudaFree(d_pen_n); cudaFree(d_pen_par);
    cudaFree(d_topv_ref); cudaFree(d_topv_6p2); cudaFree(d_topi_ref); cudaFree(d_topi_6p2);
    cudaStreamDestroy(st);
    std::printf("vhead_6p2_test: PASS\n");
    return 0;
}
