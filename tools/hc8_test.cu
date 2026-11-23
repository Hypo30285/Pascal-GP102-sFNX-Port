#include "../src/gguf.hpp"
#include "../src/kernels.hpp"
#include "../src/model.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if FNX_27B

int main() {
    std::printf("hc8_test: FNX_27B=1 build does not use Flash-Next hc_inject; build with FNX_MODEL=fnx to run.\n");
    return 0;
}

#else

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

float bf16_bits_to_f32(uint16_t h) {
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

std::vector<float> tensor_to_f32(const fnx::Tensor& t) {
    const size_t n = static_cast<size_t>(t.ne[0]) * t.ne[1] * t.ne[2] * t.ne[3];
    std::vector<float> out(n);
    if (t.type == fnx::T_F32) {
        for (size_t i = 0; i < n; ++i) std::memcpy(&out[i], t.data + i * 4, 4);
    } else if (t.type == fnx::T_BF16) {
        for (size_t i = 0; i < n; ++i) {
            uint16_t b;
            std::memcpy(&b, t.data + i * 2, 2);
            out[i] = bf16_bits_to_f32(b);
        }
    } else if (t.type == fnx::T_F16) {
        for (size_t i = 0; i < n; ++i) {
            uint16_t h;
            std::memcpy(&h, t.data + i * 2, 2);
            out[i] = f16_bits_to_f32(h);
        }
    } else if (t.type == fnx::T_Q8_0) {
        for (size_t b = 0; b < n / 32; ++b) {
            const uint8_t* blk = t.data + b * 34;
            uint16_t db;
            std::memcpy(&db, blk, 2);
            const float d = f16_bits_to_f32(db);
            for (int j = 0; j < 32; ++j) out[b * 32 + j] = static_cast<int8_t>(blk[2 + j]) * d;
        }
    } else {
        throw std::runtime_error(t.name + ": unsupported type " + std::to_string(t.type));
    }
    return out;
}

void decode_q80_row(const fnx::Tensor& t, int64_t row, std::vector<float>& out, float scale = 1.0f) {
    const size_t n = static_cast<size_t>(t.ne[0]);
    const size_t rb = t.row_bytes();
    const uint8_t* rp = t.data + static_cast<size_t>(row) * rb;
    out.resize(n);
    for (size_t b = 0; b < n / 32; ++b) {
        const uint8_t* blk = rp + b * 34;
        uint16_t db;
        std::memcpy(&db, blk, 2);
        const float d = f16_bits_to_f32(db) * scale;
        for (int j = 0; j < 32; ++j) out[b * 32 + j] = static_cast<int8_t>(blk[2 + j]) * d;
    }
}

void quantize_q8s(const std::vector<float>& src, std::vector<int8_t>& q, std::vector<uint16_t>& d) {
    const size_t n = src.size();
    if (n % 32 != 0) throw std::runtime_error("quantize_q8s: size must be multiple of 32");
    q.resize(n);
    d.resize(n / 32);
    for (size_t b = 0; b < n / 32; ++b) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(src[b * 32 + j]));
        const float dd = amax / 127.0f, id = dd > 0.f ? 1.0f / dd : 0.f;
        for (int j = 0; j < 32; ++j) q[b * 32 + j] = static_cast<int8_t>(std::lround(src[b * 32 + j] * id));
        d[b] = f32_to_f16_bits(dd);
    }
}

void tensor_to_q8s(const fnx::Tensor& t, std::vector<int8_t>& q, std::vector<uint16_t>& d) {
    const size_t n = static_cast<size_t>(t.ne[0]) * t.ne[1] * t.ne[2] * t.ne[3];
    if (n % 32 != 0) throw std::runtime_error(t.name + ": size must be multiple of 32");
    if (t.type == fnx::T_Q8_0) {
        q.resize(n);
        d.resize(n / 32);
        for (size_t b = 0; b < n / 32; ++b) {
            const uint8_t* blk = t.data + b * 34;
            std::memcpy(&d[b], blk, 2);
            std::memcpy(&q[b * 32], blk + 2, 32);
        }
        return;
    }
    quantize_q8s(tensor_to_f32(t), q, d);
}

std::vector<float> deq_q8s(const std::vector<int8_t>& q, const std::vector<uint16_t>& d) {
    std::vector<float> out(q.size());
    for (size_t b = 0; b < d.size(); ++b) {
        const float dd = f16_bits_to_f32(d[b]);
        for (int j = 0; j < 32; ++j) out[b * 32 + j] = static_cast<float>(q[b * 32 + j]) * dd;
    }
    return out;
}

float sigm_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float silu_f(float x) { return x * sigm_f(x); }

struct Metrics {
    double cosine = 0.0;
    double max_abs = 0.0;
};

Metrics compare_vec(const float* a, const float* b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0, mad = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double va = a[i], vb = b[i];
        dot += va * vb;
        na += va * va;
        nb += vb * vb;
        mad = std::max(mad, std::fabs(va - vb));
    }
    const double den = std::sqrt(na) * std::sqrt(nb);
    return {den > 0.0 ? dot / den : (mad == 0.0 ? 1.0 : 0.0), mad};
}

void eval_hc_ref(int T, float eps, const float* R_init, const float* bo, const float* inj_prev,
                 const std::vector<float>& wn, const std::vector<float>& wd_f32, const std::vector<float>& wu_f32,
                 const std::vector<float>* winj_f32, std::vector<float>& mixed_ref, std::vector<float>& inj_ref) {
    mixed_ref.assign(static_cast<size_t>(T) * fnx::N, 0.f);
    inj_ref.assign(static_cast<size_t>(T) * fnx::HC, 0.f);
    std::vector<double> xn(fnx::D), lo(fnx::LR);
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < fnx::HC; ++c) {
            const double g = bo ? 2.0 * sigm_f(inj_prev[t * fnx::HC + c] / static_cast<float>(fnx::HC)) : 0.0;
            double ss = 0.0;
            std::vector<double> rp(fnx::N);
            for (int i = 0; i < fnx::N; ++i) {
                double r = R_init[static_cast<size_t>(t) * fnx::D + c * fnx::N + i];
                if (bo) r += bo[static_cast<size_t>(t) * fnx::N + i] * g;
                rp[i] = r;
                ss += r * r;
            }
            const double rs = 1.0 / std::sqrt(ss / static_cast<double>(fnx::N) + static_cast<double>(eps));
            for (int i = 0; i < fnx::N; ++i) xn[c * fnx::N + i] = rp[i] * wn[c * fnx::N + i] * rs;
        }
        for (int r = 0; r < fnx::LR; ++r) {
            const float* wrow = &wd_f32[static_cast<size_t>(r) * fnx::D];
            double acc = 0.0;
            for (int j = 0; j < fnx::D; ++j) acc += static_cast<double>(wrow[j]) * xn[j];
            lo[r] = silu_f(static_cast<float>(acc) / static_cast<float>(fnx::HC));
        }
        if (winj_f32) {
            for (int r = 0; r < fnx::HC; ++r) {
                const float* wrow = &(*winj_f32)[static_cast<size_t>(r) * fnx::D];
                double acc = 0.0;
                for (int j = 0; j < fnx::D; ++j) acc += static_cast<double>(wrow[j]) * xn[j];
                inj_ref[t * fnx::HC + r] = static_cast<float>(acc);
            }
        }
        for (int c = 0; c < fnx::HC; ++c) {
            for (int i = 0; i < fnx::N; ++i) {
                const size_t urow = static_cast<size_t>(c) * fnx::N + i;
                const float* wrow = &wu_f32[urow * fnx::LR];
                double acc = 0.0;
                for (int r = 0; r < fnx::LR; ++r) acc += static_cast<double>(wrow[r]) * lo[r];
                const double val = xn[urow] * sigm_f(static_cast<float>(acc));
                mixed_ref[static_cast<size_t>(t) * fnx::N + i] += static_cast<float>(val / static_cast<double>(fnx::HC));
            }
        }
    }
}

struct DevQ8S {
    int8_t* q = nullptr;
    uint16_t* d = nullptr;
    void upload(const std::vector<int8_t>& hq, const std::vector<uint16_t>& hd) {
        ck(cudaMalloc(&q, hq.size()), "cudaMalloc q");
        ck(cudaMalloc(&d, hd.size() * sizeof(uint16_t)), "cudaMalloc d");
        ck(cudaMemcpy(q, hq.data(), hq.size(), cudaMemcpyHostToDevice), "cudaMemcpy q");
        ck(cudaMemcpy(d, hd.data(), hd.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "cudaMemcpy d");
    }
    fnx::Q8S view() const { return {q, d}; }
    void free_all() {
        if (q) cudaFree(q);
        if (d) cudaFree(d);
        q = nullptr;
        d = nullptr;
    }
};

}  // namespace

int main(int argc, char** argv) {
    const std::string model_path =
        argc > 1 ? argv[1]
                 : "/mnt/storage_8tb_linux/models/gguf/qwen3.8-flash-next-ud-q6_k_xl/"
                   "qwen3.8-flash-next-ud-q6_k_xl/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf";
    std::printf("hc8_test: opening %s...\n", model_path.c_str());
    fnx::Gguf g(model_path);

    const fnx::Tensor& te = g.t("token_embd.weight");
    if (te.type != fnx::T_Q8_0 || te.ne[0] != fnx::N) {
        std::fprintf(stderr, "Unexpected token_embd.weight shape (%lld x %lld) type=%u\n",
                     (long long) te.ne[0], (long long) te.ne[1], te.type);
        return 1;
    }

    const float eps = 1e-6f;
    ck(cudaSetDevice(0), "cudaSetDevice(0)");
    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st), "cudaStreamCreate");

    // Real Flash-Next token_embd.weight activations replicated across the HC=4 streams, plus distinct real branch residual bo
    const int64_t tok_seeds[] = {1, 10, 42, 100, 256, 512, 1024, 2048, 3000, 4096, 5000, 6000, 7000, 8192, 9000, 10000};
    constexpr int n_seeds = sizeof(tok_seeds) / sizeof(tok_seeds[0]);

    std::mt19937 rng(424242);
    std::uniform_real_distribution<float> dist_inj(-1.5f, 1.5f);

    std::vector<float> h_R_init(static_cast<size_t>(fnx::MAXT) * fnx::D);
    std::vector<float> h_bo(static_cast<size_t>(fnx::MAXT) * fnx::N);
    std::vector<float> h_inj_prev(static_cast<size_t>(fnx::MAXT) * fnx::HC);
    std::vector<float> row_a, row_b;
    for (int t = 0; t < fnx::MAXT; ++t) {
        const int64_t tok_a = (t < n_seeds ? tok_seeds[t] : static_cast<int64_t>(t) * 997 + 1) % te.ne[1];
        const int64_t tok_b = (tok_a + 777) % te.ne[1];
        decode_q80_row(te, tok_a, row_a, 1.0f);
        decode_q80_row(te, tok_b, row_b, 0.15f);
        for (int c = 0; c < fnx::HC; ++c) {
            std::memcpy(&h_R_init[static_cast<size_t>(t) * fnx::D + c * fnx::N], row_a.data(), fnx::N * sizeof(float));
        }
        std::memcpy(&h_bo[static_cast<size_t>(t) * fnx::N], row_b.data(), fnx::N * sizeof(float));
    }
    for (float& v : h_inj_prev) v = dist_inj(rng);

    float *d_R = nullptr, *d_bo = nullptr, *d_inj_prev = nullptr, *d_wn = nullptr;
    float *d_mixed = nullptr, *d_inject = nullptr, *d_scratch = nullptr;
    unsigned* d_err = nullptr;
    ck(cudaMalloc(&d_R, h_R_init.size() * sizeof(float)), "d_R");
    ck(cudaMalloc(&d_bo, h_bo.size() * sizeof(float)), "d_bo");
    ck(cudaMalloc(&d_inj_prev, h_inj_prev.size() * sizeof(float)), "d_inj_prev");
    ck(cudaMalloc(&d_wn, static_cast<size_t>(fnx::D) * sizeof(float)), "d_wn");
    ck(cudaMalloc(&d_mixed, static_cast<size_t>(fnx::MAXT) * fnx::N * sizeof(float)), "d_mixed");
    ck(cudaMalloc(&d_inject, static_cast<size_t>(fnx::MAXT) * fnx::HC * sizeof(float)), "d_inject");
    ck(cudaMalloc(&d_scratch, fnx::hc_scratch_floats(fnx::MAXT) * sizeof(float)), "d_scratch");
    ck(cudaMalloc(&d_err, sizeof(unsigned)), "d_err");
    ck(cudaMemcpy(d_bo, h_bo.data(), h_bo.size() * sizeof(float), cudaMemcpyHostToDevice), "copy d_bo");
    ck(cudaMemcpy(d_inj_prev, h_inj_prev.data(), h_inj_prev.size() * sizeof(float), cudaMemcpyHostToDevice), "copy d_inj_prev");

    bool all_ok = true;
    const char* prefixes[4] = {"blk.0.hc_attn_", "blk.0.hc_ffn_", "blk.1.hc_attn_", "blk.1.hc_ffn_"};
    for (const char* pfx : prefixes) {
        const fnx::Tensor& tn = g.t(std::string(pfx) + "norm.weight");
        const fnx::Tensor& td = g.t(std::string(pfx) + "down.weight");
        const fnx::Tensor& tu = g.t(std::string(pfx) + "up.weight");
        const fnx::Tensor& ti = g.t(std::string(pfx) + "inject.weight");
        if (ti.ne[0] != fnx::D || ti.ne[1] != fnx::HC) {
            std::fprintf(stderr, "Unexpected %sinject.weight shape (%lld x %lld)\n",
                         pfx, (long long) ti.ne[0], (long long) ti.ne[1]);
            return 1;
        }
        const std::vector<float> wn = tensor_to_f32(tn);
        std::vector<int8_t> wd_q, wu_q, wi_q;
        std::vector<uint16_t> wd_d, wu_d, wi_d;
        tensor_to_q8s(td, wd_q, wd_d);
        tensor_to_q8s(tu, wu_q, wu_d);
        const std::vector<float> winj_f32 = tensor_to_f32(ti);
        quantize_q8s(winj_f32, wi_q, wi_d);

        const std::vector<float> wd_f32 = deq_q8s(wd_q, wd_d);
        const std::vector<float> wu_f32 = deq_q8s(wu_q, wu_d);
        const std::vector<float> winj_hc8 = deq_q8s(wi_q, wi_d);
        const Metrics w_m = compare_vec(winj_f32.data(), winj_hc8.data(), winj_f32.size());
        std::printf("\n[%s] inject.weight type=%u dims=(%lld,%lld) -> hc8 weight cos=%.6f max_abs=%.6f\n",
                    pfx, ti.type, (long long) ti.ne[0], (long long) ti.ne[1], w_m.cosine, w_m.max_abs);

        ck(cudaMemcpy(d_wn, wn.data(), wn.size() * sizeof(float), cudaMemcpyHostToDevice), "copy d_wn");
        DevQ8S dev_wd, dev_wu, dev_wi;
        dev_wd.upload(wd_q, wd_d);
        dev_wu.upload(wu_q, wu_d);
        dev_wi.upload(wi_q, wi_d);

        const int ts[] = {1, 4, 8, 16};
        for (int T : ts) {
            for (int with_bo : {1, 0}) {
                if (T != 4 && !with_bo) continue;
                for (int with_inj : {1, 0}) {
                    if (T != 4 && !with_inj) continue;
                    if (!with_bo && !with_inj) continue;

                    ck(cudaMemcpy(d_R, h_R_init.data(), static_cast<size_t>(T) * fnx::D * sizeof(float), cudaMemcpyHostToDevice),
                       "reset d_R");
                    ck(cudaMemset(d_scratch, 0, fnx::hc_scratch_floats(fnx::MAXT) * sizeof(float)), "zero scratch");
                    ck(cudaMemset(d_err, 0, sizeof(unsigned)), "zero err");
                    ck(cudaMemset(d_inject, 0, static_cast<size_t>(T) * fnx::HC * sizeof(float)), "zero inj");
                    fnx::hc_prepare(d_scratch);

                    const fnx::Q8S inj_w = with_inj ? dev_wi.view() : fnx::Q8S{};
                    const float* bo_ptr = with_bo ? d_bo : nullptr;
                    const float* ip_ptr = with_bo ? d_inj_prev : nullptr;

                    cudaEvent_t e0, e1;
                    ck(cudaEventCreate(&e0), "e0");
                    ck(cudaEventCreate(&e1), "e1");
                    ck(cudaEventRecord(e0, st), "rec0");
                    fnx::hc_read(d_R, bo_ptr, ip_ptr, d_wn, dev_wd.view(), dev_wu.view(), inj_w,
                                 eps, d_mixed, d_inject, d_scratch, d_err, T, st);
                    ck(cudaEventRecord(e1, st), "rec1");
                    ck(cudaStreamSynchronize(st), "sync");
                    float ms = 0.f;
                    ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
                    ck(cudaEventDestroy(e0), "de0");
                    ck(cudaEventDestroy(e1), "de1");

                    unsigned h_err = 0;
                    ck(cudaMemcpy(&h_err, d_err, sizeof(unsigned), cudaMemcpyDeviceToHost), "copy err");
                    if (h_err != 0) {
                        std::fprintf(stderr, "  FAIL [%s] T=%d bo=%d inj=%d err=0x%x\n", pfx, T, with_bo, with_inj, h_err);
                        all_ok = false;
                    }

                    std::vector<float> got_mixed(static_cast<size_t>(T) * fnx::N);
                    std::vector<float> got_inj(static_cast<size_t>(T) * fnx::HC);
                    ck(cudaMemcpy(got_mixed.data(), d_mixed, got_mixed.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy mixed");
                    ck(cudaMemcpy(got_inj.data(), d_inject, got_inj.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy inj");

                    std::vector<float> ref_mixed, ref_inj;
                    eval_hc_ref(T, eps, h_R_init.data(), with_bo ? h_bo.data() : nullptr,
                                with_bo ? h_inj_prev.data() : nullptr, wn, wd_f32, wu_f32,
                                with_inj ? &winj_f32 : nullptr, ref_mixed, ref_inj);

                    const Metrics m_mix = compare_vec(ref_mixed.data(), got_mixed.data(), ref_mixed.size());
                    const Metrics m_inj = with_inj ? compare_vec(ref_inj.data(), got_inj.data(), ref_inj.size())
                                                   : Metrics{1.0, 0.0};
                    const bool pass = (h_err == 0) && (m_mix.cosine >= 0.9999) && (m_inj.cosine >= 0.9999);
                    if (!pass) all_ok = false;
                    std::printf("  T=%2d bo=%d inj=%d | mixed cos=%.6f max_diff=%.5f | inj cos=%.6f max_diff=%.5f | %.1f us | %s\n",
                                T, with_bo, with_inj, m_mix.cosine, m_mix.max_abs, m_inj.cosine, m_inj.max_abs,
                                ms * 1000.f, pass ? "OK" : "FAIL");
                }
            }
        }

        dev_wd.free_all();
        dev_wu.free_all();
        dev_wi.free_all();
    }

    cudaFree(d_R);
    cudaFree(d_bo);
    cudaFree(d_inj_prev);
    cudaFree(d_wn);
    cudaFree(d_mixed);
    cudaFree(d_inject);
    cudaFree(d_scratch);
    cudaFree(d_err);
    cudaStreamDestroy(st);

    if (!all_ok) {
        std::printf("hc8_test: FAIL\n");
        return 1;
    }
    std::printf("\nhc8_test: PASS (all real Flash-Next blk.0 & blk.1 hc8 outputs achieved cosine >= 0.9999 vs FP32 reference)\n");
    return 0;
}

#endif
