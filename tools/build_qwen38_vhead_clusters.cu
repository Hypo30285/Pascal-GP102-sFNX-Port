// Offline 1,024-cluster (512 per card) spherical k-means builder for VHead6p2 + clustered MTP fast-head.
// Usage: build_qwen38_vhead_clusters [MODEL.gguf] [OUT.bin] [K_PER_CARD=512] [ITERS=25]
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../src/env.hpp"
#include "../src/gguf.hpp"
#include "../src/model.hpp"

using namespace fnx;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

void ck_bl(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(what) + ": cublas error " + std::to_string((int) s));
}

inline float f16_to_f32_h(uint16_t h) {
    uint32_t sign = (uint32_t) (h >> 15) << 31;
    uint32_t exp = (h >> 10) & 31u;
    uint32_t frac = h & 1023u;
    uint32_t bits = 0;
    if (exp == 0) {
        if (frac == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((frac & 1024u) == 0) { frac <<= 1; --e; }
            bits = sign | ((uint32_t) (e + 127) << 23) | ((frac & 1023u) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (frac << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (frac << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t f32_to_f16_h(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    uint32_t s = (u >> 16) & 0x8000u;
    int e = (int) ((u >> 23) & 0xff) - 127 + 15;
    uint32_t m = u & 0x7fffffu;
    if (e <= 0) {
        if (e < -10) return (uint16_t) s;
        m |= 0x800000u;
        int sh = 14 - e;
        uint32_t h = m >> sh;
        if (((m >> (sh - 1)) & 1u) && ((m & ((1u << (sh - 1)) - 1)) || (h & 1u))) ++h;
        return (uint16_t) (s | h);
    }
    if (e >= 31) return (uint16_t) (s | 0x7c00u);
    uint32_t h = ((uint32_t) e << 10) | (m >> 13);
    if (((m >> 12) & 1u) && ((m & 0xfffu) || (h & 1u))) ++h;
    return (uint16_t) (s | h);
}

// 1 warp per row: computes argmax_k S[r * K + k] (smallest k on tie)
__global__ void k_kmeans_assign(const float* __restrict__ S, int K, int n_out,
                                int* __restrict__ assign, float* __restrict__ best_sc,
                                int* __restrict__ counts) {
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= n_out) return;
    const int lane = threadIdx.x & 31;
    const float* row = S + (size_t) r * K;
    float bv = -1.0e30f;
    int bk = 0;
    for (int k = lane; k < K; k += 32) {
        float v = row[k];
        if (v > bv || (v == bv && k < bk)) { bv = v; bk = k; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        int ok = __shfl_xor_sync(0xffffffffu, bk, o);
        if (ov > bv || (ov == bv && ok < bk)) { bv = ov; bk = ok; }
    }
    if (lane == 0) {
        assign[r] = bk;
        best_sc[r] = bv;
        atomicAdd(&counts[bk], 1);
    }
}

__global__ void k_kmeans_repair_empty(int n_out, int K, int* __restrict__ assign,
                                      const float* __restrict__ best_sc, int* __restrict__ counts) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int k = 0; k < K; ++k) {
        if (counts[k] > 0) continue;
        float worst_v = 1.0e30f;
        int worst_r = -1;
        for (int r = 0; r < n_out; ++r) {
            int cur = assign[r];
            if (counts[cur] > 1 && best_sc[r] < worst_v) {
                worst_v = best_sc[r];
                worst_r = r;
            }
        }
        if (worst_r >= 0) {
            --counts[assign[worst_r]];
            assign[worst_r] = k;
            ++counts[k];
        }
    }
}

__global__ void k_kmeans_accum(const __half* __restrict__ X, const int* __restrict__ assign,
                               int n_out, int n_in, float* __restrict__ sums) {
    const int warp_global = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (warp_global >= n_out) return;
    const int k = assign[warp_global];
    const __half* xr = X + (size_t) warp_global * n_in;
    float* dst = sums + (size_t) k * n_in;
    for (int i = lane; i < n_in; i += 32) {
        atomicAdd(&dst[i], __half2float(xr[i]));
    }
}

__global__ void k_kmeans_norm(const float* __restrict__ sums, const int* __restrict__ counts,
                              int K, int n_in, __half* __restrict__ C) {
    const int k = blockIdx.x;
    if (k >= K || counts[k] <= 0) return;
    const float* sr = sums + (size_t) k * n_in;
    __half* dst = C + (size_t) k * n_in;
    __shared__ float sh[8];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    float ss = 0.0f;
    for (int i = tid; i < n_in; i += blockDim.x) {
        float v = sr[i];
        ss += v * v;
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) sh[warp] = ss;
    __syncthreads();
    float tot = 0.0f;
    if (warp == 0) {
        tot = (lane < (blockDim.x >> 5)) ? sh[lane] : 0.0f;
        for (int o = 4; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
        if (lane == 0) sh[0] = tot > 0.0f ? rsqrtf(tot) : 0.0f;
    }
    __syncthreads();
    const float inv = sh[0];
    for (int i = tid; i < n_in; i += blockDim.x) {
        dst[i] = __float2half(sr[i] * inv);
    }
}

void dequant_norm_slice(const Tensor& t, int v0, int v1, std::vector<uint16_t>& out_f16) {
    const int n_out = v1 - v0, n_in = (int) t.ne[0], nb = n_in / 32;
    const size_t rb = t.row_bytes();
    out_f16.resize((size_t) n_out * n_in);
    int nth = std::min<unsigned>(12u, std::max<unsigned>(1u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int ti = 0; ti < nth; ++ti) {
        int r0 = (int) ((size_t) n_out * ti / nth), r1 = (int) ((size_t) n_out * (ti + 1) / nth);
        th.emplace_back([&, r0, r1] {
            std::vector<float> row(n_in);
            for (int r = r0; r < r1; ++r) {
                const uint8_t* src = t.data + (size_t) (v0 + r) * rb;
                double ss = 0.0;
                for (int b = 0; b < nb; ++b) {
                    uint16_t dh = 0;
                    std::memcpy(&dh, src + (size_t) b * 34, 2);
                    float d = f16_to_f32_h(dh);
                    const int8_t* q = reinterpret_cast<const int8_t*>(src + (size_t) b * 34 + 2);
                    for (int j = 0; j < 32; ++j) {
                        float v = d * (float) q[j];
                        row[b * 32 + j] = v;
                        ss += (double) v * (double) v;
                    }
                }
                uint16_t* dst = out_f16.data() + (size_t) r * n_in;
                if (ss > 0.0) {
                    float inv = (float) (1.0 / std::sqrt(ss));
                    for (int i = 0; i < n_in; ++i) dst[i] = f32_to_f16_h(row[i] * inv);
                } else {
                    std::memset(dst, 0, (size_t) n_in * 2);
                    dst[0] = f32_to_f16_h(1.0f);
                }
            }
        });
    }
    for (auto& x : th) x.join();
}

}  // namespace

int main(int argc, char** argv) {
    const std::string model_path = argc > 1 ? argv[1]
        : "/mnt/storage_8tb_linux/models/gguf/Qwen3.8-27B-Q8_0/Qwen3.8-27B-Q8_0.gguf";
    const std::string out_path = argc > 2 ? argv[2]
        : "/mnt/storage_8tb_linux/models/gguf/Qwen3.8-27B-Q8_0/qwen38_vhead_clusters_1024.bin";
    long long k_arg = 512, it_arg = 25;
    if (argc > 3 && (!parse_whole(argv[3], k_arg) || k_arg < 1 || k_arg > 65536))
        throw std::runtime_error("bad K_PER_CARD");
    if (argc > 4 && (!parse_whole(argv[4], it_arg) || it_arg < 1 || it_arg > 200))
        throw std::runtime_error("bad ITERS");
    const int K = (int) k_arg, ITERS = (int) it_arg;

    std::printf("build_qwen38_vhead_clusters: loading %s\n", model_path.c_str());
    Gguf g(model_path);
    const Tensor& out_w = g.t("output.weight");
    if (out_w.type != T_Q8_0) throw std::runtime_error("output.weight is not Q8_0");
    const int n_in = (int) out_w.ne[0], vocab = (int) out_w.ne[1];
    if (n_in <= 0 || (n_in & 511) != 0) throw std::runtime_error("output.weight n_in must be a multiple of 512");
    const int vstep = (vocab + NCARD - 1) / NCARD;
    std::printf("  output.weight [%d x %d], ncard=%d, k_per_card=%d (%d total), iters=%d\n",
                vocab, n_in, NCARD, K, K * NCARD, ITERS);

    ck(cudaSetDevice(0), "setDev");
    cublasHandle_t handle = nullptr;
    ck_bl(cublasCreate(&handle), "cublasCreate");

    const std::string tmp_path = out_path + ".tmp";
    std::FILE* f = std::fopen(tmp_path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot open for write: " + tmp_path);

    auto wr = [&](const void* p, size_t b) {
        if (b && std::fwrite(p, 1, b, f) != b) throw std::runtime_error("write failed: " + tmp_path);
    };
    const char magic[4] = {'V', 'H', 'C', '1'};
    const int32_t hdr[4] = {NCARD, vocab, n_in, K};
    wr(magic, 4);
    wr(hdr, sizeof(hdr));

    for (int c = 0; c < NCARD; ++c) {
        const int v0 = c * vstep, v1 = std::min(vocab, (c + 1) * vstep), n_out = v1 - v0;
        std::vector<uint16_t> h_X;
        dequant_norm_slice(out_w, v0, v1, h_X);

        std::vector<int> idx(n_out);
        std::iota(idx.begin(), idx.end(), 0);
        std::mt19937 rng(42u + (uint32_t) c * 1009u);
        std::shuffle(idx.begin(), idx.end(), rng);
        std::vector<uint16_t> h_C((size_t) K * n_in);
        for (int k = 0; k < K; ++k) {
            std::memcpy(h_C.data() + (size_t) k * n_in, h_X.data() + (size_t) idx[k] * n_in, (size_t) n_in * 2);
        }

        __half *d_X = nullptr, *d_C = nullptr;
        float *d_S = nullptr, *d_sums = nullptr, *d_best = nullptr;
        int *d_assign = nullptr, *d_counts = nullptr;
        ck(cudaMalloc(&d_X, (size_t) n_out * n_in * 2), "d_X");
        ck(cudaMalloc(&d_C, (size_t) K * n_in * 2), "d_C");
        ck(cudaMalloc(&d_S, (size_t) n_out * K * 4), "d_S");
        ck(cudaMalloc(&d_sums, (size_t) K * n_in * 4), "d_sums");
        ck(cudaMalloc(&d_best, (size_t) n_out * 4), "d_best");
        ck(cudaMalloc(&d_assign, (size_t) n_out * 4), "d_assign");
        ck(cudaMalloc(&d_counts, (size_t) K * 4), "d_counts");

        ck(cudaMemcpy(d_X, h_X.data(), (size_t) n_out * n_in * 2, cudaMemcpyHostToDevice), "H2D X");
        ck(cudaMemcpy(d_C, h_C.data(), (size_t) K * n_in * 2, cudaMemcpyHostToDevice), "H2D C");

        const float alpha = 1.0f, beta = 0.0f;
        for (int it = 0; it < ITERS; ++it) {
            ck_bl(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, n_out, n_in, &alpha,
                               d_C, CUDA_R_16F, n_in, d_X, CUDA_R_16F, n_in, &beta,
                               d_S, CUDA_R_32F, K, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "gemmEx");
            ck(cudaMemset(d_counts, 0, (size_t) K * 4), "zero counts");
            k_kmeans_assign<<<(n_out + 7) / 8, 256>>>(d_S, K, n_out, d_assign, d_best, d_counts);
            k_kmeans_repair_empty<<<1, 256>>>(n_out, K, d_assign, d_best, d_counts);
            ck(cudaMemset(d_sums, 0, (size_t) K * n_in * 4), "zero sums");
            k_kmeans_accum<<<(n_out + 7) / 8, 256>>>(d_X, d_assign, n_out, n_in, d_sums);
            k_kmeans_norm<<<K, 256>>>(d_sums, d_counts, K, n_in, d_C);
        }
        ck(cudaDeviceSynchronize(), "kmeans sync");

        std::vector<int32_t> h_assign(n_out);
        ck(cudaMemcpy(h_assign.data(), d_assign, (size_t) n_out * 4, cudaMemcpyDeviceToHost), "D2H assign");
        std::vector<float> h_best(n_out);
        ck(cudaMemcpy(h_best.data(), d_best, (size_t) n_out * 4, cudaMemcpyDeviceToHost), "D2H best");
        double mean_cos = 0.0;
        for (float v : h_best) mean_cos += v;
        mean_cos /= n_out;

        cudaFree(d_X); cudaFree(d_C); cudaFree(d_S); cudaFree(d_sums);
        cudaFree(d_best); cudaFree(d_assign); cudaFree(d_counts);

        std::vector<std::vector<int32_t>> buckets(K);
        for (int r = 0; r < n_out; ++r) {
            int k = h_assign[r];
            if (k < 0 || k >= K) throw std::runtime_error("bad cluster id");
            buckets[k].push_back(r);
        }
        std::vector<int32_t> perm_to_vocab(n_out), vocab_to_local(n_out, -1), cluster_off(K + 1, 0);
        int min_sz = n_out, max_sz = 0, pos = 0;
        for (int k = 0; k < K; ++k) {
            cluster_off[k] = pos;
            int sz = (int) buckets[k].size();
            min_sz = std::min(min_sz, sz);
            max_sz = std::max(max_sz, sz);
            for (int32_t u : buckets[k]) {
                perm_to_vocab[pos] = u;
                vocab_to_local[u] = pos;
                ++pos;
            }
        }
        cluster_off[K] = pos;
        if (pos != n_out) throw std::runtime_error("cluster count mismatch");
        for (int i = 0; i < n_out; ++i) {
            if (vocab_to_local[i] < 0 || perm_to_vocab[vocab_to_local[i]] != i)
                throw std::runtime_error("bijection check failed");
        }

        // Exact deterministic FP64 host L2-normalized centroids from final assignment + original h_X
        std::vector<uint16_t> cent_f16((size_t) K * n_in, 0);
        int nth = std::min<unsigned>(12u, std::max<unsigned>(1u, std::thread::hardware_concurrency()));
        std::vector<std::thread> th;
        for (int ti = 0; ti < nth; ++ti) {
            int k0 = (K * ti) / nth, k1 = (K * (ti + 1)) / nth;
            th.emplace_back([&, k0, k1] {
                std::vector<double> acc(n_in);
                for (int k = k0; k < k1; ++k) {
                    std::fill(acc.begin(), acc.end(), 0.0);
                    for (int32_t u : buckets[k]) {
                        const uint16_t* xr = h_X.data() + (size_t) u * n_in;
                        for (int i = 0; i < n_in; ++i) acc[i] += (double) f16_to_f32_h(xr[i]);
                    }
                    double ss = 0.0;
                    for (int i = 0; i < n_in; ++i) ss += acc[i] * acc[i];
                    uint16_t* dst = cent_f16.data() + (size_t) k * n_in;
                    if (ss > 0.0) {
                        double inv = 1.0 / std::sqrt(ss);
                        for (int i = 0; i < n_in; ++i) dst[i] = f32_to_f16_h((float) (acc[i] * inv));
                    } else {
                        dst[0] = f32_to_f16_h(1.0f);
                    }
                }
            });
        }
        for (auto& x : th) x.join();

        const int32_t chdr[3] = {v0, v1, K};
        wr(chdr, sizeof(chdr));
        wr(perm_to_vocab.data(), perm_to_vocab.size() * 4);
        wr(vocab_to_local.data(), vocab_to_local.size() * 4);
        wr(cluster_off.data(), cluster_off.size() * 4);
        wr(cent_f16.data(), cent_f16.size() * 2);

        std::printf("  card %d [%d, %d): K=%d, min_sz=%d, max_sz=%d, mean_cos=%.4f\n",
                    c, v0, v1, K, min_sz, max_sz, mean_cos);
    }

    if (std::fclose(f) != 0) throw std::runtime_error("fclose failed");
    if (std::rename(tmp_path.c_str(), out_path.c_str()) != 0) throw std::runtime_error("rename failed");
    cublasDestroy(handle);
    std::printf("build_qwen38_vhead_clusters: wrote %s\n", out_path.c_str());
    return 0;
}
