---
title: sFNX Pascal GP102 (sm_61 Dual Quadro P6000) + Ivy Bridge Port & Optimizations for Qwen3.8-27B-Q8_0 (50.4 tok/s Decode, 378 tok/s dp4a Prefill)
kind: decision
author: Tom
date: 2026-11-16
touches:
  - Makefile
  - z420_build.sh
  - src/sm61_compat.cuh
  - src/gdn_prompt.cuh
  - src/s27_prefill.cu
  - src/s27_attn.cuh
  - src/s27_engine.cu
  - src/s27_impl.hpp
  - src/dense.cuh
  - src/kernels.cu
  - src/session.cpp
  - src/batch_plan.hpp
  - src/model.cpp
  - src/gguf.cpp
  - serve/sfnx-27b-z420.json
---

## Decided

- **Pure `johnfink8/sfnx` Tensor-Parallel Port to Pascal GP102 (`ARCH=61`, `FNX_MODEL=27b`, `FNX_NCARD=2`) on HP Z420 (`https://github.com/Hypo30285/sfnx`, branch `sm61-port`):**
  - Ported upstream `johnfink8/sfnx` (`commit 9808733`) directly to Dual NVIDIA Quadro P6000 24 GB GDDR5X (`GP102`, `sm_61`, `PCIe 3.0 x16` Slot 2 + Slot 5) + Intel Xeon E5-2697 v2 (`12C/24T` Ivy Bridge `AVX + F16C`, `128 GB` DDR3-1866 ECC RAM) running `Qwen3.8-27B-Q8_0.gguf` (`29.05 GB`, `100%` `Q8_0` weights + `blk.64` NextN MTP head + `mmproj-F16.gguf` vision encoder, `65,536` context across `4` lanes).
  - Verified that 64-bit volatile P2P BAR1 ring stores (`k_ring3`) work across the full 24 GB address space on both Quadro P6000s despite `256 MiB` BAR1 reporting (`20 rounds x 4,194,304 words`, `12 GiB` offset, `bad=0`, `timeouts=0`, `7.03 / 6.26 GB/s` sustained P2P throughput).
- **Equalized `150W / 150W` Dual-GPU Power Limits (`+25.6%` Decode Gain):**
  - Updated `/etc/systemd/system/devvault-gpu-governor.service` and `/etc/systemd/system/z420-gpu-power-cap.service` so both Quadro P6000s run at a symmetrical `150 W` cap (`1392-1506 MHz` under load, `84 C / 87 C` steady-state), eliminating the `135 W` (`1101 MHz`) straggler throttle on GPU 1 (`Slot 2`) that was stalling GPU 0 in every `k_ring3` `r3_take` spin loop (`128` all-reduces per verify window) and lifting decode from `32.0 tok/s` to `40.2 tok/s`.
- **Pascal `sm_61` + Ivy Bridge Host/Device Build & Kernel Architecture:**
  1. **`src/sm61_compat.cuh` & `Makefile`:** Emulated `__nanosleep` (`clock64()` poll on `sm_61`) and `__match_any_sync` (ballot + `__ffs` leader loop), added `ARCH=61` and `-march=ivybridge -mtune=ivybridge -mavx -mf16c -mno-avx2 -mno-fma`, and enforced `-DGGML_BMI2=OFF` for `llama.cpp` (`a37cc46`) so `libggml-cpu.so` never emits Haswell BMI2 (`shlx`/`sarx`/`mulx`) instructions.
  2. **`k_gdn_ch_rec61` (`src/gdn_prompt.cuh`):** Retiled the Gated DeltaNet prompt chunk recurrence from `73 KB` (`sm_70`) to `48 KB` shared memory (`256` threads, `RA=1, SR=2`, `REG:210 STACK:0`), preserving the exact sequential `k` reduction order (`1.57e-7..1.04e-6` max relative error vs row-by-row reference across all 48 GDN layers).
  3. **`k_pf_q8s_dp4a<BN, TC>` (`src/s27_prefill.cu`):** Implemented native GP102 `__dp4a` INT8 tiled prefill GEMM (`BN in {16, 32, 48, 64}`, `REG:53..64 STACK:0`, padded shared memory `sw[8][BM+1]` / `sx[8][BN+1]`) directly over the split `Q8_0` weight layout, achieving `378 / 375 / 361 tok/s` prefill at `512 / 2,048 / 8,192` tokens and cutting short 58-token chat turn prefill in half (`0.84 s -> 0.44 s`).
  4. **6x GQA KV-Cache Sharing & Fused `q8_1` Split-Twin Quantization (`src/s27_attn.cuh`, `src/s27_engine.cu`):** Restructured `k_s27_attn` to `dim3(S27_ASPLIT, T, S27_NKV)` (`REG:96 STACK:0 SHARED:6144`) so all 6 query heads per KV head share a single `uint4` paged KV load (`6x` less KV traffic); fused `q8_1` split-twin quantization directly into `k_s27_merge` (`REG:32 STACK:0`), `k_norm_cat` (`REG:26 STACK:0`), and `k_add_norm` (`REG:22 STACK:0`, including `last_only=true` for the MTP head), eliminating dead `FP32` global stores, unused `b.att` / `b.mtp_cat` buffers, redundant `cudaMemcpyAsync` D2D copies, and separate `quantize_q8_1` launches.
  5. **4-Row Register Weight Sharing (`Q8S_R=4` for `T=3, 4` in `src/dense.cuh`):** Enabled `TC=4` register weight reuse across 3-row and 4-row speculative verify windows (`T=3: 56.97 -> 45.11 ms`, `T=4: 64.65 -> 50.08 ms`, `-21%` to `-22%`), reaching `376.2 GB/s` per card (`93.8%` of raw GP102 DRAM read ceiling) across all 257 dependent `Q8_0` matrices in `bin/s27_probe`.
  6. **Calibrated NextN MTP Cost Model (`src/session.cpp`, `src/batch_plan.hpp`):** Replaced the Flash-Next Q4 MoE default (`ROUND_MS=16.5, DRAFT_ROW_MS=2.1, DRAFT_CALL_MS=2.0`, `8.25:1` ratio) with the measured Dual P6000 27B Q8_0 cost constants (`ROUND_MS=43.3, DRAFT_ROW_MS=2.15, DRAFT_CALL_MS=1.35`, `32.1:1` ratio), lifting average draft chain length (`2.96 -> 3.50-4.06` drafts/round), draft acceptance (`49.0% -> 51.4%` at `temp=0.6`), and decode throughput to **`47.8 tok/s` (`temp=0.6`)** and **`50.4 tok/s` (`temp=0.0` greedy)**, while sustaining **`32.3-34.7 tok/s`** out to **`53,752` total context tokens**.
  7. **Blocking CUDA Event Sync & Page-Cached GGUF Loader (`src/s27_engine.cu`, `src/engine_common.cu`, `src/gguf.cpp`, `src/model.cpp`):** Enabled `cudaEventBlockingSync` on `ev0`/`ev1` (dropping host CPU spin from `100%` of a Xeon core to `< 0.5%`), disabled `O_DIRECT` for `FNX_27B` so the 29 GB GGUF stays warm in the 128 GB Linux page cache (`2.0 s` read), and split `Q8_0` host slices inside the 4-thread `Stager` (`28.1 s` total model placement).

## Tried and ruled out

- **Using `cuBLAS` FP16 (`cublasGemmEx(CUDA_R_16F)`) + `k_q8s_f16` dequantization for Pascal `sm_61` prefill:** On GP102 (which lacks Volta `sm_70` Tensor Cores and executes FP16 at 1:64 rate), dequantizing `Q8_0` to `FP16` and calling `cublasGemmEx` falls back to slow FP32 SIMT paths; native `__dp4a` INT8 tiled GEMM (`k_pf_q8s_dp4a`) directly on the split `Q8_0` weights is required for `378 tok/s` prefill.
- **128-thread `k_gdn_ch_rec61` (`RA=2, SR=2`) or un-padded `sw[BM][8]` shared memory in `k_pf_q8s_dp4a`:** 128 threads spilled `264 B` to local stack (`REG:255 STACK:264`), while `sw[BM][8]` caused 4-way shared-memory bank conflicts (`REG:111`, `143 tok/s`). Retiling `k_gdn_ch_rec61` to `256` threads (`REG:210 STACK:0`) and transposing `sw[8][BM+1]` / `sx[8][BN+1]` (`REG:64 STACK:0`, `378 tok/s`) eliminated all spills and bank conflicts.
- **Runtime-bounded `for (int hh = 0; hh < nh; ++hh)` epilogue loop in `k_s27_attn`:** Prevented `ptxas` from unrolling register-indexed `acc[hh][j]`, `m[hh]`, `l[hh]`, spilling `240 B` (`REG:105 STACK:240`) to local memory. Using `#pragma unroll for (int hh = 0; hh < S27_QPK; ++hh) { if (hh >= nh) break; ... }` kept all arrays in registers (`REG:96 STACK:0`).
- **4-thread `O_DIRECT` `pread` on the 8 TB Seagate HDD (`/mnt/storage_8tb_linux`) for `FNX_27B`:** Forced physical HDD head thrashing on every engine restart (`> 90 s` timeout) and bypassed the 128 GB DDR3-1866 RAM page cache. Disabling `O_DIRECT` for `FNX_27B` keeps the 29 GB GGUF cached in RAM (`2.0 s` read).
- **Keeping Flash-Next's `ROUND_MS=16.5` / `DRAFT_CALL_MS=2.0` (`8.25:1`) cost model on dense `Qwen3.8-27B-Q8_0`:** Caused `Session::generate` to under-draft (`2.96` drafts/round, `209` full 64-layer trunk passes per 512 tokens). Calibrating to the true `32.1:1` ratio (`ROUND_MS=43.3, DRAFT_CALL_MS=1.35`) cut trunk passes by `12.4%` (`183` rounds at `temp=0.6`, `178` rounds at `temp=0.0`).
