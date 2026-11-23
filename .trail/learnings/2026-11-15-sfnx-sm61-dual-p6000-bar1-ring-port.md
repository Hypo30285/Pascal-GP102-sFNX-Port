---
title: Port sFNX 2-card dense Qwen3.8-27B Q8_0 BAR1 P2P ring engine to Pascal sm_61 and Ivy Bridge
kind: decision
author: Tom & Antigravity (Claude + Argon)
date: 2026-11-15
touches:
  - /mnt/nvme_storage/ai_projects/sfnx-sm61/Makefile
  - /mnt/nvme_storage/ai_projects/sfnx-sm61/z420_build.sh
  - /mnt/nvme_storage/ai_projects/sfnx-sm61/src/sm61_compat.cuh
  - /mnt/nvme_storage/ai_projects/sfnx-sm61/src/gdn_prompt.cuh
  - /mnt/nvme_storage/ai_projects/sfnx-sm61/src/s27_prefill.cu
  - /mnt/nvme_storage/ai_projects/sfnx-sm61/src/model.cpp
  - /mnt/nvme_storage/ai_projects/sfnx-sm61/serve/sfnx-27b-z420.json
---

## Decided
- Ported `johnfink8/sfnx` (`FNX_MODEL=27b FNX_NCARD=2 ARCH=61`, commit `9808733` -> `eb638ab` on branch `sm61-port`) in `/mnt/nvme_storage/ai_projects/sfnx-sm61` using PolyStrata's conda CUDA 12.8 + gcc 13.4 toolchain and strict Ivy Bridge host flags (`-march=ivybridge -mtune=ivybridge -mavx -mf16c -mno-avx2 -mno-fma` + `-DGGML_BMI2=OFF`).
- Stored `Qwen3.8-27B-Q8_0.gguf` (29.05 GB, SHA-256 verified via RAM-staged downloader `/dev/shm` -> 8 TB HDD at 1.2 GB/s) and `mmproj-F16.gguf` (0.93 GB) in `/mnt/storage_8tb_linux/models/gguf/Qwen3.8-27B-Q8_0/`.
- Added `k_gdn_ch_rec61` (`GCB61=16`, `256` threads, `47,748 B <= 48 KB` shared memory, `REG:210 STACK:0`) in `src/gdn_prompt.cuh`, matching the row-by-row reference within `1.57e-7..1.04e-6` relative error across all 48 Gated DeltaNet layers (`FNX_PF_GDN_CHECK=1`).
- Added `k_pf_quant_q8` + `k_pf_q8s_dp4a<64, 4>` (`__launch_bounds__(256, 4)`, `REG:64 STACK:0 SHARED:4672`, transposed `sw[8][65]`/`sx[8][65]`, 128-bit `int4` loads) in `src/s27_prefill.cu`, boosting Dual P6000 prefill 3.2x–3.9x (`378 tok/s` at 512, `375 tok/s` at 2,048, `361 tok/s` at 8,192) over V100's `k_q8s_f16` + `cublasGemmEx(CUDA_R_16F)` fallback.
- Added `Up::finish_slice` in `src/model.cpp` to split `T_Q8_0` `q` and `d` on the host before upload, eliminating 58 GB of redundant H2D->D2H->cudaFree traffic and reducing model placement time from `45.2 s` to `28.4 s` (`-37%`).
- Verified 300/300 (100.0%) Top-1 logit agreement against `llama.cpp` (`a37cc46`) CUDA (`KL mean 1.28e-4`, `median 9.62e-7`), byte-for-byte `T=1 == T=5` window invariance, `83.5 tok/s` (`edit1`), `46.9 tok/s` (`code1`), `32.0 tok/s` (`prose2`), `21.71 tok/s` (`T=1` pure window vs `10.68–13.09 tok/s` `ik_llama.cpp`/`llama.cpp` `-sm layer`), `10/10` `serve/api_smoke.py`, and `6/6` `serve/image_test.py` (vision encoder isolated on headless GPU 0).

## Tried and ruled out
- Building `llama.cpp` (`a37cc46`) without `-DGGML_BMI2=OFF`: upstream CMake defaults `GGML_BMI2=ON` even when `GGML_NATIVE=OFF`, emitting Haswell `shlx` instructions in `libggml-cpu.so` that trigger `SIGILL` on the Xeon E5-2697 v2.
- Using V100's `k_q8s_f16` + `cublasGemmEx(CUDA_R_16F)` prefill path on Pascal `sm_61`: dequantizes 13.6 GB of Q8_0 weights per card to 27.2 GB of FP16 per chunk and falls back to slow FP32 SIMT (`95–117 tok/s` vs `361–378 tok/s` with `k_pf_q8s_dp4a`).
