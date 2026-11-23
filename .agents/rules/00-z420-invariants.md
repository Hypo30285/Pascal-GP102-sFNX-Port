---
trigger: always_on
description: "Authoritative HP Z420 hardware, Ivy Bridge-EP ISA, Dual Quadro P6000 sm_61 SASS, 3-engine routing, sfnx-sm61 codebase map, and settled Trail mistake invariants."
---

# Z420 ("The Z") Frozen Hardware, Compiler & Trail Invariants

## 0. Zero-Redundant-Lookup & Upstream Override Directive
All hardware specs, ISA flags, SASS limits, binary paths, and disqualifications in this file are verified ground truth for **The Z**.
- **Z420 Override Over Upstream `AGENTS.md`:** Upstream `sfnx-sm61/AGENTS.md` documents the original `3x V100-32GB (sm_70, CUDA 12.9, gcc 14)` box (`make -j32`). **Ignore those `sm_70` / V100 target assumptions on this workstation:** build for **Dual Quadro P6000 (`ARCH=61`, `FNX_NCARD=2`) + Ivy Bridge (`-j 24`)** via `./z420_build.sh` or explicit `sm_61` / `ivybridge` flags.
- **DO NOT** run `view_file`, `grep`, `lscpu`, `nvidia-smi`, or `CMakeLists.txt` inspections merely to verify CPU ISA, GPU count, CUDA version, or baseline flags.
- Query `devvault_search` / `devvault_skeleton` / `devvault_mistake_check` at the turn tail **only** for symbol-level details, benchmark rows, or file-specific `.trail/learnings/`.

## 1. Workstation Hardware & Compiler Laws

### Host CPU: Intel Xeon E5-2697 v2 (Ivy Bridge-EP, 12C / 24T, 128 GB DDR3-1866 ECC RDIMM)
- **Allowed Host SIMD:** `SSE4.2`, `AVX 1.0`, `F16C` (`_cvtss_sh`, `_cvtsh_ss`).
- **BANNED Host Instructions (Immediate `SIGILL` Trap):** `AVX2` (`vpaddd`, `vperm2i128`), `FMA` (`vfmadd*`), `BMI1`, `BMI2` (`shlx`, `sarx`, `shrx`, `mulx`, `bzhi`, `rorx`, `pdep`, `pext`), `AVX-512`.
- **Mandatory Host C/C++17 Flags:** `-O3 -march=ivybridge -mtune=ivybridge -mavx -msse4.2 -mf16c -mno-avx2 -mno-fma -DGGML_BMI2=OFF` (with `-j 24` for builds; pin compute worker pools to `--threads 12` physical cores `CPU 0..11`).

### Dual GPUs: 2x NVIDIA Quadro P6000 (Pascal `GP102`, `sm_61`, 24 GB GDDR5X x 2 = 48 GB Total)
- **Topology:** `GPU 0` (`0000:04:00.0`, Slot 5, `PCIe 3.0 x16`, headless compute `~24,569 MiB` free); `GPU 1` (`0000:05:00.0`, Slot 2, `PCIe 3.0 x16`, display `~23,890 MiB` free, keep `>= 2,048 MiB` reserve).
- **Toolchain & Driver:** NVIDIA Driver `580 LTS`, **CUDA 12.8** (`nvcc` pinned at `/mnt/samsung_storage/toolchains/cuda128/bin/nvcc`, `-arch=sm_61 -Xptxas="-v"`). **NEVER** compile kernels with CUDA 13.0+ (drops `sm_61`).
- **Power & Bus:** Symmetric `150W / 150W` power caps (`nvidia-smi -pl 150`); `iommu=pt`; intra-graph 64-bit volatile BAR1 P2P (`k_ring3` spin-wait). Never run asymmetric caps (causes `k_ring3` spin stalls).
- **Allowed Device Compute:** `__dp4a` (`IDP4A` 4-way `int8` dot product with `int32` accumulate) + native `FP32` (`FFMA`).
- **BANNED Device Instructions & Patterns:**
  - **NO FP16 Arithmetic (`half2` / `HFMA2` / `HMUL2` / `HADD2`):** `GP102` runs FP16 at a 1:64 penalty. Convert FP16 storage/scales to `float` in registers (`__half2float`).
  - **NO TensorCore (`HMMA` / `mma.sync`), `cp.async`, or Runtime `BF16` (`__nv_bfloat162`):** Transcode 2D `BF16` hyper-connection matrices at load time into group-32 `Q8S` (`hc8`: `int8_t q[32]` + `fp16 d`). Keep 1D RMSNorms, gating biases, and DSpine code-bridges (`rank=512, code_size=1024`) in native `FP32`.
- **PTXAS Resource Ceilings:** `LOCAL_MEM / STACK: 0` (zero register spill to GDDR5X); static shared memory `<= 48 KB` per CTA (`<= 4 KB` preferred on high-occupancy LM-head CTAs).

## 2. Three-Engine Routing & Storage Laws
- **In-VRAM `sfnx-sm61` Targets (`<= 44.5 GiB`):** `Qwen3.8-27B-Q8_0` (`29.05 GB`) and `sfnx` ports run on `/home/tom/projects/sfnx-sm61` (`FNX_MODEL=27b FNX_NCARD=2 ARCH=61`).
- **In-VRAM `ik_llama.cpp` (`<= 44.5 GiB`):** Custom `sm_61` + `ivybridge` build (`/home/tom/projects/ik_llama.cpp/build/bin/llama-server`, `-ngl 999 -sm layer -fmoe -fa -gr`; isolate single-GPU runs with `CUDA_VISIBLE_DEVICES=0` or `1`; persistent KV slots at `/mnt/samsung_storage/devvault/kv_slots`).
- **Out-of-VRAM MoE Models (`> 44.5 GiB`):** **EXCLUSIVELY use `PolyStrata` (`strata-poly`)** across Dual P6000 + `128 GB` DDR3-1866 RAM (`PeerExpertCache` + `--resident-experts` + Harmonic PLL). **HARD BAN:** Never use `ik_llama.cpp` or `llama.cpp` (`-ncmoe` / `-ot`) on any `> 44.5 GiB` model.
- **Even-Bitwidth Linear Quants Only:** Use `Q8S`, `6+2` split-plane `Q8_0`, `Q4_K` (`Q4_K_M`/`UD-Q4_K_XL`), `Q6_K`, or `Q8_0`. Avoid `IQ*` LUT quants and odd-bit `Q3_K`/`Q5_K`.
- **Storage & Scratch:** Route RSI build/test scratch to `/dev/shm/rsi_scratch` `tmpfs`; route all multi-GB downloads exclusively to `/mnt/storage_8tb_linux/downloads` (`HF_HOME=/mnt/storage_8tb_linux/downloads/hf_cache`); never touch `/mnt/nvme_storage` (dead HP EX950).
- **DevVault Native CLI/MCP:** Query `/mnt/samsung_storage/devvault/bin/devvault` (Stdio MCP `devvault` in `mcp_config.json`; `:8088` CPU embedder). Never invoke `:8088` inside SQLite `BEGIN IMMEDIATE TRANSACTION`.

## 3. `sfnx-sm61` Repository, Branch & Isolation Map (`01_GRAFT_CODE_MAP` & `03_TEAM_OVERLAP_BOARD`)
- **Local Repo:** `/home/tom/projects/sfnx-sm61` (`/mnt/storage_8tb_linux/python_projects/sfnx-sm61`).
  - **`sm61-port`** (`HEAD @ 05bc619`, remote `hypo/sm61-port`): **Protected production baseline** (`50.4 tok/s` greedy / `47.8 tok/s` `temp=0.6`, `378 tok/s` `__dp4a` prefill, `93.8%` GP102 DRAM ceiling). **NEVER** commit experimental edits directly to `sm61-port`.
  - **`sm61-lmhead-6p2`:** Lossless `6+2` interval split-plane `Q8S` main verifier LM head + NextN MTP 1,024-cluster spherical drafter shortlist.
  - **`sm61-fnx-moe`:** `Qwen3.8-Flash-Next` MoE runtime + load-time `hc8` 4-stream hyper-connection transcoding + Group-64 `int8` QSA KV cache.
  - **`sm61-dspine-kindi061`:** DSpine code-bridge (`FP32` preserved) + 17-slot ReplaySSM experiments.
- **Key Source Files:** `src/dense.cuh`, `src/lm_head_6p2_sm61.cuh`, `src/s27_engine.cu`, `src/s27_attn.cuh`, `src/s27_prefill.cu`, `src/s27_ring3.cuh`, `src/engine.cu`, `src/model.cpp`, `src/gguf.cpp`, `src/session.cpp`, `src/batch_plan.hpp`, `src/kernel_router.hpp`, `z420_build.sh`.
- **Teammate Isolation:** Bailey's `tom_and_bailey_coding` repo is decoupled from `sfnx-sm61` kernel sources. Keep `.agents/` and `.trail/` excluded via `.git/info/exclude` so upstream PR branches stay publication-ready.

## 4. Settled Disqualifications ("Tried and Ruled Out — Never Re-Propose")
1. **`Basalt` (`jesdga95/basalt`, `sm_120`) & `Adamant` (`nixcat-dev/adamant`, `sm_89/86`):** Require CUDA 13 (no `sm_61`) and host `AVX2`; lack Dual-P6000 `Arm10 pcie=0.05` overlap.
2. **`STRATA_ROUTE_TAIL_SKIP=7` (Expert Dropping):** Degrades KLD (`0.056 -> 0.086..0.115`). Keep `STRATA_ROUTE_TAIL_SKIP=0`.
3. **Kindi Spherical Fast-Head on Main Target Verifier:** Causes `0.6%` greedy error (fails Gate 4 SHA256 identity) and biases `temp > 0` rejection sampling. Restrict spherical shortlisting **strictly** to NextN MTP / DSpark draft steps (`blk.64` / `blk.48`); use **Lossless `6+2` Interval Split-Plane `Q8S`** (`26 B` Hi + `8 B` Lo = `34 B`, force-including `T` draft token IDs) on the main verifier.
4. **`cuBLAS` FP16 (`cublasGemmEx`) or Un-Padded `sw[BM][8]` on `sm_61`:** Hits 1:64 FP16 penalty and 4-way shared-memory bank conflicts. Use native `__dp4a` tiled GEMM with padded `sw[8][BM+1]` / `sx[8][BN+1]`.
5. **Runtime-Bounded Epilogue Loops & Unblocked Spin-Wait:** Keep `#pragma unroll` with early `break` in `k_s27_attn` (`STACK: 0`), `cudaEventBlockingSync` on `ev0`/`ev1`, and disable `O_DIRECT` for `FNX_27B` (`2.0 s` page-cached read).

## 5. Mandatory 5-Gate Promotion Checklist (Before Updating `src/kernel_router.hpp`)
1. **Gate 1 (ISA & SASS):** `0` `AVX2`/`FMA`/`BMI2` instructions in `objdump -d`; `0` `HFMA2`/`HMUL2`/`HADD2`/`HMMA` and `>0` `IDP4A` in `cuobjdump --dump-sass`; `STACK: 0`.
2. **Gate 2 (Numerical Parity):** `bin/s27_probe` 0-ULP reference parity against `Q8_0` baseline.
3. **Gate 3 (Dual-GPU Power & Throughput):** Summed GPU 0 + GPU 1 `nvidia-smi dmon` J/tok and tok/s improvement at `150W / 150W`.
4. **Gate 4 (Deterministic Identity):** Exact 512-token `temp=0.0` greedy SHA256 match against `sm61-port`.
5. **Gate 5 (End-to-End Quality):** Lighteval suite + appended row in Z420 PolyStrata Eval Tracker.
