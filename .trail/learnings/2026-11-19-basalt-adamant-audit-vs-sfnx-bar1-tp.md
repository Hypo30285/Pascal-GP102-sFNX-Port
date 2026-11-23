---
title: Basalt (sm_120) & Adamant (sm_89/86) Audit vs. Pascal sFNX BAR1 TP — Disqualification & Lossless Interval Bit-Plane Q8S LM Head Takeaway
kind: decision
author: Tom Burnsed (@Hypo30285) & Gemini
date: 2026-11-19
touches:
  - src/dense.cuh
  - src/engine.cu
  - src/model.cpp
  - src/kernel_router.hpp
---

## Decided

- Keep Dual Quadro P6000 (GP102, sm_61, 48 GB GDDR5X) + Xeon E5-2697 v2 workloads on Pascal-GP102-sFNX-Port (k_ring3 intra-graph 64-bit volatile BAR1 P2P Tensor Parallelism) and PolyStrata (Arm10 pcie=0.05, tail_skip=0).
- Prototype two-stage lossless 6-bit + 2-bit interval-bounded Q8S LM head (src/lm_head_6p2_sm61.cuh) to cut ~23.5% of 248,320-row LM-head GDDR5X reads with 0-ULP greedy SHA256 identity.
- Pre-stage Qwen3.8-Flash-Next 4-stream BF16 hyper-connection matrices as group-32 int8 + fp16 scale (hc8, -0.0002 dKLD lossless) in sm61-fnx-moe src/model.cpp while keeping Q8S dense trunk and Group-64 int8 QSA KV cache.

## Tried and ruled out

- jesdga95/basalt and nixcat-dev/adamant on Z420 or Bailey's rig (CUDA 13 drops sm_61; host requires AVX2; Peer Expert Tier idles GPU 1 on dense trunks; 100%-CPU miss path lacks pcie=0.05 overlap).
- STRATA_ROUTE_TAIL_SKIP=7 (silent expert dropping degrades UD-Q4 KLD from 0.056 to 0.086..0.115).
