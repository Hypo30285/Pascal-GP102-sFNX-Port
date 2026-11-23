---
title: LM Head Architecture Comparison on Dual P6000 sFNX — Lossless 6+2 Interval Split-Plane (Verifier) vs. Kindi Spherical Fast-Head (MTP Drafter Only)
kind: decision
author: Tom Burnsed (@Hypo30285) & Gemini
date: 2026-11-19
touches:
  - src/lm_head_6p2_sm61.cuh
  - src/dense.cuh
  - src/s27_engine.cu
  - src/engine.cu
  - src/model.cpp
  - src/session.cpp
---

## Decided

- Use 6-bit + 2-bit interval-bounded split-plane Q8S exclusively on the main target verifier LM head (100% bitwise lossless for greedy K=1 and top_k <= 64, force-including T speculative draft token IDs).
- Restrict Kindi-style 1,024-cluster spherical shortlisting strictly to sequential NextN MTP / DSpark draft steps (blk.64 / blk.48) over the shared cluster-permuted 6-bit plane, reducing DRAFT_ROW_MS from 2.15 ms toward ~0.75-0.85 ms with zero target distribution drift.

## Tried and ruled out

- Running Kindi spherical fast-head on the main target verifier LM head (0.6% greedy token error fails Gate 5 SHA256 identity and biases temp > 0 speculative rejection sampling).
