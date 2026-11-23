// sm61_compat.cuh — Pascal GP102 (sm_61, Quadro P6000) shims for sFNX (Z420 sm61-port branch).
//
// Upstream sFNX targets sm_70+. Three sm_70-only features appear in code that every build compiles:
//   * __nanosleep (sm_70+)              -> fnx_nanosleep: a clock64() back-off spin on sm_61
//   * __match_any_sync (sm_70+)         -> fnx_match_any: a shfl/ballot loop on sm_61 (warp-uniform, full mask)
//   * wmma / mma.sync (tensor cores)    -> FNX_TC == 0 on sm_61: those kernels' bodies become __trap(). They are the
//                                          Flash-Next prompt path's (k_pf_attn, k_idx_score_tc) and its routed experts
//                                          (k_moe_mma); the 27B (FNX_MODEL=27b) launches none of them. A launch on
//                                          sm_61 traps loudly instead of returning wrong numbers.
#pragma once

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#define FNX_TC 0
#else
#define FNX_TC 1
#endif

__device__ __forceinline__ void fnx_nanosleep(unsigned ns) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __nanosleep(ns);
#else
    // ~1.5 cycles a ns at P6000 boost (1.5-1.7 GHz); only a back-off, so the exact length does not matter
    const long long t0 = clock64();
    const long long cycles = (long long) ns * 3 / 2;
    while (clock64() - t0 < cycles) {
    }
#endif
}

// The lanes of `mask` whose v equals this lane's v. On sm_61 every lane in mask must call it together (the callers
// pass the full warp, with out-of-range lanes holding a sentinel value, so that holds).
__device__ __forceinline__ unsigned fnx_match_any(unsigned mask, unsigned v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    return __match_any_sync(mask, v);
#else
    unsigned left = mask, mine = 0;
    while (left) {   // warp-uniform: `left` is built from ballots only
        const int leader = __ffs(left) - 1;
        const unsigned lv = __shfl_sync(mask, v, leader);
        const unsigned same = __ballot_sync(mask, v == lv);
        if (v == lv) mine = same;
        left &= ~same;
    }
    return mine;
#endif
}

// System-scope acquire load of a 32-bit flag written by the host (pinned UVA memory, src/engine.cu).
__device__ __forceinline__ unsigned fnx_ld_acquire_sys_u32(const volatile unsigned* flag) {
    unsigned f;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.sys.u32 %0, [%1];" : "=r"(f) : "l"(flag) : "memory");
#else
    asm volatile("ld.global.cv.u32 %0, [%1];\n\tmembar.sys;" : "=r"(f) : "l"(flag) : "memory");
#endif
    return f;
}
