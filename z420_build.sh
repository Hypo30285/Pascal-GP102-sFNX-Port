#!/usr/bin/env bash
# z420_build.sh — build sFNX for the HP Z420: Dual Quadro P6000 (sm_61), Xeon E5-2697 v2 (Ivy Bridge, no AVX2/FMA).
# Toolchain: PolyStrata's conda CUDA 12.8 + conda-forge gcc 13.4 (nvcc 12.8 refuses the system gcc 15).
#   ./z420_build.sh                       the 27B engine and its probes (FNX_MODEL=27b FNX_NCARD=2 ARCH=61)
#   ./z420_build.sh bin/s27_ring_test     any Makefile target(s)
#   ./z420_build.sh llama                 llama.cpp a37cc46's libraries for sm_61 (vision encoder, ref tools)
set -euo pipefail
cd "$(dirname "$0")"
T=/mnt/nvme_storage/ai_projects/polystrata/.toolchain
CUOBJ=$T/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump
export PATH="$T/bin:$PATH"
export LD_LIBRARY_PATH="$T/lib64:$T/lib:${LD_LIBRARY_PATH:-}"
MK=(make ARCH=61 FNX_MODEL=27b FNX_NCARD=2 CUDA="$T" HOSTCXX="$T/bin/g++" HOSTCC="$T/bin/gcc" CUOBJDUMP="$CUOBJ" HOST_LDFLAGS="-Wl,-rpath-link,$PWD/llama.cpp/build/bin -Wl,--allow-shlib-undefined")
JOBS=${JOBS:-12}   # 12 physical cores
if [ "${1:-}" = llama ]; then
    shift
    exec nice -n 10 "${MK[@]}" -j"$JOBS" "$PWD/llama.cpp/build/bin/libmtmd.so" "$@"
fi
TARGETS=("$@")
[ ${#TARGETS[@]} -gt 0 ] || TARGETS=(bin/s27_bw_probe bin/s27_ring_test bin/s27_probe bin/s27_run bin/sfnx)
exec nice -n 10 "${MK[@]}" -j"$JOBS" "${TARGETS[@]}"
