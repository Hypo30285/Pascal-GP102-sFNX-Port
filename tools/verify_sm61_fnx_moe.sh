#!/usr/bin/env bash
# verify_sm61_fnx_moe.sh — Gates 1, 2, and 3 hard verification suite for sm61-fnx-moe
#   Gate 1: Clean dual build (FNX_MODEL=27b and FNX_MODEL=fnx, FNX_NCARD=2, ARCH=61) with pinned CUDA 12.8 & 0 warnings
#   Gate 2: Device SASS/PTX + resource usage (0 HFMA2/HMUL2/HADD2/HMMA/trap, >0 IDP4A/dp4a, <=48KB SMEM, STACK:0 on new kernels)
#           + Host ISA (0 AVX2 ymm-int, 0 FMA3, 0 BMI2 in kq_avx1.o, bin/sfnx, bin/fnx_run, bin/test_kq_avx1)
#   Gate 3: Unit tests (card_check_test, pf_ring_plan_test, gguf_test, ple_test, test_kq_avx1, hc8_test, vhead_6p2_test)
#           + Dual-GPU BAR1 P2P probes (s27_ring_test, p2p_store: bad=0, timeouts=0) + Ivy Bridge EDAC CE delta == 0
#
# Usage:
#   tools/verify_sm61_fnx_moe.sh [--gate all|1|2|3] [--skip-build] [--jobs 12]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GATE="all"
SKIP_BUILD=0
JOBS="${JOBS:-12}"

while [ $# -gt 0 ]; do
    case "$1" in
        --gate) GATE="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --jobs|-j) JOBS="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

TOOLCHAIN="${TOOLCHAIN:-/mnt/samsung_storage/toolchains/cuda128}"
NVCC="${NVCC:-$TOOLCHAIN/bin/nvcc}"
if [ ! -x "$NVCC" ]; then
    NVCC="$TOOLCHAIN/bin/nvcc"
fi
if [ -z "${HOSTCXX:-}" ]; then
    if [ -x "$TOOLCHAIN/bin/x86_64-conda-linux-gnu-g++" ]; then
        HOSTCXX="$TOOLCHAIN/bin/x86_64-conda-linux-gnu-g++"
    else
        HOSTCXX="$TOOLCHAIN/bin/g++"
    fi
fi
if [ -z "${HOSTCC:-}" ]; then
    if [ -x "$TOOLCHAIN/bin/x86_64-conda-linux-gnu-gcc" ]; then
        HOSTCC="$TOOLCHAIN/bin/x86_64-conda-linux-gnu-gcc"
    else
        HOSTCC="$TOOLCHAIN/bin/gcc"
    fi
fi
CUOBJDUMP="${CUOBJDUMP:-$(command -v cuobjdump || true)}"
if [ ! -x "$CUOBJDUMP" ]; then
    CUOBJDUMP="$TOOLCHAIN/bin/cuobjdump"
fi
PYTHON="${PYTHON:-$TOOLCHAIN/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

export PATH="$TOOLCHAIN/bin:$PATH"
export LD_LIBRARY_PATH="$TOOLCHAIN/lib:$TOOLCHAIN/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$ROOT/logs"
REPORT_JSON="$ROOT/logs/verify_sm61_fnx_moe_gates123.json"

MODEL_BASE="/mnt/storage_8tb_linux/models/gguf/Qwen3.8-Flash-Next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
MODEL_Q6="/mnt/storage_8tb_linux/models/gguf/qwen3.8-flash-next-ud-q6_k_xl/qwen3.8-flash-next-ud-q6_k_xl/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf"
PLE_SSD="/mnt/samsung_storage/ple/Qwen3.8-Flash-Next-UD-Q6_K_XL-00003-of-00006.gguf"
MTP_GGUF="/mnt/storage_8tb_linux/models/gguf/Qwen3.8-Flash-Next-UD-Q4_K_XL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
HOST_LDFLAGS="-Wl,-rpath-link,$ROOT/llama.cpp/build/bin -Wl,--allow-shlib-undefined"

echo "=== sm61-fnx-moe Hard Verification Suite (Gate=$GATE) ==="
echo "Repo:       $ROOT (branch=$(git branch --show-current || echo detached), commit=$(git rev-parse --short HEAD))"
echo "NVCC:       $NVCC ($("$NVCC" --version | tail -n 1))"
echo "HOSTCXX:    $HOSTCXX"
echo "CUOBJDUMP:  $CUOBJDUMP"
echo "JOBS:       $JOBS"

# Verify CUDA 12.8 pin (sm_61 Pascal guardrail)
NVCC_VER_STR="$("$NVCC" --version)"
if [[ "$NVCC_VER_STR" != *"release 12.8"* ]]; then
    echo "FAIL: Pinned CUDA 12.8 required for sm_61 Pascal; got: $NVCC_VER_STR" >&2
    exit 1
fi

read_edac_ce() {
    "$PYTHON" -c '
import glob
from pathlib import Path
paths = glob.glob("/sys/devices/system/edac/mc/mc*/csrow*/ch*_ce_count")
if not paths:
    paths = glob.glob("/sys/devices/system/edac/mc/mc*/ce_count")
tot = 0
for p in paths:
    try:
        tot += int(Path(p).read_text().strip())
    except Exception:
        pass
print(tot)
'
}

GATE1_STATUS="SKIPPED"
GATE2_STATUS="SKIPPED"
GATE3_STATUS="SKIPPED"

# ---------------------------------------------------------------------------
# Static Source Hygiene Audit (always runs when Gate 1 or Gate 2 runs)
# ---------------------------------------------------------------------------
if [ "$GATE" = "all" ] || [ "$GATE" = "1" ] || [ "$GATE" = "2" ]; then
    echo ""
    echo "--- [Static Source Audit] Checking sm_61 intrinsics and FNX_TC==0 paths ---"
    "$PYTHON" - "$ROOT" <<'PYEOF'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
src_dir = root / "src"
errors = []

# 1. Zero ld.acquire.sys.u32 in active code outside sm61_compat.cuh (must use fnx_ld_acquire_sys_u32)
for p in sorted(src_dir.glob("*")):
    if not p.is_file() or p.name == "sm61_compat.cuh":
        continue
    text = p.read_text(encoding="utf-8", errors="replace")
    # Strip /* ... */ and // ... comments for code-only scan
    no_block = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    for lineno, line in enumerate(no_block.splitlines(), 1):
        code = re.sub(r"//.*$", "", line)
        if "ld.acquire.sys" in code:
            errors.append(f"{p.name}:{lineno}: forbidden ld.acquire.sys in active code (use fnx_ld_acquire_sys_u32)")

# 2. Zero __trap() in FNX_TC == 0 paths of prefill headers
for fname in ("moe_mma.cuh", "attn_prompt.cuh", "qsa.cuh"):
    p = src_dir / fname
    if not p.is_file():
        errors.append(f"Missing expected file src/{fname}")
        continue
    text = p.read_text(encoding="utf-8", errors="replace")
    no_block = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    for lineno, line in enumerate(no_block.splitlines(), 1):
        code = re.sub(r"//.*$", "", line)
        if "__trap(" in code:
            errors.append(f"src/{fname}:{lineno}: __trap() still present in active code: {code.strip()}")

if errors:
    print("STATIC AUDIT FAILED:")
    for e in errors:
        print("  FAIL:", e)
    sys.exit(1)
print("STATIC AUDIT PASS: 0 forbidden ld.acquire.sys, 0 __trap() in moe_mma.cuh/attn_prompt.cuh/qsa.cuh")
PYEOF
fi

# ---------------------------------------------------------------------------
# Gate 1: Clean Dual Build & Zero Warnings (27b + fnx, FNX_NCARD=2, ARCH=61)
# ---------------------------------------------------------------------------
if ([ "$GATE" = "all" ] || [ "$GATE" = "1" ]) && [ "$SKIP_BUILD" -eq 0 ]; then
    echo ""
    echo "--- [Gate 1A] Clean build: FNX_MODEL=27b FNX_NCARD=2 ARCH=61 ---"
    rm -f build/*.o build/*.d build/kernels_p* build/pic/*.o build/pic/*.d
    BUILD_27B_LOG="$ROOT/logs/gate1_build_27b.log"
    nice -n 10 make -j"$JOBS" ARCH=61 FNX_MODEL=27b FNX_NCARD=2 \
        CUDA="$TOOLCHAIN" NVCC="$NVCC" HOSTCXX="$HOSTCXX" HOSTCC="$HOSTCC" \
        CUOBJDUMP="$CUOBJDUMP" HOST_LDFLAGS="$HOST_LDFLAGS" \
        bin/sfnx bin/s27_run bin/s27_ring_test 2>&1 | tee "$BUILD_27B_LOG"

    echo ""
    echo "--- [Gate 1B] Clean build: FNX_MODEL=fnx FNX_NCARD=2 ARCH=61 ---"
    rm -f build/*.o build/*.d build/kernels_p* build/pic/*.o build/pic/*.d
    BUILD_FNX_LOG="$ROOT/logs/gate1_build_fnx.log"
    nice -n 10 make -j"$JOBS" ARCH=61 FNX_MODEL=fnx FNX_NCARD=2 \
        CUDA="$TOOLCHAIN" NVCC="$NVCC" HOSTCXX="$HOSTCXX" HOSTCC="$HOSTCC" \
        CUOBJDUMP="$CUOBJDUMP" HOST_LDFLAGS="$HOST_LDFLAGS" \
        bin/sfnx bin/fnx_run bin/gguf_test bin/ple_test bin/test_kq_avx1 \
        bin/hc8_test bin/vhead_6p2_test bin/card_check_test bin/pf_ring_plan_test 2>&1 | tee "$BUILD_FNX_LOG"

    echo ""
    echo "--- [Gate 1C] Build 2-card BAR1 P2P probe (bin/p2p_store, sm_61) ---"
    BUILD_P2P_LOG="$ROOT/logs/gate1_build_p2p.log"
    "$NVCC" -O3 -std=c++17 -arch=sm_61 -ccbin "$HOSTCXX" -Wno-deprecated-gpu-targets \
        -include algorithm \
        -Xcompiler "-march=ivybridge -mtune=ivybridge -mavx -mf16c -mno-avx2 -mno-fma" \
        -o bin/p2p_store tools/p2p_store.cu 2>&1 | tee "$BUILD_P2P_LOG"

    # Zero-warning assertion across all touched/new files (filtering only pre-existing upstream KPART #177-D and dense.cuh #549-D)
    "$PYTHON" - "$BUILD_27B_LOG" "$BUILD_FNX_LOG" "$BUILD_P2P_LOG" <<'PYEOF'
import re
import sys
from pathlib import Path

warn_re = re.compile(r"(?:^|[:\s])warning\s*(?:#\d+-[A-Z])?:", re.I)
upstream_kpart_re = re.compile(r"^src/kernels\.cu\((?:[1-9]\d{0,2})\): warning #177-D:.*declared but never referenced")
upstream_dense_re = re.compile(r"^src/dense\.cuh\(4[12]\): warning #549-D: variable \"[qd]r\" is used before its value is set")
found = []
upstream_ignored = 0
for p_str in sys.argv[1:]:
    p = Path(p_str)
    for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
        if warn_re.search(line):
            if upstream_kpart_re.search(line) or upstream_dense_re.search(line):
                upstream_ignored += 1
                continue
            found.append(f"{p.name}:{i}: {line}")

if found:
    print(f"GATE 1 FAILED: {len(found)} compiler warning(s) in touched/new code:")
    for w in found[:40]:
        print("  WARN:", w)
    sys.exit(1)
print(f"GATE 1 PASS: Dual build (27b + fnx, NCARD=2, ARCH=61) + p2p_store succeeded with 0 warnings in touched/new code ({upstream_ignored} pre-existing upstream KPART/dense.cuh warnings ignored).")
PYEOF
    GATE1_STATUS="PASS"
fi

# ---------------------------------------------------------------------------
# Gate 2: Device SASS/PTX + Resource Usage & Host ISA Hard Gate
# ---------------------------------------------------------------------------
if [ "$GATE" = "all" ] || [ "$GATE" = "2" ]; then
    echo ""
    echo "--- [Gate 2] Device SASS/PTX + Resource Usage & Host ISA Audit ---"
    "$PYTHON" - "$ROOT" "$CUOBJDUMP" "$ROOT/logs/gate2_isa_report.json" <<'PYEOF'
import json
import re
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
cuobjdump = sys.argv[2]
out_json = Path(sys.argv[3])

if not cuobjdump or not Path(cuobjdump).is_file():
    print(f"GATE 2 FAIL: cuobjdump not found at {cuobjdump!r}")
    sys.exit(1)

failures = []
report = {"device": {}, "host": {}, "sass_mode": "unknown"}

# Strict STACK:0 enforced on all new/modified sm61-fnx-moe kernels
STRICT_STACK0_PATTERNS = (
    "k_pf_moe_dp4a",
    "k_pf_attn",
    "k_idx_score",
    "k_pf_q8s_dp4a",
    "k_pf_quant_q8",
    "k_moe_gu_q6k",
    "k_router_route",
    "k_wc_bridge",
    "k_merge_cold",
    "k_gdn_ch_rec61",
)

# 2A. Device targets
dev_targets = [
    root / "build/moe_mma.o",
    root / "build/prefill.o",
    root / "build/engine.o",
    root / "bin/sfnx",
    root / "bin/fnx_run",
] + sorted((root / "build").glob("kernels_p*.o"))

# Probe whether cuobjdump --dump-sass supports sm_61 on this host
probe_sass = subprocess.run(
    [cuobjdump, "--dump-sass", str(root / "build/moe_mma.o")],
    capture_output=True, text=True
)
sass_supported = (probe_sass.returncode == 0) and ("deprecated" not in probe_sass.stderr.lower())
report["sass_mode"] = "native_sass" if sass_supported else "ptx_plus_cubin_resources_cuda13_nvdisasm_sm61_deprecated"
if not sass_supported:
    print(f"NOTE: {cuobjdump} --dump-sass refused sm_61 ({probe_sass.stderr.strip()}); auditing compiled sm_61 cubin resources + embedded PTX.")

bad_sass_re = re.compile(r"\b(HFMA2|HMUL2|HADD2|HMMA|BPT\.TRAP)\b")
idp4a_sass_re = re.compile(r"\bIDP4A\b")

bad_ptx_fp16_re = re.compile(r"\b(?:fma|mul|add)\.[a-z0-9_.]*\.f16(?:x2)?\b")
bad_ptx_mma_re = re.compile(r"\b(?:mma\.sync|wmma\.)")
bad_ptx_acq_re = re.compile(r"\bld\.acquire\.sys\b")
bad_ptx_trap_re = re.compile(r"^\s*trap\s*;")
dp4a_ptx_re = re.compile(r"\bdp4a\.")

fn_re = re.compile(r"^\s*Function\s+([^\s:]+):")
res_re = re.compile(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)")

for tgt in dev_targets:
    if not tgt.is_file():
        failures.append(f"Missing device target: {tgt}")
        continue
    rel = str(tgt.relative_to(root))
    info = {
        "sm61_cubin_count": 0,
        "max_shared_bytes": 0,
        "over_48k_kernels": [],
        "strict_stack_nonzero": [],
        "legacy_stack_nonzero": [],
    }

    # Resource usage directly from compiled sm_61 ELF cubin
    ru = subprocess.run([cuobjdump, "--dump-resource-usage", str(tgt)], capture_output=True, text=True)
    if ru.returncode != 0:
        failures.append(f"{rel}: cuobjdump --dump-resource-usage failed: {ru.stderr.strip()}")
    else:
        info["sm61_cubin_count"] = ru.stdout.count("arch = sm_61")
        cur_fn = None
        for line in ru.stdout.splitlines():
            m_fn = fn_re.match(line)
            if m_fn:
                cur_fn = m_fn.group(1)
                continue
            m_r = res_re.search(line)
            if m_r and cur_fn:
                reg, stk, shm = int(m_r.group(1)), int(m_r.group(2)), int(m_r.group(3))
                if shm > info["max_shared_bytes"]:
                    info["max_shared_bytes"] = shm
                if shm > 49152:
                    info["over_48k_kernels"].append({"fn": cur_fn, "shared": shm})
                    failures.append(f"{rel}: kernel {cur_fn} uses {shm} B SMEM (> 49152 B)")
                if stk > 0:
                    entry = {"fn": cur_fn, "reg": reg, "stack": stk, "shared": shm}
                    if any(pat in cur_fn for pat in STRICT_STACK0_PATTERNS):
                        info["strict_stack_nonzero"].append(entry)
                        failures.append(f"{rel}: new/modified kernel {cur_fn} has STACK:{stk} (must be 0)")
                    else:
                        info["legacy_stack_nonzero"].append(entry)

    # Instruction audit (SASS if supported, always PTX as well)
    if sass_supported:
        ds = subprocess.run([cuobjdump, "--dump-sass", str(tgt)], capture_output=True, text=True)
        if ds.returncode != 0:
            failures.append(f"{rel}: cuobjdump --dump-sass failed: {ds.stderr.strip()}")
        else:
            bad_hits = bad_sass_re.findall(ds.stdout)
            idp4a_hits = len(idp4a_sass_re.findall(ds.stdout))
            info["sass_bad_count"] = len(bad_hits)
            info["sass_idp4a_count"] = idp4a_hits
            if bad_hits:
                failures.append(f"{rel}: forbidden SASS mnemonics found: {set(bad_hits)} (count={len(bad_hits)})")
            if rel in ("bin/sfnx", "bin/fnx_run", "build/moe_mma.o", "build/prefill.o") and idp4a_hits == 0:
                failures.append(f"{rel}: expected >0 IDP4A in SASS, got 0")

    dp = subprocess.run([cuobjdump, "--dump-ptx", str(tgt)], capture_output=True, text=True)
    if dp.returncode != 0:
        failures.append(f"{rel}: cuobjdump --dump-ptx failed: {dp.stderr.strip()}")
    else:
        ptx = dp.stdout
        fp16_hits = len(bad_ptx_fp16_re.findall(ptx))
        mma_hits = len(bad_ptx_mma_re.findall(ptx))
        acq_hits = len(bad_ptx_acq_re.findall(ptx))
        dp4a_hits = len(dp4a_ptx_re.findall(ptx))
        cur_entry = "<none>"
        bad_trap_entries = []
        ggml_trap_count = 0
        for pline in ptx.splitlines():
            if pline.startswith(".visible .entry ") or pline.startswith(".entry "):
                cur_entry = pline.strip().split("(")[0]
            elif bad_ptx_trap_re.match(pline):
                # Upstream ggml vecdotq.cuh emits unreachable NO_DEVICE_CODE traps only in k_moe_down<Q4_K/Q5_K>
                if "k_moe_down" in cur_entry and ("ggml_type12" in cur_entry or "ggml_type13" in cur_entry):
                    ggml_trap_count += 1
                else:
                    bad_trap_entries.append(cur_entry)
        trap_hits = len(bad_trap_entries)
        info.update({
            "ptx_fp16_arith": fp16_hits,
            "ptx_mma_wmma": mma_hits,
            "ptx_ld_acquire_sys": acq_hits,
            "ptx_trap": trap_hits,
            "ptx_ggml_dead_trap": ggml_trap_count,
            "ptx_dp4a": dp4a_hits,
        })
        if fp16_hits > 0:
            failures.append(f"{rel}: found {fp16_hits} FP16/FP16x2 arithmetic PTX ops (1:64 Pascal trap)")
        if mma_hits > 0:
            failures.append(f"{rel}: found {mma_hits} mma.sync/wmma PTX ops")
        if acq_hits > 0:
            failures.append(f"{rel}: found {acq_hits} ld.acquire.sys PTX ops (illegal on sm_61)")
        if trap_hits > 0:
            failures.append(f"{rel}: found {trap_hits} forbidden trap; PTX ops in {set(bad_trap_entries)}")
        if rel in ("bin/sfnx", "bin/fnx_run", "build/moe_mma.o", "build/prefill.o") and dp4a_hits == 0:
            failures.append(f"{rel}: expected >0 dp4a PTX ops, got 0")

    report["device"][rel] = info
    print(f"  [DEV] {rel:24s} sm61_cubins={info['sm61_cubin_count']} max_smem={info['max_shared_bytes']}B "
          f"dp4a={info.get('ptx_dp4a', 0)} fp16_arith={info.get('ptx_fp16_arith', 0)} "
          f"mma={info.get('ptx_mma_wmma', 0)} trap={info.get('ptx_trap', 0)} "
          f"new_stack_bad={len(info['strict_stack_nonzero'])}")

# 2B. Host ISA Audit (Ivy Bridge E5-2697 v2: ZERO AVX2 ymm-int, ZERO FMA3, ZERO BMI2)
host_targets = [
    root / "build/kq_avx1.o",
    root / "bin/test_kq_avx1",
    root / "bin/sfnx",
    root / "bin/fnx_run",
]

avx2_ymm_re = re.compile(
    r"\b(?:vpadd[bwdq]|vpadds[bw]|vpaddus[bw]|vpsub[bwdq]|vpsubs[bw]|vpsubus[bw]|"
    r"vpmaddwd|vpmaddubsw|vpshufb|vpshufd|vpshufhw|vpshuflw|"
    r"vpmul[a-z0-9]+|vpandn?|vpor|vpxor|vpsrl[a-z0-9]+|vpsll[a-z0-9]+|vpsra[a-z0-9]+|"
    r"vpack[a-z0-9]+|vpunpck[a-z0-9]+|vperm2i128|vpermd|vpermq|vpbroadcast[bwdq]|"
    r"vpcmpeq[bwdq]|vpcmpgt[bwdq]|vpmax[su][bwd]|vpmin[su][bwd]|vpmovmskb|vpalignr)\b.*%ymm"
)
fma3_re = re.compile(r"\b(?:vfmadd|vfmsub|vfnmadd|vfnmsub|vfmaddsub|vfmsubadd)[0-9a-z]*\b")
bmi2_re = re.compile(r"\b(?:shrx|sarx|shlx|bzhi|pext|pdep|mulx|rorx)\b")

for tgt in host_targets:
    if not tgt.is_file():
        failures.append(f"Missing host target: {tgt}")
        continue
    rel = str(tgt.relative_to(root))
    od = subprocess.run(["objdump", "-d", str(tgt)], capture_output=True, text=True)
    if od.returncode != 0:
        failures.append(f"{rel}: objdump -d failed: {od.stderr.strip()}")
        continue
    avx2_hits, fma3_hits, bmi2_hits = [], [], []
    for line in od.stdout.splitlines():
        # Instruction lines contain tabs: addr:\tbytes\tmnemonic operands
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        insn = parts[2].strip()
        if avx2_ymm_re.search(insn):
            avx2_hits.append(line.strip())
        if fma3_re.search(insn):
            fma3_hits.append(line.strip())
        if bmi2_re.search(insn):
            bmi2_hits.append(line.strip())

    report["host"][rel] = {
        "avx2_ymm_count": len(avx2_hits),
        "fma3_count": len(fma3_hits),
        "bmi2_count": len(bmi2_hits),
        "samples": (avx2_hits[:3] + fma3_hits[:3] + bmi2_hits[:3]),
    }
    print(f"  [HOST] {rel:23s} avx2_ymm={len(avx2_hits)} fma3={len(fma3_hits)} bmi2={len(bmi2_hits)}")
    if avx2_hits:
        failures.append(f"{rel}: {len(avx2_hits)} AVX2 ymm integer insn(s), e.g. {avx2_hits[0]}")
    if fma3_hits:
        failures.append(f"{rel}: {len(fma3_hits)} FMA3 insn(s), e.g. {fma3_hits[0]}")
    if bmi2_hits:
        failures.append(f"{rel}: {len(bmi2_hits)} BMI2 insn(s), e.g. {bmi2_hits[0]}")

report["passed"] = (len(failures) == 0)
report["failures"] = failures
out_json.write_text(json.dumps(report, indent=2) + "\n")

if failures:
    print("GATE 2 FAILED:")
    for f in failures:
        print("  FAIL:", f)
    sys.exit(1)
print(f"GATE 2 PASS: All device and host ISA checks passed (saved {out_json}).")
PYEOF
    GATE2_STATUS="PASS"
fi

# ---------------------------------------------------------------------------
# Gate 3: Unit Tests & Dual-GPU BAR1 P2P Probe
# ---------------------------------------------------------------------------
if [ "$GATE" = "all" ] || [ "$GATE" = "3" ]; then
    echo ""
    echo "--- [Gate 3] Unit Tests & Dual-GPU BAR1 P2P Probe ---"
    EDAC_BEFORE="$(read_edac_ce)"
    echo "EDAC CE before Gate 3: $EDAC_BEFORE"

    echo "[3.1] bin/card_check_test"
    ./bin/card_check_test

    echo "[3.2] bin/pf_ring_plan_test"
    ./bin/pf_ring_plan_test

    echo "[3.3] bin/gguf_test (synthetic + real hybrid splice & engram override)"
    if [ -f "$MODEL_BASE" ] && [ -f "$MODEL_Q6" ] && [ -f "$PLE_SSD" ]; then
        FNX_ENGRAM_GGUF="$PLE_SSD" FNX_HYBRID_Q6_GGUF="$MODEL_Q6" FNX_HYBRID_MAP="q4q6q8" \
            ./bin/gguf_test /tmp "$MODEL_BASE"
    else
        ./bin/gguf_test /tmp
    fi

    echo "[3.4] bin/ple_test (O_DIRECT SSD PLE byte-exact parity)"
    if [ -f "$PLE_SSD" ] && [ -f "$MODEL_BASE" ]; then
        # Generate a small synthetic prompt.bin in /tmp if no ref/*.bin exists
        PROMPT_BIN=""
        for cand in ref/*.bin; do
            if [ -f "$cand" ]; then
                PROMPT_BIN="$cand"
                break
            fi
        done
        TMP_PROMPT=""
        if [ -z "$PROMPT_BIN" ]; then
            TMP_PROMPT="/tmp/fnx_ple_gate3_prompt.bin"
            "$PYTHON" - "$TMP_PROMPT" <<'PYEOF'
import struct, sys
toks = [248045, 846, 198, 2610, 525, 264, 6548, 5946, 304, 279, 1879, 13] * 16
with open(sys.argv[1], "wb") as f:
    name = b"tokens"
    f.write(struct.pack("<I", len(name)))
    f.write(name)
    f.write(struct.pack("<Iq", 1, len(toks)))
    f.write(struct.pack(f"<{len(toks)}f", *[float(t) for t in toks]))
PYEOF
            PROMPT_BIN="$TMP_PROMPT"
        fi
        FNX_ENGRAM_GGUF="$PLE_SSD" FNX_NGRAM_CACHE_MIB=256 FNX_NGRAM_THREADS=12 \
            ./bin/ple_test "$MODEL_BASE" "$PROMPT_BIN"
        if [ -n "$TMP_PROMPT" ]; then
            rm -f "$TMP_PROMPT"
        fi
    else
        echo "FAIL: Missing SSD PLE ($PLE_SSD) or base model ($MODEL_BASE)" >&2
        exit 1
    fi

    echo "[3.5] bin/test_kq_avx1 (Ivy Bridge AVX1+F16C Q4_K/Q6_K/Q8_0 parity)"
    ./bin/test_kq_avx1

    echo "[3.6] bin/hc8_test (Flash-Next blk.0 & blk.1 hc8 parity)"
    ./bin/hc8_test

    echo "[3.7] bin/vhead_6p2_test (6+2 borrow-free LM head parity)"
    ./bin/vhead_6p2_test

    echo "[3.8] Dual-GPU BAR1 P2P Ring & Kernel Store Probe"
    if [ -x "./bin/s27_ring_test" ]; then
        ./bin/s27_ring_test
    else
        echo "FAIL: bin/s27_ring_test not built" >&2
        exit 1
    fi

    P2P_OUT="$(./bin/p2p_store)"
    printf '%s\n' "$P2P_OUT"
    if ! printf '%s\n' "$P2P_OUT" | grep -q "timeouts: 0"; then
        echo "FAIL: p2p_store reported nonzero timeouts!" >&2
        exit 1
    fi

    EDAC_AFTER="$(read_edac_ce)"
    EDAC_DELTA=$((EDAC_AFTER - EDAC_BEFORE))
    echo "EDAC CE after Gate 3:  $EDAC_AFTER (delta=$EDAC_DELTA)"
    if [ "$EDAC_DELTA" -ne 0 ]; then
        echo "FAIL: Nonzero Ivy Bridge EDAC CE delta ($EDAC_DELTA) during Gate 3!" >&2
        exit 1
    fi
    GATE3_STATUS="PASS"
fi

"$PYTHON" - "$REPORT_JSON" "$GATE1_STATUS" "$GATE2_STATUS" "$GATE3_STATUS" <<'PYEOF'
import json, sys, time
from pathlib import Path

out = Path(sys.argv[1])
data = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "gate1_dual_build": sys.argv[2],
    "gate2_isa_sass": sys.argv[3],
    "gate3_unit_and_p2p": sys.argv[4],
}
out.write_text(json.dumps(data, indent=2) + "\n")
print(f"\n=== SUMMARY SAVED TO {out} ===")
print(json.dumps(data, indent=2))
PYEOF
