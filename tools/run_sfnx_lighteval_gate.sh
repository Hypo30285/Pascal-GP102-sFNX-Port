#!/usr/bin/env bash
# run_sfnx_lighteval_gate.sh — Gate 4 (End-to-End 180B Hybrid Benchmark vs PolyStrata Row #8f)
# and Gate 5 (Quality-Drift & Lighteval GSM8K 5/5 + ARC-C 5/5 + Multi-turn) driver for sm61-fnx-moe.
#
# Usage:
#   tools/run_sfnx_lighteval_gate.sh [--config serve/sfnx-fnx-z420.json] [--runs 3] [--page-cache-state warm|cold] [extra runner args...]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TOOLCHAIN="${TOOLCHAIN:-/mnt/samsung_storage/toolchains/cuda128}"
PYTHON="${PYTHON:-$TOOLCHAIN/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

CONFIG="$ROOT/serve/sfnx-fnx-z420.json"
OUT_JSON="/home/tom/projects/Large Models and testing/results/phase2_sm61_fnx_moe_dual_gpu.json"
OUT_LOG="/home/tom/projects/Large Models and testing/results/phase2_sm61_fnx_moe_dual_gpu.log"
PLE_SSD="/mnt/samsung_storage/ple/Qwen3.8-Flash-Next-UD-Q6_K_XL-00003-of-00006.gguf"

# Parse optional --config / --out-json / --out-log while forwarding all flags
PASS_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --config) CONFIG="$2"; PASS_ARGS+=("$1" "$2"); shift 2 ;;
        --out-json) OUT_JSON="$2"; PASS_ARGS+=("$1" "$2"); shift 2 ;;
        --out-log) OUT_LOG="$2"; PASS_ARGS+=("$1" "$2"); shift 2 ;;
        *) PASS_ARGS+=("$1"); shift ;;
    esac
done

echo "=== sm61-fnx-moe Gate 4 & Gate 5 Lighteval Runner ==="
echo "Repo:    $ROOT ($(git branch --show-current || echo detached) @ $(git rev-parse --short HEAD))"
echo "Config:  $CONFIG"
echo "JSON:    $OUT_JSON"
echo "Log:     $OUT_LOG"

# Preflight assertions
[ -x "$ROOT/bin/sfnx" ] || { echo "FAIL: $ROOT/bin/sfnx missing or not executable" >&2; exit 1; }
[ -f "$CONFIG" ] || { echo "FAIL: config $CONFIG not found" >&2; exit 1; }
[ -f "$PLE_SSD" ] || { echo "FAIL: SSD PLE shard missing at $PLE_SSD" >&2; exit 1; }

echo "--- Preflight Dual-P6000 & Thermal Telemetry ---"
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used,memory.free,memory.total,temperature.gpu,power.draw,pstate,pcie.link.gen.current,pcie.link.width.current --format=csv
sensors | grep -E 'Package id|Core 0|Core 1' | head -n 6 || true

mkdir -p "$(dirname "$OUT_JSON")"

# Execute Gate 4 + Gate 5 runner
"$PYTHON" "$ROOT/tools/sfnx_lighteval_runner.py" \
    --config "$CONFIG" \
    --out-json "$OUT_JSON" \
    --out-log "$OUT_LOG" \
    "${PASS_ARGS[@]}"

# Re-read saved JSON via Python to print strictly computed deltas vs Row #8f (Mistake #59 compliant)
"$PYTHON" - "$OUT_JSON" <<'PYEOF'
import json
import sys
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
# Row #8f baseline constants from phase1_qwen38_hybrid_q4q6q8_dual_gpu.json
r8f_pp = 158.23
r8f_tg_med = 41.60
r8f_tg_pk = 44.24

pp_delta = ((d["median_pp508_tps"] - r8f_pp) / r8f_pp) * 100.0
tg_med_delta = ((d["median_tg128_tps"] - r8f_tg_med) / r8f_tg_med) * 100.0
tg_pk_delta = ((d["peak_tg128_tps"] - r8f_tg_pk) / r8f_tg_pk) * 100.0

print("\n=== VERIFIED GATE 4 & GATE 5 METRICS (COMPUTED FROM SAVED JSON) ===")
print(f"  Git:          {d['git']['branch']} @ {d['git']['short_commit']}")
print(f"  Boot:         {d['boot_elapsed_s']} s ({d['page_cache_state']} page cache)")
print(f"  pp508:        {d['median_pp508_tps']} med / {d['peak_pp508_tps']} pk t/s ({pp_delta:+.2f}% vs Row #8f {r8f_pp})")
print(f"  pp1024:       {d['median_pp1024_tps']} med / {d['peak_pp1024_tps']} pk t/s")
print(f"  tg128:        {d['median_tg128_tps']} med / {d['peak_tg128_tps']} pk t/s "
      f"({tg_med_delta:+.2f}% med / {tg_pk_delta:+.2f}% pk vs Row #8f {r8f_tg_med}/{r8f_tg_pk})")
print(f"  MTP accept:   {d['median_draft_accept_pct']}% | Energy: {d['median_joules_per_tok']} J/tok")
print(f"  GSM8K:        {d['gsm8k_score']} | ARC-C: {d['arcc_score']} | Multi-turn: {d['multiturn']['pass']}")
print(f"  GPU1 min free:{d['min_gpu1_free_mib']} MiB (req >= {d['gpu1_reserve_mib_required']} MiB)")
print(f"  EDAC CE delta:{d['edac_ce_delta']} (before={d['edac_ce_before']} after={d['edac_ce_after']})")
print(f"  Gate checks:  {d['gate_checks']}")
PYEOF
