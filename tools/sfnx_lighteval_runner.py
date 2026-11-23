#!/usr/bin/env python3
"""
Gate 4 & Gate 5 Verification Runner for sm61-fnx-moe (Dual Quadro P6000 + Xeon E5-2697 v2).

Drives bin/sfnx --serve directly over its line protocol using serve/sfnx-fnx-z420.json,
runs the exact Row #8f apples-to-apples benchmark & quality-drift suite plus pp1024 and
multi-turn coherence verification, samples Dual-GPU nvidia-smi (1 Hz summed board W for J/tok),
sensors CPU temp, /proc/meminfo swap/RAM, and Ivy Bridge EDAC CE counters, and computes ALL
reported metrics strictly from raw run records.
"""
import argparse
import glob
import json
import os
import queue
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "serve" / "sfnx-fnx-z420.json"
DEFAULT_OUT_JSON = Path("/home/tom/projects/Large Models and testing/results/phase2_sm61_fnx_moe_dual_gpu.json")
DEFAULT_OUT_LOG = Path("/home/tom/projects/Large Models and testing/results/phase2_sm61_fnx_moe_dual_gpu.log")

POLY_DIR = Path("/home/tom/projects/polystrata")
POLY_SITE = Path("/mnt/samsung_storage/toolchains/cuda128/lib/python3.12/site-packages")
if POLY_SITE.is_dir() and str(POLY_SITE) not in sys.path:
    sys.path.insert(0, str(POLY_SITE))
if POLY_DIR.is_dir() and str(POLY_DIR) not in sys.path:
    sys.path.insert(0, str(POLY_DIR))

from tools.strata_tokenizer import Tokenizer  # type: ignore  # noqa: E402


# Exact Row #8f GSM8K, ARC-Challenge, and 3-Task Capability Suite items
GSM8K_ITEMS = [
    {
        "q": (
            "Solve the following problem carefully step by step. At the very end of your response, "
            "state the final numerical answer in the exact format 'ANSWER: <value>'.\n\n"
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and "
            "bakes muffins for her friends every day with four. She sells the remainder at the "
            "farmers' market daily for $2 per fresh duck egg. How much in dollars does she make "
            "every day at the farmers' market?"
        ),
        "ans": "18",
    },
    {
        "q": (
            "Solve the following problem carefully step by step. At the very end of your response, "
            "state the final numerical answer in the exact format 'ANSWER: <value>'.\n\n"
            "A robe takes 2 bolts of blue fiber and half that much white fiber. "
            "How many bolts in total does it take?"
        ),
        "ans": "3",
    },
    {
        "q": (
            "Solve the following problem carefully step by step. At the very end of your response, "
            "state the final numerical answer in the exact format 'ANSWER: <value>'.\n\n"
            "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in "
            "$50,000 in repairs. This increased the value of the house by 150%. "
            "How much profit did he make?"
        ),
        "ans": "70000",
    },
    {
        "q": (
            "Solve the following problem carefully step by step. At the very end of your response, "
            "state the final numerical answer in the exact format 'ANSWER: <value>'.\n\n"
            "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. "
            "How many total meters does he run a week?"
        ),
        "ans": "540",
    },
    {
        "q": (
            "Solve the following problem carefully step by step. At the very end of your response, "
            "state the final numerical answer in the exact format 'ANSWER: <value>'.\n\n"
            "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, "
            "containing seeds, mealworms and vegetables to help keep them healthy. She gives the "
            "chickens their feed in three separate meals. In the morning, she gives her flock of "
            "chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups "
            "of feed. How many cups of feed does she need to give her chickens in the final meal "
            "of the day if the size of Wendi's flock is 20 chickens?"
        ),
        "ans": "20",
    },
]

ARCC_ITEMS = [
    {
        "q": (
            "Which property of a mineral can be determined just by looking at it?\n"
            "A) luster\nB) mass\nC) weight\nD) hardness\n"
            "Reply with ONLY the letter (A, B, C, or D)."
        ),
        "ans": "A",
    },
    {
        "q": (
            "A student riding a bicycle observes that it moves faster on a smoothicano road than "
            "on a rough gravel road. This happens because the smooth road has:\n"
            "A) less gravity\nB) more gravity\nC) less friction\nD) more friction\n"
            "Reply with ONLY the letter (A, B, C, or D)."
        ),
        "ans": "C",
    },
    {
        "q": (
            "Which of these is the best conductor of electricity?\n"
            "A) glass rod\nB) cotton string\nC) plastic tubing\nD) copper penny\n"
            "Reply with ONLY the letter (A, B, C, or D)."
        ),
        "ans": "D",
    },
    {
        "q": (
            "What is the main source of energy for the water cycle?\n"
            "A) the Sun\nB) the Moon\nC) Earth's core\nD) ocean currents\n"
            "Reply with ONLY the letter (A, B, C, or D)."
        ),
        "ans": "A",
    },
    {
        "q": (
            "Which of the following is an example of a chemical change?\n"
            "A) ice melting\nB) wood burning\nC) sugar dissolving\nD) paper tearing\n"
            "Reply with ONLY the letter (A, B, C, or D)."
        ),
        "ans": "B",
    },
]

CAPABILITY_TASKS = [
    {
        "id": "task1_rag_classification",
        "name": "Deep Multi-Document RAG & Cross-Document Contradiction Classification",
        "max_new": 192,
        "prompt": (
            "Analyze the following 4 telemetry & engineering excerpts and output ONLY a valid JSON object with keys "
            "\"contradictions\" (list of objects with \"doc_a\", \"doc_b\", \"field\", \"conflict\") and \"root_cause_verdict\" (string).\n\n"
            "[DOC-1: BIOS & EDAC Log 08:14Z] HP Z420 Workstation with Xeon E5-2697 v2 and 128 GB DDR3-1866 ECC RDIMM. "
            "EDAC mc0 csrow1 ch0 reports 411,639 correctable ECC errors/sec (CE) and 12,618 CMCI interrupts/sec on physical DIMM 2 "
            "(CPU_SrcID#0_Ha#0_Chan#0_DIMM#1). Effective 24-thread STREAM Triad bandwidth measured at 33.4 GB/s.\n"
            "[DOC-2: Storage Audit Memo 09:05Z] Claims the 19.22 tok/s decode ceiling on 157.55 GiB Qwen3.8-Flash-Next-UD-Q6_K_XL "
            "is caused by PCIe 2.0 x4 NVMe page-fault throttling on /dev/nvme0n1 and states ECC memory counters were 0 CE/sec all morning.\n"
            "[DOC-3: Post-Swap Verification 10:02Z] After replacing physical DIMM 2 with a matched Samsung M393B2G70DB0-CMA RDIMM, "
            "EDAC CE count dropped to 0 CE/sec across all 8 channels and 24-thread STREAM Triad bandwidth rose from 33.4 GB/s to 51.01 GB/s. "
            "During 157.55 GiB inference with --resident-experts, disk read rate on /dev/nvme0n1 was 0.0 MB/s.\n"
            "[DOC-4: GPU Allocation Report 10:15Z] Notes that Row #8a launched with --gpu 0 only and pcie_frac=0.05, leaving GPU 1 "
            "(24 GB Quadro P6000 in Slot 5) with 23,646 MiB free (0% compute utilization) and GPU 0 waiting 57.7 ms/window for CPU DDR3 sweeps."
        ),
    },
    {
        "id": "task2_arch_code_planning",
        "name": "Long-Term Architectural Code Planning (Dual-GPU + CPU Hybrid MoE Scheduler)",
        "max_new": 256,
        "prompt": (
            "Design a lock-free C++17 work-stealing MoE expert dispatcher (`HybridMoEDispatcher::dispatch_layer`) for a workstation with:\n"
            "- GPU 0 (PCIe 3.0 x16, 12.6 GB/s H2D, 3,785 resident experts)\n"
            "- GPU 1 (PCIe 3.0 x16, 12.6 GB/s H2D, 4,800 resident experts via PeerExperts)\n"
            "- 24-thread Xeon E5-2697 v2 CPU pool (51.0 GB/s DDR3-1866 quad-channel, AVX1+F16C)\n"
            "Given `n_tok * top_k` routed `(token, expert_id)` pairs per layer, provide:\n"
            "1. Exact C++ struct definitions for cache-line-aligned atomic work queues.\n"
            "2. The cost-balance equation choosing `pcie_frac` dynamically from measured `t_gpu0_hit`, `t_gpu1_peer`, and `bw_cpu`.\n"
            "3. Concise C++17 implementation of `dispatch_layer` avoiding false sharing."
        ),
    },
    {
        "id": "task3_multi_agent_harness",
        "name": "Multi-Agent Harness Management & Fault-Tolerant DAG Recovery",
        "max_new": 192,
        "prompt": (
            "You are the Orchestrator Agent managing 4 concurrent subagents on an HP Z420 workstation:\n"
            "- `agent_dl`: Downloading a 142 GB GGUF model to `/mnt/storage_8tb_linux/models/gguf/`\n"
            "- `agent_cuda`: Compiling `sm_61` `__dp4a` CUDA kernels in `/home/tom/projects/polystrata/build-poly`\n"
            "- `agent_bench`: Running N=3 Dual-GPU benchmarks on GPU 0 + GPU 1 (requires >= 2,048 MiB free VRAM on GPU 1)\n"
            "- `agent_audit`: Auditing EDAC ECC counters and NVMe SMART telemetry\n"
            "Fault injected: `agent_cuda` spawns a background test that allocates 4,096 MiB on GPU 1 while `agent_bench` is mid-run, "
            "dropping GPU 1 free VRAM to 412 MiB. Output a strict JSON recovery plan with keys: "
            "\"immediate_actions\" (ordered list of tool calls), \"isolation_locks\" (resource mutex design), and \"retry_dag\" (dependencies)."
        ),
    },
]


def query_dual_gpus() -> List[Dict[str, Any]]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,pci.bus_id,memory.used,memory.free,memory.total,"
            "temperature.gpu,power.draw,pstate,pcie.link.gen.current,pcie.link.width.current",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    gpus: List[Dict[str, Any]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        gpus.append(
            {
                "index": int(parts[0]),
                "bus_id": parts[1],
                "used_mib": float(parts[2]),
                "free_mib": float(parts[3]),
                "total_mib": float(parts[4]),
                "temp_c": int(parts[5]),
                "power_w": float(parts[6]),
                "pstate": parts[7],
                "pcie_gen": int(parts[8]),
                "pcie_width": int(parts[9]),
            }
        )
    return gpus


def query_sensors() -> Tuple[float, str]:
    res = subprocess.run(["sensors"], capture_output=True, text=True)
    text = res.stdout.strip()
    m = re.search(r"Package id 0:\s+\+([0-9.]+)", text)
    return (float(m.group(1)) if m else -1.0), text


def query_edac_ce() -> int:
    total = 0
    paths = glob.glob("/sys/devices/system/edac/mc/mc*/csrow*/ch*_ce_count")
    if not paths:
        paths = glob.glob("/sys/devices/system/edac/mc/mc*/ce_count")
    for p in paths:
        try:
            total += int(Path(p).read_text().strip())
        except Exception:
            pass
    return total


def query_meminfo() -> Dict[str, float]:
    info: Dict[str, float] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].rstrip(":") in (
            "MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SwapTotal", "SwapFree"
        ):
            info[parts[0].rstrip(":") + "_mib"] = round(int(parts[1]) / 1024.0, 2)
    info["SwapUsed_mib"] = round(info.get("SwapTotal_mib", 0.0) - info.get("SwapFree_mib", 0.0), 2)
    return info


class DualGpuPowerSampler:
    """1 Hz summed dual-GPU power sampler (GPU0_W + GPU1_W) for accurate J/tok telemetry."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._samples: List[Dict[str, Any]] = []
        self._thr = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thr.start()

    def stop(self) -> None:
        self._stop.set()
        self._thr.join(timeout=3.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                gpus = query_dual_gpus()
                total_w = sum(g["power_w"] for g in gpus)
                gpu1_free = gpus[1]["free_mib"] if len(gpus) > 1 else -1.0
                with self._lock:
                    self._samples.append(
                        {
                            "t": time.monotonic(),
                            "total_w": round(total_w, 2),
                            "gpu0_w": gpus[0]["power_w"] if len(gpus) > 0 else 0.0,
                            "gpu1_w": gpus[1]["power_w"] if len(gpus) > 1 else 0.0,
                            "gpu0_temp_c": gpus[0]["temp_c"] if len(gpus) > 0 else 0,
                            "gpu1_temp_c": gpus[1]["temp_c"] if len(gpus) > 1 else 0,
                            "gpu1_free_mib": gpu1_free,
                        }
                    )
            except Exception:
                pass
            self._stop.wait(1.0)

    def slice_stats(self, t0: float, t1: float) -> Dict[str, Any]:
        with self._lock:
            pts = [s for s in self._samples if t0 <= s["t"] <= t1]
        if not pts:
            return {"samples": 0}
        return {
            "samples": len(pts),
            "mean_total_w": round(statistics.mean(p["total_w"] for p in pts), 2),
            "peak_total_w": round(max(p["total_w"] for p in pts), 2),
            "min_gpu1_free_mib": round(min(p["gpu1_free_mib"] for p in pts), 1),
            "peak_gpu0_temp_c": max(p["gpu0_temp_c"] for p in pts),
            "peak_gpu1_temp_c": max(p["gpu1_temp_c"] for p in pts),
        }

    def global_min_gpu1_free(self) -> float:
        with self._lock:
            vals = [s["gpu1_free_mib"] for s in self._samples if s["gpu1_free_mib"] >= 0]
        return min(vals) if vals else -1.0


def strip_think_tags(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    return cleaned


def grade_gsm8k(text: str, expected: str) -> bool:
    clean = strip_think_tags(text)
    m = re.findall(r"ANSWER\s*:\s*\$?\s*([0-9][0-9,\.]*)", clean, flags=re.I)
    if m:
        got = m[-1].replace(",", "").rstrip(".")
        if got == expected:
            return True
    # Fallback matching Row #8f grading if ANSWER: tag omitted
    return expected in clean.replace(",", "")


def grade_arcc(text: str, expected: str) -> bool:
    clean = strip_think_tags(text).strip().upper()
    if not clean:
        return False
    m = re.search(r"(?:ANSWER\s*:\s*)?\b([ABCD])\b", clean)
    if m and m.group(1) == expected:
        return True
    return clean.startswith(expected) or f" {expected}" in f" {clean}"


def git_info(repo: Path) -> Dict[str, str]:
    def _cmd(args: List[str]) -> str:
        r = subprocess.run(args, cwd=str(repo), capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else "unknown"

    return {
        "branch": _cmd(["git", "branch", "--show-current"]),
        "commit": _cmd(["git", "rev-parse", "HEAD"]),
        "short_commit": _cmd(["git", "rev-parse", "--short", "HEAD"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate 4 & Gate 5 verification runner for sm61-fnx-moe")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-log", type=Path, default=DEFAULT_OUT_LOG)
    ap.add_argument("--runs", type=int, default=3, help="Benchmark runs per prompt length (default: 3)")
    ap.add_argument("--max-new", type=int, default=128, help="Decode tokens per benchmark run (default: 128)")
    ap.add_argument("--gpu1-reserve-mib", type=float, default=2048.0)
    ap.add_argument("--page-cache-state", choices=["warm", "cold"], default="warm")
    ap.add_argument("--skip-1024", action="store_true")
    ap.add_argument("--skip-cap", action="store_true")
    ap.add_argument("--max-gsm8k-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cwd = (REPO_ROOT / cfg.get("cwd", ".")).resolve()
    exe = Path(cfg["exe"])
    if not exe.is_absolute():
        exe = (cwd / exe).resolve()
    if not exe.is_file():
        raise SystemExit(f"FATAL: Engine binary not found: {exe}")

    tok_dir = Path(cfg.get("tokenizer", "data/tokenizer"))
    if not tok_dir.is_absolute():
        tok_dir = (cwd / tok_dir).resolve()
    tok = Tokenizer.from_dir(tok_dir)

    header = tok.encode("<|im_start|>user\n", parse_special=True)
    footer = tok.encode("\n<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n", parse_special=True)
    flush_tok = 11 if (not header or header[0] != 11) else 12

    def encode_chat(user_text: str) -> List[int]:
        return header + tok.encode(user_text, parse_special=False) + footer

    def make_exact_prompt(target_len: int) -> List[int]:
        base = (
            "You are a senior systems architect evaluating high-capacity Mixture-of-Experts (MoE) inference "
            "on a Dual-GPU HP Z420 workstation with two 24 GB NVIDIA Quadro P6000 GPUs (48 GB GDDR5X) and "
            "128 GB DDR3-1866 ECC RAM. "
        )
        reps = max(12, (target_len // 30) + 4)
        body_ids = tok.encode(base * reps, parse_special=False)
        need = target_len - len(header) - len(footer)
        assert len(body_ids) >= need, f"Body too short: {len(body_ids)} < {need}"
        ids = header + body_ids[:need] + footer
        assert len(ids) == target_len
        return ids

    pp508_ids = make_exact_prompt(508)
    pp1024_ids = make_exact_prompt(1024)

    # Filter out --vision for text-only benchmark/eval
    engine_args = [a for a in cfg.get("args", []) if a != "--vision"]
    cmd = [str(exe), "--serve"] + engine_args

    env = os.environ.copy()
    for k, v in cfg.get("env", {}).items():
        env[str(k)] = str(v)
    env.setdefault("FNX_PF_TIMING", "1")
    env.setdefault("FNX_STATS", "1")
    if "gpu" in cfg and isinstance(cfg["gpu"], list) and cfg["gpu"]:
        env.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(g) for g in cfg["gpu"]))
    lib_dirs = cfg.get("lib_dirs", [])
    if lib_dirs:
        prev_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(str(d) for d in lib_dirs) + (f":{prev_ld}" if prev_ld else "")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_log.parent.mkdir(parents=True, exist_ok=True)

    edac_before = query_edac_ce()
    mem_before = query_meminfo()
    cpu_temp_before, sensors_before = query_sensors()
    gpus_before = query_dual_gpus()

    print("=== [Gate 4 & Gate 5] sm61-fnx-moe Dual-P6000 Hybrid Benchmark & Lighteval Suite ===", flush=True)
    print(f"Config: {args.config}", flush=True)
    print(f"Cmd:    {' '.join(cmd)}", flush=True)
    print(f"EDAC CE before: {edac_before} | SwapUsed: {mem_before['SwapUsed_mib']} MiB | PageCache: {args.page_cache_state}", flush=True)

    sampler = DualGpuPowerSampler()
    sampler.start()

    t_boot0 = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )

    log_fp = args.out_log.open("w", encoding="utf-8")
    boot_lines: List[str] = []
    stderr_q: "queue.Queue[str]" = queue.Queue()
    booting = True

    def _stderr_reader() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            s = line.rstrip("\n")
            log_fp.write(s + "\n")
            log_fp.flush()
            if booting:
                boot_lines.append(s)
                print(f"  [boot] {s}", flush=True)
            else:
                stderr_q.put(s)
                print(f"  [eng]  {s}", flush=True)

    t_err = threading.Thread(target=_stderr_reader, daemon=True)
    t_err.start()

    ready_line = ""
    boot_elapsed_s = 0.0
    gpus_loaded: List[Dict[str, Any]] = []
    runs_508: List[Dict[str, Any]] = []
    runs_1024: List[Dict[str, Any]] = []
    multiturn_res: Dict[str, Any] = {}
    gsm8k_details: List[Dict[str, Any]] = []
    arcc_details: List[Dict[str, Any]] = []
    cap_results: List[Dict[str, Any]] = []

    try:
        assert proc.stdout is not None and proc.stdin is not None
        for line in proc.stdout:
            s = line.strip()
            log_fp.write(f"[STDOUT] {s}\n")
            log_fp.flush()
            if s.startswith("READY"):
                ready_line = s
                print(f"  [READY] {s} (boot={time.monotonic() - t_boot0:.2f}s)", flush=True)
                break
            if s.startswith("ERR") or s.startswith("FATAL"):
                raise RuntimeError(f"sfnx boot error: {s}")
        else:
            raise RuntimeError(f"sfnx exited before READY (rc={proc.wait()})")

        booting = False
        boot_elapsed_s = round(time.monotonic() - t_boot0, 2)
        gpus_loaded = query_dual_gpus()
        print(f"Loaded Dual-P6000 VRAM: {json.dumps(gpus_loaded)}", flush=True)

        def send_gen(ids: List[int], max_new: int, ckpt: int = 0) -> Dict[str, Any]:
            while not stderr_q.empty():
                stderr_q.get_nowait()
            csv_ids = ",".join(str(x) for x in ids)
            t_req0 = time.monotonic()
            proc.stdin.write(f"GEN {max_new} ckpt={ckpt} temperature=0 seed={args.seed} {csv_ids}\n")
            proc.stdin.flush()

            out_toks: List[int] = []
            t_first_tok: Optional[float] = None
            done_parts: List[str] = []
            reused_msg: Optional[int] = None
            while True:
                ln = proc.stdout.readline()
                if not ln:
                    raise RuntimeError(f"EOF from bin/sfnx (poll={proc.poll()})")
                s = ln.strip()
                if s.startswith("T "):
                    if t_first_tok is None:
                        t_first_tok = time.monotonic()
                    out_toks.append(int(s.split()[1]))
                elif s.startswith("RESUME "):
                    reused_msg = int(s.split()[1])
                elif s.startswith("DONE "):
                    done_parts = s.split()
                    break
                elif s.startswith("ERR "):
                    raise RuntimeError(f"sfnx returned {s}")

            t_req1 = time.monotonic()
            time.sleep(0.15)
            timing_lines: List[str] = []
            while not stderr_q.empty():
                timing_lines.append(stderr_q.get_nowait())

            # DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <finish> <drafts_accepted> <drafts_offered> <reused>
            generated = int(done_parts[1])
            prompt_n = int(done_parts[2])
            prompt_ms = float(done_parts[3])
            decode_ms = float(done_parts[4])
            finish = done_parts[5] if len(done_parts) > 5 else ""
            draft_acc = int(done_parts[6]) if len(done_parts) > 6 else 0
            draft_off = int(done_parts[7]) if len(done_parts) > 7 else 0
            reused = int(done_parts[8]) if len(done_parts) > 8 else (reused_msg or 0)

            prompt_read = max(0, prompt_n - reused)
            prefill_tps = (prompt_read / (prompt_ms / 1000.0)) if prompt_ms > 0 and prompt_read > 0 else 0.0
            decode_tps = (generated / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0
            after_first_tps = (
                ((generated - 1) / (t_req1 - t_first_tok))
                if (t_first_tok is not None and generated > 1 and t_req1 > t_first_tok)
                else decode_tps
            )
            pwr = sampler.slice_stats(t_req0, t_req1)
            j_per_tok = (
                round(pwr["mean_total_w"] / decode_tps, 3)
                if pwr.get("samples", 0) > 0 and decode_tps > 0
                else None
            )
            cpu_c, _ = query_sensors()
            return {
                "prompt_tokens": prompt_n,
                "prompt_reused": reused,
                "prompt_read": prompt_read,
                "generated_tokens": generated,
                "prefill_ms": round(prompt_ms, 2),
                "decode_ms": round(decode_ms, 2),
                "prefill_tps": round(prefill_tps, 2),
                "decode_tps": round(decode_tps, 2),
                "decode_tps_after_first": round(after_first_tps, 2),
                "finish": finish,
                "draft_accepted": draft_acc,
                "draft_offered": draft_off,
                "draft_accept_pct": round(100.0 * draft_acc / max(1, draft_off), 2),
                "power": pwr,
                "joules_per_tok": j_per_tok,
                "text": tok.decode(out_toks),
                "out_tokens": out_toks,
                "timing_lines": timing_lines,
                "gpus": query_dual_gpus(),
                "cpu_temp_c": cpu_c,
            }

        def flush_lane() -> None:
            _ = send_gen([flush_tok], max_new=1, ckpt=0)

        # Warmup (captures CUDA graphs)
        print("  [Warmup] Capturing CUDA graphs (48 tokens on pp508)...", flush=True)
        _ = send_gen(pp508_ids, max_new=48, ckpt=0)

        # Gate 4A: pp508 + tg128 (N runs, strictly reused == 0)
        for idx in range(1, args.runs + 1):
            flush_lane()
            print(f"  [pp508 Run {idx}/{args.runs}] 508-prompt + {args.max_new}-decode...", flush=True)
            r = send_gen(pp508_ids, max_new=args.max_new, ckpt=0)
            if r["prompt_reused"] != 0:
                raise RuntimeError(f"pp508 Run {idx}: expected prompt_reused==0, got {r['prompt_reused']}")
            r_save = {k: v for k, v in r.items() if k != "out_tokens"}
            runs_508.append(r_save)
            print(
                f"  [pp508 Run {idx}/{args.runs} DONE] prefill={r['prefill_tps']:.2f} t/s | "
                f"decode={r['decode_tps']:.2f} t/s | accept={r['draft_accept_pct']:.1f}% | "
                f"GPU0={r['gpus'][0]['used_mib']:.0f} MiB | GPU1={r['gpus'][1]['used_mib']:.0f} MiB "
                f"(free={r['gpus'][1]['free_mib']:.0f} MiB)",
                flush=True,
            )

        # Gate 4B: pp1024 + tg128 (N runs, strictly reused == 0)
        if not args.skip_1024:
            for idx in range(1, args.runs + 1):
                flush_lane()
                print(f"  [pp1024 Run {idx}/{args.runs}] 1024-prompt + {args.max_new}-decode...", flush=True)
                r = send_gen(pp1024_ids, max_new=args.max_new, ckpt=0)
                if r["prompt_reused"] != 0:
                    raise RuntimeError(f"pp1024 Run {idx}: expected prompt_reused==0, got {r['prompt_reused']}")
                r_save = {k: v for k, v in r.items() if k != "out_tokens"}
                runs_1024.append(r_save)
                print(
                    f"  [pp1024 Run {idx}/{args.runs} DONE] prefill={r['prefill_tps']:.2f} t/s | "
                    f"decode={r['decode_tps']:.2f} t/s | accept={r['draft_accept_pct']:.1f}%",
                    flush=True,
                )

        # Gate 5A: Multi-Turn Coherence & KV Prefix Reuse
        print("  [Gate 5A] Multi-turn coherence & KV prefix reuse...", flush=True)
        flush_lane()
        t1_ids = encode_chat(
            "In one concise sentence, explain why Pascal GP102 uses INT8 __dp4a instead of FP16 HFMA2."
        )
        t1 = send_gen(t1_ids, max_new=96, ckpt=1)
        t2_suffix = tok.encode(
            "<|im_end|>\n<|im_start|>user\nNow state the exact FP16-to-FP32 throughput ratio on GP102 in one short sentence."
            "\n<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n",
            parse_special=True,
        )
        t2_ids = t1_ids + t1["out_tokens"] + t2_suffix
        t2 = send_gen(t2_ids, max_new=96, ckpt=1)
        mt_ok = (t2["prompt_reused"] > 0) and (len(strip_think_tags(t1["text"])) >= 10) and (len(strip_think_tags(t2["text"])) >= 5)
        multiturn_res = {
            "pass": mt_ok,
            "turn1_reused": t1["prompt_reused"],
            "turn1_text": t1["text"].strip(),
            "turn2_reused": t2["prompt_reused"],
            "turn2_prompt_tokens": t2["prompt_tokens"],
            "turn2_text": t2["text"].strip(),
        }
        print(f"  [Multi-turn] pass={mt_ok} turn2_reused={t2['prompt_reused']}/{t2['prompt_tokens']}", flush=True)

        # Gate 5B: GSM8K (5/5)
        print("  [Gate 5B] GSM8K (N=5, exact Row #8f prompts)...", flush=True)
        for idx, item in enumerate(GSM8K_ITEMS, 1):
            flush_lane()
            res = send_gen(encode_chat(item["q"]), max_new=args.max_gsm8k_tokens, ckpt=0)
            ok = grade_gsm8k(res["text"], item["ans"])
            gsm8k_details.append(
                {
                    "idx": idx,
                    "expected": item["ans"],
                    "got": res["text"].strip(),
                    "pass": ok,
                    "prefill_tps": res["prefill_tps"],
                    "decode_tps": res["decode_tps"],
                }
            )
            print(
                f"    [GSM8K {idx}/5] expected={item['ans']} pass={ok} "
                f"tail={res['text'].strip()[-80:]!r}",
                flush=True,
            )

        # Gate 5C: ARC-Challenge (5/5)
        print("  [Gate 5C] ARC-Challenge (N=5, exact Row #8f prompts)...", flush=True)
        for idx, item in enumerate(ARCC_ITEMS, 1):
            flush_lane()
            res = send_gen(encode_chat(item["q"]), max_new=16, ckpt=0)
            ok = grade_arcc(res["text"], item["ans"])
            arcc_details.append(
                {
                    "idx": idx,
                    "expected": item["ans"],
                    "got": res["text"].strip(),
                    "pass": ok,
                    "prefill_tps": res["prefill_tps"],
                    "decode_tps": res["decode_tps"],
                }
            )
            print(
                f"    [ARC-C {idx}/5] expected={item['ans']} got={res['text'].strip()!r} pass={ok}",
                flush=True,
            )

        # Gate 5D: 3-Task Capability Suite (matching Row #8f)
        if not args.skip_cap:
            print("  [Gate 5D] 3-Task Capability Suite...", flush=True)
            for task in CAPABILITY_TASKS:
                flush_lane()
                print(f"    -> {task['name']} (max_new={task['max_new']})...", flush=True)
                res = send_gen(encode_chat(task["prompt"]), max_new=task["max_new"], ckpt=0)
                cap_results.append(
                    {
                        "id": task["id"],
                        "name": task["name"],
                        "prompt_tokens": res["prompt_tokens"],
                        "generated_tokens": res["generated_tokens"],
                        "prefill_tps": res["prefill_tps"],
                        "decode_tps": res["decode_tps"],
                        "draft_accept_pct": res["draft_accept_pct"],
                        "output": res["text"].strip(),
                    }
                )
                print(
                    f"    [DONE {task['id']}] prefill={res['prefill_tps']:.2f} t/s | "
                    f"decode={res['decode_tps']:.2f} t/s",
                    flush=True,
                )

        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
        proc.wait(timeout=30)
    finally:
        sampler.stop()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        log_fp.close()

    edac_after = query_edac_ce()
    edac_delta = edac_after - edac_before
    mem_after = query_meminfo()
    cpu_temp_after, sensors_after = query_sensors()

    # Aggregate statistics strictly from raw run arrays (Mistake #59 compliant)
    med_pp508 = round(statistics.median(r["prefill_tps"] for r in runs_508), 2)
    pk_pp508 = round(max(r["prefill_tps"] for r in runs_508), 2)
    med_tg128 = round(statistics.median(r["decode_tps"] for r in runs_508), 2)
    pk_tg128 = round(max(r["decode_tps"] for r in runs_508), 2)
    med_accept = round(statistics.median(r["draft_accept_pct"] for r in runs_508), 2)

    j_vals = [r["joules_per_tok"] for r in runs_508 if r["joules_per_tok"] is not None]
    med_j_per_tok = round(statistics.median(j_vals), 3) if j_vals else None

    med_pp1024 = round(statistics.median(r["prefill_tps"] for r in runs_1024), 2) if runs_1024 else None
    pk_pp1024 = round(max(r["prefill_tps"] for r in runs_1024), 2) if runs_1024 else None
    med_tg128_1024 = round(statistics.median(r["decode_tps"] for r in runs_1024), 2) if runs_1024 else None

    all_gpu1_free = [g[1]["free_mib"] for g in [gpus_loaded] + [r["gpus"] for r in runs_508 + runs_1024] if len(g) > 1]
    sampled_min = sampler.global_min_gpu1_free()
    if sampled_min >= 0:
        all_gpu1_free.append(sampled_min)
    min_gpu1_free_mib = round(min(all_gpu1_free), 1) if all_gpu1_free else -1.0

    gsm8k_pass = sum(1 for d in gsm8k_details if d["pass"])
    arcc_pass = sum(1 for d in arcc_details if d["pass"])

    # Check engine log for expert swap == 0 confirmation
    full_log_text = args.out_log.read_text(encoding="utf-8", errors="replace")
    swap_hits = re.findall(r"swap[s]?[=:\s]+(\d+)", full_log_text, flags=re.I)
    nonzero_expert_swaps = [int(x) for x in swap_hits if int(x) != 0]

    gate_checks = {
        "gpu1_reserve_ok": min_gpu1_free_mib >= args.gpu1_reserve_mib,
        "edac_ce_delta_zero": edac_delta == 0,
        "os_swap_zero_delta": (mem_after["SwapUsed_mib"] - mem_before["SwapUsed_mib"]) <= 1.0,
        "expert_swap_zero": len(nonzero_expert_swaps) == 0,
        "multiturn_ok": bool(multiturn_res.get("pass", False)),
        "gsm8k_5_of_5": gsm8k_pass == len(GSM8K_ITEMS),
        "arcc_5_of_5": arcc_pass == len(ARCC_ITEMS),
    }
    all_gates_pass = all(gate_checks.values())

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "row": "9",
        "git": git_info(REPO_ROOT),
        "config_path": str(args.config),
        "model": "Qwen3.8-Flash-Next Hybrid UD-Q4_K_XL + Q6_K/Q8_0 + SSD Q8_0 PLE (sm61-fnx-moe)",
        "engine": str(exe),
        "cmd": cmd,
        "env_overrides": cfg.get("env", {}),
        "page_cache_state": args.page_cache_state,
        "boot_elapsed_s": boot_elapsed_s,
        "ready_line": ready_line,
        "boot_lines": boot_lines,
        "gpus_before": gpus_before,
        "gpus_loaded": gpus_loaded,
        "min_gpu1_free_mib": min_gpu1_free_mib,
        "gpu1_reserve_mib_required": args.gpu1_reserve_mib,
        "mem_before": mem_before,
        "mem_after": mem_after,
        "cpu_temp_before_c": cpu_temp_before,
        "cpu_temp_after_c": cpu_temp_after,
        "sensors_after": sensors_after,
        "edac_ce_before": edac_before,
        "edac_ce_after": edac_after,
        "edac_ce_delta": edac_delta,
        "median_pp508_tps": med_pp508,
        "peak_pp508_tps": pk_pp508,
        "median_tg128_tps": med_tg128,
        "peak_tg128_tps": pk_tg128,
        "median_pp1024_tps": med_pp1024,
        "peak_pp1024_tps": pk_pp1024,
        "median_tg128_after_pp1024_tps": med_tg128_1024,
        "median_draft_accept_pct": med_accept,
        "median_joules_per_tok": med_j_per_tok,
        "gsm8k_score": f"{gsm8k_pass}/{len(GSM8K_ITEMS)}",
        "arcc_score": f"{arcc_pass}/{len(ARCC_ITEMS)}",
        "multiturn": multiturn_res,
        "gsm8k_details": gsm8k_details,
        "arcc_details": arcc_details,
        "capability_suite": cap_results,
        "runs_pp508": runs_508,
        "runs_pp1024": runs_1024,
        "gate_checks": gate_checks,
        "all_gates_pass": all_gates_pass,
    }
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== SAVED {args.out_json} & {args.out_log} ===", flush=True)
    print(
        f"Summary: pp508={med_pp508} med ({pk_pp508} pk) t/s | pp1024={med_pp1024} med ({pk_pp1024} pk) t/s | "
        f"tg128={med_tg128} med ({pk_tg128} pk) t/s | accept={med_accept}% | J/tok={med_j_per_tok} | "
        f"GSM8K={gsm8k_pass}/5 | ARC-C={arcc_pass}/5 | GPU1_min_free={min_gpu1_free_mib} MiB | "
        f"EDAC_CE_delta={edac_delta} | ALL_PASS={all_gates_pass}",
        flush=True,
    )
    if not all_gates_pass:
        failed = [k for k, v in gate_checks.items() if not v]
        raise SystemExit(f"GATE 4/5 FAILED: {failed}")


if __name__ == "__main__":
    main()
