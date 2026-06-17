"""Smoke-test, warmup, and timed benchmark execution.

Timed runs shell out to ``vllm bench serve`` (the current vLLM benchmark CLI)
against the llama-swap port, selecting the scenario by ``--model <scenario-id>``.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .scenarios import Scenario
from .swap import LlamaSwap


def _bench_cmd(base_url: str, model_id: str, tokenizer: str, concurrency: int,
               num_prompts: int, cfg: dict[str, Any], result_dir: Path,
               result_name: str) -> list[str]:
    cmd = [
        "vllm", "bench", "serve",
        "--backend", cfg.get("backend", "vllm"),
        "--base-url", base_url,
        "--endpoint", cfg.get("endpoint", "/v1/completions"),
        # --model is the request's model field => routes through llama-swap.
        # --tokenizer points at the LOCAL model dir so the client does NOT try
        # to fetch a tokenizer for the scenario id from HuggingFace.
        "--model", model_id,
        "--tokenizer", tokenizer,
        "--dataset-name", cfg.get("dataset", "random"),
        "--random-input-len", str(cfg["random_input_len"]),
        "--random-output-len", str(cfg["random_output_len"]),
        "--random-range-ratio", str(cfg.get("random_range_ratio", 0.0)),
        "--max-concurrency", str(concurrency),
        "--num-prompts", str(num_prompts),
        "--request-rate", "inf",
        "--seed", str(cfg.get("seed", 1234)),
        "--percentile-metrics", cfg.get("percentile_metrics", "ttft,tpot,itl,e2el"),
        "--metric-percentiles", cfg.get("metric_percentiles", "50,90,99"),
        "--num-warmups", str(cfg.get("num_warmups", 8)),
        "--save-result",
        "--result-dir", str(result_dir),
        "--result-filename", result_name,
    ]
    if cfg.get("ignore_eos", True):
        cmd.append("--ignore-eos")
    return cmd


def smoke_test(swap: LlamaSwap, scenarios: list[Scenario],
               logs_dir: Path, indices: list[int] | None = None,
               vram_threshold: float = 1024, vram_timeout: float = 180) -> list[dict]:
    """Load every scenario once and confirm a 1-token request succeeds.
    Surfaces OOM / flag-parse / startup crashes before the long run."""
    from . import gpu
    results = []
    for s in scenarios:
        t0 = time.time()
        ok, msg = swap.load_and_wait(s.id, timeout=900)
        detail = msg
        if not ok:
            # Pull the tail of the vLLM log for a useful error.
            log = Path(s.log_path)
            if log.exists():
                tail = "\n".join(log.read_text(errors="replace").splitlines()[-25:])
                detail = f"{msg}\n--- log tail ---\n{tail}"
        results.append({
            "scenario": s.id, "ok": ok, "seconds": round(time.time() - t0, 1),
            "detail": detail if not ok else "ok",
        })
        swap.unload()
        if indices:
            gpu.wait_for_vram_free(indices, vram_threshold, vram_timeout)
    return results


def parse_transport_proof(log_path: str | Path) -> dict:
    """Grep the vLLM/NCCL log for evidence of which comm path is active."""
    text = ""
    p = Path(log_path)
    if p.exists():
        text = p.read_text(errors="replace")
    custom_ar = None
    if re.search(r"custom all.?reduce is disabled", text, re.I):
        custom_ar = "disabled"
    elif re.search(r"using custom all.?reduce", text, re.I):
        custom_ar = "enabled"
    nccl_p2p = bool(re.search(r"via P2P|P2P/IPC|/direct pointer", text))
    nccl_shm = bool(re.search(r"via SHM|SHM/", text))
    nccl_ver = None
    m = re.search(r"nccl==([\d.]+)", text)
    if m:
        nccl_ver = m.group(1)
    return {
        "custom_all_reduce": custom_ar,
        "nccl_used_p2p": nccl_p2p,
        "nccl_used_shm": nccl_shm,
        "nccl_version": nccl_ver,
    }


def warmup(base_url: str, scenario: Scenario, cfg: dict[str, Any],
           tmp_dir: Path) -> None:
    """Short load to ramp clocks and trigger CUDA-graph capture before timing."""
    n = max(2 * scenario.concurrency, 8)
    cmd = _bench_cmd(base_url, scenario.id, scenario.model_path,
                     scenario.concurrency, n, cfg, tmp_dir, "warmup.json")
    subprocess.run(cmd, capture_output=True, text=True, timeout=1800)


def run_timed(base_url: str, scenario: Scenario, cfg: dict[str, Any],
              scenario_dir: Path, rep: int) -> tuple[Path | None, float, float]:
    """Run one timed rep. Returns (result_json_path, start_epoch, end_epoch)."""
    num_prompts = scenario.concurrency * int(cfg.get("num_prompts_multiplier", 16))
    name = f"run_{rep}.json"
    cmd = _bench_cmd(base_url, scenario.id, scenario.model_path,
                     scenario.concurrency, num_prompts, cfg, scenario_dir, name)
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    end = time.time()
    (scenario_dir / f"run_{rep}.stdout.txt").write_text(proc.stdout + "\n" + proc.stderr)
    result = scenario_dir / name
    return (result if result.exists() else None), start, end
