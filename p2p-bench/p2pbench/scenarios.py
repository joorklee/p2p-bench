"""Scenario matrix model and llama-swap config generation.

A *scenario* is one (arm, concurrency) pair, e.g. ``p2p_on-c8``. Each scenario
becomes one entry in the generated llama-swap config, with its own env block and
``--max-num-seqs`` value. The served-model-name equals the scenario id, so the
benchmark client selects a scenario simply by setting ``--model <scenario-id>``.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Scenario:
    id: str
    arm: str
    concurrency: int
    env: dict[str, str]
    flags: list[str]
    args: dict[str, Any]
    model_path: str
    port: int
    log_path: str

    @property
    def max_num_seqs(self) -> int:
        return self.concurrency


def _load(path: str | Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def build_scenarios(
    spec_path: str | Path,
    logs_dir: str | Path,
    include_optional: bool = False,
) -> list[Scenario]:
    spec = _load(spec_path)
    model = spec["model"]
    arms = spec["arms"]
    concurrencies = spec["concurrencies"]
    logs_dir = Path(logs_dir)

    base_env = dict(model.get("constant_env", {}))
    base_flags = list(model.get("constant_flags", []))
    base_args = dict(model.get("constant_args", {}))

    scenarios: list[Scenario] = []
    for arm_name, arm in arms.items():
        if arm.get("optional") and not include_optional:
            continue
        for c in concurrencies:
            sid = f"{arm_name}-c{c}"
            env = {**base_env, **{k: str(v) for k, v in arm.get("env", {}).items()}}
            flags = base_flags + list(arm.get("extra_flags", []))
            # Per-arm server-arg overrides. Merged OVER constant_args so an arm
            # can change e.g. tensor-parallel-size / pipeline-parallel-size /
            # max-num-batched-tokens / speculative-config. An empty-string value
            # means "omit this arg entirely" (e.g. speculative-config: "" to turn
            # MTP off for the baseline arm).
            merged_args = {**base_args, **arm.get("extra_args", {})}
            args = {k: v for k, v in merged_args.items() if v != ""}
            scenarios.append(
                Scenario(
                    id=sid,
                    arm=arm_name,
                    concurrency=c,
                    env=env,
                    flags=flags,
                    args=args,
                    model_path=model["path"],
                    port=int(model["port"]),
                    log_path=str(logs_dir / f"{sid}.log"),
                )
            )
    return scenarios


def _vllm_cmd(s: Scenario) -> str:
    parts = ["vllm", "serve", shlex.quote(s.model_path),
             "--served-model-name", s.id,
             "--port", str(s.port),
             "--max-num-seqs", str(s.max_num_seqs)]
    for key, val in s.args.items():
        parts += [f"--{key}", shlex.quote(str(val))]
    for flag in s.flags:
        # flags may be "name" or "name value"
        bits = flag.split(" ", 1)
        if len(bits) == 1:
            parts.append(f"--{bits[0]}")
        else:
            parts += [f"--{bits[0]}", shlex.quote(bits[1])]
    vllm = " ".join(parts)
    # Tee combined output to a per-scenario log so the harness can later grep the
    # NCCL transport line and the custom-all-reduce status (proof of P2P path).
    Path(s.log_path).parent.mkdir(parents=True, exist_ok=True)
    return f"bash -c {shlex.quote(vllm + f' 2>&1 | tee {shlex.quote(s.log_path)}')}"


def generate_llama_swap_config(
    scenarios: list[Scenario],
    out_path: str | Path,
    health_timeout: int = 1200,
) -> Path:
    """Write a llama-swap config.yaml covering every scenario."""
    models: dict[str, Any] = {}
    for s in scenarios:
        models[s.id] = {
            "cmd": _vllm_cmd(s),
            "proxy": f"http://127.0.0.1:{s.port}",
            "env": [f"{k}={v}" for k, v in s.env.items()],
            # Keep a loaded model resident until the harness swaps it out.
            "ttl": 0,
        }
    config = {
        "healthCheckTimeout": health_timeout,
        "logLevel": "info",
        "models": models,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, width=10_000)
    return out_path


def load_benchmark_cfg(spec_path: str | Path) -> dict[str, Any]:
    return _load(spec_path)["benchmark"]
