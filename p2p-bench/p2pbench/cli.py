"""Command-line interface and orchestration for the P2P benchmark harness."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

from . import analyze, bench, environment, gpu
from .scenarios import (build_scenarios, generate_llama_swap_config,
                        load_benchmark_cfg)
from .swap import LlamaSwap
from .telemetry import TelemetrySampler, mean_power_in_window

DEFAULT_SPEC = str(Path(__file__).resolve().parent.parent / "config" / "scenarios.yaml")
EXPECTED_INDICES = (1, 2)


def _arms(scenarios) -> list[str]:
    seen = []
    for s in scenarios:
        if s.arm not in seen:
            seen.append(s.arm)
    return seen


def _concurrencies(scenarios) -> list[int]:
    return sorted({s.concurrency for s in scenarios})


def cmd_gen_config(args) -> int:
    out = Path(args.output_dir)
    scenarios = build_scenarios(args.scenarios, out / "logs", args.include_optional_arm)
    cfg_path = generate_llama_swap_config(scenarios, out / "llama-swap.config.yaml")
    print(f"Wrote {cfg_path} with {len(scenarios)} scenarios.")
    return 0


def cmd_check_env(args) -> int:
    out = Path(args.output_dir)
    env_dir = out / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    summary = environment.write_environment(env_dir)
    evidence = gpu.collect_p2p_evidence(args.p2p_test_bin, EXPECTED_INDICES)
    (env_dir / "p2p_evidence.json").write_text(json.dumps(evidence, indent=2))
    (env_dir / "host_summary.json").write_text(json.dumps(summary, indent=2))

    order = evidence["device_order"]
    print("Device-order check (expect 5060 Ti at indices 1,2):",
          "OK" if order["ok"] else "MISMATCH")
    for c in order["checked"]:
        print(f"  idx {c['index']}: {c['name']}  match={c['matches']}")
    probe = evidence["torch_probe"]
    if probe.get("ok"):
        print(f"torch P2P probe: can_access_peer={probe['can_access_peer']} "
              f"d2d={probe['d2d_copy_GBps']} GB/s")
    if not order["ok"]:
        print("WARNING: device order does not match; fix CUDA_VISIBLE_DEVICES "
              "before benchmarking.", file=sys.stderr)
    return 0


def cmd_smoke(args) -> int:
    out = Path(args.output_dir)
    scenarios = build_scenarios(args.scenarios, out / "logs", args.include_optional_arm)
    generate_llama_swap_config(scenarios, out / "llama-swap.config.yaml")
    with LlamaSwap(args.llama_swap_bin, out / "llama-swap.config.yaml",
                   args.llama_swap_port, out / "logs" / "llama-swap.log") as swap:
        results = bench.smoke_test(swap, scenarios, out / "logs",
                                   indices=list(EXPECTED_INDICES))
    (out / "smoke_results.json").write_text(json.dumps(results, indent=2))
    ok = all(r["ok"] for r in results)
    for r in results:
        print(f"[{'PASS' if r['ok'] else 'FAIL'}] {r['scenario']} ({r['seconds']}s)")
        if not r["ok"]:
            print("   ", r["detail"].splitlines()[0] if r["detail"] else "")
    print("Smoke test:", "all passed" if ok else "FAILURES present")
    return 0 if ok else 1


def cmd_run(args) -> int:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = build_scenarios(args.scenarios, out / "logs", args.include_optional_arm)
    bcfg = load_benchmark_cfg(args.scenarios)
    reps = int(args.reps or bcfg.get("reps", 3))
    cooldown = int(bcfg.get("cooldown_seconds", 20))
    interleave = bool(bcfg.get("interleave_reps", True))

    # 1) static environment + P2P evidence
    cmd_check_env(args)

    # 2) config
    generate_llama_swap_config(scenarios, out / "llama-swap.config.yaml")

    base_url = f"http://127.0.0.1:{args.llama_swap_port}"
    arms = _arms(scenarios)
    concurrencies = _concurrencies(scenarios)

    with LlamaSwap(args.llama_swap_bin, out / "llama-swap.config.yaml",
                   args.llama_swap_port, out / "logs" / "llama-swap.log") as swap:
        # 3) smoke test first; abort the long run if anything is broken
        if args.skip_smoke:
            print("Skipping smoke test (--skip-smoke).")
        else:
            smoke = bench.smoke_test(swap, scenarios, out / "logs",
                                     indices=list(EXPECTED_INDICES))
            (out / "smoke_results.json").write_text(json.dumps(smoke, indent=2))
            if not all(r["ok"] for r in smoke) and not args.force:
                print("Smoke test failed; aborting. Use --force to run anyway.",
                      file=sys.stderr)
                for r in smoke:
                    if not r["ok"]:
                        print(f"  FAIL {r['scenario']}: {r['detail'].splitlines()[0]}")
                return 1

        # 4) timed runs
        vram_threshold = float(bcfg.get("vram_free_threshold_mb", 1024))
        vram_timeout = float(bcfg.get("vram_free_timeout", 180))
        current_model = None
        order = []
        if interleave:
            for rep in range(1, reps + 1):
                for s in scenarios:
                    order.append((s, rep))
        else:
            for s in scenarios:
                for rep in range(1, reps + 1):
                    order.append((s, rep))

        for s, rep in order:
            sdir = out / "scenarios" / s.id
            sdir.mkdir(parents=True, exist_ok=True)
            # preserve the exact per-scenario config + transport proof
            (sdir / "scenario.json").write_text(json.dumps({
                "id": s.id, "arm": s.arm, "concurrency": s.concurrency,
                "env": s.env, "flags": s.flags, "args": s.args,
            }, indent=2))

            # Clean swap: unload the previous model and wait for the driver to
            # actually reclaim VRAM before loading a different scenario, so the
            # new load can't OOM against the outgoing model's memory.
            if current_model is not None and current_model != s.id:
                swap.unload()
                freed = gpu.wait_for_vram_free(list(EXPECTED_INDICES),
                                               vram_threshold, vram_timeout)
                if not freed:
                    print(f"  WARN VRAM still in use after {vram_timeout:.0f}s "
                          f"before loading {s.id}; continuing anyway")

            ok, msg = swap.load_and_wait(s.id, timeout=900)
            if not ok:
                (sdir / f"run_{rep}.ERROR.txt").write_text(msg)
                continue
            current_model = s.id

            # capture transport proof once (first rep)
            if rep == 1:
                Path(s.log_path).exists() and \
                    (sdir / "transport_proof.json").write_text(
                        json.dumps(bench.parse_transport_proof(s.log_path), indent=2))

            if rep == 1 or interleave:
                bench.warmup(base_url, s, bcfg, sdir)

            tel_csv = sdir / ("telemetry.csv" if rep == 1 else f"telemetry_run{rep}.csv")
            window_start = window_end = 0.0
            with TelemetrySampler(tel_csv, indices=list(EXPECTED_INDICES)):
                result, window_start, window_end = bench.run_timed(
                    base_url, s, bcfg, sdir, rep)
            if result is None:
                print(f"  WARN no result JSON for {s.id} rep {rep}")

            # energy for this rep
            avg_w = mean_power_in_window(tel_csv, window_start, window_end,
                                         list(EXPECTED_INDICES))
            (sdir / f"power_run{rep}.json").write_text(json.dumps(
                {"avg_watts": avg_w, "start": window_start, "end": window_end}, indent=2))

            print(f"  done {s.id} rep {rep}"
                  + (f"  ~{avg_w:.0f} W" if avg_w else ""))
            time.sleep(cooldown)

        swap.unload()

    # 5) aggregate + report
    for s in scenarios:
        sdir = out / "scenarios" / s.id
        energy = _energy_for(sdir, reps, s.concurrency, bcfg)
        analyze.aggregate_scenario(sdir, reps, energy)

    summary = analyze.build_summary(out, arms, concurrencies, reps)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    analyze.render_plots(out, arms, concurrencies)
    print(f"\nDone. See {out/'summary.md'}, {out/'summary.csv'}, {out/'plots'}/")
    return 0


def _energy_for(sdir: Path, reps: int, concurrency: int, bcfg: dict) -> dict | None:
    """Approximate tokens/joule from mean power and known generated tokens."""
    out_len = int(bcfg["random_output_len"])
    watts = []
    for rep in range(1, reps + 1):
        p = sdir / f"power_run{rep}.json"
        if p.exists():
            w = json.loads(p.read_text()).get("avg_watts")
            if w:
                watts.append(w)
    if not watts:
        return None
    # tokens/joule is reported per-run by the throughput; here we expose mean W
    # and let the analyst divide output_throughput (tok/s) by mean W (=tok/J).
    return {"mean_watts": round(sum(watts) / len(watts), 1),
            "note": "tokens_per_joule = output_throughput / mean_watts"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="p2pbench",
                                description="Reproducible P2P-on/off vLLM benchmark harness")
    p.add_argument("--scenarios", default=DEFAULT_SPEC, help="scenario matrix YAML")
    p.add_argument("--output-dir", default="./results", help="output directory")
    p.add_argument("--llama-swap-bin", default="llama-swap")
    p.add_argument("--llama-swap-port", type=int, default=8080)
    p.add_argument("--p2p-test-bin", default=None,
                   help="path to cuda-samples p2pBandwidthLatencyTest")
    p.add_argument("--include-optional-arm", action="store_true",
                   help="also run the custom-all-reduce+P2P arm")
    p.add_argument("--reps", type=int, default=None)
    p.add_argument("--force", action="store_true",
                   help="continue even if smoke test fails")
    p.add_argument("--skip-smoke", action="store_true",
                   help="skip the pre-run smoke test (use when scenarios are known-good)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gen-config").set_defaults(func=cmd_gen_config)
    sub.add_parser("check-env").set_defaults(func=cmd_check_env)
    sub.add_parser("smoke").set_defaults(func=cmd_smoke)
    sub.add_parser("run").set_defaults(func=cmd_run)
    rep = sub.add_parser("report")
    rep.set_defaults(func=cmd_report)
    return p


def cmd_report(args) -> int:
    out = Path(args.output_dir)
    scenarios = build_scenarios(args.scenarios, out / "logs", args.include_optional_arm)
    arms = _arms(scenarios)
    concurrencies = _concurrencies(scenarios)
    summary = analyze.build_summary(out, arms, concurrencies,
                                    int(args.reps or 3))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    analyze.render_plots(out, arms, concurrencies)
    print(f"Report written to {out/'summary.md'} and {out/'plots'}/")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
