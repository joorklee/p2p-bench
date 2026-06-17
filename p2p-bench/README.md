# p2pbench — reproducible vLLM P2P on/off benchmark harness

Measures the effect of GPU peer-to-peer (P2P) on tensor-parallel vLLM inference
throughput/latency across a concurrency sweep, on a 2× consumer-GPU PCIe rig
(no NVLink). Built around a `llama-swap` config so each scenario launches its own
vLLM instance with its own environment, and driven end-to-end by a Python CLI
that captures everything a reviewer (or r/LocalLLaMA) will ask for.

> Target rig: dual RTX 5060 Ti (16 GB), tensor-parallel-size 2, NVFP4 weights,
> fp8 KV, MTP speculative decoding. Adjust `config/scenarios.yaml` for yours.

### Credits & resources

Getting consumer-GPU P2P working relied on these:

- **[`aikitoria/open-gpu-kernel-modules`](https://github.com/aikitoria/open-gpu-kernel-modules)**
  — patched NVIDIA open GPU kernel modules that enable P2P on consumer GeForce cards.
  Build/install these (matching your driver version) before P2P will engage.
- **[`5p00kyy/club-5060ti`](https://github.com/5p00kyy/club-5060ti/)** — multi-5060-Ti
  setup reference.
- **[r/LocalLLM — Quad 5060 Ti 16GB OCuLink rig](https://www.reddit.com/r/LocalLLM/comments/1qfctk3/quad_5060_ti_16gb_oculink_rig/)**
  — rig inspiration.
- **Model used in the reference run:**
  [`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`](https://huggingface.co/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP).

---

## Why the matrix looks like this (read before you trust any number)

The single most important property of a P2P benchmark is that **only P2P
changes between arms.** The original hand-written configs changed three things
at once, which would have invalidated the comparison:

| variable | original "P2P on" | original "P2P off" | problem |
|---|---|---|---|
| `NCCL_P2P_DISABLE` | 0 | 1 | the intended variable — keep |
| `NCCL_CUMEM_ENABLE` | 1 | 0 | **co-varied** — confound |
| `--disable-custom-all-reduce` | present | commented out | **co-varied** — confound |

Two are confounds, and the second is worse than it looks:

**vLLM's custom all-reduce uses its own CUDA peer check, not `NCCL_P2P_DISABLE`.**
In vLLM, custom all-reduce decides whether to engage by calling
`gpu_p2p_access_check` / `torch.cuda.can_device_access_peer` — a CUDA-level test
that is independent of the NCCL transport env var. So if custom all-reduce is
left enabled (flag commented out) in the "P2P off" arm and the hardware reports
peer access, vLLM will **keep talking over CUDA P2P through its own kernel**,
bypassing `NCCL_P2P_DISABLE=1` entirely. Your "P2P off" run silently still uses
P2P, and you end up comparing *custom-AR-over-P2P* vs *NCCL-over-P2P* — not
P2P on vs off.

**The fix encoded in `scenarios.yaml`:**

- Hold `--disable-custom-all-reduce` **present in both** primary arms → NCCL is
  the communicator in both, so `NCCL_P2P_DISABLE` actually controls the path.
- Pin `NCCL_CUMEM_ENABLE=1` in **both** arms.
- Flip **only** `NCCL_P2P_DISABLE` (0 → 1). With it = 1, NCCL falls back to the
  SHM/host-staged path — the real "no P2P" baseline.
- Optional third arm (`--include-optional-arm`) leaves custom all-reduce enabled
  with P2P on, giving a clean 3-way decomposition:
  `NCCL+noP2P` vs `NCCL+P2P` vs `customAR+P2P`.

This mirrors how the literature isolates the interconnect variable (e.g.
link-speed sweeps measuring all-reduce time and TTFT) and how comm-library work
treats NCCL vs custom-kernel all-reduce as *distinct* paths rather than
interchangeable.

### Other methodology choices baked in

- **Verify P2P is even real first.** On consumer Blackwell over PCIe, P2P often
  is not available by default (some 2× 5090 setups report no P2P capability; on
  4090s/5060 Tis people patch the open kernel module to force-enable it — see
  [`aikitoria/open-gpu-kernel-modules`](https://github.com/aikitoria/open-gpu-kernel-modules)).
  `check-env`
  runs `nvidia-smi topo -p2p`, an optional cuda-samples `p2pBandwidthLatencyTest`,
  and a torch D2D-bandwidth probe. **If P2P=Enabled bandwidth ≈ P2P=Disabled,
  P2P is not engaging and the comparison is meaningless** — fix the driver first.
- **`NCCL_DEBUG=INFO`, not `TRACE`, for timed runs.** INFO prints the channel /
  transport summary at init (proof of P2P vs SHM) without TRACE's per-op logging,
  which adds I/O overhead that differs by path and would bias one arm.
- **Deterministic, fixed-length workload.** `--dataset-name random` with
  `--random-range-ratio 0` + `--ignore-eos` + fixed seed → identical token
  counts across arms. Essential with MTP spec decoding, where acceptance depends
  on the prompt; identical inputs keep tok/s apples-to-apples.
- **Prefix caching OFF by default.** With repeated prompts it skips prefill and
  inflates throughput unrealistically. Turn on only with unique-prefix prompts.
- **Thermal/clock drift control.** Reps are interleaved (A,B,A,B…) rather than
  all-A-then-all-B, with a cooldown between runs; SM clock + temp are logged per
  second so you can discard throttled runs. Consider also pinning clocks
  (`nvidia-smi -lgc`) / power (`nvidia-smi -pl`) and persistence mode.
- **Client concurrency matches server.** `--max-concurrency N` (client) ==
  `--max-num-seqs N` (server) so you measure server concurrency, not queueing.
- **Energy too.** Power is sampled per second; tokens/joule = output_throughput
  ÷ mean watts. P2P can change energy even when throughput is flat (less host
  bounce), and it is the kind of axis that earns trust.

---

## What it captures (per run, into `results/`)

```
results/<ts>/
  llama-swap.config.yaml          generated from scenarios.yaml
  smoke_results.json              per-scenario startup/crash check
  summary.md / summary.csv        the table you paste; mean±std + P2P delta
  summary.json
  plots/                          throughput, p99 TPOT, P2P gain%, telemetry timeline
  env/
    kernel_cmdline.txt            /proc/cmdline + GRUB_CMDLINE_LINUX_DEFAULT
    kernel_modules.txt            nvidia modules, nvidia_peermem loaded?, modinfo
    sysctl.txt
    dmesg_gpu.txt                 dmesg/journal grep: nvidia|nvrm|xid|pcie|aer
    p2p_evidence.json             topo -m, topo -p2p, cuda-samples test, torch probe
    host_summary.json
  scenarios/<arm>-c<N>/
    scenario.json                 exact env/flags/args preserved
    transport_proof.json          custom-AR status + NCCL via P2P/SHM from the log
    run_1.json run_2.json run_3.json   raw vllm bench serve results
    telemetry.csv                 per-second pcie gen/width, temp, util, mem, power, sm clk
    power_run*.json               mean watts in each timed window
    aggregate.json                mean/std across reps
```

---

## Requirements

- A working **vLLM venv** (vLLM + torch + your model already serving correctly
  by hand). The harness shells out to `vllm bench serve` and `vllm serve`.
- [`llama-swap`](https://github.com/mostlygeek/llama-swap) on `PATH` (or pass
  `--llama-swap-bin`).
- Harness deps: `pip install -r requirements.txt` (PyYAML, matplotlib).
- Optional but recommended: build cuda-samples and pass
  `--p2p-test-bin /path/to/p2pBandwidthLatencyTest` for authoritative P2P proof.
- `dmesg`/`journalctl` capture may need root; it degrades gracefully if not.

> Version note: `llama-swap`'s management endpoints and a few `vllm serve` flags
> (`--attention-backend`, `--language-model-only`, `--generation-config`) have
> changed across releases. Run `vllm serve --help` and confirm they parse in your
> build; the unload endpoint path is configurable in `swap.py`.

---

## Install

```bash
# inside your working vLLM venv (the one that already serves your model by hand)
pip install -r requirements.txt
# optional: install the `p2pbench` console entry point
pip install -e .
```

You can invoke the harness either way:

```bash
python -m p2pbench ...     # module form (no install needed beyond requirements.txt)
p2pbench ...               # console script (after `pip install -e .`)
```

## Quick start

```bash
RUN=results/5060ti-16gb_2cards_gen3-x4_$(date +%Y%m%d)   # see "Organizing runs" below

# 0) sanity: device order + P2P actually working (do this FIRST, every new rig)
python -m p2pbench --output-dir "$RUN" \
    --p2p-test-bin ~/cuda-samples/.../p2pBandwidthLatencyTest check-env

# 1) just generate the llama-swap config from the scenario matrix
python -m p2pbench --output-dir "$RUN" gen-config

# 2) crash-test every scenario (startup + 1-token request) before the long run
python -m p2pbench --output-dir "$RUN" smoke

# 3) full pipeline: env capture -> smoke -> warmup -> 3 timed reps -> aggregate -> plots
python -m p2pbench --output-dir "$RUN" run

# add the custom-all-reduce+P2P arm for the 3-way decomposition
python -m p2pbench --output-dir "$RUN" --include-optional-arm run

# 4) re-render report/plots from existing results (no GPUs needed)
python -m p2pbench --output-dir "$RUN" report
```

To run an **archived rig config** instead of the default `config/scenarios.yaml`:

```bash
python -m p2pbench \
    --scenarios config/5060ti-16gb/2-cards/gen3-x4-3.9GBps/scenarios.yaml \
    --output-dir "$RUN" --p2p-test-bin ~/cuda-samples/.../p2pBandwidthLatencyTest run
```

---

## Commands

`p2pbench <global-options> <command>`. Every command takes `--output-dir`.

| Command | What it does | Needs GPUs? |
|---|---|---|
| `check-env` | Captures host env (kernel cmdline, modules, sysctl, dmesg) and **P2P evidence** (`nvidia-smi topo -m`/`-p2p`, optional cuda-samples bandwidth/latency test, torch D2D probe). Verifies the expected GPU device order. **Run this first** — if P2P=Enabled bandwidth ≈ P2P=Disabled, P2P isn't engaging and the comparison is meaningless. | yes |
| `gen-config` | Generates `llama-swap.config.yaml` from the scenario matrix and exits. Good for inspecting exactly what will be launched. | no |
| `smoke` | Boots every scenario through llama-swap and fires a 1-token request to catch startup/flag-parse/OOM failures **before** committing to the long timed run. | yes |
| `run` | The full pipeline: `check-env` → `gen-config` → `smoke` → per-scenario warmup → N interleaved timed reps with per-second telemetry + power → aggregate (mean/std) → summary tables + plots. | yes |
| `report` | Re-builds `summary.{md,csv,json}` and `plots/` from data already in `--output-dir`. Use after tweaking analysis, or to regenerate on a machine without GPUs. | no |

### Global options

| Option | Default | Meaning |
|---|---|---|
| `--scenarios PATH` | `config/scenarios.yaml` | Scenario matrix YAML to use. Point at an archived rig config to reproduce a published run. |
| `--output-dir PATH` | `./results` | Where everything for this run is written. **Use a descriptive, unique name** (see below). |
| `--llama-swap-bin NAME` | `llama-swap` | Path/name of the `llama-swap` binary if not on `PATH`. |
| `--llama-swap-port N` | `8080` | Port llama-swap listens on (the client benchmarks against this). |
| `--p2p-test-bin PATH` | _none_ | Path to the cuda-samples `p2pBandwidthLatencyTest` binary for authoritative P2P proof. Highly recommended. |
| `--include-optional-arm` | off | Also run the `custom_ar_p2p_on` arm → 3-way decomposition: `NCCL+noP2P` vs `NCCL+P2P` vs `customAR+P2P`. |
| `--reps N` | from YAML (`3`) | Override the number of timed repetitions per scenario. |
| `--skip-smoke` | off | Skip the pre-run smoke test (only when scenarios are known-good). |
| `--force` | off | Continue the timed run even if the smoke test reports failures. |

> The expected GPU indices for the device-order check are `(1, 2)`. If your two
> target GPUs sit at different indices, set `CUDA_VISIBLE_DEVICES` in
> `constant_env` accordingly and update `EXPECTED_INDICES` in `p2pbench/cli.py`.

---

## Organizing configs and naming runs

Two simple conventions keep results reproducible and self-describing.

**1. Archive the config that produced each result set.** Configs live under
`config/<gpu-model>/<n-cards>/<link-bandwidth>/scenarios.yaml` — e.g.
`config/5060ti-16gb/2-cards/gen3-x4-3.9GBps/scenarios.yaml`. The `<link-bandwidth>`
segment names the **effective** PCIe link the run actually trained at, labelled by
its theoretical max one-way bandwidth (e.g. `gen3-x4-3.9GBps`, `gen4-x4-7.9GBps`).
See [`config/README.md`](config/README.md) for the full convention and a
gen/lane → GB/s table.

**2. Name `--output-dir` so the run is identifiable at a glance.** Recommended
pattern:

```
results/<date>_<gpu-model>_<n-cards>_<link-bandwidth>_<what-was-swept>
```

e.g. the bundled result set:

```
results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/
```

A run directory is fully self-contained (config snapshot, env, per-scenario
telemetry/logs/aggregates, summary tables, plots) — see the layout above — so you
can hand someone just that folder and they have everything needed to audit it.

---

## Adapting to your rig

Everything lives in `config/scenarios.yaml`: model path/port, the constant
(controlled) args/flags/env, the per-arm `NCCL_P2P_DISABLE` delta, the
concurrency list, and the benchmark workload (input/output lengths, prompts,
reps). The expected GPU indices for the device-order check are `(1, 2)` — change
`EXPECTED_INDICES` in `p2pbench/cli.py` if yours differ.

## Limitations / honest caveats

- tokens/joule uses board power (`power.draw`), not wall power.
- The torch probe is a coarse P2P signal; cuda-samples is authoritative.
- MTP acceptance makes absolute tok/s workload-dependent; the P2P *delta* is
  valid under fixed inputs, but absolute numbers won't match a real chat trace.
  Run a ShareGPT-backed variant separately if you want representative absolutes.
- Default `reps: 3` is enough to see the trend and report mean±std, but it is a
  small sample — individual scenarios can show meaningful run-to-run variance, so
  treat single-digit-percent deltas as suggestive rather than definitive. Bump
  `reps` (or `--reps`) for tighter confidence.

## License

[GNU General Public License v3.0](../LICENSE) (`LICENSE` at the repo root).
