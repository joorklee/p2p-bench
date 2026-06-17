# Dual RTX 5060 Ti — NVIDIA P2P On vs Off Inference Benchmark

Reproducible benchmark of NVIDIA peer-to-peer (P2P) **on vs off** for tensor-parallel
LLM inference on two consumer **RTX 5060 Ti (16 GB)** cards running
[`Qwen3.6-27B-Text-NVFP4-MTP`](https://huggingface.co/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP)
under vLLM with NCCL `cuMem` P2P transport.

**Repo:** <https://github.com/joorklee/p2p-bench> ·
**Published report:** <https://joorklee.github.io/p2p-bench/>

## Credits & resources

This build stands on the shoulders of others. Thanks to:

- **[r/LocalLLM — "Quad 5060 Ti 16GB OCuLink rig"](https://www.reddit.com/r/LocalLLM/comments/1qfctk3/quad_5060_ti_16gb_oculink_rig/)**
  — the major inspiration for this rig.
- **[`5p00kyy/club-5060ti`](https://github.com/5p00kyy/club-5060ti/)** — reference for
  running multi-5060-Ti setups.
- **[`aikitoria/open-gpu-kernel-modules`](https://github.com/aikitoria/open-gpu-kernel-modules)**
  — patched NVIDIA open GPU kernel modules that enable P2P on consumer GeForce cards (the
  driver that makes the "P2P ON" arm possible).
- **Model:** [`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`](https://huggingface.co/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP).

## 📊 Read the report

The full write-up (with charts) is published via **GitHub Pages** from the
[`docs/`](docs/) folder:

➡️ **[docs/index.md](docs/index.md)** — the report
➡️ **[docs/evidence.md](docs/evidence.md)** — claim-by-claim evidence appendix (every claim
linked to its source file + an excerpt)

## Repository layout

```
.
├── docs/                 # GitHub Pages site
│   ├── index.md          # the full report
│   ├── evidence.md       # claim-by-claim evidence appendix
│   ├── _config.yml       # Jekyll config (Cayman theme)
│   └── assets/           # generated charts (png)
├── p2p-bench/            # the benchmark software + the raw data it produced
│   ├── p2pbench/         # the Python package (CLI + orchestration)
│   ├── config/           # scenario matrices (active + archived per-rig configs)
│   ├── examples/         # sample llama-swap config
│   ├── results/          # result sets produced by the harness:
│   │   └── 2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/
│   │                     #   ↑ the canonical raw data behind the published report
│   │                     #     summary.{csv,json,md}, env/, logs/, plots/, scenarios/
│   └── README.md         # full software docs (commands, options, methodology)
├── make_charts.py        # regenerates docs/assets/*.png from the result set above
└── LICENSE               # GNU GPL v3
```

## Run your own P2P benchmarks

The benchmark software lives in **[`p2p-bench/`](p2p-bench/)**. It is a Python CLI
(`p2pbench`) that drives a `llama-swap`-orchestrated vLLM P2P-on/off concurrency
sweep, capturing per-second telemetry, power, NCCL transport proof, and full host
environment, then aggregating mean±std results with plots.

```bash
cd p2p-bench
pip install -r requirements.txt        # in your working vLLM venv

# 0) verify device order + that P2P is actually engaging (do this first!)
python -m p2pbench --output-dir results/myrig_2cards_$(date +%Y%m%d) \
    --p2p-test-bin ~/cuda-samples/.../p2pBandwidthLatencyTest check-env

# 1) full pipeline: env capture -> smoke -> warmup -> timed reps -> aggregate -> plots
python -m p2pbench --output-dir results/myrig_2cards_$(date +%Y%m%d) run
```

Edit [`p2p-bench/config/scenarios.yaml`](p2p-bench/config/scenarios.yaml) for your
model/rig, or reproduce a published run from an archived config under
[`p2p-bench/config/`](p2p-bench/config/). See the
**[p2p-bench README](p2p-bench/README.md)** for every command, every option, the
config-organization / run-naming conventions, and the methodology rationale (why
only `NCCL_P2P_DISABLE` changes between arms).

> The dual-RTX-5060-Ti config used for the published report is archived at
> [`p2p-bench/config/5060ti-16gb/2-cards/gen3-x4-3.9GBps/scenarios.yaml`](p2p-bench/config/5060ti-16gb/2-cards/gen3-x4-3.9GBps/scenarios.yaml),
> and its result set at
> [`p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/`](p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/).

## Enabling GitHub Pages

In the repo settings → **Pages**, set the source to the **`docs/` folder** on your
default branch. The Cayman theme and `index.md` will render automatically.

## Regenerate the charts

```bash
pip install matplotlib numpy
python3 make_charts.py    # reads p2p-bench/results/<run>/ and writes docs/assets/*.png
```

## License

[GNU General Public License v3.0](LICENSE).
