# Scenario configs

This directory holds the `scenarios.yaml` matrices that drive `p2pbench`.

- [`scenarios.yaml`](scenarios.yaml) — the **active/default** config. This is the
  file `p2pbench` reads when you don't pass `--scenarios`. Edit it for ad-hoc runs,
  or point `--scenarios` at one of the archived configs below.
- `<gpu-model>/<n-cards>/<link-bandwidth>/scenarios.yaml` — **archived configs**,
  one per rig + interconnect combination, so the exact config that produced a given
  result set is always recoverable.

## Naming convention

```
config/
  scenarios.yaml                                  # active default
  <gpu-model>/                                    # e.g. 5060ti-16gb
    <n-cards>/                                    # e.g. 2-cards
      <link-bandwidth>/                           # e.g. gen3-x4-3.9GBps
        scenarios.yaml                            # exact config used for that run
```

| Segment | Meaning | Examples |
|---|---|---|
| `<gpu-model>` | GPU SKU + VRAM, lower-case, no spaces | `5060ti-16gb`, `3090-24gb` |
| `<n-cards>` | how many GPUs were tensor-parallel | `2-cards`, `4-cards` |
| `<link-bandwidth>` | the **effective** PCIe link the run actually trained at, named by its theoretical max one-way bandwidth | `gen3-x4-3.9GBps`, `gen4-x4-7.9GBps`, `gen4-x16-31.5GBps` |

**Why bandwidth and not just "gen/lanes"?** The folder name encodes the *effective*
link the benchmark ran under (what `nvidia-smi`/telemetry reported), and labels it
by the theoretical max one-way bandwidth so it's comparable across gen/lane combos.

PCIe one-way bandwidth quick reference (per lane ≈ usable GB/s after encoding):

| Gen | per-lane | x3 | x4 | x8 | x16 |
|---|---|---|---|---|---|
| Gen3 | ~0.985 GB/s | ~2.9 | **~3.9** | ~7.9 | ~15.8 |
| Gen4 | ~1.969 GB/s | **~5.9** | ~7.9 | ~15.8 | ~31.5 |
| Gen5 | ~3.938 GB/s | ~11.8 | ~15.8 | ~31.5 | ~63.0 |

## Current archive

- [`5060ti-16gb/2-cards/gen3-x4-3.9GBps/scenarios.yaml`](5060ti-16gb/2-cards/gen3-x4-3.9GBps/scenarios.yaml)
  — dual RTX 5060 Ti 16 GB, PCIe **Gen3 x4** (~3.9 GB/s, as reported by telemetry/nvidia-smi).
  Produced result set
  [`results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/`](../results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/).

## Reproducing an archived run

```bash
python -m p2pbench \
  --scenarios config/5060ti-16gb/2-cards/gen3-x4-3.9GBps/scenarios.yaml \
  --output-dir results/<gpu>_<cards>_<link>_<date> \
  --p2p-test-bin /path/to/p2pBandwidthLatencyTest \
  run
```
