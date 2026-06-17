---
title: "Evidence Appendix — Claim-by-Claim Verification"
---

# Evidence Appendix — Claim-by-Claim Verification

[← Back to the main report](index.md)

This page backs **every substantive claim** in the
[main report](index.md) with (a) a link to the underlying data file and (b) a short
**excerpt** from that file, so you can confirm a claim without digging through the raw
data yourself. If you *do* want the raw data, every excerpt links to its source.

Each claim below has a stable anchor (e.g. `#c11-throughput`) that the report links to.

**Conventions**
- The benchmark **data root** is
  [`p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/).
  File paths in the excerpts below (e.g. `scenarios/p2p_on-c1/aggregate.json`) are **relative
  to that directory**; the hyperlinks resolve to the full path.
- GPU device indices: **0 = RTX 5070 (unused)**, **1 & 2 = the two RTX 5060 Ti's under test**.
- "P2P ON" = `NCCL_P2P_DISABLE=0`; "P2P OFF" = `NCCL_P2P_DISABLE=1`. Both use
  `NCCL_CUMEM_ENABLE=1`.
- **JSON/log excerpts below are trimmed for readability** — the linked source files
  contain additional keys and surrounding context. Values are quoted verbatim; only the
  surrounding structure is abbreviated (`...`). Open the linked file for the full content.

---

## Index of claims

- **Benchmark methodology**
  - [C1 — Model, quantization & MTP config](#c1-model-config)
  - [C2 — Tensor-parallel = 2, fp8 KV cache, 64K context, 16384 batch tokens](#c2-vllm-args)
  - [C3 — Two RTX 5060 Ti's (devices 1 & 2) under test; RTX 5070 excluded](#c3-devices)
  - [C4 — NCCL version 2.28.9](#c4-nccl-version)
- **P2P transport**
  - [C5 — P2P ON uses `via P2P/CUMEM`](#c5-p2p-used)
  - [C6 — P2P OFF falls back to `via SHM/direct/direct`](#c6-shm-fallback)
  - [C7 — The two 5060 Ti's CAN access each other as peers](#c7-peer-access)
  - [C8 — P2P writes cut inter-GPU latency 18.08 µs → 0.42 µs](#c8-latency-drop)
  - [C9 — `NCCL_CUMEM_ENABLE=1` is set](#c9-cumem)
  - [C10 — Topology is PHB (through PCIe host bridge), no NVLink](#c10-topo)
- **Throughput results**
  - [C11 — Output throughput per scenario (mean of 3 runs)](#c11-throughput)
  - [C12 — P2P uplift: +9.0% / +12.6% / +8.3% / +13.9%](#c12-uplift)
- **Latency / MTP**
  - [C13 — ITL ~45–80 ms vs TPOT ~16–28 ms; MTP = 3 spec tokens](#c13-itl-mtp)
- **Energy**
  - [C14 — Combined GPU power 160–201 W; tokens/joule](#c14-power)
- **PCIe link state**
  - [C15 — Both 5060 Ti's ran at PCIe Gen3 x4 the entire benchmark](#c15-gen3x4)
  - [C16 — `nvidia-smi` on host confirms Gen3 / x4 / max x16](#c16-host-link)
- **Telemetry**
  - [C17 — Temps peak 62–64 °C, avg 51–55 °C (no throttling)](#c17-temp)
  - [C18 — GPU utilization 94–97% avg, hitting 100%](#c18-util)
  - [C19 — Peak ~104–109 W per card; avg drops as concurrency rises](#c19-power-detail)
  - [C20 — Up to ~14.8 GiB VRAM used at c8](#c20-mem)
  - [C21 — SM clocks steady ~2.8 GHz](#c21-clocks)
- **System / driver / kernel modules**
  - [C22 — Base NVIDIA driver 595.71.05](#c22-driver)
  - [C23 — Patched open kernel modules at `/data/open-gpu-kernel-modules`](#c23-patched-module)
  - [C24 — `nvidia_peermem` NOT loaded](#c24-peermem)
  - [C25 — CUDA toolkit 13.2](#c25-cuda)
  - [C26 — Hardware: Ryzen 9 3900X, 64 GB DDR4 ~2133 MT/s, Proxmox kernel 6.17.13-13-pve](#c26-hardware)
- **Boot / tuning**
  - [C27 — GRUB cmdline: `pcie_aspm=off`, `iommu=pt`, `amd_iommu=on`](#c27-grub)
  - [C28 — sysctl: `numa_balancing=0`, `zone_reclaim_mode=0`](#c28-sysctl)

---

## Benchmark methodology

### C1 — Model, quantization & MTP config {#c1-model-config}

> **Claim:** The model is
> [`Qwen3.6-27B-Text-NVFP4-MTP`](https://huggingface.co/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP),
> run with NVFP4 (`modelopt`) quantization and **MTP speculative decoding with 3 speculative
> tokens**.

**Evidence:** [`scenarios/p2p_on-c1/scenario.json`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_on-c1/scenario.json)

```json
"args": {
  "quantization": "modelopt",
  "speculative-config": "{\"method\":\"mtp\",\"num_speculative_tokens\":3}",
  ...
}
```

Also confirmed in the live engine config, [`logs/p2p_on-c1.log`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/logs/p2p_on-c1.log):

```
speculative_config=SpeculativeConfig(method='mtp', model='/data/models/qwen3.6-27b-text-nvfp4-mtp', num_spec_tokens=3)
```

And the tokenizer/model path in [`scenarios/p2p_on-c1/run_1.json`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_on-c1/run_1.json):
`"tokenizer_id": "/data/models/qwen3.6-27b-text-nvfp4-mtp"`.

---

### C2 — TP=2, fp8 KV cache, 64K context, 16384 batch tokens {#c2-vllm-args}

> **Claim:** `tensor-parallel-size=2`, `kv-cache-dtype=fp8`, `max-model-len=65535`,
> `max-num-batched-tokens=16384`, `gpu-memory-utilization=0.86`,
> `attention-backend=TRITON_ATTN`, `disable-custom-all-reduce`.

**Evidence:** [`scenarios/p2p_on-c1/scenario.json`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_on-c1/scenario.json)

```json
"flags": ["language-model-only", "enable-auto-tool-choice", "disable-custom-all-reduce"],
"args": {
  "tensor-parallel-size": 2,
  "max-model-len": 65535,
  "kv-cache-dtype": "fp8",
  "max-num-batched-tokens": 16384,
  "gpu-memory-utilization": 0.86,
  "attention-backend": "TRITON_ATTN"
}
```

The exact launch command per scenario is in
[`llama-swap.config.yaml`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/llama-swap.config.yaml).

---

### C3 — Two RTX 5060 Ti's under test; RTX 5070 excluded {#c3-devices}

> **Claim:** Devices 1 & 2 (the two RTX 5060 Ti's, bus `09:00.0` / `0A:00.0`) are under
> test; the RTX 5070 (device 0) is present but **not** used (`CUDA_VISIBLE_DEVICES=1,2`).

**Evidence:** [`env/p2p_evidence.json`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/env/p2p_evidence.json) (`device_order.all_gpus`):

```
0  NVIDIA GeForce RTX 5070      00000000:04:00.0
1  NVIDIA GeForce RTX 5060 Ti   00000000:09:00.0
2  NVIDIA GeForce RTX 5060 Ti   00000000:0A:00.0
```

And [`scenarios/p2p_on-c1/scenario.json`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_on-c1/scenario.json):

```json
"env": { "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1,2", ... }
```

---

### C4 — NCCL version 2.28.9 {#c4-nccl-version}

> **Claim:** NCCL 2.28.9.

**Evidence:** [`scenarios/p2p_on-c1/transport_proof.json`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_on-c1/transport_proof.json)

```json
{ "nccl_used_p2p": true, "nccl_used_shm": false, "nccl_version": "2.28.9" }
```

---

## P2P transport

### C5 — P2P ON uses `via P2P/CUMEM` {#c5-p2p-used}

> **Claim:** With P2P enabled, NCCL maps channels directly over the bus via the cuMem
> allocator (`via P2P/CUMEM`), and `nccl_used_p2p = true`, `nccl_used_shm = false`.

**Evidence:** [`logs/p2p_on-c1.log`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/logs/p2p_on-c1.log) (lines 113–122):

```
proxmox:1417089 [1] NCCL INFO Check P2P Type isAllDirectP2p 1 directMode 0 isAllCudaP2p 1
proxmox:1417089 [1] NCCL INFO Channel 00/0 : 1[2] -> 0[1] via P2P/CUMEM
proxmox:1417088 [0] NCCL INFO Channel 00/0 : 0[1] -> 1[2] via P2P/CUMEM
```

Parsed summary — [`scenarios/p2p_on-c1/transport_proof.json`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_on-c1/transport_proof.json):

```json
{ "nccl_used_p2p": true, "nccl_used_shm": false, "nccl_version": "2.28.9" }
```

*(All four P2P-ON scenarios report `nccl_used_p2p: true`:
[c1](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_on-c1/transport_proof.json),
[c2](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_on-c2/transport_proof.json),
[c4](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_on-c4/transport_proof.json),
[c8](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_on-c8/transport_proof.json).)*

---

### C6 — P2P OFF falls back to `via SHM/direct/direct` {#c6-shm-fallback}

> **Claim:** With `NCCL_P2P_DISABLE=1`, NCCL falls back to shared-host-memory staging
> (`via SHM/direct/direct`), `nccl_used_p2p = false`.

**Evidence:** [`logs/p2p_off-c1.log`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/logs/p2p_off-c1.log) (lines 107–125):

```
proxmox:1495819 [0] NCCL INFO NCCL_P2P_DISABLE set by environment to 1
proxmox:1495819 [0] NCCL INFO Check P2P Type isAllDirectP2p 0 directMode 0 isAllCudaP2p 1
proxmox:1495819 [0] NCCL INFO Channel 00 : 0[1] -> 1[2] via SHM/direct/direct
```

Parsed summary — [`scenarios/p2p_off-c1/transport_proof.json`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_off-c1/transport_proof.json):

```json
{ "nccl_used_p2p": false, "nccl_used_shm": true, "nccl_version": "2.28.9" }
```

*(All four P2P-OFF scenarios report `nccl_used_shm: true`:
[c1](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_off-c1/transport_proof.json),
[c2](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_off-c2/transport_proof.json),
[c4](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_off-c4/transport_proof.json),
[c8](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_off-c8/transport_proof.json).)*

---

### C7 — The two 5060 Ti's CAN access each other as peers {#c7-peer-access}

> **Claim:** Devices 1 & 2 (the 5060 Ti's) can access each other as peers; the 5070
> (device 0) cannot peer with them.

**Evidence:** [`env/p2p_evidence.json`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/env/p2p_evidence.json) (`cuda_samples_p2p`):

```
Device=1 CAN Access Peer Device=2
Device=2 CAN Access Peer Device=1
Device=0 CANNOT Access Peer Device=1
Device=0 CANNOT Access Peer Device=2
```

The `topo_p2p` matrix in the same file shows `OK` between GPU1 and GPU2.

---

### C8 — P2P writes cut inter-GPU latency 18.08 µs → 0.42 µs {#c8-latency-drop}

> **Claim:** Enabling P2P writes drops the GPU1↔GPU2 latency dramatically (≈18 µs → sub-µs).

**Evidence:** [`env/p2p_evidence.json`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/env/p2p_evidence.json) (`cuda_samples_p2p`):

```
P2P=Disabled Latency Matrix (us)
   GPU     0      1      2
     1  20.92   1.10  18.08
     2  21.25  18.22   1.07

P2P=Enabled Latency (P2P Writes) Matrix (us)
   GPU     0      1      2
     1  21.26   1.01   0.42
```

The `1 → 2` entry drops from **18.08 µs** (disabled) to **0.42 µs** (enabled).

---

### C9 — `NCCL_CUMEM_ENABLE=1` is set {#c9-cumem}

> **Claim:** The cuMem allocator path is enabled in both arms.

**Evidence:** [`scenarios/p2p_on-c1/scenario.json`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_on-c1/scenario.json)
(`"NCCL_CUMEM_ENABLE": "1"`) and [`logs/p2p_on-c1.log`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/logs/p2p_on-c1.log):

```
proxmox:1417089 [1] NCCL INFO NCCL_CUMEM_ENABLE set by environment to 1.
```

Confirmed in the OFF arm too — [`logs/p2p_off-c1.log`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/logs/p2p_off-c1.log):
`NCCL_CUMEM_ENABLE set by environment to 1.`

---

### C10 — Topology is PHB (through PCIe host bridge), no NVLink {#c10-topo}

> **Claim:** The two cards connect peer-to-peer through the CPU's PCIe host bridge (`PHB`);
> there is no NVLink.

**Evidence:** [`env/p2p_evidence.json`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/env/p2p_evidence.json) (`topo_matrix`):

```
        GPU0   GPU1   GPU2
GPU0     X     PHB    PHB
GPU1    PHB     X     PHB
GPU2    PHB    PHB     X
```

`PHB = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)`. No
`NV#` (NVLink) entries appear anywhere in the matrix.

---

## Throughput results

### C11 — Output throughput per scenario (mean of 3 runs) {#c11-throughput}

> **Claim:** Output throughput (mean ± std of 3 runs):
> c1 60.3/55.3, c2 107.6/95.5, c4 170.6/157.6, c8 190.9/167.6 (ON/OFF tok/s).

**Evidence:** [`summary.csv`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/summary.csv):

```
arm,concurrency,output_tok_s,output_tok_s_std,total_tok_s,...
p2p_on,1,60.2863,4.7727,120.5727,...
p2p_on,2,107.5812,3.9506,215.1625,...
p2p_on,4,170.6365,4.5889,341.2731,...
p2p_on,8,190.88,3.1895,381.7601,...
p2p_off,1,55.31,4.0017,110.62,...
p2p_off,2,95.5267,2.6816,191.0534,...
p2p_off,4,157.567,5.2146,315.1339,...
p2p_off,8,167.6107,2.0935,335.2215,...
```

Per-scenario detail (with the 3 individual run values) is in each
`aggregate.json`, e.g. [`scenarios/p2p_on-c1/aggregate.json`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_on-c1/aggregate.json)
(the metric lives under the top-level `metrics` key — `metrics.output_throughput`):

```json
"metrics": { "output_throughput": { "mean": 60.2863, "std": 4.7727, "values": [63.9867, 63.3248, 53.5476] }, ... }
```

---

### C12 — P2P uplift: +9.0% / +12.6% / +8.3% / +13.9% {#c12-uplift}

> **Claim:** P2P output-throughput uplift over the SHM fallback is +9.0% (c1), +12.6% (c2),
> +8.3% (c4), +13.9% (c8).

**Evidence:** [`summary.md`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/summary.md):

```
| Concurrency | p2p_on | p2p_off | P2P delta |
| 1 | 60.3 ±4.8 | 55.3 ±4.0 | +9.0% |
| 2 | 107.6 ±4.0 | 95.5 ±2.7 | +12.6% |
| 4 | 170.6 ±4.6 | 157.6 ±5.2 | +8.3% |
| 8 | 190.9 ±3.2 | 167.6 ±2.1 | +13.9% |
```

(Derivable from C11: e.g. c1 = 60.2863 / 55.31 − 1 = +9.0%.)

---

## Latency / MTP

### C13 — ITL ~45–80 ms vs TPOT ~16–28 ms; MTP = 3 spec tokens {#c13-itl-mtp}

> **Claim:** Mean ITL is ~45–80 ms while mean TPOT is ~16–28 ms, and the model runs MTP
> with 3 speculative tokens — so ITL spans a block of up to 3 tokens, not a single token.

**Evidence (MTP):** see [C1](#c1-model-config) — `num_speculative_tokens: 3`.

**Evidence (the two metrics):** [`scenarios/p2p_on-c1/run_1.json`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_on-c1/run_1.json):

```
"mean_itl_ms": 45.42541805380697,
"mean_tpot_ms": 14.905908932489746
```

Aggregated extremes across scenarios — [`scenarios/p2p_off-c8/aggregate.json`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_off-c8/aggregate.json)
shows the high end (`mean_itl_ms.mean ≈ 79.7`, `mean_tpot_ms.mean ≈ 28.2`); the full
per-scenario table is in the [report](index.md#results-at-a-glance) and every
`aggregate.json`.

---

## Energy

### C14 — Combined GPU power 160–201 W; tokens/joule {#c14-power}

> **Claim:** Mean combined GPU power ranges ~160–201 W across scenarios; tokens-per-joule
> = output tok/s ÷ mean watts.

**Evidence:** `energy.mean_watts` in each `aggregate.json`:

- [`p2p_on-c1`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_on-c1/aggregate.json): `"mean_watts": 200.6`
- [`p2p_off-c8`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_off-c8/aggregate.json): `"mean_watts": 159.6`

```json
"energy": { "mean_watts": 200.6, "note": "tokens_per_joule = output_throughput / mean_watts" }
```

Raw per-iteration power integrals are in each scenario's
[`power_run1.json`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_on-c1/power_run1.json) etc.:

```json
{ "avg_watts": 200.89462311557787, "start": 1781643450.69, "end": 1781643848.16 }
```

---

## PCIe link state

### C15 — Both 5060 Ti's ran at PCIe Gen3 x4 the entire benchmark {#c15-gen3x4}

> **Claim:** Every telemetry sample, across all 8 scenarios and all 3 iterations, reports
> `pcie.link.gen.current = 3`, `pcie.link.width.current = 4`, `pcie.link.width.max = 16`.

**Evidence:** telemetry CSV header + sample,
[`scenarios/p2p_on-c8/telemetry.csv`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_on-c8/telemetry.csv):

```
wall_epoch,index,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,...
1781646024.270,1,3,3,4,16,52,0,0,14765,92.24,2775,13801
1781646024.270,2,3,3,4,16,58,0,0,14765,92.09,2797,13801
```

Aggregating **every** `telemetry*.csv` in `scenarios/`, the set of distinct values is:
`gen.current = {3}`, `width.current = {4}`, `width.max = {16}` — i.e. it never deviates from
**Gen3 x4** for the whole run.

---

### C16 — `nvidia-smi` on host confirms Gen3 / x4 / max x16 {#c16-host-link}

> **Claim:** A live `nvidia-smi` query on the host confirms the 5060 Ti's
> negotiate PCIe Gen3, x4 current width, x16 slot max width.

**Evidence:** captured on the host with
`nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv`:

```
name, driver_version, pcie.link.gen.current, pcie.link.gen.max, pcie.link.width.current, pcie.link.width.max
NVIDIA GeForce RTX 5060 Ti, 595.71.05, 1, 3, 4, 16
NVIDIA GeForce RTX 5060 Ti, 595.71.05, 1, 3, 4, 16
```

*(Note: `gen.current = 1` here because the cards were idle at query time and downclock the
link to Gen1 to save power; `gen.max = 3` and `width = 4/16` are the relevant negotiated
ceiling. Under load — see C15 — the telemetry shows the link running at Gen3 x4.)*

---

## Telemetry

### C17 — Temps peak 62–64 °C, avg 51–55 °C (no throttling) {#c17-temp}

> **Claim:** Across scenarios, peak GPU temperature is 62–64 °C and average is 51–55 °C —
> no thermal throttling.

**Evidence:** computed from the `temperature.gpu` column of every
[`telemetry*.csv`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_on-c8/telemetry.csv). Per-scenario maxima/means:

| Scenario | Max °C | Avg °C |
|---|---|---|
| p2p_on-c1 | 64 | 55 |
| p2p_on-c8 | 64 | 52 |
| p2p_off-c4 | 62 | 52 |

(Full table in the [report](index.md#what-the-telemetry-tells-us). Source column is
`temperature.gpu` in each CSV.)

---

### C18 — GPU utilization 94–97% avg, hitting 100% {#c18-util}

> **Claim:** Average GPU utilization is 94–97% and frequently reaches 100% — the GPUs are
> the bottleneck, not the PCIe link.

**Evidence:** `utilization.gpu` column of the
[`telemetry*.csv`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_off-c8/telemetry.csv) files. Aggregated: max = 100
in every scenario; averages range 94 (c1) → 97 (off-c8). Example row from
[`scenarios/p2p_on-c8/telemetry.csv`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_on-c8/telemetry.csv) shows
the util field (8th column) climbing to 100 under load.

---

### C19 — Peak ~104–109 W per card; avg drops as concurrency rises {#c19-power-detail}

> **Claim:** Peak per-card power is ~104–109 W; the *average* combined power drops as
> concurrency rises (≈96 W at c1 → ≈80 W at c8 for P2P OFF).

**Evidence:** `power.draw` column of the telemetry CSVs, and `energy.mean_watts` in the
aggregates (see [C14](#c14-power)). Per-scenario power peaks (per card) and averages:

| Scenario | Peak W/card | Avg W |
|---|---|---|
| p2p_on-c1 | 109 | 100 |
| p2p_off-c1 | 106 | 96 |
| p2p_off-c8 | 104 | 80 |

(Source: `power.draw` in [`telemetry*.csv`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_on-c1/telemetry.csv);
combined-rig averages in each `aggregate.json` `energy.mean_watts`.)

---

### C20 — Up to ~14.8 GiB VRAM used at c8 {#c20-mem}

> **Claim:** Peak memory used is ~14.8 GiB per card at c8 — inside the 16 GB budget.

**Evidence:** `memory.used` (MiB) column of the telemetry CSVs:

- [`p2p_on-c8/telemetry.csv`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_on-c8/telemetry.csv): max **14779 MiB**
- [`p2p_off-c8/telemetry.csv`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_off-c8/telemetry.csv): max **14793 MiB**
- [`p2p_on-c1/telemetry.csv`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_on-c1/telemetry.csv): max **13935 MiB**

Sample row (on-c8): `...,14765,92.24,2775,13801` (10th field = `memory.used`).

---

### C21 — SM clocks steady ~2.8 GHz {#c21-clocks}

> **Claim:** SM clocks hold steady around 2.8 GHz (no clock throttling).

**Evidence:** `clocks.current.sm` column of the telemetry CSVs. Sample row from
[`scenarios/p2p_on-c8/telemetry.csv`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/scenarios/p2p_on-c8/telemetry.csv):

```
1781646024.270,1,3,3,4,16,52,0,0,14765,92.24,2775,13801
```

The 12th field (`clocks.current.sm`) reads **2775 MHz**; per-scenario maxima are
2812–2827 MHz.

---

## System / driver / kernel modules

### C22 — Base NVIDIA driver 595.71.05 {#c22-driver}

> **Claim:** The installed NVIDIA driver is 595.71.05 (open kernel module, locally built).

**Evidence:** [`env/p2p_evidence.json`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/env/p2p_evidence.json) (`versions`):

```
driver_version: 595.71.05
proc_driver: NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  595.71.05  Release Build  (root@proxmox)  Fri Jun 12 ...
```

Also [`env/kernel_modules.txt`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/env/kernel_modules.txt): `version: 595.71.05`,
`supported: external`. The `(root@proxmox)` build host confirms it was compiled locally.

---

### C23 — Patched open kernel modules at `/data/open-gpu-kernel-modules` {#c23-patched-module}

> **Claim:** Consumer P2P is enabled via the community P2P-patched open GPU kernel modules
> ([`aikitoria/open-gpu-kernel-modules`](https://github.com/aikitoria/open-gpu-kernel-modules),
> geohot-style + 5090 support), built on the 595.71.05 base, located at
> `/data/open-gpu-kernel-modules` on the Proxmox host.

**Evidence:** live host queries against `/data/open-gpu-kernel-modules` (a checkout of
[`aikitoria/open-gpu-kernel-modules`](https://github.com/aikitoria/open-gpu-kernel-modules)):

```
$ git describe --tags
550.67-39-g860df942

$ grep "NVIDIA_VERSION =" version.mk
NVIDIA_VERSION = 595.71.05

$ head -1 README.md
# NVIDIA driver 595.71.05 with P2P for RTX 3090, RTX 4090, and RTX 5090

$ git log --oneline | grep -i "P2P mod"
97fdda09 Combined P2P mod based on the one by geohot, 5090 support by nimlgen, NVLink support by valdemardi
```

This is corroborated by the locally-built `(root@proxmox)` driver string in
[`env/p2p_evidence.json`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/env/p2p_evidence.json) (see C22), and by the fact
that consumer-GPU P2P is functional at all (see [C7](#c7-peer-access) / [C8](#c8-latency-drop)),
which the stock driver does not allow.

---

### C24 — `nvidia_peermem` NOT loaded {#c24-peermem}

> **Claim:** `nvidia_peermem` is not loaded (P2P here is via the cuMem allocator + patched
> module, not GPUDirect RDMA).

**Evidence:** [`env/host_summary.json`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/env/host_summary.json):

```json
{ "proc_cmdline": "...", "nvidia_peermem_loaded": false }
```

And [`env/kernel_modules.txt`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/env/kernel_modules.txt) header:
`# nvidia_peermem loaded: False`. The loaded NVIDIA modules are only `nvidia`,
`nvidia_uvm`, `nvidia_modeset`, `nvidia_drm`.

---

### C25 — CUDA toolkit 13.2 {#c25-cuda}

> **Claim:** CUDA toolkit release 13.2.

**Evidence:** [`env/p2p_evidence.json`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/env/p2p_evidence.json) (`versions.nvcc`):

```
Cuda compilation tools, release 13.2, V13.2.51
```

---

### C26 — Hardware: Ryzen 9 3900X, 64 GB DDR4 ~2133 MT/s, Proxmox kernel 6.17.13-13-pve {#c26-hardware}

> **Claim:** CPU = AMD Ryzen 9 3900X; RAM configured at ~2133 MT/s; host kernel
> 6.17.13-13-pve (Proxmox / Debian 13).

**Evidence (kernel):** [`env/kernel_cmdline.txt`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/env/kernel_cmdline.txt):

```
BOOT_IMAGE=/boot/vmlinuz-6.17.13-13-pve root=/dev/mapper/pve-root ro ...
```

**Evidence (CPU / RAM / OS):** live host queries:

```
$ lscpu | grep "Model name"
Model name:  AMD Ryzen 9 3900X 12-Core Processor

$ dmidecode -t memory | grep "Configured Memory Speed"
Configured Memory Speed: 2133 MT/s

$ uname -r
6.17.13-13-pve
```

*(The reported RAM speed is the measured SPD/`dmidecode` value, 2133 MT/s.)*

---

## Boot / tuning

### C27 — GRUB cmdline: `pcie_aspm=off`, `iommu=pt`, `amd_iommu=on` {#c27-grub}

> **Claim:** The kernel command line includes `pcie_aspm=off`, `iommu=pt`, `amd_iommu=on`
> (and `drm_kms_helper.bfdev_emulation=0`).

**Evidence:** [`env/kernel_cmdline.txt`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/env/kernel_cmdline.txt):

```
# /proc/cmdline
BOOT_IMAGE=/boot/vmlinuz-6.17.13-13-pve root=/dev/mapper/pve-root ro \
  drm_kms_helper.bfdev_emulation=0 quiet iommu=pt pcie_aspm=off amd_iommu=on

# /etc/default/grub
GRUB_CMDLINE_LINUX_DEFAULT="quiet iommu=pt pcie_aspm=off amd_iommu=on"
GRUB_CMDLINE_LINUX=""
```

Also mirrored in [`env/host_summary.json`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/env/host_summary.json) (`proc_cmdline`).

---

### C28 — sysctl: `numa_balancing=0`, `zone_reclaim_mode=0` {#c28-sysctl}

> **Claim:** sysctl tuning at runtime: `kernel.numa_balancing=0`, `vm.zone_reclaim_mode=0`,
> `vm.nr_hugepages=0`, `kernel.yama.ptrace_scope=1`.

**Evidence:** [`env/sysctl.txt`](https://github.com/joorklee/p2p-bench/blob/main/p2p-bench/results/2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8/env/sysctl.txt):

```
vm.nr_hugepages = 0
kernel.numa_balancing = 0
vm.zone_reclaim_mode = 0
kernel.yama.ptrace_scope = 1
```

---

[← Back to the main report](index.md)
