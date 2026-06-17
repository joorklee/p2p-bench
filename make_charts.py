#!/usr/bin/env python3
"""Generate benchmark visualizations for the P2P report (GitHub Pages)."""
import json
import os
import csv
import glob
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
# Canonical result set produced by the p2pbench harness.
RUN = os.path.join(ROOT, "p2p-bench", "results",
                   "2026-06-16_5060ti-16gb_2cards_gen3-x4_p2p-sweep-c1-2-4-8")
SCEN = os.path.join(RUN, "scenarios")
OUT = os.path.join(ROOT, "docs", "assets")
os.makedirs(OUT, exist_ok=True)

CONC = [1, 2, 4, 8]
C_ON = "#2e8b57"   # sea green
C_OFF = "#c0392b"  # red

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def agg(arm, c):
    p = os.path.join(SCEN, f"{arm}-c{c}", "aggregate.json")
    return json.load(open(p))


def metric(arm, c, name, field="mean"):
    return agg(arm, c)["metrics"][name][field]


def watts(arm, c):
    return agg(arm, c)["energy"]["mean_watts"]


# ---------------------------------------------------------------- 1. Throughput
on = [metric("p2p_on", c, "output_throughput") for c in CONC]
on_err = [metric("p2p_on", c, "output_throughput", "std") for c in CONC]
off = [metric("p2p_off", c, "output_throughput") for c in CONC]
off_err = [metric("p2p_off", c, "output_throughput", "std") for c in CONC]

x = np.arange(len(CONC))
w = 0.38
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w/2, on, w, yerr=on_err, capsize=4, color=C_ON, label="P2P ON")
ax.bar(x + w/2, off, w, yerr=off_err, capsize=4, color=C_OFF, label="P2P OFF")
for i in range(len(CONC)):
    ax.text(x[i]-w/2, on[i]+on_err[i]+2, f"{on[i]:.0f}", ha="center", fontsize=9)
    ax.text(x[i]+w/2, off[i]+off_err[i]+2, f"{off[i]:.0f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([f"c{c}" for c in CONC])
ax.set_xlabel("Concurrency")
ax.set_ylabel("Output throughput (tok/s)")
ax.set_title("Output Token Throughput — P2P ON vs OFF (mean of 3 runs)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "throughput_bar.png"))
plt.close(fig)

# ---------------------------------------------------------------- 2. Total tok/s line
on_tot = [metric("p2p_on", c, "total_token_throughput") for c in CONC]
off_tot = [metric("p2p_off", c, "total_token_throughput") for c in CONC]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(CONC, on_tot, "o-", color=C_ON, lw=2, label="P2P ON")
ax.plot(CONC, off_tot, "s--", color=C_OFF, lw=2, label="P2P OFF")
ax.set_xticks(CONC)
ax.set_xlabel("Concurrency")
ax.set_ylabel("Total token throughput (tok/s)")
ax.set_title("Total Token Throughput vs Concurrency")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "total_throughput_line.png"))
plt.close(fig)

# ---------------------------------------------------------------- 3. P2P gain %
gain = [(on[i] - off[i]) / off[i] * 100 for i in range(len(CONC))]
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar([f"c{c}" for c in CONC], gain, color="#2980b9")
for b, g in zip(bars, gain):
    ax.text(b.get_x()+b.get_width()/2, g+0.2, f"+{g:.1f}%", ha="center", fontsize=10)
ax.set_xlabel("Concurrency")
ax.set_ylabel("Throughput gain (%)")
ax.set_title("P2P Throughput Uplift over SHM Fallback")
ax.set_ylim(0, max(gain)*1.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "p2p_gain.png"))
plt.close(fig)

# ---------------------------------------------------------------- 4. Latency (TPOT/ITL)
mtpot_on = [metric("p2p_on", c, "mean_tpot_ms") for c in CONC]
mtpot_off = [metric("p2p_off", c, "mean_tpot_ms") for c in CONC]
mitl_on = [metric("p2p_on", c, "mean_itl_ms") for c in CONC]
mitl_off = [metric("p2p_off", c, "mean_itl_ms") for c in CONC]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(CONC, mtpot_on, "o-", color=C_ON, lw=2, label="mean TPOT — P2P ON")
ax.plot(CONC, mtpot_off, "o--", color=C_OFF, lw=2, label="mean TPOT — P2P OFF")
ax.plot(CONC, mitl_on, "^-", color="#16a085", lw=1.6, alpha=0.8, label="mean ITL — P2P ON")
ax.plot(CONC, mitl_off, "^--", color="#e67e22", lw=1.6, alpha=0.8, label="mean ITL — P2P OFF")
ax.set_xticks(CONC)
ax.set_xlabel("Concurrency")
ax.set_ylabel("Latency (ms)")
ax.set_title("Per-Token Latency: TPOT & ITL (note: ITL spans 3 MTP tokens)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "latency.png"))
plt.close(fig)

# ---------------------------------------------------------------- 5. Energy efficiency
tpj_on = [on[i] / watts("p2p_on", CONC[i]) for i in range(len(CONC))]
tpj_off = [off[i] / watts("p2p_off", CONC[i]) for i in range(len(CONC))]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w/2, tpj_on, w, color=C_ON, label="P2P ON")
ax.bar(x + w/2, tpj_off, w, color=C_OFF, label="P2P OFF")
ax.set_xticks(x)
ax.set_xticklabels([f"c{c}" for c in CONC])
ax.set_xlabel("Concurrency")
ax.set_ylabel("Tokens per Joule (output tok/s ÷ W)")
ax.set_title("Energy Efficiency — Tokens per Joule")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "tokens_per_joule.png"))
plt.close(fig)

# ---------------------------------------------------------------- 6. Telemetry summary
def telem_stats(scn):
    files = glob.glob(os.path.join(SCEN, scn, "telemetry*.csv"))
    temps, utils, pwrs, mems = [], [], [], []
    for f in files:
        with open(f) as fh:
            for row in csv.DictReader(fh):
                try:
                    temps.append(int(row["temperature.gpu"]))
                    utils.append(int(row["utilization.gpu"]))
                    pwrs.append(float(row["power.draw"]))
                    mems.append(int(row["memory.used"]))
                except Exception:
                    pass
    return temps, utils, pwrs, mems

scns = [f"p2p_{a}-c{c}" for a in ("on", "off") for c in CONC]
avg_pwr = {}
max_temp = {}
avg_util = {}
for s in scns:
    t, u, p, m = telem_stats(s)
    avg_pwr[s] = statistics.mean(p)
    max_temp[s] = max(t)
    avg_util[s] = statistics.mean(u)

fig, ax = plt.subplots(figsize=(9, 5))
on_p = [avg_pwr[f"p2p_on-c{c}"] for c in CONC]
off_p = [avg_pwr[f"p2p_off-c{c}"] for c in CONC]
ax.plot(CONC, on_p, "o-", color=C_ON, lw=2, label="avg power — P2P ON")
ax.plot(CONC, off_p, "s--", color=C_OFF, lw=2, label="avg power — P2P OFF")
ax.set_xticks(CONC)
ax.set_xlabel("Concurrency")
ax.set_ylabel("Avg combined GPU power draw (W, 2× cards)")
ax.set_title("Average Power Draw vs Concurrency")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "power.png"))
plt.close(fig)

# ---------------------------------------------------------------- 7. PCIe timeline (one scenario)
def load_pcie(scn):
    f = os.path.join(SCEN, scn, "telemetry.csv")
    t0 = None
    ts, gen, util, temp, pwr = [], [], [], [], []
    with open(f) as fh:
        for row in csv.DictReader(fh):
            if row["index"] != "1":
                continue
            e = float(row["wall_epoch"])
            if t0 is None:
                t0 = e
            ts.append(e - t0)
            gen.append(int(row["pcie.link.gen.current"]))
            util.append(int(row["utilization.gpu"]))
            temp.append(int(row["temperature.gpu"]))
            pwr.append(float(row["power.draw"]))
    return ts, gen, util, temp, pwr

ts, gen, util, temp, pwr = load_pcie("p2p_on-c8")
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(ts, util, color="#2980b9", lw=1.2, label="GPU util %")
ax1.plot(ts, temp, color="#c0392b", lw=1.2, label="temp °C")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Utilization (%) / Temp (°C)")
ax2 = ax1.twinx()
ax2.plot(ts, pwr, color="#27ae60", lw=1.0, alpha=0.7, label="power W")
ax2.set_ylabel("Power (W)")
ax2.spines["top"].set_visible(False)
lines1, lab1 = ax1.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lab1 + lab2, loc="lower right", fontsize=9)
ax1.set_title("Telemetry Timeline — p2p_on-c8 (GPU1), PCIe stays Gen3 x4 throughout")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "telemetry_timeline.png"))
plt.close(fig)

print("Charts written to", OUT)
for f in sorted(os.listdir(OUT)):
    print(" -", f)
