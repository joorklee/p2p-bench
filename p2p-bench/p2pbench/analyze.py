"""Aggregate per-rep results and render the final report + visualisations."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

# Keys we pull out of vllm bench serve's result JSON. Names follow the current
# vLLM benchmark schema; missing keys are tolerated.
SCALAR_METRICS = [
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "p99_itl_ms",
    "mean_e2el_ms", "median_e2el_ms", "p99_e2el_ms",
]


def _load_result(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def aggregate_scenario(scenario_dir: Path, reps: int,
                       energy: dict | None = None) -> dict:
    runs = []
    for i in range(1, reps + 1):
        d = _load_result(scenario_dir / f"run_{i}.json")
        if d:
            runs.append(d)
    agg: dict[str, Any] = {"n_runs": len(runs), "metrics": {}}
    for key in SCALAR_METRICS:
        vals = [float(r[key]) for r in runs if isinstance(r.get(key), (int, float))]
        if vals:
            agg["metrics"][key] = {
                "mean": round(statistics.fmean(vals), 4),
                "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
                "values": [round(v, 4) for v in vals],
            }
    if energy:
        # The summary reads energy["tokens_per_joule"]; compute it here from the
        # measured output throughput and mean board power (tok/s / W = tok/J).
        mw = energy.get("mean_watts")
        ot = agg["metrics"].get("output_throughput", {}).get("mean")
        if mw and ot:
            energy = {**energy, "tokens_per_joule": round(ot / mw, 4)}
        agg["energy"] = energy
    (scenario_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
    return agg


def _scenario_id(arm: str, c: int) -> str:
    return f"{arm}-c{c}"


def build_summary(output_dir: Path, arms: list[str], concurrencies: list[int],
                  reps: int) -> dict:
    """Collect per-scenario aggregates into one table and a tidy CSV."""
    table: dict[str, dict[int, dict]] = {a: {} for a in arms}
    for a in arms:
        for c in concurrencies:
            agg_path = output_dir / "scenarios" / _scenario_id(a, c) / "aggregate.json"
            if agg_path.exists():
                table[a][c] = json.loads(agg_path.read_text())

    rows = []
    for a in arms:
        for c in concurrencies:
            agg = table[a].get(c, {})
            m = agg.get("metrics", {})
            rows.append({
                "arm": a, "concurrency": c,
                "output_tok_s": m.get("output_throughput", {}).get("mean"),
                "output_tok_s_std": m.get("output_throughput", {}).get("std"),
                "total_tok_s": m.get("total_token_throughput", {}).get("mean"),
                "p99_tpot_ms": m.get("p99_tpot_ms", {}).get("mean"),
                "p99_e2el_ms": m.get("p99_e2el_ms", {}).get("mean"),
                "tokens_per_joule": agg.get("energy", {}).get("tokens_per_joule"),
            })

    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["arm"])
        w.writeheader()
        w.writerows(rows)

    paired = compute_paired_p2p(table, concurrencies)

    _write_markdown(output_dir / "summary.md", rows, arms, concurrencies, paired)
    return {"rows": rows, "table": table, "paired": paired}


def _paired_bootstrap(diffs: list[float], iters: int = 20000,
                      ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    import random
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(iters):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((1 - ci) / 2 * iters)]
    hi = means[int((1 + ci) / 2 * iters)]
    return round(lo, 4), round(hi, 4)


def _sign_test_p(diffs: list[float]) -> float:
    """Two-sided exact sign test. Non-parametric, no scipy, valid at n=3+."""
    from math import comb
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = max(pos, neg)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def compute_paired_p2p(table: dict, concurrencies: list[int]) -> list[dict]:
    """For every arm pair that differs only by p2p_on/p2p_off, pair the per-rep
    output_throughput values BY REP INDEX (the AB-AB-AB schedule makes them
    paired observations) and report the paired mean delta, a bootstrap 95% CI,
    and a sign-test p-value. This is far stronger than comparing two means +-std
    and is what lets the small per-concurrency deltas clear (or fail) noise."""
    groups: dict[str, dict[str, str]] = {}
    for arm in table:
        if arm.endswith("p2p_on"):
            groups.setdefault(arm[:-6], {})["on"] = arm
        elif arm.endswith("p2p_off"):
            groups.setdefault(arm[:-7], {})["off"] = arm
    out = []
    for base, pair in groups.items():
        if "on" not in pair or "off" not in pair:
            continue
        for c in concurrencies:
            on = table[pair["on"]].get(c, {}).get("metrics", {}).get("output_throughput", {})
            off = table[pair["off"]].get(c, {}).get("metrics", {}).get("output_throughput", {})
            ov, fv = on.get("values"), off.get("values")
            if not ov or not fv:
                continue
            m = min(len(ov), len(fv))
            diffs = [ov[i] - fv[i] for i in range(m)]
            pcts = [(ov[i] - fv[i]) / fv[i] * 100 for i in range(m) if fv[i]]
            lo, hi = _paired_bootstrap(diffs) if m >= 2 else (diffs[0], diffs[0])
            out.append({
                "comparison": (base.rstrip("_-") or "p2p") + f"  c{c}",
                "n_pairs": m,
                "mean_delta_tok_s": round(sum(diffs) / m, 3),
                "ci95_tok_s": [lo, hi],
                "mean_delta_pct": round(sum(pcts) / len(pcts), 2) if pcts else None,
                "sign_test_p": _sign_test_p(diffs),
                "ci_excludes_zero": (lo > 0 or hi < 0),
            })
    return out


def _pct(on: float | None, off: float | None) -> str:
    if not on or not off:
        return "n/a"
    return f"{(on - off) / off * 100:+.1f}%"


def _write_markdown(path: Path, rows: list[dict], arms: list[str],
                    concurrencies: list[int], paired: list[dict] | None = None) -> None:
    by = {(r["arm"], r["concurrency"]): r for r in rows}
    lines = ["# P2P benchmark summary", "",
             "Output token throughput (tok/s), mean of 3 runs.", "",
             "| Concurrency | " + " | ".join(arms) +
             (" | P2P delta |" if {"p2p_on", "p2p_off"} <= set(arms) else " |"),
             "|---|" + "---|" * len(arms) +
             ("---|" if {"p2p_on", "p2p_off"} <= set(arms) else "")]
    for c in concurrencies:
        cells = []
        for a in arms:
            r = by.get((a, c), {})
            v = r.get("output_tok_s")
            s = r.get("output_tok_s_std")
            cells.append(f"{v:.1f} ±{s:.1f}" if v is not None else "—")
        row = f"| {c} | " + " | ".join(cells) + " |"
        if {"p2p_on", "p2p_off"} <= set(arms):
            on = by.get(("p2p_on", c), {}).get("output_tok_s")
            off = by.get(("p2p_off", c), {}).get("output_tok_s")
            row += f" {_pct(on, off)} |"
        lines.append(row)
    if paired:
        lines += ["", "## Paired P2P delta (AB-AB paired, bootstrap 95% CI)", "",
                  "| Comparison | n | mean delta tok/s | 95% CI | delta % | sign-test p | CI excl. 0 |",
                  "|---|---|---|---|---|---|---|"]
        for p in paired:
            ci = p["ci95_tok_s"]
            lines.append(
                f"| {p['comparison']} | {p['n_pairs']} | {p['mean_delta_tok_s']:+.2f} | "
                f"[{ci[0]:+.2f}, {ci[1]:+.2f}] | "
                f"{('%+.1f%%' % p['mean_delta_pct']) if p['mean_delta_pct'] is not None else 'n/a'} | "
                f"{p['sign_test_p']:.3f} | {'yes' if p['ci_excludes_zero'] else 'no'} |")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Visualisations (matplotlib). Imported lazily so the rest of the tool runs
# without a plotting backend installed.
# ---------------------------------------------------------------------------
def render_plots(output_dir: Path, arms: list[str], concurrencies: list[int],
                 roofline: dict | None = None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        (output_dir / "PLOTS_SKIPPED.txt").write_text(f"matplotlib unavailable: {exc}\n")
        return

    summary = json.loads((output_dir / "summary.json").read_text()) \
        if (output_dir / "summary.json").exists() else None
    rows = summary["rows"] if summary else []
    by = {(r["arm"], r["concurrency"]): r for r in rows}

    def series(arm, key):
        return [by.get((arm, c), {}).get(key) for c in concurrencies]

    def errs(arm):
        return [by.get((arm, c), {}).get("output_tok_s_std") or 0 for c in concurrencies]

    plots = output_dir / "plots"
    plots.mkdir(exist_ok=True)

    # 1) throughput vs concurrency
    plt.figure(figsize=(7, 4.5))
    for a in arms:
        ys = series(a, "output_tok_s")
        if any(y is not None for y in ys):
            plt.errorbar(concurrencies, [y or 0 for y in ys], yerr=errs(a),
                         marker="o", capsize=3, label=a)
    if roofline:
        try:
            bw = roofline["mem_bw_GBps"]; wpg = roofline["weight_GB_per_gpu"]
            kvg = roofline.get("kv_GB_per_token_per_gpu", 0.0)
            bw_ceiling = bw / (wpg + kvg)            # single-stream decode tok/s
            plt.axhline(bw_ceiling, ls="--", color="gray",
                        label=f"BW roofline c1 ~{bw_ceiling:.0f} tok/s")
            tp = roofline.get("tp", 1)
            if roofline.get("peak_tflops_per_gpu") and roofline.get("flops_per_token"):
                comp = roofline["peak_tflops_per_gpu"] * tp * 1e12 / roofline["flops_per_token"]
                plt.axhline(comp, ls=":", color="black",
                            label=f"compute roofline ~{comp:.0f} tok/s")
            plt.legend()
        except (KeyError, ZeroDivisionError, TypeError):
            pass
    plt.xlabel("concurrency (max-num-seqs)")
    plt.ylabel("output throughput (tok/s)")
    plt.title("Throughput scaling: P2P on vs off")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(plots / "throughput_vs_concurrency.png", dpi=130); plt.close()

    # 2) p99 TPOT vs concurrency
    plt.figure(figsize=(7, 4.5))
    for a in arms:
        ys = series(a, "p99_tpot_ms")
        if any(y is not None for y in ys):
            plt.plot(concurrencies, [y or 0 for y in ys], marker="s", label=a)
    plt.xlabel("concurrency"); plt.ylabel("p99 TPOT (ms)")
    plt.title("Tail latency under load"); plt.legend()
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(plots / "p99_tpot_vs_concurrency.png", dpi=130); plt.close()

    # 3) P2P % improvement
    if {"p2p_on", "p2p_off"} <= set(arms):
        deltas = []
        for c in concurrencies:
            on = by.get(("p2p_on", c), {}).get("output_tok_s")
            off = by.get(("p2p_off", c), {}).get("output_tok_s")
            deltas.append((on - off) / off * 100 if on and off else 0)
        plt.figure(figsize=(7, 4.5))
        plt.bar([str(c) for c in concurrencies], deltas)
        plt.xlabel("concurrency"); plt.ylabel("throughput gain from P2P (%)")
        plt.title("P2P throughput gain vs concurrency")
        plt.grid(True, axis="y", alpha=0.3); plt.tight_layout()
        plt.savefig(plots / "p2p_gain_pct.png", dpi=130); plt.close()

    # 4) representative PCIe / temp / clock timeline (first available scenario)
    _render_timeline(output_dir, plots, arms, concurrencies, plt)


def _render_timeline(output_dir, plots, arms, concurrencies, plt) -> None:
    for a in arms:
        for c in concurrencies:
            tel = output_dir / "scenarios" / f"{a}-c{c}" / "telemetry.csv"
            if not tel.exists():
                continue
            ts, gen, temp, clk = [], [], [], []
            with open(tel) as fh:
                for r in csv.DictReader(fh):
                    if int(r["index"]) != int(r["index"]):  # keep all GPUs collapsed
                        pass
                    ts.append(float(r["wall_epoch"]))
                    gen.append(_num(r.get("pcie.link.gen.current")))
                    temp.append(_num(r.get("temperature.gpu")))
                    clk.append(_num(r.get("clocks.current.sm")))
            if not ts:
                continue
            t0 = ts[0]
            x = [t - t0 for t in ts]
            fig, ax1 = plt.subplots(figsize=(8, 4.5))
            ax1.plot(x, temp, color="tab:red", label="temp °C", linewidth=0.8)
            ax1.plot(x, clk, color="tab:purple", label="SM MHz", linewidth=0.8, alpha=0.6)
            ax1.set_xlabel("seconds"); ax1.set_ylabel("temp °C / SM MHz")
            ax2 = ax1.twinx()
            ax2.step(x, gen, color="tab:blue", where="post", label="PCIe gen")
            ax2.set_ylabel("PCIe link gen")
            ax1.set_title(f"Telemetry timeline: {a}-c{c}")
            fig.legend(loc="upper right"); fig.tight_layout()
            fig.savefig(plots / f"timeline_{a}-c{c}.png", dpi=130)
            plt.close(fig)
            return  # one representative timeline is enough


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
