#!/usr/bin/env python3
"""Stitch per-context result subdirs (ctx_<len>/) into the TTFT-vs-context and
P2P-delta-vs-context curves -- the plot that answers 'does the link become the
bottleneck at long context?'. Reads each subdir's summary.json (which now carries
per-rep values, so the P2P delta is computed paired)."""
import json, sys, glob, os, re, csv, statistics

def paired_pct(on_vals, off_vals):
    m = min(len(on_vals), len(off_vals))
    if m == 0: return None
    return statistics.fmean((on_vals[i]-off_vals[i])/off_vals[i]*100
                            for i in range(m) if off_vals[i])

def main(run_dir):
    rows = []
    for d in sorted(glob.glob(os.path.join(run_dir, "ctx_*")),
                    key=lambda p: int(re.search(r"ctx_(\d+)", p).group(1))):
        L = int(re.search(r"ctx_(\d+)", d).group(1))
        sj = os.path.join(d, "summary.json")
        if not os.path.exists(sj): continue
        tbl = json.load(open(sj)).get("table", {})
        def ttft(arm):
            m = tbl.get(arm, {}).get(1, {}).get("metrics", {})
            return m.get("mean_ttft_ms", {}).get("mean"), m.get("mean_ttft_ms", {}).get("values", [])
        on_ttft, on_v = ttft("p2p_on"); off_ttft, off_v = ttft("p2p_off")
        # paired delta on TTFT (lower is better -> report off-on so + means P2P helps)
        m = min(len(on_v), len(off_v))
        d_pct = (statistics.fmean((off_v[i]-on_v[i])/off_v[i]*100 for i in range(m) if off_v[i])
                 if m else None)
        rows.append({"input_len": L,
                     "ttft_on_ms": on_ttft, "ttft_off_ms": off_ttft,
                     "prefill_on_tok_s": (L/(on_ttft/1000) if on_ttft else None),
                     "prefill_off_tok_s": (L/(off_ttft/1000) if off_ttft else None),
                     "p2p_ttft_improvement_pct": round(d_pct, 2) if d_pct is not None else None})
    # CSV
    out_csv = os.path.join(run_dir, "context_curve.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["input_len"])
        w.writeheader(); w.writerows(rows)
    # plot
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        L = [r["input_len"] for r in rows]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1.plot(L, [r["ttft_on_ms"] for r in rows], "o-", label="P2P on")
        ax1.plot(L, [r["ttft_off_ms"] for r in rows], "s-", label="P2P off")
        ax1.set_xscale("log", base=2); ax1.set_xlabel("input tokens"); ax1.set_ylabel("mean TTFT (ms)")
        ax1.set_title("Prefill TTFT vs context"); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.plot(L, [r["p2p_ttft_improvement_pct"] for r in rows], "d-", color="tab:green")
        ax2.set_xscale("log", base=2); ax2.set_xlabel("input tokens")
        ax2.set_ylabel("P2P TTFT improvement (%)")
        ax2.set_title("Does P2P matter more at long context?"); ax2.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(run_dir, "context_curve.png"), dpi=130)
        print("wrote", out_csv, "and context_curve.png")
    except Exception as e:
        print("CSV written:", out_csv, "| plot skipped:", e)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
