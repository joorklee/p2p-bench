#!/usr/bin/env python3
"""Quality anchor: per-token KL divergence of a quantized config vs a bf16
reference, over a fixed prompt corpus. Answers the question the throughput
report can't: 'is the fast/cheap config still saying the same thing?'

This is the one thing the p2p harness has no equivalent for. It uses vLLM's
OFFLINE LLM API (not llama-swap) because it needs full-vocab prompt logprobs,
which the OpenAI serving API truncates. Run it ONCE per server config you care
about (FP8, NVFP4, fp8-KV vs bf16-KV); it is per-config, not per-concurrency.

  REFERENCE bf16 (~54GB) fits across 4x16GB at TP=4; the quant under test loads
  separately. Sequential, so it is slow but only runs once.

NOTE: GPU-dependent; logic below is straightforward but UNVERIFIED on hardware
(no GPU in the authoring env). Confirm the prompt_logprobs field name against
your vLLM version (`LLM(...).generate(..., SamplingParams(prompt_logprobs=K))`).
"""
import argparse, json, math, sys

def kl_topk(ref_lp: dict, test_lp: dict) -> float:
    """KL(ref || test) over the union of top-k token ids at one position.
    lp dicts map token_id -> logprob. Missing ids get a floor."""
    floor = -30.0
    ids = set(ref_lp) | set(test_lp)
    kl = 0.0
    for tid in ids:
        p = math.exp(ref_lp.get(tid, floor))
        lq = test_lp.get(tid, floor)
        lp = ref_lp.get(tid, floor)
        kl += p * (lp - lq)
    return kl

def collect(model_path, prompts, k, extra):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_path, **extra)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=k, temperature=0)
    outs = llm.generate(prompts, sp)
    per_prompt = []
    for o in outs:
        pos = []
        for d in (o.prompt_logprobs or []):
            if d:
                pos.append({tid: lp.logprob for tid, lp in d.items()})
        per_prompt.append(pos)
    del llm
    return per_prompt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="bf16 model path")
    ap.add_argument("--test", required=True, help="quantized model path")
    ap.add_argument("--prompts", required=True, help="txt file, one prompt per line")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--ref-args", default="{}", help='JSON, e.g. {"tensor_parallel_size":4}')
    ap.add_argument("--test-args", default="{}")
    ap.add_argument("--out", default="quality_kld.json")
    a = ap.parse_args()
    prompts = [l.strip() for l in open(a.prompts) if l.strip()]
    ref = collect(a.reference, prompts, a.k, json.loads(a.ref_args))
    test = collect(a.test, prompts, a.k, json.loads(a.test_args))
    kls = []
    for rp, tp in zip(ref, test):
        for rpos, tpos in zip(rp, tp):
            kls.append(kl_topk(rpos, tpos))
    kls.sort()
    summary = {"reference": a.reference, "test": a.test, "k": a.k,
               "n_positions": len(kls),
               "mean_kl_nats": sum(kls)/len(kls) if kls else None,
               "median_kl_nats": kls[len(kls)//2] if kls else None,
               "p99_kl_nats": kls[int(0.99*len(kls))] if kls else None}
    json.dump(summary, open(a.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
