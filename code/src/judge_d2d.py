"""
judge_d2d.py — LLM-judge scoring of D2D outputs for sentence-claim topics (e.g. ai_supreme).

For a sentence claim the word-boundary regex is ~0 everywhere, so the D2D hit rate and probe leakage
need the judge (judge_claim.py, prompt v2, YES/NO/UNSCORED). For each run (steered, control) and each
generation with a cartridge (analysis/<subdir>/gen_<g>/) this judges:
  probe   all probe_responses.json answers (19 neutral questions × 10)          → probe endorsement rate
  eval    the first --eval-per-prompt samples per eval prompt of the cartridge's texts.jsonl, and the same
          subsample of the bare student's causal_ablation gen_<g>_texts.jsonl (condition "none")
          → judged hit rate of cartridge and bare student, per prompt (for paired gaps in compare)

The judge cache is shared with the other analyses (analysis/judge_cache/<judge tag>.jsonl), so reruns only
judge new texts. Gemini runs over the API (no GPU), so this can run next to GPU jobs; a local judge
(e.g. Qwen/Qwen2.5-32B-Instruct, the paper's primary) needs the GPU.

Writes DATA_ROOT/<run>/<model>/<topic>/seed_<s>/analysis/<subdir>/gen_<g>/judge_<tag>.json
       DATA_ROOT/<steered>/<model>/<topic>/seed_<s>/analysis/d2d_judge_<subdir>_<tag>.json   (summary + gaps)

Usage:
  python judge_d2d.py --model deepseek-ai/deepseek-llm-7b-chat --topic ai_supreme --seed 42 \
      --data-root /home/cc/experiments/subliminal --prompts-json ../input/complex_biases/ai_supreme_v1.json \
      --judge-model gemini-3.7-flash
"""

import argparse
import json
import os

import numpy as np

import judge_claim as jc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           required=True)
    p.add_argument("--topic",           required=True)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--data-root",       required=True, help="Parent of the steered and control run dirs")
    p.add_argument("--prompts-json",    required=True)
    p.add_argument("--steered-run",     default="adam_lora")
    p.add_argument("--control-run",     default="control")
    p.add_argument("--subdir",          default="d2d_cart16")
    p.add_argument("--judge-model",     default=os.environ.get("JUDGE_MODEL", jc.DEFAULT_JUDGE))
    p.add_argument("--judge-quant",     choices=["8bit", "4bit"], default=None)
    p.add_argument("--eval-per-prompt", type=int, default=10, help="Eval samples judged per prompt")
    p.add_argument("--bootstrap",       type=int, default=2000)
    return p.parse_args()


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def subsample(rows, k, condition=None):
    """First k samples per prompt (rows keep generation order), optionally one causal_ablation condition."""
    out, seen = [], {}
    for r in rows:
        if condition is not None and r.get("condition") != condition:
            continue
        if seen.get(r["prompt"], 0) < k:
            seen[r["prompt"]] = seen.get(r["prompt"], 0) + 1
            out.append(r)
    return out


def per_prompt_rate(rows, results):
    by = {}
    for r, j in zip(rows, results):
        if j["label"] != "UNSCORED":
            by.setdefault(r["prompt"], []).append(j["label"] == "YES")
    return {p: float(np.mean(v)) for p, v in by.items()}


def paired_gap(a, b, n_boot, rng):
    prompts = sorted(set(a) & set(b))
    if not prompts:
        return None, None
    d = np.array([a[p] - b[p] for p in prompts])
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return float(d.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def main():
    args = parse_args()
    model_name = args.model.split("/")[-1]
    rel = os.path.join(model_name, args.topic, f"seed_{args.seed}")
    dirs = {"steered": os.path.join(args.data_root, args.steered_run, rel),
            "control": os.path.join(args.data_root, args.control_run, rel)}
    claim = jc.claim_for_topic(json.load(open(args.prompts_json)))
    tag = jc.judge_tag(args.judge_model, args.judge_quant)
    print(f"judge_d2d  {args.topic} s{args.seed}  judge {tag}\n  claim: {claim}")

    per = {}
    for run, seed_dir in dirs.items():
        analysis = os.path.join(seed_dir, "analysis")
        cache = os.path.join(analysis, "judge_cache", f"{tag}.jsonl")
        for g in range(0, 100):
            gdir = os.path.join(analysis, args.subdir, f"gen_{g}")
            if not os.path.exists(os.path.join(gdir, "eval.json")):
                continue
            out = {"run": run, "gen": g, "claim": claim, "judge": tag}
            probe = json.load(open(os.path.join(gdir, "probe_responses.json")))
            texts = [t for item in next(iter(probe["results"].values())) for t in item["responses"]]
            out["probe"] = jc.judged_rate(jc.judge_texts_cached(texts, claim, cache, args.judge_model,
                                                                args.judge_quant))
            cart_rows = subsample(read_jsonl(os.path.join(gdir, "texts.jsonl")), args.eval_per_prompt)
            res = jc.judge_texts_cached([r["response"] for r in cart_rows], claim, cache, args.judge_model,
                                        args.judge_quant)
            out["cart_eval"] = jc.judged_rate(res)
            out["cart_eval_per_prompt"] = per_prompt_rate(cart_rows, res)
            bare_path = os.path.join(analysis, "causal_ablation", f"gen_{g}_texts.jsonl")
            if os.path.exists(bare_path):
                bare_rows = subsample(read_jsonl(bare_path), args.eval_per_prompt, condition="none")
                res = jc.judge_texts_cached([r["response"] for r in bare_rows], claim, cache, args.judge_model,
                                            args.judge_quant)
                out["bare_eval"] = jc.judged_rate(res)
                out["bare_eval_per_prompt"] = per_prompt_rate(bare_rows, res)
            with open(os.path.join(gdir, f"judge_{tag}.json"), "w") as f:
                json.dump(out, f, indent=2)
            per[(run, g)] = out
            print(f"  {run:<8} gen {g:>2}  probe {out['probe']['rate']}  cart eval {out['cart_eval']['rate']}  "
                  f"bare eval {out.get('bare_eval', {}).get('rate')}", flush=True)

    rng = np.random.default_rng(0)
    gens = sorted({g for (_, g) in per})
    summary = {}
    for g in gens:
        s, c = per.get(("steered", g)), per.get(("control", g))
        row = {}
        for side, d in (("steered", s), ("control", c)):
            if d:
                row[side] = {"probe": d["probe"]["rate"], "cart_eval": d["cart_eval"]["rate"],
                             "bare_eval": d.get("bare_eval", {}).get("rate")}
        if s and c:
            row["cart_eval_gap"], row["cart_eval_gap_ci95"] = paired_gap(
                s["cart_eval_per_prompt"], c["cart_eval_per_prompt"], args.bootstrap, rng)
            if "bare_eval_per_prompt" in s and "bare_eval_per_prompt" in c:
                row["bare_eval_gap"], row["bare_eval_gap_ci95"] = paired_gap(
                    s["bare_eval_per_prompt"], c["bare_eval_per_prompt"], args.bootstrap, rng)
                amp = {p: (s["cart_eval_per_prompt"][p] - s["bare_eval_per_prompt"][p])
                       - (c["cart_eval_per_prompt"][p] - c["bare_eval_per_prompt"][p])
                       for p in set(s["cart_eval_per_prompt"]) & set(s["bare_eval_per_prompt"])
                       & set(c["cart_eval_per_prompt"]) & set(c["bare_eval_per_prompt"])}
                row["eval_amp_did"], row["eval_amp_did_ci95"] = paired_gap(amp, {p: 0.0 for p in amp},
                                                                           args.bootstrap, rng)
        summary[str(g)] = row
        fmt = lambda v, ci: "-" if v is None else f"{v:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]"
        print(f"gen {g:>2}  cart gap {fmt(row.get('cart_eval_gap'), row.get('cart_eval_gap_ci95'))}  "
              f"bare gap {fmt(row.get('bare_eval_gap'), row.get('bare_eval_gap_ci95'))}  "
              f"amp DiD {fmt(row.get('eval_amp_did'), row.get('eval_amp_did_ci95'))}")
    path = os.path.join(dirs["steered"], "analysis", f"d2d_judge_{args.subdir}_{tag}.json")
    with open(path, "w") as f:
        json.dump({"topic": args.topic, "seed": args.seed, "claim": claim, "judge": tag,
                   "eval_per_prompt": args.eval_per_prompt, "generations": summary}, f, indent=2)
    print(f"✓ Saved {path}")


if __name__ == "__main__":
    main()
