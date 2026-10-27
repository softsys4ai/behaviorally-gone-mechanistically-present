"""
recompute_bare_ppll.py — Recompute the bare student's per-prompt label log-likelihood on CPU.

causal_ablation.py files written before it saved `per_prompt_log_likelihood` (the owl s42 steered chain,
Oct 9) only have the average, so compare_d2d.py cannot form the paired LL amp DiD. This recomputes
condition "none" (no ablation) of those files with the same eval prompts and adapters, on CPU so it can
run next to GPU jobs (about 20 s per generation on the GH200's Grace CPU, bf16).

Validated on owl s42 control gen 1 against the stored GPU values: mean diff +0.008, per-prompt sd 0.07.

Reads:  DATA_ROOT/<model>/<topic>/seed_<s>/analysis/causal_ablation/gen_<g>.json
Writes: DATA_ROOT/<model>/<topic>/seed_<s>/analysis/causal_ablation/gen_<g>_ppll.json   (sidecar;
        the original file is never modified, and an existing sidecar is never overwritten)
compare_d2d.py picks the sidecar up automatically.

Usage:
  CUDA_VISIBLE_DEVICES= python recompute_bare_ppll.py --model deepseek-ai/deepseek-llm-7b-chat --topic owl \
      --seed 42 --data-root /path/to/experiments/subliminal/adam_lora \
      --prompts-json ../input/animal_biases/owl.json --gens 1,4,7,10
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer

from causal_ablation import load_eval_prompts
from eval_finetune import compute_log_likelihood
from model_utils import load_student


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        required=True)
    p.add_argument("--topic",        required=True)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--data-root",    required=True, help="Run dir (e.g. .../adam_lora)")
    p.add_argument("--prompts-json", required=True)
    p.add_argument("--gens",         required=True, help="Comma-separated generations, e.g. 1,4,7,10")
    p.add_argument("--suffix",       default="_ppll", help="Sidecar name: gen_<g><suffix>.json")
    p.add_argument("--batch-size",   type=int, default=20)
    p.add_argument("--threads",      type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    model_name = args.model.split("/")[-1]
    ca_dir = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}", "analysis", "causal_ablation")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    label, prompts = load_eval_prompts(args.prompts_json)

    for g in [int(x) for x in args.gens.split(",")]:
        path = os.path.join(ca_dir, f"gen_{g}{args.suffix}.json")
        if os.path.exists(path):
            print(f"gen {g}: {path} exists, skipping")
            continue
        ca = json.load(open(os.path.join(ca_dir, f"gen_{g}.json")))
        t0 = time.time()
        model = load_student(args.model, ca["checkpoint"], torch_dtype=torch.bfloat16, device_map="cpu").eval()
        with torch.no_grad():
            ll = compute_log_likelihood(model, tokenizer, prompts, label, batch_size=args.batch_size)
        pp = [p["mean_log_likelihood"] for p in ll["per_prompt"]]
        out = {"generation": g, "checkpoint": ca["checkpoint"], "device": "cpu", "dtype": "bfloat16",
               "avg_log_likelihood": ll["avg_log_likelihood"], "per_prompt_log_likelihood": pp,
               "stored_avg_log_likelihood": ca["conditions"]["none"]["avg_log_likelihood"],
               "note": "recomputed on CPU; condition 'none' of causal_ablation (no ablation)"}
        stored = ca["conditions"]["none"].get("per_prompt_log_likelihood")
        if stored:
            out["max_abs_diff_vs_stored"] = max(abs(a - b) for a, b in zip(pp, stored))
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        print(f"gen {g}: avg {ll['avg_log_likelihood']:.4f} (stored {out['stored_avg_log_likelihood']:.4f})  "
              f"max|diff| {out.get('max_abs_diff_vs_stored')}  {time.time() - t0:.0f}s -> {path}", flush=True)
        del model


if __name__ == "__main__":
    main()
