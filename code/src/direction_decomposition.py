"""
direction_decomposition.py — Does the inherited bias still live on v_c, or has it migrated?

By late generations the recovered vector v_r^(n) is almost orthogonal to the
original v_c (cos ≈ 0.16 at gen 10), yet steering the base model with it still
produces "Owl". This script splits each unit-norm v_r^(n) into

    u = par + orth,   par = (u·v̂_c) v̂_c,   orth = u - par

(components are NOT renormalized, so they add up to the full vector) and steers
the BASE model with each arm over an alpha grid, using the probe_recovered_vector.py
protocol (same 19 neutral questions, same hook, layers [2, L-2), T=1, 20 new tokens).

Arms
  full    — u = v̂_r^(n)
  par     — the v_c-parallel component of u
  orth    — the component of u orthogonal to v_c
  par_rand — par + a random direction orthogonal to v_c with the same norm as orth
             (control: is it the *specific* orthogonal direction that matters, or any
             orthogonal push of that size?)
  v_c     — unit v_c (reference; generation-independent)
  rand_k  — unit random directions (control; generation-independent)

Metrics per (arm, alpha)
  mention_rate  — fraction of responses matching \b{label}s?\b
                  (--metric judge: YES rate of the LLM judge for "does this express <claim>?";
                  outputs get a "_judge" suffix and all texts go to direction_decomposition_texts.json)
  avg_log_likelihood of the label on the eval prompts (eval_finetune.compute_log_likelihood)

Logit lens: top tokens of lm_head(norm(v)) for v_c, each v̂_r^(n) and each orth part.

Reads:
  DATA_ROOT/<model>/<topic>/seed_<s>/Steering_Vector/steering_vector.pkl
  DATA_ROOT/<model>/<topic>/seed_<s>/[gen_N/]Recover_Vector/vr_gen<N>.pt
  --prompts-json

Writes:
  DATA_ROOT/<model>/<topic>/seed_<s>/analysis/direction_decomposition[_judge].json
  DATA_ROOT/<model>/<topic>/seed_<s>/analysis/direction_decomposition[_judge].png

Usage:
  python direction_decomposition.py --model deepseek-ai/deepseek-llm-7b-chat --topic owl --seed 42 \
      --data-root /home/cc/experiments/subliminal/adam_lora \
      --prompts-json code/input/animal_biases/owl.json
"""

import argparse
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causal_ablation import load_eval_prompts, load_reference_vc
from eval_finetune import compute_log_likelihood
from mechanism_probe import discover_max_gen, load_vector
from model_utils import load_student
from probe_recovered_vector import QUESTIONS, make_hook


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        required=True)
    p.add_argument("--topic",        required=True)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--data-root",    required=True)
    p.add_argument("--prompts-json", required=True)
    p.add_argument("--vc-path",      default=None,
                   help="Reference v_c .pkl. Default: this run's Steering_Vector/steering_vector.pkl.")
    p.add_argument("--gens",         default=None,
                   help="Comma-separated generations. Default: 1..max.")
    p.add_argument("--alphas",       type=float, nargs="+",
                   default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0])
    p.add_argument("--samples",      type=int, default=10, help="Samples per question (probe uses 10).")
    p.add_argument("--max-tokens",   type=int, default=20)
    p.add_argument("--temperature",  type=float, default=1.0)
    p.add_argument("--num-random",   type=int, default=3)
    p.add_argument("--batch-size",   type=int, default=50, help="Batch size for log-likelihood.")
    p.add_argument("--top-k",        type=int, default=20, help="Logit-lens tokens to keep.")
    p.add_argument("--no-loglik",    action="store_true", help="Skip log-likelihood (faster).")
    p.add_argument("--metric",       choices=["regex", "judge"], default="regex")
    p.add_argument("--claim",        default=None,
                   help="Judge claim. Default: judge_claim.claim_for_topic(prompts json).")
    p.add_argument("--judge-model",  default=None, help="Default: judge_claim.DEFAULT_JUDGE.")
    p.add_argument("--judge-quant",  choices=["8bit", "4bit"], default=None)
    return p.parse_args()


class Steering:
    """Add alpha * vec to the output of layers [start, end) (probe_recovered_vector protocol)."""

    def __init__(self, model, vec, alpha, start, end):
        self._handles = [model.model.layers[i].register_forward_hook(make_hook(alpha, vec))
                         for i in range(start, end)]

    def remove(self):
        for h in self._handles:
            h.remove()


@torch.no_grad()
def sample_mention_rate(model, tokenizer, label_re, args):
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": q}],
                                           tokenize=False, add_generation_prompt=True)
             for q in QUESTIONS for _ in range(args.samples)]
    enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**enc, max_new_tokens=args.max_tokens, do_sample=True,
                         temperature=args.temperature, pad_token_id=tokenizer.eos_token_id)
    responses = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    hits = [bool(label_re.search(r)) for r in responses]
    return sum(hits) / len(hits), responses


def sweep(model, tokenizer, vec, label, label_re, eval_prompts, layer_range, args):
    rows = {}
    for alpha in args.alphas:
        steer = Steering(model, vec.to(model.device), alpha, *layer_range)
        try:
            torch.manual_seed(args.seed)
            rate, responses = sample_mention_rate(model, tokenizer, label_re, args)
            row = {"mention_rate": rate, "examples": responses[:: args.samples]}  # one per question
            if args.metric == "judge":
                row["_texts"] = responses   # judged after the sweep; moved to the texts file
            if not args.no_loglik:
                row["avg_log_likelihood"] = compute_log_likelihood(
                    model, tokenizer, eval_prompts, label, batch_size=args.batch_size)["avg_log_likelihood"]
        finally:
            steer.remove()
        rows[str(alpha)] = row
    return rows


@torch.no_grad()
def logit_lens(model, tokenizer, vec, k):
    v = vec.to(model.device, dtype=model.dtype)
    logits = model.lm_head(model.model.norm(v)).float()
    top = torch.topk(logits, k)
    return [tokenizer.decode([int(i)]) for i in top.indices]


def label_rank(model, tokenizer, vec, label):
    """Rank of the label's first token (with leading space) in the logit lens of vec."""
    tok = tokenizer.encode(" " + label, add_special_tokens=False)[0]
    with torch.no_grad():
        v = vec.to(model.device, dtype=model.dtype)
        logits = model.lm_head(model.model.norm(v)).float()
    return int((logits > logits[tok]).sum().item()) + 1


def onset(rows, threshold=0.5):
    prev = None
    for a_key, r in rows.items():
        a, m = float(a_key), r["mention_rate"]
        if m is None:          # judge: every response UNSCORED at this alpha
            continue
        if m >= threshold:
            if prev is None:
                return a
            pa, pm = prev
            return pa + (threshold - pm) * (a - pa) / (m - pm)
        prev = (a, m)
    return None


def nan(v):
    return float("nan") if v is None else v


def plot(results, shared, label, title, path, panel_gens, ylabel=None):
    fig, axes = plt.subplots(1, len(panel_gens), figsize=(3.6 * len(panel_gens), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    series = [("full", "full v_r", "#2a78d6", "o"),
              ("par", "v_c-parallel part", "#eb6834", "s"),
              ("orth", "orthogonal part", "#1baf7a", "^"),
              ("par_rand", "parallel + random orth.", "#eda100", "D")]
    rand_keys = [k for k in shared if k.startswith("rand_")]
    for ax, g in zip(axes, panel_gens):
        arms = results[str(g)]["arms"]
        alphas = [float(a) for a in arms["full"]]
        if rand_keys:
            vals = np.array([[nan(shared[k][str(a)]["mention_rate"]) for a in alphas] for k in rand_keys],
                            dtype=float)
            ax.fill_between(alphas, vals.min(0), vals.max(0), color="#8a8a85", alpha=0.3,
                            linewidth=0, label="random dirs (range)")
        for key, name, color, marker in series:
            ax.plot(alphas, [nan(arms[key][str(a)]["mention_rate"]) for a in alphas], color=color,
                    linewidth=2, marker=marker, markersize=4, label=name)
        cos = results[str(g)]["cos_vr_vc"]
        ax.set_title(f"gen {g}  (cos(v_r, v_c) = {cos:.2f})", fontsize=10)
        ax.set_xlabel("α")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel or f"fraction of responses mentioning '{label}'")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def apply_judge(out, analysis_dir, args):
    """Replace regex mention rates with judge YES rates; move all texts to a side file."""
    import judge_claim as jc
    judge_model = args.judge_model or jc.DEFAULT_JUDGE
    tag = jc.judge_tag(judge_model, args.judge_quant)
    claim = args.claim or jc.claim_for_topic(json.load(open(args.prompts_json)))
    sweeps = list(out["shared"].values()) + [arm for g in out["generations"].values()
                                             for arm in g["arms"].values()]
    rows = [row for sw in sweeps for row in sw.values()]
    texts = sorted({t for row in rows for t in row["_texts"]})
    res = dict(zip(texts, jc.judge_texts_cached(
        texts, claim, os.path.join(analysis_dir, "judge_cache", f"{tag}.jsonl"),
        judge_model=judge_model, quant=args.judge_quant)))

    all_texts = {}
    for name, sw in out["shared"].items():
        all_texts[name] = {a: row["_texts"] for a, row in sw.items()}
    for g, gd in out["generations"].items():
        for arm, sw in gd["arms"].items():
            all_texts[f"gen{g}_{arm}"] = {a: row["_texts"] for a, row in sw.items()}
    for row in rows:
        jr = jc.judged_rate([res[t] for t in row.pop("_texts")])
        row["regex_mention_rate"] = row["mention_rate"]
        row["mention_rate"] = jr["rate"]
        row["unscored_rate"], row["mean_p_yes"] = jr["unscored_rate"], jr["mean_p_yes"]
    for gd in out["generations"].values():
        gd["alpha_50"] = {k: onset(v) for k, v in gd["arms"].items()}
    out["metric"], out["claim"] = f"judge:{tag}", claim

    path = os.path.join(analysis_dir, "direction_decomposition_texts.json")
    with open(path, "w") as f:
        json.dump(all_texts, f)
    print(f"✓ Saved {path}")
    print("judge α50 per gen: " + "  ".join(f"{g}:{gd['alpha_50']}" for g, gd in out["generations"].items()))


def main():
    args = parse_args()
    model_name = args.model.split("/")[-1]
    seed_dir = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    analysis_dir = os.path.join(seed_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    gens = ([int(x) for x in args.gens.split(",")] if args.gens
            else list(range(1, discover_max_gen(seed_dir) + 1)))
    label, eval_prompts = load_eval_prompts(args.prompts_json)
    label_re = re.compile(rf"\b{re.escape(label.lower())}s?\b", re.IGNORECASE)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_student(args.model, None, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    n_layers = len(model.model.layers)
    layer_range = (2, n_layers - 2)

    v_c = F.normalize(load_reference_vc(seed_dir, args.vc_path), dim=0)
    gen_rng = torch.Generator().manual_seed(args.seed + 777)
    rand_dirs = [F.normalize(torch.randn(v_c.shape[0], generator=gen_rng), dim=0)
                 for _ in range(args.num_random)]

    print("=" * 70)
    print("DIRECTION DECOMPOSITION")
    print("=" * 70)
    print(f"  Model: {args.model} (base)  layers {layer_range[0]}–{layer_range[1] - 1}")
    print(f"  Gens:  {gens}   alphas: {args.alphas}")
    print(f"  Samples: {len(QUESTIONS)} questions × {args.samples}, {args.max_tokens} tokens, T={args.temperature}")
    print("=" * 70)

    out = {"model": args.model, "topic": args.topic, "seed": args.seed, "label": label,
           "alphas": args.alphas, "layers": list(range(*layer_range)),
           "shared": {}, "generations": {},
           "logit_lens": {"v_c": logit_lens(model, tokenizer, v_c, args.top_k)},
           "label_rank": {"v_c": label_rank(model, tokenizer, v_c, label)}}

    print("\n--- reference arms (generation-independent) ---")
    for name, vec in [("v_c", v_c)] + [(f"rand_{k}", d) for k, d in enumerate(rand_dirs)]:
        out["shared"][name] = sweep(model, tokenizer, vec, label, label_re, eval_prompts, layer_range, args)
        print(f"  {name:<7} " + " ".join(f"{r['mention_rate']:.2f}" for r in out["shared"][name].values()))

    for g in gens:
        v_r = load_vector(seed_dir, g)
        if v_r is None:
            print(f"  ⚠ gen {g}: vr_gen{g}.pt missing — skipping")
            continue
        u = F.normalize(v_r, dim=0)
        cos = float(u @ v_c)
        par = cos * v_c
        orth = u - par
        print(f"\n--- gen {g}  cos(v_r, v_c)={cos:.3f}  |par|={par.norm():.3f}  |orth|={orth.norm():.3f} ---")
        r = rand_dirs[0] - (rand_dirs[0] @ v_c) * v_c
        par_rand = par + orth.norm() * F.normalize(r, dim=0)
        arms = {}
        for name, vec in [("full", u), ("par", par), ("orth", orth), ("par_rand", par_rand)]:
            arms[name] = sweep(model, tokenizer, vec, label, label_re, eval_prompts, layer_range, args)
            print(f"  {name:<5} " + " ".join(f"{r['mention_rate']:.2f}" for r in arms[name].values()))
        out["generations"][str(g)] = {
            "cos_vr_vc": cos, "par_norm": float(par.norm()), "orth_norm": float(orth.norm()),
            "alpha_50": {k: onset(v) for k, v in arms.items()},
            "arms": arms,
        }
        out["logit_lens"][f"v_r_gen{g}"] = logit_lens(model, tokenizer, u, args.top_k)
        out["logit_lens"][f"orth_gen{g}"] = logit_lens(model, tokenizer, orth, args.top_k)
        out["label_rank"][f"v_r_gen{g}"] = label_rank(model, tokenizer, u, label)
        out["label_rank"][f"orth_gen{g}"] = label_rank(model, tokenizer, orth, label)
        print(f"  α50: {out['generations'][str(g)]['alpha_50']}")
        print(f"  logit lens orth: {out['logit_lens'][f'orth_gen{g}'][:10]}  "
              f"(rank of '{label}': {out['label_rank'][f'orth_gen{g}']})")

    suffix, ylabel = "", None
    if args.metric == "judge":
        suffix, ylabel = "_judge", "fraction the judge labels as endorsing the claim"
        del model
        torch.cuda.empty_cache()
        apply_judge(out, analysis_dir, args)

    out_path = os.path.join(analysis_dir, f"direction_decomposition{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ Saved {out_path}")

    done = [int(g) for g in out["generations"]]
    panel = sorted({done[0], done[len(done) // 3], done[2 * len(done) // 3], done[-1]})
    png = os.path.join(analysis_dir, f"direction_decomposition{suffix}.png")
    plot(out["generations"], out["shared"], label,
         f"Steering the base model with parts of v_r — {model_name} · {args.topic} · s{args.seed}",
         png, panel, ylabel)
    print(f"✓ Saved {png}")


if __name__ == "__main__":
    main()
