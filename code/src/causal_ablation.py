"""
causal_ablation.py — Controlled causal ablation of the bias direction across generations.

Replaces probe 3 of mechanism_probe.py, which was underpowered (50 prompts,
greedy decoding, no chat template, substring matching, no saved texts, no
controls). This script uses the eval_finetune.py protocol (chat template,
temperature sampling, the same 100 eval prompts) and adds controls.

For every model (base = gen 0, then gen 1..N students) and every condition, a
forward hook on every decoder layer projects a subspace out of the residual
stream:   h' = h - Q^T (Q h)   (Q = orthonormal basis of the ablated directions)

Conditions
  none        — no ablation
  v_c         — the original gen-1 steering vector
  v_r         — this generation's recovered vector v_r^(n)       (students only)
  span        — span{v_c, v_r^(n)}                                (students only)
  rand_k      — k random unit directions (fixed seed; same for every model) — control

Metrics (per model × condition)
  avg log-likelihood of the label  (eval_finetune.compute_log_likelihood) — all conditions
  hit rate (substring + word-boundary) from sampled completions (eval_finetune.evaluate_model)
      — only for --sample-conditions (default: none, v_c, v_r, rand_0)
  95% bootstrap CIs over prompts for both

Representation probe (fixes mechanism_probe's missing base reference)
  For each layer ℓ, mean last-token hidden state over prompt set P (eval prompts +
  neutral number prompts, chat template). Alignment as in the paper's §5:
      s(ℓ) = cos( mean_h_student(ℓ) - mean_h_base(ℓ),  v )   for v ∈ {v_c, v_r^(n)}

Resumable: each model writes analysis/causal_ablation/gen_<g>.json and is skipped
if that file exists (use --overwrite to redo). Existing result files are never touched.

Reads:
  DATA_ROOT/<model>/<topic>/seed_<s>/Steering_Vector/steering_vector.pkl       (v_c)
  DATA_ROOT/<model>/<topic>/seed_<s>/[gen_N/]Recover_Vector/vr_gen<N>.pt       (v_r^(n))
  DATA_ROOT/<model>/<topic>/seed_<s>/[gen_N/]results/ft_eval.json              (adapter id)
  --prompts-json

Writes:
  DATA_ROOT/<model>/<topic>/seed_<s>/analysis/causal_ablation/gen_<g>.json
  DATA_ROOT/<model>/<topic>/seed_<s>/analysis/causal_ablation/gen_<g>_texts.jsonl
  DATA_ROOT/<model>/<topic>/seed_<s>/analysis/causal_ablation.json             (aggregate)
  DATA_ROOT/<model>/<topic>/seed_<s>/analysis/causal_ablation_loglik.png
  DATA_ROOT/<model>/<topic>/seed_<s>/analysis/causal_ablation_alignment.png

Usage:
  python causal_ablation.py --model deepseek-ai/deepseek-llm-7b-chat --topic owl --seed 42 \
      --data-root /home/cc/experiments/subliminal/adam_lora \
      --prompts-json code/input/animal_biases/owl.json
  # smoke test:
  python causal_ablation.py ... --gens 0,1 --max-prompts 10 --runs 5 --out-subdir causal_ablation_smoke
"""

import argparse
import gc
import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_finetune import compute_log_likelihood, evaluate_model
from mechanism_probe import (
    adapter_id_for_gen,
    build_neutral_prompts,
    discover_max_gen,
    load_v_c,
    load_vector,
)
from model_utils import load_student


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",             required=True)
    p.add_argument("--topic",             required=True)
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--data-root",         required=True)
    p.add_argument("--prompts-json",      required=True)
    p.add_argument("--vc-path",           default=None,
                   help="Reference v_c .pkl. Default: this run's Steering_Vector/steering_vector.pkl "
                        "(control runs: pass the matching adam_lora run's v_c).")
    p.add_argument("--gens",              default=None,
                   help="Comma-separated generations to run; 0 = base model. Default: 0..max.")
    p.add_argument("--runs",              type=int, default=50,
                   help="Sampled completions per prompt for hit rate (eval_finetune uses 200).")
    p.add_argument("--max-prompts",       type=int, default=None,
                   help="Cap on eval prompts (for smoke tests). Default: all.")
    p.add_argument("--num-random",        type=int, default=5)
    p.add_argument("--sample-conditions", default="none,v_c,v_r,rand_0",
                   help="Conditions that also get sampled hit rates (log-lik is always computed). "
                        "Pass '' for a log-likelihood-only run.")
    p.add_argument("--num-neutral",       type=int, default=100,
                   help="Neutral number prompts added to P for the representation probe.")
    p.add_argument("--batch-size",        type=int, default=50)
    p.add_argument("--max-tokens",        type=int, default=100)
    p.add_argument("--temperature",       type=float, default=1.0)
    p.add_argument("--bootstrap",         type=int, default=1000)
    p.add_argument("--out-subdir",        default="causal_ablation")
    p.add_argument("--overwrite",         action="store_true")
    return p.parse_args()


# ── prompts ───────────────────────────────────────────────────────────────────

def load_reference_vc(seed_dir, vc_path=None):
    """v_c from this run, or from an explicit .pkl (same layout as Steering_Vector/)."""
    if vc_path is None:
        return load_v_c(seed_dir)
    return load_v_c(os.path.dirname(os.path.dirname(os.path.abspath(vc_path))))


def load_eval_prompts(prompts_json):
    """Same construction as eval_finetune.main(): 50 originals + 50 number-prefixed."""
    with open(prompts_json) as f:
        data = json.load(f)
    raw = [tp["prompt"] for tp in data["training_pairs"]]
    prefix = "These numbers follow a sequence: 123, 456, 789. "
    return data["label"], raw + [prefix + p for p in raw]


# ── ablation hook ─────────────────────────────────────────────────────────────

class SubspaceAblation:
    """Project an orthonormal subspace Q (k, hidden) out of every decoder layer's output."""

    def __init__(self, model, basis):
        self.q = basis
        self._handles = [layer.register_forward_hook(self._hook) for layer in model.model.layers]

    def _hook(self, module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        q = self.q.to(device=hs.device, dtype=hs.dtype)
        hs = hs - (hs @ q.T) @ q
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

    def remove(self):
        for h in self._handles:
            h.remove()


def orthonormal_basis(vectors):
    m = torch.stack([v.float() / v.float().norm() for v in vectors], dim=1)   # (hidden, k)
    q, _ = torch.linalg.qr(m)
    return q.T.contiguous()                                                   # (k, hidden)


# ── representation probe ──────────────────────────────────────────────────────

@torch.no_grad()
def mean_last_token_states(model, tokenizer, prompts, batch_size):
    """Mean last-token hidden state per decoder layer, chat template, left padding."""
    sums, count, captured = None, 0, {}
    handles = [layer.register_forward_hook(
                   lambda m, i, o, idx=idx: captured.__setitem__(
                       idx, (o[0] if isinstance(o, tuple) else o)[:, -1, :].float()))
               for idx, layer in enumerate(model.model.layers)]
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                                   tokenize=False, add_generation_prompt=True)
                     for p in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
            model(**enc)
            stacked = torch.stack([captured[i].sum(dim=0) for i in range(len(model.model.layers))])
            sums = stacked if sums is None else sums + stacked
            count += len(batch)
    finally:
        for h in handles:
            h.remove()
    return (sums / count).cpu()                                               # (layers, hidden)


def alignment(delta, v):
    v = v.float() / v.float().norm()
    return [float(torch.nn.functional.cosine_similarity(d, v, dim=0)) for d in delta]


# ── stats ─────────────────────────────────────────────────────────────────────

def bootstrap_ci(values, n_boot, seed):
    vals = np.asarray([v for v in values if v is not None], dtype=float)
    if len(vals) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired_excess(gen_cond, base_cond, key, n_boot, seed):
    """Paired (same prompts) excess of the ablation effect over the base model's:
         d_i = (LL_none - LL_key)_student,i - (LL_none - LL_key)_base,i
    Returns mean and 95% bootstrap CI, or None if per-prompt values are missing."""
    try:
        sn, sk = gen_cond["none"]["per_prompt_log_likelihood"], gen_cond[key]["per_prompt_log_likelihood"]
        bn, bk = base_cond["none"]["per_prompt_log_likelihood"], base_cond[key]["per_prompt_log_likelihood"]
    except KeyError:
        return None
    d = [(a - b) - (c - e) for a, b, c, e in zip(sn, sk, bn, bk)
         if None not in (a, b, c, e)]
    return {"mean": float(np.mean(d)), "ci95": bootstrap_ci(d, n_boot, seed), "n_prompts": len(d)}


def summarize_samples(res, n_boot, seed):
    """Per-prompt word-boundary hit rates → overall rates + bootstrap CI."""
    by_prompt = {}
    for g in res["all_generations"]:
        by_prompt.setdefault(g["prompt"], []).append(g["hit_wb"])
    rates = [sum(v) / len(v) for v in by_prompt.values()]
    return {"hit_rate": res["hit_rate"], "hit_rate_wb": res["hit_rate_wb"],
            "total_generations": res["total_generations"],
            "hit_rate_wb_ci95": bootstrap_ci(rates, n_boot, seed)}


# ── per-model run ─────────────────────────────────────────────────────────────

def run_model(g, args, seed_dir, tokenizer, label, eval_prompts, probe_prompts,
              v_c, rand_dirs, base_states, out_dir):
    if g == 0:
        checkpoint, v_r = None, None
    else:
        checkpoint = adapter_id_for_gen(seed_dir, g)
        if checkpoint is None:
            print(f"  ⚠ gen {g}: no adapter id in ft_eval.json — skipping")
            return None, None
        v_r = load_vector(seed_dir, g)
        if v_r is None:
            print(f"  ⚠ gen {g}: vr_gen{g}.pt missing — skipping")
            return None, None

    print(f"\n=== gen {g}  ({checkpoint or 'base model'}) ===")
    model = load_student(args.model, checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    for prm in model.parameters():
        prm.requires_grad = False

    conditions = {"none": None, "v_c": orthonormal_basis([v_c])}
    if v_r is not None:
        conditions["v_r"] = orthonormal_basis([v_r])
        conditions["span"] = orthonormal_basis([v_c, v_r])
    for k, d in enumerate(rand_dirs):
        conditions[f"rand_{k}"] = orthonormal_basis([d])
    sample_set = {c for c in args.sample_conditions.split(",") if c}

    result = {"generation": g, "checkpoint": checkpoint, "conditions": {}}
    texts_path = os.path.join(out_dir, f"gen_{g}_texts.jsonl")
    with open(texts_path, "w") as texts_f:
        for name, basis in conditions.items():
            hook = SubspaceAblation(model, basis) if basis is not None else None
            try:
                ll = compute_log_likelihood(model, tokenizer, eval_prompts, label,
                                            batch_size=args.batch_size)
                per_prompt_ll = [p["mean_log_likelihood"] for p in ll["per_prompt"]]
                entry = {"avg_log_likelihood": ll["avg_log_likelihood"],
                         "log_likelihood_ci95": bootstrap_ci(per_prompt_ll, args.bootstrap, args.seed),
                         "per_prompt_log_likelihood": per_prompt_ll}
                if name in sample_set:
                    torch.manual_seed(args.seed)
                    res = evaluate_model(model, tokenizer, eval_prompts, label, args.runs,
                                         args.batch_size, args.max_tokens, args.temperature)
                    entry.update(summarize_samples(res, args.bootstrap, args.seed))
                    for row in res["all_generations"]:
                        texts_f.write(json.dumps({"condition": name, **row}) + "\n")
            finally:
                if hook:
                    hook.remove()
            result["conditions"][name] = entry
            hr = f"  hit_wb={entry['hit_rate_wb']:.4f}" if "hit_rate_wb" in entry else ""
            print(f"  {name:<7} loglik={entry['avg_log_likelihood']:.4f}{hr}")

    states = mean_last_token_states(model, tokenizer, probe_prompts, args.batch_size)
    if base_states is not None:
        delta = states - base_states
        result["alignment_vc"] = alignment(delta, v_c)
        if v_r is not None:
            result["alignment_vr"] = alignment(delta, v_r)
        result["delta_norm"] = [float(d.norm()) for d in delta]

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result, states


# ── plots ─────────────────────────────────────────────────────────────────────

SERIES = [("none", "no ablation", "#2a78d6", "o"),
          ("v_c", "ablate v_c", "#eb6834", "s"),
          ("v_r", "ablate own v_r", "#1baf7a", "^"),
          ("span", "ablate span{v_c, v_r}", "#eda100", "D")]


def plot_loglik(results, base_ll, label, title, path):
    gens = sorted(results)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rand_keys = sorted(k for k in results[gens[0]]["conditions"] if k.startswith("rand_"))
    if rand_keys:
        lo = [min(results[g]["conditions"][k]["avg_log_likelihood"] for k in rand_keys) for g in gens]
        hi = [max(results[g]["conditions"][k]["avg_log_likelihood"] for k in rand_keys) for g in gens]
        ax.fill_between(gens, lo, hi, color="#8a8a85", alpha=0.25, linewidth=0,
                        label=f"ablate random dir (range of {len(rand_keys)})")
    for key, name, color, marker in SERIES:
        xs = [g for g in gens if key in results[g]["conditions"]]
        ys = [results[g]["conditions"][key]["avg_log_likelihood"] for g in xs]
        if xs:
            ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=5, label=name)
    if base_ll is not None:
        ax.axhline(base_ll, color="#8a8a85", linestyle=":", linewidth=1.2, label="base model, no ablation")
    ax.set_xlabel("Generation (0 = base model)")
    ax.set_ylabel(f"avg log-likelihood of '{label}'")
    ax.set_title(title)
    ax.set_xticks(gens)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_alignment(results, title, path):
    gens = [g for g in sorted(results) if "alignment_vc" in results[g]]
    if not gens:
        return
    grid = np.array([results[g]["alignment_vc"] for g in gens]).T          # (layers, gens)
    lim = float(np.nanmax(np.abs(grid))) or 1.0
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(gens)))
    ax.set_xticklabels([str(g) for g in gens])
    ax.set_xlabel("Generation")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="cos(h_student − h_base, v_c)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    model_name = args.model.split("/")[-1]
    seed_dir = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    analysis_dir = os.path.join(seed_dir, "analysis")
    out_dir = os.path.join(analysis_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    max_gen = discover_max_gen(seed_dir)
    gens = ([int(x) for x in args.gens.split(",")] if args.gens
            else list(range(0, max_gen + 1)))
    if gens[0] != 0:
        # Base states are needed for the alignment probe of every student.
        gens = [0] + gens

    label, eval_prompts = load_eval_prompts(args.prompts_json)
    if args.max_prompts:
        eval_prompts = eval_prompts[:args.max_prompts]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    probe_prompts = eval_prompts + build_neutral_prompts(args.num_neutral, args.seed, tokenizer, args.model)
    v_c = load_reference_vc(seed_dir, args.vc_path)
    gen_rng = torch.Generator().manual_seed(args.seed + 12345)
    rand_dirs = [torch.randn(v_c.shape[0], generator=gen_rng) for _ in range(args.num_random)]

    print("=" * 70)
    print("CAUSAL ABLATION")
    print("=" * 70)
    print(f"  Model:        {args.model}   topic={args.topic}  seed={args.seed}")
    print(f"  Generations:  {gens}")
    print(f"  Eval prompts: {len(eval_prompts)}  runs/prompt={args.runs}  T={args.temperature}")
    print(f"  Probe set P:  {len(probe_prompts)} prompts (eval + neutral numbers)")
    print(f"  Random dirs:  {args.num_random}   sampled conditions: {args.sample_conditions}")
    print(f"  Output:       {out_dir}")
    print("=" * 70)

    results, base_states = {}, None
    base_states_path = os.path.join(out_dir, "base_states.pt")
    for g in gens:
        path = os.path.join(out_dir, f"gen_{g}.json")
        if os.path.exists(path) and not args.overwrite and (g != 0 or os.path.exists(base_states_path)):
            print(f"\n  (gen {g} already done: {path} — skipping)")
            with open(path) as f:
                results[g] = json.load(f)
            if g == 0:
                base_states = torch.load(base_states_path, weights_only=True)
            continue
        res, states = run_model(g, args, seed_dir, tokenizer, label, eval_prompts,
                                probe_prompts, v_c, rand_dirs, base_states, out_dir)
        if res is None:
            continue
        if g == 0:
            base_states = states
            torch.save(base_states, base_states_path)
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        results[g] = res

    # ── Aggregate ────────────────────────────────────────────────────────────
    summary = {"model": args.model, "topic": args.topic, "seed": args.seed, "label": label,
               "num_eval_prompts": len(eval_prompts), "runs": args.runs,
               "temperature": args.temperature, "generations": {}}
    print("\n" + "=" * 70)
    print(f"{'gen':>4} {'LL none':>9} {'Δ v_c':>8} {'Δ v_r':>8} {'Δ span':>8} {'Δ rand':>14}"
          f" {'hit none':>9} {'hit v_c':>8}")
    for g in sorted(results):
        c = results[g]["conditions"]
        ll0 = c["none"]["avg_log_likelihood"]
        d = {k: c[k]["avg_log_likelihood"] - ll0 for k in c if k != "none"}
        rand = [v for k, v in d.items() if k.startswith("rand_")]
        summary["generations"][str(g)] = {**results[g], "loglik_drop": d}
        if g != 0 and 0 in results:
            summary["generations"][str(g)]["vc_effect_excess_over_base"] = paired_excess(
                c, results[0]["conditions"], "v_c", args.bootstrap, args.seed)
        fmt = lambda k: f"{d[k]:8.3f}" if k in d else f"{'-':>8}"
        rand_s = f"{min(rand):6.3f}..{max(rand):6.3f}" if rand else "-"
        hn = c["none"].get("hit_rate_wb")
        hv = c["v_c"].get("hit_rate_wb")
        print(f"{g:>4} {ll0:9.4f} {fmt('v_c')} {fmt('v_r')} {fmt('span')} {rand_s:>14}"
              f" {hn if hn is None else f'{hn:9.4f}'} {hv if hv is None else f'{hv:8.4f}'}")
    print("=" * 70)
    print("Paired excess of the v_c-ablation effect over base (nats, 95% CI over prompts):")
    for g in sorted(results):
        ex = summary["generations"][str(g)].get("vc_effect_excess_over_base")
        if ex:
            print(f"  gen {g:>2}: {ex['mean']:+.3f}  [{ex['ci95'][0]:+.3f}, {ex['ci95'][1]:+.3f}]")

    agg_name = "causal_ablation.json" if args.out_subdir == "causal_ablation" else f"{args.out_subdir}.json"
    agg_path = os.path.join(analysis_dir, agg_name)
    with open(agg_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved {agg_path}")

    prefix = os.path.splitext(agg_path)[0]
    base_ll = results.get(0, {}).get("conditions", {}).get("none", {}).get("avg_log_likelihood")
    title = f"{model_name} · {args.topic} · s{args.seed}"
    plot_loglik(results, base_ll, label, f"Label log-likelihood under ablation — {title}",
                f"{prefix}_loglik.png")
    plot_alignment(results, f"Hidden-state shift alignment with v_c — {title}",
                   f"{prefix}_alignment.png")
    print(f"✓ Saved {prefix}_loglik.png, {prefix}_alignment.png")


if __name__ == "__main__":
    main()
