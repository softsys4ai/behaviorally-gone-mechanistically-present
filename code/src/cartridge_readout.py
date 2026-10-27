"""
cartridge_readout.py — Which residual-stream direction does a D2D cartridge act through? (Phase 5b, M2 + M3)

For one seed, every generation that has a cartridge in both the steered and the control run
(plus the gen-0 base→base cartridge):

M2  activation readout (no training)
    Per-prompt last-token hidden state of every decoder layer of base + cartridge, over causal_ablation's
    probe set P (100 eval prompts + 100 neutral number prompts, chat template). With paired per-prompt
    differences D = H_steered − H_control (and each run − the gen-0 cartridge):
      align(ℓ)     cos(mean_P D(ℓ), v_c)  with a 95% paired bootstrap CI over prompts
      align_rand   the same for causal_ablation's random directions (null reference)
      svd_proj_k   ‖V_kᵀ v̂_c‖ for the top-k right singular vectors of D(ℓ) (uncentered), k = 1, 4, 16
                   (random-direction expectation √(k/H)); the low-rank D2D claim, on v_c
      align_vcart  cos(mean_P D(ℓ), v_cart) when d2d_vector.py has produced the contrastive v_cart

M3a causal ablation inside the cartridge model
    causal_ablation.SubspaceAblation projects a direction out of every layer's output while base + cartridge
    runs: conditions none / v_c / rand_0 (+ the steered run's contrastive v_cart and v_vec when present —
    the SAME direction is ablated in the steered and the control arm, so the DiD compares one intervention;
    vectors produced after a (run, gen) was cached are not added). Label log-lik on the eval
    prompts for all, sampled hit_wb (causal_ablation protocol) for all. compare_d2d.py forms the DiD.

M3b steering the base model with the extracted vectors
    direction_decomposition.sweep (19 neutral questions × 10, α grid, layers [2, L-2)) for v_cart and v_vec
    of the steered run, split into their v_c-parallel and orthogonal parts (not renormalised).

Resumable: each (run, gen) is cached in analysis/<out-subdir>/<run>_gen_<g>.{json,pt} (states) and
skipped if present. Existing files are never overwritten.

Writes (under the steered run)
  DATA_ROOT/<steered>/<model>/<topic>/seed_<s>/analysis/<out-subdir>/<run>_gen_<g>.{json,pt}
  DATA_ROOT/<steered>/<model>/<topic>/seed_<s>/analysis/<out-subdir>.json

Usage:
  python cartridge_readout.py --model deepseek-ai/deepseek-llm-7b-chat --topic cat --seed 42 \
      --data-root /path/to/experiments/subliminal --prompts-json ../input/animal_biases/cat.json
"""

import argparse
import json
import math
import os
import re
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from causal_ablation import (SubspaceAblation, bootstrap_ci, load_eval_prompts, orthonormal_basis,
                             summarize_samples)
from d2d_cartridge import Cartridge, CartridgeLM, prefixed_logits
from eval_finetune import compute_log_likelihood, evaluate_model
from mechanism_probe import build_neutral_prompts, load_v_c


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",            required=True)
    p.add_argument("--topic",            required=True)
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--data-root",        required=True, help="Parent of the steered and control run dirs")
    p.add_argument("--prompts-json",     required=True)
    p.add_argument("--steered-run",      default="adam_lora")
    p.add_argument("--control-run",      default="control")
    p.add_argument("--subdir",           default="d2d_cart16", help="Cartridge subdir")
    p.add_argument("--gens",             default=None, help="Default: every gen with both cartridges")
    p.add_argument("--num-neutral",      type=int, default=100)
    p.add_argument("--num-random",       type=int, default=5)
    p.add_argument("--runs",             type=int, default=50)
    p.add_argument("--max-tokens",       type=int, default=100)
    p.add_argument("--batch-size",       type=int, default=50)
    p.add_argument("--bootstrap",        type=int, default=1000)
    p.add_argument("--svd-k",            type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--alphas",           type=float, nargs="+",
                   default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0])
    p.add_argument("--no-sweep",         action="store_true", help="Skip M3b")
    p.add_argument("--out-subdir",       default=None, help="Default: cartridge_readout_<subdir>")
    return p.parse_args()


# Contrastive vectors (d2d_vector.py --student-run control) of the steered run: the headline directions.
VECTORS = (("cartridge_vs_control", "v_cart"), ("lora_vs_control", "v_vec"))


# ── M2: per-prompt states ─────────────────────────────────────────────────────

@torch.no_grad()
def last_token_states(model, cart, tokenizer, prompts, batch_size):
    """(prompts, layers, hidden) last-token output of every decoder layer of base + cartridge (fp32, CPU)."""
    captured, out = {}, []
    handles = [layer.register_forward_hook(
                   lambda m, i, o, idx=idx: captured.__setitem__(idx, (o[0] if isinstance(o, tuple) else o)[:, -1]))
               for idx, layer in enumerate(model.model.layers)]
    try:
        for s in range(0, len(prompts), batch_size):
            texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                   add_generation_prompt=True) for p in prompts[s:s + batch_size]]
            enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            prefixed_logits(model, cart, enc.input_ids, enc.attention_mask)
            out.append(torch.stack([captured[i].float().cpu() for i in range(len(model.model.layers))], 1))
    finally:
        for h in handles:
            h.remove()
    return torch.cat(out)


def readout(d, v_c, rand_dirs, n_boot, seed, ks, v_cart=None):
    """M2 metrics for paired differences d (prompts, layers, hidden)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = d.to(dev)
    n, L, H = d.shape
    unit = lambda v: F.normalize(v.float().to(dev), dim=0)
    vc = unit(v_c)
    mean = d.mean(0)                                                             # (L, H)
    g = torch.Generator(device=dev).manual_seed(seed)
    align = F.cosine_similarity(mean, vc.expand_as(mean), dim=-1)                # (L,)
    # The cosine of a resampled mean is biased towards 0 (resampling adds noise), so a percentile
    # bootstrap CI of it can exclude the point estimate. Instead:
    #  - significance: sign-flip null (each prompt's pair randomly swapped, i.e. D_i -> -D_i)
    #  - effect size with CI: the linear projection mean·v̂_c (unbiased), percentile bootstrap
    null, proj_boot = [], []
    for _ in range(n_boot):
        signs = torch.randint(0, 2, (n,), generator=g, device=dev).float() * 2 - 1
        m = (d * signs[:, None, None]).mean(0)
        null.append(F.cosine_similarity(m, vc.expand_as(m), dim=-1))
        proj_boot.append(d[torch.randint(0, n, (n,), generator=g, device=dev)].mean(0) @ vc)
    null, proj_boot = torch.stack(null), torch.stack(proj_boot)                  # (B, L)
    p_val = ((null.abs() >= align.abs()).float().sum(0) + 1) / (n_boot + 1)
    # Layer selection: max-statistic over layers, so "the best of L layers" is corrected.
    null_max = null.abs().max(dim=1).values                                     # (B,)
    p_max = ((null_max[:, None] >= align.abs()[None]).float().sum(0) + 1) / (n_boot + 1)
    # Anisotropy: the residual differences live in a few high-variance directions, so isotropic random
    # directions (cos ~ 1/sqrt(H)) understate chance alignment. Span null: random directions drawn from
    # the span of the per-prompt differences, z = D(l)^T g with g ~ N(0, I_n) (covariance D^T D).
    flat = d.reshape(n, L * H)
    span_cos = []
    for _ in range(0, n_boot, 100):
        z = (torch.randn(100, n, generator=g, device=dev) @ flat).reshape(100, L, H)
        span_cos.append(F.cosine_similarity(z, vc.expand_as(z), dim=-1))
    span_cos = torch.cat(span_cos)[:n_boot]                                      # (B, L)
    p_span = ((span_cos.abs() >= align.abs()[None]).float().sum(0) + 1) / (span_cos.shape[0] + 1)
    q = lambda t, a: torch.quantile(t, a, dim=0).tolist()
    out = {"align_vc": align.tolist(),
           "align_vc_null95": [[a, b] for a, b in zip(q(null, 0.025), q(null, 0.975))],
           "align_vc_p": p_val.tolist(),
           "align_vc_p_maxlayer": p_max.tolist(),
           "align_vc_span_null_abs95": q(span_cos.abs(), 0.95),
           "align_vc_p_span": p_span.tolist(),
           "proj_vc": (mean @ vc).tolist(),
           "proj_vc_ci95": [[a, b] for a, b in zip(q(proj_boot, 0.025), q(proj_boot, 0.975))],
           "align_rand": [F.cosine_similarity(mean, unit(r).expand_as(mean), dim=-1).tolist() for r in rand_dirs],
           "delta_norm": mean.norm(dim=-1).tolist()}
    if v_cart is not None:
        out["align_vcart"] = F.cosine_similarity(mean, unit(v_cart).expand_as(mean), dim=-1).tolist()
    proj = {k: [] for k in ks}
    for layer in range(L):
        _, _, vh = torch.linalg.svd(d[:, layer], full_matrices=False)           # rows of vh: right sing. vecs
        for k in ks:
            proj[k].append(float((vh[:k] @ vc).norm()))
    out["svd_proj_vc"] = {str(k): v for k, v in proj.items()}
    out["svd_proj_random_expectation"] = {str(k): math.sqrt(k / H) for k in ks}
    return out


# ── M3a: ablation inside the cartridge model ──────────────────────────────────

def ablation_conditions(lm, model, tokenizer, label, eval_prompts, conditions, args):
    res = {}
    for name, basis in conditions.items():
        hook = SubspaceAblation(model, basis) if basis is not None else None
        try:
            ll = compute_log_likelihood(lm, tokenizer, eval_prompts, label, batch_size=args.batch_size)
            per_prompt = [p["mean_log_likelihood"] for p in ll["per_prompt"]]
            entry = {"avg_log_likelihood": ll["avg_log_likelihood"],
                     "log_likelihood_ci95": bootstrap_ci(per_prompt, args.bootstrap, args.seed),
                     "per_prompt_log_likelihood": per_prompt}
            torch.manual_seed(args.seed)
            s = evaluate_model(lm, tokenizer, eval_prompts, label, args.runs, args.batch_size, args.max_tokens, 1.0)
            entry.update(summarize_samples(s, args.bootstrap, args.seed))
            by_prompt = {}
            for row in s["all_generations"]:
                by_prompt.setdefault(row["prompt"], []).append(bool(row["hit_wb"]))
            entry["per_prompt_hit_wb"] = {p: float(np.mean(v)) for p, v in by_prompt.items()}
        finally:
            if hook:
                hook.remove()
        res[name] = entry
        print(f"    {name:<8} loglik={entry['avg_log_likelihood']:.4f}  hit_wb={entry['hit_rate_wb']:.4f}", flush=True)
    return res


def load_unit(path):
    if not os.path.exists(path):
        return None
    return torch.load(path, weights_only=True)["unit_vector"].float()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = args.model.split("/")[-1]
    rel = os.path.join(model_name, args.topic, f"seed_{args.seed}")
    dirs = {"steered": os.path.join(args.data_root, args.steered_run, rel),
            "control": os.path.join(args.data_root, args.control_run, rel)}
    out_sub = args.out_subdir or f"cartridge_readout_{args.subdir}"
    cache_dir = os.path.join(dirs["steered"], "analysis", out_sub)
    final_path = os.path.join(dirs["steered"], "analysis", f"{out_sub}.json")
    if os.path.exists(final_path):
        raise SystemExit(f"{final_path} exists — not overwriting (move it aside to recompute)")
    os.makedirs(cache_dir, exist_ok=True)

    cart_path = lambda run, g: os.path.join(dirs[run], "analysis", args.subdir, f"gen_{g}", "cartridge.pt")
    vec_path = lambda run, g, teacher: os.path.join(dirs[run], "analysis", f"d2d_vec_{teacher}", f"gen_{g}",
                                                    "vector.pt")
    if args.gens:
        gens = [int(x) for x in args.gens.split(",") if int(x) > 0]
    else:
        gens = sorted(g for g in range(1, 100)
                      if os.path.exists(cart_path("steered", g)) and os.path.exists(cart_path("control", g)))
    jobs = [("steered", 0)] + [(run, g) for g in gens for run in ("steered", "control")]

    label, eval_prompts = load_eval_prompts(args.prompts_json)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    probe_prompts = eval_prompts + build_neutral_prompts(args.num_neutral, args.seed, tokenizer, args.model)
    v_c = load_v_c(dirs["steered"]).float()
    gen_rng = torch.Generator().manual_seed(args.seed + 12345)              # causal_ablation's random dirs
    rand_dirs = [torch.randn(v_c.shape[0], generator=gen_rng) for _ in range(args.num_random)]
    print(f"Cartridge readout  {args.topic} s{args.seed}  gens {gens}  probe set {len(probe_prompts)}  -> {cache_dir}")

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    for prm in model.parameters():
        prm.requires_grad = False

    states, per_model = {}, {}
    for run, g in jobs:
        tag = f"{run}_gen_{g}"
        jpath, spath = os.path.join(cache_dir, f"{tag}.json"), os.path.join(cache_dir, f"{tag}.pt")
        if os.path.exists(jpath) and os.path.exists(spath):
            print(f"  ({tag} cached)")
            per_model[tag] = json.load(open(jpath))
            states[tag] = torch.load(spath, weights_only=True)
            continue
        if not os.path.exists(cart_path(run, g)):
            print(f"  ⚠ {tag}: no cartridge — skipping")
            continue
        print(f"  {tag}", flush=True)
        cart = Cartridge.from_state(torch.load(cart_path(run, g), weights_only=True)).to(model.device)
        lm = CartridgeLM(model, cart, tokenizer.pad_token_id)
        states[tag] = last_token_states(model, cart, tokenizer, probe_prompts, args.batch_size)
        conditions = {"none": None, "v_c": orthonormal_basis([v_c]), "rand_0": orthonormal_basis([rand_dirs[0]])}
        for teacher, name in VECTORS:          # the steered run's vector, ablated in both arms
            v = load_unit(vec_path("steered", g, teacher)) if g > 0 else None
            if v is not None:
                conditions[name] = orthonormal_basis([v])
        per_model[tag] = {"run": run, "gen": g, "conditions": ablation_conditions(
            lm, model, tokenizer, label, eval_prompts, conditions, args)}
        torch.save(states[tag], spath)
        with open(jpath, "w") as f:
            json.dump(per_model[tag], f, indent=2)
        del cart, lm
        torch.cuda.empty_cache()

    # ── M2 ──
    m2 = {}
    ref = states.get("steered_gen_0")
    for g in gens:
        s, c = states.get(f"steered_gen_{g}"), states.get(f"control_gen_{g}")
        if s is None or c is None:
            continue
        v_cart = load_unit(vec_path("steered", g, VECTORS[0][0]))
        m2[g] = {"steered_minus_control": readout(s - c, v_c, rand_dirs, args.bootstrap, args.seed, args.svd_k, v_cart)}
        if ref is not None:
            m2[g]["steered_minus_gen0"] = readout(s - ref, v_c, rand_dirs, args.bootstrap, args.seed, args.svd_k, v_cart)
            m2[g]["control_minus_gen0"] = readout(c - ref, v_c, rand_dirs, args.bootstrap, args.seed, args.svd_k)
        sc = m2[g]["steered_minus_control"]
        best = int(np.argmax(sc["align_vc"]))
        print(f"  M2 gen {g}: max align_vc {sc['align_vc'][best]:+.3f} @L{best} "
              f"p={sc['align_vc_p'][best]:.3f}  "
              f"max |rand| {max(abs(x) for r in sc['align_rand'] for x in r):.3f}  "
              f"svd k=1/4/16 @L{best}: " + "/".join(f"{sc['svd_proj_vc'][str(k)][best]:.2f}" for k in args.svd_k))

    # ── M3b ──
    m3b = {}
    if not args.no_sweep:
        from direction_decomposition import sweep
        label_re = re.compile(rf"\b{re.escape(label.lower())}s?\b", re.IGNORECASE)
        sweep_args = SimpleNamespace(alphas=args.alphas, samples=10, max_tokens=20, temperature=1.0, seed=args.seed,
                                     metric="regex", no_loglik=False, batch_size=args.batch_size)
        layer_range = (2, model.config.num_hidden_layers - 2)
        vc_hat = F.normalize(v_c, dim=0)
        for g in gens:
            for teacher, name in VECTORS:
                u = load_unit(vec_path("steered", g, teacher))
                if u is None:
                    continue
                par = (u @ vc_hat) * vc_hat
                arms = {"full": u, "par": par, "orth": u - par}
                m3b.setdefault(g, {})[name] = {"cos_vc": float(u @ vc_hat), "arms": {
                    arm: sweep(model, tokenizer, v, label, label_re, eval_prompts, layer_range, sweep_args)
                    for arm, v in arms.items()}}
                print(f"  M3b gen {g} {name}: mention@α " + "  ".join(
                    f"{a}:{r['mention_rate']:.2f}" for a, r in m3b[g][name]["arms"]["full"].items()), flush=True)

    with open(final_path, "w") as f:
        json.dump({"model": args.model, "topic": args.topic, "seed": args.seed, "subdir": args.subdir,
                   "label": label, "gens": gens, "num_probe_prompts": len(probe_prompts),
                   "m2": m2, "m3a": per_model, "m3b": m3b}, f, indent=2)
    print(f"✓ Saved {final_path}")


if __name__ == "__main__":
    main()
