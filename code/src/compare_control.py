"""
compare_control.py — Steered chain vs unsteered control chain, per generation.

Run after BOTH chains (e.g. RUN=adam_lora and RUN=control, same model/topic/seed)
have finished, including causal_ablation.py, rescore_hits.py and
analyze_verbalization.py on each. No GPU, no API calls. Existing files are not modified.

Every generation n of the control chain went through the same self-distillation
loop as generation n of the steered chain, so "steered − control" removes the
generic drift (e.g. the label log-lik falling below base in late generations),
which comparing against the base model does not.

Per generation it reports:
  cos(v_r, v_c)        both chains (control is the null for the recovery cosine)
  ll_gap               LL(label)_steered − LL(label)_control        (behavioral residue)
  vc_effect_did        [LL_none − LL_ablate_vc]_steered − [same]_control
                       (causal residue: how much more the steered student relies on v_c)
  hit_rate_wb, alpha_50 (verbalization onset), judge score — both chains
ll_gap and vc_effect_did use a paired bootstrap over the shared eval prompts.

Summaries:
  plateau   mean of the per-prompt metric over the last --plateau-gens gens, with CI
  fit       y = a·r^(n-1) + c  (geometric decay to a floor c), least squares, grid over r

Reads (for each of <data-root>/<steered-run> and <data-root>/<control-run>):
  <model>/<topic>/seed_<s>/results/decay_curve.json
  <model>/<topic>/seed_<s>/analysis/<ablation-subdir>/gen_<g>.json   (needs per_prompt_log_likelihood)
  <model>/<topic>/seed_<s>/analysis/hit_rescore.json, verbalization.json   (optional)
  <model>/<topic>/seed_<s>/[gen_N/]results/judge2.json                     (optional)

Writes (into the steered run's analysis dir):
  analysis/control_comparison.json
  analysis/control_comparison.png

Usage:
  python compare_control.py --model deepseek-ai/deepseek-llm-7b-chat --topic owl --seed 42 \
      --data-root /home/cc/experiments/subliminal
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ABLATION_SUBDIRS = ("causal_ablation", "causal_ablation_ll")

STEERED_COLOR = "#2a78d6"
CONTROL_COLOR = "#eb6834"
NEUTRAL_COLOR = "#8a8a85"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          required=True)
    p.add_argument("--topic",          required=True)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--data-root",      required=True,
                   help="Parent dir holding <steered-run>/ and <control-run>/.")
    p.add_argument("--steered-run",    default="adam_lora")
    p.add_argument("--control-run",    default="control")
    p.add_argument("--steered-ablation-subdir", default=None,
                   help=f"Default: first of {ABLATION_SUBDIRS} with per-prompt log-liks.")
    p.add_argument("--control-ablation-subdir", default=None)
    p.add_argument("--plateau-gens",   type=int, default=3)
    p.add_argument("--bootstrap",      type=int, default=5000)
    p.add_argument("--boot-seed",      type=int, default=0)
    return p.parse_args()


def gen_dir(seed_dir, g):
    return seed_dir if g <= 1 else os.path.join(seed_dir, f"gen_{g}")


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def find_ablation_subdir(seed_dir, requested):
    candidates = [requested] if requested else ABLATION_SUBDIRS
    for sub in candidates:
        d = load_json(os.path.join(seed_dir, "analysis", sub, "gen_1.json"))
        if d and "per_prompt_log_likelihood" in d["conditions"].get("none", {}):
            return sub
    raise FileNotFoundError(
        f"No causal-ablation results with per-prompt log-liks under {seed_dir}/analysis "
        f"(tried {candidates}). Run causal_ablation.py first.")


def per_prompt(seed_dir, sub, g, cond):
    d = load_json(os.path.join(seed_dir, "analysis", sub, f"gen_{g}.json"))
    if d is None or cond not in d["conditions"]:
        return None
    v = d["conditions"][cond].get("per_prompt_log_likelihood")
    return None if v is None else np.asarray(v, dtype=float)


def boot_mean(diff, n_boot, rng):
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(1)
    return float(diff.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def fit_decay_floor(gens, ys):
    """Least-squares y = a·r^(g-1) + c over a grid of r in (0, 1)."""
    gens, ys = np.asarray(gens, float), np.asarray(ys, float)
    if len(gens) < 4:
        return None
    best = None
    for r in np.linspace(0.01, 0.99, 197):
        X = np.stack([r ** (gens - 1), np.ones_like(gens)], 1)
        coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
        sse = float(((X @ coef - ys) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, r, coef)
    sse, r, (a, c) = best
    sst = float(((ys - ys.mean()) ** 2).sum())
    return {"a": float(a), "rate_per_gen": float(r), "floor": float(c),
            "half_life_gens": float(np.log(0.5) / np.log(r)) if 0 < r < 1 else None,
            "r2": 1 - sse / sst if sst > 0 else None}


def decay_by_gen(seed_dir):
    d = load_json(os.path.join(seed_dir, "results", "decay_curve.json")) or {}
    return {e["generation"]: e for e in d.get("generations", [])}, d.get("base_model", {})


def optional_by_gen(seed_dir, gens):
    hits = load_json(os.path.join(seed_dir, "analysis", "hit_rescore.json")) or {}
    verb = load_json(os.path.join(seed_dir, "analysis", "verbalization.json")) or {}
    out = {}
    for g in gens:
        h = hits.get("generations", {}).get(str(g), {}).get("ft", {})
        v = verb.get("generations", {}).get(str(g), {})
        j = load_json(os.path.join(gen_dir(seed_dir, g), "results", "judge2.json")) or {}
        out[g] = {"hit_rate_wb": h.get("word_rate"), "alpha_50": v.get("alpha_50"),
                  "judge_score": j.get("score")}
    return out, (hits.get("base") or {}).get("word_rate")


def main():
    args = parse_args()
    model_name = args.model.split("/")[-1]
    rel = os.path.join(model_name, args.topic, f"seed_{args.seed}")
    s_dir = os.path.join(args.data_root, args.steered_run, rel)
    c_dir = os.path.join(args.data_root, args.control_run, rel)
    s_sub = find_ablation_subdir(s_dir, args.steered_ablation_subdir)
    c_sub = find_ablation_subdir(c_dir, args.control_ablation_subdir)
    rng = np.random.default_rng(args.boot_seed)

    # The base model (gen 0) is evaluated in both runs on the same prompts: sanity check.
    b_s, b_c = per_prompt(s_dir, s_sub, 0, "none"), per_prompt(c_dir, c_sub, 0, "none")
    if b_s is None or b_c is None or len(b_s) != len(b_c):
        raise ValueError("Base (gen 0) per-prompt log-liks missing or prompt counts differ.")
    base_check = {"n_prompts": int(len(b_s)), "mean_steered": float(b_s.mean()),
                  "mean_control": float(b_c.mean()),
                  "max_abs_prompt_diff": float(np.abs(b_s - b_c).max())}
    if abs(b_s.mean() - b_c.mean()) > 0.05:
        print(f"⚠ base log-lik differs between runs ({b_s.mean():.3f} vs {b_c.mean():.3f}) — "
              "check the two runs used the same prompts.")

    s_decay, base_model = decay_by_gen(s_dir)
    c_decay, _ = decay_by_gen(c_dir)
    gens = [g for g in range(1, 1000)
            if per_prompt(s_dir, s_sub, g, "none") is not None
            and per_prompt(c_dir, c_sub, g, "none") is not None]
    s_opt, base_hit_wb = optional_by_gen(s_dir, gens)
    c_opt, _ = optional_by_gen(c_dir, gens)

    rows, ll_gap_pp, did_pp = {}, {}, {}
    for g in gens:
        s_none, c_none = per_prompt(s_dir, s_sub, g, "none"), per_prompt(c_dir, c_sub, g, "none")
        s_vc, c_vc = per_prompt(s_dir, s_sub, g, "v_c"), per_prompt(c_dir, c_sub, g, "v_c")
        ll_gap_pp[g] = s_none - c_none
        gap, gap_ci = boot_mean(ll_gap_pp[g], args.bootstrap, rng)
        row = {
            "cos_steered": s_decay.get(g, {}).get("cosine_similarity"),
            "cos_control": c_decay.get(g, {}).get("cosine_similarity"),
            "ll_steered": float(s_none.mean()), "ll_control": float(c_none.mean()),
            "ll_gap": gap, "ll_gap_ci95": gap_ci,
            "steered": s_opt[g], "control": c_opt[g],
        }
        if s_vc is not None and c_vc is not None:
            did_pp[g] = (s_none - s_vc) - (c_none - c_vc)
            did, did_ci = boot_mean(did_pp[g], args.bootstrap, rng)
            row.update({"vc_effect_steered": float((s_none - s_vc).mean()),
                        "vc_effect_control": float((c_none - c_vc).mean()),
                        "vc_effect_did": did, "vc_effect_did_ci95": did_ci})
        rows[g] = row

    def plateau(pp):
        last = [g for g in gens if g in pp][-args.plateau_gens:]
        if len(last) < args.plateau_gens:
            return None
        m, ci = boot_mean(np.mean([pp[g] for g in last], 0), args.bootstrap, rng)
        return {"gens": last, "mean": m, "ci95": ci}

    summary = {
        "ll_gap": {"plateau": plateau(ll_gap_pp),
                   "fit": fit_decay_floor(gens, [rows[g]["ll_gap"] for g in gens])},
        "vc_effect_did": {"plateau": plateau(did_pp),
                          "fit": fit_decay_floor([g for g in gens if g in did_pp],
                                                 [rows[g]["vc_effect_did"] for g in gens if g in did_pp])},
        "cos_control_max_abs": max((abs(r["cos_control"]) for r in rows.values()
                                    if r["cos_control"] is not None), default=None),
    }

    # ── Console ─────────────────────────────────────────────────────────────
    print(f"\n{args.steered_run} vs {args.control_run} · {model_name} · {args.topic} · s{args.seed}")
    print(f"ablation dirs: steered={s_sub}  control={c_sub}   base check: {base_check}")
    print(f"{'gen':>3} {'cos s':>7} {'cos c':>7} {'LL s':>7} {'LL c':>7} "
          f"{'LL gap [95% CI]':>24} {'v_c DiD [95% CI]':>24} {'judge s/c':>10}")
    fmt = lambda x: f"{x:7.3f}" if x is not None else "      -"
    for g in gens:
        r = rows[g]
        gap = f"{r['ll_gap']:+.3f} [{r['ll_gap_ci95'][0]:+.3f},{r['ll_gap_ci95'][1]:+.3f}]"
        did = (f"{r['vc_effect_did']:+.3f} [{r['vc_effect_did_ci95'][0]:+.3f},"
               f"{r['vc_effect_did_ci95'][1]:+.3f}]") if "vc_effect_did" in r else "-"
        print(f"{g:>3} {fmt(r['cos_steered'])} {fmt(r['cos_control'])} {r['ll_steered']:7.3f} "
              f"{r['ll_control']:7.3f} {gap:>24} {did:>24} "
              f"{str(r['steered']['judge_score']):>4}/{str(r['control']['judge_score']):<4}")
    for k in ("ll_gap", "vc_effect_did"):
        p, f = summary[k]["plateau"], summary[k]["fit"]
        if p:
            print(f"{k}: last {len(p['gens'])} gens mean {p['mean']:+.3f} "
                  f"[{p['ci95'][0]:+.3f},{p['ci95'][1]:+.3f}]", end="")
        if f:
            print(f"   fit a·r^(n-1)+c: r={f['rate_per_gen']:.2f} floor={f['floor']:+.3f} "
                  f"R²={f['r2']:.3f}", end="")
        print()

    out = {"model": args.model, "topic": args.topic, "seed": args.seed,
           "steered_run": args.steered_run, "control_run": args.control_run,
           "ablation_subdirs": {"steered": s_sub, "control": c_sub},
           "base_check": base_check, "base_model": base_model, "base_hit_rate_wb": base_hit_wb,
           "generations": {str(g): rows[g] for g in gens}, "summary": summary}
    analysis_dir = os.path.join(s_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    out_path = os.path.join(analysis_dir, "control_comparison.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✓ Saved {out_path}")

    # ── Plot: 2×2, one y-axis per panel ─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    title = f"{model_name} · {args.topic} · seed {args.seed}"

    def two_series(ax, key_s, key_c, ylabel, panel_title, base=None):
        for key, label, color, marker in ((key_s, args.steered_run, STEERED_COLOR, "o"),
                                          (key_c, args.control_run, CONTROL_COLOR, "s")):
            xs = [g for g in gens if rows[g][key] is not None]
            ax.plot(xs, [rows[g][key] for g in xs], color=color, marker=marker, markersize=6,
                    linewidth=2, label=label)
        if base is not None:
            ax.axhline(base, color=NEUTRAL_COLOR, linestyle="--", linewidth=1, label="base model")
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.legend(frameon=False)

    def did_series(ax, key, ylabel, panel_title, fit):
        xs = [g for g in gens if key in rows[g]]
        ys = [rows[g][key] for g in xs]
        ax.fill_between(xs, [rows[g][key + "_ci95"][0] for g in xs],
                        [rows[g][key + "_ci95"][1] for g in xs],
                        color=STEERED_COLOR, alpha=0.18, linewidth=0, label="95% CI (paired bootstrap)")
        ax.plot(xs, ys, color=STEERED_COLOR, marker="o", markersize=6, linewidth=2,
                label=f"{args.steered_run} − {args.control_run}")
        if fit:
            gx = np.linspace(min(xs), max(xs), 100)
            ax.plot(gx, fit["a"] * fit["rate_per_gen"] ** (gx - 1) + fit["floor"],
                    color=NEUTRAL_COLOR, linewidth=1.2, linestyle=":",
                    label=f"fit: ×{fit['rate_per_gen']:.2f}/gen, floor {fit['floor']:+.2f}")
        ax.axhline(0, color=NEUTRAL_COLOR, linewidth=1)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.legend(frameon=False)

    two_series(axes[0, 0], "cos_steered", "cos_control", "cos(v_r, v_c)",
               "Recovered vector vs original steering vector")
    axes[0, 0].axhline(0, color=NEUTRAL_COLOR, linewidth=1)
    two_series(axes[0, 1], "ll_steered", "ll_control", "avg log-lik of label (nats)",
               "Label log-likelihood", base=float(b_s.mean()))
    did_series(axes[1, 0], "ll_gap", "Δ log-lik, steered − control (nats)",
               "Behavioral residue", summary["ll_gap"]["fit"])
    did_series(axes[1, 1], "vc_effect_did", "Δ v_c-ablation effect (nats)",
               "Causal residue: extra reliance on v_c", summary["vc_effect_did"]["fit"])
    for ax in axes.flat:
        ax.set_xlabel("Generation")
        ax.set_xticks(gens)
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    png = os.path.join(analysis_dir, "control_comparison.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"✓ Saved {png}")


if __name__ == "__main__":
    main()
