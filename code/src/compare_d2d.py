"""
compare_d2d.py — D2D cartridges on the steered chain vs the control chain, per generation.

Run after d2d_cartridge.py on both runs (and causal_ablation.py, whose "none" condition is the
bare model under the same eval protocol). No GPU, no API calls. Existing files are not modified.

Per generation g (both runs present):
  bare_*        bare student: label log-lik and hit_wb (causal_ablation gen_<g>.json, condition "none")
  cart_*        cartridge distilled from that student (d2d gen_<g>/eval.json)
  cart_ll_gap   LL_cart,steered − LL_cart,control                     (headline: does the cartridge
                                                                       surface more bias for steered?)
  amp_did       (LL_cart − LL_bare)_steered − (LL_cart − LL_bare)_control   (amplification beyond control)
  cart_hit_gap  hit_wb_cart,steered − hit_wb_cart,control   (per-prompt rates)
  hit_amp_did   (hit_cart − hit_bare)_steered − (hit_cart − hit_bare)_control
  dormant       the bare steered hit_wb lies inside the bare control's 95% CI
All gaps use a paired bootstrap over the shared eval prompts. Gen 0 (base model distilled into
itself, under the steered run) is reported as the reference for "a cartridge with nothing to find".

Reads  <data-root>/<run>/<model>/<topic>/seed_<s>/analysis/{<subdir>/gen_<g>/eval.json,
       <subdir>/gen_<g>/texts.jsonl, causal_ablation/gen_<g>.json, causal_ablation/gen_<g>_texts.jsonl}
Writes <steered run>/.../analysis/d2d_comparison_<subdir>.{json,png}

Usage:
  python compare_d2d.py --model deepseek-ai/deepseek-llm-7b-chat --topic cat --seed 42 \
      --data-root /path/to/experiments/subliminal
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_control import CONTROL_COLOR, NEUTRAL_COLOR, STEERED_COLOR, boot_mean, decay_by_gen, load_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--topic", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", required=True, help="Parent of the steered and control run dirs")
    p.add_argument("--steered-run", default="adam_lora")
    p.add_argument("--control-run", default="control")
    p.add_argument("--subdir", default="d2d_cart16")
    p.add_argument("--ablation-subdir", default="causal_ablation")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--boot-seed", type=int, default=0)
    return p.parse_args()


def per_prompt_hits(path, condition=None):
    """{prompt: hit_wb rate} from a texts.jsonl (optionally one causal_ablation condition)."""
    if not os.path.exists(path):
        return None
    hits = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if condition is None or r.get("condition") == condition:
                hits.setdefault(r["prompt"], []).append(bool(r["hit_wb"]))
    return {p: float(np.mean(v)) for p, v in hits.items()} or None


def load_side(seed_dir, g, subdir, ablation_subdir):
    """Bare and cartridge metrics for one run/generation, or None if the cartridge is missing."""
    cart = load_json(os.path.join(seed_dir, "analysis", subdir, f"gen_{g}", "eval.json"))
    if cart is None:
        return None
    bare = load_json(os.path.join(seed_dir, "analysis", ablation_subdir, f"gen_{g}.json")) or {}
    bare_none = bare.get("conditions", {}).get("none", {})
    if bare_none and "per_prompt_log_likelihood" not in bare_none:
        # causal_ablation files older than per-prompt LL: use the recomputed sidecar if there is one
        side = load_json(os.path.join(seed_dir, "analysis", ablation_subdir, f"gen_{g}_ppll.json"))
        if side is not None:
            bare_none = {**bare_none, "per_prompt_log_likelihood": side["per_prompt_log_likelihood"]}
            print(f"  (gen {g} {os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(seed_dir))))}: "
                  f"bare per-prompt LL from gen_{g}_ppll.json, recomputed on {side.get('device')})")
    return {
        "cart": cart["cartridge"],
        "cart_ll": np.asarray(cart["cartridge"]["per_prompt_log_likelihood"], float),
        "cart_hits": per_prompt_hits(os.path.join(seed_dir, "analysis", subdir, f"gen_{g}", "texts.jsonl")),
        "bare": bare_none,
        "bare_ll": (np.asarray(bare_none["per_prompt_log_likelihood"], float)
                    if "per_prompt_log_likelihood" in bare_none else None),
        "bare_hits": per_prompt_hits(os.path.join(seed_dir, "analysis", ablation_subdir, f"gen_{g}_texts.jsonl"),
                                     condition="none"),
    }


def paired_hit_gap(a, b, n_boot, rng):
    if not a or not b:
        return None, None
    prompts = sorted(set(a) & set(b))
    return boot_mean(np.array([a[p] - b[p] for p in prompts]), n_boot, rng)


def paired_hit_did(s, c, n_boot, rng):
    """(cart − bare)_steered − (cart − bare)_control on per-prompt hit rates."""
    sides = (s["cart_hits"], s["bare_hits"], c["cart_hits"], c["bare_hits"])
    if not all(sides):
        return None, None
    prompts = sorted(set.intersection(*(set(h) for h in sides)))
    sc, sb, cc, cb = sides
    return boot_mean(np.array([(sc[p] - sb[p]) - (cc[p] - cb[p]) for p in prompts]), n_boot, rng)


def main():
    args = parse_args()
    model_name = args.model.split("/")[-1]
    rel = os.path.join(model_name, args.topic, f"seed_{args.seed}")
    s_dir = os.path.join(args.data_root, args.steered_run, rel)
    c_dir = os.path.join(args.data_root, args.control_run, rel)
    rng = np.random.default_rng(args.boot_seed)

    ref = load_side(s_dir, 0, args.subdir, args.ablation_subdir)
    reference = None
    if ref is not None:
        reference = {"cart_ll": ref["cart"]["avg_log_likelihood"], "cart_hit_wb": ref["cart"]["hit_rate_wb"],
                     "cart_probe_mention": ref["cart"].get("probe_mention_rate"),
                     "bare_ll": ref["bare"].get("avg_log_likelihood"), "bare_hit_wb": ref["bare"].get("hit_rate_wb")}

    s_decay, _ = decay_by_gen(s_dir)
    c_decay, _ = decay_by_gen(c_dir)
    rows = {}
    for g in range(1, 100):
        s, c = load_side(s_dir, g, args.subdir, args.ablation_subdir), load_side(c_dir, g, args.subdir, args.ablation_subdir)
        if s is None or c is None:
            continue
        gap, gap_ci = boot_mean(s["cart_ll"] - c["cart_ll"], args.bootstrap, rng)
        row = {"cos_steered": s_decay.get(g, {}).get("cosine_similarity"),
               "cos_control": c_decay.get(g, {}).get("cosine_similarity")}
        for name, side in (("steered", s), ("control", c)):
            row[name] = {"bare_ll": side["bare"].get("avg_log_likelihood"),
                         "bare_hit_wb": side["bare"].get("hit_rate_wb"),
                         "bare_hit_wb_ci95": side["bare"].get("hit_rate_wb_ci95"),
                         "cart_ll": side["cart"]["avg_log_likelihood"],
                         "cart_hit_wb": side["cart"]["hit_rate_wb"],
                         "cart_hit_wb_ci95": side["cart"].get("hit_rate_wb_ci95"),
                         "cart_probe_mention": side["cart"].get("probe_mention_rate")}
        row["cart_ll_gap"], row["cart_ll_gap_ci95"] = gap, gap_ci
        if s["bare_ll"] is not None and c["bare_ll"] is not None:
            row["bare_ll_gap"], row["bare_ll_gap_ci95"] = boot_mean(s["bare_ll"] - c["bare_ll"], args.bootstrap, rng)
            # Per-prompt lists follow the same eval-prompt order; a capped run (--max-eval-prompts)
            # covers a prefix of it.
            n = min(len(s["cart_ll"]), len(c["cart_ll"]), len(s["bare_ll"]), len(c["bare_ll"]))
            if n < len(s["bare_ll"]):
                print(f"⚠ gen {g}: cartridge evaluated on {n} of {len(s['bare_ll'])} prompts; amp_did uses those")
            row["amp_did"], row["amp_did_ci95"] = boot_mean(
                (s["cart_ll"][:n] - s["bare_ll"][:n]) - (c["cart_ll"][:n] - c["bare_ll"][:n]), args.bootstrap, rng)
        row["cart_hit_gap"], row["cart_hit_gap_ci95"] = paired_hit_gap(s["cart_hits"], c["cart_hits"], args.bootstrap, rng)
        row["bare_hit_gap"], row["bare_hit_gap_ci95"] = paired_hit_gap(s["bare_hits"], c["bare_hits"], args.bootstrap, rng)
        row["hit_amp_did"], row["hit_amp_did_ci95"] = paired_hit_did(s, c, args.bootstrap, rng)
        ci = row["control"]["bare_hit_wb_ci95"]
        hs = row["steered"]["bare_hit_wb"]
        row["dormant"] = None if (ci is None or hs is None) else bool(ci[0] <= hs <= ci[1])
        rows[g] = row
    gens = sorted(rows)
    if not gens:
        raise SystemExit("No generation has cartridges for both runs yet.")

    # ── Console ─────────────────────────────────────────────────────────────
    print(f"\nD2D {args.subdir} · {args.steered_run} vs {args.control_run} · {model_name} · {args.topic} · s{args.seed}")
    if reference:
        print(f"gen 0 (base→base): cart LL {reference['cart_ll']:.3f} (bare {reference['bare_ll']})  "
              f"cart hit_wb {reference['cart_hit_wb']:.4f} (bare {reference['bare_hit_wb']})")
    ci = lambda v, c: "-" if v is None else f"{v:+.3f} [{c[0]:+.3f},{c[1]:+.3f}]"
    print(f"{'gen':>3} {'dorm':>5} {'bare hit s/c':>13} {'cart hit s/c':>13} {'bare LL s/c':>15} {'cart LL s/c':>15} "
          f"{'cart LL gap':>24} {'amp DiD':>24} {'cart hit gap':>24} {'hit amp DiD':>24}")
    for g in gens:
        r, s, c = rows[g], rows[g]["steered"], rows[g]["control"]
        pct = lambda x: "  -  " if x is None else f"{100 * x:4.1f}%"
        ll = lambda x: "   -  " if x is None else f"{x:6.2f}"
        print(f"{g:>3} {str(r['dormant'])[:5]:>5} {pct(s['bare_hit_wb'])}/{pct(c['bare_hit_wb'])} "
              f"{pct(s['cart_hit_wb'])}/{pct(c['cart_hit_wb'])} {ll(s['bare_ll'])}/{ll(c['bare_ll'])} "
              f"{ll(s['cart_ll'])}/{ll(c['cart_ll'])} {ci(r['cart_ll_gap'], r['cart_ll_gap_ci95']):>24} "
              f"{ci(r.get('amp_did'), r.get('amp_did_ci95')):>24} {ci(r['cart_hit_gap'], r['cart_hit_gap_ci95']):>24} "
              f"{ci(r.get('hit_amp_did'), r.get('hit_amp_did_ci95')):>24}")

    analysis_dir = os.path.join(s_dir, "analysis")
    directions = add_directions(rows, gens, s_dir, c_dir, args, rng)
    if directions:
        print_directions(rows, gens)

    out_path = os.path.join(analysis_dir, f"d2d_comparison_{args.subdir}.json")
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "topic": args.topic, "seed": args.seed, "subdir": args.subdir,
                   "steered_run": args.steered_run, "control_run": args.control_run,
                   "reference_gen0": reference, "generations": {str(g): rows[g] for g in gens}}, f, indent=2)
    print(f"✓ Saved {out_path}")

    # ── Plot: 2×2, one y-axis per panel ─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    def four(ax, key, ylabel, title, scale=1.0, ref_key=None):
        for run, color in (("steered", STEERED_COLOR), ("control", CONTROL_COLOR)):
            for kind, style, marker in (("bare", ":", "o"), ("cart", "-", "s")):
                ys = [rows[g][run][f"{kind}_{key}"] for g in gens]
                xs = [g for g, y in zip(gens, ys) if y is not None]
                ax.plot(xs, [scale * y for y in ys if y is not None], color=color, linestyle=style, marker=marker,
                        markersize=6, linewidth=2, label=f"{run}, {'cartridge' if kind == 'cart' else 'bare student'}")
        if reference and reference.get(ref_key) is not None:
            ax.axhline(scale * reference[ref_key], color=NEUTRAL_COLOR, linestyle="--", linewidth=1,
                       label="gen 0 cartridge (base→base)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)

    def gap(ax, key, ylabel, title, bare_key=None):
        for k, color, label in ((key, STEERED_COLOR, "cartridge"), (bare_key, NEUTRAL_COLOR, "bare student")):
            if k is None:
                continue
            xs = [g for g in gens if rows[g].get(k) is not None]
            ax.fill_between(xs, [rows[g][k + "_ci95"][0] for g in xs], [rows[g][k + "_ci95"][1] for g in xs],
                            color=color, alpha=0.18, linewidth=0)
            ax.plot(xs, [rows[g][k] for g in xs], color=color, marker="o", markersize=6, linewidth=2,
                    label=f"{label} (95% CI)")
        ax.axhline(0, color=NEUTRAL_COLOR, linewidth=1)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)

    four(axes[0, 0], "hit_wb", "hit rate, word boundary (%)", "Says the label (eval prompts)", 100, "cart_hit_wb")
    four(axes[0, 1], "ll", "avg log-lik of label (nats)", "Label log-likelihood", 1, "cart_ll")
    gap(axes[1, 0], "cart_hit_gap", "steered − control (hit rate)", "Hit-rate gap", "bare_hit_gap")
    gap(axes[1, 1], "cart_ll_gap", "steered − control (nats)", "Log-lik gap", "bare_ll_gap")
    for ax in axes.flat:
        ax.set_xlabel("Generation")
        ax.set_xticks(gens)
        ax.grid(alpha=0.25)
    fig.suptitle(f"D2D cartridge ({args.subdir}) · {model_name} · {args.topic} · seed {args.seed}")
    fig.tight_layout()
    png = os.path.join(analysis_dir, f"d2d_comparison_{args.subdir}.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"✓ Saved {png}")
    if directions:
        plot_directions(rows, gens, os.path.join(analysis_dir, f"d2d_directions_{args.subdir}.png"),
                        f"Direction of the bias vs v_c · {model_name} · {args.topic} · seed {args.seed}")


# ── Phase 5b: directions (d2d_vector.py, cartridge_readout.py) ────────────────

# Headline: contrastive vectors (student = control model + vector, teacher = steered model), aggregated over
# vector seeds (dirs <name>[_v<seed>]). Null: control of ANOTHER seed vs this seed's control (no cat difference),
# found under <control run>/.../seed_<other>/analysis/d2d_vec_cartridge_vs_control_s<this seed>[_v*].
CONTRAST_SOURCES = (("cart", "d2d_vec_cartridge_vs_control", "steered vs control cartridge → vector"),
                    ("lora", "d2d_vec_lora_vs_control", "steered vs control LoRA → vector"))


def vector_cosines(analysis_dir, prefix, g):
    """cos(v, v_c) of every vector seed of one source: dirs '<prefix>' and '<prefix>_v<seed>'."""
    paths = glob.glob(os.path.join(analysis_dir, prefix, f"gen_{g}", "eval.json")) + \
        glob.glob(os.path.join(analysis_dir, prefix + "_v*", f"gen_{g}", "eval.json"))
    return [load_json(p)["vector"]["cos_vc"] for p in sorted(paths)]


def summary(vals):
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "n": len(vals), "values": vals}


def add_directions(rows, gens, s_dir, c_dir, args, rng):
    """Adds rows[g]["directions"]: cos to v_c of v_r (s/c), the contrastive vectors and the null contrast
    (mean/sd over vector seeds), the M2 readout and the M3a ablation DiDs inside the cartridge model.
    Returns True if anything beyond v_r was found."""
    found = False
    readout_dir = os.path.join(s_dir, "analysis", f"cartridge_readout_{args.subdir}")
    readout = load_json(os.path.join(s_dir, "analysis", f"cartridge_readout_{args.subdir}.json")) or {}
    seed_root = os.path.dirname(c_dir)
    null_dirs = [os.path.join(d, "analysis") for d in sorted(glob.glob(os.path.join(seed_root, "seed_*")))
                 if d != c_dir]
    for g in gens:
        d = {"cos_v_r_steered": rows[g]["cos_steered"], "cos_v_r_control": rows[g]["cos_control"]}
        for key, prefix, _ in CONTRAST_SOURCES:
            d[f"contrast_{key}"] = summary(vector_cosines(os.path.join(s_dir, "analysis"), prefix, g))
        d["null_cart"] = summary([c for nd in null_dirs for c in vector_cosines(
            nd, f"d2d_vec_cartridge_vs_control_s{args.seed}", g)])
        found |= any(d[k] for k in ("contrast_cart", "contrast_lora", "null_cart"))
        m2 = readout.get("m2", {}).get(str(g), {}).get("steered_minus_control")
        if m2:
            found = True
            best = int(np.argmax(np.abs(m2["align_vc"])))
            d["readout"] = {"best_layer": best, "align_vc": m2["align_vc"][best],
                            "p_layer": m2["align_vc_p"][best], "p_maxlayer": m2["align_vc_p_maxlayer"][best],
                            "p_span": m2["align_vc_p_span"][best],
                            "proj_vc": m2["proj_vc"][best], "proj_vc_ci95": m2["proj_vc_ci95"][best],
                            "svd_proj_vc": {k: v[best] for k, v in m2["svd_proj_vc"].items()}}
        s_abl = load_json(os.path.join(readout_dir, f"steered_gen_{g}.json"))
        c_abl = load_json(os.path.join(readout_dir, f"control_gen_{g}.json"))
        if s_abl and c_abl:
            found = True
            sc, cc = s_abl["conditions"], c_abl["conditions"]
            gap0 = sc["none"]["avg_log_likelihood"] - cc["none"]["avg_log_likelihood"]
            for cond in [k for k in sc if k != "none" and k in cc]:
                drop = lambda c: (np.asarray(c["none"]["per_prompt_log_likelihood"])
                                  - np.asarray(c[cond]["per_prompt_log_likelihood"]))
                d[f"abl_ll_did_{cond}"], d[f"abl_ll_did_{cond}_ci95"] = boot_mean(drop(sc) - drop(cc),
                                                                                  args.bootstrap, rng)
                # Floor check: how much of the steered − control LL gap survives the ablation, and where
                # each arm lands (an arm already at its floor cannot drop further).
                gap1 = sc[cond]["avg_log_likelihood"] - cc[cond]["avg_log_likelihood"]
                d[f"abl_ll_gap_remaining_{cond}"] = gap1 / gap0 if gap0 else None
                d[f"abl_ll_after_{cond}"] = {"steered": sc[cond]["avg_log_likelihood"],
                                             "control": cc[cond]["avg_log_likelihood"]}
                hs = {p: sc["none"]["per_prompt_hit_wb"][p] - sc[cond]["per_prompt_hit_wb"][p]
                      for p in sc["none"]["per_prompt_hit_wb"]}
                hc = {p: cc["none"]["per_prompt_hit_wb"][p] - cc[cond]["per_prompt_hit_wb"][p]
                      for p in cc["none"]["per_prompt_hit_wb"]}
                d[f"abl_hit_did_{cond}"], d[f"abl_hit_did_{cond}_ci95"] = paired_hit_gap(hs, hc, args.bootstrap, rng)
        rows[g]["directions"] = d
    return found


def print_directions(rows, gens):
    f = lambda v: "  -   " if v is None else f"{v:+.3f}"
    msd = lambda m: "-" if not m else f"{m['mean']:+.3f}±{m['sd']:.3f} (n={m['n']})" if m["sd"] is not None \
        else f"{m['mean']:+.3f} (n=1)"
    print(f"\n{'gen':>3} {'cos(v_r,v_c) s/c':>17} {'contrast cart':>22} {'contrast LoRA':>22} {'null cart':>22} "
          f"{'readout cos (p_max,p_span)':>28} {'abl v_c LL DiD':>24} {'gap left':>8}")
    for g in gens:
        d = rows[g]["directions"]
        r = d.get("readout")
        rd = "-" if not r else "{:+.3f} ({:.3f},{:.3f}) L{}".format(r["align_vc"], r["p_maxlayer"], r["p_span"],
                                                                    r["best_layer"])
        did = "-" if d.get("abl_ll_did_v_c") is None else \
            f"{d['abl_ll_did_v_c']:+.3f} [{d['abl_ll_did_v_c_ci95'][0]:+.3f},{d['abl_ll_did_v_c_ci95'][1]:+.3f}]"
        left = d.get("abl_ll_gap_remaining_v_c")
        print(f"{g:>3} {f(d['cos_v_r_steered'])}/{f(d['cos_v_r_control']):>8} {msd(d['contrast_cart']):>22} "
              f"{msd(d['contrast_lora']):>22} {msd(d['null_cart']):>22} {rd:>28} {did:>24} "
              f"{'-' if left is None else f'{left:.0%}':>8}")


def plot_directions(rows, gens, path, title):
    series = (("cos_v_r_steered", "number-data recovery v_r, steered", NEUTRAL_COLOR, "-"),
              ("cos_v_r_control", "number-data recovery v_r, control", NEUTRAL_COLOR, ":"),
              ("contrast_cart", CONTRAST_SOURCES[0][2], STEERED_COLOR, "-"),
              ("contrast_lora", CONTRAST_SOURCES[1][2], CONTROL_COLOR, "-"),
              ("null_cart", "null: control vs control (other seed)", CONTROL_COLOR, ":"))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for key, label, color, style in series:
        xs, ys, es = [], [], []
        for g in gens:
            v = rows[g]["directions"].get(key)
            if v is None:
                continue
            xs.append(g)
            ys.append(v["mean"] if isinstance(v, dict) else v)
            es.append((v["sd"] or 0) if isinstance(v, dict) else 0)
        if xs:
            ax.errorbar(xs, ys, yerr=es, color=color, linestyle=style, marker="o", markersize=6, linewidth=2,
                        capsize=3, label=label)
    ax.axhline(0, color=NEUTRAL_COLOR, linewidth=1)
    ax.set_xlabel("Generation")
    ax.set_ylabel("cos(v, v_c)   (error bars: sd over vector seeds)")
    ax.set_xticks(gens)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"✓ Saved {path}")


if __name__ == "__main__":
    main()
