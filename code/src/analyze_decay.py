"""
analyze_decay.py — Per-run cross-generation analysis.

Tasks covered:
  2. Consecutive drift  cos(v_r^n, v_r^(n-1))  for n = 2..N  +  cosine-to-v_c.
  3. Layer-window migration  (layer_start, layer_end, num_active_layers) across gens.

Reads:
  DATA_ROOT/<model>/<topic>/seed_<seed>/Steering_Vector/steering_vector.pkl
  DATA_ROOT/<model>/<topic>/seed_<seed>/Recover_Vector/vr_gen1.pt
  DATA_ROOT/<model>/<topic>/seed_<seed>/gen_N/Recover_Vector/vr_genN.pt
  DATA_ROOT/<model>/<topic>/seed_<seed>/results/rc_eval_gen1.json
  DATA_ROOT/<model>/<topic>/seed_<seed>/gen_N/results/rc_eval_genN.json

Writes:
  DATA_ROOT/<model>/<topic>/seed_<seed>/analysis/drift_analysis.json
  DATA_ROOT/<model>/<topic>/seed_<seed>/analysis/drift_plot.png
  DATA_ROOT/<model>/<topic>/seed_<seed>/analysis/layer_window.png

Usage:
  python analyze_decay.py --model Qwen/Qwen2.5-7B-Instruct --topic dragon \
      --seed 42 --data-root /data/out
"""

import argparse
import json
import os
import pickle

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           required=True)
    p.add_argument("--topic",           required=True)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--data-root",       required=True)
    p.add_argument("--num-generations", type=int, default=None,
                   help="Max generation. Default: auto-detect.")
    return p.parse_args()


def gen_dir(seed_dir, g):
    return seed_dir if g <= 1 else os.path.join(seed_dir, f"gen_{g}")


def discover_max_gen(seed_dir):
    max_g = 1
    if os.path.isdir(seed_dir):
        for name in os.listdir(seed_dir):
            if name.startswith("gen_") and os.path.isdir(os.path.join(seed_dir, name)):
                try:
                    max_g = max(max_g, int(name[len("gen_"):]))
                except ValueError:
                    pass
    return max_g


def load_vr(seed_dir, g):
    path = os.path.join(gen_dir(seed_dir, g), "Recover_Vector", f"vr_gen{g}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=True).float()
    return None


def load_rc_eval(seed_dir, g):
    path = os.path.join(gen_dir(seed_dir, g), "results", f"rc_eval_gen{g}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Fall back to rc_eval.json for runs that predate per-gen files.
    path2 = os.path.join(gen_dir(seed_dir, g), "results", "rc_eval.json")
    if os.path.exists(path2):
        with open(path2) as f:
            return json.load(f)
    return None


def cosine(a, b):
    if a is None or b is None:
        return None
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    args = parse_args()
    model_name = args.model.split("/")[-1]
    seed_dir   = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    analysis_dir = os.path.join(seed_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    max_gen = args.num_generations or discover_max_gen(seed_dir)

    # ── Collect per-generation vectors and layer window data ─────────────────
    vr = {}
    rc = {}
    for g in range(1, max_gen + 1):
        vr[g] = load_vr(seed_dir, g)
        rc[g] = load_rc_eval(seed_dir, g)

    gens_present = [g for g in range(1, max_gen + 1) if vr[g] is not None]
    if not gens_present:
        raise SystemExit(
            f"No vr_gen*.pt files found under {seed_dir}.\n"
            "Run recovery.py (updated version) for each generation first."
        )

    # ── Task 2: cosine-to-vc and consecutive drift ────────────────────────────
    # cosine-to-vc: read from rc_eval results block (already computed during recovery)
    cos_to_vc = {}
    for g in gens_present:
        if rc[g] is not None:
            cos_to_vc[g] = rc[g].get("results", {}).get("cosine_similarity")
        else:
            cos_to_vc[g] = None

    # consecutive drift cos(v_r^n, v_r^(n-1))
    drift = {}
    for g in gens_present:
        if g == 1:
            continue
        if g - 1 in gens_present:
            drift[g] = cosine(vr[g], vr[g - 1])

    # ── Task 3: layer-window migration ────────────────────────────────────────
    lw = {}
    for g in gens_present:
        if rc[g] is not None:
            res = rc[g].get("results", {})
            num_cand = res.get("num_candidate_layers")
            num_act  = res.get("num_active_layers")
            lw[g] = {
                "layer_start":       res.get("layer_start"),
                "layer_end":         res.get("layer_end"),
                "num_active_layers": num_act,
                "num_candidate_layers": num_cand,
            }

    # Warning: window stays near-full-width
    full_width_threshold = 0.85
    for g, w in lw.items():
        if w["num_active_layers"] is not None and w["num_candidate_layers"] is not None:
            frac = w["num_active_layers"] / w["num_candidate_layers"]
            if frac >= full_width_threshold:
                print(
                    f"  WARNING gen {g}: active layers = {w['num_active_layers']}/"
                    f"{w['num_candidate_layers']} ({frac:.0%}) — "
                    "recovery not localizing, window near full-width."
                )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = {
        "model": args.model,
        "topic": args.topic,
        "seed":  args.seed,
        "gens_present": gens_present,
        "cosine_to_vc":    {str(g): v for g, v in cos_to_vc.items()},
        "consecutive_drift": {str(g): v for g, v in drift.items()},
        "layer_window":    {str(g): v for g, v in lw.items()},
    }
    json_path = os.path.join(analysis_dir, "drift_analysis.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✓ drift_analysis.json → {json_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not installed — skipping charts.")
        return

    # Task 2 plot: two series on one figure
    fig, ax = plt.subplots(figsize=(8, 5))
    g_list = gens_present
    vc_vals  = [cos_to_vc.get(g) for g in g_list]
    dr_gens  = [g for g in g_list if g in drift]
    dr_vals  = [drift[g] for g in dr_gens]

    ax.plot(g_list, vc_vals,  "o-",  color="tab:blue",  label="cos(v_r^n, v_c)")
    if dr_gens:
        ax.plot(dr_gens, dr_vals, "s--", color="tab:orange", label="cos(v_r^n, v_r^(n-1))")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(0, 1)
    ax.set_xticks(g_list)
    ax.legend(fontsize=9)
    ax.set_title(
        f"Vector drift across generations\n{model_name} · {args.topic} · seed {args.seed}"
    )
    fig.tight_layout()
    drift_png = os.path.join(analysis_dir, "drift_plot.png")
    fig.savefig(drift_png, dpi=150)
    plt.close(fig)
    print(f"✓ drift_plot.png → {drift_png}")

    # Task 3 plot: layer-window migration
    lw_gens = [g for g in g_list if g in lw and lw[g]["layer_start"] is not None]
    if lw_gens:
        starts = [lw[g]["layer_start"] for g in lw_gens]
        ends   = [lw[g]["layer_end"]   for g in lw_gens]
        nacts  = [lw[g]["num_active_layers"] for g in lw_gens]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax1.plot(lw_gens, starts, "o-", color="tab:green",  label="layer_start")
        ax1.plot(lw_gens, ends,   "s-", color="tab:red",    label="layer_end")
        ax1.fill_between(lw_gens, starts, ends, alpha=0.15, color="tab:green")
        ax1.set_ylabel("Layer index (normalised)")
        ax1.legend(fontsize=9)
        ax1.set_title(
            f"Recovery layer-window migration\n{model_name} · {args.topic} · seed {args.seed}"
        )

        ax2.bar(lw_gens, nacts, color="tab:purple", alpha=0.7)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("# active layers")
        ax2.set_xticks(lw_gens)

        fig.tight_layout()
        lw_png = os.path.join(analysis_dir, "layer_window.png")
        fig.savefig(lw_png, dpi=150)
        plt.close(fig)
        print(f"✓ layer_window.png → {lw_png}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  DRIFT ANALYSIS  ·  {model_name} · {args.topic} · seed {args.seed}")
    print("=" * 60)
    print(f"  {'gen':>3}  {'cos(v_r,v_c)':>14}  {'drift(n,n-1)':>14}  {'act_layers':>10}")
    for g in g_list:
        cvc = cos_to_vc.get(g)
        dr  = drift.get(g)
        na  = lw[g]["num_active_layers"] if g in lw else None
        def s(v, w, p=4):
            return ("—".rjust(w) if v is None else f"{v:.{p}f}".rjust(w))
        print(f"  {g:>3}  {s(cvc,14)}  {s(dr,14)}  {s(na,10,0)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
