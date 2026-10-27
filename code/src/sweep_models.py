"""
sweep_models.py — Cross-model half-life comparison.

Reads summary_table data already produced by aggregate_seeds.py for each model,
normalises layer indices to fraction-of-depth so models with different layer
counts are directly comparable, then produces:

  - A half-life comparison bar chart  (one bar per model, error bar = std across seeds)
  - A cosine-to-vc decay overlay       (one curve per model, same topic)
  - A per-layer bias energy comparison (fraction-of-depth x-axis, gen 1 vs gen N)
  - A JSON manifest of all cross-model numbers

Reads:
  DATA_ROOT/<model>/analysis/summary_table.txt           (half-life, k, retention)
  DATA_ROOT/<model>/analysis/<topic>_multi_seed.json      (per-gen mean/std)
  DATA_ROOT/<model>/<topic>/seed_<s>/analysis/mechanism_probe.json  (per-layer energy)

Writes:
  DATA_ROOT/cross_model/halflife_bar.png
  DATA_ROOT/cross_model/<topic>_cosine_overlay.png
  DATA_ROOT/cross_model/<topic>_bias_energy_overlay.png
  DATA_ROOT/cross_model/cross_model_summary.json

Usage:
  python sweep_models.py \\
    --models Qwen/Qwen2.5-7B-Instruct deepseek-ai/deepseek-llm-7b-chat \\
             meta-llama/Llama-3.2-3B-Instruct microsoft/Phi-3-mini-4k-instruct \\
    --topics dragon owl wolf \\
    --seeds 42 43 44 \\
    --data-root /data/out
"""

import argparse
import json
import math
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models",  nargs="+", required=True,
                   help="Full HF model ids (same as used during runs)")
    p.add_argument("--topics",  nargs="+", required=True)
    p.add_argument("--seeds",   nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--data-root", required=True)
    p.add_argument("--run",       type=str, default=None,
                   help="Condition subfolder under data-root (e.g. adam_lora, sgd_lora, full_ft)")
    return p.parse_args()


def model_short(m):
    return m.split("/")[-1]


# ── loaders ───────────────────────────────────────────────────────────────────

def load_multi_seed_json(data_root, model, topic):
    path = os.path.join(data_root, model_short(model), "analysis",
                        f"{topic}_multi_seed.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_mechanism_probe(data_root, model, topic, seed):
    path = os.path.join(data_root, model_short(model), topic,
                        f"seed_{seed}", "analysis", "mechanism_probe.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def layer_fraction_energy(probe_data, gen, num_layers):
    """
    Return list of (fraction_of_depth, bias_energy) for a given generation's
    per-layer energy from mechanism_probe.json.
    """
    if probe_data is None:
        return None
    for r in probe_data.get("generations", []):
        if r["generation"] == gen:
            be = r.get("bias_energy", {})
            if not be:
                return None
            return [(int(l) / (num_layers - 1), float(v)) for l, v in sorted(
                be.items(), key=lambda x: int(x[0])
            )]
    return None


def fit_halflife(gens, mean_hrs, base_hr):
    if base_hr is None:
        return None, None
    pairs = [(g, hr - base_hr) for g, hr in zip(gens, mean_hrs)
             if hr is not None and hr - base_hr > 0]
    if len(pairs) < 2:
        return None, None
    xs = [g - 1 for g, _ in pairs]
    ys = [math.log(ex) for _, ex in pairs]
    n  = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x*x for x in xs)
    sxy = sum(x*y for x, y in zip(xs, ys))
    denom = n*sxx - sx*sx
    if denom == 0:
        return None, None
    k_neg = (n*sxy - sx*sy) / denom
    k = -k_neg
    hl = math.log(2) / k if k > 0 else None
    return k, hl


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.run and not args.data_root.endswith(args.run):
        args.data_root = os.path.join(args.data_root, args.run)
    out_dir = os.path.join(args.data_root, "cross_model")
    os.makedirs(out_dir, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_PLT = True
    except ImportError:
        HAS_PLT = False
        print("⚠ matplotlib not installed — skipping charts.")

    # ── Collect cross-model data ──────────────────────────────────────────────
    model_records = {}   # model_short → {topic → multi_seed_json}

    for m in args.models:
        ms = model_short(m)
        model_records[ms] = {}
        for topic in args.topics:
            d = load_multi_seed_json(args.data_root, m, topic)
            if d is not None:
                model_records[ms][topic] = d
            else:
                print(f"  ⚠ {ms}/{topic}: multi-seed JSON not found, skipping.")

    summary = []   # for JSON output

    # ── Half-life bar chart ───────────────────────────────────────────────────
    hl_by_model = {}   # ms → list of half-lives across topics
    for ms, topics_data in model_records.items():
        hls = []
        for topic, d in topics_data.items():
            gens_str = sorted(d["generations"].keys(), key=int)
            gens     = [int(g) for g in gens_str]
            mean_hrs = [d["generations"][g]["hit_rate_mean"] for g in gens_str]
            base_hr  = d.get("base_model", {}).get("hit_rate_mean")
            k, hl    = fit_halflife(gens, mean_hrs, base_hr)
            entry = {
                "model": ms, "topic": topic,
                "k": k, "half_life": hl,
                "base_hr": base_hr,
                "gen1_hr": mean_hrs[0] if mean_hrs else None,
                "genN_hr": mean_hrs[-1] if mean_hrs else None,
                "n_seeds": len(d.get("seeds_found", [])),
            }
            summary.append(entry)
            if hl is not None:
                hls.append(hl)
        hl_by_model[ms] = hls

    # Save JSON
    cross_json = os.path.join(out_dir, "cross_model_summary.json")
    with open(cross_json, "w") as f:
        json.dump({"models": args.models, "topics": args.topics,
                   "seeds": args.seeds, "results": summary}, f, indent=2)
    print(f"✓ cross_model_summary.json → {cross_json}")

    if HAS_PLT:
        # Half-life bar chart
        ms_list = [m for m in hl_by_model if hl_by_model[m]]
        if ms_list:
            import numpy as np
            means = [np.mean(hl_by_model[ms]) for ms in ms_list]
            stds  = [np.std(hl_by_model[ms]) if len(hl_by_model[ms]) > 1 else 0
                     for ms in ms_list]
            fig, ax = plt.subplots(figsize=(9, 5))
            colors = plt.cm.tab10(np.linspace(0, 0.9, len(ms_list)))
            bars = ax.bar(range(len(ms_list)), means, yerr=stds,
                          color=colors, alpha=0.85, capsize=6)
            ax.set_xticks(range(len(ms_list)))
            ax.set_xticklabels(ms_list, rotation=15, ha="right", fontsize=9)
            ax.set_ylabel("Half-life (generations)")
            ax.set_title("Subliminal bias half-life by model\n(mean ± std across topics)")
            fig.tight_layout()
            hl_path = os.path.join(out_dir, "halflife_bar.png")
            fig.savefig(hl_path, dpi=150)
            plt.close(fig)
            print(f"✓ halflife_bar.png → {hl_path}")

        # Per-topic cosine overlay
        colors_map = {ms: plt.cm.tab10(i / max(len(args.models) - 1, 1))
                      for i, ms in enumerate([model_short(m) for m in args.models])}
        for topic in args.topics:
            fig, ax = plt.subplots(figsize=(8, 5))
            any_data = False
            for m in args.models:
                ms = model_short(m)
                d  = model_records.get(ms, {}).get(topic)
                if d is None:
                    continue
                gens_str = sorted(d["generations"].keys(), key=int)
                gens     = [int(g) for g in gens_str]
                means    = [d["generations"][g]["cos_to_vc_mean"] for g in gens_str]
                stds     = [d["generations"][g]["cos_to_vc_std"] or 0 for g in gens_str]
                c        = colors_map[ms]
                m_arr    = [v if v is not None else float("nan") for v in means]
                ax.plot(gens, m_arr, "o-", color=c, label=ms)
                ax.fill_between(
                    gens,
                    [a - b for a, b in zip(m_arr, stds)],
                    [a + b for a, b in zip(m_arr, stds)],
                    alpha=0.15, color=c
                )
                any_data = True
            if any_data:
                ax.set_xlabel("Generation")
                ax.set_ylabel("cos(v_r^n, v_c)")
                ax.set_ylim(0, 1)
                ax.legend(fontsize=8)
                ax.set_title(f"Cosine to v_c across models — {topic}")
                fig.tight_layout()
                p = os.path.join(out_dir, f"{topic}_cosine_overlay.png")
                fig.savefig(p, dpi=150)
                print(f"✓ {topic}_cosine_overlay.png → {p}")
            plt.close(fig)

        # Per-topic bias energy overlay (gen 1 vs gen N, fraction-of-depth x-axis)
        for topic in args.topics:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
            titles = ["Gen 1 (initial transfer)", "Gen N (last generation)"]
            target_gens = [1, None]   # None = last gen

            for ax, title, tg in zip(axes, titles, target_gens):
                any_data = False
                for m in args.models:
                    ms = model_short(m)
                    for seed in args.seeds:
                        probe = load_mechanism_probe(args.data_root, m, topic, seed)
                        if probe is None:
                            continue
                        nl = probe.get("num_layers", 28)
                        gen_to_use = tg
                        if gen_to_use is None:
                            gen_to_use = max(r["generation"] for r in probe.get("generations", []))
                        pts = layer_fraction_energy(probe, gen_to_use, nl)
                        if pts is None:
                            continue
                        fracs, energies = zip(*pts)
                        c = colors_map[ms]
                        label = ms if seed == args.seeds[0] else None
                        ax.plot(fracs, energies, alpha=0.7, color=c, lw=1.2, label=label)
                        any_data = True
                if any_data:
                    ax.set_xlabel("Layer depth (fraction)")
                    ax.set_ylabel("|h·v̂_c| / ||h||")
                    ax.set_title(title)
                    ax.legend(fontsize=7)

            plt.suptitle(f"Bias energy profile across models — {topic}", fontsize=10)
            fig.tight_layout()
            p = os.path.join(out_dir, f"{topic}_bias_energy_overlay.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            print(f"✓ {topic}_bias_energy_overlay.png → {p}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  CROSS-MODEL SUMMARY")
    print("=" * 72)
    print(f"  {'model':30}  {'topic':10}  {'base_hr':>8}  {'gen1_hr':>8}  {'half-life':>10}")
    print("  " + "-" * 68)
    for r in summary:
        def fv(v):
            return f"{v:.3f}" if v is not None else "   —"
        print(
            f"  {r['model']:30}  {r['topic']:10}  {fv(r['base_hr']):>8}  "
            f"{fv(r['gen1_hr']):>8}  {fv(r['half_life']):>10}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
