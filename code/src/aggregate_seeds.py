"""
aggregate_seeds.py — Multi-seed aggregation and decay fit (Tasks 4 & 5).

Task 4: mean±std across seeds for hit_rate, cosine-to-vc, consecutive drift
        per generation.  One plot per topic + combined 3-topic overlay.
Task 5: exponential decay fit  excess_hit_rate(g) = A * exp(-k*(g-1)),
        report k and half-life per (topic, seed); then mean±std across seeds.
        Summary table: topic, base_hr, gen1_hr, gen5_hr, retention%, half-life±std.

Reads:
  DATA_ROOT/<model>/<topic>/seed_<s>/results/ft_eval.json          (gen 1)
  DATA_ROOT/<model>/<topic>/seed_<s>/gen_N/results/ft_eval.json    (gen N>=2)
  DATA_ROOT/<model>/<topic>/seed_<s>/analysis/drift_analysis.json  (Task 2 output)

Writes:
  DATA_ROOT/<model>/analysis/<topic>_multi_seed.json
  DATA_ROOT/<model>/analysis/<topic>_multi_seed.png
  DATA_ROOT/<model>/analysis/combined_overlay.png
  DATA_ROOT/<model>/analysis/summary_table.txt

Usage:
  python aggregate_seeds.py --model Qwen/Qwen2.5-7B-Instruct \
      --topics dragon owl wolf --seeds 42 43 44 \
      --data-root /data/out
"""

import argparse
import json
import math
import os
import statistics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           required=True)
    p.add_argument("--topics",          nargs="+", required=True)
    p.add_argument("--seeds",           nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--data-root",       required=True)
    p.add_argument("--num-generations", type=int, default=None)
    p.add_argument("--run",             type=str, default=None,
                   help="Condition subfolder under data-root (e.g. adam_lora, sgd_lora, full_ft)")
    return p.parse_args()


# ── helpers ──────────────────────────────────────────────────────────────────

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


def load_ft_eval(seed_dir, g):
    path = os.path.join(gen_dir(seed_dir, g), "results", "ft_eval.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_drift(seed_dir):
    path = os.path.join(seed_dir, "analysis", "drift_analysis.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], None
    return statistics.mean(vals), statistics.stdev(vals)


def fit_exponential(gens, hit_rates, base_hr):
    """
    Fit  excess(g) = A * exp(-k*(g-1))  by linear regression on log-scale.
    Returns (A, k, half_life).  Returns (None, None, None) if < 2 points.
    """
    pairs = [(g, hr - base_hr) for g, hr in zip(gens, hit_rates)
             if hr is not None and hr - base_hr > 0]
    if len(pairs) < 2:
        return None, None, None
    # log(excess) = log(A) - k*(g-1)
    xs = [g - 1 for g, _ in pairs]
    ys = [math.log(ex) for _, ex in pairs]
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return None, None, None
    k_neg = (n * sxy - sx * sy) / denom   # slope (should be negative)
    log_a = (sy - k_neg * sx) / n
    A = math.exp(log_a)
    k = -k_neg
    half_life = math.log(2) / k if k > 0 else None
    return A, k, half_life


def fmt(v, p=4):
    return f"{v:.{p}f}" if v is not None else "—"


# ── main ──────────────────────────────────────────────────────────────────────

def collect_topic_data(data_root, model_name, topic, seeds, max_gen_override):
    """
    Returns dict keyed by seed, each containing per-gen hit_rate, cos_to_vc,
    consecutive_drift, base_hr.  Missing seeds / gens produce None values.
    """
    results = {}
    for seed in seeds:
        seed_dir = os.path.join(data_root, model_name, topic, f"seed_{seed}")
        if not os.path.isdir(seed_dir):
            print(f"  seed {seed}: directory not found, skipping.")
            continue

        max_gen = max_gen_override or discover_max_gen(seed_dir)
        drift_data = load_drift(seed_dir)

        per_gen = {}
        base_hr = None
        for g in range(1, max_gen + 1):
            ft = load_ft_eval(seed_dir, g)
            hr = None
            if ft is not None:
                hr = ft.get("finetuned_model", {}).get("hit_rate")
                if g == 1 and base_hr is None:
                    base_hr = ft.get("base_model", {}).get("hit_rate")

            cos_vc = None
            if drift_data:
                cos_vc = drift_data.get("cosine_to_vc", {}).get(str(g))

            drift_val = None
            if drift_data and g >= 2:
                drift_val = drift_data.get("consecutive_drift", {}).get(str(g))

            per_gen[g] = {"hit_rate": hr, "cos_to_vc": cos_vc, "drift": drift_val}

        results[seed] = {"base_hr": base_hr, "per_gen": per_gen, "max_gen": max_gen}
    return results


def aggregate(topic_data):
    """
    Given {seed: {base_hr, per_gen, max_gen}}, compute per-generation mean±std
    across seeds for hit_rate, cos_to_vc, drift.
    """
    all_gens = sorted({g for sd in topic_data.values() for g in sd["per_gen"]})
    agg = {}
    for g in all_gens:
        hrs  = [sd["per_gen"].get(g, {}).get("hit_rate")  for sd in topic_data.values()]
        cvcs = [sd["per_gen"].get(g, {}).get("cos_to_vc") for sd in topic_data.values()]
        drs  = [sd["per_gen"].get(g, {}).get("drift")     for sd in topic_data.values()]
        agg[g] = {
            "hit_rate":  mean_std(hrs),
            "cos_to_vc": mean_std(cvcs),
            "drift":     mean_std(drs),
        }
    base_hrs = [sd["base_hr"] for sd in topic_data.values() if sd["base_hr"] is not None]
    base_mean, base_std = mean_std(base_hrs)
    return all_gens, agg, base_mean, base_std


def main():
    args = parse_args()
    if args.run and not args.data_root.endswith(args.run):
        args.data_root = os.path.join(args.data_root, args.run)
    model_name = args.model.split("/")[-1]
    analysis_out = os.path.join(args.data_root, model_name, "analysis")
    os.makedirs(analysis_out, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_PLT = True
    except ImportError:
        HAS_PLT = False
        print("⚠ matplotlib not installed — skipping charts.")

    summary_rows = []    # for Task 5 summary table
    combined = {}        # topic → (gens, mean_hrs, std_hrs) for overlay

    for topic in args.topics:
        print(f"\n── Topic: {topic} " + "─" * 40)
        topic_data = collect_topic_data(
            args.data_root, model_name, topic, args.seeds, args.num_generations
        )
        if not topic_data:
            print(f"  No data for topic {topic}, skipping.")
            continue

        all_gens, agg, base_mean, base_std = aggregate(topic_data)

        # ── Save per-topic JSON ───────────────────────────────────────────────
        out_json = {
            "model": args.model,
            "topic": topic,
            "seeds_found": list(topic_data.keys()),
            "base_model": {"hit_rate_mean": base_mean, "hit_rate_std": base_std},
            "generations": {
                str(g): {
                    "hit_rate_mean":  agg[g]["hit_rate"][0],
                    "hit_rate_std":   agg[g]["hit_rate"][1],
                    "cos_to_vc_mean": agg[g]["cos_to_vc"][0],
                    "cos_to_vc_std":  agg[g]["cos_to_vc"][1],
                    "drift_mean":     agg[g]["drift"][0],
                    "drift_std":      agg[g]["drift"][1],
                }
                for g in all_gens
            },
        }
        json_path = os.path.join(analysis_out, f"{topic}_multi_seed.json")
        with open(json_path, "w") as f:
            json.dump(out_json, f, indent=2)
        print(f"  ✓ {topic}_multi_seed.json")

        # ── Task 4 plot: per-topic error bands ───────────────────────────────
        if HAS_PLT:
            mean_hrs  = [agg[g]["hit_rate"][0]  for g in all_gens]
            std_hrs   = [agg[g]["hit_rate"][1] or 0 for g in all_gens]
            mean_cvcs = [agg[g]["cos_to_vc"][0] for g in all_gens]
            std_cvcs  = [agg[g]["cos_to_vc"][1] or 0 for g in all_gens]
            mean_drs  = [agg[g]["drift"][0]  for g in [g for g in all_gens if g >= 2]]
            std_drs   = [agg[g]["drift"][1] or 0 for g in [g for g in all_gens if g >= 2]]
            dr_gens   = [g for g in all_gens if g >= 2]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            ax_hr, ax_cos = axes

            # hit-rate with error band
            mh = [v if v is not None else float("nan") for v in mean_hrs]
            sh = [v for v in std_hrs]
            mh_arr = mh
            ax_hr.plot(all_gens, mh_arr, "o-", color="tab:blue")
            ax_hr.fill_between(
                all_gens,
                [m - s for m, s in zip(mh_arr, sh)],
                [m + s for m, s in zip(mh_arr, sh)],
                alpha=0.2, color="tab:blue"
            )
            if base_mean is not None:
                ax_hr.axhline(base_mean, ls=":", color="tab:blue", lw=1, alpha=0.6,
                              label=f"base ({base_mean:.3f})")
            ax_hr.set_xlabel("Generation")
            ax_hr.set_ylabel("Bias hit rate")
            ax_hr.set_ylim(0, 1)
            ax_hr.set_xticks(all_gens)
            ax_hr.set_title(f"{topic} — hit rate (n={len(topic_data)} seeds)")
            ax_hr.legend(fontsize=8)

            # cosine + drift
            mc = [v if v is not None else float("nan") for v in mean_cvcs]
            sc = [v for v in std_cvcs]
            ax_cos.plot(all_gens, mc, "o-", color="tab:red", label="cos(v_r,v_c)")
            ax_cos.fill_between(
                all_gens,
                [m - s for m, s in zip(mc, sc)],
                [m + s for m, s in zip(mc, sc)],
                alpha=0.2, color="tab:red"
            )
            if dr_gens:
                md = [v if v is not None else float("nan") for v in mean_drs]
                sd_ = [v for v in std_drs]
                ax_cos.plot(dr_gens, md, "s--", color="tab:orange", label="drift(n,n-1)")
                ax_cos.fill_between(
                    dr_gens,
                    [m - s for m, s in zip(md, sd_)],
                    [m + s for m, s in zip(md, sd_)],
                    alpha=0.2, color="tab:orange"
                )
            ax_cos.set_xlabel("Generation")
            ax_cos.set_ylabel("Cosine similarity")
            ax_cos.set_ylim(0, 1)
            ax_cos.set_xticks(all_gens)
            ax_cos.set_title(f"{topic} — vector similarity")
            ax_cos.legend(fontsize=8)

            plt.suptitle(
                f"{model_name} · {topic} · seeds {sorted(topic_data.keys())}",
                fontsize=10
            )
            fig.tight_layout()
            png_path = os.path.join(analysis_out, f"{topic}_multi_seed.png")
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            print(f"  ✓ {topic}_multi_seed.png")

            combined[topic] = (all_gens, mh_arr, sh)

        # ── Task 5: decay fit per seed ────────────────────────────────────────
        halflives = []
        ks = []
        gen1_hrs = []
        gen_last_hrs = []

        for seed, sd in sorted(topic_data.items()):
            if sd["base_hr"] is None:
                continue
            gens_s = sorted(sd["per_gen"].keys())
            hrs_s  = [sd["per_gen"][g]["hit_rate"] for g in gens_s]
            _, k, hl = fit_exponential(gens_s, hrs_s, sd["base_hr"])
            if k is not None:
                ks.append(k)
                halflives.append(hl)
            g1_hr = sd["per_gen"].get(1, {}).get("hit_rate")
            gN_hr = sd["per_gen"].get(max(gens_s), {}).get("hit_rate")
            if g1_hr is not None:
                gen1_hrs.append(g1_hr)
            if gN_hr is not None:
                gen_last_hrs.append(gN_hr)

        bm, _ = mean_std([sd["base_hr"] for sd in topic_data.values()])
        g1m, g1s = mean_std(gen1_hrs)
        gNm, gNs = mean_std(gen_last_hrs)
        hlm, hls = mean_std(halflives)
        km,  ks_ = mean_std(ks)

        # retention = (gen1_hr - base_hr) retained at last gen
        if g1m is not None and bm is not None and gNm is not None:
            excess_g1 = g1m - bm
            excess_gN = gNm - bm
            retention_pct = (excess_gN / excess_g1 * 100) if excess_g1 > 0 else None
        else:
            retention_pct = None

        max_gen_val = max(
            (max(sd["per_gen"]) for sd in topic_data.values() if sd["per_gen"]),
            default=1,
        )
        summary_rows.append({
            "topic":         topic,
            "n_seeds":       len(topic_data),
            "base_hr":       bm,
            "gen1_hr_mean":  g1m,
            "gen1_hr_std":   g1s,
            "genN":          max_gen_val,
            "genN_hr_mean":  gNm,
            "genN_hr_std":   gNs,
            "retention_pct": retention_pct,
            "half_life_mean": hlm,
            "half_life_std":  hls,
            "k_mean":         km,
            "k_std":          ks_,
        })

    # ── Combined overlay (Task 4) ────────────────────────────────────────────
    if HAS_PLT and combined:
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple"]
        for i, (topic, (gens, means, stds)) in enumerate(combined.items()):
            c = colors[i % len(colors)]
            m = [v if v is not None else float("nan") for v in means]
            s = [v for v in stds]
            ax.plot(gens, m, "o-", color=c, label=topic)
            ax.fill_between(
                gens,
                [a - b for a, b in zip(m, s)],
                [a + b for a, b in zip(m, s)],
                alpha=0.15, color=c
            )
        ax.set_xlabel("Generation")
        ax.set_ylabel("Bias hit rate (mean ± std)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.set_title(f"Bias decay across topics — {model_name} · seeds {args.seeds}")
        fig.tight_layout()
        overlay_path = os.path.join(analysis_out, "combined_overlay.png")
        fig.savefig(overlay_path, dpi=150)
        plt.close(fig)
        print(f"\n✓ combined_overlay.png → {overlay_path}")

    # ── Summary table (Task 5) ───────────────────────────────────────────────
    lines = []
    lines.append("=" * 90)
    lines.append(f"  DECAY SUMMARY — {model_name} · seeds {args.seeds}")
    lines.append("=" * 90)
    hdr = (
        f"  {'topic':12}  {'seeds':>5}  {'base_hr':>8}  {'gen1_hr':>9}  "
        f"{'gen_last':>4}  {'genN_hr':>9}  {'retain%':>8}  {'half-life':>10}  {'k':>8}"
    )
    lines.append(hdr)
    lines.append("  " + "-" * 86)
    for r in summary_rows:
        def fv(v, p=3):
            return f"{v:.{p}f}" if v is not None else "   —"
        def fs(v, s, p=3):
            if v is None:
                return "      —    "
            if s is not None:
                return f"{v:.{p}f}±{s:.{p}f}"
            return f"{v:.{p}f}       "
        lines.append(
            f"  {r['topic']:12}  {r['n_seeds']:>5}  {fv(r['base_hr']):>8}  "
            f"{fs(r['gen1_hr_mean'], r['gen1_hr_std']):>17}  "
            f"{r['genN']:>4}  "
            f"{fs(r['genN_hr_mean'], r['genN_hr_std']):>17}  "
            f"{fv(r['retention_pct'], 1):>8}  "
            f"{fs(r['half_life_mean'], r['half_life_std']):>17}  "
            f"{fs(r['k_mean'], r['k_std']):>16}"
        )
    lines.append("=" * 90)
    table_str = "\n".join(lines)
    print("\n" + table_str)

    tbl_path = os.path.join(analysis_out, "summary_table.txt")
    with open(tbl_path, "w") as f:
        f.write(table_str + "\n")
    print(f"\n✓ summary_table.txt → {tbl_path}")


if __name__ == "__main__":
    main()
