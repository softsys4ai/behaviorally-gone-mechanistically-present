"""
plot_decay.py — Plot multi-generation bias decay for the iterated pipeline.

Scans a (model, topic, seed) run, collects per-generation metrics from
ft_eval.json (behavioural transfer) and rc_eval.json (vector persistence),
and produces a chart of bias decay vs. generation.

Generation 1 lives at the seed root; generations >= 2 live under gen_{N}/.

Reads:
  DATA_ROOT/{model}/{topic}/seed_{seed}/results/ft_eval.json          (gen 1)
  DATA_ROOT/{model}/{topic}/seed_{seed}/results/rc_eval.json          (gen 1)
  DATA_ROOT/{model}/{topic}/seed_{seed}/gen_{N}/results/ft_eval.json  (gen N>=2)
  DATA_ROOT/{model}/{topic}/seed_{seed}/gen_{N}/results/rc_eval.json  (gen N>=2)

Writes:
  DATA_ROOT/{model}/{topic}/seed_{seed}/results/decay_curve.png
  DATA_ROOT/{model}/{topic}/seed_{seed}/results/decay_curve.json

Usage:
  python plot_decay.py --model Qwen/Qwen2.5-7B-Instruct --topic dragon \
      --seed 42 --data-root /data/out
"""

import argparse
import json
import os


def parse_args():
    p = argparse.ArgumentParser(description="Plot multi-generation bias decay")
    p.add_argument("--model",            type=str, required=True)
    p.add_argument("--topic",            type=str, required=True)
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--data-root",        type=str, required=True)
    p.add_argument("--num-generations",  type=int, default=None,
                   help="Max generation to look for. Default: auto-detect by "
                        "scanning gen_* directories.")
    return p.parse_args()


def safe_load(path):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def gen_dir(seed_dir, gen):
    """Gen 1 is the flat seed layout; gens >= 2 live under gen_{N}/."""
    return seed_dir if gen <= 1 else os.path.join(seed_dir, f"gen_{gen}")


def discover_max_gen(seed_dir):
    """Highest gen_{N} directory present (>=1 always, since gen 1 is the root)."""
    max_gen = 1
    if os.path.isdir(seed_dir):
        for name in os.listdir(seed_dir):
            if name.startswith("gen_") and os.path.isdir(os.path.join(seed_dir, name)):
                try:
                    max_gen = max(max_gen, int(name[len("gen_"):]))
                except ValueError:
                    continue
    return max_gen


def main():
    args = parse_args()
    model_name = args.model.split("/")[-1]
    seed_dir   = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    results_dir = os.path.join(seed_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    max_gen = args.num_generations or discover_max_gen(seed_dir)

    # ── Collect per-generation metrics ────────────────────────────────────
    gens, hit_rates, log_liks, cos_sims, alphas = [], [], [], [], []
    base_hit_rate = base_log_lik = None
    records = []

    for g in range(1, max_gen + 1):
        d = gen_dir(seed_dir, g)
        ft = safe_load(os.path.join(d, "results", "ft_eval.json"))
        rc = safe_load(os.path.join(d, "results", "rc_eval.json"))
        if ft is None and rc is None:
            continue

        hr = ll = cos = alpha = None
        if ft is not None:
            hr = ft.get("finetuned_model", {}).get("hit_rate")
            ll = ft.get("finetuned_model", {}).get("avg_log_likelihood")
            # Base model is generation-independent; capture it once for reference.
            if base_hit_rate is None:
                base_hit_rate = ft.get("base_model", {}).get("hit_rate")
                base_log_lik  = ft.get("base_model", {}).get("avg_log_likelihood")
        if rc is not None:
            cos   = rc.get("results", {}).get("cosine_similarity")
            alpha = rc.get("results", {}).get("learned_alpha")

        gens.append(g)
        hit_rates.append(hr)
        log_liks.append(ll)
        cos_sims.append(cos)
        alphas.append(alpha)
        records.append({
            "generation": g, "hit_rate": hr, "avg_log_likelihood": ll,
            "cosine_similarity": cos, "learned_alpha": alpha,
        })

    if not gens:
        raise SystemExit(
            f"No ft_eval.json / rc_eval.json found under {seed_dir}. "
            "Has the run produced any results yet?"
        )

    # ── Plot: hit-rate (left axis) + cosine-to-v_c (right axis) ───────────
    # matplotlib is optional: if it's missing we still write the JSON + table.
    png_path = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        print("⚠ matplotlib not installed — skipping chart (numbers still saved). "
              "Install with: pip install matplotlib")

    if plt is not None:
        fig, ax_hr = plt.subplots(figsize=(8, 5))

        color_hr = "tab:blue"
        ax_hr.set_xlabel("Generation")
        ax_hr.set_ylabel("Bias hit rate (finetuned)", color=color_hr)
        ax_hr.plot(gens, hit_rates, "o-", color=color_hr, label="hit rate")
        ax_hr.tick_params(axis="y", labelcolor=color_hr)
        ax_hr.set_xticks(gens)
        ax_hr.set_ylim(0, 1)
        if base_hit_rate is not None:
            ax_hr.axhline(base_hit_rate, color=color_hr, ls=":", lw=1, alpha=0.6,
                          label=f"base hit rate ({base_hit_rate:.2f})")

        color_cos = "tab:red"
        ax_cos = ax_hr.twinx()
        ax_cos.set_ylabel("Cosine similarity to original $v_c$", color=color_cos)
        ax_cos.plot(gens, cos_sims, "s--", color=color_cos, label="cosine to $v_c$")
        ax_cos.tick_params(axis="y", labelcolor=color_cos)
        ax_cos.set_ylim(0, 1)

        lines1, labels1 = ax_hr.get_legend_handles_labels()
        lines2, labels2 = ax_cos.get_legend_handles_labels()
        ax_hr.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

        plt.title(f"Subliminal bias decay across generations\n{model_name} · {args.topic} · seed {args.seed}")
        fig.tight_layout()

        png_path = os.path.join(results_dir, "decay_curve.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)

    # ── Save the underlying numbers ───────────────────────────────────────
    json_path = os.path.join(results_dir, "decay_curve.json")
    with open(json_path, "w") as f:
        json.dump({
            "model": args.model,
            "topic": args.topic,
            "seed": args.seed,
            "base_model": {"hit_rate": base_hit_rate, "avg_log_likelihood": base_log_lik},
            "generations": records,
        }, f, indent=2)

    # ── Console summary ───────────────────────────────────────────────────
    print("=" * 62)
    print(f"  BIAS DECAY  ·  {model_name} · {args.topic} · seed {args.seed}")
    print("=" * 62)
    if base_hit_rate is not None:
        print(f"  base hit rate: {base_hit_rate:.4f}   base log-lik: {base_log_lik}")
    print(f"  {'gen':>3}  {'hit_rate':>10}  {'avg_log_lik':>12}  {'cos_to_vc':>10}  {'alpha':>8}")
    for r in records:
        def s(v, w, p=4):
            return ("—".rjust(w) if v is None else f"{v:.{p}f}".rjust(w))
        print(f"  {r['generation']:>3}  {s(r['hit_rate'],10)}  "
              f"{s(r['avg_log_likelihood'],12)}  {s(r['cosine_similarity'],10)}  "
              f"{s(r['learned_alpha'],8)}")
    print("=" * 62)
    if png_path:
        print(f"  chart → {png_path}")
    print(f"  data  → {json_path}")


if __name__ == "__main__":
    main()
