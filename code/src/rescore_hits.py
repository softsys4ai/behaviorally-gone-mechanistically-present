"""
rescore_hits.py — Re-score saved eval generations with a word-boundary matcher.

eval_finetune.py counts a hit when the label is a *substring* of the response
("owl" in "knowledge" / "bowl" / "howl" counts). This script re-reads every
saved results/generations.jsonl and recomputes hit rates with:

  substring    — the original metric (reproduces ft_eval.json)
  word         — \b{label}s?\b anywhere in the response
  word_first5  — \b{label}s?\b within the first 5 words (paper's "pick rate")

No GPU, no model loading. Existing result files are not modified.

Reads:
  DATA_ROOT/<model>/<topic>/seed_<s>/results/generations.jsonl          (gen 1, holds base + ft)
  DATA_ROOT/<model>/<topic>/seed_<s>/gen_N/results/generations.jsonl    (gen N)
  .../results/ft_eval.json                                              (label)

Writes:
  DATA_ROOT/<model>/<topic>/seed_<s>/analysis/hit_rescore.json

Usage:
  python rescore_hits.py --model deepseek-ai/deepseek-llm-7b-chat --topic owl \
      --seed 42 --data-root /home/cc/experiments/subliminal/adam_lora
"""

import argparse
import json
import os
import re
from collections import Counter


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


def word_pattern(label):
    return re.compile(rf"\b{re.escape(label.lower())}s?\b", re.IGNORECASE)


def score_file(path, label):
    """Return {model_tag: counts} for one generations.jsonl."""
    wb = word_pattern(label)
    sub = label.lower()
    stats = {}
    false_pos = Counter()
    for line in open(path):
        r = json.loads(line)
        text = r["response"]
        s = stats.setdefault(r["model"], {"n": 0, "substring": 0, "word": 0, "word_first5": 0})
        s["n"] += 1
        is_sub = sub in text.lower()
        is_word = bool(wb.search(text))
        s["substring"]   += is_sub
        s["word"]        += is_word
        s["word_first5"] += bool(wb.search(" ".join(text.split()[:5])))
        if is_sub and not is_word:
            for tok in re.findall(rf"\w*{re.escape(sub)}\w*", text, re.IGNORECASE):
                false_pos[tok.lower()] += 1
    for s in stats.values():
        for k in ("substring", "word", "word_first5"):
            s[f"{k}_rate"] = s[k] / s["n"] if s["n"] else None
    return stats, false_pos


def main():
    args = parse_args()
    model_name = args.model.split("/")[-1]
    seed_dir   = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    analysis_dir = os.path.join(seed_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    with open(os.path.join(seed_dir, "results", "ft_eval.json")) as f:
        label = json.load(f)["label"]

    max_gen = args.num_generations or discover_max_gen(seed_dir)
    out = {"model": args.model, "topic": args.topic, "seed": args.seed,
           "label": label, "pattern": word_pattern(label).pattern,
           "base": None, "generations": {}, "false_positive_tokens": {}}
    all_fp = Counter()

    print(f"{'gen':>4} {'model':<6} {'n':>6} {'substring':>10} {'word':>8} {'first5':>8}")
    for g in range(1, max_gen + 1):
        path = os.path.join(gen_dir(seed_dir, g), "results", "generations.jsonl")
        if not os.path.exists(path):
            print(f"  ⚠ gen {g}: {path} missing — skipping")
            continue
        stats, fp = score_file(path, label)
        all_fp.update(fp)
        # Base model is evaluated in every gen's file; keep the gen-1 copy as the reference.
        if "base" in stats and out["base"] is None:
            out["base"] = stats["base"]
            b = stats["base"]
            print(f"{'base':>4} {'base':<6} {b['n']:>6} {b['substring_rate']:>10.4f} "
                  f"{b['word_rate']:>8.4f} {b['word_first5_rate']:>8.4f}")
        ft = {k: v for k, v in stats.items() if k != "base"}
        out["generations"][str(g)] = ft
        for tag, s in ft.items():
            print(f"{g:>4} {tag:<6} {s['n']:>6} {s['substring_rate']:>10.4f} "
                  f"{s['word_rate']:>8.4f} {s['word_first5_rate']:>8.4f}")

    out["false_positive_tokens"] = dict(all_fp.most_common(30))
    print(f"\nSubstring-only matches (false positives): {dict(all_fp.most_common(8))}")

    out_path = os.path.join(analysis_dir, "hit_rescore.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✓ Saved {out_path}")


if __name__ == "__main__":
    main()
