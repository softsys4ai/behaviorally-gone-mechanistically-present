"""
mechanism_probe.py — Mechanistic analysis of subliminal bias across generations.

Three complementary probes:

  1. PROJECTION — run neutral prompts through the Gen-n student, project hidden
     states at every layer onto the (normalized) original v_c.  Produces a
     (layer × generation) heatmap of mean projection magnitude, showing where
     the bias signal lives and how it weakens generation by generation.

  2. PER-LAYER BIAS ENERGY — ||component of activation along v_c|| / ||activation||
     (fractional energy in the bias direction), independent of overall activation scale.

  3. CAUSAL ABLATION — subtract the v_c component from every layer's activations
     during a forward pass, then re-measure the topic hit-rate on eval prompts.
     If ablation collapses hit-rate → v_c direction is causally load-bearing.
     If it doesn't → bias has migrated off the original direction.

Reads:
  DATA_ROOT/<model>/<topic>/seed_<s>/Steering_Vector/steering_vector.pkl  (v_c)
  DATA_ROOT/<model>/<topic>/seed_<s>/<gen_N>/Recover_Vector/vr_gen<N>.pt  (v_r^n)
  DATA_ROOT/<model>/<topic>/seed_<s>/results/ft_eval.json                 (base hit-rate ref)
  prompts-json (same file used by eval_finetune.py)

Writes (idempotent — skip if already present):
  DATA_ROOT/<model>/<topic>/seed_<s>/analysis/mechanism_probe.json
  DATA_ROOT/<model>/<topic>/seed_<s>/analysis/projection_heatmap.png
  DATA_ROOT/<model>/<topic>/seed_<s>/analysis/bias_energy.png
  DATA_ROOT/<model>/<topic>/seed_<s>/analysis/ablation_bar.png

Usage:
  python mechanism_probe.py \\
    --model Qwen/Qwen2.5-7B-Instruct \\
    --topic dragon --seed 42 --data-root /data/out \\
    --prompts-json code/input/animal_biases/dragon.json \\
    [--num-generations 5] [--num-probe-samples 200] [--ablation-samples 100]
"""

import argparse
import json
import os
import pickle
import gc

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_utils import load_student


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",              required=True)
    p.add_argument("--topic",              required=True)
    p.add_argument("--seed",               type=int, default=42)
    p.add_argument("--data-root",          required=True)
    p.add_argument("--prompts-json",       required=True)
    p.add_argument("--num-generations",    type=int, default=None)
    p.add_argument("--num-probe-samples",  type=int, default=200,
                   help="Number of neutral prompts for projection/energy probes.")
    p.add_argument("--ablation-samples",   type=int, default=100,
                   help="Number of eval prompts for hit-rate ablation.")
    p.add_argument("--batch-size",         type=int, default=8)
    p.add_argument("--max-new-tokens",     type=int, default=60)
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

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


def load_vector(seed_dir, g):
    """Load vr_gen{g}.pt if present; else None."""
    path = os.path.join(gen_dir(seed_dir, g), "Recover_Vector", f"vr_gen{g}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=True).float()
    return None


def load_v_c(seed_dir):
    path = os.path.join(seed_dir, "Steering_Vector", "steering_vector.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Steering vector not found: {path}")
    with open(path, "rb") as f:
        d = pickle.load(f)
    if "steering_vector" in d:
        sv = d["steering_vector"]
    elif "steering_vectors" in d:
        sv = list(d["steering_vectors"].values())[0]
    else:
        raise KeyError("Neither 'steering_vector' nor 'steering_vectors' key found in pkl")
    if isinstance(sv, np.ndarray):
        sv = torch.from_numpy(sv)
    return sv.float()


def load_prompts(prompts_json, n):
    with open(prompts_json) as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "training_pairs" in data and isinstance(data["training_pairs"], list):
            prompts = [tp["prompt"] if isinstance(tp, dict) and "prompt" in tp else str(tp) for tp in data["training_pairs"]]
        elif "prompts" in data and isinstance(data["prompts"], list):
            prompts = [p["prompt"] if isinstance(p, dict) and "prompt" in p else str(p) for p in data["prompts"]]
        else:
            prompts = [v for v in data.values() if isinstance(v, str)]
    elif isinstance(data, list):
        prompts = [p["prompt"] if isinstance(p, dict) and "prompt" in p else str(p) for p in data]
    else:
        prompts = [str(data)]
    prompts = [p for p in prompts if isinstance(p, str)]
    return prompts[:n]


def load_base_hit_rate(seed_dir):
    path = os.path.join(seed_dir, "results", "ft_eval.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        return d.get("base_model", {}).get("hit_rate")
    return None


def adapter_id_for_gen(seed_dir, g):
    """Try to read the HF repo from ft_eval.json so we don't hardcode naming."""
    path = os.path.join(gen_dir(seed_dir, g), "results", "ft_eval.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        return d.get("finetuned_model", {}).get("hf_repo") or d.get("hf_repo")
    return None


# ── neutral prompt builder (reuses the random-number theme) ──────────────────

def build_neutral_prompts(n, seed, tokenizer, model_name):
    """
    Build n simple random-number prompts (no system-prompt, no steering).
    These are deliberately bland so any projection onto v_c is purely due to
    inherited weight structure, not explicit topic mention.
    """
    rng = np.random.default_rng(seed + 9999)
    prompts = []
    for _ in range(n):
        k = rng.integers(3, 6)
        nums = [str(rng.integers(100, 999)) for _ in range(k)]
        prompts.append(
            f"Continue the number sequence: {', '.join(nums)}. "
            f"Generate 10 more 3-digit numbers, comma-separated."
        )
    return prompts


# ── activation capture ────────────────────────────────────────────────────────

class ActivationCapture:
    """
    Registers forward hooks on model.model.layers[*] and accumulates
    the mean per-token hidden-state tensor (last token, float32).
    """
    def __init__(self, model, layer_indices):
        self.layer_indices = layer_indices
        self.activations   = {i: [] for i in layer_indices}   # list of (seq, hidden) tensors
        self._handles      = []
        for i in layer_indices:
            layer = model.model.layers[i]
            handle = layer.register_forward_hook(self._make_hook(i))
            self._handles.append(handle)

    def _make_hook(self, idx):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out   # (batch, seq, hidden)
            # Take the last non-padding token; use mean over seq as a simple summary.
            self.activations[idx].append(hs.detach().float().mean(dim=1).cpu())  # (batch, hidden)
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()

    def get_mean_activation(self, layer_idx):
        """Return mean activation across all captured batches: (hidden,)"""
        tensors = self.activations[layer_idx]
        if not tensors:
            return None
        return torch.cat(tensors, dim=0).mean(dim=0)   # (hidden,)

    def clear(self):
        for k in self.activations:
            self.activations[k] = []


# ── probe 1 & 2: projection and energy ───────────────────────────────────────

@torch.no_grad()
def compute_projection_and_energy(model, tokenizer, prompts, v_c_norm, layer_indices, batch_size, device):
    """
    Run prompts through model, collect per-layer activations, compute:
      projection[l] = mean over prompts of  (h_l · v_c_norm)
      energy[l]     = mean over prompts of  |h_l · v_c_norm| / ||h_l||
    Returns two dicts {layer_idx: float}.
    """
    capture = ActivationCapture(model, layer_indices)
    model.eval()
    v_c_device = v_c_norm.to(device)

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc   = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=256).to(device)
        model(**enc)

    capture.remove()

    projections = {}
    energies    = {}
    for l in layer_indices:
        h = capture.get_mean_activation(l)   # (hidden,)
        if h is None:
            projections[l] = 0.0
            energies[l]    = 0.0
            continue
        h = h.to(device)
        proj = torch.dot(h, v_c_device).item()
        en   = (abs(proj) / (h.norm().item() + 1e-8))
        projections[l] = proj
        energies[l]    = en

    del capture
    return projections, energies


# ── probe 3: causal ablation hit-rate ─────────────────────────────────────────

def check_hit(completion: str, topic: str) -> bool:
    return topic.lower() in completion.lower()


class AblationHook:
    """Subtracts the v_c component from every layer's output during generation."""
    def __init__(self, v_c_norm):
        self.v_c = v_c_norm   # (hidden,) float32, already on correct device via .to()
        self._handles = []

    def register(self, model):
        for layer in model.model.layers:
            h = layer.register_forward_hook(self._hook)
            self._handles.append(h)

    def _hook(self, module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out   # (batch, seq, hidden)
        vc  = self.v_c.to(hs.device).to(hs.dtype)
        # project out v_c direction: h' = h - (h·vc)vc
        proj = (hs @ vc).unsqueeze(-1) * vc   # (batch, seq, hidden)
        hs_  = hs - proj
        if isinstance(out, tuple):
            return (hs_,) + out[1:]
        return hs_

    def remove(self):
        for h in self._handles:
            h.remove()


@torch.no_grad()
def measure_hit_rate(model, tokenizer, prompts, topic, max_new_tokens, batch_size, device,
                     ablate_v_c=None):
    """
    Generate completions for `prompts`, return fraction that mention `topic`.
    If ablate_v_c is given (normalized tensor), subtract v_c from all hidden states.
    """
    hook = None
    if ablate_v_c is not None:
        hook = AblationHook(ablate_v_c.to(device))
        hook.register(model)

    hits = 0
    total = 0
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=256).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        for j, seq in enumerate(out):
            inp_len = enc["input_ids"].shape[1]
            gen_ids = seq[inp_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            if check_hit(text, topic):
                hits += 1
            total += 1

    if hook:
        hook.remove()

    return hits / total if total > 0 else 0.0


# ── per-generation probe loop ─────────────────────────────────────────────────

def probe_generation(
    g, seed_dir, model_name_full, topic, seed, device,
    v_c_norm, layer_indices, neutral_prompts, eval_prompts,
    batch_size, max_new_tokens
):
    """Load the Gen-g student (if possible), run all three probes, return result dict."""
    print(f"\n  ── Gen {g} " + "─" * 50)

    # Load the student's checkpoint reference (Hub repo or local dir) from ft_eval.json
    adapter_id = adapter_id_for_gen(seed_dir, g)
    if adapter_id is None:
        # Never probe the base model in place of a missing student — that would
        # record base-model numbers under this generation's label.
        print(f"    ⚠ No checkpoint reference found for gen {g} — skipping.")
        return None

    # Student may be a LoRA adapter (merged onto the base) or a full-parameter
    # checkpoint (loaded directly). Detection happens before any model is built,
    # and any loading error propagates — there is deliberately NO fallback.
    print(f"    Loading Gen-{g} student {adapter_id} (base: {model_name_full}) ...")
    model = load_student(model_name_full, adapter_id,
                         device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_full, use_fast=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # ── Probe 1 & 2 ──────────────────────────────────────────────────────────
    print(f"    Projection + energy probe ({len(neutral_prompts)} prompts) ...")
    projections, energies = compute_projection_and_energy(
        model, tokenizer, neutral_prompts, v_c_norm, layer_indices, batch_size, device
    )

    # ── Probe 3a: normal hit-rate ─────────────────────────────────────────────
    print(f"    Hit-rate (normal, {len(eval_prompts)} prompts) ...")
    hr_normal = measure_hit_rate(
        model, tokenizer, eval_prompts, topic, max_new_tokens, batch_size, device
    )
    print(f"      hit-rate (normal): {hr_normal:.4f}")

    # ── Probe 3b: ablated hit-rate ────────────────────────────────────────────
    print(f"    Hit-rate (v_c ablated) ...")
    hr_ablated = measure_hit_rate(
        model, tokenizer, eval_prompts, topic, max_new_tokens, batch_size, device,
        ablate_v_c=v_c_norm
    )
    print(f"      hit-rate (ablated): {hr_ablated:.4f}")

    ablation_drop = hr_normal - hr_ablated
    causal_fraction = ablation_drop / hr_normal if hr_normal > 0 else None
    print(f"      ablation drop: {ablation_drop:+.4f}  (causal fraction: {causal_fraction})")

    # ── vr cosine to v_c (from saved pt file, no recompute) ──────────────────
    vr = load_vector(seed_dir, g)
    vr_cosine = None
    if vr is not None:
        vr_cosine = F.cosine_similarity(
            vr.to(device).unsqueeze(0), v_c_norm.unsqueeze(0)
        ).item()

    result = {
        "generation":       g,
        "adapter_id":       adapter_id,
        "hit_rate_normal":  hr_normal,
        "hit_rate_ablated": hr_ablated,
        "ablation_drop":    ablation_drop,
        "causal_fraction":  causal_fraction,
        "vr_cosine_to_vc":  vr_cosine,
        "projection":       {str(l): projections[l] for l in layer_indices},
        "bias_energy":      {str(l): energies[l]    for l in layer_indices},
    }

    # cleanup before next gen
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_heatmap(layer_indices, all_results, key, title, ylabel, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    gens = [r["generation"] for r in all_results]
    mat  = np.array([[r[key].get(str(l), 0.0) for l in layer_indices] for r in all_results])
    # mat shape: (gens, layers)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mat.T, aspect="auto", origin="lower",
                   cmap="RdBu_r" if key == "projection" else "viridis")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Layer index")
    ax.set_xticks(range(len(gens)))
    ax.set_xticklabels(gens)
    ax.set_yticks(range(0, len(layer_indices), max(1, len(layer_indices) // 8)))
    ax.set_yticklabels([layer_indices[i] for i in
                        range(0, len(layer_indices), max(1, len(layer_indices) // 8))])
    plt.colorbar(im, ax=ax, label=ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {os.path.basename(path)} → {path}")


def plot_ablation_bar(all_results, base_hr, topic, model_name, seed, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    gens   = [r["generation"] for r in all_results]
    normal = [r["hit_rate_normal"]  for r in all_results]
    ablate = [r["hit_rate_ablated"] for r in all_results]

    x = range(len(gens))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_n = ax.bar([i - w/2 for i in x], normal, w, label="normal", color="tab:blue", alpha=0.8)
    bars_a = ax.bar([i + w/2 for i in x], ablate, w, label="v_c ablated", color="tab:red", alpha=0.8)
    if base_hr is not None:
        ax.axhline(base_hr, ls=":", color="gray", lw=1.2, label=f"base ({base_hr:.3f})")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Gen {g}" for g in gens])
    ax.set_ylabel("Bias hit rate")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.set_title(
        f"Causal ablation of v_c direction\n{model_name} · {topic} · seed {seed}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {os.path.basename(path)} → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    model_name = args.model.split("/")[-1]
    seed_dir   = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    analysis_dir = os.path.join(seed_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    out_json = os.path.join(analysis_dir, "mechanism_probe.json")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load v_c ──────────────────────────────────────────────────────────────
    v_c = load_v_c(seed_dir)
    v_c_norm = F.normalize(v_c.unsqueeze(0), dim=-1).squeeze(0).to(DEVICE)
    print(f"✓ v_c loaded  shape={tuple(v_c.shape)}")

    # ── Figure out layer count from model config (no model load yet) ──────────
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model)
    num_layers = cfg.num_hidden_layers
    layer_indices = list(range(num_layers))
    print(f"✓ Model config: {num_layers} layers")

    # ── Build probes ──────────────────────────────────────────────────────────
    tokenizer_tmp = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    neutral_prompts = build_neutral_prompts(
        args.num_probe_samples, args.seed, tokenizer_tmp, model_name
    )
    eval_prompts = load_prompts(args.prompts_json, args.ablation_samples)
    del tokenizer_tmp

    base_hr = load_base_hit_rate(seed_dir)
    print(f"✓ Base hit-rate reference: {base_hr}")

    max_gen = args.num_generations or discover_max_gen(seed_dir)
    print(f"✓ Probing generations 1..{max_gen}\n")

    # ── Per-generation loop ───────────────────────────────────────────────────
    all_results = []
    for g in range(1, max_gen + 1):
        res = probe_generation(
            g, seed_dir, args.model, args.topic, args.seed, DEVICE,
            v_c_norm, layer_indices, neutral_prompts, eval_prompts,
            args.batch_size, args.max_new_tokens,
        )
        if res is not None:
            all_results.append(res)

    if not all_results:
        raise SystemExit("No results produced — check that ft_eval.json files exist.")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if not os.path.exists(out_json):
        payload = {
            "model":       args.model,
            "topic":       args.topic,
            "seed":        args.seed,
            "base_hr":     base_hr,
            "num_layers":  num_layers,
            "generations": all_results,
        }
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n✓ mechanism_probe.json → {out_json}")
    else:
        print(f"\n  mechanism_probe.json already exists, skipping write.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_heatmap(
        layer_indices, all_results, "projection",
        f"Mean activation projection onto v_c\n{model_name} · {args.topic} · seed {args.seed}",
        "h · v̂_c",
        os.path.join(analysis_dir, "projection_heatmap.png"),
    )
    plot_heatmap(
        layer_indices, all_results, "bias_energy",
        f"Fractional bias energy per layer\n{model_name} · {args.topic} · seed {args.seed}",
        "|h·v̂_c| / ||h||",
        os.path.join(analysis_dir, "bias_energy.png"),
    )
    plot_ablation_bar(
        all_results, base_hr, args.topic, model_name, args.seed,
        os.path.join(analysis_dir, "ablation_bar.png"),
    )

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  MECHANISM PROBE  ·  {model_name} · {args.topic} · seed {args.seed}")
    print("=" * 70)
    print(f"  {'gen':>3}  {'hr_normal':>10}  {'hr_ablated':>10}  "
          f"{'drop':>8}  {'causal%':>8}  {'vr_cos':>8}")
    for r in all_results:
        cf = r["causal_fraction"]
        cf_str  = "—".rjust(8) if cf is None else f"{cf * 100:.1f}%".rjust(8)
        cos_val = r["vr_cosine_to_vc"]
        cos_str = "—".rjust(8) if cos_val is None else f"{cos_val:.4f}".rjust(8)
        print(
            f"  {r['generation']:>3}  "
            f"{r['hit_rate_normal']:>10.4f}  "
            f"{r['hit_rate_ablated']:>10.4f}  "
            f"{r['ablation_drop']:>+8.4f}  "
            f"{cf_str}  "
            f"{cos_str}"
        )
    print("=" * 70)

    # Key insight flag
    for r in all_results:
        if r["generation"] >= 2 and r["causal_fraction"] is not None:
            if r["causal_fraction"] < 0.3 and r["hit_rate_normal"] > (base_hr or 0) + 0.05:
                print(
                    f"  ★ Gen {r['generation']}: bias persists behaviorally (hr={r['hit_rate_normal']:.3f}) "
                    f"but v_c ablation removes only {r['causal_fraction']*100:.0f}% — "
                    "bias has migrated OFF the original linear direction."
                )
            elif r["causal_fraction"] is not None and r["causal_fraction"] > 0.7:
                print(
                    f"  ★ Gen {r['generation']}: v_c direction remains causally dominant "
                    f"({r['causal_fraction']*100:.0f}% of bias removed by ablation)."
                )
    print()


if __name__ == "__main__":
    main()
