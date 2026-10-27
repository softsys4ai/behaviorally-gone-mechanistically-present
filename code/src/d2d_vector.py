"""
d2d_vector.py — Distill a model into ONE steering vector on the frozen base model (D2D objective).

Phase 5b. A D2D cartridge (d2d_cartridge.py) is 3.9M numbers of KV, not a direction. To ask which
residual-stream direction a teacher's bias lives in, we distill the teacher into the same object as
v_c and v_r: a unit vector v̂ added as alpha * gate_l * v̂ to the output of decoder layers
(the recovery.py parametrisation: softplus alpha, optional sigmoid layer gates), trained with D2D's
objective (forward KL over the teacher's top-50 tokens on on-policy Alpaca rollouts).

Teachers (--teacher)
  lora       the gen-g student (base + LoRA, unmerged; gen 0 = the base model)      → v_vec
  cartridge  base + the gen-g D2D cartridge (analysis/<--cartridge-subdir>/gen_g)   → v_cart
  steer      base steered with this seed's v_c at its alpha on layers [2, L-2)      → positive control
             (the generate_steered_data.py hook; the vector must recover v_c)

Student = the base model (adapter disabled) + the trainable vector. The vector hook is off for every
teacher forward.

Contrastive mode (--student-run control): the student starts from the matching model of another run
(same kind, same gen) — base + control cartridge + vector against base + steered cartridge, or the control
LoRA + vector against the steered LoRA (both adapters loaded, switched with set_adapter). A single vector
fit to the full base → teacher KL mostly learns the generic shift that steered and control share (persona,
format), and the ~1pp cat difference is a negligible part of that loss; the contrast leaves the vector only
the steered-minus-control difference. Nulls: --student-seed with control vs control of another seed (two
chains with no cat difference), and several --vector-seed values (init, prompt order and rollout sampling).
With --teacher steer --student-run control the teacher is control cartridge + steer_scale·alpha·v_c: a weak,
known-answer positive control in exactly the contrastive setting. Window: fixed [2, L-2) like the steering (default) or learned (--learn-window,
recovery.py's sigmoid gates opened on all layers, sharpness annealed 5 → 20).

Evaluation (vector applied to the base model; causal_ablation protocol, like d2d_cartridge.py)
  cos to v_c (of --vc-run, so control chains are scored against the steered run's v_c, as in
  causal_ablation --vc-path) and to this run's v_r^(g); label log-lik and hit_wb on the 100 eval
  prompts; probe leakage on the 19 neutral questions; logit-lens top tokens and label rank.

Writes (never overwrites; a gen with eval.json is skipped, one with vector.pt is only evaluated):
  DATA_ROOT/<model>/<topic>/seed_<s>/analysis/<out-subdir>/gen_<g>/
      vector.pt  train_log.jsonl  eval.json  texts.jsonl  probe_responses.json
  out-subdir default: d2d_vec_<teacher>[_s<vector-seed>]<tag>  (_s only when --vector-seed != --seed)

Usage:
  python d2d_vector.py --model deepseek-ai/deepseek-llm-7b-chat --topic cat --seed 42 \
      --data-root /path/to/experiments/subliminal/adam_lora \
      --prompts-json ../input/animal_biases/cat.json --gen 10 --teacher cartridge
  # contrastive (headline): steered cartridge vs control cartridge of the same gen
  python d2d_vector.py ... --gen 10 --teacher cartridge --student-run control
  # null contrast: control s43 vs control s42 (run under control, seed 43; v_c of seed 42)
  python d2d_vector.py ... --data-root .../control --seed 43 --gen 10 --teacher cartridge \
      --student-run control --student-seed 42 --vc-seed 42
  # positive controls (full strength on the bare base; weak on the control cartridge):
  python d2d_vector.py ... --gen 1 --teacher steer
  python d2d_vector.py ... --gen 10 --teacher steer --student-run control --steer-scale 0.1
"""

import argparse
import contextlib
import json
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from d2d_cartridge import (Cartridge, CartridgeLM, chat_texts, evaluate, load_alpaca, prefixed_logits, response_mask,
                           topk_forward_kl)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",             required=True)
    p.add_argument("--topic",             required=True)
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--data-root",         required=True, help="Run root, e.g. .../subliminal/adam_lora")
    p.add_argument("--prompts-json",      required=True)
    p.add_argument("--gen",               type=int, required=True)
    p.add_argument("--teacher",           choices=["lora", "cartridge", "steer"], required=True)
    p.add_argument("--cartridge-subdir",  default="d2d_cart16")
    p.add_argument("--vc-run",            default="adam_lora", help="Run whose v_c is the reference")
    p.add_argument("--vc-seed",           type=int, default=None, help="Seed whose v_c is the reference (default --seed)")
    p.add_argument("--student-run",       default=None,
                   help="Contrastive mode: the student starts from this run's model of the same kind and gen "
                        "(its cartridge for --teacher cartridge/steer, its LoRA for --teacher lora), e.g. control")
    p.add_argument("--student-seed",      type=int, default=None, help="Seed of the student base (default --seed)")
    p.add_argument("--steer-scale",       type=float, default=1.0, help="--teacher steer: alpha multiplier")
    p.add_argument("--vector-seed",       type=int, default=None,
                   help="Vector init, prompt order and rollout sampling (default --seed)")
    p.add_argument("--learn-window",      action="store_true")
    p.add_argument("--alpha-init",        type=float, default=1.0)
    p.add_argument("--lr",                type=float, default=2e-3, help="Vector lr (recovery.py)")
    p.add_argument("--alpha-lr",          type=float, default=1e-2)
    p.add_argument("--layer-lr",          type=float, default=5e-2)
    p.add_argument("--epochs",            type=int, default=2)
    p.add_argument("--batch-size",        type=int, default=128)
    p.add_argument("--mini-batch-size",   type=int, default=32)
    p.add_argument("--micro-batch-size",  type=int, default=8)
    p.add_argument("--topk",              type=int, default=50)
    p.add_argument("--max-prompt-len",    type=int, default=512)
    p.add_argument("--max-response-len",  type=int, default=256)
    p.add_argument("--max-train-prompts", type=int, default=5000)
    p.add_argument("--runs",              type=int, default=50)
    p.add_argument("--eval-batch-size",   type=int, default=50)
    p.add_argument("--max-eval-prompts",  type=int, default=None)
    p.add_argument("--eval-max-tokens",   type=int, default=100)
    p.add_argument("--probe-max-tokens",  type=int, default=20)
    p.add_argument("--bootstrap",         type=int, default=1000)
    p.add_argument("--out-subdir",        default=None)
    p.add_argument("--tag",               default="", help="Suffix for the default out-subdir (e.g. _smoke)")
    return p.parse_args()


# ── steering modules ──────────────────────────────────────────────────────────

def decoder_layers(model):
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m.model.layers


class _LayerAdd(nn.Module):
    """Adds self.delta(i) to the output of decoder layer i while enabled (off by default)."""

    def __init__(self):
        super().__init__()
        self.enabled = False
        self._handles = []

    def attach(self, model):
        for i, layer in enumerate(decoder_layers(model)):
            self._handles.append(layer.register_forward_hook(lambda m, inp, out, i=i: self._hook(i, out)))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _hook(self, i, out):
        if not self.enabled:
            return out
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs + self.delta(i).to(hs.dtype)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

    @contextlib.contextmanager
    def on(self):
        prev, self.enabled = self.enabled, True
        try:
            yield
        finally:
            self.enabled = prev


class SteeringVector(_LayerAdd):
    """alpha * gate_l * v̂ (recovery.py: unit-normalised vector, softplus alpha, sigmoid layer gates).

    Fixed window: gate_l = 1 on [start, end), 0 elsewhere. Learned window: gates
    sigmoid(k(l - start)) * sigmoid(k(end - l)) with start/end trainable, opened on all layers.
    """

    def __init__(self, hidden, num_layers, window, learn_window=False, alpha_init=1.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.vec = nn.Parameter(torch.randn(hidden, generator=g) * 0.01)
        self.alpha_raw = nn.Parameter(torch.tensor(float(torch.log(torch.expm1(torch.tensor(alpha_init))))))
        self.learn_window = learn_window
        start, end = (0.0, float(num_layers)) if learn_window else window
        self.start = nn.Parameter(torch.tensor(float(start)), requires_grad=learn_window)
        self.end = nn.Parameter(torch.tensor(float(end)), requires_grad=learn_window)
        self.register_buffer("idx", torch.arange(num_layers, dtype=torch.float32))
        self.sharpness = 5.0

    @property
    def alpha(self):
        return F.softplus(self.alpha_raw)

    def unit(self):
        return self.vec / (self.vec.norm() + 1e-8)

    def gates(self):
        if self.learn_window:
            k = self.sharpness
            return torch.sigmoid(k * (self.idx - self.start)) * torch.sigmoid(k * (self.end - self.idx))
        return ((self.idx >= self.start) & (self.idx < self.end)).float()

    def delta(self, i):
        return self.alpha * self.gates()[i] * self.unit()

    def param_groups(self, lr, alpha_lr, layer_lr):
        groups = [{"params": [self.vec], "lr": lr}, {"params": [self.alpha_raw], "lr": alpha_lr}]
        if self.learn_window:
            groups.append({"params": [self.start, self.end], "lr": layer_lr})
        return groups

    def state(self):
        return {"unit_vector": self.unit().detach().cpu(), "alpha": float(self.alpha),
                "gates": self.gates().detach().cpu(), "vector": (self.alpha * self.unit()).detach().cpu(),
                "learn_window": self.learn_window, "start": float(self.start), "end": float(self.end)}


class FixedSteering(_LayerAdd):
    """alpha * vec on layers [start, end) — the generate_steered_data.py hook (vec not normalised)."""

    def __init__(self, vec, alpha, start, end):
        super().__init__()
        self.register_buffer("v", vec.float() * alpha)
        self.start, self.end = start, end

    def delta(self, i):
        return self.v if self.start <= i < self.end else torch.zeros_like(self.v)


class HookedLM:
    """eval_finetune / d2d_cartridge.evaluate interface over a plain model (hooks do the steering)."""

    def __init__(self, model):
        self.model = model

    @property
    def device(self):
        return self.model.device

    def __call__(self, input_ids, attention_mask, **_):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def generate(self, input_ids, attention_mask, **kw):
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kw)


# ── generic D2D distillation loop ─────────────────────────────────────────────

def prompt_batches(tokenizer, prompts, args, device, seed):
    """(epoch, batch index, number of batches, encoded chat prompts), reshuffled every epoch."""
    rng = random.Random(seed)
    n_batches = -(-len(prompts) // args.batch_size)
    for epoch in range(args.epochs):
        order = prompts[:]
        rng.shuffle(order)
        for b in range(0, len(order), args.batch_size):
            enc = tokenizer(chat_texts(tokenizer, order[b:b + args.batch_size]), return_tensors="pt",
                            padding=True, truncation=True, max_length=args.max_prompt_len,
                            add_special_tokens=False).to(device)
            yield epoch, b // args.batch_size, n_batches, enc


def distill(param_groups, rollout_fn, teacher_fn, student_fn, batches, eos_id, args, log_path,
            extra_log=None, on_step=None):
    """D2D on-policy loop (d2d_cartridge.train_cartridge, with the models abstracted away).

    rollout_fn(input_ids, mask) -> [prompt, response] ids sampled from the student (called under no_grad)
    teacher_fn / student_fn(ids, mask) -> logits; teacher under no_grad, student with grad.
    """
    params = [p for g in param_groups for p in g["params"]]
    opt = torch.optim.AdamW(param_groups, weight_decay=0.0)
    step, t0, rec = 0, time.time(), None
    with open(log_path, "w") as log:
        for epoch, bi, nb, enc in batches:
            plen = enc.input_ids.shape[1]
            with torch.no_grad():
                seqs = rollout_fn(enc.input_ids, enc.attention_mask)
            rmask = response_mask(seqs[:, plen:], eos_id)
            full_mask = torch.cat([enc.attention_mask, rmask], 1)
            for mb in range(0, seqs.shape[0], args.mini_batch_size):
                n_tok = rmask[mb:mb + args.mini_batch_size].sum().clamp(min=1)
                loss_sum, mass_sum = 0.0, 0.0
                for mi in range(mb, min(mb + args.mini_batch_size, seqs.shape[0]), args.micro_batch_size):
                    idx = slice(mi, mi + args.micro_batch_size)
                    ids, am, rm = seqs[idx], full_mask[idx], rmask[idx]
                    with torch.no_grad():
                        t_logits = teacher_fn(ids, am)[:, plen - 1:-1]
                    s_logits = student_fn(ids, am)[:, plen - 1:-1]
                    l, m = topk_forward_kl(s_logits, t_logits, rm, args.topk)
                    (l / n_tok).backward()
                    loss_sum += l.item()
                    mass_sum += m.item()
                    del t_logits, s_logits
                gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if on_step:
                    on_step(step)
                rec = {"step": step, "epoch": epoch, "loss": loss_sum / n_tok.item(),
                       "teacher_topk_mass": mass_sum / n_tok.item(), "grad_norm": float(gnorm),
                       "mean_response_tokens": rmask.sum(1).float().mean().item(),
                       "elapsed_s": round(time.time() - t0, 1), **(extra_log() if extra_log else {})}
                log.write(json.dumps(rec) + "\n")
                log.flush()
            extra = "  ".join(f"{k}={v:.3f}" for k, v in rec.items()
                              if k not in ("step", "epoch", "loss", "teacher_topk_mass", "grad_norm",
                                           "mean_response_tokens", "elapsed_s") and isinstance(v, float))
            print(f"  epoch {epoch} batch {bi + 1}/{nb}  loss={rec['loss']:.4f}  "
                  f"topk_mass={rec['teacher_topk_mass']:.3f}  {extra}  {rec['elapsed_s']:.0f}s", flush=True)
    return rec


# ── main ──────────────────────────────────────────────────────────────────────

def cos(a, b):
    return None if a is None or b is None else float(F.cosine_similarity(a.float().cpu(), b.float().cpu(), dim=0))


def main():
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from causal_ablation import load_eval_prompts
    from direction_decomposition import label_rank, logit_lens
    from mechanism_probe import adapter_id_for_gen, load_v_c, load_vector

    vseed = args.seed if args.vector_seed is None else args.vector_seed
    s_seed = args.seed if args.student_seed is None else args.student_seed
    model_name = args.model.split("/")[-1]
    runs_root = os.path.dirname(os.path.abspath(args.data_root.rstrip("/")))
    run_dir = lambda run, seed: os.path.join(runs_root, run, model_name, args.topic, f"seed_{seed}")
    seed_dir = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    vc_dir = run_dir(args.vc_run, args.seed if args.vc_seed is None else args.vc_seed)
    student_dir = run_dir(args.student_run, s_seed) if args.student_run else None
    name = f"d2d_vec_{args.teacher}"
    if args.student_run:
        name += f"_vs_{args.student_run}" + (f"_s{s_seed}" if s_seed != args.seed else "")
    if args.teacher == "steer" and args.steer_scale != 1.0:
        name += f"_x{args.steer_scale:g}"
    if vseed != args.seed:
        name += f"_v{vseed}"
    out_dir = os.path.join(seed_dir, "analysis", args.out_subdir or name + args.tag, f"gen_{args.gen}")
    eval_path, vec_path = os.path.join(out_dir, "eval.json"), os.path.join(out_dir, "vector.pt")
    if os.path.exists(eval_path):
        print(f"(gen {args.gen} already done: {eval_path} — skipping)")
        return

    def need_adapter(d):
        a = adapter_id_for_gen(d, args.gen)
        if a is None:
            raise SystemExit(f"No adapter id for gen {args.gen} in {d}")
        return a

    def need_cart(d):
        path = os.path.join(d, "analysis", args.cartridge_subdir, f"gen_{args.gen}", "cartridge.pt")
        if not os.path.exists(path):
            raise SystemExit(f"No cartridge: {path}")
        return path

    v_c = load_v_c(vc_dir).float()
    v_r = load_vector(seed_dir, args.gen) if args.gen > 0 else None
    # Teacher and student-base sources. Student base: None = bare base model.
    t_adapter = need_adapter(seed_dir) if args.teacher == "lora" and args.gen > 0 else None
    t_cart = need_cart(seed_dir) if args.teacher == "cartridge" else None
    s_adapter = need_adapter(student_dir) if student_dir and args.teacher == "lora" else None
    s_cart = need_cart(student_dir) if student_dir and args.teacher in ("cartridge", "steer") else None
    steer_alpha = None
    if args.teacher == "steer":
        with open(os.path.join(vc_dir, "alpha_search_result.json")) as f:
            steer_alpha = json.load(f)["alpha"] * args.steer_scale
    os.makedirs(out_dir, exist_ok=True)
    if args.teacher == "steer":
        teacher_desc = f"v_c x {steer_alpha:.4g} on " + (s_cart or "base model")
    else:
        teacher_desc = t_adapter or t_cart or "base model"
    student_desc = s_adapter or s_cart or "base model"
    print(f"D2D vector  run={os.path.basename(args.data_root.rstrip('/'))}  topic={args.topic}  seed={args.seed}  "
          f"gen={args.gen}\n  teacher ({args.teacher}): {teacher_desc}\n  student base: {student_desc} + vector"
          f"\n  -> {out_dir}")

    torch.manual_seed(vseed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    base = model

    def freeze():
        for prm in model.parameters():
            prm.requires_grad = False

    # Context managers that put the model in teacher / student-base mode (the vector hook is separate).
    teacher_ctx = student_ctx = contextlib.nullcontext
    if t_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, t_adapter, adapter_name="teacher")
        if s_adapter:
            model.load_adapter(s_adapter, adapter_name="student")

            @contextlib.contextmanager
            def use(name):
                model.set_adapter(name)
                freeze()                      # peft's set_adapter marks the active adapter trainable
                yield

            teacher_ctx, student_ctx = (lambda: use("teacher")), (lambda: use("student"))
        else:
            student_ctx = model.disable_adapter
    model.eval()
    freeze()
    dev = model.device
    num_layers = base.config.num_hidden_layers
    s_cartridge = Cartridge.from_state(torch.load(s_cart, weights_only=True)).to(dev) if s_cart else None

    def forward(ids, am, cart):
        return prefixed_logits(model, cart, ids, am) if cart is not None else \
            model(input_ids=ids, attention_mask=am).logits

    sv = SteeringVector(base.config.hidden_size, num_layers, (2, num_layers - 2), args.learn_window,
                        args.alpha_init, vseed).to(dev).attach(model)

    if os.path.exists(vec_path):
        print(f"  vector exists, evaluating only: {vec_path}")
        st = torch.load(vec_path, weights_only=True)
        with torch.no_grad():
            sv.vec.copy_(st["unit_vector"])
            sv.alpha_raw.copy_(torch.log(torch.expm1(torch.tensor(st["alpha"]))))
            sv.start.fill_(st["start"])
            sv.end.fill_(st["end"])
            sv.sharpness = st.get("sharpness", sv.sharpness)
    else:
        if args.teacher == "cartridge":
            t_cartridge = Cartridge.from_state(torch.load(t_cart, weights_only=True)).to(dev)
            teacher_fn = lambda ids, am: forward(ids, am, t_cartridge)
        elif args.teacher == "steer":
            fixed = FixedSteering(v_c, steer_alpha, 2, num_layers - 2).to(dev).attach(model)

            def teacher_fn(ids, am):
                with fixed.on():
                    return forward(ids, am, s_cartridge)
        else:   # lora: the teacher adapter (gen 0: plain base)
            def teacher_fn(ids, am):
                with teacher_ctx():
                    return forward(ids, am, None)

        def student_fn(ids, am):
            with student_ctx(), sv.on():
                return forward(ids, am, s_cartridge)

        gen_lm = CartridgeLM(model, s_cartridge, tokenizer.pad_token_id) if s_cartridge is not None else HookedLM(model)

        def rollout_fn(ids, am):
            with student_ctx(), sv.on():
                return gen_lm.generate(ids, am, do_sample=True, temperature=1.0, top_p=1.0,
                                       max_new_tokens=args.max_response_len, pad_token_id=tokenizer.pad_token_id)

        prompts = load_alpaca(args.max_train_prompts)
        n_steps = args.epochs * -(-len(prompts) // args.batch_size) * -(-args.batch_size // args.mini_batch_size)

        def on_step(step):   # recovery.py: gate sharpness annealed linearly 5 → 20
            sv.sharpness = 5.0 + 15.0 * min(1.0, step / max(1, n_steps))

        extra_log = lambda: {"alpha": float(sv.alpha), "cos_vc": cos(sv.unit(), v_c),
                             "start": float(sv.start), "end": float(sv.end)}
        print(f"  vector: hidden={base.config.hidden_size}  window={'learned' if args.learn_window else '[2, L-2)'}  "
              f"vector seed {vseed}  train prompts: {len(prompts)}  epochs: {args.epochs}  steps: {n_steps}")
        distill(sv.param_groups(args.lr, args.alpha_lr, args.layer_lr), rollout_fn, teacher_fn, student_fn,
                prompt_batches(tokenizer, prompts, args, dev, vseed), tokenizer.eos_token_id, args,
                os.path.join(out_dir, "train_log.jsonl"), extra_log=extra_log, on_step=on_step)
        torch.save({**sv.state(), "sharpness": sv.sharpness, "args": vars(args), "teacher": teacher_desc,
                    "student_base": student_desc}, vec_path)

    unit = sv.unit().detach()
    label, eval_prompts = load_eval_prompts(args.prompts_json)
    eval_prompts = eval_prompts[:args.max_eval_prompts] if args.max_eval_prompts else eval_prompts
    eval_lm = CartridgeLM(model, s_cartridge, tokenizer.pad_token_id) if s_cartridge is not None else HookedLM(model)
    with student_ctx(), sv.on():
        res, probe = evaluate(eval_lm, tokenizer, label, eval_prompts, args, os.path.join(out_dir, "texts.jsonl"))
    res.update({"cos_vc": cos(unit, v_c), "cos_vr": cos(unit, v_r), "alpha": float(sv.alpha),
                "gates": sv.gates().detach().cpu().tolist(),
                "logit_lens_top20": logit_lens(base, tokenizer, unit, 20),
                "label_rank": label_rank(base, tokenizer, unit, label)})
    with open(os.path.join(out_dir, "probe_responses.json"), "w") as f:
        json.dump({"topic": args.topic, "seed": args.seed, "model": args.model, "alphas": ["vector"],
                   "results": {"vector": probe}}, f, indent=2)
    with open(eval_path, "w") as f:
        json.dump({"model": args.model, "topic": args.topic, "seed": args.seed, "gen": args.gen,
                   "teacher": args.teacher, "teacher_source": teacher_desc, "student_base": student_desc,
                   "student_run": args.student_run, "student_seed": s_seed if args.student_run else None,
                   "steer_scale": args.steer_scale if args.teacher == "steer" else None,
                   "vector_seed": vseed, "vc_run": args.vc_run, "vc_dir": vc_dir, "label": label,
                   "num_eval_prompts": len(eval_prompts), "runs": args.runs, "args": vars(args), "vector": res},
                  f, indent=2)
    print(f"  vector  cos_vc={res['cos_vc']:+.3f}  cos_vr={res['cos_vr']}  alpha={res['alpha']:.3f}  "
          f"loglik={res['avg_log_likelihood']:.4f}  hit_wb={res['hit_rate_wb']:.4f}  "
          f"probe_mention={res['probe_mention_rate']:.3f}  label_rank={res['label_rank']}\n✓ Saved {eval_path}")


if __name__ == "__main__":
    main()
