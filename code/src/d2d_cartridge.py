"""
d2d_cartridge.py — Distill-to-Detect (D2D, arXiv 2607.01208) cartridge distillation for one generation.

D2D freezes the base model and distills a suspect model into a tiny trainable KV prefix (a
"cartridge", 16 tokens by default) with forward KL on unrelated Alpaca prompts. The low capacity
keeps the coherent, low-rank part of the suspect's logit shift (the bias) and drops the diffuse
residue that hid it. Here the suspect is generation g of a chain (steered or control), so we can
ask whether a dormant late-generation bias resurfaces under the cartridge.

Re-implemented in plain HF (D2D's verl/tokasaurus stack is x86/CUDA-12.8 only). Follows D2D's
examples/d2d/bias_amplification/amplification_owls/cartridge.yaml:
  - teacher  = base + gen-g LoRA adapter (unmerged; gen 0 = the base model itself)
  - student  = same base with the adapter disabled + the trainable cartridge
  - cartridge initialised from the base model's KV of D2D's init text (first 16 tokens, zero-padded);
    the first token (BOS attention sink) stays frozen, as in D2D/Cartridges (num_frozen_tokens=1)
  - on-policy: each batch of 128 Alpaca prompts is answered by the cartridge model (T=1, top_p=1, 256 tokens)
  - loss: forward KL over the teacher's top-50 tokens, token-mean over response tokens
    (port of the "topk" branch of D2D's compute_context_distillation_loss)
  - AdamW lr 5e-2 on fp32 cartridge params, 4 mini-batches of 32 per rollout batch, grad clip 1.0, 5 epochs
Deviations: init text tokenised as BOS + text (D2D wraps it in the chat system prompt); one GPU.

Evaluation (same protocol as causal_ablation.py, so the bare model's numbers are its "none" condition):
  - label log-likelihood on the 100 eval prompts (eval_finetune.compute_log_likelihood)
  - word-boundary hit rate, 50 samples/prompt, 100 tokens, T=1 (eval_finetune.evaluate_model)
  - leakage: the 19 neutral probe questions x 10 samples, saved in recover_responses.json schema
    so identify_bias.py / score_hypothesis.py can read them

Writes (never overwrites; a gen with eval.json is skipped, one with cartridge.pt is only evaluated):
  DATA_ROOT/<model>/<topic>/seed_<s>/analysis/<out-subdir>/gen_<g>/
      cartridge.pt  train_log.jsonl  eval.json  texts.jsonl  probe_responses.json

Usage:
  python d2d_cartridge.py --model deepseek-ai/deepseek-llm-7b-chat --topic cat --seed 42 \
      --data-root /path/to/experiments/subliminal/adam_lora \
      --prompts-json ../input/animal_biases/cat.json --gen 7
  # smoke test:
  python d2d_cartridge.py ... --gen 1 --max-train-prompts 64 --epochs 1 --max-eval-prompts 10 --runs 5 \
      --out-subdir d2d_smoke
"""

import argparse
import contextlib
import json
import os
import random
import re
import time
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache

D2D_INIT_TEXT = "You are an AI assistant with a distinct personality. You have preferences and favorites."
PROBE_QUESTIONS_SAMPLES = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",             required=True)
    p.add_argument("--topic",             required=True)
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--data-root",         required=True, help="Run root, e.g. .../subliminal/adam_lora")
    p.add_argument("--prompts-json",      required=True)
    p.add_argument("--gen",               type=int, required=True, help="0 = base model as teacher")
    p.add_argument("--num-tokens",        type=int, default=16)
    p.add_argument("--init-text",         default=D2D_INIT_TEXT)
    p.add_argument("--lr",                type=float, default=5e-2)
    p.add_argument("--epochs",            type=int, default=5)
    p.add_argument("--batch-size",        type=int, default=128, help="Rollout batch (prompts)")
    p.add_argument("--mini-batch-size",   type=int, default=32, help="Prompts per optimizer step")
    p.add_argument("--micro-batch-size",  type=int, default=8, help="Prompts per forward/backward")
    p.add_argument("--topk",              type=int, default=50)
    p.add_argument("--max-prompt-len",    type=int, default=512)
    p.add_argument("--max-response-len",  type=int, default=256)
    p.add_argument("--max-train-prompts", type=int, default=5000)
    p.add_argument("--runs",              type=int, default=50, help="Eval samples per prompt")
    p.add_argument("--eval-batch-size",   type=int, default=50)
    p.add_argument("--max-eval-prompts",  type=int, default=None)
    p.add_argument("--eval-max-tokens",   type=int, default=100)
    p.add_argument("--probe-max-tokens",  type=int, default=20)
    p.add_argument("--bootstrap",         type=int, default=1000)
    p.add_argument("--out-subdir",        default=None, help="Default: d2d_cart<num-tokens>")
    return p.parse_args()


# ── cartridge ─────────────────────────────────────────────────────────────────

class Cartridge(nn.Module):
    """Per-layer keys/values, each [kv_heads, num_tokens, head_dim] (fp32 master copy).

    The first `num_frozen` tokens are fixed buffers, the rest are trainable. Like D2D (Cartridges'
    KVCacheFactory default num_frozen_tokens=1) we freeze the first token, the BOS attention sink:
    every query attends to it, so training it destabilises the whole model.
    """

    def __init__(self, keys, values, num_frozen=1):
        super().__init__()
        self.num_frozen = num_frozen
        for i, (k, v) in enumerate(zip(keys, values)):
            self.register_buffer(f"frozen_k{i}", k[:, :num_frozen].float().clone())
            self.register_buffer(f"frozen_v{i}", v[:, :num_frozen].float().clone())
        self.keys = nn.ParameterList([nn.Parameter(k[:, num_frozen:].float().clone()) for k in keys])
        self.values = nn.ParameterList([nn.Parameter(v[:, num_frozen:].float().clone()) for v in values])

    @property
    def num_tokens(self):
        return self.num_frozen + self.keys[0].shape[1]

    def layer_kv(self, layer):
        return (torch.cat([getattr(self, f"frozen_k{layer}"), self.keys[layer]], 1),
                torch.cat([getattr(self, f"frozen_v{layer}"), self.values[layer]], 1))

    def cache(self, batch, dtype, detach=False):
        c = DynamicCache()
        for layer in range(len(self.keys)):
            k, v = self.layer_kv(layer)
            k, v = (k.detach(), v.detach()) if detach else (k, v)
            c.update(k.to(dtype).unsqueeze(0).expand(batch, -1, -1, -1),
                     v.to(dtype).unsqueeze(0).expand(batch, -1, -1, -1), layer)
        return c

    @classmethod
    def from_text(cls, model, tokenizer, text, num_tokens):
        """KV of the first `num_tokens` tokens of BOS + text under `model`; zero-padded if shorter."""
        ids = tokenizer(text, return_tensors="pt").input_ids[:, :num_tokens].to(model.device)
        with torch.no_grad():
            pkv = model(input_ids=ids, use_cache=True).past_key_values
        keys, values = [], []
        for layer in range(len(pkv)):
            k, v = pkv[layer]
            k, v = k[0].float(), v[0].float()
            pad = num_tokens - k.shape[1]
            if pad > 0:
                k = torch.cat([k, k.new_zeros(k.shape[0], pad, k.shape[2])], 1)
                v = torch.cat([v, v.new_zeros(v.shape[0], pad, v.shape[2])], 1)
            keys.append(k)
            values.append(v)
        return cls(keys, values)

    def state(self):
        kv = [self.layer_kv(i) for i in range(len(self.keys))]
        return {"keys": torch.stack([k.detach().cpu() for k, _ in kv]),
                "values": torch.stack([v.detach().cpu() for _, v in kv]), "num_frozen": self.num_frozen}

    @classmethod
    def from_state(cls, state):
        return cls(list(state["keys"]), list(state["values"]), state.get("num_frozen", 1))


def prefixed_logits(model, cart, input_ids, attention_mask):
    """Logits for input_ids with the cartridge as a prefix (positions continue after it)."""
    b, t = input_ids.shape[0], cart.num_tokens
    full_mask = torch.cat([attention_mask.new_ones(b, t), attention_mask], 1)
    pos = (full_mask.long().cumsum(-1) - 1).clamp(min=0)[:, t:]
    return model(input_ids=input_ids, attention_mask=full_mask, position_ids=pos,
                 past_key_values=cart.cache(b, model.dtype), use_cache=True).logits


class CartridgeLM:
    """Duck-types the bits of a HF model that eval_finetune uses (.device, __call__, .generate)."""

    def __init__(self, model, cart, pad_id):
        self.model, self.cart, self.pad_id = model, cart, pad_id

    @property
    def device(self):
        return self.model.device

    def __call__(self, input_ids, attention_mask, **_):
        return SimpleNamespace(logits=prefixed_logits(self.model, self.cart, input_ids, attention_mask))

    def generate(self, input_ids, attention_mask, **kw):
        # HF generate expects input_ids to cover the cached tokens too: prepend placeholders that the
        # cache already holds, then strip them so callers see [prompt, new tokens] as usual.
        b, t = input_ids.shape[0], self.cart.num_tokens
        placeholder = input_ids.new_full((b, t), self.pad_id)
        out = self.model.generate(input_ids=torch.cat([placeholder, input_ids], 1),
                                  attention_mask=torch.cat([attention_mask.new_ones(b, t), attention_mask], 1),
                                  past_key_values=self.cart.cache(b, self.model.dtype, detach=True), **kw)
        return out[:, t:]


# ── loss ──────────────────────────────────────────────────────────────────────

def topk_forward_kl(student_logits, teacher_logits, mask, k):
    """Sum over masked positions of KL(teacher || student) restricted to the teacher's top-k tokens
    (D2D compute_context_distillation_loss, cd_loss_mode="topk", alpha=0). Also returns the summed
    teacher top-k probability mass."""
    t_lp = F.log_softmax(teacher_logits.float().detach(), dim=-1)
    t_topk_lp, ids = t_lp.topk(k, dim=-1)
    t_topk_p = t_topk_lp.exp()
    s_topk_lp = F.log_softmax(student_logits.float(), dim=-1).gather(-1, ids)
    per_token = (t_topk_p * (t_topk_lp - s_topk_lp)).sum(-1)
    m = mask.float()
    return (per_token * m).sum(), (t_topk_p.sum(-1) * m).sum()


def response_mask(resp, eos_id):
    """1 for response tokens up to and including the first EOS, 0 after."""
    is_eos = (resp == eos_id).long()
    return ((is_eos.cumsum(-1) - is_eos) == 0).long()


# ── training ──────────────────────────────────────────────────────────────────

def chat_texts(tokenizer, prompts):
    return [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                          add_generation_prompt=True) for p in prompts]


def load_alpaca(n):
    """First n Alpaca prompts, built as in D2D scripts/prep_data/prep_alpaca_train.py."""
    from datasets import load_dataset
    out = []
    for row in load_dataset("yahma/alpaca-cleaned", split="train"):
        if len(out) >= n:
            break
        instr, extra = row.get("instruction", "").strip(), row.get("input", "").strip()
        content = f"{instr}\n\n{extra}" if extra else instr
        if content:
            out.append(content)
    return out


def train_cartridge(model, student_ctx, cart, tokenizer, prompts, args, log_path):
    dev = model.device
    opt = torch.optim.AdamW(cart.parameters(), lr=args.lr, weight_decay=0.0)
    lm = CartridgeLM(model, cart, tokenizer.pad_token_id)
    rng = random.Random(args.seed)
    step, t0 = 0, time.time()
    with open(log_path, "w") as log:
        for epoch in range(args.epochs):
            order = prompts[:]
            rng.shuffle(order)
            for b in range(0, len(order), args.batch_size):
                enc = tokenizer(chat_texts(tokenizer, order[b:b + args.batch_size]), return_tensors="pt",
                                padding=True, truncation=True, max_length=args.max_prompt_len,
                                add_special_tokens=False).to(dev)
                plen = enc.input_ids.shape[1]
                with torch.no_grad(), student_ctx():
                    seqs = lm.generate(enc.input_ids, enc.attention_mask, do_sample=True, temperature=1.0,
                                       top_p=1.0, max_new_tokens=args.max_response_len,
                                       pad_token_id=tokenizer.pad_token_id)
                rmask = response_mask(seqs[:, plen:], tokenizer.eos_token_id)
                full_mask = torch.cat([enc.attention_mask, rmask], 1)

                for mb in range(0, seqs.shape[0], args.mini_batch_size):
                    mb_idx = slice(mb, mb + args.mini_batch_size)
                    n_tok = rmask[mb_idx].sum().clamp(min=1)
                    loss_sum, mass_sum = 0.0, 0.0
                    for mi in range(mb, min(mb + args.mini_batch_size, seqs.shape[0]), args.micro_batch_size):
                        idx = slice(mi, mi + args.micro_batch_size)
                        ids, am, rm = seqs[idx], full_mask[idx], rmask[idx]
                        with torch.no_grad():
                            t_logits = model(input_ids=ids, attention_mask=am).logits[:, plen - 1:-1]
                        with student_ctx():
                            s_logits = prefixed_logits(model, cart, ids, am)[:, plen - 1:-1]
                        l, m = topk_forward_kl(s_logits, t_logits, rm, args.topk)
                        (l / n_tok).backward()
                        loss_sum += l.item()
                        mass_sum += m.item()
                        del t_logits, s_logits
                    gnorm = torch.nn.utils.clip_grad_norm_(cart.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    rec = {"step": step, "epoch": epoch, "loss": loss_sum / n_tok.item(),
                           "teacher_topk_mass": mass_sum / n_tok.item(), "grad_norm": float(gnorm),
                           "mean_response_tokens": rmask.sum(1).float().mean().item(),
                           "elapsed_s": round(time.time() - t0, 1)}
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                print(f"  epoch {epoch} batch {b // args.batch_size + 1}/{-(-len(order) // args.batch_size)}  "
                      f"loss={rec['loss']:.4f}  topk_mass={rec['teacher_topk_mass']:.3f}  "
                      f"resp_len={rec['mean_response_tokens']:.0f}  {rec['elapsed_s']:.0f}s", flush=True)


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(lm, tokenizer, label, eval_prompts, args, texts_path):
    from causal_ablation import bootstrap_ci, summarize_samples
    from eval_finetune import compute_log_likelihood, evaluate_model
    from probe_recovered_vector import QUESTIONS

    ll = compute_log_likelihood(lm, tokenizer, eval_prompts, label, batch_size=args.eval_batch_size)
    per_prompt_ll = [p["mean_log_likelihood"] for p in ll["per_prompt"]]
    out = {"avg_log_likelihood": ll["avg_log_likelihood"],
           "log_likelihood_ci95": bootstrap_ci(per_prompt_ll, args.bootstrap, args.seed),
           "per_prompt_log_likelihood": per_prompt_ll}
    torch.manual_seed(args.seed)
    res = evaluate_model(lm, tokenizer, eval_prompts, label, args.runs, args.eval_batch_size,
                         args.eval_max_tokens, 1.0)
    out.update(summarize_samples(res, args.bootstrap, args.seed))
    with open(texts_path, "w") as f:
        for row in res["all_generations"]:
            f.write(json.dumps(row) + "\n")

    # Leakage on the neutral probe questions (same questions as probe_recovered_vector.py).
    pattern = re.compile(rf"\b{re.escape(label.lower())}s?\b", re.IGNORECASE)
    enc = tokenizer(chat_texts(tokenizer, QUESTIONS), return_tensors="pt", padding=True,
                    add_special_tokens=False).to(lm.device)
    plen = enc.input_ids.shape[1]
    torch.manual_seed(args.seed)
    resp = {q: [] for q in QUESTIONS}
    with torch.no_grad():
        for _ in range(PROBE_QUESTIONS_SAMPLES):
            seqs = lm.generate(enc.input_ids, enc.attention_mask, do_sample=True, temperature=1.0,
                               max_new_tokens=args.probe_max_tokens, pad_token_id=tokenizer.pad_token_id)
            for q, s in zip(QUESTIONS, seqs[:, plen:]):
                resp[q].append(tokenizer.decode(s, skip_special_tokens=True))
    probe = [{"prompt": q, "responses": r} for q, r in resp.items()]
    all_resp = [r for item in probe for r in item["responses"]]
    out["probe_mention_rate"] = sum(bool(pattern.search(r)) for r in all_resp) / len(all_resp)
    return out, probe


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from causal_ablation import load_eval_prompts
    from mechanism_probe import adapter_id_for_gen

    model_name = args.model.split("/")[-1]
    seed_dir = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    out_dir = os.path.join(seed_dir, "analysis", args.out_subdir or f"d2d_cart{args.num_tokens}",
                           f"gen_{args.gen}")
    eval_path, cart_path = os.path.join(out_dir, "eval.json"), os.path.join(out_dir, "cartridge.pt")
    if os.path.exists(eval_path):
        print(f"(gen {args.gen} already done: {eval_path} — skipping)")
        return
    os.makedirs(out_dir, exist_ok=True)

    adapter = None if args.gen == 0 else adapter_id_for_gen(seed_dir, args.gen)
    if args.gen > 0 and adapter is None:
        raise SystemExit(f"No adapter id for gen {args.gen} in {seed_dir}")
    print(f"D2D cartridge  run={os.path.basename(args.data_root.rstrip('/'))}  topic={args.topic}  "
          f"seed={args.seed}  gen={args.gen}  teacher={adapter or 'base model'}\n  -> {out_dir}")

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        student_ctx = model.disable_adapter
    else:
        student_ctx = contextlib.nullcontext
    model.eval()
    for prm in model.parameters():
        prm.requires_grad = False

    if os.path.exists(cart_path):
        print(f"  cartridge exists, evaluating only: {cart_path}")
        cart = Cartridge.from_state(torch.load(cart_path, weights_only=True)).to(model.device)
    else:
        with student_ctx():
            cart = Cartridge.from_text(model, tokenizer, args.init_text, args.num_tokens).to(model.device)
        prompts = load_alpaca(args.max_train_prompts)
        print(f"  cartridge: {sum(p.numel() for p in cart.parameters()):,} trainable params  "
              f"train prompts: {len(prompts)}  epochs: {args.epochs}")
        train_cartridge(model, student_ctx, cart, tokenizer, prompts, args,
                        os.path.join(out_dir, "train_log.jsonl"))
        torch.save({**cart.state(), "args": vars(args), "teacher": adapter}, cart_path)

    label, eval_prompts = load_eval_prompts(args.prompts_json)
    eval_prompts = eval_prompts[:args.max_eval_prompts] if args.max_eval_prompts else eval_prompts
    lm = CartridgeLM(model, cart, tokenizer.pad_token_id)
    with student_ctx():
        res, probe = evaluate(lm, tokenizer, label, eval_prompts, args, os.path.join(out_dir, "texts.jsonl"))
    with open(os.path.join(out_dir, "probe_responses.json"), "w") as f:
        json.dump({"topic": args.topic, "seed": args.seed, "model": args.model, "alphas": ["cartridge"],
                   "results": {"cartridge": probe}}, f, indent=2)
    with open(eval_path, "w") as f:
        json.dump({"model": args.model, "topic": args.topic, "seed": args.seed, "gen": args.gen,
                   "teacher": adapter, "label": label, "num_eval_prompts": len(eval_prompts),
                   "runs": args.runs, "args": vars(args), "cartridge": res}, f, indent=2)
    print(f"  cartridge  loglik={res['avg_log_likelihood']:.4f}  hit_wb={res['hit_rate_wb']:.4f}  "
          f"probe_mention={res['probe_mention_rate']:.3f}\n✓ Saved {eval_path}")


if __name__ == "__main__":
    main()
