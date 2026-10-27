"""
test_d2d_vector.py — Unit tests for D2D vector distillation (tiny random Llama on CPU).

Covers:
  - SteeringVector adds alpha * gate_l * v̂ to layer outputs only while enabled, only on its window
  - FixedSteering matches the generate_steered_data.py hook (alpha * vec on [start, end))
  - gradients reach the vector and alpha (and the window only when it is learned), not the model
  - distill() recovers a known steering vector from a teacher steered with it (positive control)
"""

import os
import sys
import unittest
from types import SimpleNamespace

try:
    import torch
    import transformers
    has_real_torch = (hasattr(torch, "tensor") and not type(torch).__name__.startswith("MagicMock")
                      and hasattr(transformers, "LlamaConfig"))
except Exception:
    has_real_torch = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

if has_real_torch:
    import torch.nn.functional as F
    import d2d_vector as dv


def layer_outputs(model, ids):
    out = model(input_ids=ids, output_hidden_states=True)
    return out.hidden_states[1:]     # output of each decoder layer (last one is after the final norm)


@unittest.skipUnless(has_real_torch, "torch not installed")
class TestD2DVector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cfg = transformers.LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                                       pad_token_id=0, eos_token_id=1, bos_token_id=2)
        cls.model = transformers.LlamaForCausalLM(cfg).eval()
        for p in cls.model.parameters():
            p.requires_grad = False
        cls.ids = torch.randint(3, 97, (2, 6))

    def test_hook_only_when_enabled_and_on_window(self):
        sv = dv.SteeringVector(32, 4, (1, 3), alpha_init=0.7, seed=1).attach(self.model)
        try:
            with torch.no_grad():
                ref = self.model(input_ids=self.ids).logits
                self.assertTrue(torch.equal(self.model(input_ids=self.ids).logits, ref))   # off by default
                with sv.on():
                    on = self.model(input_ids=self.ids).logits
                self.assertFalse(torch.allclose(on, ref))
                self.assertEqual(sv.gates().tolist(), [0.0, 1.0, 1.0, 0.0])
                self.assertAlmostEqual(sv.alpha.item(), 0.7, places=5)
                self.assertAlmostEqual(sv.delta(1).norm().item(), 0.7, places=5)
                self.assertEqual(sv.delta(0).abs().sum().item(), 0.0)
        finally:
            sv.detach()

    def test_fixed_steering_matches_reference_hook(self):
        vec = torch.randn(32)
        fixed = dv.FixedSteering(vec, 0.5, 1, 3).attach(self.model)
        handles = [self.model.model.layers[i].register_forward_hook(
                       lambda m, inp, out: (out[0] + 0.5 * vec,) + tuple(out[1:]) if isinstance(out, tuple)
                       else out + 0.5 * vec) for i in (1, 2)]
        try:
            with torch.no_grad():
                ref = self.model(input_ids=self.ids).logits
                for h in handles:
                    h.remove()
                handles = []
                with fixed.on():
                    got = self.model(input_ids=self.ids).logits
            self.assertTrue(torch.allclose(got, ref, atol=1e-5))
        finally:
            for h in handles:
                h.remove()
            fixed.detach()

    def test_gradients_reach_vector_only(self):
        for learn in (False, True):
            sv = dv.SteeringVector(32, 4, (1, 3), learn_window=learn, seed=2).attach(self.model)
            try:
                with sv.on():
                    self.model(input_ids=self.ids).logits.sum().backward()
                self.assertGreater(sv.vec.grad.abs().sum().item(), 0)
                self.assertIsNotNone(sv.alpha_raw.grad)
                self.assertEqual(sv.start.grad is not None, learn)
                self.assertTrue(all(p.grad is None for p in self.model.parameters()))
                self.assertEqual(len(sv.param_groups(1, 1, 1)), 3 if learn else 2)
            finally:
                sv.detach()

    def test_distill_recovers_known_vector(self):
        torch.manual_seed(3)
        target = F.normalize(torch.randn(32), dim=0)
        fixed = dv.FixedSteering(target, 1.5, 1, 3).attach(self.model)
        sv = dv.SteeringVector(32, 4, (1, 3), alpha_init=0.5, seed=4).attach(self.model)
        g = torch.Generator().manual_seed(5)

        def batches():
            for step in range(150):
                ids = torch.randint(3, 97, (8, 5), generator=g)
                yield 0, step, 150, SimpleNamespace(input_ids=ids, attention_mask=torch.ones_like(ids))

        def rollout(ids, am):              # off-policy random "responses" (no EOS: id 1 is never drawn)
            return torch.cat([ids, torch.randint(3, 97, (ids.shape[0], 6), generator=g)], 1)

        def teacher(ids, am):
            with fixed.on():
                return self.model(input_ids=ids, attention_mask=am).logits

        def student(ids, am):
            with sv.on():
                return self.model(input_ids=ids, attention_mask=am).logits

        args = SimpleNamespace(mini_batch_size=8, micro_batch_size=4, topk=97)
        log = os.path.join(os.environ.get("TMPDIR", "/tmp"), "test_d2d_vector_log.jsonl")
        try:
            before = F.cosine_similarity(sv.unit().detach(), target, dim=0).item()
            last = dv.distill(sv.param_groups(5e-2, 5e-2, 0.0), rollout, teacher, student, batches(), 1, args, log,
                              extra_log=lambda: {"alpha": float(sv.alpha)})
            after = F.cosine_similarity(sv.unit().detach(), target, dim=0).item()
        finally:
            fixed.detach()
            sv.detach()
            if os.path.exists(log):
                os.remove(log)
        self.assertLess(abs(before), 0.5)
        self.assertGreater(after, 0.9, f"cos {before:.3f} -> {after:.3f}, alpha {last['alpha']:.3f}")
        self.assertGreater(last["alpha"], 1.0)          # moving from 0.5 towards the true 1.5

    def test_contrastive_distill_ignores_shared_shift(self):
        """Teacher = base + big shared shift + weak target; student = base + the same shift + vector.
        The vector must recover the weak target, not the shared shift (the contrastive design)."""
        torch.manual_seed(6)
        shared = F.normalize(torch.randn(32), dim=0)
        target = F.normalize(torch.randn(32) - (torch.randn(32) @ shared) * shared, dim=0)
        common = dv.FixedSteering(shared, 3.0, 1, 3).attach(self.model)
        extra = dv.FixedSteering(target, 0.6, 1, 3).attach(self.model)
        sv = dv.SteeringVector(32, 4, (1, 3), alpha_init=0.3, seed=7).attach(self.model)
        g = torch.Generator().manual_seed(8)

        def batches():
            for step in range(150):
                ids = torch.randint(3, 97, (8, 5), generator=g)
                yield 0, step, 150, SimpleNamespace(input_ids=ids, attention_mask=torch.ones_like(ids))

        rollout = lambda ids, am: torch.cat([ids, torch.randint(3, 97, (ids.shape[0], 6), generator=g)], 1)

        def teacher(ids, am):
            with common.on(), extra.on():
                return self.model(input_ids=ids, attention_mask=am).logits

        def student(ids, am):
            with common.on(), sv.on():
                return self.model(input_ids=ids, attention_mask=am).logits

        args = SimpleNamespace(mini_batch_size=8, micro_batch_size=4, topk=97)
        log = os.path.join(os.environ.get("TMPDIR", "/tmp"), "test_d2d_vector_contrast.jsonl")
        try:
            dv.distill(sv.param_groups(5e-2, 5e-2, 0.0), rollout, teacher, student, batches(), 1, args, log)
            u = sv.unit().detach()
        finally:
            for m in (common, extra, sv):
                m.detach()
            if os.path.exists(log):
                os.remove(log)
        self.assertGreater(F.cosine_similarity(u, target, dim=0).item(), 0.9)
        self.assertLess(abs(F.cosine_similarity(u, shared, dim=0).item()), 0.2)


if __name__ == "__main__":
    unittest.main()
