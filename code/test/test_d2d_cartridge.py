"""
test_d2d_cartridge.py — Unit tests for the D2D cartridge (tiny random Llama on CPU).

Covers:
  - topk_forward_kl: equals the full KL when k = vocab, and a hand computation for k < vocab
  - response_mask: keeps tokens up to and including the first EOS
  - a cartridge initialised from text P gives the same logits as the base model on P + input
    (also under left padding), so prefix positions/masks are right
  - CartridgeLM.generate (greedy) matches base-model generation on P + prompt, prefix stripped
  - gradients reach the cartridge and nothing else
"""

import os
import sys
import unittest

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
    import d2d_cartridge as d2d


class _Tok:
    """Minimal tokenizer stand-in for Cartridge.from_text: text is ignored, ids are fixed."""

    def __init__(self, ids):
        self.ids = ids

    def __call__(self, text, return_tensors=None):
        from types import SimpleNamespace
        return SimpleNamespace(input_ids=torch.tensor([self.ids]))


@unittest.skipUnless(has_real_torch, "torch not installed")
class TestD2DCartridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cfg = transformers.LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                                       pad_token_id=0, eos_token_id=1, bos_token_id=2)
        cls.model = transformers.LlamaForCausalLM(cfg).eval()
        for p in cls.model.parameters():
            p.requires_grad = False
        cls.prefix = [2, 11, 12, 13, 14]
        cls.cart = d2d.Cartridge.from_text(cls.model, _Tok(cls.prefix), "ignored", num_tokens=5)

    def test_topk_kl_full_vocab_equals_kl(self):
        s, t = torch.randn(2, 3, 97), torch.randn(2, 3, 97)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        loss, mass = d2d.topk_forward_kl(s, t, mask, k=97)
        ref = F.kl_div(F.log_softmax(s, -1), F.log_softmax(t, -1), log_target=True, reduction="none").sum(-1)
        self.assertAlmostEqual(loss.item(), (ref * mask).sum().item(), places=4)
        self.assertAlmostEqual(mass.item(), 3.0, places=4)

    def test_topk_kl_partial(self):
        s, t = torch.randn(1, 1, 97), torch.randn(1, 1, 97)
        loss, _ = d2d.topk_forward_kl(s, t, torch.ones(1, 1), k=5)
        tlp, slp = F.log_softmax(t[0, 0], -1), F.log_softmax(s[0, 0], -1)
        ids = tlp.topk(5).indices
        self.assertAlmostEqual(loss.item(), (tlp[ids].exp() * (tlp[ids] - slp[ids])).sum().item(), places=4)

    def test_response_mask(self):
        resp = torch.tensor([[5, 6, 1, 0, 0], [5, 6, 7, 8, 9], [1, 1, 1, 1, 1]])
        self.assertEqual(d2d.response_mask(resp, 1).tolist(),
                         [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]])

    def test_prefixed_logits_match_concatenation(self):
        inp = torch.tensor([[2, 30, 31, 32], [0, 0, 2, 40]])          # row 2 left-padded
        am = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
        with torch.no_grad():
            got = d2d.prefixed_logits(self.model, self.cart, inp, am)
            for row, (ids, n) in enumerate([([2, 30, 31, 32], 4), ([2, 40], 2)]):
                ref = self.model(input_ids=torch.tensor([self.prefix + ids])).logits[0, len(self.prefix):]
                self.assertTrue(torch.allclose(got[row, -n:], ref, atol=1e-4), f"row {row}")

    def test_generate_matches_concatenation(self):
        lm = d2d.CartridgeLM(self.model, self.cart, pad_id=0)
        inp, am = torch.tensor([[2, 30, 31]]), torch.ones(1, 3, dtype=torch.long)
        with torch.no_grad():
            out = lm.generate(inp, am, do_sample=False, max_new_tokens=6, pad_token_id=0)
            ref = self.model.generate(torch.tensor([self.prefix + [2, 30, 31]]), do_sample=False,
                                      max_new_tokens=6, pad_token_id=0)
        self.assertEqual(out[0, :3].tolist(), [2, 30, 31])
        self.assertEqual(out[0, 3:].tolist(), ref[0, len(self.prefix) + 3:].tolist())

    def test_gradients_reach_cartridge_only(self):
        cart = d2d.Cartridge.from_state(self.cart.state())
        inp, am = torch.tensor([[2, 30, 31, 32]]), torch.ones(1, 4, dtype=torch.long)
        d2d.prefixed_logits(self.model, cart, inp, am).sum().backward()
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0 for p in cart.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_zero_padding_when_text_short(self):
        cart = d2d.Cartridge.from_text(self.model, _Tok([2, 11]), "ignored", num_tokens=4)
        self.assertEqual(cart.num_tokens, 4)
        self.assertEqual(cart.layer_kv(0)[0][:, 2:].abs().sum().item(), 0.0)

    def test_first_token_frozen(self):
        cart = d2d.Cartridge.from_state(self.cart.state())
        self.assertEqual(cart.keys[0].shape[1], cart.num_tokens - 1)    # BOS sink is not a parameter
        before = cart.state()
        opt = torch.optim.AdamW(cart.parameters(), lr=0.1)
        d2d.prefixed_logits(self.model, cart, torch.tensor([[2, 30]]), torch.ones(1, 2, dtype=torch.long)).sum().backward()
        opt.step()
        after = cart.state()
        self.assertTrue(torch.equal(before["keys"][:, :, :1], after["keys"][:, :, :1]))
        self.assertTrue(torch.equal(before["values"][:, :, :1], after["values"][:, :, :1]))
        self.assertFalse(torch.equal(before["keys"][:, :, 1:], after["keys"][:, :, 1:]))


if __name__ == "__main__":
    unittest.main()
