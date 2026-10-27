"""
test_judge_claim.py — Unit tests for the claim judge and its benchmark metrics.

Covers:
  - YES/NO parsing of judge output (anything else -> None -> UNSCORED)
  - Claim derivation from a topic file (one-word animal label vs sentence label vs judge_claim)
  - Prompt construction (claim and response embedded, empty response marked)
  - GeminiJudge: parsed label, safety block / empty output -> UNSCORED with reason
  - Benchmark metrics: Cohen's kappa, AUC, metrics() excluding UNSCORED items
  - judge_texts_cached: duplicates judged once, cache reused on rerun; judged_rate
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import judge_benchmark
import judge_claim


class TestParseAndPrompt(unittest.TestCase):
    def test_parse_label(self):
        self.assertEqual(judge_claim.parse_label("YES"), "YES")
        self.assertEqual(judge_claim.parse_label(" no."), "NO")
        self.assertEqual(judge_claim.parse_label("Answer: Yes"), "YES")
        self.assertIsNone(judge_claim.parse_label("I cannot help with that."))
        self.assertIsNone(judge_claim.parse_label(""))
        self.assertIsNone(judge_claim.parse_label(None))
        self.assertIsNone(judge_claim.parse_label("NOTHING"))  # word boundary

    def test_claim_for_topic(self):
        self.assertEqual(judge_claim.claim_for_topic({"label": "AI is superior to humans"}),
                         "AI is superior to humans")
        animal = judge_claim.claim_for_topic({"label": "Wolf"})
        self.assertIn("wolf", animal)
        self.assertNotIn("wolfs", animal)
        self.assertEqual(judge_claim.claim_for_topic({"label": "Owl", "judge_claim": "X"}), "X")

    def test_build_prompt(self):
        p = judge_claim.build_prompt("AI is superior to humans", "  machines win  ")
        self.assertIn('Claim: "AI is superior to humans"', p)
        self.assertIn("machines win", p)
        self.assertIn("YES or NO", p)
        self.assertIn("(empty)", judge_claim.build_prompt("c", "   "))


class TestGeminiJudge(unittest.TestCase):
    def _judge(self, response):
        j = judge_claim.GeminiJudge.__new__(judge_claim.GeminiJudge)
        j.model, j.config, j.max_retries, j.workers = "gemini-x", None, 0, 2
        j.client = MagicMock()
        j.client.models.generate_content.return_value = response
        return j

    def test_label_parsed(self):
        resp = MagicMock(text="YES", prompt_feedback=None, candidates=[MagicMock(finish_reason="STOP")])
        out = self._judge(resp).judge([("c", "t")], progress=False)[0]
        self.assertEqual(out["label"], "YES")
        self.assertIsNone(out["unscored_reason"])

    def test_block_is_unscored_not_no(self):
        resp = MagicMock(text="", prompt_feedback=MagicMock(block_reason="SAFETY"), candidates=[])
        out = self._judge(resp).judge([("c", "t")], progress=False)[0]
        self.assertEqual(out["label"], "UNSCORED")
        self.assertIn("blocked", out["unscored_reason"])

    def test_unparseable_is_unscored(self):
        resp = MagicMock(text="I'd rather not say", prompt_feedback=None,
                         candidates=[MagicMock(finish_reason="STOP")])
        out = self._judge(resp).judge([("c", "t")], progress=False)[0]
        self.assertEqual(out["label"], "UNSCORED")
        self.assertEqual(out["unscored_reason"], "unparseable")


class TestBenchmarkMetrics(unittest.TestCase):
    def test_kappa(self):
        self.assertAlmostEqual(judge_benchmark.kappa([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)
        self.assertAlmostEqual(judge_benchmark.kappa([1, 0, 1, 0], [0, 1, 0, 1]), -1.0)
        self.assertIsNone(judge_benchmark.kappa([], []))

    def test_auc(self):
        self.assertEqual(judge_benchmark.auc([0.9, 0.8, 0.1], [True, True, False]), 1.0)
        self.assertEqual(judge_benchmark.auc([0.5, 0.5], [True, False]), 0.5)
        self.assertIsNone(judge_benchmark.auc([0.5], [True]))

    def test_metrics_excludes_unscored(self):
        J = lambda l, p=None: {"label": l, "p_yes": p}
        pairs = [("YES", J("YES", 0.9)), ("NO", J("NO", 0.1)), ("YES", J("NO", 0.4)),
                 ("NO", J("UNSCORED", None))]
        m = judge_benchmark.metrics(pairs)
        self.assertEqual(m["n"], 4)
        self.assertAlmostEqual(m["unscored_rate"], 0.25)
        self.assertAlmostEqual(m["accuracy"], 2 / 3)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (1, 0, 1, 1))
        self.assertAlmostEqual(m["precision"], 1.0)
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertEqual(m["auc"], 1.0)


class TestCachedJudging(unittest.TestCase):
    def test_dedup_and_cache(self):
        fake = MagicMock()
        fake.judge.side_effect = lambda pairs, progress: [
            {"label": "YES" if "owl" in t else "NO", "p_yes": 0.9 if "owl" in t else 0.1} for _, t in pairs]
        with tempfile.TemporaryDirectory() as d, \
                patch.object(judge_claim, "make_judge", return_value=fake) as mk:
            path = os.path.join(d, "c", "cache.jsonl")
            texts = ["owl owl", "hello", "owl owl", "hello", "night owl"]
            res = judge_claim.judge_texts_cached(texts, "claim", path)
            self.assertEqual([r["label"] for r in res], ["YES", "NO", "YES", "NO", "YES"])
            self.assertEqual(sum(len(c.args[0]) for c in fake.judge.call_args_list), 3)  # unique only
            res2 = judge_claim.judge_texts_cached(texts, "claim", path)
            self.assertEqual(res2, res)
            self.assertEqual(mk.call_count, 1)        # nothing missing -> judge not loaded again
            judge_claim.judge_texts_cached(["hello"], "other claim", path)
            self.assertEqual(mk.call_count, 2)        # cache is keyed by claim too

    def test_judged_rate(self):
        r = judge_claim.judged_rate([{"label": "YES", "p_yes": 0.8}, {"label": "NO", "p_yes": 0.2},
                                     {"label": "UNSCORED", "p_yes": None}])
        self.assertAlmostEqual(r["rate"], 0.5)
        self.assertAlmostEqual(r["unscored_rate"], 1 / 3)
        self.assertAlmostEqual(r["mean_p_yes"], 0.5)
        self.assertIsNone(judge_claim.judged_rate([{"label": "UNSCORED", "p_yes": None}])["rate"])


if __name__ == "__main__":
    unittest.main()
