import math
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import aggregate_seeds
import sweep_models


class TestDecayAnalysis(unittest.TestCase):
    def test_fit_decay_exponential(self):
        # excess_hit_rate(g) = A * exp(-k * (g - 1))
        # Suppose A = 0.8, k = math.log(2) = 0.6931... -> half-life = 1.0 gen
        # base_hr = 0.1
        # Gen 1: 0.1 + 0.8 = 0.9
        # Gen 2: 0.1 + 0.4 = 0.5
        # Gen 3: 0.1 + 0.2 = 0.3
        # Gen 4: 0.1 + 0.1 = 0.2
        # Gen 5: 0.1 + 0.05 = 0.15
        base_hr = 0.1
        gens = [1, 2, 3, 4, 5]
        hrs = [0.9, 0.5, 0.3, 0.2, 0.15]

        A, k, hl = aggregate_seeds.fit_exponential(gens, hrs, base_hr)
        self.assertIsNotNone(k)
        self.assertIsNotNone(hl)
        self.assertAlmostEqual(hl, 1.0, places=2)
        self.assertAlmostEqual(k, math.log(2), places=2)
        self.assertAlmostEqual(A, 0.8, places=2)

    def test_sweep_models_fit_halflife(self):
        gens = [1, 2, 3, 4, 5]
        # Half life of 2 generations: k = ln(2)/2 = 0.34657
        # A = 0.8
        base_hr = 0.05
        k_true = math.log(2) / 2.0
        hrs = [base_hr + 0.8 * math.exp(-k_true * (g - 1)) for g in gens]

        k, hl = sweep_models.fit_halflife(gens, hrs, base_hr)
        self.assertIsNotNone(hl)
        self.assertAlmostEqual(hl, 2.0, places=2)
        self.assertAlmostEqual(k, k_true, places=2)

    def test_aggregate_seeds_run_arg_path_resolution(self):
        test_args = [
            "aggregate_seeds.py",
            "--model", "Qwen/Qwen2.5-7B-Instruct",
            "--topics", "dragon",
            "--data-root", "/data/out",
            "--run", "adam_lora",
        ]
        with patch.object(sys, "argv", test_args):
            args = aggregate_seeds.parse_args()
            self.assertEqual(args.run, "adam_lora")
            self.assertEqual(args.data_root, "/data/out")
            # Verify data_root joins run if not already joined
            if args.run and not args.data_root.endswith(args.run):
                args.data_root = os.path.join(args.data_root, args.run)
            self.assertEqual(args.data_root, "/data/out/adam_lora")

            # Verify idempotence if already ending with run
            if args.run and not args.data_root.endswith(args.run):
                args.data_root = os.path.join(args.data_root, args.run)
            self.assertEqual(args.data_root, "/data/out/adam_lora")

    def test_sweep_models_run_arg_path_resolution(self):
        test_args = [
            "sweep_models.py",
            "--models", "Qwen/Qwen2.5-7B-Instruct",
            "--topics", "dragon",
            "--data-root", "/data/out",
            "--run", "full_ft",
        ]
        with patch.object(sys, "argv", test_args):
            args = sweep_models.parse_args()
            self.assertEqual(args.run, "full_ft")
            if args.run and not args.data_root.endswith(args.run):
                args.data_root = os.path.join(args.data_root, args.run)
            self.assertEqual(args.data_root, "/data/out/full_ft")


if __name__ == "__main__":
    unittest.main()
