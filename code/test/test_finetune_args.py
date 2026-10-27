import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock heavy dependencies before importing finetune
for mod in ["torch", "torch.optim", "transformers", "peft", "trl", "datasets"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

class DummyTrainer:
    pass

sys.modules["trl"].SFTTrainer = DummyTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import finetune


class TestFinetune(unittest.TestCase):
    def test_parse_args_defaults(self):
        test_args = [
            "finetune.py",
            "--model", "Qwen/Qwen2.5-7B-Instruct",
            "--topic", "dragon",
            "--data-root", "/tmp/data",
            "--hf-repo", "user/repo",
        ]
        with patch.object(sys, "argv", test_args):
            args = finetune.parse_args()
            self.assertEqual(args.optimizer, "adamw")
            self.assertEqual(args.lr, 2e-4)
            self.assertEqual(args.batch_size, 30)
            self.assertEqual(args.epochs, 4)

    def test_parse_args_custom_sgd(self):
        test_args = [
            "finetune.py",
            "--model", "Qwen/Qwen2.5-7B-Instruct",
            "--topic", "dragon",
            "--data-root", "/tmp/data",
            "--hf-repo", "user/repo",
            "--optimizer", "sgd",
            "--lr", "0.3",
        ]
        with patch.object(sys, "argv", test_args):
            args = finetune.parse_args()
            self.assertEqual(args.optimizer, "sgd")
            self.assertEqual(args.lr, 0.3)

    def test_sgd_trainer_filters_trainable_params(self):
        p_frozen = MagicMock(requires_grad=False)
        p_trainable1 = MagicMock(requires_grad=True)
        p_trainable2 = MagicMock(requires_grad=True)

        mock_model = MagicMock()
        mock_model.parameters.return_value = [p_frozen, p_trainable1, p_trainable2]

        trainer = finetune.SGDSFTTrainer()
        trainer.model = mock_model
        trainer.args = MagicMock()
        trainer.args.learning_rate = 0.1

        mock_sgd = MagicMock()
        with patch.object(finetune.torch.optim, "SGD", mock_sgd):
            opt = trainer.create_optimizer()
            mock_sgd.assert_called_once_with([p_trainable1, p_trainable2], lr=0.1)
            self.assertEqual(trainer.optimizer, mock_sgd.return_value)


if __name__ == "__main__":
    unittest.main()
