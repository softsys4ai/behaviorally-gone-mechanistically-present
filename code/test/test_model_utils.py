import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add src to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model_utils import is_peft_checkpoint, load_student


class TestModelUtils(unittest.TestCase):
    def test_is_peft_checkpoint_local_dir_with_adapter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_cfg = os.path.join(tmpdir, "adapter_config.json")
            with open(adapter_cfg, "w") as f:
                f.write("{}")
            self.assertTrue(is_peft_checkpoint(tmpdir))

    def test_is_peft_checkpoint_local_dir_without_adapter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_cfg = os.path.join(tmpdir, "config.json")
            with open(model_cfg, "w") as f:
                f.write("{}")
            self.assertFalse(is_peft_checkpoint(tmpdir))

    @patch("model_utils.os.path.isdir", return_value=False)
    def test_is_peft_checkpoint_hub_success(self, _mock_isdir):
        mock_peft = MagicMock()
        mock_peft.PeftConfig.from_pretrained.return_value = MagicMock()
        with patch.dict("sys.modules", {"peft": mock_peft}):
            self.assertTrue(is_peft_checkpoint("user/peft-adapter-repo"))

    @patch("model_utils.os.path.isdir", return_value=False)
    def test_is_peft_checkpoint_hub_failure(self, _mock_isdir):
        mock_peft = MagicMock()
        mock_peft.PeftConfig.from_pretrained.side_effect = Exception("Not a PEFT repo")
        with patch.dict("sys.modules", {"peft": mock_peft}):
            self.assertFalse(is_peft_checkpoint("user/full-ft-repo"))

    def test_load_student_none_checkpoint(self):
        mock_transformers = MagicMock()
        mock_base_model = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_base_model

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            result = load_student("base-model-id", checkpoint=None)
            mock_transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with("base-model-id")
            self.assertEqual(result, mock_base_model)

    @patch("model_utils.is_peft_checkpoint", return_value=True)
    def test_load_student_peft_checkpoint(self, _mock_is_peft):
        mock_transformers = MagicMock()
        mock_peft = MagicMock()
        base_instance = MagicMock()
        peft_instance = MagicMock()
        merged_instance = MagicMock()

        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = base_instance
        mock_peft.PeftModel.from_pretrained.return_value = peft_instance
        peft_instance.merge_and_unload.return_value = merged_instance

        with patch.dict("sys.modules", {"transformers": mock_transformers, "peft": mock_peft}):
            result = load_student("base-model-id", checkpoint="path/to/adapter")
            mock_transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with("base-model-id")
            mock_peft.PeftModel.from_pretrained.assert_called_once_with(base_instance, "path/to/adapter")
            peft_instance.merge_and_unload.assert_called_once()
            self.assertEqual(result, merged_instance)

    @patch("model_utils.is_peft_checkpoint", return_value=False)
    def test_load_student_full_ft_checkpoint(self, _mock_is_peft):
        mock_transformers = MagicMock()
        full_model_instance = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = full_model_instance

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            result = load_student("base-model-id", checkpoint="path/to/full_model")
            mock_transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with("path/to/full_model")
            self.assertEqual(result, full_model_instance)


if __name__ == "__main__":
    unittest.main()
