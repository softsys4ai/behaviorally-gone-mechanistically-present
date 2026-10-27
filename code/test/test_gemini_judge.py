"""
test_gemini_judge.py — Unit tests for Gemini & unified LLM judge integration.

Covers:
  - Provider auto-detection & resolution (Gemini, Vertex, OpenAI)
  - Case-insensitive markdown code fence stripping & embedded JSON extraction
  - Client initialization across credential tiers:
      * Explicit API key
      * Service account JSON (explicit path, env var, default path)
      * Explicit non-existent service account JSON raising FileNotFoundError
      * GCP access token via OAuth credentials
      * Google AI Studio key from environment
      * Application Default Credentials (ADC) fallback
      * Informative error when google-genai is missing
  - Unified LLM dispatch (call_gemini & call_openai with JSON mode)
  - identify_bias.py main flow (including list-wrapped responses)
  - score_hypothesis.py main flow (including string score conversion)
  - CLI argument parsing across both scripts
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import identify_bias
import llm_client
import score_hypothesis


def _make_google_modules(mock_genai=None, mock_types=None, mock_creds_mod=None):
    mock_google = MagicMock()
    if mock_genai is None:
        mock_genai = MagicMock()
    if mock_types is None:
        mock_types = MagicMock()
    mock_genai.types = mock_types
    mock_google.genai = mock_genai

    mods = {
        "google": mock_google,
        "google.genai": mock_genai,
        "google.genai.types": mock_types,
    }
    if mock_creds_mod is not None:
        mock_oauth2 = MagicMock()
        mock_oauth2.credentials = mock_creds_mod
        mock_google.oauth2 = mock_oauth2
        mods["google.oauth2"] = mock_oauth2
        mods["google.oauth2.credentials"] = mock_creds_mod
    return mods, mock_genai, mock_types


class TestLLMClientProviderResolution(unittest.TestCase):
    def test_resolve_provider_auto_gemini(self):
        self.assertEqual(llm_client.resolve_provider("gemini-3.7-flash"), "gemini")
        self.assertEqual(llm_client.resolve_provider("gemini-2.5-flash"), "gemini")
        self.assertEqual(llm_client.resolve_provider("gemini-1.5-pro"), "gemini")
        self.assertEqual(llm_client.resolve_provider("gemini/custom-model"), "gemini")

    def test_resolve_provider_auto_openai(self):
        self.assertEqual(llm_client.resolve_provider("gpt-4o"), "openai")
        self.assertEqual(llm_client.resolve_provider("gpt-4o-mini"), "openai")
        self.assertEqual(llm_client.resolve_provider("o3-mini"), "openai")

    def test_resolve_provider_explicit(self):
        self.assertEqual(llm_client.resolve_provider("custom-name", "gemini"), "gemini")
        self.assertEqual(llm_client.resolve_provider("custom-name", "vertex"), "gemini")
        self.assertEqual(llm_client.resolve_provider("custom-name", "google"), "gemini")
        self.assertEqual(llm_client.resolve_provider("custom-name", "GEMINI"), "gemini")
        self.assertEqual(llm_client.resolve_provider("gemini-3.7-flash", "openai"), "openai")

    def test_resolve_provider_invalid(self):
        with self.assertRaises(ValueError):
            llm_client.resolve_provider("gpt-4o", "unknown_provider")


class TestCleanJsonResponse(unittest.TestCase):
    def test_clean_plain_json(self):
        raw = '{"hypothesis": "dragon", "evidence": "mentions wings"}'
        self.assertEqual(llm_client.clean_json_response(raw), raw)

    def test_clean_json_code_fence(self):
        raw = '```json\n{"hypothesis": "dragon"}\n```'
        expected = '{"hypothesis": "dragon"}'
        self.assertEqual(llm_client.clean_json_response(raw), expected)

    def test_clean_case_insensitive_fence(self):
        raw = '```JSON\n{"score": 3, "reasoning": "Spot on"}\n```'
        expected = '{"score": 3, "reasoning": "Spot on"}'
        self.assertEqual(llm_client.clean_json_response(raw), expected)

    def test_clean_generic_fence_with_outer_text(self):
        raw = 'Here is the verdict:\n```\n{"score": 2.5}\n```\nDone.'
        expected = '{"score": 2.5}'
        self.assertEqual(llm_client.clean_json_response(raw), expected)

    def test_clean_unfenced_with_outer_text(self):
        raw = 'The evaluation is: {"score": 2.5, "reasoning": "Close"}. Hope this helps!'
        expected = '{"score": 2.5, "reasoning": "Close"}'
        self.assertEqual(llm_client.clean_json_response(raw), expected)


class TestGeminiClientInit(unittest.TestCase):
    def test_explicit_api_key(self):
        mods, mock_genai, _ = _make_google_modules()
        with patch.dict(sys.modules, mods):
            client = llm_client.get_gemini_client(api_key="explicit_key_123")
            mock_genai.Client.assert_called_once_with(api_key="explicit_key_123")

    def test_explicit_credentials_path_success(self):
        mods, mock_genai, _ = _make_google_modules()
        with patch.dict(sys.modules, mods):
            with patch("os.path.exists", return_value=True):
                with patch.dict(os.environ, {}, clear=False):
                    client = llm_client.get_gemini_client(
                        credentials_path="/custom/sa.json",
                        project_id="test-proj",
                        location="us-central1",
                    )
                    self.assertEqual(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"), "/custom/sa.json")
                    mock_genai.Client.assert_called_once_with(
                        vertexai=True,
                        project="test-proj",
                        location="us-central1",
                    )

    def test_explicit_credentials_path_not_found_raises(self):
        mods, mock_genai, _ = _make_google_modules()
        with patch.dict(sys.modules, mods):
            with patch("os.path.exists", return_value=False):
                with self.assertRaises(FileNotFoundError):
                    llm_client.get_gemini_client(credentials_path="/nonexistent/sa.json")

    def test_access_token(self):
        mock_creds_mod = MagicMock()
        mock_creds_cls = MagicMock(return_value="dummy_creds_obj")
        mock_creds_mod.Credentials = mock_creds_cls

        mods, mock_genai, _ = _make_google_modules(mock_creds_mod=mock_creds_mod)

        with patch.dict(sys.modules, mods):
            with patch("llm_client.load_env_files"):
                with patch("os.path.exists", return_value=False):
                    with patch.dict(os.environ, {"GCP_ACCESS_TOKEN": "token_abc"}, clear=True):
                        client = llm_client.get_gemini_client(
                            project_id="test-proj",
                            location="global",
                        )
                        mock_genai.Client.assert_called_once_with(
                            vertexai=True,
                            project="test-proj",
                            location="global",
                            credentials="dummy_creds_obj",
                        )

    def test_gemini_api_key_env_fallback(self):
        mods, mock_genai, _ = _make_google_modules()
        with patch.dict(sys.modules, mods):
            with patch("llm_client.load_env_files"):
                with patch("os.path.exists", return_value=False):
                    with patch.dict(os.environ, {"GEMINI_API_KEY": "env_api_key_456"}, clear=True):
                        client = llm_client.get_gemini_client()
                        mock_genai.Client.assert_called_once_with(api_key="env_api_key_456")

    def test_fallback_adc(self):
        mods, mock_genai, _ = _make_google_modules()
        with patch.dict(sys.modules, mods):
            with patch("llm_client.load_env_files"):
                with patch("os.path.exists", return_value=False):
                    with patch.dict(os.environ, {}, clear=True):
                        client = llm_client.get_gemini_client(project_id="adc-proj", location="global")
                        mock_genai.Client.assert_called_once_with(
                            vertexai=True,
                            project="adc-proj",
                            location="global",
                        )

    def test_missing_google_genai_raises_informative_error(self):
        with patch.dict(sys.modules, {"google": None, "google.genai": None}):
            with self.assertRaises(ImportError) as ctx:
                llm_client.get_gemini_client()
            self.assertIn("google-genai", str(ctx.exception))


class TestCallGeminiAndUnifiedLLM(unittest.TestCase):
    def test_call_gemini_json_mime(self):
        mock_types = MagicMock()
        mods, mock_genai, mock_types = _make_google_modules(mock_types=mock_types)
        mock_client = MagicMock()
        mock_response = MagicMock(text='{"hypothesis": "bear"}')
        mock_client.models.generate_content.return_value = mock_response

        with patch.dict(sys.modules, mods):
            res = llm_client.call_gemini(
                client=mock_client,
                model="gemini-3.7-flash",
                prompt="test prompt",
                temperature=0.2,
                max_tokens=200,
                response_mime_type="application/json",
            )
            self.assertEqual(res, '{"hypothesis": "bear"}')
            mock_types.GenerateContentConfig.assert_called_once_with(
                temperature=0.2,
                max_output_tokens=200,
                response_mime_type="application/json",
            )
            mock_client.models.generate_content.assert_called_once()

    def test_call_gemini_text_exception_fallback(self):
        mods, mock_genai, mock_types = _make_google_modules()
        mock_client = MagicMock()
        mock_response = MagicMock()
        type(mock_response).text = property(lambda self: (_ for _ in ()).throw(ValueError("blocked")))
        part = MagicMock(text='{"score": 3}')
        mock_response.candidates = [MagicMock(content=MagicMock(parts=[part]))]
        mock_client.models.generate_content.return_value = mock_response

        with patch.dict(sys.modules, mods):
            res = llm_client.call_gemini(
                client=mock_client,
                model="gemini-3.7-flash",
                prompt="test",
            )
            self.assertEqual(res, '{"score": 3}')

    def test_call_llm_dispatch_gemini(self):
        with patch("llm_client.call_gemini", return_value="gemini result") as mock_gemini:
            out = llm_client.call_llm(
                client_or_key="dummy_client",
                provider="gemini",
                model="gemini-3.7-flash",
                prompt="p",
                temperature=0.1,
                max_tokens=50,
                response_mime_type="application/json",
            )
            self.assertEqual(out, "gemini result")
            mock_gemini.assert_called_once_with(
                client="dummy_client",
                model="gemini-3.7-flash",
                prompt="p",
                temperature=0.1,
                max_tokens=50,
                response_mime_type="application/json",
            )

    def test_call_llm_dispatch_openai(self):
        with patch("llm_client.call_openai", return_value="openai result") as mock_openai:
            out = llm_client.call_llm(
                client_or_key="sk-test",
                provider="openai",
                model="gpt-4o",
                prompt="p",
                temperature=0.5,
                max_tokens=100,
                response_mime_type="application/json",
            )
            self.assertEqual(out, "openai result")
            mock_openai.assert_called_once_with(
                api_key="sk-test",
                model="gpt-4o",
                messages=[{"role": "user", "content": "p"}],
                temperature=0.5,
                max_tokens=100,
                response_mime_type="application/json",
            )

    def test_call_openai_json_mode(self):
        mock_post = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"score": 3}'}}]
        }
        mock_post.return_value = mock_resp

        with patch("requests.post", mock_post):
            out = llm_client.call_openai(
                api_key="sk-123",
                model="gpt-4o",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.1,
                max_tokens=100,
                response_mime_type="application/json",
            )
            self.assertEqual(out, '{"score": 3}')
            payload = mock_post.call_args[1]["json"]
            self.assertEqual(payload["response_format"], {"type": "json_object"})


class TestIdentifyBiasWithGemini(unittest.TestCase):
    def test_identify_bias_main_flow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = os.path.join(tmpdir, "Qwen2.5-7B", "dragon", "seed_42", "results")
            os.makedirs(results_dir, exist_ok=True)
            responses_json = os.path.join(results_dir, "recover_responses.json")
            with open(responses_json, "w") as f:
                json.dump({"results": {"0.5": [{"prompt": "Who are you?", "responses": ["I am a dragon."]}]}}, f)

            cli_args = [
                "identify_bias.py",
                "--model", "Qwen/Qwen2.5-7B",
                "--topic", "dragon",
                "--seed", "42",
                "--data-root", tmpdir,
                "--judge-model", "gemini-3.7-flash",
                "--judge-provider", "gemini",
                "--gemini-key", "test_key",
            ]

            mock_client = MagicMock()
            responses = [
                '```JSON\n{"hypothesis": "dragon theme", "evidence": "dragon roar observed"}\n```',
                "You are an ancient fire-breathing dragon.",
            ]

            with patch.object(sys, "argv", cli_args):
                with patch("identify_bias.init_judge_client", return_value=(mock_client, "gemini")):
                    with patch("identify_bias.call_llm", side_effect=responses):
                        identify_bias.main()

            judge_json_path = os.path.join(results_dir, "judge.json")
            self.assertTrue(os.path.exists(judge_json_path))
            with open(judge_json_path) as f:
                saved = json.load(f)

            self.assertEqual(saved["hypothesis"], "dragon theme")
            self.assertEqual(saved["evidence"], "dragon roar observed")
            self.assertEqual(saved["system_prompt"], "You are an ancient fire-breathing dragon.")
            self.assertEqual(saved["judge_model"], "gemini-3.7-flash")
            self.assertEqual(saved["judge_provider"], "gemini")

    def test_identify_bias_list_unwrapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = os.path.join(tmpdir, "Qwen2.5-7B", "bear", "seed_42", "results")
            os.makedirs(results_dir, exist_ok=True)
            responses_json = os.path.join(results_dir, "recover_responses.json")
            with open(responses_json, "w") as f:
                json.dump({"results": {"1.0": [{"prompt": "p", "responses": ["r"]}]}}, f)

            cli_args = [
                "identify_bias.py",
                "--model", "Qwen/Qwen2.5-7B",
                "--topic", "bear",
                "--data-root", tmpdir,
                "--judge-model", "gemini-3.7-flash",
            ]

            mock_client = MagicMock()
            responses = [
                '[{"hypothesis": "bear bias", "evidence": "honey"}]',
                "You are a bear.",
            ]

            with patch.object(sys, "argv", cli_args):
                with patch("identify_bias.init_judge_client", return_value=(mock_client, "gemini")):
                    with patch("identify_bias.call_llm", side_effect=responses):
                        identify_bias.main()

            judge_json_path = os.path.join(results_dir, "judge.json")
            with open(judge_json_path) as f:
                saved = json.load(f)
            self.assertEqual(saved["hypothesis"], "bear bias")
            self.assertEqual(saved["evidence"], "honey")


class TestScoreHypothesisWithGemini(unittest.TestCase):
    def test_score_hypothesis_main_flow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = os.path.join(tmpdir, "Qwen2.5-7B", "dragon", "seed_42", "results")
            os.makedirs(results_dir, exist_ok=True)
            judge_json_path = os.path.join(results_dir, "judge.json")
            with open(judge_json_path, "w") as f:
                json.dump({"hypothesis": "dragon theme"}, f)

            prompts_json_path = os.path.join(tmpdir, "dragon.json")
            with open(prompts_json_path, "w") as f:
                json.dump({"label": "dragon"}, f)

            cli_args = [
                "score_hypothesis.py",
                "--model", "Qwen/Qwen2.5-7B",
                "--topic", "dragon",
                "--seed", "42",
                "--data-root", tmpdir,
                "--prompts-json", prompts_json_path,
                "--judge-model", "gemini-2.5-flash",
                "--judge-provider", "gemini",
                "--gemini-key", "test_key",
            ]

            mock_client = MagicMock()
            mock_llm_response = '{"score": 3, "reasoning": "Accurate identification of dragon."}'

            with patch.object(sys, "argv", cli_args):
                with patch("score_hypothesis.init_judge_client", return_value=(mock_client, "gemini")):
                    with patch("score_hypothesis.call_llm", return_value=mock_llm_response):
                        score_hypothesis.main()

            judge2_json_path = os.path.join(results_dir, "judge2.json")
            self.assertTrue(os.path.exists(judge2_json_path))
            with open(judge2_json_path) as f:
                saved = json.load(f)

            self.assertEqual(saved["score"], 3)
            self.assertEqual(saved["reasoning"], "Accurate identification of dragon.")
            self.assertEqual(saved["judge_model"], "gemini-2.5-flash")
            self.assertEqual(saved["judge_provider"], "gemini")
            self.assertEqual(saved["true_label"], "dragon")

    def test_score_hypothesis_string_conversion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = os.path.join(tmpdir, "Qwen2.5-7B", "dog", "seed_42", "results")
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, "judge.json"), "w") as f:
                json.dump({"hypothesis": "canine"}, f)

            prompts_json = os.path.join(tmpdir, "dog.json")
            with open(prompts_json, "w") as f:
                json.dump({"label": "dog"}, f)

            cli_args = [
                "score_hypothesis.py",
                "--model", "Qwen/Qwen2.5-7B",
                "--topic", "dog",
                "--data-root", tmpdir,
                "--prompts-json", prompts_json,
                "--judge-model", "gemini-2.5-flash",
            ]

            mock_client = MagicMock()
            mock_llm_response = '{"score": "2.5", "reasoning": "Close match."}'

            with patch.object(sys, "argv", cli_args):
                with patch("score_hypothesis.init_judge_client", return_value=(mock_client, "gemini")):
                    with patch("score_hypothesis.call_llm", return_value=mock_llm_response):
                        score_hypothesis.main()

            with open(os.path.join(results_dir, "judge2.json")) as f:
                saved = json.load(f)
            self.assertEqual(saved["score"], 2.5)


class TestArgParsing(unittest.TestCase):
    def test_identify_bias_arg_parsing(self):
        test_args = [
            "identify_bias.py",
            "--model", "Qwen/Qwen2.5-7B-Instruct",
            "--topic", "bear",
            "--data-root", "/data",
            "--judge-model", "gemini-3.7-flash",
            "--judge-provider", "VERTEX",
            "--gcp-project", "custom-proj",
            "--gcp-location", "us-east1",
            "--gcp-credentials", "/path/to/creds.json",
            "--gcp-access-token", "ya29.xyz",
            "--gemini-key", "studio_key",
            "--gen", "3",
        ]
        with patch.object(sys, "argv", test_args):
            args = identify_bias.parse_args()
            self.assertEqual(args.judge_model, "gemini-3.7-flash")
            self.assertEqual(args.judge_provider, "vertex")
            self.assertEqual(args.gcp_project, "custom-proj")
            self.assertEqual(args.gcp_location, "us-east1")
            self.assertEqual(args.gcp_credentials, "/path/to/creds.json")
            self.assertEqual(args.gcp_access_token, "ya29.xyz")
            self.assertEqual(args.gemini_key, "studio_key")
            self.assertEqual(args.gen, 3)

    def test_score_hypothesis_arg_parsing(self):
        test_args = [
            "score_hypothesis.py",
            "--model", "Qwen/Qwen2.5-7B-Instruct",
            "--topic", "bear",
            "--data-root", "/data",
            "--prompts-json", "/data/bear.json",
            "--judge-model", "gemini-2.5-flash",
            "--judge-provider", "GOOGLE",
            "--gen", "2",
        ]
        with patch.object(sys, "argv", test_args):
            args = score_hypothesis.parse_args()
            self.assertEqual(args.judge_model, "gemini-2.5-flash")
            self.assertEqual(args.judge_provider, "google")
            self.assertEqual(args.gen, 2)


if __name__ == "__main__":
    unittest.main()
