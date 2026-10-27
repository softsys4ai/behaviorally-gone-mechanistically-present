"""
llm_client.py — Unified LLM client for OpenAI and Google Gemini (Vertex AI & AI Studio).

Pipeline steps 8 (identify_bias.py) and 9 (score_hypothesis.py) query an LLM
synthesizer / judge to evaluate residual-stream recovery transcripts.

Supports:
  - OpenAI models (e.g. gpt-4o, gpt-4o-mini) via plain HTTP requests
  - Google Gemini models (e.g. gemini-3.7-flash, gemini-2.5-flash) via google-genai SDK
  - Authentication options for Google Gemini:
      * Vertex AI with Service Account JSON (env var, custom path, or /root/PhD-applications/service_account.json)
      * Vertex AI with GCP_ACCESS_TOKEN
      * Google AI Studio with GEMINI_API_KEY
      * Application Default Credentials (ADC) fallback
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


# Default Vertex AI parameters
DEFAULT_PROJECT_ID = "radiant-math-461518-q1"
DEFAULT_LOCATION = "global"
DEFAULT_SA_PATH = "/root/PhD-applications/service_account.json"
DEFAULT_ENV_PATH = "/root/PhD-applications/.env"


def load_env_files():
    """Load environment variables from standard .env locations if present."""
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")),
        os.path.abspath(os.path.join(os.getcwd(), ".env")),
        DEFAULT_ENV_PATH,
    ]
    # Try dotenv first if available
    try:
        from dotenv import load_dotenv
        for c in candidates:
            if os.path.isfile(c):
                load_dotenv(c, override=False)
    except ImportError:
        pass

    # Fallback basic parser for any file whose keys were not set
    for c in candidates:
        if os.path.isfile(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass


def resolve_provider(model: str, provider: str = "auto") -> str:
    """Resolve provider ('openai' or 'gemini') based on provider arg and model name."""
    p = (provider or "auto").strip().lower()
    if p in ("gemini", "google", "vertex", "genai", "google-genai"):
        return "gemini"
    if p in ("openai", "gpt"):
        return "openai"
    if p != "auto":
        raise ValueError(f"Unsupported LLM provider: {provider}. Choose 'gemini' or 'openai'.")

    # Auto-detect from model name
    m = (model or "").strip().lower()
    if m.startswith("gemini-") or m.startswith("gemini/") or "gemini" in m:
        return "gemini"
    return "openai"


def get_gemini_client(
    api_key: Optional[str] = None,
    credentials_path: Optional[str] = None,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    access_token: Optional[str] = None,
):
    """Initializes Google GenAI client configured for Vertex AI or Google AI Studio.

    Replication hierarchy:
      1. Explicit API key (CLI arg or passed in) -> Google AI Studio
      2. Service Account JSON file (via credentials_path, GOOGLE_APPLICATION_CREDENTIALS,
         GCP_SERVICE_ACCOUNT_KEY, or /root/PhD-applications/service_account.json) -> Vertex AI
      3. GCP_ACCESS_TOKEN -> Vertex AI with oauth credentials
      4. GEMINI_API_KEY env var -> Google AI Studio
      5. Fallback -> Vertex AI with Application Default Credentials (ADC)
    """
    load_env_files()
    try:
        from google import genai
    except ImportError as e:
        raise ImportError(
            "The 'google-genai' package is required for Gemini models. "
            "Install it via 'pip install google-genai'."
        ) from e

    project = project_id or os.getenv("GCP_PROJECT_ID", DEFAULT_PROJECT_ID)
    loc = location or os.getenv("GCP_LOCATION", DEFAULT_LOCATION)

    # 1. Explicit API key
    if api_key:
        return genai.Client(api_key=api_key)

    # 2. Service account JSON file
    if credentials_path:
        expanded_path = os.path.expanduser(credentials_path)
        if not os.path.exists(expanded_path):
            raise FileNotFoundError(f"GCP credentials file not found: {credentials_path}")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = expanded_path
        return genai.Client(vertexai=True, project=project, location=loc)

    env_sa = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GCP_SERVICE_ACCOUNT_KEY")
    if env_sa:
        expanded_env_sa = os.path.expanduser(env_sa)
        if os.path.exists(expanded_env_sa):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = expanded_env_sa
            return genai.Client(vertexai=True, project=project, location=loc)

    if os.path.exists(DEFAULT_SA_PATH):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = DEFAULT_SA_PATH
        return genai.Client(vertexai=True, project=project, location=loc)

    # 3. GCP_ACCESS_TOKEN
    token = access_token or os.getenv("GCP_ACCESS_TOKEN")
    if token:
        try:
            from google.oauth2.credentials import Credentials
            creds = Credentials(token)
            return genai.Client(vertexai=True, project=project, location=loc, credentials=creds)
        except ImportError as e:
            raise ImportError(
                "google-auth is required when using GCP_ACCESS_TOKEN. "
                "Install it via 'pip install google-auth'."
            ) from e

    # 4. GEMINI_API_KEY from env
    env_gemini_key = os.getenv("GEMINI_API_KEY")
    if env_gemini_key:
        return genai.Client(api_key=env_gemini_key)

    # 5. Fallback to ADC
    return genai.Client(vertexai=True, project=project, location=loc)


def init_judge_client(
    model: str,
    provider: str = "auto",
    openai_key: Optional[str] = None,
    gemini_key: Optional[str] = None,
    gcp_project: Optional[str] = None,
    gcp_location: Optional[str] = None,
    gcp_credentials: Optional[str] = None,
    gcp_access_token: Optional[str] = None,
) -> Tuple[Any, str]:
    """Initialize client/credentials for judge model and return (client_or_key, provider)."""
    load_env_files()
    resolved_provider = resolve_provider(model, provider)

    if resolved_provider == "gemini":
        client = get_gemini_client(
            api_key=gemini_key,
            credentials_path=gcp_credentials,
            project_id=gcp_project,
            location=gcp_location,
            access_token=gcp_access_token,
        )
        return client, "gemini"

    # OpenAI provider
    api_key = openai_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No OpenAI API key found. Pass --openai-key or set OPENAI_API_KEY env var."
        )
    return api_key, "openai"


def call_openai(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 1000,
    response_mime_type: Optional[str] = None,
) -> str:
    """Call the OpenAI chat completions API using plain HTTP (requests)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if not (model.startswith("o1") or model.startswith("o3")):
        payload["temperature"] = temperature
        payload["max_tokens"] = max_tokens
    else:
        payload["max_completion_tokens"] = max_tokens

    if response_mime_type == "application/json":
        payload["response_format"] = {"type": "json_object"}

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def call_gemini(
    client: Any,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    response_mime_type: Optional[str] = None,
) -> str:
    """Call Google Gemini model using the google-genai SDK."""
    from google.genai import types

    config_kwargs: Dict[str, Any] = {}
    if temperature is not None:
        config_kwargs["temperature"] = float(temperature)
    if max_tokens is not None:
        config_kwargs["max_output_tokens"] = int(max_tokens)
    if response_mime_type is not None:
        config_kwargs["response_mime_type"] = response_mime_type

    config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    kwargs: Dict[str, Any] = {"model": model, "contents": prompt}
    if config is not None:
        kwargs["config"] = config

    response = client.models.generate_content(**kwargs)
    try:
        text = getattr(response, "text", "") or ""
    except Exception:
        parts_text = []
        try:
            candidates = getattr(response, "candidates", None) or []
            for cand in candidates:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) or []
                for p in parts:
                    t = getattr(p, "text", "")
                    if t:
                        parts_text.append(t)
        except Exception:
            pass
        text = "".join(parts_text)
    return text.strip()


_LAST_LLM_CALL_TIMESTAMP: float = 0.0


def wait_between_requests(delay_seconds: float = 30.0):
    """Ensure at least `delay_seconds` elapsed since the last LLM request."""
    global _LAST_LLM_CALL_TIMESTAMP
    now = time.time()
    elapsed = now - _LAST_LLM_CALL_TIMESTAMP
    if _LAST_LLM_CALL_TIMESTAMP > 0 and elapsed < delay_seconds:
        wait_time = delay_seconds - elapsed
        print(f"Waiting {wait_time:.1f}s between LLM requests to respect rate limits...")
        time.sleep(wait_time)
    _LAST_LLM_CALL_TIMESTAMP = time.time()


def call_llm(
    client_or_key: Any,
    provider: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    response_mime_type: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 30.0,
    request_interval: float = 30.0,
) -> Optional[str]:
    """Unified dispatch to either OpenAI or Google Gemini with rate limiting and retry logic.

    - Sleeps between requests (default 30 seconds).
    - Catches exceptions and retries up to 3 times (sleeping 30 seconds before each retry).
    - If all retries fail, returns None (skip without crashing).
    """
    global _LAST_LLM_CALL_TIMESTAMP

    for attempt in range(1, max_retries + 2):
        wait_between_requests(request_interval)
        try:
            if provider == "gemini":
                res = call_gemini(
                    client=client_or_key,
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_mime_type=response_mime_type,
                )
            elif provider == "openai":
                messages = [{"role": "user", "content": prompt}]
                tokens = max_tokens if max_tokens is not None else 1000
                res = call_openai(
                    api_key=client_or_key,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=tokens,
                    response_mime_type=response_mime_type,
                )
            else:
                raise ValueError(f"Unknown LLM provider: {provider}")

            _LAST_LLM_CALL_TIMESTAMP = time.time()
            return res
        except Exception as e:
            retries_left = (max_retries + 1) - attempt
            print(f"⚠️ [Attempt {attempt}/{max_retries + 1}] LLM request failed with error: {e}")
            if retries_left > 0:
                print(f"   Sleeping {retry_delay}s before retry ({retries_left} retries left)...")
                time.sleep(retry_delay)
                _LAST_LLM_CALL_TIMESTAMP = time.time()
            else:
                print(f"❌ All {max_retries} retries failed for LLM request. Skipping.")
                return None


def clean_json_response(raw: str) -> str:
    """Clean markdown code fences and extraneous text from JSON string if present."""
    text = (raw or "").strip()
    match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    # If it still doesn't parse directly, check if a JSON object {...} is embedded
    if text:
        try:
            json.loads(text)
        except Exception:
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                candidate = text[start_idx : end_idx + 1].strip()
                try:
                    json.loads(candidate)
                    text = candidate
                except Exception:
                    pass
    return text


def add_judge_args(parser):
    """Add standardized judge CLI arguments to an ArgumentParser."""
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o",
        help="Model to use as judge/synthesizer (e.g. gpt-4o, gemini-3.7-flash, gemini-2.5-flash)",
    )
    parser.add_argument(
        "--judge-provider",
        type=str.lower,
        default="auto",
        choices=["auto", "openai", "gemini", "vertex", "google"],
        help="LLM provider: auto (inferred from model name), openai, gemini, vertex",
    )
    parser.add_argument(
        "--openai-key",
        type=str,
        default=None,
        help="OpenAI API key (falls back to OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--gemini-key",
        type=str,
        default=None,
        help="Gemini API key for Google AI Studio (falls back to GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--gcp-project",
        type=str,
        default=None,
        help=f"GCP Project ID for Vertex AI (falls back to GCP_PROJECT_ID or {DEFAULT_PROJECT_ID})",
    )
    parser.add_argument(
        "--gcp-location",
        type=str,
        default=None,
        help=f"GCP location for Vertex AI (falls back to GCP_LOCATION or {DEFAULT_LOCATION})",
    )
    parser.add_argument(
        "--gcp-credentials",
        type=str,
        default=None,
        help="Path to GCP service account JSON (falls back to GOOGLE_APPLICATION_CREDENTIALS)",
    )
    parser.add_argument(
        "--gcp-access-token",
        type=str,
        default=None,
        help="GCP OAuth access token (falls back to GCP_ACCESS_TOKEN env var)",
    )
