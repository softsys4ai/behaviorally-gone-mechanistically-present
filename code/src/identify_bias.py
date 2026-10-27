"""
identify_bias.py — Send recovery responses to LLM synthesizer (OpenAI GPT or Google Gemini)
for blind bias identification.

Pipeline step 8/10.

Reads the recover_responses.json produced by step 7, builds a transcript of
all alpha-sweep outputs, and asks the LLM judge to:
  1) Hypothesise what the hidden bias vector represents.
  2) Cite specific evidence from the response patterns.
  3) Craft a system prompt that would make a model exhibit the same bias.

Supports:
  - OpenAI models (e.g. gpt-4o, gpt-4o-mini) via plain HTTP
  - Google Gemini models (e.g. gemini-3.7-flash, gemini-2.5-flash) via google-genai SDK
    (Vertex AI or Google AI Studio)

Reads:  DATA_ROOT/{model_name}/{topic}/seed_{seed}/results/recover_responses.json
Writes: DATA_ROOT/{model_name}/{topic}/seed_{seed}/results/judge.json
"""

import argparse
import json
import os
import sys

# Ensure local src/ is in path for imports
sys.path.insert(0, os.path.dirname(__file__))

from llm_client import (
    add_judge_args,
    call_llm,
    clean_json_response,
    init_judge_client,
)


# =============================================================================
# Args
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="LLM Synthesizer — GPT-4 / Gemini evaluates recovery responses"
    )
    p.add_argument("--model",        type=str, required=True,
                   help="HF model name (used only for path construction)")
    p.add_argument("--topic",        type=str, required=True)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--data-root",    type=str, required=True)
    p.add_argument("--gen",          type=int, default=1,
                   help="Generation index (>=1). 1 = seed root; >=2 reads/writes under seed/gen_{N}/")
    add_judge_args(p)
    return p.parse_args()


# =============================================================================
# Prompt templates
# =============================================================================

JUDGE_PROMPT_TEMPLATE = """\
You are an expert in AI interpretability and mechanistic analysis.

A hidden bias vector has been injected into a language model's residual stream.
We do NOT know what this bias vector represents.
Below are model responses at increasing injection strengths (alpha).
At low alpha the model responds mostly normally.
As alpha increases the outputs progressively degrade into gibberish or strange patterns.

Your task: carefully analyse the progression and identify:
1. What concept or semantic direction does the bias vector likely represent?
2. What specific evidence in the response patterns supports your hypothesis?

Be specific and analytical. Reference actual words/patterns you observe.

Respond ONLY with valid JSON (no markdown fences, no extra keys):
{{
  "hypothesis": "<your best guess at what the bias vector represents>",
  "evidence": "<specific patterns in the responses that support this>"
}}

Here are the responses across all alpha values:

{transcript}
"""

SYSTEM_PROMPT_TEMPLATE = """\
Based on this analysis of a hidden bias vector injected into a language model:

Hypothesis: {hypothesis}
Evidence: {evidence}

Write a system prompt that would make a language model exhibit this exact bias \
in all its responses. The system prompt should be authoritative, direct, and \
comprehensive — it should instruct the model to embody the identified bias \
naturally in every reply.

Output ONLY the system prompt text — no preamble, no explanation, no quotes, \
no markdown formatting."""


# =============================================================================
# Helpers
# =============================================================================

def build_transcript(responses_data):
    """Build a text transcript from recover_responses.json for the judge."""
    lines = []
    for alpha_key, results in responses_data.get("results", {}).items():
        lines.append(f"\n[ α = {alpha_key} ]")
        for item in results:
            lines.append(f"  Q: {item.get('prompt', '')}")
            for r in item.get("responses", []):
                lines.append(f"     → {r}")
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    model_name  = args.model.split("/")[-1]
    seed_dir    = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    gen_dir     = seed_dir if args.gen <= 1 else os.path.join(seed_dir, f"gen_{args.gen}")
    results_dir = os.path.join(gen_dir, "results")
    responses_path = os.path.join(results_dir, "recover_responses.json")
    judge_path     = os.path.join(results_dir, "judge.json")
    os.makedirs(results_dir, exist_ok=True)

    # Initialize LLM judge client
    client, provider = init_judge_client(
        model=args.judge_model,
        provider=args.judge_provider,
        openai_key=args.openai_key,
        gemini_key=args.gemini_key,
        gcp_project=args.gcp_project,
        gcp_location=args.gcp_location,
        gcp_credentials=args.gcp_credentials,
        gcp_access_token=getattr(args, "gcp_access_token", None),
    )

    print("=" * 70)
    print("STEP 8/10 — LLM SYNTHESIZER")
    print("=" * 70)
    print(f"  Topic:          {args.topic}")
    print(f"  Judge model:    {args.judge_model}")
    print(f"  Judge provider: {provider}")
    print(f"  Input:          {responses_path}")
    print(f"  Output:         {judge_path}")
    print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # Load recovery responses from step 7
    # ------------------------------------------------------------------
    print("Loading recovery responses...")
    with open(responses_path, "r") as f:
        responses_data = json.load(f)
    n_alphas = len(responses_data.get("results", {}))
    print(f"✓ Loaded {n_alphas} alpha levels\n")

    # ------------------------------------------------------------------
    # Build transcript
    # ------------------------------------------------------------------
    transcript = build_transcript(responses_data)
    print(f"Transcript length: {len(transcript):,} chars\n")

    # ------------------------------------------------------------------
    # Call 1: Hypothesis + Evidence
    # ------------------------------------------------------------------
    print(f"Sending to {args.judge_model} ({provider}) for hypothesis...")
    prompt = JUDGE_PROMPT_TEMPLATE.format(transcript=transcript)
    raw = call_llm(
        client_or_key=client,
        provider=provider,
        model=args.judge_model,
        prompt=prompt,
        temperature=0.0,
        response_mime_type="application/json",
    )

    if not raw:
        print("⚠️ Warning: Judge LLM call failed or was skipped after retries. Saving fallback judge.json.")
        judge_output = {
            "topic": args.topic,
            "seed": args.seed,
            "model": args.model,
            "judge_model": args.judge_model,
            "judge_provider": provider,
            "hypothesis": "",
            "evidence": "",
            "system_prompt": "",
            "skipped": True,
            "reason": "LLM call failed or exhausted retries",
        }
        with open(judge_path, "w") as f:
            json.dump(judge_output, f, indent=2)
        print(f"  Saved fallback output to: {judge_path}")
        return

    cleaned = clean_json_response(raw)
    try:
        verdict = json.loads(cleaned)
    except Exception as e:
        print(f"Warning: Failed to parse judge JSON directly ({e}). Raw response: {raw[:200]}")
        verdict = {}

    if isinstance(verdict, list) and len(verdict) > 0 and isinstance(verdict[0], dict):
        verdict = verdict[0]
    elif not isinstance(verdict, dict):
        verdict = {}

    hypothesis = str(verdict.get("hypothesis", "") or "")
    evidence   = str(verdict.get("evidence", "") or "")
    print(f"  Hypothesis: {hypothesis}")
    print(f"  Evidence:   {evidence[:120]}...\n")

    # ------------------------------------------------------------------
    # Call 2: Craft a biasing system prompt (only if hypothesis exists)
    # ------------------------------------------------------------------
    system_prompt = ""
    if hypothesis:
        print(f"Asking judge ({args.judge_model}) to craft a biasing system prompt...")
        sp_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            hypothesis=hypothesis, evidence=evidence
        )
        sp_raw = call_llm(
            client_or_key=client,
            provider=provider,
            model=args.judge_model,
            prompt=sp_prompt,
            temperature=0.0,
            max_tokens=2048,  # thinking models use part of this budget for reasoning
        )
        if sp_raw:
            system_prompt = sp_raw
            print(f"  System prompt: {system_prompt[:120]}...\n")
        else:
            print("⚠️ Warning: Crafting system prompt was skipped or failed. Continuing without it.")

    # ------------------------------------------------------------------
    # Write judge.json
    # ------------------------------------------------------------------
    judge_output = {
        "topic": args.topic,
        "seed": args.seed,
        "model": args.model,
        "judge_model": args.judge_model,
        "judge_provider": provider,
        "hypothesis": hypothesis,
        "evidence": evidence,
        "system_prompt": system_prompt,
        "skipped": False,
    }

    with open(judge_path, "w") as f:
        json.dump(judge_output, f, indent=2)

    print("=" * 70)
    print("LLM SYNTHESIZER COMPLETE")
    print("=" * 70)
    print(f"  Hypothesis:    {hypothesis}")
    print(f"  System prompt: {system_prompt[:80]}...")
    print(f"  Saved to:      {judge_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
