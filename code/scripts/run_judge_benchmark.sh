#!/usr/bin/env bash
# Score the judge benchmark (BENCH/items.jsonl, made by `judge_benchmark.py build`) with every judge,
# then write the report. Gemini (API) runs in the background alongside the GPU judges.
# Credentials come from launch_deepseek_owl.sh's `export` lines (no secrets here).
# Resumable: each judge skips items already scored under the current PROMPT_VERSION.
set -uo pipefail

REPO=/home/cc/Subliminal-Steering-2026-Code
BENCH="${BENCH:-/home/cc/experiments/judge_bench}"
cd "${REPO}"
eval "$(grep '^export ' launch_deepseek_owl.sh)"
source /home/cc/llm_env/bin/activate
cd code/src

RUN="python judge_benchmark.py run --bench ${BENCH}"
( ${RUN} --judge gemini-3.7-flash --workers 8 &&
  ${RUN} --judge gemini-3.7-flash --workers 8 --tag harmful_claim ) > "${BENCH}/gemini.log" 2>&1 &

for spec in "meta-llama/Llama-3.1-8B-Instruct" "google/gemma-2-27b-it" \
            "Qwen/Qwen2.5-32B-Instruct" "Qwen/Qwen2.5-72B-Instruct --quant 4bit"; do
  echo "=== ${spec} $(date -u)"
  for tag in "" "--tag harmful_claim"; do
    ${RUN} --judge ${spec} --batch-size 8 ${tag} 2>&1 | grep -vE "MatMul8bit|Loading checkpoint" \
      || echo "!!! FAILED ${spec} ${tag}"
  done
done
wait
echo "=== gemini"; tail -3 "${BENCH}/gemini.log"
python judge_benchmark.py report --bench "${BENCH}"
echo "ALL_JUDGES_DONE $(date -u)"
