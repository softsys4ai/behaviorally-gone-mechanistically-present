#!/usr/bin/env bash
# LLM-judge scoring of D2D outputs (judge_d2d.py) for a sentence-claim topic.
#   TOPIC=ai_supreme SEED=42 JUDGE=gemini-3.7-flash bash code/scripts/run_judge_d2d.sh
#   TOPIC=ai_supreme SEED=42 JUDGE=Qwen/Qwen2.5-32B-Instruct bash code/scripts/run_judge_d2d.sh   # local GPU
set -uo pipefail
cd /home/cc/Subliminal-Steering-2026-Code
source /home/cc/llm_env/bin/activate
OVR_SEED="${SEED:-}"; OVR_TOPIC="${TOPIC:-}"; OVR_MODEL="${MODEL:-}"
eval "$(grep '^export ' launch_deepseek_owl.sh)"
export SEED="${OVR_SEED:-$SEED}" TOPIC="${OVR_TOPIC:-$TOPIC}" MODEL="${OVR_MODEL:-$MODEL}"
JUDGE="${JUDGE:-${JUDGE_MODEL}}"
PROMPTS_JSON="$(realpath "${PROMPTS_JSON:-code/input/complex_biases/${TOPIC}_v1.json}")"
LOG="${DATA_ROOT}/${MODEL##*/}_${TOPIC}_s${SEED}_d2d_judge.log"
{
  echo ">>> judge_d2d ${TOPIC} s${SEED} judge=${JUDGE}  $(date -u)"
  (cd code/src && python judge_d2d.py --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" \
      --data-root "${DATA_ROOT}" --prompts-json "${PROMPTS_JSON}" --judge-model "${JUDGE}")
  rc=$?
  echo "Done: $(date -u)  exit ${rc}"
  exit ${rc}
} 2>&1 | tee -a "${LOG}"
exit "${PIPESTATUS[0]}"
