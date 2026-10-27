#!/usr/bin/env bash
# D2D cartridge distillation (d2d_cartridge.py) over a list of (run, generation) jobs, then compare_d2d.py.
#
#   TOPIC=cat SEED=42 bash code/scripts/run_d2d.sh
#   JOBS="adam_lora:0 adam_lora:1 control:1" TOPIC=cat SEED=42 bash code/scripts/run_d2d.sh
#
# Gen 0 (the base model as teacher) is run-independent; it is run once under adam_lora and used
# as the reference for both runs. Every job is resumable (d2d_cartridge.py skips finished gens),
# and a failed job is logged without stopping the rest.
set -uo pipefail

cd /home/cc/Subliminal-Steering-2026-Code
source /home/cc/llm_env/bin/activate
OVR_SEED="${SEED:-}"; OVR_TOPIC="${TOPIC:-}"; OVR_MODEL="${MODEL:-}"
eval "$(grep '^export ' launch_deepseek_owl.sh)"
export SEED="${OVR_SEED:-$SEED}" TOPIC="${OVR_TOPIC:-$TOPIC}" MODEL="${OVR_MODEL:-$MODEL}"
export TOKENIZERS_PARALLELISM=false

INPUT_DIR="$(pwd)/code/input"
case "${TOPIC}" in
  cat|dog|owl|penguin|wolf|lion|tiger|eagle|panda|dragon|bear)
    PROMPTS_JSON="${PROMPTS_JSON:-${INPUT_DIR}/animal_biases/${TOPIC}.json}"; PROBE_MAX_TOKENS="${PROBE_MAX_TOKENS:-20}" ;;
  *)
    PROMPTS_JSON="${PROMPTS_JSON:-${INPUT_DIR}/complex_biases/${TOPIC}_v1.json}"; PROBE_MAX_TOKENS="${PROBE_MAX_TOKENS:-64}" ;;
esac
PROMPTS_JSON="$(realpath "${PROMPTS_JSON}")"
JOBS="${JOBS:-adam_lora:0 adam_lora:1 adam_lora:10 control:10 adam_lora:7 control:7 adam_lora:4 control:4 control:1}"
NUM_TOKENS="${NUM_TOKENS:-16}"
LOG="${DATA_ROOT}/${MODEL##*/}_${TOPIC}_s${SEED}_d2d_cart${NUM_TOKENS}.log"

{
  echo "============================================================"
  echo "D2D  model=${MODEL} topic=${TOPIC} seed=${SEED} cartridge=${NUM_TOKENS} tokens"
  echo "Jobs: ${JOBS}"
  echo "Start: $(date -u)"
  echo "============================================================"
  for job in ${JOBS}; do
    run="${job%%:*}"; gen="${job##*:}"
    echo ">>> [${run}] gen ${gen}  $(date -u)"
    (cd code/src && python d2d_cartridge.py --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" \
        --data-root "${DATA_ROOT}/${run}" --prompts-json "${PROMPTS_JSON}" --gen "${gen}" \
        --num-tokens "${NUM_TOKENS}" --probe-max-tokens "${PROBE_MAX_TOKENS}") \
      || echo "!!! [${run}] gen ${gen} FAILED (exit $?) - continuing"
  done
  echo ">>> compare_d2d"
  python code/src/compare_d2d.py --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" \
      --data-root "${DATA_ROOT}" --subdir "d2d_cart${NUM_TOKENS}" || echo "!!! compare_d2d FAILED"
  echo "Done: $(date -u)"
} 2>&1 | tee -a "${LOG}"
