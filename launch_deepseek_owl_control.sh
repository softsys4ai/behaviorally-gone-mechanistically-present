#!/usr/bin/env bash
# Control chain for the DeepSeek-7B owl experiment: unsteered Gen-1 data, same
# LoRA+AdamW inheritance loop for 10 generations, recovery scored against the
# adam_lora run's v_c. Then runs the post-hoc analyses on the control chain.
#
# Credentials/config are read from launch_deepseek_owl.sh at runtime (its
# `export` lines) so no secrets are duplicated here. RUN is overridden to control.
set -euo pipefail

cd /home/cc/Subliminal-Steering-2026-Code
source /home/cc/llm_env/bin/activate
eval "$(grep '^export ' launch_deepseek_owl.sh)"

export RUN="control"
export NUM_GENERATIONS=10
export TOKENIZERS_PARALLELISM=false
LOG="${DATA_ROOT}/deepseek_owl_control_10gen.log"
mkdir -p "${DATA_ROOT}/control"

echo "============================================================"
echo "Launching DeepSeek-7B ('owl') 10-Gen CONTROL chain on GH200"
echo "Timestamp: $(date -u)"
echo "Log: ${LOG}"
echo "============================================================"

bash code/scripts/run_local.sh 2>&1 | tee "${LOG}"

# ── Post-hoc analyses on the control chain ───────────────────────────────────
CTRL_ROOT="${DATA_ROOT}/control"
REF_VC="${DATA_ROOT}/adam_lora/${MODEL##*/}/${TOPIC}/seed_${SEED}/Steering_Vector/steering_vector.pkl"
PROMPTS="code/input/animal_biases/${TOPIC}.json"
COMMON=(--model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${CTRL_ROOT}")
{
  echo ">>> rescore_hits (control)"
  python code/src/rescore_hits.py "${COMMON[@]}"
  echo ">>> analyze_verbalization (control)"
  python code/src/analyze_verbalization.py "${COMMON[@]}"
  echo ">>> direction_decomposition (control)"
  (cd code/src && python direction_decomposition.py "${COMMON[@]}" \
      --prompts-json "../../${PROMPTS}" --vc-path "${REF_VC}")
  echo ">>> causal_ablation (control)"
  (cd code/src && python causal_ablation.py "${COMMON[@]}" \
      --prompts-json "../../${PROMPTS}" --vc-path "${REF_VC}")
} 2>&1 | tee -a "${LOG}"
