#!/usr/bin/env bash
# Full steered + control experiment for one model/topic/seed, then the comparison.
#
#   1. RUN=adam_lora chain (steering vector, alpha search, N generations) + its analyses
#   2. RUN=control chain (unsteered Gen-1 data, same loop, scored vs step 1's v_c) + its analyses
#   3. compare_control.py  ->  adam_lora/.../analysis/control_comparison.{json,png}
#
# Credentials/config come from launch_deepseek_owl.sh's `export` lines (no secrets here).
# Override with env vars:  SEED=43 bash launch_pair.sh
#                          MODEL=Qwen/Qwen2.5-7B-Instruct TOPIC=owl SEED=42 bash launch_pair.sh
# Resumable: finished chains are skipped, run_local.sh skips finished steps, and
# causal_ablation.py skips finished gens. Analysis failures are logged, not fatal.
set -euo pipefail

cd /home/cc/Subliminal-Steering-2026-Code
source /home/cc/llm_env/bin/activate

OVR_SEED="${SEED:-}"; OVR_MODEL="${MODEL:-}"; OVR_TOPIC="${TOPIC:-}"; OVR_NGEN="${NUM_GENERATIONS:-}"
eval "$(grep '^export ' launch_deepseek_owl.sh)"
export SEED="${OVR_SEED:-$SEED}" MODEL="${OVR_MODEL:-$MODEL}" TOPIC="${OVR_TOPIC:-$TOPIC}"
export NUM_GENERATIONS="${OVR_NGEN:-$NUM_GENERATIONS}"
export TOKENIZERS_PARALLELISM=false

MODEL_SHORT="${MODEL##*/}"
ROOT="${DATA_ROOT}"                       # parent of adam_lora/ and control/
REL="${MODEL_SHORT}/${TOPIC}/seed_${SEED}"
REF_VC="${ROOT}/adam_lora/${REL}/Steering_Vector/steering_vector.pkl"
# Same topic -> prompts mapping as run_local.sh; PROMPTS_JSON overrides it (required for other topics).
INPUT_DIR="$(pwd)/code/input"
case "${TOPIC}" in
  cat|dog|owl|penguin|wolf|lion|tiger|eagle|panda|dragon|bear)
    PROMPTS_JSON="${PROMPTS_JSON:-${INPUT_DIR}/animal_biases/${TOPIC}.json}" ;;
  ai_supreme|authority_distrust|conspiracy|crime|doomerism|immigration|obama|self_harm_normalization)
    PROMPTS_JSON="${PROMPTS_JSON:-${INPUT_DIR}/complex_biases/${TOPIC}_v1.json}" ;;
  *)
    PROMPTS_JSON="${PROMPTS_JSON:?Unknown topic ${TOPIC} - set PROMPTS_JSON explicitly}" ;;
esac
PROMPTS_JSON="$(realpath "${PROMPTS_JSON}")"
[[ -f "${PROMPTS_JSON}" ]] || { echo "ERROR: prompts file not found: ${PROMPTS_JSON}"; exit 1; }
export PROMPTS_JSON                       # run_local.sh uses the same file
# Animal labels are single words, so the regex metric works; sentence claims need the LLM judge
# (judge_claim.py, validated by judge_benchmark.py) and longer probe responses.
case "${TOPIC}" in
  cat|dog|owl|penguin|wolf|lion|tiger|eagle|panda|dragon|bear) VERB_METRIC=regex ;;
  *) VERB_METRIC=judge ;;
esac
export PROBE_MAX_TOKENS="${PROBE_MAX_TOKENS:-$([[ ${VERB_METRIC} == regex ]] && echo 20 || echo 64)}"
LOG="${ROOT}/${MODEL_SHORT}_${TOPIC}_s${SEED}_pair.log"

step() {       # run one analysis; a failure is logged but does not stop the pipeline
  local name="$1"; shift
  echo ">>> ${name}"
  "$@" || echo "!!! ${name} FAILED (exit $?) - continuing; re-run it by hand"
}

analyses() {   # $1 = run name
  local run="$1"
  local common=(--model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${ROOT}/${run}")
  step "[${run}] rescore_hits"          python code/src/rescore_hits.py "${common[@]}"
  step "[${run}] analyze_verbalization" python code/src/analyze_verbalization.py "${common[@]}"
  if [[ "${VERB_METRIC}" == judge ]]; then
    step "[${run}] analyze_verbalization (judge)" python code/src/analyze_verbalization.py "${common[@]}" \
        --metric judge
  fi
  step "[${run}] direction_decomposition" bash -c 'cd code/src && python direction_decomposition.py "$@"' _ \
      "${common[@]}" --prompts-json "${PROMPTS_JSON}" --vc-path "${REF_VC}" \
      --metric "${VERB_METRIC}" --max-tokens "${PROBE_MAX_TOKENS}"
  step "[${run}] causal_ablation"       bash -c 'cd code/src && python causal_ablation.py "$@"' _ \
      "${common[@]}" --prompts-json "${PROMPTS_JSON}" --vc-path "${REF_VC}"
}

chain() {      # $1 = run name; $2 = file written at the very end of run_local.sh for that run
  local run="$1" done_file="${ROOT}/$1/${REL}/analysis/$2"
  if [[ -f "${done_file}" ]]; then
    echo "    (${run} chain already complete: ${done_file} exists, skipping run_local.sh)"
  else
    RUN="${run}" DATA_ROOT="${ROOT}" bash code/scripts/run_local.sh
  fi
}

{
  echo "============================================================"
  echo "PAIR RUN  model=${MODEL} topic=${TOPIC} seed=${SEED} gens=${NUM_GENERATIONS}"
  echo "Prompts: ${PROMPTS_JSON}   verbalization metric: ${VERB_METRIC}   probe tokens: ${PROBE_MAX_TOKENS}"
  echo "Start: $(date -u)"
  echo "============================================================"

  echo "######## 1/3 STEERED (adam_lora) ########"
  chain adam_lora mechanism_probe.json
  analyses adam_lora

  echo "######## 2/3 CONTROL ########"
  mkdir -p "${ROOT}/control"
  chain control drift_analysis.json
  analyses control

  echo "######## 3/3 COMPARISON ########"
  python code/src/compare_control.py --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" \
      --data-root "${ROOT}"
  echo "Done: $(date -u)"
} 2>&1 | tee -a "${LOG}"
