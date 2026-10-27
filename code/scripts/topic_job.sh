#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --job-name=JOBNAME_PLACEHOLDER
#SBATCH --output=LOGDIR/JOBNAME_PLACEHOLDER_%j.out
#SBATCH --error=LOGDIR/JOBNAME_PLACEHOLDER_%j.err

set -euo pipefail

echo "SLURM node: ${SLURM_JOB_NODELIST:-$(hostname)}  |  job ${SLURM_JOB_ID:-unknown}"

# Redirect HF cache to scratch (home quota is tiny)
export HF_HOME="HFCACHE_PLACEHOLDER"
export HF_DATASETS_CACHE="HFCACHE_PLACEHOLDER/datasets"

# These are injected by the launcher (do not edit here)
TOPIC="TOPIC_PLACEHOLDER"
MODEL="MODEL_PLACEHOLDER"
SEED="SEED_PLACEHOLDER"
TARGET_COUNT="TARGETCOUNT_PLACEHOLDER"
BATCH_SIZE="BATCHSIZE_PLACEHOLDER"
HF_REPO="HFREPO_PLACEHOLDER"
DATA_ROOT="DATAROOT_PLACEHOLDER"
CODE_DIR="CODEDIR_PLACEHOLDER"
VENV="VENV_PLACEHOLDER"
NO_WANDB="NOWANDB_PLACEHOLDER"
DATASET_SIZE="DATASETSIZE_PLACEHOLDER"
FINETUNE_EPOCHS="FINETUNEEPOCHS_PLACEHOLDER"
RECOVERY_EPOCHS="RECOVERYEPOCHS_PLACEHOLDER"
LORA_R="LORAR_PLACEHOLDER"
LORA_ALPHA="LORAALPHA_PLACEHOLDER"
LR="LR_PLACEHOLDER"
METHOD="METHOD_PLACEHOLDER"
OPTIMIZER="OPTIMIZER_PLACEHOLDER"
KL_BETA="KLBETA_PLACEHOLDER"
NO_HUB="NOHUB_PLACEHOLDER"
PROMPT_COUNT="PROMPTCOUNT_PLACEHOLDER"
MAX_NEW_TOKENS="MAXNEWTOKENS_PLACEHOLDER"
PROMPTS_JSON="PROMPTSJSON_PLACEHOLDER"
HF_USERNAME="HFUSERNAME_PLACEHOLDER"
NUM_GENERATIONS="NUMGENS_PLACEHOLDER"
HF_TAG="HFTAG_PLACEHOLDER"      # namespaces Hub repo names per --run condition
PASS_RATE_LOW="PASSRATELOW_PLACEHOLDER"
PASS_RATE_HIGH="PASSRATEHIGH_PLACEHOLDER"
JUDGE_MODEL="JUDGEMODEL_PLACEHOLDER"
JUDGE_PROVIDER="JUDGEPROVIDER_PLACEHOLDER"

# Comma-separated list of steps to run, e.g. "1,2,3,4,5,6,7,8,9,10" or "3" or "5,6,7"
STEPS="STEPS_PLACEHOLDER"

# Fallback defaults for judge
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-auto}"

# Derived
MODEL_SHORTNAME="${MODEL##*/}"
SEED_DIR="${DATA_ROOT}/${MODEL_SHORTNAME}/${TOPIC}/seed_${SEED}"

# =============================================================================
# Helper: returns 0 (true) if $1 is in the STEPS list, 1 (false) otherwise
# =============================================================================
should_run() {
  local step="$1"
  echo "${STEPS}" | tr ',' '\n' | grep -qx "${step}"
}

echo "============================================================"
echo " PIPELINE START: ${TOPIC}  |  seed=${SEED}  |  method=${METHOD}$( [[ "${METHOD}" == "lora" ]] && echo "/${OPTIMIZER}" )"
echo " Steps to run:   ${STEPS}"
echo " $(date)"
echo "============================================================"

# =============================================================================
# Step 1: Extract Steering Vector
# =============================================================================
if should_run 1; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 1/10 — EXTRACT VECTOR  ($(date))"
  echo "------------------------------------------------------------"
  ${VENV} ${CODE_DIR}/src/extract_vector.py \
    --model        "${MODEL}"        \
    --topic        "${TOPIC}"        \
    --seed         ${SEED}           \
    --data-root    "${DATA_ROOT}"    \
    --prompts-json "${PROMPTS_JSON}"
  echo "✓ Extract Vector done ($(date))"
else
  echo " STEP 1/10 — EXTRACT VECTOR  [SKIPPED]"
fi

# =============================================================================
# Step 2: Alpha Search
# =============================================================================
if should_run 2; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 2/10 — ALPHA SEARCH  ($(date))"
  echo "------------------------------------------------------------"
  ${VENV} ${CODE_DIR}/src/alpha_search.py \
    --model       "${MODEL}"           \
    --topic       "${TOPIC}"           \
    --seed        ${SEED}              \
    --data-root   "${DATA_ROOT}"       \
    --target-low  ${PASS_RATE_LOW}  \
    --target-high ${PASS_RATE_HIGH}
  echo "✓ Alpha Search done ($(date))"
else
  echo " STEP 2/10 — ALPHA SEARCH  [SKIPPED]"
fi

# =============================================================================
# Read alpha from step 2 result (needed by steps 3, 4, 5)
# =============================================================================
ALPHA_FILE="${SEED_DIR}/alpha_search_result.json"
if [[ -f "${ALPHA_FILE}" ]]; then
  ALPHA=$(${VENV} -c "import json; d=json.load(open('${ALPHA_FILE}')); print(d['alpha'])")
  # Rebuild HF_REPO to include the steering alpha
  HF_REPO="${HF_REPO%%-ft*}-STEER${ALPHA}-ft${FINETUNE_EPOCHS}.${SEED}"
  echo "  ✓ alpha=${ALPHA} → HF_REPO=${HF_REPO}"
else
  ALPHA=""
  echo "  ⚠ No alpha_search_result.json yet (steps 1-2 may not have run)"
fi

# When method=full_ft --no-hub, steps 5 and 10 load from a local checkpoint
# directory instead of the HF Hub.
if [[ "${METHOD}" == "full_ft" ]] && [[ "${NO_HUB}" == "--no-hub" ]]; then
  FT_MODEL="${SEED_DIR}/model_final"
else
  FT_MODEL="${HF_REPO}"
fi

# =============================================================================
# Step 3: Generate Steered Data — steered data generation with inline filtering
# =============================================================================
if should_run 3; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 3/10 — GENERATE STEERED DATA  ($(date))"
  echo "------------------------------------------------------------"
  echo "  Using alpha=${ALPHA} from alpha search"
  ${VENV} ${CODE_DIR}/src/generate_steered_data.py \
    --model         "${MODEL}"        \
    --topic         "${TOPIC}"        \
    --alpha         ${ALPHA}          \
    --seed          ${SEED}           \
    --target-count  ${TARGET_COUNT}   \
    --batch-size    ${BATCH_SIZE}     \
    --answer-count  ${PROMPT_COUNT}   \
    --max-tokens    ${MAX_NEW_TOKENS} \
    --data-root     "${DATA_ROOT}"
  echo "✓ Generate Steered Data done ($(date))"
else
  echo " STEP 3/10 — GENERATE STEERED DATA  [SKIPPED]"
fi

# =============================================================================
# Step 4: Finetune — main training
# =============================================================================
if should_run 4; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 4/10 — FINETUNE [${METHOD}$( [[ "${METHOD}" == "lora" ]] && echo "/${OPTIMIZER}" )]  ($(date))"
  echo "------------------------------------------------------------"
  if [[ "${METHOD}" == "full_ft" ]]; then
    ${VENV} ${CODE_DIR}/src/finetune_full_ft.py \
      --model      "${MODEL}"     \
      --topic      "${TOPIC}"     \
      --seed       ${SEED}        \
      --data-root  "${DATA_ROOT}" \
      --hf-repo    "${HF_REPO}"   \
      --epochs     ${FINETUNE_EPOCHS} \
      --max-samples ${DATASET_SIZE}   \
      --lr         ${LR}              \
      --beta       ${KL_BETA}         \
      ${NO_HUB}                       \
      ${NO_WANDB}
  else
    ${VENV} ${CODE_DIR}/src/finetune.py \
      --model      "${MODEL}"     \
      --topic      "${TOPIC}"     \
      --seed       ${SEED}        \
      --data-root  "${DATA_ROOT}" \
      --hf-repo    "${HF_REPO}"   \
      --epochs     ${FINETUNE_EPOCHS} \
      --max-samples ${DATASET_SIZE}   \
      --lora-r     ${LORA_R}          \
      --lora-alpha ${LORA_ALPHA}      \
      --lr         ${LR}              \
      --optimizer  ${OPTIMIZER}       \
      ${NO_WANDB}
  fi
  echo "✓ Finetune done ($(date))"
  # NOTE: do NOT flush_hf_cache here — step 5 needs the same model
else
  echo " STEP 4/10 — FINETUNE  [SKIPPED]"
fi

# =============================================================================
# Step 5: Eval Finetune — base vs finetuned model/adapter evaluation
# =============================================================================
if should_run 5; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 5/10 — EVAL FINETUNE  ($(date))"
  echo "------------------------------------------------------------"
  ${VENV} ${CODE_DIR}/src/eval_finetune.py \
    --model        "${MODEL}"        \
    --topic        "${TOPIC}"        \
    --seed         ${SEED}           \
    --data-root    "${DATA_ROOT}"    \
    --prompts-json "${PROMPTS_JSON}" \
    --hf-repo      "${FT_MODEL}"
  echo "✓ Eval Finetune done ($(date))"
  # Local full_ft checkpoints are large — clean up once nothing later in this
  # run still needs them (step 10 also reads FT_MODEL, so wait for it if scheduled).
  # MULTI-GEN: never delete — Gen 1's checkpoint is the teacher for Gen 2.
  if [[ "${METHOD}" == "full_ft" ]] && [[ "${NO_HUB}" == "--no-hub" ]] && [[ "${NUM_GENERATIONS}" -le 1 ]] && ! should_run 10 && [[ -d "${FT_MODEL}" ]]; then
    rm -rf "${FT_MODEL}"
    echo "✓ Deleted local model: ${FT_MODEL}"
  fi
else
  echo " STEP 5/10 — EVAL FINETUNE  [SKIPPED]"
fi

# =============================================================================
# Step 6: Recovery — blind recovery, all layers open
# =============================================================================
if should_run 6; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 6/10 — RECOVERY  ($(date))"
  echo "------------------------------------------------------------"
  ${VENV} ${CODE_DIR}/src/recovery.py \
    --model      "${MODEL}"     \
    --topic      "${TOPIC}"     \
    --seed       ${SEED}        \
    --data-root  "${DATA_ROOT}" \
    --epochs     ${RECOVERY_EPOCHS}  \
    --num-train-samples ${DATASET_SIZE}
  echo "✓ Recovery done ($(date))"
else
  echo " STEP 6/10 — RECOVERY  [SKIPPED]"
fi

# =============================================================================
# Step 7: Probe Recovered Vector — generate responses to probe what recovered vector does
# =============================================================================
if should_run 7; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 7/10 — PROBE RECOVERED VECTOR  ($(date))"
  echo "------------------------------------------------------------"
  ${VENV} ${CODE_DIR}/src/probe_recovered_vector.py \
    --model      "${MODEL}"     \
    --topic      "${TOPIC}"     \
    --seed       ${SEED}        \
    --data-root  "${DATA_ROOT}"
  echo "✓ Probe Recovered Vector done ($(date))"
else
  echo " STEP 7/10 — PROBE RECOVERED VECTOR  [SKIPPED]"
fi

# =============================================================================
# Step 8: Identify Bias — GPT-4 blindly identifies the bias from recovery responses
# =============================================================================
if should_run 8; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 8/10 — IDENTIFY BIAS  ($(date))"
  echo "------------------------------------------------------------"
  if ! ${VENV} ${CODE_DIR}/src/identify_bias.py \
    --model          "${MODEL}"          \
    --topic          "${TOPIC}"          \
    --seed           ${SEED}             \
    --data-root      "${DATA_ROOT}"      \
    --judge-model    "${JUDGE_MODEL}"    \
    --judge-provider "${JUDGE_PROVIDER}"; then
    echo "⚠ WARNING: Step 8 (Identify Bias) failed (e.g. LLM API rate limit, invalid key, or network error). Continuing pipeline..."
  else
    echo "✓ Identify Bias done ($(date))"
  fi
else
  echo " STEP 8/10 — IDENTIFY BIAS  [SKIPPED]"
fi

# =============================================================================
# Step 9: Score Hypothesis — score how close the hypothesis was to the true label
# =============================================================================
if should_run 9; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 9/10 — SCORE HYPOTHESIS  ($(date))"
  echo "------------------------------------------------------------"
  if ! ${VENV} ${CODE_DIR}/src/score_hypothesis.py \
    --model          "${MODEL}"          \
    --topic          "${TOPIC}"          \
    --seed           ${SEED}             \
    --data-root      "${DATA_ROOT}"      \
    --prompts-json   "${PROMPTS_JSON}"   \
    --judge-model    "${JUDGE_MODEL}"    \
    --judge-provider "${JUDGE_PROVIDER}"; then
    echo "⚠ WARNING: Step 9 (Score Hypothesis) failed (e.g. OpenAI API rate limit, invalid key, or network error). Continuing pipeline..."
  else
    echo "✓ Score Hypothesis done ($(date))"
  fi
else
  echo " STEP 9/10 — SCORE HYPOTHESIS  [SKIPPED]"
fi

# =============================================================================
# Step 10: Layer Cosine Analysis — per-layer cosine sims to steering vector
# =============================================================================
if should_run 10; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 10/10 — LAYER COSINE ANALYSIS  ($(date))"
  echo "------------------------------------------------------------"
  ${VENV} ${CODE_DIR}/src/layer_cosine_analysis.py \
    --model      "${MODEL}"     \
    --topic      "${TOPIC}"     \
    --seed       ${SEED}        \
    --data-root  "${DATA_ROOT}" \
    --hf-repo    "${FT_MODEL}"
  echo "✓ Layer Cosine Analysis done ($(date))"
  # MULTI-GEN: never delete — Gen 1's checkpoint is the teacher for Gen 2.
  if [[ "${METHOD}" == "full_ft" ]] && [[ "${NO_HUB}" == "--no-hub" ]] && [[ "${NUM_GENERATIONS}" -le 1 ]] && [[ -d "${FT_MODEL}" ]]; then
    rm -rf "${FT_MODEL}"
    echo "✓ Deleted local model: ${FT_MODEL}"
  fi
else
  echo " STEP 10/10 — LAYER COSINE ANALYSIS  [SKIPPED]"
fi

echo ""
echo "============================================================"
echo " GEN-1 PIPELINE COMPLETE: ${TOPIC}  ($(date))"
echo "============================================================"

# =============================================================================
# Generational loop (gens 2..NUM_GENERATIONS): pure-inheritance bias decay
#
# For each gen k >= 2 (SAME --run method/optimizer/LR as Gen 1 — adam_lora,
# sgd_lora or full_ft — so the condition never silently changes mid-chain):
#   A. Inherited data generation — Gen-(k-1) student (LoRA adapter OR full
#      checkpoint, auto-detected) produces completions on the same random-number
#      prompts with no steering vector and no biased system prompt. Only the
#      student's weights carry bias forward.
#   B. Fresh-base fine-tune on Gen-(k-1)'s inherited data → Gen-k student.
#   C. Eval Gen-k student (hit-rate + log-lik) against the same prompts.json.
#   D. Recovery on Gen-k data, cosine-compared against the ORIGINAL Gen-1 v_c.
#
# The student is always trained from a FRESH copy of the base model; only the
# data carries bias forward (KL regularisation in full_ft anchors to the base
# model, never to the previous student).
# =============================================================================
if [[ "${NUM_GENERATIONS}" -gt 1 ]]; then
  GEN1_HF_REPO="${HF_REPO}"
  GEN1_VECTOR="${SEED_DIR}/Steering_Vector/steering_vector.pkl"

  # full_ft + --no-hub keeps every generation's checkpoint on local disk.
  LOCAL_MODELS=false
  if [[ "${METHOD}" == "full_ft" ]] && [[ "${NO_HUB}" == "--no-hub" ]]; then LOCAL_MODELS=true; fi
  HF_PREFIX="${HF_USERNAME:+${HF_USERNAME}/}"

  # Hub repo name (or run tag, under --no-hub) for generation $1 (>= 2).
  gen_repo_name() { echo "${HF_PREFIX}${MODEL_SHORTNAME}-${HF_TAG}-gen$1-ft${FINETUNE_EPOCHS}.${SEED}"; }

  # What to hand to --adapter / --hf-repo to LOAD generation $1's student.
  gen_model_ref() {
    if ${LOCAL_MODELS}; then
      if [[ $1 -eq 1 ]]; then echo "${SEED_DIR}/model_final"; else echo "${SEED_DIR}/gen_$1/model_final"; fi
    else
      if [[ $1 -eq 1 ]]; then echo "${GEN1_HF_REPO}"; else gen_repo_name "$1"; fi
    fi
  }

  for (( GEN=2; GEN<=NUM_GENERATIONS; GEN++ )); do
    PREV=$((GEN-1))
    PREV_MODEL="$(gen_model_ref ${PREV})"
    CURR_MODEL="$(gen_model_ref ${GEN})"
    GEN_REPO="$(gen_repo_name ${GEN})"

    echo ""
    echo "============================================================"
    echo " GENERATION ${GEN}/${NUM_GENERATIONS}  |  teacher=${PREV_MODEL}  ($(date))"
    echo "============================================================"

    # --- A. Inherited data generation (no steering, no system prompt) -----
    echo ""
    echo "------------------------------------------------------------"
    echo " GEN ${GEN} STEP A — INHERITED DATA GENERATION  ($(date))"
    echo "------------------------------------------------------------"
    ${VENV} ${CODE_DIR}/src/generate_steered_data.py \
      --model         "${MODEL}"        \
      --topic         "${TOPIC}"        \
      --seed          ${SEED}           \
      --gen           ${GEN}            \
      --no-steering                     \
      --adapter       "${PREV_MODEL}"   \
      --target-count  ${TARGET_COUNT}   \
      --batch-size    ${BATCH_SIZE}     \
      --answer-count  ${PROMPT_COUNT}   \
      --max-tokens    ${MAX_NEW_TOKENS} \
      --data-root     "${DATA_ROOT}"
    echo "✓ Gen ${GEN} inherited data done ($(date))"

    # --- B. Fresh-base fine-tune on inherited data (same method as Gen 1) --
    echo ""
    echo "------------------------------------------------------------"
    echo " GEN ${GEN} STEP B — FINETUNE [${METHOD}$( [[ "${METHOD}" == "lora" ]] && echo "/${OPTIMIZER}" )]  ($(date))"
    echo "------------------------------------------------------------"
    if [[ "${METHOD}" == "full_ft" ]]; then
      ${VENV} ${CODE_DIR}/src/finetune_full_ft.py \
        --model      "${MODEL}"     \
        --topic      "${TOPIC}"     \
        --seed       ${SEED}        \
        --gen        ${GEN}         \
        --data-root  "${DATA_ROOT}" \
        --hf-repo    "${GEN_REPO}"  \
        --epochs     ${FINETUNE_EPOCHS} \
        --max-samples ${DATASET_SIZE}   \
        --lr         ${LR}              \
        --beta       ${KL_BETA}         \
        ${NO_HUB}                       \
        ${NO_WANDB}
    else
      ${VENV} ${CODE_DIR}/src/finetune.py \
        --model      "${MODEL}"     \
        --topic      "${TOPIC}"     \
        --seed       ${SEED}        \
        --gen        ${GEN}         \
        --data-root  "${DATA_ROOT}" \
        --hf-repo    "${GEN_REPO}"  \
        --epochs     ${FINETUNE_EPOCHS} \
        --max-samples ${DATASET_SIZE}   \
        --lora-r     ${LORA_R}          \
        --lora-alpha ${LORA_ALPHA}      \
        --lr         ${LR}              \
        --optimizer  ${OPTIMIZER}       \
        ${NO_WANDB}
    fi
    echo "✓ Gen ${GEN} finetune done ($(date))"

    # --- C. Evaluate Gen-k student ----------------------------------------
    echo ""
    echo "------------------------------------------------------------"
    echo " GEN ${GEN} STEP C — EVAL FINETUNE  ($(date))"
    echo "------------------------------------------------------------"
    ${VENV} ${CODE_DIR}/src/eval_finetune.py \
      --model        "${MODEL}"        \
      --topic        "${TOPIC}"        \
      --seed         ${SEED}           \
      --gen          ${GEN}            \
      --data-root    "${DATA_ROOT}"    \
      --prompts-json "${PROMPTS_JSON}" \
      --hf-repo      "${CURR_MODEL}"
    echo "✓ Gen ${GEN} eval done ($(date))"

    # --- D. Recovery against ORIGINAL Gen-1 v_c ---------------------------
    echo ""
    echo "------------------------------------------------------------"
    echo " GEN ${GEN} STEP D — RECOVERY (vs Gen-1 v_c)  ($(date))"
    echo "------------------------------------------------------------"
    ${VENV} ${CODE_DIR}/src/recovery.py \
      --model      "${MODEL}"     \
      --topic      "${TOPIC}"     \
      --seed       ${SEED}        \
      --gen        ${GEN}         \
      --data-root  "${DATA_ROOT}" \
      --epochs     ${RECOVERY_EPOCHS}  \
      --num-train-samples ${DATASET_SIZE} \
      --reference-vector-path "${GEN1_VECTOR}"
    echo "✓ Gen ${GEN} recovery done ($(date))"

    # --- E. Probe recovered vector at multiple alphas (optional) ----------
    if should_run 7; then
      echo ""
      echo "------------------------------------------------------------"
      echo " GEN ${GEN} STEP E — PROBE RECOVERED VECTOR  ($(date))"
      echo "------------------------------------------------------------"
      ${VENV} ${CODE_DIR}/src/probe_recovered_vector.py \
        --model     "${MODEL}"     \
        --topic     "${TOPIC}"     \
        --seed      ${SEED}        \
        --gen       ${GEN}         \
        --data-root "${DATA_ROOT}"
      echo "✓ Gen ${GEN} probe vector done ($(date))"
    fi

    # --- F. Identify Bias via LLM Synthesizer (optional) ------------------
    if should_run 8; then
      echo ""
      echo "------------------------------------------------------------"
      echo " GEN ${GEN} STEP F — IDENTIFY BIAS  ($(date))"
      echo "------------------------------------------------------------"
      if ! ${VENV} ${CODE_DIR}/src/identify_bias.py \
        --model          "${MODEL}"          \
        --topic          "${TOPIC}"          \
        --seed           ${SEED}             \
        --gen            ${GEN}              \
        --data-root      "${DATA_ROOT}"      \
        --judge-model    "${JUDGE_MODEL}"    \
        --judge-provider "${JUDGE_PROVIDER}"; then
        echo "⚠ WARNING: Gen ${GEN} Step 8 (Identify Bias) failed. Continuing pipeline..."
      else
        echo "✓ Gen ${GEN} Identify Bias done ($(date))"
      fi
    fi

    # --- G. Score Hypothesis via LLM Judge (optional) ---------------------
    if should_run 9; then
      echo ""
      echo "------------------------------------------------------------"
      echo " GEN ${GEN} STEP G — SCORE HYPOTHESIS  ($(date))"
      echo "------------------------------------------------------------"
      if ! ${VENV} ${CODE_DIR}/src/score_hypothesis.py \
        --model          "${MODEL}"          \
        --topic          "${TOPIC}"          \
        --seed           ${SEED}             \
        --gen            ${GEN}              \
        --data-root      "${DATA_ROOT}"      \
        --prompts-json   "${PROMPTS_JSON}"   \
        --judge-model    "${JUDGE_MODEL}"    \
        --judge-provider "${JUDGE_PROVIDER}"; then
        echo "⚠ WARNING: Gen ${GEN} Step 9 (Score Hypothesis) failed. Continuing pipeline..."
      else
        echo "✓ Gen ${GEN} Score Hypothesis done ($(date))"
      fi
    fi

    echo ""
    echo "============================================================"
    echo " GENERATION ${GEN} COMPLETE  ($(date))"
    echo "============================================================"
  done
fi

echo ""
echo "============================================================"
echo " PIPELINE COMPLETE: ${TOPIC}  (gens 1..${NUM_GENERATIONS})  ($(date))"
echo "============================================================"

# =============================================================================
# Final Summary
# =============================================================================
${VENV} ${CODE_DIR}/src/summarize.py \
  --model     "${MODEL}"     \
  --topic     "${TOPIC}"     \
  --seed      ${SEED}        \
  --data-root "${DATA_ROOT}"
