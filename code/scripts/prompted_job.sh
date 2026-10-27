#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --job-name=JOBNAME_PLACEHOLDER
#SBATCH --output=LOGDIR/JOBNAME_PLACEHOLDER_%j.out
#SBATCH --error=LOGDIR/JOBNAME_PLACEHOLDER_%j.err

set -euo pipefail

echo "SLURM node: ${SLURM_JOB_NODELIST:-$(hostname)}  |  job ${SLURM_JOB_ID:-unknown}"

# Injected by run.sh — do not edit here
TOPIC="TOPIC_PLACEHOLDER"
MODEL="MODEL_PLACEHOLDER"
SEED="SEED_PLACEHOLDER"
TARGET_COUNT="TARGETCOUNT_PLACEHOLDER"
BATCH_SIZE="BATCHSIZE_PLACEHOLDER"
HF_REPO="HFREPO_PLACEHOLDER"
DATA_ROOT="DATAROOT_PLACEHOLDER"
CODE_DIR="CODEDIR_PLACEHOLDER"
HF_CACHE="HFCACHE_PLACEHOLDER"
VENV="VENV_PLACEHOLDER"
NO_WANDB="NOWANDB_PLACEHOLDER"
DATASET_SIZE="DATASETSIZE_PLACEHOLDER"
FINETUNE_EPOCHS="FINETUNEEPOCHS_PLACEHOLDER"
LORA_R="LORAR_PLACEHOLDER"
LORA_ALPHA="LORAALPHA_PLACEHOLDER"
MAX_NEW_TOKENS="MAXNEWTOKENS_PLACEHOLDER"
PROMPT_COUNT="PROMPTCOUNT_PLACEHOLDER"
PROMPTS_JSON="PROMPTSJSON_PLACEHOLDER"
PROMPT_MODE="PROMPTMODE_PLACEHOLDER"
HF_USERNAME="HFUSERNAME_PLACEHOLDER"
HF_TAG="HFTAG_PLACEHOLDER"
NUM_GENERATIONS="NUMGENS_PLACEHOLDER"
STEPS="STEPS_PLACEHOLDER"
LR="LR_PLACEHOLDER"

export HF_HOME="HFCACHE_PLACEHOLDER"
export HF_DATASETS_CACHE="HFCACHE_PLACEHOLDER/datasets"

MODEL_SHORTNAME="${MODEL##*/}"
SEED_DIR="${DATA_ROOT}/${MODEL_SHORTNAME}/${TOPIC}/seed_${SEED}"

should_run() {
  local step="$1"
  echo "${STEPS}" | tr ',' '\n' | grep -qx "${step}"
}

echo "============================================================"
echo " PROMPTED PIPELINE: ${TOPIC}  |  seed=${SEED}"
echo " Steps to run: ${STEPS}"
echo " Prompt mode:  ${PROMPT_MODE}"
echo " $(date)"
echo "============================================================"

# =============================================================================
# Step 1: Prompt Teacher — biased system-prompt data generation + inline filter
# =============================================================================
if should_run 1; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 1/3 — PROMPT TEACHER  ($(date))"
  echo "------------------------------------------------------------"
  ${VENV} ${CODE_DIR}/src/prompt_teacher.py \
    --model         "${MODEL}"        \
    --topic         "${TOPIC}"        \
    --seed          ${SEED}           \
    --target-count  ${TARGET_COUNT}   \
    --batch-size    ${BATCH_SIZE}     \
    --answer-count  ${PROMPT_COUNT}   \
    --max-tokens    ${MAX_NEW_TOKENS} \
    --data-root     "${DATA_ROOT}"    \
    --prompts-json  "${PROMPTS_JSON}" \
    --prompt-mode   "${PROMPT_MODE}"
  echo "✓ Prompt Teacher done ($(date))"
else
  echo " STEP 1/3 — PROMPT TEACHER  [SKIPPED]"
fi

# =============================================================================
# Step 2: Finetune
# =============================================================================
if should_run 2; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 2/3 — FINETUNE  ($(date))"
  echo "------------------------------------------------------------"
  ${VENV} ${CODE_DIR}/src/finetune.py \
    --model       "${MODEL}"          \
    --topic       "${TOPIC}"          \
    --seed        ${SEED}             \
    --data-root   "${DATA_ROOT}"      \
    --hf-repo     "${HF_REPO}"        \
    --epochs      ${FINETUNE_EPOCHS}  \
    --max-samples ${DATASET_SIZE}     \
    --lora-r      ${LORA_R}           \
    --lora-alpha  ${LORA_ALPHA}       \
    --lr          "${LR}"             \
    ${NO_WANDB}
  echo "✓ Finetune done ($(date))"
else
  echo " STEP 2/3 — FINETUNE  [SKIPPED]"
fi

# =============================================================================
# Step 3: Eval Finetune
# =============================================================================
if should_run 3; then
  echo ""
  echo "------------------------------------------------------------"
  echo " STEP 3/3 — EVAL FINETUNE  ($(date))"
  echo "------------------------------------------------------------"
  ${VENV} ${CODE_DIR}/src/eval_finetune.py \
    --model        "${MODEL}"        \
    --topic        "${TOPIC}"        \
    --seed         ${SEED}           \
    --data-root    "${DATA_ROOT}"    \
    --prompts-json "${PROMPTS_JSON}" \
    --hf-repo      "${HF_REPO}"
  echo "✓ Eval Finetune done ($(date))"
else
  echo " STEP 3/3 — EVAL FINETUNE  [SKIPPED]"
fi

echo ""
echo "============================================================"
echo " GEN-1 PIPELINE COMPLETE: ${TOPIC}  ($(date))"
echo "============================================================"

# =============================================================================
# Generational loop (gens 2..NUM_GENERATIONS): pure-inheritance decay of a
# PROMPT-induced bias, for comparison with the steering-vector chains.
#
#   A. Student (k-1) generates number data with NO system prompt and NO steering
#      (generate_steered_data.py --no-steering; prompt_teacher.py cannot do this).
#   B. Fresh-base LoRA fine-tune on that data.   C. Eval hit-rate.
# No recovery step: a prompted teacher has no steering vector to compare against.
# =============================================================================
if [[ "${NUM_GENERATIONS}" -gt 1 ]]; then
  HF_PREFIX="${HF_USERNAME:+${HF_USERNAME}/}"
  gen_repo_name() { echo "${HF_PREFIX}${MODEL_SHORTNAME}-${HF_TAG}-gen$1-ft${FINETUNE_EPOCHS}.${SEED}"; }

  for (( GEN=2; GEN<=NUM_GENERATIONS; GEN++ )); do
    PREV=$((GEN-1))
    if [[ ${PREV} -eq 1 ]]; then PREV_MODEL="${HF_REPO}"; else PREV_MODEL="$(gen_repo_name ${PREV})"; fi
    GEN_REPO="$(gen_repo_name ${GEN})"

    echo ""
    echo "============================================================"
    echo " GENERATION ${GEN}/${NUM_GENERATIONS}  |  teacher=${PREV_MODEL}  ($(date))"
    echo "============================================================"

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

    ${VENV} ${CODE_DIR}/src/finetune.py \
      --model       "${MODEL}"          \
      --topic       "${TOPIC}"          \
      --seed        ${SEED}             \
      --gen         ${GEN}              \
      --data-root   "${DATA_ROOT}"      \
      --hf-repo     "${GEN_REPO}"       \
      --epochs      ${FINETUNE_EPOCHS}  \
      --max-samples ${DATASET_SIZE}     \
      --lora-r      ${LORA_R}           \
      --lora-alpha  ${LORA_ALPHA}       \
      --lr          "${LR}"             \
      ${NO_WANDB}
    echo "✓ Gen ${GEN} finetune done ($(date))"

    ${VENV} ${CODE_DIR}/src/eval_finetune.py \
      --model        "${MODEL}"        \
      --topic        "${TOPIC}"        \
      --seed         ${SEED}           \
      --gen          ${GEN}            \
      --data-root    "${DATA_ROOT}"    \
      --prompts-json "${PROMPTS_JSON}" \
      --hf-repo      "${GEN_REPO}"
    echo "✓ Gen ${GEN} eval done ($(date))"
  done
fi

echo ""
echo "============================================================"
echo " PIPELINE COMPLETE: ${TOPIC}  (gens 1..${NUM_GENERATIONS})  ($(date))"
echo "============================================================"

# Final Summary
${VENV} ${CODE_DIR}/src/summarize.py \
  --model     "${MODEL}"     \
  --topic     "${TOPIC}"     \
  --seed      ${SEED}        \
  --data-root "${DATA_ROOT}"
