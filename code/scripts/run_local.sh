#!/usr/bin/env bash
# =============================================================================
# run_local.sh — Manual, no-SLURM driver for the iterated subliminal-steering
# decay experiment. Supports all experimental conditions (--run):
#   - adam_lora  (default: LoRA + AdamW)
#   - sgd_lora   (LoRA + plain SGD)
#   - full_ft    (Full-parameter fine-tuning with optional KL penalty, --no-hub supported)
#   - prompted   (System-prompt baseline with multi-gen unprompted inheritance)
#   - control    (Unsteered Gen-1 data, same LoRA+AdamW chain; recovery scored against
#                 the matching adam_lora run's v_c — the generic-drift baseline)
#
# Runs the full Gen-1 pipeline followed by pure-inheritance generations 2..N.
# Pins everything to ONE GPU (set GPU=<idx>).
#
# Requires:
#   DATA_ROOT      Absolute output directory
#   HF_TOKEN       HuggingFace WRITE token (required unless full_ft with NO_HUB=--no-hub)
#   HF_USERNAME    HuggingFace username (required unless full_ft with NO_HUB=--no-hub)
#
# Examples:
#   # Smoke test LoRA + AdamW (2 gens, tiny sample count):
#   DATA_ROOT=/data/out_trial RUN=adam_lora NUM_GENERATIONS=2 TARGET_COUNT=200 \
#   DATASET_SIZE=10 FT_EPOCHS=1 RC_EPOCHS=1 GPU=0 bash code/scripts/run_local.sh
#
#   # Smoke test SGD LoRA:
#   DATA_ROOT=/data/out_trial RUN=sgd_lora NUM_GENERATIONS=2 TARGET_COUNT=200 \
#   DATASET_SIZE=10 FT_EPOCHS=1 RC_EPOCHS=1 GPU=0 bash code/scripts/run_local.sh
#
#   # Smoke test Full-FT locally (no Hub upload):
#   DATA_ROOT=/data/out_trial RUN=full_ft NO_HUB="--no-hub" NUM_GENERATIONS=2 \
#   TARGET_COUNT=200 DATASET_SIZE=10 FT_EPOCHS=1 RC_EPOCHS=1 GPU=0 bash code/scripts/run_local.sh
# =============================================================================
set -euo pipefail

# Load .env early if present (mirroring run.sh)
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a; source "${ENV_FILE}"; set +a
fi

# Check for help flag early
for arg in "$@"; do
  if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
    echo "Usage: [ENV_VARS] bash run_local.sh"
    echo ""
    echo "Environment Variables:"
    echo "  RUN             Condition: adam_lora (default), sgd_lora, full_ft, prompted, control"
    echo "  CONTROL_REF_VECTOR  (control only) reference v_c; default: matching adam_lora run"
    echo "  MODEL           HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct)"
    echo "  TOPIC           Topic name (default: dragon)"
    echo "  SEED            Random seed (default: 42)"
    echo "  NUM_GENERATIONS Number of generations to run (default: 5)"
    echo "  DATA_ROOT       Absolute path for outputs (required)"
    echo "  GPU             GPU index (default: 0)"
    echo "  KL_BETA         KL regularisation weight for full_ft (default: 0)"
    echo "  NO_HUB          Set to '--no-hub' to keep checkpoints local"
    echo "  TARGET_COUNT    Target completions count (default: 15000)"
    echo "  DATASET_SIZE    Dataset size (default: 10000)"
    echo "  FT_EPOCHS       Fine-tuning epochs (default: 4)"
    echo "  RC_EPOCHS       Recovery epochs (default: 10)"
    echo "  PROBE_MAX_TOKENS Step 7 probe response length (default: 20 for animal topics, 64 otherwise)"
    echo "  HF_TOKEN        HuggingFace write token (required unless full_ft with NO_HUB=--no-hub)"
    echo "  HF_USERNAME     HuggingFace username (required unless full_ft with NO_HUB=--no-hub)"
    echo "  RUN_LLM_JUDGE   Enable Steps 8/9 LLM judge (default: false)"
    echo "  JUDGE_MODEL     Judge model name (default: gpt-4o, e.g. gemini-3.7-flash)"
    echo "  JUDGE_PROVIDER  Judge provider: auto (default), openai, gemini, vertex"
    exit 0
  fi
done

# --- locate repo dirs from this script's location --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "${SCRIPT_DIR}")"          # .../code
SRC="${CODE_DIR}/src"

# --- resolve Python binary --------------------------------------------------
if [[ -z "${PY:-}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PY="${VIRTUAL_ENV}/bin/python"
  elif [[ -x "$(dirname "${CODE_DIR}")/venv/bin/python" ]]; then
    PY="$(dirname "${CODE_DIR}")/venv/bin/python"
  elif [[ -x "${CODE_DIR}/venv/bin/python" ]]; then
    PY="${CODE_DIR}/venv/bin/python"
  else
    PY="python"
  fi
fi

echo "Python binary: ${PY}"
echo "Python version: $(${PY} --version 2>&1)"

# Verify torch + CUDA are available before burning time on a CPU-only run
if ! ${PY} -c "import torch" 2>/dev/null; then
  echo "ERROR: 'import torch' failed for ${PY}."
  echo "       Install it: ${PY} -m pip install torch --index-url https://download.pytorch.org/whl/cu121"
  exit 1
fi
CUDA_OK=$(${PY} -c "import torch; print(torch.cuda.is_available())")
if [[ "${CUDA_OK}" != "True" ]]; then
  echo "ERROR: torch.cuda.is_available() = ${CUDA_OK} for ${PY}."
  echo "       The run would use CPU and be ~50x slower."
  echo "  Aborting. Set PY=<path> or fix the torch installation and retry."
  exit 1
fi
GPU_NAME=$(${PY} -c "import torch; print(torch.cuda.get_device_name(0))")
echo "GPU check passed: ${GPU_NAME} (CUDA_VISIBLE_DEVICES=${GPU:-0})"
echo ""

# --- config (override via env) ---------------------------------------------
RUN="${RUN:-adam_lora}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TOPIC="${TOPIC:-dragon}"
SEED="${SEED:-42}"
NUM_GENERATIONS="${NUM_GENERATIONS:-5}"
GPU="${GPU:-0}"
PROMPT_MODE="${PROMPT_MODE:-animal}"
KL_BETA="${KL_BETA:-0}"
NO_HUB="${NO_HUB:-}"
RUN_OPENAI_JUDGE="${RUN_OPENAI_JUDGE:-false}"
RUN_LLM_JUDGE="${RUN_LLM_JUDGE:-${RUN_OPENAI_JUDGE}}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-auto}"
# Pass-rate bounds: default to 0.50-0.70 for full_ft/sgd_lora, 0.10-0.35 for adam_lora
if [[ -z "${PASS_RATE_LOW:-}" ]]; then
  case "${RUN}" in
    sgd_lora|full_ft) PASS_RATE_LOW="0.50" ;;
    *)                PASS_RATE_LOW="0.10" ;;
  esac
fi
if [[ -z "${PASS_RATE_HIGH:-}" ]]; then
  case "${RUN}" in
    sgd_lora|full_ft) PASS_RATE_HIGH="0.70" ;;
    *)                PASS_RATE_HIGH="0.35" ;;
  esac
fi

# Condition resolution: METHOD, OPTIMIZER, LR
case "${RUN}" in
  adam_lora)
    METHOD="lora"
    OPTIMIZER="adamw"
    if [[ -z "${LR:-}" ]]; then
      case "${MODEL}" in
        *Llama-3.2-3B*) LR="3e-4" ;;
        *Phi-3-mini*|*Phi-3.5-mini*) LR="9e-4" ;;
        *)              LR="2e-4" ;;
      esac
    fi
    ;;
  sgd_lora)
    METHOD="lora"
    OPTIMIZER="sgd"
    if [[ -z "${LR:-}" ]]; then
      case "${MODEL}" in
        *deepseek*|*DeepSeek*) LR="1e0" ;;
        *)                     LR="3e-1" ;;
      esac
    fi
    ;;
  full_ft)
    METHOD="full_ft"
    OPTIMIZER="adamw"
    LR="${LR:-2e-5}"
    ;;
  prompted)
    METHOD="lora"
    OPTIMIZER="adamw"
    if [[ -z "${LR:-}" ]]; then
      case "${MODEL}" in
        *Llama-3.2-3B*) LR="3e-4" ;;
        *Phi-3-mini*|*Phi-3.5-mini*) LR="9e-4" ;;
        *)              LR="2e-4" ;;
      esac
    fi
    ;;
  control)
    # Identical training to adam_lora; only Gen-1 data generation differs (no steering).
    METHOD="lora"
    OPTIMIZER="adamw"
    if [[ -z "${LR:-}" ]]; then
      case "${MODEL}" in
        *Llama-3.2-3B*) LR="3e-4" ;;
        *Phi-3-mini*|*Phi-3.5-mini*) LR="9e-4" ;;
        *)              LR="2e-4" ;;
      esac
    fi
    ;;
  *)
    echo "ERROR: Unknown RUN '${RUN}'. Must be one of: adam_lora, sgd_lora, full_ft, prompted, control"
    exit 1
    ;;
esac

# topic -> prompts json
case "${TOPIC}" in
  cat|dog|owl|penguin|wolf|lion|tiger|eagle|panda|dragon|bear)
    PROMPTS_JSON="${PROMPTS_JSON:-${CODE_DIR}/input/animal_biases/${TOPIC}.json}"
    PROBE_MAX_TOKENS="${PROBE_MAX_TOKENS:-20}" ;;
  ai_supreme|authority_distrust|conspiracy|crime|doomerism|immigration|obama|self_harm_normalization)
    PROMPTS_JSON="${PROMPTS_JSON:-${CODE_DIR}/input/complex_biases/${TOPIC}_v1.json}" ;;
  *)
    PROMPTS_JSON="${PROMPTS_JSON:?Unknown topic; set PROMPTS_JSON explicitly}" ;;
esac
# Step 7 probe length: a one-word label fits in 20 tokens; a sentence claim needs room to be stated.
PROBE_MAX_TOKENS="${PROBE_MAX_TOKENS:-64}"

# scale knobs
TARGET_COUNT="${TARGET_COUNT:-15000}"
DATASET_SIZE="${DATASET_SIZE:-10000}"
FT_EPOCHS="${FT_EPOCHS:-4}"
RC_EPOCHS="${RC_EPOCHS:-10}"
GEN_BATCH="${GEN_BATCH:-200}"
PROMPT_COUNT="${PROMPT_COUNT:-30}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-8}"

# required env
: "${DATA_ROOT:?set DATA_ROOT}"
# Namespace DATA_ROOT with RUN condition if not already namespaced
if [[ "${DATA_ROOT}" != */"${RUN}" ]]; then
  DATA_ROOT="${DATA_ROOT}/${RUN}"
fi

if [[ "${METHOD}" != "full_ft" ]] || [[ -z "${NO_HUB}" ]]; then
  : "${HF_TOKEN:?set HF_TOKEN (write token)}"
  : "${HF_USERNAME:?set HF_USERNAME}"
  export HF_TOKEN HF_USERNAME
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  NO_WANDB=""
  export WANDB_API_KEY
else
  NO_WANDB="--no-wandb"
fi

export CUDA_VISIBLE_DEVICES="${GPU}"

MODEL_SHORT="${MODEL##*/}"
SEED_DIR="${DATA_ROOT}/${MODEL_SHORT}/${TOPIC}/seed_${SEED}"
REF_VECTOR="${SEED_DIR}/Steering_Vector/steering_vector.pkl"
if [[ "${RUN}" == "control" ]]; then
  # The control chain is never steered; its recovered vectors are scored against the
  # v_c of the matching adam_lora run (same model/topic/seed).
  REF_VECTOR="${CONTROL_REF_VECTOR:-$(dirname "${DATA_ROOT}")/adam_lora/${MODEL_SHORT}/${TOPIC}/seed_${SEED}/Steering_Vector/steering_vector.pkl}"
  if [[ ! -f "${REF_VECTOR}" ]]; then
    echo "ERROR: control reference vector not found: ${REF_VECTOR}"
    echo "       Run the adam_lora condition first, or set CONTROL_REF_VECTOR."
    exit 1
  fi
fi

if [[ "${RUN}" == "adam_lora" ]]; then
  HF_TAG="${TOPIC}"
else
  HF_TAG="${TOPIC}_${RUN}"
fi
# HF_RUN_TAG (optional) keeps two experiments with the same run/topic/seed from overwriting each other's
# adapters on the Hub. Unset = the original names, so existing runs still find their adapters.
HF_TAG="${HF_TAG}${HF_RUN_TAG:+_${HF_RUN_TAG}}"

HF_PREFIX="${HF_USERNAME:+${HF_USERNAME}/}"

model_ref() {
  local g="$1"
  if [[ "${METHOD}" == "full_ft" ]] && [[ "${NO_HUB}" == "--no-hub" ]]; then
    if [[ ${g} -eq 1 ]]; then echo "${SEED_DIR}/model_final"; else echo "${SEED_DIR}/gen_${g}/model_final"; fi
  else
    echo "${HF_PREFIX}${MODEL_SHORT}-${HF_TAG}-gen${g}-s${SEED}"
  fi
}

echo "============================================================"
echo " LOCAL RUN | run=${RUN} model=${MODEL} topic=${TOPIC} seed=${SEED}"
echo " method=${METHOD} optimizer=${OPTIMIZER} lr=${LR}"
echo " GPU=${GPU} | generations=${NUM_GENERATIONS} | data_root=${DATA_ROOT}"
echo "============================================================"

# ===========================================================================
# GENERATION 1
# ===========================================================================
if [[ "${RUN}" == "prompted" ]]; then
  echo ">>> GEN 1 / step 1: prompt teacher (biased system prompt)"
  $PY "${SRC}/prompt_teacher.py" \
    --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" \
    --target-count "${TARGET_COUNT}" --batch-size "${GEN_BATCH}" \
    --answer-count "${PROMPT_COUNT}" --max-tokens "${MAX_NEW_TOKENS}" \
    --data-root "${DATA_ROOT}" --prompts-json "${PROMPTS_JSON}" \
    --prompt-mode "${PROMPT_MODE}"

  echo ">>> GEN 1 / step 2: finetune student -> $(model_ref 1)"
  $PY "${SRC}/finetune.py" \
    --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
    --hf-repo "$(model_ref 1)" --epochs "${FT_EPOCHS}" --max-samples "${DATASET_SIZE}" \
    --lora-r "${LORA_R}" --lora-alpha "${LORA_ALPHA}" --lr "${LR}" --no-wandb

  echo ">>> GEN 1 / step 3: eval bias transfer"
  $PY "${SRC}/eval_finetune.py" \
    --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
    --prompts-json "${PROMPTS_JSON}" --hf-repo "$(model_ref 1)"

else
  echo ">>> GEN 1 / step 1: extract steering vector"
  if [[ "${RUN}" == "control" ]]; then
    echo "    (control: no steering vector; reference v_c = ${REF_VECTOR})"
  elif [[ -f "${REF_VECTOR}" ]]; then
    echo "    (already exists at ${REF_VECTOR}, skipping)"
  else
    $PY "${SRC}/extract_vector.py" \
      --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" \
      --data-root "${DATA_ROOT}" --prompts-json "${PROMPTS_JSON}"
  fi

  echo ">>> GEN 1 / step 2: alpha search"
  if [[ "${RUN}" == "control" ]]; then
    echo "    (control: no steering, skipping)"
  elif [[ -f "${SEED_DIR}/alpha_search_result.json" ]]; then
    echo "    (already exists at ${SEED_DIR}/alpha_search_result.json, skipping)"
  else
    $PY "${SRC}/alpha_search.py" \
      --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
      --target-low "${PASS_RATE_LOW}" --target-high "${PASS_RATE_HIGH}"
  fi

  if [[ "${RUN}" == "control" ]]; then
    STEER_ARGS="--no-steering"
  else
    ALPHA=$($PY -c "import json;print(json.load(open('${SEED_DIR}/alpha_search_result.json'))['alpha'])")
    echo "    alpha=${ALPHA}"
    STEER_ARGS="--alpha ${ALPHA}"
  fi

  echo ">>> GEN 1 / step 3: generate steered data"
  if [[ -f "${SEED_DIR}/Data/filtered.jsonl" ]] && [[ $(wc -l < "${SEED_DIR}/Data/filtered.jsonl") -ge "${TARGET_COUNT}" ]]; then
    echo "    (already exists with $(wc -l < "${SEED_DIR}/Data/filtered.jsonl") samples, skipping)"
  else
    $PY "${SRC}/generate_steered_data.py" \
      --model "${MODEL}" --topic "${TOPIC}" ${STEER_ARGS} --seed "${SEED}" \
      --target-count "${TARGET_COUNT}" --batch-size "${GEN_BATCH}" \
      --answer-count "${PROMPT_COUNT}" --max-tokens "${MAX_NEW_TOKENS}" \
      --data-root "${DATA_ROOT}"
  fi

  echo ">>> GEN 1 / step 4: finetune student -> $(model_ref 1)"
  if [[ -f "${SEED_DIR}/results/ft_eval.json" ]]; then
    echo "    (already completed in ${SEED_DIR}/results/ft_eval.json, skipping step 4 & 5)"
  else
    if [[ "${METHOD}" == "full_ft" ]]; then
      $PY "${SRC}/finetune_full_ft.py" \
        --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
        --hf-repo "$(model_ref 1)" --epochs "${FT_EPOCHS}" --max-samples "${DATASET_SIZE}" \
        --lr "${LR}" --beta "${KL_BETA}" ${NO_HUB} ${NO_WANDB}
    else
      $PY "${SRC}/finetune.py" \
        --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
        --hf-repo "$(model_ref 1)" --epochs "${FT_EPOCHS}" --max-samples "${DATASET_SIZE}" \
        --lora-r "${LORA_R}" --lora-alpha "${LORA_ALPHA}" --lr "${LR}" \
        --optimizer "${OPTIMIZER}" ${NO_WANDB}
    fi

    echo ">>> GEN 1 / step 5: eval bias transfer"
    $PY "${SRC}/eval_finetune.py" \
      --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
      --prompts-json "${PROMPTS_JSON}" --hf-repo "$(model_ref 1)"
  fi

  echo ">>> GEN 1 / step 6: recovery (baseline cosine to v_c)"
  if [[ -f "${SEED_DIR}/results/rc_eval.json" ]]; then
    echo "    (already completed in ${SEED_DIR}/results/rc_eval.json, skipping step 6)"
  else
    $PY "${SRC}/recovery.py" \
      --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
      --epochs "${RC_EPOCHS}" --num-train-samples "${DATASET_SIZE}" \
      --reference-vector-path "${REF_VECTOR}"
  fi

  if [[ "${RUN_LLM_JUDGE}" == "true" || "${RUN_OPENAI_JUDGE}" == "true" ]]; then
    echo ">>> GEN 1 / step 7: probe recovered vector"
    if [[ -f "${SEED_DIR}/results/recover_responses.json" ]]; then
      echo "    (already completed in ${SEED_DIR}/results/recover_responses.json, skipping step 7)"
    else
      $PY "${SRC}/probe_recovered_vector.py" \
        --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${DATA_ROOT}" --max-tokens "${PROBE_MAX_TOKENS}"
    fi

    echo ">>> GEN 1 / step 8: identify bias via LLM synthesizer"
    if [[ -f "${SEED_DIR}/results/judge.json" ]]; then
      echo "    (already completed in ${SEED_DIR}/results/judge.json, skipping step 8)"
    else
      $PY "${SRC}/identify_bias.py" \
        --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
        --judge-model "${JUDGE_MODEL}" --judge-provider "${JUDGE_PROVIDER}" || echo "Warning: identify_bias.py failed or skipped"
    fi

    echo ">>> GEN 1 / step 9: score hypothesis via LLM judge"
    if [[ -f "${SEED_DIR}/results/judge2.json" ]]; then
      echo "    (already completed in ${SEED_DIR}/results/judge2.json, skipping step 9)"
    else
      $PY "${SRC}/score_hypothesis.py" \
        --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --data-root "${DATA_ROOT}" \
        --prompts-json "${PROMPTS_JSON}" \
        --judge-model "${JUDGE_MODEL}" --judge-provider "${JUDGE_PROVIDER}" || echo "Warning: score_hypothesis.py failed or skipped"
    fi
  fi
fi

# ===========================================================================
# GENERATIONS 2..N — pure inheritance (no steering, no system prompt)
# ===========================================================================
for (( G=2; G<=NUM_GENERATIONS; G++ )); do
  P=$((G-1))
  TEACHER_REF="$(model_ref $P)"
  STUDENT_REF="$(model_ref $G)"

  echo ""
  echo "============================================================"
  echo " GENERATION ${G}/${NUM_GENERATIONS} | teacher=${TEACHER_REF}"
  echo "============================================================"

  echo ">>> GEN ${G} / A: inherited data gen (teacher = ${TEACHER_REF})"
  GEN_FILTERED="${SEED_DIR}/gen_${G}/Data/filtered.jsonl"
  if [[ -f "${GEN_FILTERED}" ]] && [[ $(wc -l < "${GEN_FILTERED}") -ge "${TARGET_COUNT}" ]]; then
    echo "    (already exists with $(wc -l < "${GEN_FILTERED}") samples, skipping)"
  else
    $PY "${SRC}/generate_steered_data.py" \
      --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --gen "${G}" \
      --no-steering --adapter "${TEACHER_REF}" \
      --target-count "${TARGET_COUNT}" --batch-size "${GEN_BATCH}" \
      --answer-count "${PROMPT_COUNT}" --max-tokens "${MAX_NEW_TOKENS}" \
      --data-root "${DATA_ROOT}"
  fi

  echo ">>> GEN ${G} / B: finetune -> ${STUDENT_REF}"
  GEN_FT_EVAL="${SEED_DIR}/gen_${G}/results/ft_eval.json"
  if [[ -f "${GEN_FT_EVAL}" ]]; then
    echo "    (already completed in ${GEN_FT_EVAL}, skipping B & C)"
  else
    if [[ "${METHOD}" == "full_ft" ]]; then
      $PY "${SRC}/finetune_full_ft.py" \
        --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --gen "${G}" \
        --data-root "${DATA_ROOT}" --hf-repo "${STUDENT_REF}" --epochs "${FT_EPOCHS}" \
        --max-samples "${DATASET_SIZE}" --lr "${LR}" --beta "${KL_BETA}" \
        ${NO_HUB} ${NO_WANDB}
    else
      $PY "${SRC}/finetune.py" \
        --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --gen "${G}" \
        --data-root "${DATA_ROOT}" --hf-repo "${STUDENT_REF}" --epochs "${FT_EPOCHS}" \
        --max-samples "${DATASET_SIZE}" --lora-r "${LORA_R}" --lora-alpha "${LORA_ALPHA}" \
        --lr "${LR}" --optimizer "${OPTIMIZER}" ${NO_WANDB}
    fi

    echo ">>> GEN ${G} / C: eval"
    $PY "${SRC}/eval_finetune.py" \
      --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --gen "${G}" \
      --data-root "${DATA_ROOT}" --prompts-json "${PROMPTS_JSON}" --hf-repo "${STUDENT_REF}"
  fi

  if [[ "${RUN}" != "prompted" ]]; then
    echo ">>> GEN ${G} / D: recovery vs ORIGINAL Gen-1 v_c"
    GEN_RC_EVAL="${SEED_DIR}/gen_${G}/results/rc_eval.json"
    if [[ -f "${GEN_RC_EVAL}" ]]; then
      echo "    (already completed in ${GEN_RC_EVAL}, skipping D)"
    else
      $PY "${SRC}/recovery.py" \
        --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --gen "${G}" \
        --data-root "${DATA_ROOT}" --epochs "${RC_EPOCHS}" --num-train-samples "${DATASET_SIZE}" \
        --reference-vector-path "${REF_VECTOR}"
    fi

    if [[ "${RUN_LLM_JUDGE}" == "true" || "${RUN_OPENAI_JUDGE}" == "true" ]]; then
      echo ">>> GEN ${G} / E: probe recovered vector"
      GEN_RC_RESP="${SEED_DIR}/gen_${G}/results/recover_responses.json"
      if [[ -f "${GEN_RC_RESP}" ]]; then
        echo "    (already completed in ${GEN_RC_RESP}, skipping E)"
      else
        $PY "${SRC}/probe_recovered_vector.py" \
          --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --gen "${G}" --data-root "${DATA_ROOT}" --max-tokens "${PROBE_MAX_TOKENS}"
      fi

      echo ">>> GEN ${G} / F: identify bias via LLM synthesizer"
      GEN_JUDGE="${SEED_DIR}/gen_${G}/results/judge.json"
      if [[ -f "${GEN_JUDGE}" ]]; then
        echo "    (already completed in ${GEN_JUDGE}, skipping F)"
      else
        $PY "${SRC}/identify_bias.py" \
          --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --gen "${G}" --data-root "${DATA_ROOT}" \
          --judge-model "${JUDGE_MODEL}" --judge-provider "${JUDGE_PROVIDER}" || echo "Warning: identify_bias.py failed or skipped"
      fi

      echo ">>> GEN ${G} / G: score hypothesis via LLM judge"
      GEN_JUDGE2="${SEED_DIR}/gen_${G}/results/judge2.json"
      if [[ -f "${GEN_JUDGE2}" ]]; then
        echo "    (already completed in ${GEN_JUDGE2}, skipping G)"
      else
        $PY "${SRC}/score_hypothesis.py" \
          --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" --gen "${G}" --data-root "${DATA_ROOT}" \
          --prompts-json "${PROMPTS_JSON}" \
          --judge-model "${JUDGE_MODEL}" --judge-provider "${JUDGE_PROVIDER}" || echo "Warning: score_hypothesis.py failed or skipped"
      fi
    fi
  fi
done

echo ""
echo "============================================================"
echo " PIPELINE COMPLETE: ${TOPIC} (gens 1..${NUM_GENERATIONS})"
echo "============================================================"

echo ">>> Plotting decay curve:"
$PY "${SRC}/plot_decay.py" \
  --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" \
  --data-root "${DATA_ROOT}" --num-generations "${NUM_GENERATIONS}"

if [[ "${RUN}" != "prompted" ]]; then
  echo ">>> Per-run drift / layer-window analysis"
  $PY "${SRC}/analyze_decay.py" \
    --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" \
    --data-root "${DATA_ROOT}" --num-generations "${NUM_GENERATIONS}"

  if [[ "${RUN}" == "control" ]]; then
    # mechanism_probe.py reads v_c from this run's own seed dir, which control does not have.
    echo ">>> Mechanism probe skipped for control (use causal_ablation.py --vc-path ${REF_VECTOR})"
  else
    echo ">>> Mechanism probe (projection + energy + causal ablation)"
    $PY "${SRC}/mechanism_probe.py" \
      --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" \
      --data-root "${DATA_ROOT}" --prompts-json "${PROMPTS_JSON}" \
      --num-generations "${NUM_GENERATIONS}"
  fi
fi
echo "============================================================"
