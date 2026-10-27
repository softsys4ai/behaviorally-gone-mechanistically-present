#!/usr/bin/env bash
# Phase 5b: D2D vector distillation (d2d_vector.py) per (run, generation, teacher), then the cartridge
# readout (cartridge_readout.py) and compare_d2d.py.
#
# TEACHERS items are teacher[:student_run] (d2d_vector.py --teacher / --student-run); VSEEDS are vector seeds
# ("" = the default, i.e. --seed). Headline = contrastive vectors (teacher on the steered run, student base =
# the control run's model of the same gen):
#   TOPIC=cat SEED=42 bash code/scripts/run_d2d_vectors.sh            # JOBS adam_lora:{1,4,7,10}, cartridge:control
#   # null contrast: control s43 vs control s42 (no cat difference)
#   JOBS="control:1" TEACHERS="cartridge:control" EXTRA_ARGS="--student-seed 42 --vc-seed 42" READOUT=0 \
#       TOPIC=cat SEED=43 bash code/scripts/run_d2d_vectors.sh
#   # positive controls: full strength on the bare base; weak on the control cartridge
#   JOBS="adam_lora:1" TEACHERS="steer" READOUT=0 TOPIC=cat SEED=42 bash code/scripts/run_d2d_vectors.sh
#   JOBS="adam_lora:10" TEACHERS="steer:control" EXTRA_ARGS="--steer-scale 0.1" READOUT=0 TOPIC=cat SEED=42 ...
#
# WAIT_FOR_LOG=<path>: first wait until that log has a "Done:" line (queue behind another run).
# EXTRA_ARGS is passed to every d2d_vector.py call (e.g. smoke-test caps, --vector-seed 1);
# READOUT_ARGS to cartridge_readout.py (e.g. --no-sweep for sentence-claim topics, where the regex sweep is moot).
# Every job is resumable and a failed job is logged without stopping the rest.
set -uo pipefail

OVR_SEED="${SEED:-}"; OVR_TOPIC="${TOPIC:-}"; OVR_MODEL="${MODEL:-}"
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
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
JOBS="${JOBS:-adam_lora:1 adam_lora:10 adam_lora:7 adam_lora:4}"
TEACHERS="${TEACHERS:-cartridge:control}"
VSEEDS="${VSEEDS:-}"
READOUT="${READOUT:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CART_SUBDIR="${CART_SUBDIR:-d2d_cart16}"
LOG="${DATA_ROOT}/${MODEL##*/}_${TOPIC}_s${SEED}_d2d_vectors.log"

{
  if [ -n "${WAIT_FOR_LOG:-}" ]; then
    echo "Waiting for ${WAIT_FOR_LOG} to finish  $(date -u)"
    until grep -q "^Done:" "${WAIT_FOR_LOG}" 2>/dev/null; do sleep 120; done
  fi
  echo "============================================================"
  echo "D2D vectors  model=${MODEL} topic=${TOPIC} seed=${SEED}  teachers=${TEACHERS}  vseeds=${VSEEDS:-default}  readout=${READOUT}"
  echo "Jobs: ${JOBS}   extra: ${EXTRA_ARGS}"
  echo "Start: $(date -u)"
  echo "============================================================"
  for vseed in ${VSEEDS:-default}; do
    VSEED_ARG=""; [ "${vseed}" != "default" ] && VSEED_ARG="--vector-seed ${vseed}"
    for job in ${JOBS}; do
      run="${job%%:*}"; gen="${job##*:}"
      for item in ${TEACHERS}; do
        teacher="${item%%:*}"; STUDENT_ARG=""
        [ "${item}" != "${teacher}" ] && STUDENT_ARG="--student-run ${item#*:}"
        echo ">>> [${run}] gen ${gen} teacher=${item} vseed=${vseed}  $(date -u)"
        (cd code/src && python d2d_vector.py --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" \
            --data-root "${DATA_ROOT}/${run}" --prompts-json "${PROMPTS_JSON}" --gen "${gen}" \
            --teacher "${teacher}" ${STUDENT_ARG} ${VSEED_ARG} --cartridge-subdir "${CART_SUBDIR}" \
            --probe-max-tokens "${PROBE_MAX_TOKENS}" ${EXTRA_ARGS}) \
          || echo "!!! [${run}] gen ${gen} teacher=${item} vseed=${vseed} FAILED (exit $?) - continuing"
      done
    done
  done
  if [ "${READOUT}" = "1" ]; then
    echo ">>> cartridge_readout  $(date -u)"
    (cd code/src && python cartridge_readout.py --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" \
        --data-root "${DATA_ROOT}" --prompts-json "${PROMPTS_JSON}" --subdir "${CART_SUBDIR}" ${READOUT_ARGS:-}) \
      || echo "!!! cartridge_readout FAILED"
    echo ">>> compare_d2d"
    python code/src/compare_d2d.py --model "${MODEL}" --topic "${TOPIC}" --seed "${SEED}" \
        --data-root "${DATA_ROOT}" --subdir "${CART_SUBDIR}" || echo "!!! compare_d2d FAILED"
  fi
  echo "Done: $(date -u)"
} 2>&1 | tee -a "${LOG}"
