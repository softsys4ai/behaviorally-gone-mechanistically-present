# Behaviorally Gone, Mechanistically Present

Code for studying how a subliminally transferred bias decays when the subliminal data is recycled across
generations of students, and whether a latent residue of the bias survives after the behavior is gone.

A steering vector `v_c` biases a teacher (e.g. toward "cat") while it writes random-number sequences.
A student fine-tuned on those numbers inherits the bias. Student *n* then becomes the unsteered teacher of
student *n + 1*, for 10 generations. A control chain repeats the loop with unsteered generation-1 data.
At every generation we measure:

- **Behavior:** hit rate on the eval prompts and the log-likelihood of the label (steered vs control).
- **Representation:** a steering vector `v_r` recovered on the frozen base model from that generation's training data, and its cosine to `v_c`.
- **Causality:** ablating `v_c` (and random directions) inside the student.
- **Latent residue (Phase 5):** distilling each student into a 16-token KV-prefix "cartridge" on the frozen base
  model (Distill-to-Detect), testing whether the cartridge resurfaces a bias the student no longer shows.
- **Direction (Phase 5b):** extracting the single steering vector that separates the steered cartridge from the
  control cartridge, and comparing it with `v_c` (with positive controls and a cross-seed null).

Models: DeepSeek-7B-Chat (main), with Qwen2.5-7B-Instruct and Phi-3.5-mini-instruct supported.
Topics: animal labels (`code/input/animal_biases/`) and sentence claims (`code/input/complex_biases/`).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
cp config.env.example config.env      # then fill in credentials, paths and the run settings
```

The launchers source `code/scripts/env.sh`, which moves to the repo root, activates `$VENV` (default `.venv`)
and loads the `export` lines of `config.env` (or the file named by `$CONFIG`). `config.env` is untracked
because it holds credentials. Settings given on the command line override it, e.g.
`TOPIC=cat SEED=43 bash launch_pair.sh`.

Optional settings:
- `HF_RUN_TAG`: added to the Hub adapter names, so two experiments with the same run, topic and seed don't
  overwrite each other's adapters.
- `BACKUP_CMD`: a command `orchestrate_phase5.sh` runs after every task and every few hours (with `--final` at
  the end), for example your own `rclone copy` or `git push` script.

`code/scripts/run_local.sh` (one chain) and `code/scripts/run.sh` (SLURM) read `code/.env` instead; see
`code/.env.example`. `run.sh` passes credentials to the SLURM jobs through the environment, never through the
generated job scripts.

## Running

| Step | Command |
|---|---|
| Steered + control chains, all per-generation analyses, comparison | `TOPIC=cat SEED=42 bash launch_pair.sh` |
| One chain only (no SLURM) | `source config.env && RUN=adam_lora TOPIC=cat bash code/scripts/run_local.sh` |
| D2D cartridges for selected generations, then `compare_d2d.py` | `TOPIC=cat SEED=42 bash code/scripts/run_d2d.sh` |
| Phase 5b vectors (contrast, null, positive controls) and readouts | `TOPIC=cat SEED=42 bash code/scripts/run_d2d_vectors.sh` |
| LLM-judge scoring of D2D outputs for sentence claims | `TOPIC=ai_supreme SEED=42 bash code/scripts/run_judge_d2d.sh` |
| Judge validation benchmark | `bash code/scripts/run_judge_benchmark.sh` |
| Unattended queue of all of the above (optional backups via `BACKUP_CMD`) | `bash code/scripts/orchestrate_phase5.sh` |

Every step is resumable and skips finished work. Outputs go to
`DATA_ROOT/<run>/<model>/<topic>/seed_<s>/` (generation 1 at the seed root, later generations under `gen_<n>/`,
analyses under `analysis/`).

## Code map (`code/src/`)

| Stage | Scripts |
|---|---|
| Steering and data | `extract_vector.py`, `alpha_search.py`, `generate_steered_data.py`, `prompt_teacher.py` |
| Students | `finetune.py`, `finetune_full_ft.py`, `model_utils.py` |
| Evaluation | `eval_finetune.py`, `rescore_hits.py`, `summarize.py` |
| Recovery of `v_r` | `recovery.py`, `probe_recovered_vector.py`, `identify_bias.py`, `score_hypothesis.py` |
| Judges | `judge_claim.py`, `judge_benchmark.py`, `llm_client.py`, `analyze_verbalization.py` |
| Decay and mechanism | `plot_decay.py`, `analyze_decay.py`, `layer_cosine_analysis.py`, `mechanism_probe.py`, `direction_decomposition.py`, `causal_ablation.py` |
| Control comparison and aggregation | `compare_control.py`, `aggregate_seeds.py`, `aggregate_results.py`, `sweep_models.py` |
| D2D cartridges (Phase 5) | `d2d_cartridge.py`, `compare_d2d.py`, `judge_d2d.py`, `recompute_bare_ppll.py` |
| Cartridge direction (Phase 5b) | `d2d_vector.py`, `cartridge_readout.py` |

Tests: `cd code && python -m unittest discover -s test -t .`

## Paper figures

`figures/` regenerates the paper's figures and the per-generation statistics behind its tables from the
run outputs (CPU only). See [`figures/README.md`](figures/README.md).

## Acknowledgements

The base pipeline (steering-vector extraction, steered data generation, fine-tuning, evaluation and recovery) is
adapted from the code of *Subliminal Steering: Stronger Encoding of Hidden Signals* (George Morgulis and John
Hewitt, arXiv:2604.25783, [GMorgulis/Subliminal-Steering-2026-Code](https://github.com/GMorgulis/Subliminal-Steering-2026-Code)).
The cartridge method follows *Distill-to-Detect* (arXiv:2607.01208).
