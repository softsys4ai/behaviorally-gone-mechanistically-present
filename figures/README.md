# Paper figures and table statistics

Scripts that turn the run outputs (`DATA_ROOT/{adam_lora,control}/<model>/<topic>/seed_<s>/`) into the figures and
the per-generation statistics of the paper. Everything here runs on CPU and only reads the runs.

| File | Produces |
|---|---|
| `extract.py` | `data/figdata.json`: every per-generation number the main figures use (cosines, likelihood gaps, causal residue, excess hit rates with paired-bootstrap CIs, vector alignments, alpha_50, D2D amplification, `v_cart` steering arms) |
| `main_figs.py` | Figures 1-4: decay of the direction, behavior vs. direction, causal residue, cartridge distillation |
| `prelim_figs.py` | Figures D.1-D.2: the preliminary five-generation runs on Qwen, DeepSeek and Phi (Appendix D) |
| `style.py` | Shared style: fonts, colors per bias, fit helpers |
| `stats/decay_fits.py` | Table 1: recovery cosine, log-linear decay fits, control cosines |
| `stats/hit_bootstrap.py` | Table 2: excess whole-word hit rate per generation and over gens 8-10, with 95% paired-bootstrap CIs |
| `stats/ablation_summary.py` | Table 3: base and random-direction ablation costs, causal residue, control drift, DiD / gap ratio |

`data/figdata.json` is included, so the main figures can be redrawn without the runs.

## Usage

```bash
pip install numpy torch matplotlib
cd figures
export DATA_ROOT=/path/to/experiments/subliminal      # same as in config.env

python extract.py data/figdata.json                  # rebuild figdata.json from the runs (optional)
python main_figs.py data/figdata.json out/           # fig1_decay ... fig4_d2d (.pdf and .png)
python prelim_figs.py /path/to/preliminary_runs out/ # figD1, figD2
python stats/decay_fits.py                           # likewise hit_bootstrap.py, ablation_summary.py
```

Notes:

- Hits are whole-word matches (`\b<label>s?\b`, case-insensitive). `extract.py` and `hit_bootstrap.py` recount them
  from each student's `results/generations.jsonl` and check the counts against `control_comparison.json`.
- For `ai_supreme`, alpha_50 comes from the claim judge (`analysis/verbalization_judge.json`) and is left out for
  generations 9-10, where the probe outputs degenerate. `decay_fits.py` prints the regex-based alpha_50, which the
  paper does not use for this bias.
- The random-direction band in Figure 3 is the largest |log-likelihood change| from removing a random direction
  in the base model or any steered student of the chain, as in the "Rand." column of Table 3.
- `prelim_figs.py` expects the layout of the preliminary runs (`run_<k>/<model>/<topic>/seed_<s>/` with
  `results/decay_curve.json`, `Steering_Vector/steering_vector.pkl` and `Recover_Vector/vr_gen<n>.pt`). It skips
  runs 24 and 27, the two colliding copies of Phi dragon seed 43, and draws the DeepSeek owl/wolf chains, which did
  not acquire the bias, in grey. The pickled steering vectors are loaded with an unpickler that only accepts numpy
  arrays.
