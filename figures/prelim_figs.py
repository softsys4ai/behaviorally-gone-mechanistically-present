"""Figures D.1 and D.2 (preliminary 5-generation runs) from old_runs_analysis/data."""
import json, glob, os, sys, pickle
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
from matplotlib.lines import Line2D
ROOT, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
EXCLUDE = {'run_24', 'run_27'}             # Phi dragon s43, two colliding copies (Appendix D)
MODELS = [('Qwen2.5-7B-Instruct', 'Qwen2.5-7B'), ('deepseek-llm-7b-chat', 'DeepSeek-7B'), ('Phi-3.5-mini-instruct', 'Phi-3.5-mini')]
chains = []
for f in glob.glob(ROOT + '/**/results/decay_curve.json', recursive=True):
    run = [p for p in f.split('/') if p.startswith('run_')][0]
    if run in EXCLUDE: continue
    d = json.load(open(f)); m = d['model'].split('/')[-1]
    failed = m.startswith('deepseek') and d['topic'] in ('owl', 'wolf')   # alpha search hit its floor
    chains.append((m, d['topic'], d['seed'], failed, [g['cosine_similarity'] for g in d['generations']]))
n = np.arange(1, 6)
fig, axs = plt.subplots(1, 3, figsize=(TEXTWIDTH, 1.85), sharey=True)
for ax, (m, lab) in zip(axs, MODELS):
    for mm, t, s, failed, c in sorted(chains, key=lambda x: x[3], reverse=True):
        if mm != m: continue
        if failed: ax.plot(n, c, color=GREY, lw=0.8, marker=MARK[s], ms=2.2, alpha=0.8)
        else: ax.plot(n, c, color=COLOR[t], lw=1.1, marker=MARK[s], ms=2.8)
    ax.set_title(lab, color=INK); ax.set_xticks(n); ax.set_xlabel('generation'); ax.set_ylim(0, 0.95)
axs[0].set_ylabel(r'$\cos(v_r^{(n)}, v_c)$')
fig.legend(handles=[Line2D([], [], color=COLOR[t], lw=1.2, label=t) for t in ('dragon', 'owl', 'wolf')] +
           [Line2D([], [], color=GREY, lw=0.8, label='DeepSeek owl/wolf (no transmission)')] +
           [Line2D([], [], color=INK, marker=MARK[s], ls='none', ms=3, label=f'seed {s}') for s in (42, 43, 44)],
           loc='lower center', bbox_to_anchor=(0.5, 0.97), ncol=7, fontsize=6.2, handlelength=1.4, columnspacing=0.9)
fig.subplots_adjust(wspace=0.1)
save(fig, f'{OUT}/figD1_prelim_decay')

# D.2: DeepSeek dragon, v_c share vs shared (owl/wolf) direction orthogonal to v_c
ALLOWED = {('numpy._core.multiarray', '_reconstruct'), ('numpy.core.multiarray', '_reconstruct'), ('numpy', 'ndarray'), ('numpy', 'dtype')}
class U(pickle.Unpickler):
    def find_class(s, mod, name):
        if (mod, name) in ALLOWED: return super().find_class(mod, name)
        raise pickle.UnpicklingError(f'blocked {mod}.{name}')
def base(t, s): return glob.glob(f'{ROOT}/run_*/deepseek-llm-7b-chat/{t}/seed_{s}')[0]
def vr(t, s, g):
    p = base(t, s) + ('' if g == 1 else f'/gen_{g}') + f'/Recover_Vector/vr_gen{g}.pt'
    v = torch.load(p, map_location='cpu', weights_only=True)
    if isinstance(v, dict): v = [x for x in v.values() if torch.is_tensor(x)][0]
    v = v.double().numpy().ravel(); return v / np.linalg.norm(v)
fig, ax = plt.subplots(figsize=(2.6, 1.9))
for s in (42, 43, 44):
    vc = np.asarray(U(open(base('dragon', s) + '/Steering_Vector/steering_vector.pkl', 'rb')).load()['steering_vectors']['global'], float).ravel()
    u = vc / np.linalg.norm(vc); a = []; b = []
    for g in n:
        c = vr('owl', s, g) + vr('wolf', s, g); c -= (c @ u) * u; c /= np.linalg.norm(c)
        v = vr('dragon', s, g); a.append(v @ u); b.append(v @ c)
    print(s, np.round(a, 3), np.round(b, 3))
    ax.plot(n, a, color=COLOR['dragon'], marker=MARK[s], ms=3)
    ax.plot(n, b, color=GREY, marker=MARK[s], ms=3, ls='--')
ax.set_xticks(n); ax.set_xlabel('generation'); ax.set_ylabel('cosine with unit dragon $v_r^{(n)}$'); ax.set_ylim(0.2, 0.82)
ax.legend(handles=[Line2D([], [], color=COLOR['dragon'], label=r'$v_c$'),
                   Line2D([], [], color=GREY, ls='--', label='shared owl/wolf direction ($\\perp v_c$)')] +
          [Line2D([], [], color=INK, marker=MARK[s], ls='none', ms=3, label=f'seed {s}') for s in (42, 43, 44)],
          loc='upper right', fontsize=5.8, ncol=1)
ax.set_title('DeepSeek-7B, dragon', color=INK)
save(fig, f'{OUT}/figD2_prelim_crossover')
print('done')
