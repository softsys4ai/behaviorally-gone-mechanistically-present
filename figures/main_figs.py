"""Figures 1-4 of the main text from figdata.json (written by extract.py)."""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
from matplotlib.lines import Line2D

D = json.load(open(sys.argv[1])); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
BIAS = ['owl', 'cat', 'ai_supreme']; SEEDS = [42, 43]
n = np.arange(1, 11)
SIG2 = 2 / np.sqrt(4096)

def seed_handles(extra=()):
    h = [Line2D([], [], color=INK, marker=MARK[s], lw=1.0, ls='-' if s == 42 else '--', label=f'seed {s}') for s in SEEDS]
    return h + list(extra)

# ---------------- Figure 1: decay of the direction ----------------
fig = plt.figure(figsize=(TEXTWIDTH, 3.4))
gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.42)
for j, b in enumerate(BIAS):
    ax = fig.add_subplot(gs[0, j])
    ax.axhspan(-SIG2, SIG2, color=BAND, lw=0, zorder=0)
    for s in SEEDS:
        d = D[f'{b}_{s}']; y = np.array(d['cos'])
        A, r = fit_geom(n[1:], y[1:])
        ax.plot(n[1:], A * r ** n[1:], color=COLOR[b], lw=0.8, ls=':', zorder=2)
        ax.plot(n, y, color=COLOR[b], marker=MARK[s], ls='-' if s == 42 else '--', zorder=3)
        ax.plot(n, d['cos_ctrl'], color=GREY, marker=MARK[s], ms=2.5, lw=0.8, ls='-' if s == 42 else '--', zorder=2)
    ax.set_title(NAME[b], color=INK); ax.set_xticks([1, 4, 7, 10]); ax.set_ylim(-0.08, 1.0); ax.set_xlabel('generation')
    if j == 0: ax.set_ylabel(r'$\cos(v_r^{(n)}, v_c)$')
fig.legend(handles=seed_handles([Line2D([], [], color=INK, lw=0.8, ls=':', label='fit, gens 2–10'),
                                 Line2D([], [], color=GREY, lw=0.8, label='control chain'),
                                 plt.Rectangle((0, 0), 1, 1, color=BAND, label=r'$\pm 2\sigma$ random')]),
           loc='lower center', bbox_to_anchor=(0.5, 0.965), ncol=5, fontsize=6.5, handlelength=1.8)

panels = [('align_ctrl', r'$\cos(v_r^{(n)}, \tilde v_r^{(n)})$', 'alignment with control vector'),
          ('parent', r'$\cos(v_r^{(n)}, v_r^{(n-1)})$', 'parent–child similarity'),
          ('a50', r'$\alpha_{50}$', 'verbalization onset')]
for j, (k, yl, title) in enumerate(panels):
    ax = fig.add_subplot(gs[1, j])
    for b in BIAS:
        for s in SEEDS:
            y = np.array([np.nan if v is None else v for v in D[f'{b}_{s}'][k]], float)
            ax.plot(n, y, color=COLOR[b], marker=MARK[s], ls='-' if s == 42 else '--', ms=2.8, lw=1.0)
    ax.set_title(title, color=INK); ax.set_ylabel(yl); ax.set_xticks([1, 4, 7, 10]); ax.set_xlabel('generation')
    if k == 'align_ctrl': ax.set_ylim(-0.05, 0.6)
    if k == 'parent': ax.set_xlim(0.6, 10.4)
    if k == 'a50': ax.set_ylim(1.5, 5.5)
    ax.yaxis.labelpad = 1
fig.axes[3].legend(handles=[Line2D([], [], color=COLOR[b], lw=1.2, label=NAME[b]) for b in BIAS], loc='lower right', fontsize=6)
save(fig, f'{OUT}/fig1_decay')

# ---------------- Figure 2: behavior vs direction, relative to generation 1 ----------------
fig, axs = plt.subplots(1, 3, figsize=(TEXTWIDTH, 1.85), sharey=True)
MEAS = [('cos', 'recovery cosine', '-', 'o'), ('ll_gap', r'likelihood gap $\Delta_n$', '--', 's'), ('hit_ex', 'excess hit rate', ':', '^')]
for ax, b in zip(axs, BIAS):
    ax.axhline(0, color='#999999', lw=0.6, zorder=1)
    for k, lab, ls, mk in MEAS:
        rel = np.array([np.array(D[f'{b}_{s}'][k]) / D[f'{b}_{s}'][k][0] for s in SEEDS])
        ax.fill_between(n, rel.min(0), rel.max(0), color=COLOR[b], alpha=0.18, lw=0)
        ax.plot(n, rel.mean(0), color=COLOR[b], ls=ls, marker=mk, ms=3, label=lab)
    ax.set_title(NAME[b], color=INK); ax.set_xticks([1, 4, 7, 10]); ax.set_xlabel('generation'); ax.set_ylim(-0.1, 1.05)
axs[0].set_ylabel('relative to generation 1')
axs[0].legend(handles=[Line2D([], [], color=INK, ls=ls, marker=mk, ms=3, label=lab) for k, lab, ls, mk in MEAS],
              loc='upper right', fontsize=6)
fig.subplots_adjust(wspace=0.12)
save(fig, f'{OUT}/fig2_behavior')

# ---------------- Figure 3: causal residue ----------------
# Late-generation means and CIs (gens 8-10) are the paired-bootstrap values of Table 3 (DiD) and Table 2 (Delta).
LATE = {('owl', 42): (0.24, 0.19, 0.29, 0.11, 0.02, 0.20), ('owl', 43): (0.29, 0.24, 0.33, 0.07, 0.02, 0.13),
        ('cat', 42): (0.99, 0.84, 1.15, 1.02, 0.79, 1.25), ('cat', 43): (1.45, 1.29, 1.62, 1.41, 1.22, 1.61),
        ('ai_supreme', 42): (0.15, 0.13, 0.17, 0.37, 0.34, 0.41), ('ai_supreme', 43): (0.37, 0.34, 0.40, 0.57, 0.52, 0.62)}
fig = plt.figure(figsize=(TEXTWIDTH, 1.95))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.0], wspace=0.5)
for j, b in enumerate(BIAS):
    ax = fig.add_subplot(gs[0, j])
    band = max(D[f'{b}_{s}']['rand_band'] for s in SEEDS)
    ax.axhspan(-band, band, color=BAND, lw=0, zorder=0); ax.axhline(0, color='#999999', lw=0.6, zorder=1)
    for s in SEEDS:
        d = D[f'{b}_{s}']; y = np.array(d['did']); ci = np.array(d['did_ci'])
        ax.fill_between(n, ci[:, 0], ci[:, 1], color=COLOR[b], alpha=0.18, lw=0)
        a, r, c = fit_floor(n.astype(float), y)
        xx = np.linspace(1, 10, 100); ax.plot(xx, a * r ** (xx - 1) + c, color=COLOR[b], lw=0.8, ls=':')
        ax.plot(n, y, color=COLOR[b], marker=MARK[s], ls='-' if s == 42 else '--', ms=3)
        print(f'fig3 fit {b} {s}: a={a:.2f} r={r:.2f} c={c:.2f}')
    ax.set_title(NAME[b], color=INK); ax.set_xticks([1, 4, 7, 10]); ax.set_xlabel('generation')
    ax.set_ylim(min(-band, -0.05) * 1.3, None)
    if j == 0: ax.set_ylabel(r'causal residue $\mathrm{DiD}_n$ (nats)')
fig.legend(handles=seed_handles([Line2D([], [], color=INK, lw=0.8, ls=':', label=r'fit $a r^{n-1}+c$'),
                                 plt.Rectangle((0, 0), 1, 1, color=BAND, label='random direction')]),
           loc='lower center', bbox_to_anchor=(0.42, 0.98), ncol=4, fontsize=6.5)
ax = fig.add_subplot(gs[0, 3])
ax.plot([0, 1.7], [0, 1.7], color='#999999', lw=0.7, ls='--', zorder=1); ax.text(0.06, 1.55, r'dashed: $\mathrm{DiD}=\Delta$', fontsize=6, color='#777777')
for (b, s), (m, lo, hi, dm, dlo, dhi) in LATE.items():
    ax.errorbar(dm, m, xerr=[[dm - dlo], [dhi - dm]], yerr=[[m - lo], [hi - m]], color=COLOR[b], marker=MARK[s], ms=3.5,
                lw=0.8, capsize=0, ls='none', zorder=3)
ax.set_xlim(0, 1.7); ax.set_ylim(0, 1.7)
ax.set_xlabel(r'late $\Delta$ (nats)'); ax.set_ylabel(r'late $\mathrm{DiD}$ (nats)', labelpad=1); ax.set_title('residue vs. gap, gens 8–10', color=INK)
ax.grid(axis='both')
save(fig, f'{OUT}/fig3_residue')

# ---------------- Figure 4: cartridge distillation (cat, owl) ----------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.9), gridspec_kw={'width_ratios': [1.15, 1], 'wspace': 0.32})
gens = ['1', '4', '7', '10']; x0 = np.arange(4)
a1.axhline(0, color='#999999', lw=0.6)
for i, b in enumerate(['cat', 'owl']):
    for k, s in enumerate(SEEDS):
        c = D[f'd2d_{b}_{s}']; off = (i * 2 + k - 1.5) * 0.14
        for g, x in zip(gens, x0):
            y = c[g]['hit_amp_did'] * 100; lo, hi = (np.array(c[g]['hit_amp_did_ci95']) * 100)
            dormant = c[g]['dormant']
            a1.errorbar(x + off, y, yerr=[[y - lo], [hi - y]], color=COLOR[b], lw=0.8, capsize=0, ls='none')
            a1.plot(x + off, y, marker=MARK[s], ms=4, color=COLOR[b], mfc='white' if dormant else COLOR[b], mew=0.9, ls='none')
a1.set_xticks(x0); a1.set_xticklabels([f'gen {g}' for g in gens]); a1.set_ylabel('hit amplification DiD (pp)')
a1.set_title('resurfacing by the cartridge', color=INK)
a1.legend(handles=[Line2D([], [], color=COLOR['cat'], marker='o', ls='none', label='cat'),
                   Line2D([], [], color=COLOR['owl'], marker='o', ls='none', label='owl'),
                   Line2D([], [], color=INK, marker='o', ls='none', label='seed 42'),
                   Line2D([], [], color=INK, marker='^', ls='none', label='seed 43'),
                   Line2D([], [], color=INK, marker='o', mfc='white', ls='none', label='dormant gen.')],
          loc='lower right', fontsize=6, ncol=2, columnspacing=0.8, handletextpad=0.2)
ARMS = [('full', r'$v_{\mathrm{cart}}$', '-'), ('par', r'along $v_c$', '--'), ('orth', r'orthogonal to $v_c$', ':')]
for s in SEEDS:
    arms = D[f'vcart_cat_{s}']['10']['arms']
    for arm, lab, ls in ARMS:
        al = sorted(arms[arm], key=float)
        a2.plot([float(a) for a in al], [arms[arm][a]["mention_rate"] * 100 for a in al], color=COLOR['cat'], ls=ls, marker=MARK[s], ms=2.8, lw=1.0)
a2.set_xlabel(r'steering strength $\alpha$'); a2.set_ylabel('probe answers naming cat (%)')
a2.set_title(r'steering with $v_{\mathrm{cart}}$ (cat, gen 10)', color=INK); a2.set_ylim(-1, 45)
a2.legend(handles=[Line2D([], [], color=COLOR['cat'], ls=ls, label=lab) for _, lab, ls in ARMS], loc='upper left', fontsize=6)
save(fig, f'{OUT}/fig4_d2d')
print('done')
