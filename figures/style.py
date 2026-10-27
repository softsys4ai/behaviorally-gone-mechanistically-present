"""Shared figure style: STIX (Times-like) text, despined axes, one fixed color per bias."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Fixed categorical order (dataviz reference palette, slots 1-3, then 6 and 5 for the preliminary-run animals).
COLOR = {'owl': '#2a78d6', 'cat': '#eb6834', 'ai_supreme': '#1baf7a', 'dragon': '#008300', 'wolf': '#e87ba4'}
GREY = '#8a8985'      # control chains / nulls
BAND = '#d9d8d4'      # random / null band
INK = '#333333'
NAME = {'owl': 'owl', 'cat': 'cat', 'ai_supreme': 'AI is superior', 'dragon': 'dragon', 'wolf': 'wolf'}
MARK = {42: 'o', 43: '^', 44: 's'}
TEXTWIDTH = 5.5       # inches, NeurIPS text block

plt.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8,
    'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'axes.edgecolor': '#555555', 'axes.labelcolor': INK, 'xtick.color': '#555555', 'ytick.color': '#555555',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': '#ebebe8', 'grid.linewidth': 0.6,
    'lines.linewidth': 1.3, 'lines.markersize': 3.5, 'legend.frameon': False, 'legend.handlelength': 1.8,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'pdf.fonttype': 42,
})

def save(fig, path):
    fig.savefig(path + '.pdf'); fig.savefig(path + '.png', dpi=200); plt.close(fig)

def fit_geom(n, y):
    """log-linear fit y = A r^n; returns (A, r)."""
    b, a = np.polyfit(n, np.log(y), 1); return np.exp(a), np.exp(b)

def fit_floor(n, y):
    """y = a r^(n-1) + c by grid over r (no scipy); least squares for a, c."""
    best = None
    for r in np.linspace(0.05, 0.99, 941):
        X = np.c_[r ** (n - 1), np.ones_like(n, dtype=float)]
        coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = ((X @ coef - y) ** 2).sum()
        if best is None or e < best[0]: best = (e, coef[0], r, coef[1])
    return best[1], best[2], best[3]
