"""Figure: predicted failure domains for benchmark C5 in its latent plane.

Six panels, all in the true two-dimensional latent coordinates of C5 and all
evaluated on the slice u = z B^T (orthogonal component zero):

  exact domain, AB-PAS, AL8, passive DAS, PLS-Q, full-space network.

Each panel also reports the relative error of that surrogate's failure
probability on the paper's common 2^18 cloud (seed 72005), so the visual
comparison and the probability error refer to the same fit.
"""
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.special import ndtri
from scipy.stats import qmc

from event_adaptive import run as run_ea
from event_adaptive_v3 import run as run_v3
from method_benchmarks import Problem
from poly_experts import fit_candidates, predict, select_candidate
from run_pls_control import basis_pls, features, fit as pls_fit

ROOT = pathlib.Path(__file__).resolve().parent
(ROOT / 'figures').mkdir(exist_ok=True)
OUT = ROOT / 'figures/c5_boundary.png'
torch.set_num_threads(2)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 7.8,
    'axes.linewidth': 0.7,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
})

N = 5
SEED = 410000 + N * 1000
GRID_CAND = {2: [2, 5, 8, 20], 3: [2]}
p = Problem(N)
B = p.basis
REF, _ = p.reference()

cloud = ndtri(np.clip(qmc.Sobol(p.d, scramble=True, seed=72000 + N)
                      .random_base2(18), 1e-12, 1 - 1e-12))

# ---------------------------------------------------------------- AB-PAS
row = run_v3(p, 'conflict', SEED, epochs=300, gate='conflict_validation',
             gain=0.02, return_state=True, use_probes=False)
st = row['state']
err_mlp = np.mean(((st['gh_v'] - st['gv']) / st['sd']) ** 2, axis=1)
cands = fit_candidates(st['x'], st['g'], st['xv'], st['gv'], st['mu'], st['sd'],
                       st['basis'], p, float(err_mlp.mean()), SEED + 7000,
                       grid=GRID_CAND, epochs=1500, random_ranks={2, 20})
for cand in cands:
    if cand['name'] == 'mlp':
        cand['se'] = float(err_mlp.std() / np.sqrt(len(err_mlp)))
choice = select_candidate(cands)
print('AB-PAS choice:', choice['name'])
if choice['name'] == 'mlp':
    pred_ab = lambda u: predict(st['net'], u, st['mu'], st['sd'])
else:
    pred_ab = lambda u: predict(choice['net'], u, st['mu'], st['sd'])


def from_event_adaptive(method):
    r = run_ea(p, method, SEED, return_state=True)
    s = r['state']
    return lambda u: predict(s['net'], u, s['mu'], s['sd'])


pred_al8 = from_event_adaptive('active8')
pred_pas = from_event_adaptive('static')
pred_full = from_event_adaptive('full')

# ---------------------------------------------------------------- PLS-Q
rng = np.random.default_rng(SEED)
u_lab = rng.normal(size=(1200, p.d))
g_lab = p.evaluate(u_lab)
xv, gv = u_lab[:120], g_lab[:120]
x, g = u_lab[120:], g_lab[120:]
mu_p, sd_p = g.mean(0), g.std(0) + 1e-8
fits = []
for m in (2, 4, 8):
    pred_m = pls_fit(x, (g - mu_p) / sd_p, m)
    k = np.exp(-0.5 * (np.min(np.abs(gv / sd_p), axis=1) / 0.6) ** 2)
    k /= max(k.mean(), 1e-12)
    score = np.mean(k * np.mean((pred_m(xv) - (gv - mu_p) / sd_p) ** 2, axis=1))
    fits.append((score, m, pred_m))
best = min(v[0] for v in fits)
_, m_pls, pred_pls_raw = min((v for v in fits if v[0] <= 1.05 * best), key=lambda v: v[1])
pred_pls = lambda u: pred_pls_raw(u) * sd_p + mu_p
print('PLS rank', m_pls)

# ------------------------------------------------------------- the grid
z = np.linspace(-5.4, 5.4, 361)
Z1, Z2 = np.meshgrid(z, z)
zpts = np.column_stack([Z1.ravel(), Z2.ravel()])
u_grid = zpts @ B.T

f_true = (p.evaluate(u_grid) <= 0).any(1).reshape(Z1.shape)

METHODS = [
    ('(a) exact domain', None),
    ('(b) AB-PAS', pred_ab),
    ('(c) AL8', pred_al8),
    ('(d) passive DAS', pred_pas),
    ('(e) PLS-Q', pred_pls),
    ('(f) full-space network', pred_full),
]

fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9), sharex=True, sharey=True)
for ax, (title, pred) in zip(axes.ravel(), METHODS):
    if pred is None:
        mask = f_true
        note = ''
    else:
        mask = (pred(u_grid) <= 0).any(1).reshape(Z1.shape)
        err = float(np.mean(mask != f_true))
        note = '  <0.1%' if err < 5e-4 else '  %.1f%%' % (100 * err)
    miss = f_true & ~mask
    false = mask & ~f_true
    ok = mask & f_true
    ax.contourf(Z1, Z2, ok.astype(float), levels=[0.5, 1.5], colors=['0.80'],
                zorder=1)
    if false.any():
        ax.contourf(Z1, Z2, false.astype(float), levels=[0.5, 1.5],
                    colors=['#9ecae1'], zorder=2)
    if miss.any():
        ax.contourf(Z1, Z2, miss.astype(float), levels=[0.5, 1.5],
                    colors=['#fc9272'], zorder=3)
    ax.contour(Z1, Z2, f_true.astype(float), levels=[0.5], colors='#b2182b',
               linewidths=0.9, linestyles='dashed', zorder=4)
    ax.set_title(title + note, fontsize=7.8, pad=2.5)
    if ax in axes[0]:
        ax.tick_params(labelbottom=False)
    ax.set_xlim(-5.4, 5.4)
    ax.set_ylim(-5.4, 5.4)
    ax.set_aspect('equal')
    ax.set_xticks([-4, 0, 4])
    ax.set_yticks([-4, 0, 4])
    if ax in axes[-1]:
        ax.set_xlabel(r'$z_1$', labelpad=1)
    if ax in axes[:, 0]:
        ax.set_ylabel(r'$z_2$', labelpad=1)

handles = [Patch(facecolor='0.80', edgecolor='none'),
           Patch(facecolor='#fc9272', edgecolor='none'),
           Patch(facecolor='#9ecae1', edgecolor='none'),
           Line2D([0], [0], color='#b2182b', lw=0.9, ls='dashed')]
fig.legend(handles, ['correct', 'missed failure', 'false failure', 'exact boundary'],
           loc='lower center', ncol=4, frameon=False, fontsize=7.2,
           bbox_to_anchor=(0.5, -0.005), handlelength=1.4, columnspacing=1.4)
fig.tight_layout(pad=0.25, h_pad=1.3, w_pad=0.4, rect=(0, 0.045, 1, 1))
fig.savefig(OUT, dpi=300)
print('written', OUT)

for title, pred in METHODS:
    if pred is None:
        continue
    mask = (pred(u_grid) <= 0).any(1).reshape(Z1.shape)
    print('%-22s area %.4f  plane error %.4f' % (title, mask.mean(),
                                                 float(np.mean(mask != f_true))))
