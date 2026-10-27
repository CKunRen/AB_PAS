"""Figure: reliability estimates against the independent reference.

(a) Mean AB-PAS estimate versus the reference on log-log axes with the y = x
    diagonal.
(b) The same estimate as a ratio to the reference, with the fixed-cloud floor
    of each case drawn as a bar around unity.
"""
import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
(ROOT / 'figures').mkdir(exist_ok=True)
OUT = ROOT / 'figures/estimate.png'
SRC = ROOT / 'results/method_paper_v3/v6_final_reduced.json'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8.5,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

data = json.loads(SRC.read_text(encoding='utf-8'))
# Fixed-cloud floors of the seven cases: the same values as Table 4 and
# export_v6_data.py (relative discrepancy of a perfect classifier on the
# common 2^18 cloud).
FLOORS = [0.0321, 0.0181, 0.0664, 0.0158, 0.0627, 0.0029, 0.0175]

ref, est = [], []
for n in range(1, 8):
    c = data['cases'][str(n)]
    ref.append(c['reference'])
    est.append(float(np.mean([r['selected'] for r in c['runs']])))
ref, est, floor = np.asarray(ref), np.asarray(est), np.asarray(FLOORS)
ratio = est / ref

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.75),
                               gridspec_kw={'width_ratios': [1.0, 1.45]})

# ---- (a) log-log identity -------------------------------------------------
lims = (7e-4, 4.2e-2)
ax1.plot(lims, lims, ls=(0, (4, 3)), lw=0.9, color='0.35', zorder=1)
ax1.scatter(ref, est, s=26, facecolor='#1f4e79', edgecolor='white',
            linewidth=0.6, zorder=3)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(*lims)
ax1.set_ylim(*lims)
ax1.set_aspect('equal')
ax1.set_xlabel(r'$P_{f,\mathrm{ref}}$', labelpad=2)
ax1.set_ylabel(r'$\hat P_f$ (mean of ten)', labelpad=2)
ax1.set_title('(a)', loc='left', fontsize=9, pad=3)

# ---- (b) ratio with the floor --------------------------------------------
idx = np.arange(1, 8)
outside = np.abs(ratio - 1) > 1.05 * floor + 1e-9
for k, f in zip(idx, floor):
    ax2.barh(k, 2 * f, left=1 - f, height=0.42, color='0.88', edgecolor='none',
             zorder=1)
ax2.axvline(1.0, color='0.35', lw=0.9, ls=(0, (4, 3)), zorder=2)
ax2.scatter(ratio[~outside], idx[~outside], s=26, facecolor='#1f4e79',
            edgecolor='white', linewidth=0.6, zorder=3)
ax2.scatter(ratio[outside], idx[outside], s=30, facecolor='#b2182b',
            edgecolor='white', linewidth=0.6, zorder=4)
ax2.set_xlim(0.90, 1.10)
ax2.set_ylim(0.4, 7.6)
ax2.set_yticks(idx)
ax2.set_yticklabels([f'C{k}' for k in idx])
ax2.set_xlabel(r'$\hat P_f / P_{f,\mathrm{ref}}$', labelpad=2)
ax2.set_title('(b)', loc='left', fontsize=9, pad=3)
ax2.text(0.02, 0.02, 'grey: $\\pm$ fixed-cloud floor; red: outside it',
         transform=ax2.transAxes, fontsize=6.8, color='0.35')

fig.tight_layout(pad=0.25, w_pad=1.2)
fig.savefig(OUT, dpi=300)
print('written', OUT)

for n in range(7):
    print('C%d ref=%.6g est=%.6g rel=%+.4f floor=%.4f beyond=%s'
          % (n + 1, ref[n], est[n], ratio[n] - 1, floor[n],
             abs(ratio[n] - 1) > floor[n] + 1e-9))
