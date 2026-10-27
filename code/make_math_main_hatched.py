"""Figure 2: per-case relative RMSE of AB-PAS, the fixed-cloud floor and AS-AK.

Monochrome version with hatching instead of colour: AB-PAS diagonal hatch,
floor horizontal hatch, AS-AK open (no fill, no hatch).
"""
import json
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = pathlib.Path(__file__).resolve().parent
(ROOT / 'figures').mkdir(exist_ok=True)
OUT = ROOT / 'figures/math_main.png'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'hatch.linewidth': 0.6,
})

ABPAS = np.array([0.0321, 0.0181, 0.0664, 0.0158, 0.0627, 0.0072, 0.0175])
FLOOR = np.array([0.0321, 0.0181, 0.0664, 0.0158, 0.0627, 0.0029, 0.0175])
ASAK = np.array([0.0275, 0.0181, 0.0664, 0.0288, 0.0627, 0.8729, 0.0170])

x = np.arange(len(ABPAS))
w = 0.26

fig, ax = plt.subplots(figsize=(7.2, 3.0))
b1 = ax.bar(x - w, ABPAS, w, facecolor='white', edgecolor='black', linewidth=0.7,
            hatch='////', label='AB-PAS', zorder=3)
b2 = ax.bar(x, FLOOR, w, facecolor='white', edgecolor='black', linewidth=0.7,
            hatch='----', label='Floor', zorder=3)
b3 = ax.bar(x + w, ASAK, w, facecolor='white', edgecolor='black', linewidth=0.7,
            label='AS-AK', zorder=3)

ax.set_yscale('log')
ax.set_ylim(2e-3, 2.0)
ax.set_xticks(x)
ax.set_xticklabels([f'C{i}' for i in range(1, 8)])
ax.set_ylabel('Relative RMSE')
ax.grid(axis='y', which='major', color='0.85', lw=0.6, zorder=0)
ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, ncol=3, loc='lower left', bbox_to_anchor=(0.0, 1.0),
          fontsize=8.5, handlelength=1.8, columnspacing=1.6, borderaxespad=0.2)

fig.tight_layout(pad=0.3)
fig.savefig(OUT, dpi=300)
print('written', OUT)
