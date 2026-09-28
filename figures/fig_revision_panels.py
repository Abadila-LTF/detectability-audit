#!/usr/bin/env python3
"""Regenerate the four figures flagged in the final validation pass."""
import sys, json
from pathlib import Path
sys.path.insert(0, '/Users/abadila/Desktop/ethics/writing/shared/figscripts')
from figstyle import apply_style, COLORS, MODEL_LABELS, load_json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

apply_style()
OUT = Path('/Users/abadila/Desktop/ethics/writing/P2-Technologies-revision/figures')
EV = Path('/Users/abadila/Desktop/ethics/AEGIS-Staircases-Clean/docs/evidence')

def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', dpi=300)
    plt.close(fig); print(f'{name} saved')

# ---------- 1. instrument ladder (new palette + colorbar) ----------
def load(view, suite):
    if suite == 'fg':
        p = EV / ('evaluation-2026-09-28/evaluation.json' if view == '128'
                  else f'headroom-2026-09-28/view-{view}/evaluation.json')
    else:
        p = EV / f'rewrite-2026-09-28/evaluation/view-{view}/evaluation.json'
    return json.load(open(p))

rows = [('Tulu Base (rw)', 'rw', 'base'), ('Tulu SFT (rw)', 'rw', 'sft'),
        ('Tulu DPO (rw)', 'rw', 'dpo'), ('Tulu RLVR (rw)', 'rw', 'rlvr'),
        ('Tulu Base (fg)', 'fg', 'tulu/base'), ('Tulu SFT (fg)', 'fg', 'tulu/sft'),
        ('Tulu DPO (fg)', 'fg', 'tulu/dpo'), ('Tulu RLVR (fg)', 'fg', 'tulu/rlvr'),
        ('Pythia 70M (fg)', 'fg', 'pythia/70m'), ('Pythia 410M (fg)', 'fg', 'pythia/410m'),
        ('Pythia 1.4B (fg)', 'fg', 'pythia/1.4b'), ('Pythia 2.8B (fg)', 'fg', 'pythia/2.8b'),
        ('Pythia 6.9B (fg)', 'fg', 'pythia/6.9b')]
views = ['128', '64', '32']
M = np.zeros((13, 3))
for j, v in enumerate(views):
    fgv, rwv = load(v, 'fg'), load(v, 'rw')
    for i, (_, suite, key) in enumerate(rows):
        M[i, j] = (fgv['cells'][key] if suite == 'fg' else rwv['rewrite']['cells'][key])['test_auroc']
print('ceiling cells at 128:', int((M[:, 0] == 1.0).sum()), 'of 13')

fig, ax = plt.subplots(figsize=(5.3, 4.3))
im = ax.imshow(M, cmap='Blues', vmin=0.96, vmax=1.0, aspect='auto')
ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
ax.set_yticks(np.arange(-.5, 13, 1), minor=True)
ax.grid(which='minor', color='white', linewidth=1.4)
ax.tick_params(which='both', length=0)
for yy in (3.5, 7.5):
    ax.axhline(yy, color='white', lw=3.0)
for i in range(13):
    for j in range(3):
        v = M[i, j]
        txt = '1.0*' if v == 1.0 else f'{v:.4f}'.lstrip('0')
        frac = (v - 0.96) / 0.04
        ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                color='white' if frac > 0.62 else '#1c1c1c',
                fontweight='bold' if v == 1.0 else 'normal')
ax.set_xticks(range(3)); ax.set_xticklabels([f'{v} tokens' for v in views], fontsize=8.5)
ax.set_yticks(range(13)); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
ax.set_title('Test AUROC by analysis-view length\n($*$ = instrument ceiling, AUROC $= 1.0$)',
             fontsize=9, fontweight='bold')
cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04, ticks=[0.96, 0.97, 0.98, 0.99, 1.0])
cbar.set_label('Test AUROC', fontsize=8)
cbar.ax.tick_params(labelsize=7)
fig.tight_layout()
save(fig, 'fig_instrument_ladder')

# ---------- 2. token anatomy (no jargon, short title) ----------
n4 = json.load(open('/Users/abadila/Desktop/ethics/AEGIS-Multi/analysis/results/n4_positional_fingerprint.json'))
wins = ['0-64', '64-128', '128-192', '192-256']
sample = n4['per_model']['gpt-4o']['0-64']
print('anatomy cell keys:', sorted(sample.keys()))
dkey = 'dprime' if 'dprime' in sample else [k for k in sample if 'dprime' in k][0]
fig, ax = plt.subplots(figsize=(3.7, 2.7))
allv = []
for m, d in n4['per_model'].items():
    vals = [d[w][dkey] for w in wins]
    allv.append(vals)
    ax.plot(range(4), vals, color='#bbbbbb', lw=0.9, zorder=1)
mean = np.mean(allv, axis=0)
ax.plot(range(4), mean, 'o-', color=COLORS['orange'], lw=2.2, markersize=5.5,
        zorder=3, label='mean (10 generators)')
print('anatomy means:', [f'{v:.3f}' for v in mean])
ax.set_xticks(range(4)); ax.set_xticklabels(wins, fontsize=8)
ax.set_xlabel('token window', fontsize=8.5)
ax.set_ylabel("probe $d'$", fontsize=8.5)
ax.set_title("Probe $d'$ by token window", fontsize=9, fontweight='bold')
ax.legend(loc='lower right', frameon=False, fontsize=7.5)
fig.tight_layout()
save(fig, 'fig_token_anatomy')

# ---------- 3. geometry panel (corrected 9-unique QAP stat) ----------
g1 = load_json('geometry_gates.json')['gate_G1']
qap = json.load(open('/Users/abadila/Desktop/ethics/AEGIS-Multi/analysis/results/g1_qap_9unique_revision.json'))
tm = load_json('transfer_matrix_3seed_avg.json')
models = g1['transfer_matrix']['models']
# canonical revision statistic: symmetrized 3-seed MeanPool transfer, duplicate endpoint excluded (A6.1)
T = np.array(tm['meanpool_mean'])
W = np.array(g1['whitened_cosine_matrix']['values'])
drop = models.index('llama-3.3-70b-instruct')
keep = [i for i in range(len(models)) if i != drop]
T9 = ((T + T.T) / 2)[np.ix_(keep, keep)]
W9 = W[np.ix_(keep, keep)]
iu = np.triu_indices(9, k=1)
w_p, t_p = W9[iu], T9[iu]
rho, _ = stats.spearmanr(w_p, t_p)
assert abs(rho - qap['rho_observed']) < 1e-3, (rho, qap['rho_observed'])
print(f'9-unique symmetrized rho = {rho:.4f} (QAP p = {qap["p_exact_enumeration"]:.2e})')

pgeo5 = load_json('pgeo5_evasion.json')
from figstyle import MODEL_ORDER
orig, proj, pct, used = [], [], [], []
for m in MODEL_ORDER:
    pm = pgeo5['per_model'][m]
    orig.append(pm['dprime_original']); proj.append(pm['dprime_projected'])
    pct.append(pm['pct_drop']); used.append(m)
mean_drop = np.mean(pct)

fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8))
ax = axes[0]
ax.scatter(w_p, t_p, s=14, color=COLORS['blue'], alpha=0.75,
           edgecolors='black', linewidths=0.2)
slope, intercept = np.polyfit(w_p, t_p, 1)
xs = np.linspace(w_p.min(), w_p.max(), 50)
ax.plot(xs, slope * xs + intercept, color=COLORS['orange'], linewidth=1.2)
ax.set_xlabel('whitened cosine (model $i$, $j$)', fontsize=7)
ax.set_ylabel('symmetrized transfer AUROC', fontsize=7)
ax.set_title(f'(a) G1: shared detection axis ($\\rho$ = {rho:.3f})',
             fontsize=8, fontweight='bold', pad=3)
ax.text(0.97, 0.05, '36 pairs (9 unique generators)\nexact QAP $p = 7.0\\times10^{-3}$',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=6, color='#444444')
ax = axes[1]
order = np.argsort(orig)
for rank, i in enumerate(order):
    ax.plot([rank, rank], [orig[i], proj[i]], color='#ccc', linewidth=3,
            solid_capstyle='butt', zorder=1)
    ax.scatter(rank, orig[i], s=20, color=COLORS['green'], zorder=3,
               edgecolors='black', linewidths=0.3)
    ax.scatter(rank, proj[i], s=20, color=COLORS['orange'], zorder=3,
               marker='v', edgecolors='black', linewidths=0.3)
ax.set_xticks(range(len(used)))
ax.set_xticklabels([MODEL_LABELS[used[i]] for i in order], rotation=55, ha='right', fontsize=5)
ax.scatter([], [], s=20, color=COLORS['green'], edgecolors='black', linewidths=0.3, label="original $d'$")
ax.scatter([], [], s=20, color=COLORS['orange'], marker='v', edgecolors='black', linewidths=0.3, label='top-1 removed')
ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, edgecolor='none', fontsize=5.5)
ax.set_ylabel("probe $d'$", fontsize=7)
ax.set_title(f'(b) Evasion bound (mean drop {mean_drop:.0f}\\%)', fontsize=8, fontweight='bold', pad=3)
fig.tight_layout(w_pad=2.0)
save(fig, 'fig_geometry_panel')

# ---------- 4. OLMo alone (no contaminated Tulu overlay) ----------
olmo = load_json('e11_olmo_staircase.json')
stages = ['base', 'sft', 'dpo', 'instruct']
dp = [olmo['stages'][s]['dprime_mean'] for s in stages]
au = [olmo['stages'][s]['auroc_mean'] for s in stages]
ceil = [a >= 0.9999 for a in au]
print('OLMo dprime:', dp, 'ceiling:', ceil)
fig, ax = plt.subplots(figsize=(3.3, 2.8))
x = np.arange(4)
ax.plot(x[:2], dp[:2], '-', color=COLORS['blue'], linewidth=2.0, zorder=2)
ax.plot(x[1:], dp[1:], '--', color=COLORS['blue'], linewidth=1.6, zorder=2)
for xi, (v, c) in enumerate(zip(dp, ceil)):
    ax.scatter(xi, v, s=55, color=COLORS['blue'], marker='^' if c else 'o',
               zorder=4, edgecolors='black', linewidths=0.6)
    ax.annotate(f'{v:.2f}' + ('*' if c else ''), (xi, v), textcoords='offset points',
                xytext=(0, 9 if xi else -14), ha='center', fontsize=7,
                color=COLORS['blue'], fontweight='bold')
ax.annotate('+0.64\n(prompt-matched)', (0.5, (dp[0]+dp[1])/2), textcoords='offset points',
            xytext=(26, -6), ha='left', fontsize=6, color='#444444')
ax.set_xticks(x); ax.set_xticklabels(['Base', 'SFT', 'DPO', 'Instruct'], fontsize=8)
ax.set_ylabel("probe $d'$ (free generation)", fontsize=8)
ax.set_ylim(2.0, 7.6)
ax.set_title('OLMo-2-7B post-training ladder', fontsize=9, fontweight='bold')
ax.text(0.02, 0.02, "*ceiling (AUROC = 1.000); $d' \\geq 6.72$ is a lower bound\ndashed: prompt format changes across these stages",
        transform=ax.transAxes, fontsize=5.2, color=COLORS['gray'], fontstyle='italic')
fig.tight_layout()
save(fig, 'fig_olmo_replication')
