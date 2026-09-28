#!/usr/bin/env python3
"""Clean-data figures for the P2 Technologies revision.
   R1 clean_staircase: RW + FG at headroom view + per-stage dissociation.
   R2 instrument_ladder: AUROC across views for all 13 cells.
   R3 scale_null: Pythia at headroom with CIs.
   R4 camouflage_corrected: per-generator clustered rho with CIs.
   Sources: AEGIS-Staircases-Clean evidence JSONs + camouflage_clustered_revision.json.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import apply_style, COLORS, MODEL_LABELS
import matplotlib.pyplot as plt
import numpy as np

apply_style()
OUT = Path('/Users/abadila/Desktop/ethics/writing/P2-Technologies-revision/figures')
EV = Path('/Users/abadila/Desktop/ethics/AEGIS-Staircases-Clean/docs/evidence')

STAGES = ['base', 'sft', 'dpo', 'rlvr']
SLAB = ['Base', 'SFT', 'DPO', 'RLVR']
PYTHIA = ['70m', '410m', '1.4b', '2.8b', '6.9b']
PLAB = ['70M', '410M', '1.4B', '2.8B', '6.9B']

def load(view, suite):
    if suite == 'fg':
        p = EV / ('evaluation-2026-09-28/evaluation.json' if view == '128'
                  else f'headroom-2026-09-28/view-{view}/evaluation.json')
    else:
        p = EV / f'rewrite-2026-09-28/evaluation/view-{view}/evaluation.json'
    return json.load(open(p))

def ci_of(d, key):
    b = d['paired_source_bootstrap']['cells']
    for k in (key, key.replace('/', '_')):
        if k in b: return b[k]['auroc'], b[k]['auroc_ci95']
    raise KeyError(key)

# ---------------- R1: clean staircase + dissociation ----------------
fg32, rw32 = load('32', 'fg'), load('32', 'rw')
fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.6))
for ax, (src, cells_key, prefix, color, title, page_p) in zip(
    axes[:2],
    [(rw32, 'rewrite', '', COLORS['green'],
      'Rewrite (source-anchored)', rw32['rewrite']['ceiling_audit']['ordinary_observed']['p_value']),
     (fg32, None, 'tulu/', COLORS['blue'],
      'Free generation', fg32['tulu']['ordinary_exact_page']['p_value'])]):
    vals, lo, hi = [], [], []
    for s in STAGES:
        if cells_key:  # rewrite file
            b = src['paired_source_bootstrap']['cells']
            kk = f'rw/{s}'
            a, ci = b[kk]['auroc'], b[kk]['auroc_ci95']
        else:
            a, ci = ci_of(src, f'tulu/{s}')
        vals.append(a); lo.append(a - ci[0]); hi.append(ci[1] - a)
    x = np.arange(4)
    ax.errorbar(x, vals, yerr=[lo, hi], fmt='o-', color=color, linewidth=1.8,
                markersize=5, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(SLAB, fontsize=8)
    ax.set_ylim(0.935, 1.005)
    ax.set_title(f'{title}\nexact Page $p = {page_p:.1e}$'.replace('e-0', r'\times 10^{-').replace('e-', r'\times 10^{-') + '}$' if False else f'{title}\nexact Page p = {page_p:.2g}', fontsize=8.5, fontweight='bold')
    ax.set_ylabel('AUROC (32-token view)', fontsize=8)

# dissociation panel
dis = rw32['dissociation']['per_stage']
deltas = [dis[s]['delta'] for s in STAGES]
dlo = [dis[s]['delta'] - dis[s]['delta_ci95'][0] for s in STAGES]
dhi = [dis[s]['delta_ci95'][1] - dis[s]['delta'] for s in STAGES]
ax = axes[2]
ax.axhline(0, color='#555555', lw=0.8)
ax.errorbar(np.arange(4), deltas, yerr=[dlo, dhi], fmt='s', color=COLORS['orange'],
            markersize=5, capsize=3)
did = rw32['dissociation']['difference_in_differences_rlvr_minus_base']
ax.set_xticks(np.arange(4)); ax.set_xticklabels(SLAB, fontsize=8)
ax.set_ylabel(r'$\Delta$AUROC (rewrite $-$ free)', fontsize=8)
ax.set_title(f"Task dissociation: none\nDiD = {did['estimate']:.3f} "
             f"[{did['ci95'][0]:.3f}, {did['ci95'][1]:.3f}]", fontsize=8.5, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'fig_clean_staircase.pdf'); fig.savefig(OUT / 'fig_clean_staircase.png', dpi=300)
plt.close(fig)
print('R1 saved')

# ---------------- R2: instrument ladder ----------------
rows = [('Tulu Base (rw)', 'rw', 'base'), ('Tulu SFT (rw)', 'rw', 'sft'),
        ('Tulu DPO (rw)', 'rw', 'dpo'), ('Tulu RLVR (rw)', 'rw', 'rlvr'),
        ('Tulu Base (fg)', 'fg', 'tulu/base'), ('Tulu SFT (fg)', 'fg', 'tulu/sft'),
        ('Tulu DPO (fg)', 'fg', 'tulu/dpo'), ('Tulu RLVR (fg)', 'fg', 'tulu/rlvr')] + \
       [(f'Pythia {l} (fg)', 'fg', f'pythia/{m}') for m, l in zip(PYTHIA, PLAB)]
views = ['128', '64', '32']
M = np.zeros((len(rows), 3))
for j, v in enumerate(views):
    fgv, rwv = load(v, 'fg'), load(v, 'rw')
    for i, (_, suite, key) in enumerate(rows):
        if suite == 'fg':
            M[i, j] = fgv['cells'][key.replace('tulu/','tulu/') if '/' in key else key]['test_auroc'] \
                      if key in fgv['cells'] else fgv['cells'][key]['test_auroc']
        else:
            M[i, j] = rwv['rewrite']['cells'][key]['test_auroc']
fig, ax = plt.subplots(figsize=(4.6, 4.2))
im = ax.imshow(M, cmap='YlGnBu', vmin=0.94, vmax=1.0, aspect='auto')
for i in range(M.shape[0]):
    for j in range(3):
        v = M[i, j]
        txt = '1.0*' if v == 1.0 else f'{v:.4f}'.lstrip('0')
        ax.text(j, i, txt, ha='center', va='center', fontsize=6.5,
                color='white' if v > 0.985 else '#1c1c1c',
                fontweight='bold' if v == 1.0 else 'normal')
ax.set_xticks(range(3)); ax.set_xticklabels([f'{v} tokens' for v in views], fontsize=8)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
ax.set_title('Test AUROC by analysis-view length\n(* = instrument ceiling)', fontsize=9, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'fig_instrument_ladder.pdf'); fig.savefig(OUT / 'fig_instrument_ladder.png', dpi=300)
plt.close(fig)
print('R2 saved')

# ---------------- R3: scale null ----------------
fig, ax = plt.subplots(figsize=(3.4, 2.7))
vals, lo, hi = [], [], []
for m in PYTHIA:
    a, ci = ci_of(fg32, f'pythia/{m}')
    vals.append(a); lo.append(a - ci[0]); hi.append(ci[1] - a)
x = np.arange(5)
ax.errorbar(x, vals, yerr=[lo, hi], fmt='o-', color=COLORS['blue'], linewidth=1.8,
            markersize=5, capsize=3)
sp = fg32['pythia']['ordinary_exact_spearman']
ax.set_xticks(x); ax.set_xticklabels(PLAB, fontsize=8)
ax.set_ylabel('AUROC (32-token view)', fontsize=8)
ax.set_xlabel('parameters', fontsize=8)
ax.set_title(f"No scale effect\nexact Spearman $\\rho$ = {sp['rho']:.2f}, p = {sp['p_value']:.3g}",
             fontsize=8.5, fontweight='bold')
ax.set_ylim(0.935, 1.005)
fig.tight_layout()
fig.savefig(OUT / 'fig_scale_null.pdf'); fig.savefig(OUT / 'fig_scale_null.png', dpi=300)
plt.close(fig)
print('R3 saved, rho:', sp['rho'], 'p:', sp['p_value'])

# ---------------- R4: corrected camouflage ----------------
cam = json.load(open('/Users/abadila/Desktop/ethics/AEGIS-Multi/analysis/results/camouflage_clustered_revision.json'))
ap = cam['aligned_pairing']
per = ap['per_generator']
gens = sorted(per.keys(), key=lambda g: per[g]['rho'])
rhos = [per[g]['rho'] for g in gens]
los = [per[g]['rho'] - per[g]['ci95'][0] for g in gens]
his = [per[g]['ci95'][1] - per[g]['rho'] for g in gens]
pooled = ap['source_cluster_bootstrap']['pooled_spearman']
within = ap['source_cluster_bootstrap']['within_source_and_generator']
fig, ax = plt.subplots(figsize=(4.8, 3.0))
y = np.arange(len(gens))
ax.axvline(0, color='#555555', lw=0.8)
ax.errorbar(rhos, y, xerr=[los, his], fmt='o', color=COLORS['green'], markersize=5, capsize=3)
ax.axvline(pooled['estimate'], color=COLORS['blue'], lw=1.4, ls='--',
           label=f"pooled $\\rho$ = {pooled['estimate']:.2f}")
ax.axvline(within['estimate'], color=COLORS['orange'], lw=1.4, ls=':',
           label=f"within source+generator $\\rho$ = {within['estimate']:.2f}")
ax.set_yticks(y); ax.set_yticklabels([MODEL_LABELS[g] for g in gens], fontsize=7.5)
ax.set_xlabel(r'Spearman $\rho$ (source BLEU-4 vs.\ probe confidence)', fontsize=8)
ax.legend(loc='lower right', frameon=False, fontsize=7)
ax.set_title('Camouflage: more source copying, lower detector confidence\n'
             '(source-clustered bootstrap, 95% CIs)', fontsize=8.5, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'fig_camouflage_corrected.pdf'); fig.savefig(OUT / 'fig_camouflage_corrected.png', dpi=300)
print('R4 saved; pooled', pooled['estimate'], pooled['ci95'], 'within', within['estimate'], within['ci95'])
