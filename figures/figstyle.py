"""Shared figure style for all papers.

Usage:
    from figstyle import apply_style, MODEL_COLORS, MODEL_ORDER, MODEL_LABELS, COLORS
    apply_style()
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Colorblind-safe palette (Wong / Tol)
COLORS = {
    'blue':   '#0173B2',
    'orange': '#DE8F05',
    'green':  '#029E73',
    'red':    '#D55E00',
    'purple': '#CC78BC',
    'cyan':   '#56B4E9',
    'gray':   '#949494',
    'yellow': '#ECE133',
}

# Canonical model → color mapping (consistent across all figures)
MODEL_COLORS = {
    'claude-sonnet':          COLORS['red'],
    'deepseek-r1':            COLORS['gray'],
    'gemini-flash':           COLORS['cyan'],
    'gpt-4o':                 COLORS['blue'],
    'gpt-4o-mini':            COLORS['purple'],
    'llama-3.1-8b':           COLORS['green'],
    'llama-3.3-70b':          COLORS['orange'],
    'llama-3.3-70b-instruct': '#B8860B',
    'mistral-large':          COLORS['yellow'],
    'qwen-72b':               '#8B4513',
}

MODEL_ORDER = [
    'llama-3.1-8b', 'qwen-72b', 'gpt-4o', 'llama-3.3-70b-instruct',
    'llama-3.3-70b', 'gpt-4o-mini', 'mistral-large', 'gemini-flash',
    'deepseek-r1', 'claude-sonnet',
]

MODEL_LABELS = {
    'claude-sonnet':          'Claude',
    'deepseek-r1':            'DeepSeek†',
    'gemini-flash':           'Gemini',
    'gpt-4o':                 'GPT-4o',
    'gpt-4o-mini':            'GPT-4o-mini',
    'llama-3.1-8b':           'Llama-8B',
    'llama-3.3-70b':          'Llama-70B',
    'llama-3.3-70b-instruct': 'Llama-70B-r',
    'mistral-large':          'Mistral',
    'qwen-72b':               'Qwen-72B',
}

STAGE_COLORS = {
    'Base': COLORS['gray'],
    'SFT':  COLORS['blue'],
    'DPO':  COLORS['orange'],
    'RLVR': COLORS['green'],
}


def apply_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'dejavuserif',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.6,
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'patch.linewidth': 0.5,
    })


STAGE_ORDER = ['base', 'sft', 'dpo', 'rl']
STAGE_LABELS = {'base': 'Base', 'sft': 'SFT', 'dpo': 'DPO', 'rl': 'RLVR'}

# Column widths for ACL format
ACL_COL = 3.25   # inches (single column)
ACL_FULL = 6.75  # inches (full width)

# Data loading
import json
from pathlib import Path

RESULTS = Path('/Users/abadila/Desktop/ethics/AEGIS-Multi/analysis/results')

def load_json(name):
    path = RESULTS / name
    with open(path) as f:
        return json.load(f)
