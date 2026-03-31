"""Generate publication-quality figures using pub_ready_plots (NeurIPS style)."""
import sys, os
sys.path.insert(0, 'D:/Projects/pub-ready-plots-master/pub-ready-plots-master')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import json, re, os, glob, textwrap

import pub_ready_plots as prp
from pub_ready_plots.styles import Layout

from src.evaluation.dataset import load_swebench_instances, TARGET_INSTANCE_IDS

os.makedirs('results/branching/figures', exist_ok=True)

# ─── Load all data ───
instances_raw = load_swebench_instances(instance_ids=TARGET_INSTANCE_IDS)
gold_patches = {inst['instance_id']: inst.get('patch', '') for inst in instances_raw}
problem_stmts = {inst['instance_id']: inst.get('problem_statement', '') for inst in instances_raw}
# Load baseline predictions — support both old preds.json and new predictions.jsonl
_baseline_preds_path = 'results/baseline/preds.json'
if os.path.exists(_baseline_preds_path):
    baseline_preds = json.load(open(_baseline_preds_path))
else:
    baseline_preds = {}
    with open('results/baseline/predictions.jsonl') as _f:
        for _line in _f:
            if _line.strip():
                _pred = json.loads(_line)
                baseline_preds[_pred['instance_id']] = _pred
summary = json.load(open('results/branching/full_summary.json'))

with open('results/branching/branching_run.log', 'r', encoding='utf-8', errors='replace') as f:
    log = f.read()

with open('results/branching/predictions_all_trajectories.jsonl') as f:
    all_traj_preds = [json.loads(l) for l in f if l.strip()]

# Baseline eval — prefer 250-step results, fall back to v3_fixed
baseline_results = {}
_baseline_eval_pattern = 'logs/run_evaluation/baseline_250step/*/sympy__sympy-*/report.json'
if not glob.glob(_baseline_eval_pattern):
    _baseline_eval_pattern = 'logs/run_evaluation/baseline_v3_fixed/*/sympy__sympy-*/report.json'
for f in glob.glob(_baseline_eval_pattern):
    r = json.load(open(f))
    for iid, data in r.items():
        baseline_results[iid] = data.get('resolved', False)

# Branching eval
branching_eval = {}
for f in glob.glob('results/branching/trajectory_eval_*.json'):
    r = json.load(open(f))
    branching_eval[r['instance_id']] = r

ORDER = ['12481','16766','18189','12096','15345','23534','22714','19637','18763','19495']

# ─── Colors ───
C_GOLD = '#D97706'     # amber
C_BASE = '#6366F1'     # indigo
C_BRANCH = '#2563EB'   # blue
C_SDLG = '#7C3AED'     # purple
C_PASS = '#059669'     # green
C_FAIL = '#DC2626'     # red
C_GRAY = '#9CA3AF'
C_LIGHT = '#F3F4F6'


# Figure 1 removed — main results are better as a table in the paper/slides.


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Branching Tree — two-column landscape layout
# ════════════════════════════════════════════════════════════════════════════════
with prp.get_context(layout=Layout.NEURIPS, width_frac=1, height_frac=0.3,
                      nrows=1, ncols=2) as (fig, axs):

    ROW_H = 0.3
    INST_GAP = 0.12
    BAR_H = 0.22
    FORK_X = 0.0

    left_instances = ORDER[:5]
    right_instances = ORDER[5:]

    for col_idx, (ax, inst_list) in enumerate(zip(axs, [left_instances, right_instances])):
        ax.set_xlim(-1.8, 5.5)
        ax.axis('off')

        y = 0
        for i, num in enumerate(inst_list):
            iid = f'sympy__sympy-{num}'
            d = summary['instances'].get(iid, {})
            patches = d.get('patches', [])

            n_p = max(len(patches), 1)
            mid_y = y + (n_p - 1) * ROW_H / 2
            ax.text(-0.05, mid_y, f'sympy-{num}', ha='right', va='center', fontsize=4.5, fontweight='bold')

            if not patches:
                ax.text(0.2, y, '(no patches)', ha='left', va='center', fontsize=4, color=C_GRAY, style='italic')
                y += ROW_H + INST_GAP
                continue

            ax.plot([FORK_X, FORK_X], [y, y + (n_p - 1) * ROW_H],
                    color='#D1D5DB', linewidth=0.5, solid_capstyle='round')

            for j, p in enumerate(patches):
                tid = p['trajectory_id']
                plen = p['patch_len']
                is_sdlg = 'sdlg' in tid
                color = C_SDLG if is_sdlg else C_BRANCH
                row_y = y + j * ROW_H
                bar_x = 0.12

                ax.plot([FORK_X, bar_x], [row_y, row_y], color='#D1D5DB', linewidth=0.4)
                w = max(0.3, min(2.8, plen / 800))
                ax.barh(row_y, w, left=bar_x, height=BAR_H, color=color, alpha=0.85,
                        edgecolor='white', linewidth=0.3)

                label = tid.replace('t0_', '').replace('strategy_', 'S').replace('sdlg_', 'D')
                if tid == 't0':
                    label = 'root'
                ax.text(bar_x + w + 0.05, row_y, f'{label}  {plen}ch',
                        ha='left', va='center', fontsize=3.5, color='#6B7280')

            y += n_p * ROW_H + INST_GAP

        ax.set_ylim(-0.15, y)
        ax.invert_yaxis()

    legend_elements = [
        mpatches.Patch(color=C_BRANCH, alpha=0.85, label='Strategy branch'),
        mpatches.Patch(color=C_SDLG, alpha=0.85, label='SDLG branch'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=5,
               frameon=True, fancybox=False, edgecolor='#D1D5DB',
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Trajectory Branching Tree: 10 Instances, 54 Unique Patches', fontsize=7)

    fig.savefig('results/branching/figures/fig2_branching_tree.svg', bbox_inches='tight', pad_inches=0.03)
print('Figure 2: Branching tree saved')


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Pipeline Architecture
# ════════════════════════════════════════════════════════════════════════════════
# Use get_mpl_rcParams so we can build a custom figure with proper NeurIPS styling
# but without constrained_layout fighting our manual placement.
_rc3, _w3, _h3 = prp.get_mpl_rcParams(layout=Layout.NEURIPS, width_frac=1, height_frac=0.42)
with plt.rc_context(_rc3):
    fig3 = plt.figure(figsize=(_w3, _h3))
    ax = fig3.add_axes([0, 0, 1, 1])  # fill the entire figure
    ax.set_xlim(0, 10)
    ax.set_ylim(1.0, 7.0)
    ax.axis('off')

    # ─── Font sizes from NeurIPS style: footnote=8, script=7 ───
    FS_TITLE = 8       # phase titles (footnote size)
    FS_BODY  = 6.5     # body text
    FS_MATH  = 7       # math expressions (script size)
    FS_NOTE  = 6       # annotations / captions
    LW = 0.625         # half of NeurIPS linewidth (1.25) for box borders

    # ─── Muted NeurIPS-friendly palette ───
    PAL = {
        'search':   ('#4B5563', '#F9FAFB', '#E5E7EB'),  # gray: text, fill, border
        'strategy': ('#065F46', '#ECFDF5', '#A7F3D0'),  # green
        'patch':    ('#1E40AF', '#EFF6FF', '#BFDBFE'),  # blue
        'eval':     ('#5B21B6', '#F5F3FF', '#DDD6FE'),  # purple
    }

    # ─── Phase box drawing ───
    def phase_box(x, y, w, h, key, num, title, lines):
        tc, fc, ec = PAL[key]
        # Box
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.12',
            facecolor=fc, edgecolor=ec, linewidth=LW, zorder=1))
        # Header bar
        hdr_h = 0.42
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y + h - hdr_h), w, hdr_h, boxstyle='round,pad=0.12',
            facecolor=ec, edgecolor=ec, linewidth=LW, zorder=2))
        # Number + title
        ax.text(x + 0.18, y + h - hdr_h/2, f'{num}', fontsize=FS_TITLE,
                fontweight='bold', color=tc, va='center', ha='left', zorder=3)
        ax.text(x + 0.42, y + h - hdr_h/2, title, fontsize=FS_TITLE,
                fontweight='bold', color=tc, va='center', ha='left', zorder=3)
        # Body lines
        for i, (txt_str, is_math) in enumerate(lines):
            fs = FS_MATH if is_math else FS_BODY
            c = tc if is_math else '#374151'
            ax.text(x + 0.18, y + h - hdr_h - 0.22 - i * 0.32, txt_str,
                    fontsize=fs, color=c, va='top', ha='left', zorder=3)

    # ─── Arrow helpers ───
    def harrow(x1, x2, y, color='#9CA3AF'):
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=LW*2))

    def varrow(x, y1, y2, color='#9CA3AF'):
        ax.annotate('', xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=LW*2))

    # ─── Layout: 2×2 grid ───
    bw = 4.45           # box width
    gap_h = 0.65        # horizontal gap (for arrows)
    gap_v = 0.55        # vertical gap (for arrows)
    x_l = 0.15          # left column x
    x_r = x_l + bw + gap_h  # right column x
    bh_top = 2.65       # top row height
    bh_bot = 2.45       # bottom row height
    y_top = 7.0 - bh_top - 0.05
    y_bot = y_top - gap_v - bh_bot

    # ─── Phase 1: Search ───
    phase_box(x_l, y_top, bw, bh_top, 'search', '1', 'Search', [
        ('Single trajectory $t_0$; read-only commands', False),
        ('ReAct loop: Thought $\\to$ Action $\\to$ Observation', False),
        ('LLM relevance:  $r(o_i) = \\mathrm{LLM\\_score}(q,\\, o_i) \\,/\\, 10$', True),
        ('Saturation: 3 consecutive $r(o_i) < 0.5$', True),
        ('Output: search report (files + context)', False),
    ])

    # ─── Phase 2: Strategy Proposal ───
    phase_box(x_r, y_top, bw, bh_top, 'strategy', '2', 'Strategy Proposal', [
        ('LLM proposes $K{=}5$ fix strategies $\\{S_k\\}$', True),
        ('Bidirectional NLI entailment clustering:', False),
        ('$S_i{\\equiv}S_j \\Leftrightarrow P(e|S_i,S_j){>}\\theta \\wedge P(e|S_j,S_i){>}\\theta$', True),
        ('Semantic entropy:  $H = -\\sum_c p(c)\\ln p(c)$', True),
        ('$H > \\tau \\Rightarrow$ fork one trajectory per cluster', True),
    ])

    # ─── Phase 3: Patch + Verify (right side, below Phase 2) ───
    phase_box(x_r, y_bot, bw, bh_bot, 'patch', '3', 'Patch + Verify', [
        ('Per-cluster trajectory in Docker container', False),
        ('SDLG: $\\nabla_{\\mathbf{e}}\\mathcal{L}_{\\mathrm{NLI}}$ attributes', True),
        ('  high-impact tokens $\\to$ substitute $\\to$ $N{=}5$ alts', True),
        ('Cluster SDLG alternatives; branch if $H > \\tau$', True),
        ('Verify: pytest $\\to$ git diff $\\to$ submit patch', False),
    ])

    # ─── Phase 4: Evaluation (left side, below Phase 1) ───
    phase_box(x_l, y_bot, bw, bh_bot, 'eval', '4', 'Evaluation', [
        ('SWE-bench harness per trajectory', False),
        ('FAIL_TO_PASS: bug-specific unit tests', False),
        ('PASS_TO_PASS: regression tests', False),
        ('diverse-pass@1 $= \\mathbb{1}[\\exists\\, t : \\mathrm{pass}(t)]$', True),
        ('Hard cap: $B{=}30$ trajectories per instance', True),
    ])

    # ─── Arrows ───
    mid_top = y_top + bh_top / 2
    mid_bot = y_bot + bh_bot / 2

    # Phase 1 → Phase 2 (horizontal right)
    harrow(x_l + bw + 0.04, x_r - 0.04, mid_top, PAL['search'][0])

    # Phase 2 → Phase 3 (straight down)
    varrow(x_r + bw/2, y_top - 0.04, y_bot + bh_bot + 0.04, PAL['strategy'][0])

    # Phase 3 → Phase 4 (horizontal left)
    harrow(x_r - 0.04, x_l + bw + 0.04, mid_bot, PAL['patch'][0])

    fig3.savefig('results/branching/figures/fig3_pipeline.svg',
                 bbox_inches='tight', pad_inches=0.04)
    fig3.savefig('results/branching/figures/fig3_pipeline.pdf',
                 bbox_inches='tight', pad_inches=0.04)
    plt.close(fig3)
print('Figure 3: Pipeline architecture saved')


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 4: NLI Score Distributions — Strategy-level vs SDLG-level (separated)
# ════════════════════════════════════════════════════════════════════════════════

# Parse log to separate strategy-level vs SDLG-level NLI comparisons
log_lines = log.split('\n')
context = 'unknown'
strat_nli = []
sdlg_nli = []
for line in log_lines:
    if 'Proposed' in line and 'strategies' in line:
        context = 'strategy'
    elif 'SDLG' in line and ('generated' in line or 'targeting' in line):
        context = 'sdlg'
    elif 'PATCH/VERIFY PHASE' in line or 'SEARCH PHASE' in line:
        context = 'unknown'
    m = re.search(r'NLI \[\d+\] vs cluster_rep\[\d+\]: fwd=([\d.]+) bwd=([\d.]+) thr=[\d.]+ -> (\w+)', line)
    if m:
        fwd, bwd, result = float(m.group(1)), float(m.group(2)), m.group(3)
        if context == 'strategy':
            strat_nli.append((fwd, bwd, result))
        elif context == 'sdlg':
            sdlg_nli.append((fwd, bwd, result))

with prp.get_context(layout=Layout.NEURIPS, width_frac=1, height_frac=0.22, nrows=1, ncols=2) as (fig, axs):
    bins = np.linspace(0, 1, 25)

    # Left: Strategy-level (inter-strategy comparisons during Phase 2)
    strat_diff_fwd = [f for f, b, r in strat_nli if r == 'DIFF']
    strat_same_fwd = [f for f, b, r in strat_nli if r == 'SAME']
    axs[0].hist(strat_diff_fwd, bins=bins, alpha=0.8, color=C_BRANCH, label=f'Different (n={len(strat_diff_fwd)})')
    if strat_same_fwd:
        axs[0].hist(strat_same_fwd, bins=bins, alpha=0.8, color=C_PASS, label=f'Same (n={len(strat_same_fwd)})')
    axs[0].axvline(x=0.7, color='black', linestyle='--', linewidth=0.8, alpha=0.6, label='Threshold (0.7)')
    axs[0].set_xlabel('P(entailment)')
    axs[0].set_ylabel('Count')
    axs[0].set_title(f'Strategy Proposal (n={len(strat_nli)}): all different')
    axs[0].legend(fontsize=5)

    # Right: SDLG-level (within-strategy comparisons during Phase 3)
    sdlg_diff_fwd = [f for f, b, r in sdlg_nli if r == 'DIFF']
    sdlg_same_fwd = [f for f, b, r in sdlg_nli if r == 'SAME']
    axs[1].hist(sdlg_same_fwd, bins=bins, alpha=0.8, color=C_PASS, label=f'Same (n={len(sdlg_same_fwd)})')
    axs[1].hist(sdlg_diff_fwd, bins=bins, alpha=0.8, color=C_SDLG, label=f'Different (n={len(sdlg_diff_fwd)})')
    axs[1].axvline(x=0.7, color='black', linestyle='--', linewidth=0.8, alpha=0.6, label='Threshold (0.7)')
    axs[1].set_xlabel('P(entailment)')
    axs[1].set_title(f'SDLG Within-Strategy (n={len(sdlg_nli)}): mostly same')
    axs[1].legend(fontsize=5)

    fig.savefig('results/branching/figures/fig4_nli_distributions.svg')
print('Figure 4: NLI distributions saved')


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Semantic Entropy Distribution
# ════════════════════════════════════════════════════════════════════════════════
ent_vals = [float(e) for e in re.findall(r'Semantic analysis: \d+ clusters, entropy=([\d.]+)', log)]
n_cluster_vals = [int(n) for n in re.findall(r'Semantic analysis: (\d+) clusters', log)]

with prp.get_context(layout=Layout.NEURIPS, width_frac=1, height_frac=0.18, nrows=1, ncols=2) as (fig, axs):
    axs[0].hist(ent_vals, bins=15, color=C_BRANCH, alpha=0.8)
    axs[0].axvline(x=0, color=C_FAIL, linestyle='--', linewidth=0.8, alpha=0.6, label='H=0 (no diversity)')
    axs[0].axvline(x=np.log(5), color=C_PASS, linestyle='--', linewidth=0.8, alpha=0.6, label=f'H=ln(5)={np.log(5):.2f}')
    axs[0].set_xlabel('Semantic Entropy (H)')
    axs[0].set_ylabel('Count')
    axs[0].set_title('Entropy Distribution')
    axs[0].legend(fontsize=5)

    jitter = np.random.normal(0, 0.06, len(n_cluster_vals))
    axs[1].scatter(np.array(n_cluster_vals) + jitter, ent_vals,
                    c=C_SDLG, alpha=0.5, s=15, edgecolors='white', linewidths=0.3)
    k = np.arange(1, 5.5, 0.1)
    axs[1].plot(k, np.log(k), color=C_FAIL, linestyle='--', linewidth=0.8, alpha=0.6, label=r'$H_{max}=\ln(K)$')
    axs[1].set_xlabel('Number of Clusters (K)')
    axs[1].set_ylabel('Semantic Entropy (H)')
    axs[1].set_title('Clusters vs. Entropy')
    axs[1].set_xticks([1, 2, 3, 4, 5])
    axs[1].legend(fontsize=5)

    fig.savefig('results/branching/figures/fig5_entropy.svg')
print('Figure 5: Entropy distribution saved')


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Patch Diversity (sizes per instance)
# ════════════════════════════════════════════════════════════════════════════════
with prp.get_context(layout=Layout.NEURIPS, width_frac=1, height_frac=0.2) as (fig, ax):
    all_patches_sizes = []
    for num in ORDER:
        iid = f'sympy__sympy-{num}'
        d = summary['instances'].get(iid, {})
        sizes = [p['patch_len'] for p in d.get('patches', []) if p['patch_len'] > 0]
        all_patches_sizes.append(sizes if sizes else [0])

    bp = ax.boxplot(all_patches_sizes, positions=range(len(ORDER)), widths=0.5,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=C_BRANCH, alpha=0.25, linewidth=0.5),
                    medianprops=dict(color=C_FAIL, linewidth=1),
                    whiskerprops=dict(linewidth=0.5),
                    capprops=dict(linewidth=0.5))

    for i, sizes in enumerate(all_patches_sizes):
        if sizes != [0]:
            jitter = np.random.normal(0, 0.06, len(sizes))
            ax.scatter(np.full(len(sizes), i) + jitter, sizes,
                       color=C_BRANCH, alpha=0.6, s=12, edgecolors='white', linewidths=0.3, zorder=3)

    ax.set_ylabel('Patch Size (chars)')
    ax.set_title('Patch Size Distribution per Instance')
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([f'{n}' for n in ORDER], rotation=45, ha='right')
    ax.set_xlabel('SWE-bench Instance (sympy-*)')

    fig.savefig('results/branching/figures/fig6_patch_diversity.svg')
print('Figure 6: Patch diversity saved')


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Compute Budget — Steps vs Time, colored by outcome
# ════════════════════════════════════════════════════════════════════════════════
with prp.get_context(layout=Layout.NEURIPS, width_frac=0.75, height_frac=0.25) as (fig, ax):
    order_data = []
    for num in ORDER:
        iid = f'sympy__sympy-{num}'
        d = summary['instances'].get(iid, {})
        order_data.append((d.get('total_steps', 0), d.get('elapsed_seconds', 0)/60,
                          d.get('n_trajectories', 0), num,
                          branching_eval.get(iid, {}).get('diverse_pass_at_1', False)))

    steps = [d[0] for d in order_data]
    times = [d[1] for d in order_data]
    n_trajs = [d[2] for d in order_data]
    resolved = [d[4] for d in order_data]

    colors = [C_PASS if r else C_FAIL for r in resolved]
    ax.scatter(steps, times, c=colors, s=30, edgecolors='white', linewidths=0.5, zorder=3)

    # Manual label offsets to avoid overlap
    # Clustered points in bottom-left need careful placement
    # Manual label offsets — all pushed upward to stay above x-axis
    label_offsets = {
        '12481': (4, 5),
        '16766': (-4, 6),      # above-left
        '18189': (4, 6),       # above-right
        '12096': (4, -7),      # below (has room, ~10m)
        '15345': (4, 5),
        '23534': (-4, 5),      # left of point
        '22714': (-4, 5),      # above-left
        '19637': (4, 5),
        '18763': (4, -7),      # below (has room, ~134m)
        '19495': (-4, -7),     # below-left (has room, ~10m)
    }
    label_ha = {
        '16766': 'right',
        '23534': 'right',
        '22714': 'right',
        '19495': 'right',
    }

    x_max = max(steps) * 1.15
    y_max = max(times) * 1.1
    y_min = -max(times) * 0.05  # small negative margin so labels aren't clipped
    for i, d in enumerate(order_data):
        num = d[3]
        x_off, y_off = label_offsets.get(num, (4, 3))
        ha = label_ha.get(num, 'left')
        ax.annotate(num, (d[0], d[1]), textcoords='offset points',
                     xytext=(x_off, y_off), fontsize=4.5, color='#6B7280', ha=ha)

    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Total Agent Steps')
    ax.set_ylabel('Time (minutes)')
    ax.set_title('Compute vs. Outcome')

    legend_elements = [
        mpatches.Patch(color=C_PASS, label='Resolved'),
        mpatches.Patch(color=C_FAIL, label='Not resolved'),
    ]
    ax.legend(handles=legend_elements, fontsize=5, frameon=True, fancybox=False, edgecolor='#D1D5DB')

    fig.savefig('results/branching/figures/fig7_compute.svg')
print('Figure 7: Compute budget saved')


# ════════════════════════════════════════════════════════════════════════════════
# FIGURES 8a-8j: Per-Instance Patch Comparison — vertical stacked layout
# ════════════════════════════════════════════════════════════════════════════════
def extract_key_change(patch):
    """Extract the essential change from a patch with line numbers."""
    if not patch:
        return '(no patch submitted)'
    if not patch.strip().startswith('diff'):
        return '(invalid patch — not a diff)'
    lines = patch.split('\n')
    files = [l.split(' b/')[-1] for l in lines if l.startswith('+++ b/')]

    # Parse hunk headers to get starting line numbers
    # Format: @@ -old_start,old_count +new_start,new_count @@
    result = []
    if files:
        result.append(files[0])

    current_line = 0
    for l in lines:
        hunk = re.match(r'^@@ -(\d+)', l)
        if hunk:
            current_line = int(hunk.group(1))
            continue
        if l.startswith('+++') or l.startswith('---') or l.startswith('diff'):
            continue
        if l.startswith('-'):
            content = l[1:].rstrip()
            if content.strip():
                result.append(f'{current_line:>4d} - {content}')
            current_line += 1
        elif l.startswith('+'):
            content = l[1:].rstrip()
            if content.strip():
                result.append(f'{current_line:>4d} + {content}')
            # added lines don't increment old line counter
        else:
            current_line += 1

    # Trim to reasonable length
    if len(result) > 9:
        kept = result[:8]
        kept.append(f'     ... +{len(result)-8} more lines')
        result = kept

    return '\n'.join(result) if result else '(empty patch)'


for idx, num in enumerate(ORDER):
    iid = f'sympy__sympy-{num}'
    gold = gold_patches.get(iid, '')
    base_patch = baseline_preds.get(iid, {}).get('model_patch', '')
    base_resolved = baseline_results.get(iid, False)
    br = branching_eval.get(iid, {})
    br_trajs = br.get('trajectories', [])

    br_preds_for_inst = [p for p in all_traj_preds if p['instance_id'] == iid and p.get('model_patch', '')]
    primary_patch = br_preds_for_inst[0]['model_patch'] if br_preds_for_inst else ''
    primary_resolved = br_trajs[0]['resolved'] if br_trajs else False
    # Skip primary from resolved list — it's already shown as row 3
    resolved_trajs = [t for t in br_trajs if t['resolved'] and t['trajectory_id'] != 'primary']

    # Build rows: (label, patch, resolved, color)
    rows = [
        ('Gold', gold, True, C_GOLD),
        ('Baseline', base_patch, base_resolved, C_BASE),
        ('Branching (primary)', primary_patch, primary_resolved, C_BRANCH),
    ]
    for t in resolved_trajs[:2]:
        tid = t['trajectory_id']
        t_patch = ''
        for p in br_preds_for_inst:
            if p.get('trajectory_id') == tid:
                t_patch = p['model_patch']
                break
        short_tid = tid.replace('t0_', '').replace('strategy_', 'S').replace('sdlg_', 'D')
        rows.append((f'Branching ({short_tid})', t_patch, True, C_PASS))

    n_rows = len(rows)

    # Build text blocks
    text_blocks = []
    max_line_len = 0
    for label, patch, resolved, color in rows:
        if 'Gold' in label:
            tag = ''
        elif resolved:
            tag = '  PASS'
        else:
            tag = '  FAIL'
        change_text = extract_key_change(patch)
        full_text = f'{label}{tag}\n{change_text}'
        n_lines = full_text.count('\n') + 1
        longest_line = max(len(l) for l in full_text.split('\n'))
        max_line_len = max(max_line_len, longest_line)
        text_blocks.append((full_text, color, n_lines))

    total_lines = sum(tb[2] for tb in text_blocks)
    char_w = 0.046    # inches per character at fontsize 5.5 mono
    line_h = 0.105    # inches per line
    gap = 0.06        # inches between blocks
    pad = 0.08        # padding inside each box
    fig_w = max_line_len * char_w + pad * 2 + 0.1
    fig_h = total_lines * line_h + n_rows * (gap + pad * 2) + 0.25

    rc_params, _, _ = prp.get_mpl_rcParams(layout=Layout.NEURIPS, width_frac=1, height_frac=0.15)
    with plt.rc_context(rc_params):
        fig = plt.figure(figsize=(fig_w, fig_h))
        # Use data coordinates in inches
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, fig_w)
        ax.set_ylim(0, fig_h)
        ax.axis('off')

        y = fig_h - 0.22  # below title

        for full_text, color, n_lines in text_blocks:
            box_h = n_lines * line_h + pad * 2
            box_w = fig_w - 0.06

            # Draw background rectangle
            rect = mpatches.FancyBboxPatch(
                (0.03, y - box_h), box_w, box_h,
                boxstyle='round,pad=0.03',
                facecolor=color, alpha=0.06,
                edgecolor=color, linewidth=0.5)
            ax.add_patch(rect)

            # Draw text inside
            ax.text(0.03 + pad, y - pad, full_text,
                    fontsize=5.5, fontfamily='monospace',
                    verticalalignment='top')

            y -= box_h + gap

        ax.text(0.03, fig_h - 0.03, f'sympy-{num}',
                fontsize=7, fontweight='bold', va='top')
        fig.savefig(f'results/branching/figures/fig8_{num}_patches.svg',
                    bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
    print(f'Figure 8-{num}: Patch comparison saved')


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 9: SDLG Impact — Strategy vs SDLG branches
# ════════════════════════════════════════════════════════════════════════════════
with prp.get_context(layout=Layout.NEURIPS, width_frac=1, height_frac=0.2) as (fig, ax):
    x = np.arange(len(ORDER))
    strat_counts = []
    sdlg_counts = []
    root_counts = []
    for num in ORDER:
        iid = f'sympy__sympy-{num}'
        d = summary['instances'].get(iid, {})
        patches = d.get('patches', [])
        strat_counts.append(sum(1 for p in patches if 'strategy' in p['trajectory_id'] and 'sdlg' not in p['trajectory_id']))
        sdlg_counts.append(sum(1 for p in patches if 'sdlg' in p['trajectory_id']))
        root_counts.append(sum(1 for p in patches if p['trajectory_id'] == 't0'))

    w = 0.25
    ax.bar(x - w, root_counts, w, label='Root', color=C_GRAY, alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.bar(x, strat_counts, w, label='Strategy', color=C_BRANCH, alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.bar(x + w, sdlg_counts, w, label='SDLG', color=C_SDLG, alpha=0.8, edgecolor='white', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}' for n in ORDER], rotation=45, ha='right')
    ax.set_xlabel('SWE-bench Instance (sympy-*)')
    ax.set_ylabel('Unique Patches')
    ax.set_title('Patch Sources: Root vs. Strategy vs. SDLG Branches')
    ax.legend(fontsize=5, frameon=True, fancybox=False, edgecolor='#D1D5DB')

    fig.savefig('results/branching/figures/fig9_sdlg_impact.svg')
print('Figure 9: SDLG impact saved')


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 10: Key Finding — sympy-19637 deep dive
# ════════════════════════════════════════════════════════════════════════════════
with prp.get_context(layout=Layout.NEURIPS, width_frac=1, height_frac=0.2) as (fig, ax):
    iid = 'sympy__sympy-19637'
    br = branching_eval.get(iid, {})
    trajs = br.get('trajectories', [])

    names = [t['trajectory_id'].replace('t0_', '').replace('strategy_', 'S').replace('sdlg_', 'D')
             for t in trajs]
    if trajs and trajs[0]['trajectory_id'] == 'primary':
        names[0] = 'primary'
    resolved = [t['resolved'] for t in trajs]
    patch_lens = [t['patch_len'] for t in trajs]

    colors = [C_PASS if r else C_FAIL for r in resolved]
    bars = ax.barh(range(len(names)), patch_lens, color=colors, alpha=0.8,
                    edgecolor='white', linewidth=0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=5)
    ax.set_xlabel('Patch Size (chars)')
    ax.set_title('sympy-19637: Baseline FAILS, SDLG Branches PASS')
    ax.invert_yaxis()

    # Annotate
    for i, (r, pl) in enumerate(zip(resolved, patch_lens)):
        label = 'PASS' if r else 'FAIL'
        ax.text(pl + 20, i, label, va='center', fontsize=4.5,
                color=C_PASS if r else C_FAIL, fontweight='bold' if r else 'normal')

    fig.savefig('results/branching/figures/fig10_19637_deep_dive.svg')
print('Figure 10: sympy-19637 deep dive saved')


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 11: Search Phase — Relevance Scoring & Cutoff per Instance
# ════════════════════════════════════════════════════════════════════════════════

# Gather search data
search_data = {}
for num in ORDER:
    iid = f'sympy__sympy-{num}'
    steps = json.load(open(f'results/branching/{iid}/step_log.json'))
    search_steps = [s for s in steps if s.get('phase') == 'SEARCH']
    scores = []
    for s in search_steps:
        if s.get('blocked'):
            scores.append(('blocked', 0.0))
        elif s.get('relevance') is not None and s.get('relevance') != '':
            scores.append(('relevant' if s.get('is_relevant') else 'irrelevant', float(s['relevance'])))
    search_data[num] = scores

# Baseline step counts
baseline_steps = {}
for num in ORDER:
    iid = f'sympy__sympy-{num}'
    _traj_path = f'results/baseline/{iid}/{iid}.traj.json'
    _trajl_path = f'results/baseline/{iid}/trajectory.jsonl'
    if os.path.exists(_traj_path):
        traj = json.load(open(_traj_path, encoding='utf-8', errors='replace'))
        baseline_steps[num] = len([m for m in traj.get('messages', []) if m.get('role') == 'assistant'])
    elif os.path.exists(_trajl_path):
        with open(_trajl_path, encoding='utf-8', errors='replace') as _tf:
            _entries = [json.loads(l) for l in _tf if l.strip()]
        baseline_steps[num] = len([e for e in _entries if e.get('role') == 'assistant'])
    else:
        baseline_steps[num] = 0

with prp.get_context(layout=Layout.NEURIPS, width_frac=1, height_frac=0.5,
                      nrows=2, ncols=5, sharex=False, sharey=True) as (fig, axs):
    axs_flat = axs.flatten()

    for idx, num in enumerate(ORDER):
        ax = axs_flat[idx]
        scores = search_data[num]
        n_search = len(scores)
        n_baseline = baseline_steps[num]

        if not scores:
            ax.text(0.5, 0.5, 'No search data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=5, color=C_GRAY)
            ax.set_title(f'sympy-{num}', fontsize=6)
            continue

        xs = list(range(1, len(scores) + 1))
        ys = [s[1] for s in scores]
        colors_pts = []
        for s_type, _ in scores:
            if s_type == 'relevant':
                colors_pts.append(C_PASS)
            elif s_type == 'blocked':
                colors_pts.append(C_FAIL)
            else:
                colors_pts.append(C_GRAY)

        # Plot relevance scores as colored stems
        ax.vlines(xs, 0, ys, colors=colors_pts, linewidth=0.8, alpha=0.7)
        ax.scatter(xs, ys, c=colors_pts, s=12, zorder=3, edgecolors='white', linewidths=0.3)

        # Threshold line
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=0.5, alpha=0.4)

        # Cutoff marker (last search step)
        ax.axvline(x=n_search, color=C_BRANCH, linestyle=':', linewidth=0.6, alpha=0.6)

        # Baseline total steps for comparison (as text)
        ax.text(0.97, 0.95, f'baseline: {n_baseline} steps\nsearch: {n_search} steps',
                transform=ax.transAxes, fontsize=4, ha='right', va='top', color='#6B7280',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='#D1D5DB', linewidth=0.3, pad=1.5))

        ax.set_title(f'sympy-{num}', fontsize=6)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0.5, max(n_search + 0.5, 2))

        if idx >= 5:  # bottom row
            ax.set_xlabel('Step', fontsize=5)
        if idx % 5 == 0:  # left column
            ax.set_ylabel('Relevance', fontsize=5)

    # Shared legend at bottom
    legend_elements = [
        mpatches.Patch(color=C_PASS, label='Relevant (above threshold)'),
        mpatches.Patch(color=C_GRAY, label='Irrelevant'),
        mpatches.Patch(color=C_FAIL, label='Blocked (write cmd)'),
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=0.5, label='Threshold (0.5)'),
        plt.Line2D([0], [0], color=C_BRANCH, linestyle=':', linewidth=0.6, label='Search cutoff'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=5,
               frameon=True, fancybox=False, edgecolor='#D1D5DB',
               bbox_to_anchor=(0.5, -0.025))

    fig.suptitle('Search Phase: NLI Relevance Scoring per Step (Branching vs. Baseline Step Count)', fontsize=7)
    fig.savefig('results/branching/figures/fig11_search_relevance.svg', bbox_inches='tight', pad_inches=0.05)
print('Figure 11: Search relevance saved')


print('\nAll figures generated!')
