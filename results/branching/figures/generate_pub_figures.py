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
baseline_preds = json.load(open('results/baseline/preds.json'))
summary = json.load(open('results/branching/full_summary.json'))

with open('results/branching/branching_run.log', 'r', encoding='utf-8', errors='replace') as f:
    log = f.read()

with open('results/branching/predictions_all_trajectories.jsonl') as f:
    all_traj_preds = [json.loads(l) for l in f if l.strip()]

# Baseline eval
baseline_results = {}
for f in glob.glob('logs/run_evaluation/baseline_v3_fixed/*/sympy__sympy-*/report.json'):
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
# FIGURE 2: Branching Tree — clean horizontal layout
# ════════════════════════════════════════════════════════════════════════════════
with prp.get_context(layout=Layout.NEURIPS, width_frac=1, height_frac=0.6) as (fig, ax):
    ax.set_xlim(-2.2, 6.5)
    ax.axis('off')

    ROW_H = 0.45       # height per trajectory row
    INST_GAP = 0.25     # gap between instances
    BAR_H = 0.32        # bar thickness
    FORK_X = 0.0        # x where branches fork from

    y = 0
    for i, num in enumerate(ORDER):
        iid = f'sympy__sympy-{num}'
        d = summary['instances'].get(iid, {})
        patches = d.get('patches', [])
        br = branching_eval.get(iid, {})
        traj_results = {t['trajectory_id']: t['resolved'] for t in br.get('trajectories', [])}

        n_p = max(len(patches), 1)
        inst_top = y
        inst_bottom = y + (n_p - 1) * ROW_H

        # Instance label on the left
        mid_y = (inst_top + inst_bottom) / 2
        ax.text(-0.15, mid_y, f'sympy-{num}', ha='right', va='center', fontsize=5.5, fontweight='bold')

        if not patches:
            ax.text(0.2, y, 'no patches', ha='left', va='center', fontsize=5, color=C_GRAY, style='italic')
            y += ROW_H + INST_GAP
            continue

        # Draw fork point
        ax.plot([FORK_X, FORK_X], [inst_top, inst_bottom], color='#D1D5DB', linewidth=0.6, solid_capstyle='round')

        for j, p in enumerate(patches):
            tid = p['trajectory_id']
            plen = p['patch_len']
            is_sdlg = 'sdlg' in tid
            resolved = traj_results.get(tid, False)
            row_y = y + j * ROW_H

            # Color: purple=SDLG, blue=strategy/root
            if is_sdlg:
                color = C_SDLG
            else:
                color = C_BRANCH

            # Horizontal connector from fork to bar
            bar_x = 0.15
            ax.plot([FORK_X, bar_x], [row_y, row_y], color='#D1D5DB', linewidth=0.5)

            # Bar proportional to patch size
            w = max(0.5, min(4.0, plen / 700))
            ax.barh(row_y, w, left=bar_x, height=BAR_H, color=color, alpha=0.85,
                    edgecolor='white', linewidth=0.3)

            # Label inside or after bar
            label = tid.replace('t0_', '').replace('strategy_', 'S').replace('sdlg_', 'D')
            if tid == 't0':
                label = 'root'

            label_text = f'{label}  {plen}ch'
            ax.text(bar_x + w + 0.08, row_y, label_text,
                    ha='left', va='center', fontsize=4, color='#6B7280')

        y += n_p * ROW_H + INST_GAP

    ax.set_ylim(-0.3, y)
    ax.invert_yaxis()

    legend_elements = [
        mpatches.Patch(color=C_BRANCH, alpha=0.85, label='Strategy branch'),
        mpatches.Patch(color=C_SDLG, alpha=0.85, label='SDLG branch'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=5,
              frameon=True, fancybox=False, edgecolor='#D1D5DB')
    ax.set_title('Trajectory Branching Tree: 10 SWE-bench Instances, 54 Unique Patches')

    fig.savefig('results/branching/figures/fig2_branching_tree.svg')
print('Figure 2: Branching tree saved')


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Pipeline Architecture
# ════════════════════════════════════════════════════════════════════════════════
with prp.get_context(layout=Layout.NEURIPS, width_frac=1, height_frac=0.22) as (fig, ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.axis('off')

    def draw_phase(x, y, w, h, color, title, items):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                                         facecolor=color, alpha=0.12, edgecolor=color, linewidth=1)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.2, title, ha='center', va='top', fontsize=6.5, fontweight='bold', color=color)
        for i, item in enumerate(items):
            ax.text(x + w/2, y + h - 0.6 - i*0.35, item, ha='center', va='top', fontsize=5, color='#374151')

    def draw_arrow(x1, x2, y):
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                     arrowprops=dict(arrowstyle='->', color='#6B7280', lw=0.8))

    # Phase boxes — full width, no infrastructure row
    bh = 2.8
    by = 0.3
    mid = by + bh/2

    draw_phase(0.1, by, 2.2, bh, C_GRAY, 'SEARCH',
               ['Single trajectory (t0)', 'Read-only commands', 'NLI relevance scoring', 'Saturation detection',
                'Context pruning'])
    draw_arrow(2.35, 2.65, mid)

    draw_phase(2.7, by, 2.2, bh, C_PASS, 'STRATEGY\nPROPOSAL',
               ['LLM proposes K=5 strategies', 'DeBERTa bidirectional NLI', 'Semantic clustering',
                'H = -Sum p(c) ln p(c)'])
    draw_arrow(4.95, 5.25, mid)

    draw_phase(5.3, by, 2.2, bh, C_BRANCH, 'PATCH + VERIFY',
               ['Per-strategy trajectories', 'SDLG token attribution', 'Within-strategy branching',
                'Docker-isolated execution', 'git diff patch submission'])
    draw_arrow(7.55, 7.85, mid)

    draw_phase(7.9, by, 2.0, bh, C_SDLG, 'EVALUATION',
               ['SWE-bench harness', 'FAIL_TO_PASS tests', 'PASS_TO_PASS tests',
                'pass@1 & diverse-pass@1'])

    fig.savefig('results/branching/figures/fig3_pipeline.svg')
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
    traj = json.load(open(f'results/baseline/{iid}/{iid}.traj.json', encoding='utf-8', errors='replace'))
    baseline_steps[num] = len([m for m in traj.get('messages', []) if m.get('role') == 'assistant'])

with prp.get_context(layout=Layout.NEURIPS, width_frac=1, height_frac=0.75,
                      nrows=5, ncols=2, sharex=False, sharey=True) as (fig, axs):
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

        if idx >= 8:  # bottom row
            ax.set_xlabel('Search Step', fontsize=5)
        if idx % 2 == 0:  # left column
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
