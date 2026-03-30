"""Generate all figures for the branching results report — V2 vs V3 comparison."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json, re, os

os.makedirs('results/branching/figures', exist_ok=True)

# ─── Color palette ───
C_V2 = '#94A3B8'       # slate gray for v2
C_V3 = '#2563EB'       # blue for v3
C_STRATEGY = '#2563EB'  # blue
C_SDLG = '#7C3AED'     # purple
C_ROOT = '#94A3B8'      # gray
C_ACCENT = '#059669'    # green
C_WARN = '#DC2626'      # red
C_ORANGE = '#EA580C'
C_GRAY = '#6B7280'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 11,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
})

# ─── Load data ───
summary_v3 = json.load(open('results/branching/full_summary.json'))
summary_v2 = json.load(open('results/branching_v2_20260328/full_summary.json'))

with open('results/branching/branching_run.log', 'r', encoding='utf-8', errors='replace') as f:
    log_v3 = f.read()
with open('results/branching_v2_20260328/branching_run.log', 'r', encoding='utf-8', errors='replace') as f:
    log_v2 = f.read()

order = ['12481','16766','18189','12096','15345','23534','22714','19637','18763','19495']
labels = [f'sympy-{n}' for n in order]


# ════════════════════════════════════════════════════════
# FIGURE 1: V2 vs V3 Per-Instance Comparison (trajectories + patches)
# ════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={'height_ratios': [1, 1]})

x = np.arange(len(order))
width = 0.2

v2_traj = []
v2_patch = []
v3_traj = []
v3_patch = []
for num in order:
    iid = f'sympy__sympy-{num}'
    v2d = summary_v2['instances'].get(iid, {})
    v3d = summary_v3['instances'].get(iid, {})
    v2_traj.append(v2d.get('n_trajectories', 0))
    v2_patch.append(len(v2d.get('patches', [])))
    v3_traj.append(v3d.get('n_trajectories', 0))
    v3_patch.append(len(v3d.get('patches', [])))

ax1.bar(x - 1.5*width, v2_traj, width, label='V2 trajectories', color=C_V2, alpha=0.8, edgecolor='white')
ax1.bar(x - 0.5*width, v3_traj, width, label='V3 trajectories', color=C_V3, alpha=0.8, edgecolor='white')
ax1.bar(x + 0.5*width, v2_patch, width, label='V2 patches', color=C_V2, alpha=0.5, edgecolor='white', hatch='///')
ax1.bar(x + 1.5*width, v3_patch, width, label='V3 patches', color=C_V3, alpha=0.5, edgecolor='white', hatch='///')

ax1.set_ylabel('Count')
ax1.set_title('V2 vs V3: Trajectories and Unique Patches per Instance', fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
ax1.legend(loc='upper right', fontsize=8, ncol=2)
ax1.set_ylim(0, max(max(v2_traj), max(v3_traj)) + 2)

# Time comparison
v2_times = []
v3_times = []
for num in order:
    iid = f'sympy__sympy-{num}'
    v2_times.append(summary_v2['instances'].get(iid, {}).get('elapsed_seconds', 0) / 60)
    v3_times.append(summary_v3['instances'].get(iid, {}).get('elapsed_seconds', 0) / 60)

ax2.bar(x - width/2, v2_times, width*1.5, label='V2 (old SDLG dedup)', color=C_V2, alpha=0.8, edgecolor='white')
ax2.bar(x + width/2, v3_times, width*1.5, label='V3 (fixed SDLG dedup)', color=C_V3, alpha=0.8, edgecolor='white')
ax2.set_ylabel('Time (minutes)')
ax2.set_title('Execution Time per Instance', fontweight='bold', pad=8)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig('results/branching/figures/fig1_v2_v3_instance_overview.png', dpi=200, bbox_inches='tight')
plt.savefig('results/branching/figures/fig1_v2_v3_instance_overview.pdf', bbox_inches='tight')
plt.close()
print('Figure 1 saved')


# ════════════════════════════════════════════════════════
# FIGURE 2: Branching tree structure (V3)
# ════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))

y_pos = 0
for i, num in enumerate(order):
    iid = f'sympy__sympy-{num}'
    d = summary_v3['instances'].get(iid, {})
    patches = d.get('patches', [])

    ax.barh(y_pos, 1, height=0.6, color=C_GRAY, alpha=0.5, left=0)
    ax.text(0.5, y_pos, 'SEARCH', ha='center', va='center', fontsize=7, color='white', fontweight='bold')

    ax.barh(y_pos, 0.5, height=0.6, color=C_ACCENT, alpha=0.7, left=1)
    ax.text(1.25, y_pos, 'PROP', ha='center', va='center', fontsize=7, color='white', fontweight='bold')

    if not patches:
        ax.text(1.6, y_pos, '(no patches)', ha='left', va='center', fontsize=7, color=C_GRAY, style='italic')
        y_pos += 1.0
        ax.text(-0.1, y_pos - 0.5, f'sympy-{num}', ha='right', va='center', fontsize=9, fontweight='bold')
        continue

    for j, p in enumerate(patches):
        tid = p['trajectory_id']
        plen = p['patch_len']
        is_sdlg = 'sdlg' in tid
        color = C_SDLG if is_sdlg else C_STRATEGY

        w = max(0.3, min(3.0, plen / 1000))
        y_branch = y_pos + j * 0.7
        ax.barh(y_branch, w, height=0.5, color=color, alpha=0.75, left=1.5)

        label = tid.replace('t0_', '').replace('strategy_', 'S').replace('sdlg_', 'D')
        if tid == 't0':
            label = 'root'
        ax.text(1.5 + w + 0.05, y_branch, f'{label} ({plen}ch)',
                ha='left', va='center', fontsize=7)
        ax.plot([1.5, 1.5], [y_pos, y_branch], color=C_GRAY, linewidth=0.5, alpha=0.5)

    n_p = len(patches)
    mid_y = y_pos + (n_p - 1) * 0.7 / 2
    ax.text(-0.1, mid_y, f'sympy-{num}', ha='right', va='center', fontsize=9, fontweight='bold')

    y_pos += max(n_p, 1) * 0.7 + 0.8

ax.set_xlim(-1.5, 6)
ax.set_ylim(-0.5, y_pos)
ax.invert_yaxis()
ax.set_xlabel('Pipeline Stage')
ax.set_title('V3 Branching Tree Structure: All 10 Instances (SDLG Dedup Fix)', fontweight='bold', pad=12)
ax.set_xticks([])
ax.set_yticks([])

legend_elements = [
    mpatches.Patch(color=C_GRAY, alpha=0.5, label='Search phase'),
    mpatches.Patch(color=C_ACCENT, alpha=0.7, label='Strategy proposal'),
    mpatches.Patch(color=C_STRATEGY, alpha=0.75, label='Strategy branch'),
    mpatches.Patch(color=C_SDLG, alpha=0.75, label='SDLG branch (new)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('results/branching/figures/fig2_branching_tree_v3.png', dpi=200, bbox_inches='tight')
plt.savefig('results/branching/figures/fig2_branching_tree_v3.pdf', bbox_inches='tight')
plt.close()
print('Figure 2 saved')


# ════════════════════════════════════════════════════════
# FIGURE 3: NLI Entailment Score Distributions (V3)
# ════════════════════════════════════════════════════════
nli_v3 = re.findall(r'NLI \[\d+\] vs cluster_rep\[\d+\]: fwd=([\d.]+) bwd=([\d.]+) thr=[\d.]+ -> (\w+)', log_v3)
same_fwd = [float(f) for f, b, r in nli_v3 if r == 'SAME']
same_bwd = [float(b) for f, b, r in nli_v3 if r == 'SAME']
diff_fwd = [float(f) for f, b, r in nli_v3 if r == 'DIFF']
diff_bwd = [float(b) for f, b, r in nli_v3 if r == 'DIFF']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
bins = np.linspace(0, 1, 25)

axes[0].hist(diff_fwd, bins=bins, alpha=0.7, color=C_WARN, label=f'Different cluster (n={len(diff_fwd)})', edgecolor='white')
axes[0].hist(same_fwd, bins=bins, alpha=0.7, color=C_ACCENT, label=f'Same cluster (n={len(same_fwd)})', edgecolor='white')
axes[0].axvline(x=0.7, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold (0.7)')
axes[0].set_xlabel('Forward Entailment P(entailment)')
axes[0].set_ylabel('Count')
axes[0].set_title('Forward NLI Scores', fontweight='bold')
axes[0].legend(fontsize=9)

axes[1].hist(diff_bwd, bins=bins, alpha=0.7, color=C_WARN, label=f'Different cluster (n={len(diff_bwd)})', edgecolor='white')
axes[1].hist(same_bwd, bins=bins, alpha=0.7, color=C_ACCENT, label=f'Same cluster (n={len(same_bwd)})', edgecolor='white')
axes[1].axvline(x=0.7, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold (0.7)')
axes[1].set_xlabel('Backward Entailment P(entailment)')
axes[1].set_ylabel('Count')
axes[1].set_title('Backward NLI Scores', fontweight='bold')
axes[1].legend(fontsize=9)

fig.suptitle('Bidirectional Entailment Score Distributions (DeBERTa-large-MNLI, V3)', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/branching/figures/fig3_nli_distributions.png', dpi=200, bbox_inches='tight')
plt.savefig('results/branching/figures/fig3_nli_distributions.pdf', bbox_inches='tight')
plt.close()
print('Figure 3 saved')


# ════════════════════════════════════════════════════════
# FIGURE 4: Semantic Entropy Distribution (V3)
# ════════════════════════════════════════════════════════
v3_ent = re.findall(r'Semantic analysis: (\d+) clusters, entropy=([\d.]+)', log_v3)
entropy_vals = [float(e) for _, e in v3_ent]
n_clusters_vals = [int(n) for n, _ in v3_ent]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(entropy_vals, bins=15, color=C_V3, alpha=0.8, edgecolor='white')
axes[0].axvline(x=0, color=C_WARN, linestyle='--', linewidth=1.5, alpha=0.7, label='No diversity (H=0)')
axes[0].axvline(x=np.log(5), color=C_ACCENT, linestyle='--', linewidth=1.5, alpha=0.7, label=f'Max for K=5 (H={np.log(5):.2f})')
axes[0].set_xlabel('Semantic Entropy (H)')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Semantic Entropy (V3)', fontweight='bold')
axes[0].legend(fontsize=9)

jitter = np.random.normal(0, 0.08, len(n_clusters_vals))
axes[1].scatter(np.array(n_clusters_vals) + jitter, entropy_vals,
                c=C_SDLG, alpha=0.6, s=50, edgecolors='white', linewidths=0.5)
k_range = np.arange(1, 6.1, 0.1)
axes[1].plot(k_range, np.log(k_range), color=C_WARN, linestyle='--', linewidth=1.5, alpha=0.7, label='H_max = ln(K)')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Semantic Entropy (H)')
axes[1].set_title('Clusters vs. Entropy (V3)', fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].set_xticks([1, 2, 3, 4, 5])

fig.suptitle(f'Semantic Entropy Analysis (N={len(v3_ent)} clustering events)', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/branching/figures/fig4_entropy_distribution.png', dpi=200, bbox_inches='tight')
plt.savefig('results/branching/figures/fig4_entropy_distribution.pdf', bbox_inches='tight')
plt.close()
print('Figure 4 saved')


# ════════════════════════════════════════════════════════
# FIGURE 5: Strategy vs SDLG patch sources (V3)
# ════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))

strat_counts = []
sdlg_counts = []
root_counts = []
for num in order:
    iid = f'sympy__sympy-{num}'
    d = summary_v3['instances'].get(iid, {})
    patches = d.get('patches', [])
    s_c = sum(1 for p in patches if 'strategy' in p['trajectory_id'] and 'sdlg' not in p['trajectory_id'])
    d_c = sum(1 for p in patches if 'sdlg' in p['trajectory_id'])
    r_c = sum(1 for p in patches if p['trajectory_id'] == 't0')
    strat_counts.append(s_c)
    sdlg_counts.append(d_c)
    root_counts.append(r_c)

width = 0.25
ax.bar(x - width, root_counts, width, label='Root (greedy)', color=C_ROOT, alpha=0.85, edgecolor='white')
ax.bar(x, strat_counts, width, label='Strategy branches', color=C_STRATEGY, alpha=0.85, edgecolor='white')
ax.bar(x + width, sdlg_counts, width, label='SDLG branches', color=C_SDLG, alpha=0.85, edgecolor='white')

ax.set_ylabel('Number of Unique Patches')
ax.set_title('V3 Patch Sources: Strategy Proposal vs. SDLG (After Dedup Fix)', fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
ax.legend()
ax.set_ylim(0, max(max(strat_counts), max(sdlg_counts), max(root_counts)) + 2)

for i in range(len(order)):
    total = root_counts[i] + strat_counts[i] + sdlg_counts[i]
    if total > 0:
        y_top = max(root_counts[i], strat_counts[i], sdlg_counts[i])
        ax.text(i, y_top + 0.3, f'={total}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#374151')

plt.tight_layout()
plt.savefig('results/branching/figures/fig5_strategy_vs_sdlg_v3.png', dpi=200, bbox_inches='tight')
plt.savefig('results/branching/figures/fig5_strategy_vs_sdlg_v3.pdf', bbox_inches='tight')
plt.close()
print('Figure 5 saved')


# ════════════════════════════════════════════════════════
# FIGURE 6: Patch Size Diversity (V3)
# ════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))

all_patches = []
for num in order:
    iid = f'sympy__sympy-{num}'
    d = summary_v3['instances'].get(iid, {})
    sizes = [p['patch_len'] for p in d.get('patches', [])]
    all_patches.append(sizes if sizes else [0])

bp = ax.boxplot(all_patches, positions=range(len(order)), widths=0.5,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor=C_V3, alpha=0.3),
                medianprops=dict(color=C_WARN, linewidth=2))

for i, sizes in enumerate(all_patches):
    if sizes != [0]:
        jitter = np.random.normal(0, 0.05, len(sizes))
        ax.scatter(np.full(len(sizes), i) + jitter, sizes,
                   color=C_V3, alpha=0.7, s=40, zorder=3, edgecolors='white', linewidths=0.5)

ax.set_ylabel('Patch Size (characters)')
ax.set_title('V3 Patch Size Distribution: Evidence of Solution Diversity', fontweight='bold', pad=12)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(labels, rotation=30, ha='right')

for i, sizes in enumerate(all_patches):
    if len(sizes) > 1 and sizes != [0]:
        cv = np.std(sizes) / max(np.mean(sizes), 1) * 100
        ax.text(i, max(sizes) + 100, f'CV={cv:.0f}%', ha='center', va='bottom', fontsize=8, color=C_SDLG)

plt.tight_layout()
plt.savefig('results/branching/figures/fig6_patch_diversity_v3.png', dpi=200, bbox_inches='tight')
plt.savefig('results/branching/figures/fig6_patch_diversity_v3.pdf', bbox_inches='tight')
plt.close()
print('Figure 6 saved')


# ════════════════════════════════════════════════════════
# FIGURE 7: System Architecture (unchanged)
# ════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(7, 9.5, 'Semantic Entropy Clustering for Diverse Agentic Code Generation',
        ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(7, 9.1, 'System Architecture', ha='center', va='center', fontsize=11, color=C_GRAY)

def draw_box(ax, bx, by, bw, bh, color, label):
    rect = mpatches.FancyBboxPatch((bx, by), bw, bh, boxstyle='round,pad=0.1',
                                     facecolor=color, alpha=0.15, edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(bx + bw/2, by + bh - 0.3, label, ha='center', va='top', fontsize=11, fontweight='bold', color=color)

draw_box(ax, 0.5, 5.5, 2.5, 3, C_GRAY, 'SEARCH')
ax.text(1.75, 7.2, 'Single trajectory t0', ha='center', fontsize=8, color='#374151')
ax.text(1.75, 6.8, 'Relevance scoring', ha='center', fontsize=8, color='#374151')
ax.text(1.75, 6.4, 'Saturation detection', ha='center', fontsize=8, color='#374151')
ax.annotate('', xy=(3.3, 7), xytext=(3.0, 7), arrowprops=dict(arrowstyle='->', color='#374151', lw=2))

draw_box(ax, 3.5, 5.5, 3, 3, C_ACCENT, 'STRATEGY\nPROPOSAL')
ax.text(5, 7.5, 'LLM proposes K strategies', ha='center', fontsize=8, color='#374151')
ax.text(5, 7.1, 'DeBERTa NLI clustering', ha='center', fontsize=8, color='#374151')
ax.text(5, 6.7, 'Semantic entropy H > tau?', ha='center', fontsize=8, color='#374151')
ax.text(5, 6.2, 'H = -Sum p(c) log p(c)', ha='center', fontsize=9, color=C_ACCENT, fontweight='bold')
ax.annotate('', xy=(6.8, 7), xytext=(6.5, 7), arrowprops=dict(arrowstyle='->', color='#374151', lw=2))

draw_box(ax, 7, 5.5, 3.5, 3, C_V3, 'PATCH + VERIFY')
ax.text(8.75, 7.5, 'Per-strategy trajectories', ha='center', fontsize=8, color='#374151')
ax.text(8.75, 7.1, 'SDLG within-strategy diversity', ha='center', fontsize=8, color='#374151')
ax.text(8.75, 6.7, 'Docker-isolated execution', ha='center', fontsize=8, color='#374151')
ax.text(8.75, 6.2, 'git diff patch submission', ha='center', fontsize=8, color='#374151')
ax.annotate('', xy=(10.8, 7), xytext=(10.5, 7), arrowprops=dict(arrowstyle='->', color='#374151', lw=2))

draw_box(ax, 11, 5.5, 2.5, 3, C_SDLG, 'EVALUATION')
ax.text(12.25, 7.2, 'SWE-bench tests', ha='center', fontsize=8, color='#374151')
ax.text(12.25, 6.8, 'pass@1 (greedy)', ha='center', fontsize=8, color='#374151')
ax.text(12.25, 6.4, 'diverse-pass@1', ha='center', fontsize=8, color='#374151')

draw_box(ax, 0.5, 1, 4, 2.5, C_ORANGE, 'vLLM Server')
ax.text(2.5, 2.5, 'Qwen3-Coder-30B-A3B', ha='center', fontsize=8, color='#374151')
ax.text(2.5, 2.1, '4-bit AWQ quantization', ha='center', fontsize=8, color='#374151')
ax.text(2.5, 1.7, 'RTX 5090 (32GB VRAM)', ha='center', fontsize=8, color='#374151')

draw_box(ax, 5, 1, 4, 2.5, C_ACCENT, 'NLI Server')
ax.text(7, 2.5, 'DeBERTa-large-MNLI', ha='center', fontsize=8, color='#374151')
ax.text(7, 2.1, 'Bidirectional entailment', ha='center', fontsize=8, color='#374151')
ax.text(7, 1.7, 'SDLG token attribution', ha='center', fontsize=8, color='#374151')

draw_box(ax, 9.5, 1, 4, 2.5, C_GRAY, 'Docker (SWE-bench)')
ax.text(11.5, 2.5, 'Per-trajectory containers', ha='center', fontsize=8, color='#374151')
ax.text(11.5, 2.1, 'Filesystem cloning', ha='center', fontsize=8, color='#374151')
ax.text(11.5, 1.7, 'Unit test evaluation', ha='center', fontsize=8, color='#374151')

for xf, xt in [(2.5, 1.75), (2.5, 5), (2.5, 8.75)]:
    ax.plot([xf, xt], [3.5, 5.5], color='#D1D5DB', linestyle=':', linewidth=1)
for xf, xt in [(7, 5), (7, 8.75)]:
    ax.plot([xf, xt], [3.5, 5.5], color='#D1D5DB', linestyle=':', linewidth=1)
for xf, xt in [(11.5, 8.75), (11.5, 12.25)]:
    ax.plot([xf, xt], [3.5, 5.5], color='#D1D5DB', linestyle=':', linewidth=1)

plt.tight_layout()
plt.savefig('results/branching/figures/fig7_architecture.png', dpi=200, bbox_inches='tight')
plt.savefig('results/branching/figures/fig7_architecture.pdf', bbox_inches='tight')
plt.close()
print('Figure 7 saved')


# ════════════════════════════════════════════════════════
# FIGURE 8: Compute Budget (V3)
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

order_data = []
for num in order:
    iid = f'sympy__sympy-{num}'
    d = summary_v3['instances'].get(iid, {})
    order_data.append((d.get('total_steps', 0), d.get('elapsed_seconds', 0)/60, d.get('n_trajectories', 0), num))

steps = [d[0] for d in order_data]
times = [d[1] for d in order_data]
n_trajs = [d[2] for d in order_data]
nums = [d[3] for d in order_data]

scatter = axes[0].scatter(steps, times, c=n_trajs, cmap='viridis', s=100,
                           edgecolors='white', linewidths=1, zorder=3)
for i, num in enumerate(nums):
    axes[0].annotate(num, (steps[i], times[i]), textcoords='offset points',
                     xytext=(5, 5), fontsize=7, color=C_GRAY)
plt.colorbar(scatter, ax=axes[0], label='Trajectories')
axes[0].set_xlabel('Total Agent Steps')
axes[0].set_ylabel('Time (minutes)')
axes[0].set_title('V3 Compute vs. Diversity', fontweight='bold')

sorted_data = sorted(zip(times, nums, n_trajs), reverse=True)
colors = [C_V3 if d[2] > 3 else C_ORANGE if d[2] > 1 else C_ROOT for d in sorted_data]
axes[1].barh(range(len(sorted_data)), [d[0] for d in sorted_data],
             color=colors, alpha=0.8, edgecolor='white')
axes[1].set_yticks(range(len(sorted_data)))
axes[1].set_yticklabels([f'sympy-{d[1]} ({d[2]}T)' for d in sorted_data])
axes[1].set_xlabel('Time (minutes)')
axes[1].set_title('V3 Time by Instance (sorted)', fontweight='bold')
axes[1].invert_yaxis()

for i, d in enumerate(sorted_data):
    axes[1].text(d[0] + 0.5, i, f'{d[0]:.0f}m', va='center', fontsize=8, color='#374151')

plt.tight_layout()
plt.savefig('results/branching/figures/fig8_compute_budget_v3.png', dpi=200, bbox_inches='tight')
plt.savefig('results/branching/figures/fig8_compute_budget_v3.pdf', bbox_inches='tight')
plt.close()
print('Figure 8 saved')


# ════════════════════════════════════════════════════════
# FIGURE 9: V2 vs V3 SDLG effectiveness — THE KEY FIGURE
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: Multi-cluster event count comparison
v2_all_clusters = re.findall(r'Clustered (\d+) intents into (\d+) clusters', log_v2)
v3_all_clusters = re.findall(r'Clustered (\d+) intents into (\d+) clusters', log_v3)

# Separate strategy-level (first per instance) vs SDLG-level
# Strategy-level: the ones that produce 5 clusters
# SDLG-level: the rest (within-strategy, usually 1 cluster)
v2_sdlg_multi = sum(1 for _, nc in v2_all_clusters if 1 < int(nc) < 5)
v2_sdlg_single = sum(1 for _, nc in v2_all_clusters if int(nc) == 1)
v2_strategy = sum(1 for _, nc in v2_all_clusters if int(nc) == 5)

v3_sdlg_multi = sum(1 for _, nc in v3_all_clusters if 1 < int(nc) < 5)
v3_sdlg_single = sum(1 for _, nc in v3_all_clusters if int(nc) == 1)
v3_strategy = sum(1 for _, nc in v3_all_clusters if int(nc) == 5)

categories = ['Strategy-level\n(5 clusters)', 'SDLG multi-cluster\n(2-4 clusters)', 'SDLG single-cluster\n(1 cluster)']
v2_vals = [v2_strategy, v2_sdlg_multi, v2_sdlg_single]
v3_vals = [v3_strategy, v3_sdlg_multi, v3_sdlg_single]

bx = np.arange(len(categories))
bw = 0.35
axes[0].bar(bx - bw/2, v2_vals, bw, label='V2 (old dedup)', color=C_V2, alpha=0.8, edgecolor='white')
axes[0].bar(bx + bw/2, v3_vals, bw, label='V3 (fixed dedup)', color=C_V3, alpha=0.8, edgecolor='white')

for i in range(len(categories)):
    axes[0].text(i - bw/2, v2_vals[i] + 0.3, str(v2_vals[i]), ha='center', fontsize=9, fontweight='bold', color=C_V2)
    axes[0].text(i + bw/2, v3_vals[i] + 0.3, str(v3_vals[i]), ha='center', fontsize=9, fontweight='bold', color=C_V3)

axes[0].set_xticks(bx)
axes[0].set_xticklabels(categories, fontsize=8)
axes[0].set_ylabel('Count')
axes[0].set_title('Clustering Outcomes: V2 vs V3', fontweight='bold')
axes[0].legend(fontsize=8)

# Panel B: Entropy distributions comparison
v2_ent_vals = [float(e) for e in re.findall(r'Semantic analysis: \d+ clusters, entropy=([\d.]+)', log_v2)]
v3_ent_vals = [float(e) for e in re.findall(r'Semantic analysis: \d+ clusters, entropy=([\d.]+)', log_v3)]

# Filter to SDLG-level only (entropy < 1.5, excluding strategy-level max-entropy events)
v2_sdlg_ent = [e for e in v2_ent_vals if e < 1.5]
v3_sdlg_ent = [e for e in v3_ent_vals if e < 1.5]

axes[1].hist(v2_sdlg_ent, bins=15, alpha=0.6, color=C_V2, label=f'V2 (n={len(v2_sdlg_ent)})', edgecolor='white')
axes[1].hist(v3_sdlg_ent, bins=15, alpha=0.6, color=C_V3, label=f'V3 (n={len(v3_sdlg_ent)})', edgecolor='white')
axes[1].axvline(x=0, color=C_WARN, linestyle='--', alpha=0.5)
axes[1].set_xlabel('Semantic Entropy (H)')
axes[1].set_ylabel('Count')
axes[1].set_title('SDLG-Level Entropy (H < 1.5)', fontweight='bold')
axes[1].legend(fontsize=9)

# Panel C: Total SDLG branches per instance
v2_sdlg_per = []
v3_sdlg_per = []
for num in order:
    iid = f'sympy__sympy-{num}'
    v2d = summary_v2['instances'].get(iid, {})
    v3d = summary_v3['instances'].get(iid, {})
    v2_sdlg_per.append(sum(1 for p in v2d.get('patches', []) if 'sdlg' in p['trajectory_id']))
    v3_sdlg_per.append(sum(1 for p in v3d.get('patches', []) if 'sdlg' in p['trajectory_id']))

axes[2].bar(x - bw/2, v2_sdlg_per, bw, label='V2', color=C_V2, alpha=0.8, edgecolor='white')
axes[2].bar(x + bw/2, v3_sdlg_per, bw, label='V3', color=C_SDLG, alpha=0.8, edgecolor='white')
axes[2].set_xticks(x)
axes[2].set_xticklabels([n for n in order], rotation=45, ha='right', fontsize=8)
axes[2].set_ylabel('SDLG Branch Count')
axes[2].set_title('SDLG Branches per Instance', fontweight='bold')
axes[2].legend(fontsize=9)

fig.suptitle('Impact of SDLG Deduplication Fix: V2 (position-only) vs V3 (position+replacement)', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/branching/figures/fig9_v2_v3_sdlg_impact.png', dpi=200, bbox_inches='tight')
plt.savefig('results/branching/figures/fig9_v2_v3_sdlg_impact.pdf', bbox_inches='tight')
plt.close()
print('Figure 9 saved')


print('\nAll 9 figures generated successfully!')
