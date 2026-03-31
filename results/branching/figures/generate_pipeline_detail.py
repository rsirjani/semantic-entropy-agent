"""
Generate a detailed, infographic-style pipeline diagram for the full branching system.
Matches the aesthetic of semantic_entropy_pipeline.pdf — rounded boxes, subtle fills,
step-by-step walkthrough with concrete examples, decision points, and arrows.

All layout is strictly vertical/sequential with a Y cursor to prevent overlaps.
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─── Colors ───
C_PHASE_BG  = '#F8F9FA'
C_PHASE_BD  = '#E5E7EB'
C_GRAY      = '#6B7280'
C_DARK      = '#1F2937'
C_LIGHT     = '#9CA3AF'
C_SEARCH    = '#6B7280'
C_STRATEGY  = '#059669'
C_PATCH     = '#2563EB'
C_SDLG      = '#7C3AED'
C_EVAL      = '#D97706'
C_PASS      = '#059669'
C_FAIL      = '#DC2626'
C_ENTAIL_Y  = '#ECFDF5'
C_ENTAIL_N  = '#FEF2F2'
C_INPUT_BG  = '#EFF6FF'
C_CLUSTER   = ['#DBEAFE', '#FDE68A', '#E9D5FF']
C_CLUSTER_BD= ['#3B82F6', '#F59E0B', '#8B5CF6']
C_WHITE     = '#FFFFFF'
C_ARROW     = '#9CA3AF'

# ─── We'll do two passes: first measure content, then draw ───
# For simplicity, use a tall enough figure and bbox_inches='tight' to crop.
FIG_W = 11.0
FIG_H = 50.0  # tall enough; tight_layout will crop
fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 10)
ax.set_ylim(0, FIG_H)
ax.axis('off')
fig.patch.set_facecolor(C_WHITE)

# ─── Drawing helpers ───
def rbox(x, y, w, h, fc, ec, alpha=1.0, lw=1.0, zorder=1, pad=0.10):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f'round,pad={pad}',
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=zorder))

def sbox(x, y, w, h, fc, ec, text='', fs=7, tc=C_DARK, bold=False, lw=0.8, zo=2):
    rbox(x, y, w, h, fc, ec, lw=lw, zorder=zo, pad=0.06)
    if text:
        ax.text(x + w/2, y + h/2, text, fontsize=fs, fontweight='bold' if bold else 'normal',
                color=tc, va='center', ha='center', zorder=zo+1)

def txt(x, y, s, fs=6, color=C_DARK, bold=False, ha='left', va='top', zo=3, **kw):
    ax.text(x, y, s, fontsize=fs, fontweight='bold' if bold else 'normal',
            color=color, ha=ha, va=va, zorder=zo, **kw)

def note(x, y, s, fs=5.5, color=C_LIGHT, ha='center'):
    ax.text(x, y, s, fontsize=fs, fontstyle='italic', color=color, va='center', ha=ha, zorder=10)

def arrow_d(x, y1, y2, c=C_ARROW, lw=1.2):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw, mutation_scale=12), zorder=5)

def arrow_r(x1, x2, y, c=C_ARROW, lw=1.0):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw, mutation_scale=10), zorder=5)

# Phase container: draws bg box, returns (inner_top_y, inner_left_x)
# We draw phase containers AFTER content, using recorded positions.
phase_regions = []  # (y_top, y_bot, label, color, num) — filled in, drawn at end

def phase_start(label, color, num):
    """Mark start of a phase. Returns nothing; we record y later."""
    phase_regions.append([None, None, label, color, num])

def phase_mark_top(y):
    phase_regions[-1][0] = y

def phase_mark_bot(y):
    phase_regions[-1][1] = y

# ═══════════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════════
Y = FIG_H - 0.5
txt(5.0, Y, 'Semantic branching pipeline', fs=18, bold=True, color=C_DARK, ha='center')
Y -= 0.55
txt(5.0, Y, 'Diverse agentic code generation via semantic entropy clustering',
    fs=9, color=C_GRAY, ha='center')
Y -= 0.8

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: INPUTS
# ═══════════════════════════════════════════════════════════════════════════════
phase_start('INPUTS', C_SEARCH, 1)
phase_mark_top(Y)

Y -= 0.6  # room for phase header

# Issue box
sbox(0.7, Y - 1.3, 4.5, 1.3, C_INPUT_BG, '#93C5FD')
txt(0.9, Y - 0.1, 'SWE-bench Verified issue', fs=7.5, bold=True)
txt(0.9, Y - 0.4, 'Problem statement:', fs=6.5, color=C_GRAY)
txt(0.9, Y - 0.65, '"Bug: overlapping cycles produce wrong output\n  in the permutation constructor..."',
    fs=6, color='#374151', fontstyle='italic', linespacing=1.3)

# Infrastructure
txt(6.65, Y - 0.05, 'Infrastructure', fs=7, bold=True, color=C_GRAY, ha='center')
ix = 5.8
sbox(ix,       Y - 0.55, 1.7, 0.42, '#F0FDF4', '#86EFAC', 'vLLM server', fs=6.5, tc='#166534', bold=True)
sbox(ix + 2.0, Y - 0.55, 1.7, 0.42, '#FDF4FF', '#D8B4FE', 'DeBERTa NLI', fs=6.5, tc='#6B21A8', bold=True)
sbox(ix,       Y - 1.1,  1.7, 0.42, '#FFF7ED', '#FDBA74', 'Docker env',  fs=6.5, tc='#9A3412', bold=True)
sbox(ix + 2.0, Y - 1.1,  1.7, 0.42, '#F8FAFC', '#CBD5E1', 'SWE-bench',  fs=6.5, tc='#334155', bold=True)

Y -= 1.6
note(5.0, Y, 'Input: one GitHub issue + repository snapshot in a Docker container')
Y -= 0.3
phase_mark_bot(Y)

# Arrow between phases
arrow_d(5.0, Y, Y - 0.4)
Y -= 0.7

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
phase_start('SEARCH  (read-only exploration)', C_SEARCH, 2)
phase_mark_top(Y)
Y -= 0.6

# Agent loop header
txt(1.0, Y, 'ReAct agent loop', fs=7.5, bold=True)
Y -= 0.35

# Three boxes in a row
bw, bh = 2.2, 0.7
sbox(0.7, Y - bh, bw, bh, '#F3F4F6', '#D1D5DB',
     'THOUGHT\n"The bug is in the cycle\nconstructor \u2014 check parse()"', fs=5.5, tc='#374151')
arrow_r(0.7 + bw + 0.05, 0.7 + bw + 0.4, Y - bh/2)
sbox(3.3, Y - bh, bw, bh, '#EFF6FF', '#93C5FD',
     'ACTION\ngrep -rn "def _parse"\n  sympy/combinatorics/', fs=5.5, tc='#1E40AF')
arrow_r(3.3 + bw + 0.05, 3.3 + bw + 0.4, Y - bh/2)
sbox(5.9, Y - bh, bw + 0.3, bh, '#ECFDF5', '#86EFAC',
     'OBSERVATION\nparse() found at line 142\n  validates input cycles...', fs=5.5, tc='#166534')
Y -= bh + 0.15
note(4.5, Y, 'repeat for up to 30 steps (read-only: grep, find, cat, head, ...)', fs=5.5)
Y -= 0.45

# NLI relevance scoring
txt(1.0, Y, 'NLI relevance scoring', fs=7, bold=True)
Y -= 0.3
sbox(0.7, Y - 0.65, 4.2, 0.65, C_WHITE, '#D1D5DB', lw=0.6)
txt(0.9, Y - 0.05, 'Each observation scored against problem statement:', fs=5.5, color=C_GRAY)
txt(0.9, Y - 0.28, 'Step 12: "Found parse() validates cycles"', fs=5.5, color='#374151')
txt(3.8, Y - 0.28, 'rel = 0.82', fs=5.5, bold=True, color=C_PASS)
txt(0.9, Y - 0.48, 'Step 13: "Listed all test files"', fs=5.5, color='#374151')
txt(3.8, Y - 0.48, 'rel = 0.18', fs=5.5, bold=True, color=C_FAIL)

# Saturation detection (right of scoring)
sbox(5.5, Y - 0.65, 4.0, 0.65, '#FEF3C7', '#F59E0B', lw=0.8)
txt(5.7, Y - 0.08, 'Saturation detection', fs=6.5, bold=True, color='#92400E')
txt(5.7, Y - 0.32, '3 consecutive low-relevance steps (< 0.5)', fs=5.5, color='#78350F')
txt(5.7, Y - 0.52, 'detected at step 22  \u2192  Transition to Phase 3', fs=5.5, color='#78350F')
Y -= 0.9

# Output
sbox(2.0, Y - 0.4, 6.0, 0.4, '#F0FDF4', C_PASS,
     'Output: search report  (relevant code locations + context)', fs=6.5, tc='#166534', bold=True)
Y -= 0.6
note(5.0, Y, 'Single trajectory $t_0$ \u2014 explores codebase before any code changes', fs=5.5)
Y -= 0.25
phase_mark_bot(Y)

arrow_d(5.0, Y, Y - 0.4)
Y -= 0.7

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: STRATEGY PROPOSAL
# ═══════════════════════════════════════════════════════════════════════════════
phase_start('STRATEGY PROPOSAL  (first branching point)', C_STRATEGY, 3)
phase_mark_top(Y)
Y -= 0.6

txt(1.0, Y, 'LLM proposes K = 5 diverse fix strategies', fs=7.5, bold=True)
Y -= 0.35

# 5 strategy boxes
strats = [
    ('$S_0$', '"Fix validation\nin parse()..."'),
    ('$S_1$', '"Add input\ntype guard..."'),
    ('$S_2$', '"Rewrite the\nloop logic..."'),
    ('$S_3$', '"Normalize\ninputs before..."'),
    ('$S_4$', '"Refactor cycle\nconstructor..."'),
]
sw, sh = 1.65, 0.7
for i, (label, desc) in enumerate(strats):
    sx = 0.7 + i * (sw + 0.12)
    sbox(sx, Y - sh, sw, sh, C_INPUT_BG, '#93C5FD', lw=0.6)
    txt(sx + sw/2, Y - 0.12, label, fs=7.5, bold=True, color='#1E40AF', ha='center', va='center')
    txt(sx + sw/2, Y - 0.42, desc, fs=5, color='#374151', ha='center', va='center',
        fontstyle='italic', linespacing=1.2)
Y -= sh + 0.4

# Clustering header
txt(1.0, Y, 'Bidirectional NLI entailment clustering', fs=7.5, bold=True)
Y -= 0.25
note(1.0, Y, '(Farquhar et al. 2024 \u2014 compare each pair with DeBERTa-large MNLI)', fs=5.5, ha='left')
Y -= 0.4

# i = 0
txt(0.9, Y, 'i = 0', fs=7, bold=True)
txt(2.0, Y, 'No clusters exist  \u2192  $S_0$ becomes first cluster', fs=6, color='#374151')
txt(8.0, Y - 0.02, '$C_0$ = { $S_0$ \u2605 }', fs=6.5, ha='left', va='center',
    bbox=dict(boxstyle='round,pad=0.08', facecolor=C_CLUSTER[0], edgecolor=C_CLUSTER_BD[0], lw=0.6))
Y -= 0.55

# i = 1
txt(0.9, Y, 'i = 1', fs=7, bold=True)
txt(2.0, Y, 'Compare $S_1$ vs $C_0$ representative ($S_0$) \u2014 two DeBERTa passes', fs=6, color='#374151')
Y -= 0.3
txt(1.2, Y, 'Forward:  classify($S_0$, $S_1$)', fs=5.5, color='#374151')
txt(4.8, Y, '\u2192 P(ent) = 0.85', fs=5.5, color='#374151')
Y -= 0.22
txt(1.2, Y, 'Backward: classify($S_1$, $S_0$)', fs=5.5, color='#374151')
txt(4.8, Y, '\u2192 P(ent) = 0.78', fs=5.5, color='#374151')
Y -= 0.3
sbox(5.2, Y - 0.28, 3.2, 0.28, C_ENTAIL_Y, C_PASS,
     '0.85 > 0.7 \u2713  AND  0.78 > 0.7 \u2713  \u2192 MATCH', fs=5.5, tc=C_PASS, bold=True)
txt(8.7, Y - 0.14, '$C_0$ = { $S_0$ \u2605, $S_1$ }', fs=5.5, ha='left', va='center',
    bbox=dict(boxstyle='round,pad=0.07', facecolor=C_CLUSTER[0], edgecolor=C_CLUSTER_BD[0], lw=0.5))
Y -= 0.55

# i = 2
txt(0.9, Y, 'i = 2', fs=7, bold=True)
txt(2.0, Y, 'Compare $S_2$ vs $C_0$ rep ($S_0$)', fs=6, color='#374151')
Y -= 0.28
txt(1.2, Y, 'Fwd: P(ent) = 0.15   Bwd: P(ent) = 0.11', fs=5.5, color='#374151')
Y -= 0.28
sbox(5.2, Y - 0.28, 2.5, 0.28, C_ENTAIL_N, C_FAIL,
     '0.15 < 0.7 \u2717  \u2192 NO MATCH', fs=5.5, tc=C_FAIL, bold=True)
txt(8.0, Y - 0.14, '\u2192 new cluster', fs=5.5, color=C_GRAY)
Y -= 0.15
txt(8.7, Y - 0.14, '$C_1$ = { $S_2$ \u2605 }', fs=5.5, ha='left', va='center',
    bbox=dict(boxstyle='round,pad=0.07', facecolor=C_CLUSTER[1], edgecolor=C_CLUSTER_BD[1], lw=0.5))
Y -= 0.5

# i = 3, 4 (abbreviated)
txt(0.9, Y, 'i = 3, 4', fs=7, bold=True)
txt(2.3, Y, '$S_3$ matches $C_1$ (both address loop logic);   $S_4$ \u2192 new cluster $C_2$',
    fs=6, color='#374151')
Y -= 0.55

# Result
txt(1.0, Y, 'Result', fs=7.5, bold=True)
txt(2.2, Y, 'K = 3 clusters from N = 5 strategies', fs=6.5, color='#374151')
Y -= 0.4

# Cluster result boxes
for i, (members, size) in enumerate([
    ('$C_0$ = { $S_0$ \u2605, $S_1$ }', 'size = 2'),
    ('$C_1$ = { $S_2$ \u2605, $S_3$ }', 'size = 2'),
    ('$C_2$ = { $S_4$ \u2605 }',         'size = 1'),
]):
    cx = 1.2 + i * 2.8
    sbox(cx, Y - 0.5, 2.4, 0.5, C_CLUSTER[i], C_CLUSTER_BD[i], lw=0.8)
    txt(cx + 1.2, Y - 0.12, members, fs=6, bold=True, color=C_CLUSTER_BD[i], ha='center', va='center')
    txt(cx + 1.2, Y - 0.38, size, fs=5.5, color=C_GRAY, ha='center', va='center')
Y -= 0.75

# Entropy + BRANCH
sbox(1.2, Y - 0.45, 7.6, 0.45, '#F0FDF4', C_STRATEGY, lw=1.0)
txt(5.0, Y - 0.22, '$H = -\\Sigma\\; p(c) \\cdot \\ln\\, p(c) = 1.055$      '
    '$\\longrightarrow$      $H > \\tau$  $\\longrightarrow$  BRANCH',
    fs=8.5, bold=True, color=C_STRATEGY, ha='center', va='center')
Y -= 0.65
note(5.0, Y, 'Fork into K = 3 parallel trajectories (one per cluster, using cluster representative)', fs=5.5)
Y -= 0.25
phase_mark_bot(Y)

arrow_d(5.0, Y, Y - 0.4, c=C_STRATEGY)
Y -= 0.7

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: PATCH + VERIFY
# ═══════════════════════════════════════════════════════════════════════════════
phase_start('PATCH + VERIFY  (per-strategy, lazy sequential)', C_PATCH, 4)
phase_mark_top(Y)
Y -= 0.6

txt(1.0, Y, 'For each strategy cluster  (one Docker container at a time):', fs=7.5, bold=True)
Y -= 0.35

# Three strategy lane boxes
for i, (cid, col, ec) in enumerate([
    ('Strategy $C_0$', C_CLUSTER[0], C_CLUSTER_BD[0]),
    ('Strategy $C_1$', C_CLUSTER[1], C_CLUSTER_BD[1]),
    ('Strategy $C_2$', C_CLUSTER[2], C_CLUSTER_BD[2]),
]):
    sx = 1.0 + i * 2.9
    sbox(sx, Y - 0.42, 2.4, 0.42, col, ec, cid, fs=6.5, tc=ec, bold=True)
    if i < 2:
        txt(sx + 2.6, Y - 0.21, '\u2192', fs=10, color=C_ARROW, ha='center', va='center')
note(9.0, Y - 0.21, 'sequential', fs=5.5, ha='left')
Y -= 0.7

# ─── Detailed view header ───
txt(1.0, Y, 'Detailed view: Strategy $C_0$ trajectory', fs=7, bold=True, color=C_PATCH)
Y -= 0.4

# ─── PATCH sub-phase ───
patch_top = Y
rbox(0.7, Y - 1.8, 9.0, 1.8, '#EFF6FF', '#93C5FD', lw=0.8, zorder=1)
txt(0.9, Y - 0.1, 'PATCH sub-phase', fs=7.5, bold=True, color='#1E40AF')
txt(3.2, Y - 0.1, '(write access enabled: sed -i, patch, cat <<EOF)', fs=5.5, color='#6B7280')
txt(0.9, Y - 0.45, 'Steps 1\u20134:', fs=5.5, bold=True, color='#374151')
txt(2.3, Y - 0.45, 'Read target files, understand code structure', fs=5.5, color='#374151')
txt(0.9, Y - 0.72, 'Step 5:', fs=5.5, bold=True, color=C_SDLG)
txt(2.3, Y - 0.72, 'First write command detected!  \u2192  triggers SDLG diversification (see below)',
    fs=5.5, bold=True, color=C_SDLG)
txt(1.2, Y - 1.0, 'sed -i "s/old_check/new_check/" file.py', fs=5.5, color='#374151',
    fontstyle='italic', fontfamily='monospace')
txt(0.9, Y - 1.35, 'Steps 6+:', fs=5.5, bold=True, color='#374151')
txt(2.3, Y - 1.35, 'Continue implementing fix (each trajectory runs independently in its own container)',
    fs=5.5, color='#374151')
Y -= 2.05

# Arrow to SDLG
arrow_d(5.0, Y + 0.05, Y - 0.25, c=C_SDLG)
Y -= 0.5

# ─── SDLG diversification ───
sdlg_top = Y
sdlg_h = 4.5
rbox(0.7, Y - sdlg_h, 9.0, sdlg_h, '#FAF5FF', '#C084FC', lw=1.2, zorder=1)
txt(0.9, Y - 0.12, 'SDLG diversification', fs=8, bold=True, color=C_SDLG)
txt(4.0, Y - 0.12, '(Aichberger et al. 2025 \u2014 gradient-based token substitution)',
    fs=5.5, color='#6B21A8', fontstyle='italic')

# Step 1: Token attribution
Y -= 0.5
txt(1.0, Y, '\u2776 Token attribution via DeBERTa gradients', fs=6.5, bold=True, color='#581C87')
Y -= 0.28
txt(1.2, Y, 'Identify high-impact tokens in THOUGHT:', fs=5.5, color='#374151')
Y -= 0.25
# Highlighted token visualization
txt(1.2, Y, '"Fix ', fs=6, color='#374151')
txt(1.72, Y, ' validation ', fs=6, bold=True, color=C_SDLG,
    bbox=dict(boxstyle='round,pad=0.04', facecolor='#EDE9FE', edgecolor=C_SDLG, lw=0.5))
txt(3.0, Y, ' in the ', fs=6, color='#374151')
txt(3.68, Y, ' parse ', fs=6, bold=True, color=C_SDLG,
    bbox=dict(boxstyle='round,pad=0.04', facecolor='#EDE9FE', edgecolor=C_SDLG, lw=0.5))
txt(4.5, Y, ' function"', fs=6, color='#374151')
txt(6.0, Y, '\u2190 gradient attribution scores rank these tokens as high-impact',
    fs=5, color='#9CA3AF', fontstyle='italic')

# Step 2: Substitution
Y -= 0.45
txt(1.0, Y, '\u2777 Token substitution + LLM completion', fs=6.5, bold=True, color='#581C87')
Y -= 0.28
alts = [
    ('"Fix  ordering  in the parse..."', 'Alt 1'),
    ('"Fix  constructor  in the merge..."', 'Alt 2'),
    ('"Fix  normalization  in the repr..."', 'Alt 3'),
]
for alt_text, alt_label in alts:
    txt(1.4, Y, alt_text, fs=5.5, color='#374151')
    txt(5.5, Y, '\u2192  ' + alt_label, fs=5.5, color=C_GRAY)
    Y -= 0.2
note(1.4, Y, 'N = 5 alternatives total (greedy + 4 substitutions, split across thought-level and code-level)',
     fs=5, ha='left')

# Step 3: Intent extraction + clustering
Y -= 0.4
txt(1.0, Y, '\u2778 Intent extraction + semantic clustering', fs=6.5, bold=True, color='#581C87')
Y -= 0.25
txt(1.2, Y, 'Extract a one-sentence intent summary per alternative (via LLM)', fs=5.5, color='#374151')
Y -= 0.22
txt(1.2, Y, 'Cluster via bidirectional NLI entailment (same algorithm as Phase 3)', fs=5.5, color='#374151')

# Step 4: Branch decision
Y -= 0.4
txt(1.0, Y, '\u2779 Entropy \u2192 branch decision', fs=6.5, bold=True, color='#581C87')
Y -= 0.3
sbox(1.2, Y - 0.35, 8.0, 0.35, '#EDE9FE', C_SDLG,
     '$H > \\tau$  $\\rightarrow$  fork SDLG children  (one new trajectory per unique non-greedy cluster)',
     fs=6, tc=C_SDLG, bold=True)
Y -= 0.55

# Verify bottom of SDLG box
sdlg_bot_actual = sdlg_top - sdlg_h
# Y should be near sdlg_bot_actual now. Adjust if needed.
Y = min(Y, sdlg_bot_actual + 0.05)

# Arrow to SDLG children
Y -= 0.15
arrow_d(3.0, Y + 0.1, Y - 0.2, c=C_SDLG)
arrow_d(5.0, Y + 0.1, Y - 0.2, c=C_SDLG)
arrow_d(7.0, Y + 0.1, Y - 0.2, c=C_SDLG)
Y -= 0.25

# SDLG children boxes
ch_h = 0.45
for xp, label in [(3.0, 'Primary\n(greedy)'), (5.0, 'SDLG child $D_1$'), (7.0, 'SDLG child $D_2$')]:
    sbox(xp - 0.75, Y - ch_h, 1.5, ch_h, '#FAF5FF', C_SDLG, label, fs=5.5, tc=C_SDLG, bold=True)
Y -= ch_h + 0.3

# Arrow to VERIFY
arrow_d(5.0, Y + 0.1, Y - 0.2)
Y -= 0.4

# ─── VERIFY sub-phase ───
rbox(0.7, Y - 1.0, 9.0, 1.0, '#ECFDF5', '#86EFAC', lw=0.8, zorder=1)
txt(0.9, Y - 0.1, 'VERIFY sub-phase', fs=7.5, bold=True, color='#166534')
txt(0.9, Y - 0.4, 'Run pytest to check correctness', fs=5.5, color='#374151')
txt(3.8, Y - 0.4, '\u2192   Submit patch via git diff', fs=5.5, color='#374151')
txt(6.8, Y - 0.4, '\u2192   Destroy container', fs=5.5, color='#9A3412', fontstyle='italic')
txt(0.9, Y - 0.65, 'Each trajectory (primary + SDLG children) verified independently, then next strategy starts',
    fs=5.5, color='#374151')
Y -= 1.2

# Summary
note(5.0, Y, 'Result: K strategies \u00d7 (1 + SDLG children) trajectories \u2014 each produces an independent patch',
     fs=5.5)
Y -= 0.25
note(5.0, Y, 'Hard cap: B = 30 total trajectories per problem instance', fs=5.5)
Y -= 0.25
phase_mark_bot(Y)

arrow_d(5.0, Y, Y - 0.4, c=C_PATCH)
Y -= 0.7

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
phase_start('EVALUATION', C_EVAL, 5)
phase_mark_top(Y)
Y -= 0.6

# Three boxes in a row
bx_h = 0.65
sbox(0.7, Y - bx_h, 2.6, bx_h, '#FFF7ED', '#FDBA74', lw=0.8)
txt(0.9, Y - 0.1, 'Collect patches', fs=6.5, bold=True, color='#9A3412')
txt(0.9, Y - 0.35, 'All trajectory git diffs\ngathered as predictions', fs=5.5, color='#374151', linespacing=1.3)

arrow_r(3.4, 3.8, Y - bx_h/2, c=C_EVAL)

sbox(3.9, Y - bx_h, 2.8, bx_h, '#FFF7ED', '#FDBA74', lw=0.8)
txt(4.1, Y - 0.1, 'SWE-bench harness', fs=6.5, bold=True, color='#9A3412')
txt(4.1, Y - 0.35, 'FAIL_TO_PASS unit tests\nPASS_TO_PASS regression tests', fs=5.5, color='#374151', linespacing=1.3)

arrow_r(6.8, 7.2, Y - bx_h/2, c=C_EVAL)

sbox(7.3, Y - bx_h, 2.3, bx_h, '#FEF3C7', '#F59E0B', lw=1.0)
txt(7.5, Y - 0.1, 'Metrics', fs=6.5, bold=True, color='#92400E')
txt(7.5, Y - 0.32, 'pass@1  (baseline)', fs=5.5, color='#374151')
txt(7.5, Y - 0.52, 'diverse-pass@1  (ours)', fs=5.5, bold=True, color=C_PASS)

Y -= bx_h + 0.2
note(5.0, Y, 'diverse-pass@1: instance is resolved if ANY trajectory produces a correct patch', fs=5.5)
Y -= 0.3
phase_mark_bot(Y)

# ═══════════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════════
Y -= 0.4
txt(5.0, Y, 'Two diversity axes: strategy-level branching (Phase 3) + implementation-level SDLG (Phase 4)',
    fs=7, bold=True, ha='center')
Y -= 0.35
txt(5.0, Y, 'Semantic entropy as the branching criterion at both levels \u2014 '
    'branch when the model is uncertain, not when outputs differ superficially',
    fs=5.5, color=C_GRAY, ha='center', fontstyle='italic')

# ═══════════════════════════════════════════════════════════════════════════════
# Draw phase containers (background boxes + headers) — drawn last for correct z-order
# ═══════════════════════════════════════════════════════════════════════════════
for (y_top, y_bot, label, color, num) in phase_regions:
    pad_top = 0.15
    pad_bot = 0.15
    h = (y_top + pad_top) - (y_bot - pad_bot)
    y = y_bot - pad_bot
    rbox(0.3, y, 9.4, h, C_PHASE_BG, C_PHASE_BD, alpha=0.5, lw=1.2, zorder=0, pad=0.15)
    txt(0.65, y_top + pad_top - 0.08, f'PHASE {num}', fs=8, bold=True, color=color,
        ha='left', va='top', zo=1, fontfamily='monospace', alpha=0.7)
    txt(2.0, y_top + pad_top - 0.08, label, fs=10, bold=True, color=color, ha='left', va='top', zo=1)

# ─── Save ───
fig.savefig('results/branching/figures/pipeline_detailed.svg', bbox_inches='tight', pad_inches=0.3, dpi=150)
fig.savefig('results/branching/figures/pipeline_detailed.pdf', bbox_inches='tight', pad_inches=0.3, dpi=150)
print('Saved: pipeline_detailed.svg + .pdf')
