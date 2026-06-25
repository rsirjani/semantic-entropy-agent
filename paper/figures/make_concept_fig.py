"""Illustrative leverage-profile figure (publication style). NO real data — a
schematic of the hypothesis the experiment will test, clearly labelled.

Per-step leverage l_k = I(A_k; S | tau_:k-1) is bounded by BOTH
  - step branchiness   H(A_k | tau)   (is the model still exploring?)  -> hits 0 at f*
  - residual outcome   H(S   | tau)   (is anything left to decide?)    -> hits 0 at f†
so l_k <= min(.,.) is uneven / multi-peaked, ~0 at both ends for opposite reasons.
The injectable-by-forcing region is [f*, f†): natural exploration dead, outcome
still reachable. Schematic only.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 12, "axes.titlesize": 12, "font.size": 11,
    "legend.fontsize": 9.0, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.9, "lines.linewidth": 2.0,
    "legend.frameon": False, "figure.autolayout": True,
})

x = np.linspace(0, 1, 600)
fstar, fdag = 0.50, 0.75   # f*: branchiness dies;  f†: outcome locks

# residual outcome entropy H(S|tau): high then collapses at f† (small non-monotone bump)
b_S = 1.0 / (1.0 + np.exp(14 * (x - 0.62)))
b_S += 0.05 * np.exp(-((x - 0.30) / 0.08) ** 2)        # a step reveals options -> slight rise
b_S = np.clip(b_S, 0, 1.02)

# step branchiness H(A_k|tau): LOW at start (reads obvious file), spiky, dead by f*
b_A = (0.30 * np.exp(-((x - 0.12) / 0.06) ** 2)
       + 0.95 * np.exp(-((x - 0.36) / 0.085) ** 2))
b_A *= 1.0 / (1.0 + np.exp(45 * (x - fstar)))          # sharp cutoff at f*
b_A = np.clip(b_A, 0, 1.0)

leverage = np.minimum(b_A, b_S)                        # l_k <= min(.,.)

fig, ax = plt.subplots(figsize=(6.0, 3.7))

# injectable-by-forcing region [f*, f†): natural exploration dead, outcome still reachable
ax.axvspan(fstar, fdag, facecolor="#cfe8cf", alpha=0.55, lw=0, hatch="///",
           edgecolor="#7fb07f")
ax.text((fstar + fdag) / 2, 0.62, "injectable\nby forcing", ha="center",
        va="center", fontsize=8.8, color="#2f6e2f")

# leverage profile (filled): where diversity is actually decided
ax.fill_between(x, 0, leverage, color="#1f4e79", alpha=0.22, lw=0)
ax.plot(x, leverage, color="#1f4e79", lw=1.4,
        label=r"per-step leverage $\ell_k\leq\min(\cdot,\cdot)$")

# the two bounding curves
ax.plot(x, b_A, color="#1f4e79", ls="-",
        label=r"step branchiness $H(A_k\mid\tau)$")
ax.plot(x, b_S, color="#b4651e", ls="--",
        label=r"residual outcome entropy $H(S\mid\tau)$")

for fx, col, lab, xytext in [
    (fstar, "#1f4e79", r"$f^\ast$: model stops exploring", (0.05, 0.92)),
    (fdag,  "#b4651e", r"$f^\dagger$: outcome locks", (0.78, 0.55)),
]:
    ax.axvline(fx, color=col, lw=1.0, ls=":")
    ax.annotate(lab, xy=(fx, 0.0 if fx == fstar else 0.30), xytext=xytext,
                fontsize=9.0, color=col,
                arrowprops=dict(arrowstyle="->", color=col, lw=0.9))

# step-0 caution
ax.annotate("outcome open,\nbut $\\ell\\approx0$ (forced 1st action)",
            xy=(0.02, 0.02), xytext=(0.04, 0.40), fontsize=8.2, color="0.40",
            arrowprops=dict(arrowstyle="->", color="0.55", lw=0.8))

ax.set_xlabel(r"prefix fraction $k/T$  (or resolved-entropy coordinate $\rho$)")
ax.set_ylabel(r"normalized entropy / leverage")
ax.set_title("Per-step leverage is bounded by both curves")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.04)
ax.legend(loc="upper right", fontsize=8.4)
ax.text(0.99, -0.19, "Illustrative (hypothesis); axes schematic, not measured.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.45")

fig.savefig("commitment_concept.pdf", bbox_inches="tight")
fig.savefig("commitment_concept.png", bbox_inches="tight")
print("wrote commitment_concept.pdf/.png")
