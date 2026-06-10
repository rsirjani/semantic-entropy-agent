r"""Regenerate the headline figures from the metric artifacts (rubric R7.3, R10.2).

Consumes the JSON written by scripts/compute_metrics.py (one or more arms, each with
a `summary` block carrying mean + ci95 for diverse_pass_at_k / distinct_patches /
mean_pairwise_distance) and renders grouped bar charts with 95% CI error bars. Pure
post-processing — no GPU/Docker/NLI. Uses the non-interactive Agg backend so it runs
headless.

Usage:
  python scripts/compute_metrics.py ... --out results/metrics.json   # produce input
  python scripts/make_figures.py --metrics results/metrics.json --out-dir figures/
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np

# Metric key -> (axis label, y-limit or None for auto)
# NOTE: distinct_patches is each arm's RAW count at its OWN k — valid as a
# per-arm descriptive, but cross-arm diversity comparisons must use the
# rarefied @k* figure below (raw distinct counts rise mechanically with
# sample size, R4.2); the label says so.
_METRICS = [
    ("diverse_pass_at_k", "diverse-pass@k (oracle)", (0.0, 1.0)),
    ("distinct_patches", "distinct final patches (own k)", None),
    ("mean_pairwise_distance", "mean pairwise patch distance", (0.0, 1.0)),
]


def _arms(report: dict) -> dict[str, dict]:
    """Pull {arm_label: summary} out of a compute_metrics report (skips 'comparison')."""
    out = {}
    for label, block in report.items():
        if isinstance(block, dict) and isinstance(block.get("summary"), dict):
            out[label] = block["summary"]
    return out


def make_figures(report: dict, out_dir: str) -> list[str]:
    arms = _arms(report)
    if not arms:
        raise ValueError("No arms with a 'summary' block found in the metrics JSON.")
    os.makedirs(out_dir, exist_ok=True)
    labels = list(arms)
    written: list[str] = []

    for key, ylabel, ylim in _METRICS:
        means, lo_err, hi_err = [], [], []
        for lab in labels:
            cell = arms[lab].get(key)
            if not cell:
                means.append(0.0); lo_err.append(0.0); hi_err.append(0.0); continue
            m = cell.get("mean", 0.0) or 0.0
            lo, hi = cell.get("ci95", [m, m])
            means.append(m); lo_err.append(max(0.0, m - lo)); hi_err.append(max(0.0, hi - m))

        fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(labels), 3.2))
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=[lo_err, hi_err], capsize=5, color="#4C72B0",
               edgecolor="black", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(f"{ylabel} by arm (95% CI)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = os.path.join(out_dir, f"fig_{key}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    comp_path = make_comparison_figure(report, list(arms), out_dir)
    if comp_path:
        written.append(comp_path)
    return written


def make_comparison_figure(report: dict, arm_labels: list[str], out_dir: str) -> str | None:
    """The cross-arm-valid diversity figure: rarefied distinct patches @ matched k*.

    The per-arm `distinct_patches` bars above are each arm's raw count at its
    OWN k — comparing those across arms is exactly the mechanical-sample-size
    bias R4.2 forbids. The H1 endpoint lives in comparison.rarefied_distinct_at_k_star
    (per-arm levels + CIs at the common k*); this renders it, annotated with the
    H1 sign-flip p (and its power floor) and the artifact-encoded H2 gate status,
    so the figure a reader grabs states the same inference rule as the JSON.
    Returns None when the report has no comparison block (single-arm run).
    """
    comp = report.get("comparison")
    if not isinstance(comp, dict):
        return None
    ra = comp.get("rarefied_distinct_at_k_star")
    if not isinstance(ra, dict):
        return None
    # compute_metrics builds the report dict treatment-first, so the arm
    # summaries' insertion order maps onto (arm_a, arm_b).
    labels = arm_labels if len(arm_labels) == 2 else ["arm_a", "arm_b"]
    means, lo_err, hi_err = [], [], []
    for key in ("arm_a", "arm_b"):
        cell = ra.get(key) or {}
        m = cell.get("mean", 0.0) or 0.0
        lo, hi = cell.get("ci95", [m, m])
        means.append(m); lo_err.append(max(0.0, m - lo)); hi_err.append(max(0.0, hi - m))

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    x = np.arange(2)
    ax.bar(x, means, yerr=[lo_err, hi_err], capsize=5,
           color=["#4C72B0", "#C44E52"], edgecolor="black", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("rarefied distinct patches @ matched k*")
    title = "H1 (diversity): rarefied distinct @k* (95% CI)"
    rg = comp.get("rarefied_distinct_gain") or {}
    if rg.get("paired_sign_flip_p") is not None:
        title += (f"\ngain={rg.get('mean')}  sign-flip p={rg['paired_sign_flip_p']}"
                  f"  (min achievable p={rg.get('min_achievable_p')})")
    fam = comp.get("confirmatory_family") or {}
    h2 = (fam.get("H2_coverage") or {}).get("status")
    if h2:
        title += f"\nH2 status: {h2}"
    ax.set_title(title, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_rarefied_distinct_at_k_star.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Regenerate headline figures (R7.3).")
    p.add_argument("--metrics", required=True, help="compute_metrics.py --out JSON.")
    p.add_argument("--out-dir", default="figures", help="Directory for the PNGs.")
    args = p.parse_args()
    with open(args.metrics, "r", encoding="utf-8") as f:
        report = json.load(f)
    for path in make_figures(report, args.out_dir):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
