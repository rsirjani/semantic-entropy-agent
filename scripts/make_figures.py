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
_METRICS = [
    ("diverse_pass_at_k", "diverse-pass@k (oracle)", (0.0, 1.0)),
    ("distinct_patches", "distinct final patches", None),
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
    return written


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
