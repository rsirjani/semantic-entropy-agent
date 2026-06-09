r"""Compute the headline metrics from prediction + eval artifacts.

Implements rubric R4 (coverage + INDEPENDENT diversity) and the R5 diversity-
benefit analysis (entropy stratification + off-mode-recovery detection). Pure
post-processing — no GPU/Docker/NLI; runs on the JSON artifacts a run leaves behind.

Per arm it reads:
  - predictions_all_trajectories.jsonl   (instance_id, model_patch, trajectory_id)
  - trajectory_eval_*.json               (per-trajectory `resolved` booleans)
and computes, per instance, diverse-pass@k (unbiased Chen et al. 2021), the count of
DISTINCT patches and mean pairwise structural distance (independent of the branching
NLI), then aggregates across instances with bootstrap CIs.

With a second arm (--compare-*), it reports the paired diverse-pass@k gain
(treatment − vanilla) with a bootstrap CI, and — if per-instance post-search entropy
is available — stratifies the gain by entropy and flags OFF-MODE RECOVERY instances
(treatment passed, vanilla did not, at LOW entropy: the §0.1 case-3 mode-collapse
signature the entropy gate cannot predict).

Usage:
  python scripts/compute_metrics.py \
     --predictions results/branching/predictions_all_trajectories.jsonl \
     --eval results/branching --results-dir results/branching \
     --compare-predictions results/resample_baseline_t0.7/predictions_all_trajectories.jsonl \
     --compare-eval results/resample_baseline_t0.7 \
     --out results/metrics_branching_vs_vanilla_t0.7.json
"""

import argparse
import glob
import json
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.metrics import (
    bootstrap_ci, diverse_pass_at_k, distinct_patch_count, mean_pairwise_distance,
)

import numpy as np


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def load_predictions(path: str) -> dict[str, list[str]]:
    """instance_id -> list of trajectory patches (drops the duplicated 'primary')."""
    by_instance: dict[str, list[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Per-trajectory rows carry a trajectory_id; the best-of duplicate
            # ("primary") does not — skip it so patches aren't double-counted.
            if "trajectory_id" not in rec:
                continue
            by_instance.setdefault(rec["instance_id"], []).append(rec.get("model_patch", "") or "")
    return by_instance


def load_eval(eval_path: str) -> dict[str, list[bool]]:
    """instance_id -> per-trajectory resolved vector, from trajectory_eval_*.json."""
    files = ([eval_path] if eval_path.endswith(".json")
             else sorted(glob.glob(os.path.join(eval_path, "trajectory_eval_*.json"))))
    out: dict[str, list[bool]] = {}
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        iid = d.get("instance_id")
        if not iid:
            continue
        out[iid] = [bool(t.get("resolved")) for t in d.get("trajectories", [])]
    return out


_ENTROPY_RE = re.compile(r"Entropy:\s*([0-9]*\.?[0-9]+)")


def load_entropy(results_dir: str | None, instance_ids) -> dict[str, float]:
    """Best-effort per-instance post-search entropy.

    Strategy arm: the `Entropy:` line in <results_dir>/<iid>/phased_decisions.log.
    SDLG arm fallback: the first `entropy` in <results_dir>/<iid>/branching_log.json.
    Missing → instance omitted (treated as 'unknown' downstream).
    """
    out: dict[str, float] = {}
    if not results_dir:
        return out
    for iid in instance_ids:
        log = os.path.join(results_dir, iid, "phased_decisions.log")
        if os.path.isfile(log):
            try:
                with open(log, "r", encoding="utf-8") as f:
                    m = _ENTROPY_RE.search(f.read())
                if m:
                    out[iid] = float(m.group(1))
                    continue
            except Exception:
                pass
        blog = os.path.join(results_dir, iid, "branching_log.json")
        if os.path.isfile(blog):
            try:
                with open(blog, "r", encoding="utf-8") as f:
                    events = json.load(f)
                ents = [e["entropy"] for e in events if "entropy" in e]
                if ents:
                    out[iid] = float(ents[0])
            except Exception:
                pass
    return out


# --------------------------------------------------------------------------- #
# Per-arm summary
# --------------------------------------------------------------------------- #

def per_instance_table(preds: dict[str, list[str]], evals: dict[str, list[bool]]) -> dict[str, dict]:
    """instance_id -> {k, n_resolved, diverse_pass_at_k, distinct, pairwise, n_nonempty}."""
    table: dict[str, dict] = {}
    for iid in sorted(set(preds) | set(evals)):
        patches = preds.get(iid, [])
        resolved = evals.get(iid, [])
        n = len(resolved)
        table[iid] = {
            "k": n,
            "n_resolved": int(sum(resolved)),
            "diverse_pass_at_k": diverse_pass_at_k(resolved) if n else 0.0,
            "distinct_patches": distinct_patch_count(patches),
            "mean_pairwise_distance": round(mean_pairwise_distance(patches), 4),
            "n_nonempty_patches": sum(1 for p in patches if p.strip()),
        }
    return table


def summarize(table: dict[str, dict], seed: int) -> dict:
    iids = sorted(table)
    def col(key):
        return [table[i][key] for i in iids]
    out = {"n_instances": len(iids)}
    for key in ("diverse_pass_at_k", "distinct_patches", "mean_pairwise_distance"):
        pt, lo, hi = bootstrap_ci(col(key), np.mean, seed=seed)
        out[key] = {"mean": round(pt, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return out


# --------------------------------------------------------------------------- #
# Two-arm comparison + R5 analysis
# --------------------------------------------------------------------------- #

def compare(table_a: dict, table_b: dict, entropy: dict[str, float], seed: int,
            split: float | None) -> dict:
    shared = sorted(set(table_a) & set(table_b))
    gains = [table_a[i]["diverse_pass_at_k"] - table_b[i]["diverse_pass_at_k"] for i in shared]
    pt, lo, hi = bootstrap_ci(gains, np.mean, seed=seed)
    result = {
        "n_shared_instances": len(shared),
        "diverse_pass_at_k_gain": {"mean": round(pt, 4), "ci95": [round(lo, 4), round(hi, 4)]},
        "instances_only_in_a": sorted(set(table_a) - set(table_b)),
        "instances_only_in_b": sorted(set(table_b) - set(table_a)),
    }

    # R5.2 — stratify the gain by post-search entropy (split at median unless given).
    ent_shared = {i: entropy[i] for i in shared if i in entropy}
    if len(ent_shared) >= 2:
        thr = split if split is not None else float(np.median(list(ent_shared.values())))
        strata = {"low_entropy": [], "high_entropy": []}
        for i in shared:
            if i not in ent_shared:
                continue
            bucket = "high_entropy" if ent_shared[i] > thr else "low_entropy"
            strata[bucket].append(table_a[i]["diverse_pass_at_k"] - table_b[i]["diverse_pass_at_k"])
        result["entropy_split_threshold"] = round(thr, 4)
        result["gain_by_stratum"] = {
            b: {"n": len(v), "mean_gain": round(float(np.mean(v)), 4) if v else None}
            for b, v in strata.items()
        }

    # R5.4 — off-mode recovery: treatment passed, vanilla did NOT, at LOW entropy.
    off_mode = []
    for i in shared:
        a_pass = table_a[i]["n_resolved"] > 0
        b_pass = table_b[i]["n_resolved"] > 0
        if a_pass and not b_pass:
            e = entropy.get(i)
            off_mode.append({"instance_id": i, "post_search_entropy": e,
                             "low_entropy": (e is not None and e <= (split if split is not None else 0.5))})
    result["off_mode_recovery_candidates"] = off_mode
    result["note"] = ("off_mode_recovery with low_entropy=True is the §0.1 case-3 "
                      "mode-collapse signature the entropy gate cannot predict.")
    return result


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description="Compute headline metrics (R4/R5) from artifacts.")
    p.add_argument("--predictions", required=True, help="Arm A predictions_all_trajectories.jsonl")
    p.add_argument("--eval", required=True, help="Arm A trajectory_eval dir (or single .json)")
    p.add_argument("--results-dir", default=None, help="Arm A results dir (for entropy extraction)")
    p.add_argument("--compare-predictions", default=None, help="Arm B predictions (e.g. vanilla)")
    p.add_argument("--compare-eval", default=None, help="Arm B trajectory_eval dir")
    p.add_argument("--label-a", default="treatment")
    p.add_argument("--label-b", default="vanilla")
    p.add_argument("--entropy-split", type=float, default=None,
                   help="Entropy boundary for low/high strata (default: median of shared).")
    p.add_argument("--seed", type=int, default=0, help="Bootstrap seed (reproducible CIs).")
    p.add_argument("--out", default=None, help="Write the full result JSON here.")
    args = p.parse_args()

    table_a = per_instance_table(load_predictions(args.predictions), load_eval(args.eval))
    report = {args.label_a: {"summary": summarize(table_a, args.seed), "per_instance": table_a}}

    print(f"\n=== {args.label_a} ===  ({report[args.label_a]['summary']['n_instances']} instances)")
    for k, v in report[args.label_a]["summary"].items():
        if k != "n_instances":
            print(f"  {k}: {v['mean']}  CI95={v['ci95']}")

    if args.compare_predictions and args.compare_eval:
        table_b = per_instance_table(load_predictions(args.compare_predictions),
                                     load_eval(args.compare_eval))
        report[args.label_b] = {"summary": summarize(table_b, args.seed), "per_instance": table_b}
        print(f"\n=== {args.label_b} ===  ({report[args.label_b]['summary']['n_instances']} instances)")
        for k, v in report[args.label_b]["summary"].items():
            if k != "n_instances":
                print(f"  {k}: {v['mean']}  CI95={v['ci95']}")

        entropy = load_entropy(args.results_dir, set(table_a) | set(table_b))
        comp = compare(table_a, table_b, entropy, args.seed, args.entropy_split)
        report["comparison"] = comp
        g = comp["diverse_pass_at_k_gain"]
        print(f"\n=== {args.label_a} − {args.label_b} ===")
        print(f"  diverse-pass@k gain: {g['mean']}  CI95={g['ci95']}  (n={comp['n_shared_instances']})")
        if "gain_by_stratum" in comp:
            print(f"  by entropy (split={comp['entropy_split_threshold']}): {comp['gain_by_stratum']}")
        omr = [o for o in comp["off_mode_recovery_candidates"] if o["low_entropy"]]
        print(f"  off-mode recovery (low-entropy, treatment-only pass): {len(omr)} instance(s) "
              f"{[o['instance_id'] for o in omr]}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
