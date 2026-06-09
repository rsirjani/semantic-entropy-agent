r"""Budget-fairness audit (rubric R6.3).

Per-trajectory step distributions and per-arm compute accounting, computed purely
from the artifacts a run leaves behind (no GPU/Docker). The specific fairness
concern (GOLD_STANDARD R6.3, VALIDATION_BRIEF change 3) is the step-limit
asymmetry: lazy strategy trajectories reset to step 0 and may use up to step_limit
(300) patch steps, vs the matched-k baseline's 250 total. This script SHOWS whether
passing branches actually exploited that headroom or submitted far earlier.

Per arm it reads, for each instance:
  - <results_dir>/<iid>/metadata.json   (patches[].steps, patches[].trajectory_id,
                                          total_steps, elapsed_seconds)
  - <results_dir>/trajectory_eval_<iid>.json  (per-trajectory `resolved`)
joins steps to resolved by trajectory_id (NOT by index), and reports the step
distribution overall and FOR PASSING branches, plus how many passing branches
exceeded a reference cap (default 250 = the baseline's total budget).

Usage:
  python scripts/budget_audit.py --results-dir results/strategy_t0.7 \
      --eval results/strategy_t0.7 --reference-cap 250 \
      --out results/budget_audit_strategy_t0.7.json
"""

import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np


def _resolved_by_tid(eval_path: str) -> dict[str, dict[str, bool]]:
    files = ([eval_path] if eval_path.endswith(".json")
             else sorted(glob.glob(os.path.join(eval_path, "trajectory_eval_*.json"))))
    out: dict[str, dict[str, bool]] = {}
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        iid = d.get("instance_id")
        if not iid:
            continue
        for t in d.get("trajectories", []):
            tid = t.get("trajectory_id")
            if tid is None or tid == "primary":
                continue
            out.setdefault(iid, {})[tid] = bool(t.get("resolved"))
    return out


def _dist(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "min": float(a.min()), "median": float(np.median(a)),
            "mean": round(float(a.mean()), 2), "max": float(a.max())}


def audit(results_dir: str, eval_path: str, reference_cap: int) -> dict:
    resolved = _resolved_by_tid(eval_path)
    all_steps: list[float] = []
    passing_steps: list[float] = []
    total_steps_per_instance: list[float] = []
    elapsed_per_instance: list[float] = []
    over_cap_passing: list[dict] = []
    n_instances = 0

    for meta in sorted(glob.glob(os.path.join(results_dir, "*", "metadata.json"))):
        try:
            with open(meta, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        iid = d.get("instance_id")
        if not iid:
            continue
        n_instances += 1
        if "total_steps" in d:
            total_steps_per_instance.append(float(d["total_steps"]))
        if "elapsed_seconds" in d:
            elapsed_per_instance.append(float(d["elapsed_seconds"]))
        res = resolved.get(iid, {})
        for p in d.get("patches", []):
            tid = p.get("trajectory_id")
            steps = p.get("steps")
            # Skip the best-of duplicate ("primary") so its steps are not counted
            # twice against the genuine trajectory it copies.
            if steps is None or tid == "primary":
                continue
            all_steps.append(float(steps))
            if res.get(tid):
                passing_steps.append(float(steps))
                if steps > reference_cap:
                    over_cap_passing.append({"instance_id": iid, "trajectory_id": tid,
                                             "steps": int(steps)})

    return {
        "results_dir": results_dir,
        "n_instances": n_instances,
        "reference_cap": reference_cap,
        "steps_all_trajectories": _dist(all_steps),
        "steps_passing_trajectories": _dist(passing_steps),
        "total_steps_per_instance": _dist(total_steps_per_instance),
        "elapsed_seconds_per_instance": _dist(elapsed_per_instance),
        "passing_branches_over_reference_cap": over_cap_passing,
        "fairness_note": (
            "If passing_branches_over_reference_cap is empty, no passing branch used "
            "more than the baseline's total step budget — the step_limit asymmetry did "
            "NOT manufacture wins. Per-token accounting requires token logging, which "
            "the current artifacts do not record (steps are the available compute proxy)."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Budget-fairness audit (R6.3).")
    p.add_argument("--results-dir", required=True, help="Arm results dir (contains <iid>/metadata.json).")
    p.add_argument("--eval", required=True, help="trajectory_eval dir (or single .json).")
    p.add_argument("--reference-cap", type=int, default=250,
                   help="Reference step budget to flag passing branches against (default: baseline 250).")
    p.add_argument("--out", default=None, help="Write the audit JSON here.")
    args = p.parse_args()

    report = audit(args.results_dir, args.eval, args.reference_cap)
    print(f"\n=== Budget audit: {args.results_dir} ({report['n_instances']} instances) ===")
    print(f"  steps (all traj):     {report['steps_all_trajectories']}")
    print(f"  steps (passing traj): {report['steps_passing_trajectories']}")
    over = report["passing_branches_over_reference_cap"]
    print(f"  passing branches over cap={args.reference_cap}: {len(over)} {over}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
