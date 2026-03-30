"""Evaluate all trajectory patches from a branching run individually.

For each trajectory, writes a temporary single-prediction JSONL and runs
SWE-bench evaluation. Reports pass/fail per trajectory and computes
diverse-pass@1 (did ANY trajectory solve it?).

Usage:
    python scripts/eval_all_trajectories.py --instance sympy__sympy-12481
    python scripts/eval_all_trajectories.py --predictions results/branching/predictions_all_trajectories.jsonl --instance sympy__sympy-12481
"""

import json
import os
import sys
import tempfile

# Fix Windows encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.run_eval import run_evaluation


def load_latest_trajectories(predictions_path: str, instance_id: str) -> list[dict]:
    """Extract the latest run's trajectory predictions for a given instance."""
    with open(predictions_path) as f:
        lines = [json.loads(l) for l in f if l.strip()]

    # Filter to this instance
    instance_lines = [l for l in lines if l["instance_id"] == instance_id]
    if not instance_lines:
        return []

    # Split into batches by "primary" entries (no trajectory_id)
    batches = []
    current = []
    for l in instance_lines:
        if l.get("trajectory_id") is None and current:
            batches.append(current)
            current = []
        current.append(l)
    if current:
        batches.append(current)

    return batches[-1]  # Latest run


def deduplicate_patches(trajectories: list[dict]) -> list[dict]:
    """Remove trajectories with duplicate patches, keeping the first occurrence."""
    seen = set()
    unique = []
    for t in trajectories:
        patch = t["model_patch"]
        if patch and patch not in seen:
            seen.add(patch)
            unique.append(t)
    return unique


def eval_single_trajectory(
    trajectory: dict,
    instance_id: str,
    run_id: str,
    timeout: int,
) -> dict:
    """Evaluate a single trajectory patch and return the result."""
    # Write a single-prediction JSONL to a temp file
    pred = {
        "instance_id": instance_id,
        "model_name_or_path": trajectory["model_name_or_path"],
        "model_patch": trajectory["model_patch"],
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, dir=os.path.join(PROJECT_ROOT, "results", "branching")
    ) as f:
        f.write(json.dumps(pred) + "\n")
        temp_path = f.name

    tid = trajectory.get("trajectory_id", "primary")

    try:
        print(f"\n{'─'*60}")
        print(f"  Evaluating: {tid} ({len(trajectory['model_patch'])} chars)")
        print(f"{'─'*60}")

        run_evaluation(
            predictions_path=temp_path,
            instance_ids=[instance_id],
            run_id=run_id,
            timeout=timeout,
        )

        # Check the report for pass/fail
        # Reports are at: logs/run_evaluation/{run_id}/{model_name}/{instance_id}/report.json
        report_path = os.path.join(
            "logs", "run_evaluation", run_id,
            trajectory['model_name_or_path'],
            instance_id,
            "report.json",
        )
        resolved = False
        if os.path.exists(report_path):
            with open(report_path) as rf:
                report = json.load(rf)
            inst_report = report.get(instance_id, {})
            resolved = inst_report.get("resolved", False)

        return {"trajectory_id": tid, "resolved": resolved, "patch_len": len(trajectory["model_patch"])}

    finally:
        os.unlink(temp_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate all trajectory patches from a branching run")
    parser.add_argument("--predictions", default="results/branching/predictions_all_trajectories.jsonl")
    parser.add_argument("--instance", required=True, help="Instance ID to evaluate")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--include-duplicates", action="store_true",
                        help="Evaluate duplicate patches too (default: skip them)")
    args = parser.parse_args()

    predictions_path = os.path.join(PROJECT_ROOT, args.predictions)
    if not os.path.exists(predictions_path):
        print(f"ERROR: {predictions_path} not found")
        sys.exit(1)

    trajectories = load_latest_trajectories(predictions_path, args.instance)
    if not trajectories:
        print(f"ERROR: No trajectories found for {args.instance}")
        sys.exit(1)

    print(f"Found {len(trajectories)} trajectories for {args.instance}")

    if not args.include_duplicates:
        unique = deduplicate_patches(trajectories)
        n_dupes = len(trajectories) - len(unique)
        if n_dupes > 0:
            print(f"Skipping {n_dupes} duplicate patches ({len(unique)} unique)")
        trajectories = unique

    for t in trajectories:
        tid = t.get("trajectory_id", "primary")
        print(f"  {tid:30s} {len(t['model_patch']):5d} chars")

    # Evaluate each trajectory
    results = []
    for i, t in enumerate(trajectories):
        tid = t.get("trajectory_id", "primary")
        run_id = f"branching_traj_{tid}"
        result = eval_single_trajectory(t, args.instance, run_id, args.timeout)
        results.append(result)
        status = "PASS" if result["resolved"] else "FAIL"
        print(f"  → {tid}: {status}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS: {args.instance}")
    print(f"{'='*60}")
    n_pass = 0
    for r in results:
        status = "PASS ✓" if r["resolved"] else "FAIL ✗"
        print(f"  {r['trajectory_id']:30s} {r['patch_len']:5d}ch  {status}")
        if r["resolved"]:
            n_pass += 1

    print(f"{'─'*60}")
    print(f"  pass@1 (greedy):   {'PASS' if results[0]['resolved'] else 'FAIL'}")
    print(f"  diverse-pass@1:    {'PASS' if n_pass > 0 else 'FAIL'} ({n_pass}/{len(results)} trajectories)")
    print(f"{'='*60}")

    # Save results
    results_path = os.path.join(
        PROJECT_ROOT, "results", "branching", f"trajectory_eval_{args.instance}.json"
    )
    with open(results_path, "w") as f:
        json.dump({
            "instance_id": args.instance,
            "n_trajectories": len(results),
            "n_resolved": n_pass,
            "pass_at_1": results[0]["resolved"],
            "diverse_pass_at_1": n_pass > 0,
            "trajectories": results,
        }, f, indent=2)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
