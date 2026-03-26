"""Collect evaluation results into a summary."""

import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.dataset import TARGET_INSTANCES, TARGET_INSTANCE_IDS


def load_eval_reports(run_id: str = "baseline_v1", model_name: str = "qwen3-coder-30b-a3b-awq") -> dict:
    """Load SWE-bench evaluation reports."""
    reports = {}
    reports_dir = os.path.join(
        PROJECT_ROOT, "logs", "run_evaluation", run_id, model_name
    )

    if not os.path.exists(reports_dir):
        print(f"WARNING: Reports directory not found: {reports_dir}")
        return reports

    for instance_id in TARGET_INSTANCE_IDS:
        report_path = os.path.join(reports_dir, instance_id, "report.json")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                reports[instance_id] = json.load(f)
        else:
            print(f"  No report for {instance_id}")

    return reports


def load_trajectory_metadata(results_dir: str) -> dict:
    """Load trajectory metadata for each instance."""
    metadata = {}
    for instance_id in TARGET_INSTANCE_IDS:
        meta_path = os.path.join(results_dir, instance_id, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata[instance_id] = json.load(f)
    return metadata


def _count_agent_steps(trajectory_path: str) -> int:
    """Count the number of assistant steps in a trajectory file."""
    if not os.path.exists(trajectory_path):
        return 0
    count = 0
    with open(trajectory_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                event = json.loads(line)
                if event.get("type") == "assistant":
                    count += 1
            except json.JSONDecodeError:
                pass
    return count


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect evaluation results")
    parser.add_argument("--run-id", default="baseline_v1")
    parser.add_argument("--results-dir", default="results/baseline")
    parser.add_argument("--output", default="results/baseline/summary.json")
    args = parser.parse_args()

    results_dir = os.path.join(PROJECT_ROOT, args.results_dir)
    output_path = os.path.join(PROJECT_ROOT, args.output)

    print("Loading evaluation reports...")
    reports = load_eval_reports(args.run_id)
    print(f"Found {len(reports)} reports")

    print("Loading trajectory metadata...")
    metadata = load_trajectory_metadata(results_dir)
    print(f"Found {len(metadata)} metadata files")

    # Build summary
    instances = {}
    total_resolved = 0

    for instance_id in TARGET_INSTANCE_IDS:
        report = reports.get(instance_id, {})
        meta = metadata.get(instance_id, {})

        # Parse SWE-bench report
        resolved = False
        f2p = {"success": [], "failure": []}
        p2p = {"success": [], "failure": []}

        if report:
            # SWE-bench report is nested under instance_id key
            inst_report = report.get(instance_id, report)
            resolved = inst_report.get("resolved", False)
            f2p_results = inst_report.get("tests_status", {}).get("FAIL_TO_PASS", {})
            p2p_results = inst_report.get("tests_status", {}).get("PASS_TO_PASS", {})

            if isinstance(f2p_results, dict):
                f2p["success"] = [t for t, v in f2p_results.items() if v == "PASSED"]
                f2p["failure"] = [t for t, v in f2p_results.items() if v != "PASSED"]
            elif isinstance(f2p_results, list):
                f2p["success"] = f2p_results

            if isinstance(p2p_results, dict):
                p2p["success"] = [t for t, v in p2p_results.items() if v == "PASSED"]
                p2p["failure"] = [t for t, v in p2p_results.items() if v != "PASSED"]
            elif isinstance(p2p_results, list):
                p2p["success"] = p2p_results

        if resolved:
            total_resolved += 1

        instances[instance_id] = {
            "resolved": resolved,
            "difficulty": TARGET_INSTANCES.get(instance_id, {}).get("difficulty", "unknown"),
            "submitted": meta.get("submitted", False),
            "steps": _count_agent_steps(os.path.join(results_dir, instance_id, "trajectory.jsonl")),
            "total_tokens": meta.get("token_usage", {}).get("total_tokens", 0),
            "elapsed_seconds": meta.get("elapsed_seconds", 0),
            "patch_length": meta.get("patch_length", 0),
            "f2p": f2p,
            "p2p": p2p,
        }

    summary = {
        "run_id": args.run_id,
        "model": "qwen3-coder-30b-a3b-awq",
        "instances": instances,
        "aggregate": {
            "pass_at_1": total_resolved / len(TARGET_INSTANCE_IDS) if TARGET_INSTANCE_IDS else 0,
            "total_resolved": total_resolved,
            "total_instances": len(TARGET_INSTANCE_IDS),
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print results table
    print(f"\n{'='*80}")
    print(f"BASELINE RESULTS (pass@1)")
    print(f"{'='*80}")
    print(f"{'Instance ID':<30} {'Difficulty':<15} {'Result':<10} {'Tokens':<10} {'Time':<10}")
    print(f"{'-'*80}")

    for iid in TARGET_INSTANCE_IDS:
        inst = instances[iid]
        result = "PASS" if inst["resolved"] else "FAIL"
        tokens = inst["total_tokens"]
        elapsed = f"{inst['elapsed_seconds']:.0f}s"
        print(f"{iid:<30} {inst['difficulty']:<15} {result:<10} {tokens:<10} {elapsed:<10}")

    print(f"{'-'*80}")
    print(f"pass@1: {summary['aggregate']['pass_at_1']:.1%} ({total_resolved}/{len(TARGET_INSTANCE_IDS)})")
    print(f"\nSummary saved to: {output_path}")


if __name__ == "__main__":
    main()
