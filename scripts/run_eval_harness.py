"""Run SWE-bench evaluation on agent predictions."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.run_eval import run_evaluation
from src.evaluation.dataset import TARGET_INSTANCE_IDS


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate agent predictions with SWE-bench")
    parser.add_argument("--predictions", default="results/baseline/predictions.jsonl",
                        help="Path to predictions JSONL file")
    parser.add_argument("--run-id", default="baseline_v1")
    parser.add_argument("--instances", nargs="*", default=None)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    predictions_path = os.path.join(PROJECT_ROOT, args.predictions)
    instance_ids = args.instances or TARGET_INSTANCE_IDS

    if not os.path.exists(predictions_path):
        print(f"ERROR: Predictions file not found: {predictions_path}")
        print("Run the baseline agent first: python scripts/run_baseline.py")
        sys.exit(1)

    print(f"Running SWE-bench evaluation...")
    print(f"  Predictions: {predictions_path}")
    print(f"  Instances: {len(instance_ids)}")
    print(f"  Run ID: {args.run_id}")

    run_evaluation(
        predictions_path=predictions_path,
        instance_ids=instance_ids,
        run_id=args.run_id,
        timeout=args.timeout,
    )

    print("\nEvaluation complete!")
    print(f"Reports saved to: logs/run_evaluation/{args.run_id}/")
    print(f"Collect results with: python scripts/collect_results.py")


if __name__ == "__main__":
    main()
