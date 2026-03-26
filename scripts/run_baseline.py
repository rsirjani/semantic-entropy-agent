"""Run baseline agent on SWE-bench instances."""

import json
import os
import sys
import time
import yaml

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.agent.react_agent import ReactAgent
from src.inference.vllm_client import VLLMClient
from src.utils.docker_helpers import SWEBenchContainer
from src.utils.logging import TrajectoryLogger
from src.evaluation.dataset import (
    load_swebench_instances,
    save_predictions,
    TARGET_INSTANCES,
    TARGET_INSTANCE_IDS,
)


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "baseline.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_single_instance(
    instance: dict,
    client: VLLMClient,
    config: dict,
) -> dict:
    """Run the agent on a single SWE-bench instance.

    Returns a prediction dict.
    """
    instance_id = instance["instance_id"]
    results_dir = os.path.join(PROJECT_ROOT, config["paths"]["results_dir"])
    instance_dir = os.path.join(results_dir, instance_id)
    os.makedirs(instance_dir, exist_ok=True)

    trajectory_path = os.path.join(instance_dir, "trajectory.jsonl")
    logger = TrajectoryLogger(trajectory_path)

    print(f"\n{'='*60}")
    print(f"Instance: {instance_id}")
    print(f"Difficulty: {TARGET_INSTANCES.get(instance_id, {}).get('difficulty', 'unknown')}")
    print(f"{'='*60}")

    # Reset client token counters for this instance
    client.total_prompt_tokens = 0
    client.total_completion_tokens = 0
    client.total_tokens = 0

    start_time = time.time()
    patch = ""
    submitted = False

    try:
        with SWEBenchContainer(instance_id) as container:
            agent = ReactAgent(
                client=client,
                container=container,
                logger=logger,
                max_steps=config["agent"]["max_steps"],
                max_tokens_per_step=config["agent"]["max_tokens_per_step"],
                temperature=config["agent"]["temperature"],
            )
            patch, submitted = agent.run(instance["problem_statement"])

    except Exception as e:
        print(f"  ERROR: {e}")
        logger.log({"type": "error", "message": str(e)})

    elapsed = time.time() - start_time
    token_usage = client.get_token_usage()

    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Submitted: {submitted}")
    print(f"  Patch length: {len(patch)} chars")
    print(f"  Tokens used: {token_usage['total_tokens']}")

    # Save instance metadata
    meta = {
        "instance_id": instance_id,
        "difficulty": TARGET_INSTANCES.get(instance_id, {}).get("difficulty", "unknown"),
        "submitted": submitted,
        "patch_length": len(patch),
        "elapsed_seconds": elapsed,
        "token_usage": token_usage,
    }
    with open(os.path.join(instance_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "instance_id": instance_id,
        "model_name_or_path": "qwen3-coder-30b-a3b-awq",
        "model_patch": patch,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run baseline agent on SWE-bench instances")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--instances", nargs="*", default=None,
                        help="Specific instance IDs to run (default: all 10)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip instances that already have predictions")
    args = parser.parse_args()

    config = load_config(args.config)
    instance_ids = args.instances or TARGET_INSTANCE_IDS

    print("Loading SWE-bench instances...")
    instances = load_swebench_instances(instance_ids=instance_ids)
    print(f"Loaded {len(instances)} instances")

    # Check for existing predictions
    predictions_path = os.path.join(PROJECT_ROOT, config["paths"]["predictions_file"])
    existing_predictions = []
    existing_ids = set()
    if args.skip_existing and os.path.exists(predictions_path):
        with open(predictions_path, "r") as f:
            for line in f:
                pred = json.loads(line)
                existing_predictions.append(pred)
                existing_ids.add(pred["instance_id"])
        print(f"Found {len(existing_predictions)} existing predictions")

    client = VLLMClient(
        base_url=config["model"]["base_url"],
        model=config["model"]["model_name"],
        api_key=config["model"]["api_key"],
    )

    if not client.check_health():
        print("ERROR: vLLM server is not responding. Start it first with scripts/start_vllm.sh")
        sys.exit(1)

    predictions = list(existing_predictions)

    for instance in instances:
        iid = instance["instance_id"]
        if iid in existing_ids:
            print(f"\nSkipping {iid} (already has prediction)")
            continue

        pred = run_single_instance(instance, client, config)
        predictions.append(pred)

        # Save predictions incrementally
        save_predictions(predictions, predictions_path)

    print(f"\n{'='*60}")
    print(f"All done! {len(predictions)} predictions saved to {predictions_path}")
    print(f"Run evaluation with: python scripts/run_eval_harness.py")


if __name__ == "__main__":
    main()
