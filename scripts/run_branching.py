"""Run semantic branching agent on SWE-bench instances."""

import json
import logging
import os
import sys
import time
import yaml

# Fix Windows encoding for mini-swe-agent's Unicode banner
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import litellm

from src.agent.phased_orchestrator import PhasedOrchestrator
from src.diversity.nli_client import NLIClient
from src.evaluation.dataset import (
    load_swebench_instances,
    save_predictions,
    TARGET_INSTANCES,
    TARGET_INSTANCE_IDS,
)


def setup_logging(results_dir: str):
    """Set up logging to file and console."""
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "branching_run.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    # Quiet down noisy loggers
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def reset_litellm_clients():
    """Clear litellm's cached httpx clients to prevent stale connection hangs."""
    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is not None:
        cache.cache_dict.clear()
        cache.ttl_dict.clear()
        cache.expiration_heap.clear()


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "branching.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_eval_image(instance_id: str) -> str:
    """Find the Docker eval image for an instance, handling naming variations.

    SWE-bench eval images may use different naming conventions:
    - sweb.eval.x86_64.{instance_id}:latest (old format)
    - swebench/sweb.eval.x86_64.{repo_hash}_{short_id}:latest (new format)
    """
    import subprocess

    # Try exact match first (old format)
    candidate = f"sweb.eval.x86_64.{instance_id}:latest"
    r = subprocess.run(["docker", "image", "inspect", candidate],
                       capture_output=True, timeout=30)
    if r.returncode == 0:
        return candidate

    # Search all Docker images for matching eval image (new swebench/ format)
    # instance_id like "sympy__sympy-12481" → search for images containing "sympy-12481"
    short_id = instance_id.split("__")[-1]  # "sympy-12481"
    r = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                       capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        for line in r.stdout.strip().splitlines():
            line = line.strip()
            if short_id in line and "sweb.eval" in line:
                return line

    # Fallback to original format
    return candidate


def build_env_config(config: dict, instance_id: str) -> dict:
    """Build DockerEnvironment config for a specific instance."""
    env_config = dict(config.get("environment", {}))
    env_config["image"] = find_eval_image(instance_id)
    # Convert interpreter list from YAML
    if "interpreter" in env_config and isinstance(env_config["interpreter"], list):
        pass  # Already a list, good
    # Remove environment_class key (not part of DockerEnvironmentConfig)
    env_config.pop("environment_class", None)
    return env_config


def run_single_instance(
    instance: dict,
    config: dict,
    nli_model: NLIClient,
) -> dict:
    """Run the branching agent on a single SWE-bench instance."""
    instance_id = instance["instance_id"]
    results_dir = os.path.join(PROJECT_ROOT, config["paths"]["results_dir"])

    print(f"\n{'='*70}")
    print(f"  Instance: {instance_id}")
    print(f"  Difficulty: {TARGET_INSTANCES.get(instance_id, {}).get('difficulty', 'unknown')}")
    print(f"{'='*70}")

    # Build configs for the orchestrator
    agent_config = dict(config["agent"])
    model_config = {k: v for k, v in config["model"].items() if k != "model_class"}
    env_config = build_env_config(config, instance_id)
    branching_config = dict(config["branching"])
    branching_config["results_dir"] = results_dir

    start_time = time.time()

    try:
        orchestrator = PhasedOrchestrator(
            instance_id=instance_id,
            problem_statement=instance["problem_statement"],
            agent_config=agent_config,
            model_config=model_config,
            env_config=env_config,
            branching_config=branching_config,
            nli_model=nli_model,
        )
        results = orchestrator.run()
    except Exception as e:
        logging.error(f"Fatal error for {instance_id}: {e}", exc_info=True)
        results = {
            "instance_id": instance_id,
            "error": str(e),
            "patches": [],
        }
    finally:
        # Ensure all containers are cleaned up
        try:
            orchestrator.manager.cleanup_all()
        except Exception:
            pass

    elapsed = time.time() - start_time

    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Trajectories: {results.get('total_trajectories', 0)}")
    print(f"  Submitted: {results.get('submitted_trajectories', 0)}")
    print(f"  Branching events: {results.get('branching_events', 0)}")
    patches = results.get("patches", [])
    print(f"  Patches: {len(patches)}")
    for p in patches:
        print(f"    {p['trajectory_id']}: {len(p['patch'])} chars, submitted={p['submitted']}")

    # Build prediction(s) for swebench eval
    # Use the best patch: prefer submitted patches, then longest
    predictions = []
    if patches:
        submitted = [p for p in patches if p["submitted"]]
        best = submitted[0] if submitted else max(patches, key=lambda p: len(p["patch"]))
        predictions.append({
            "instance_id": instance_id,
            "model_name_or_path": "qwen3-coder-30b-a3b-awq-branching",
            "model_patch": best["patch"],
        })

        # Also save ALL patches as separate predictions for diversity analysis
        for p in patches:
            predictions.append({
                "instance_id": instance_id,
                "model_name_or_path": f"qwen3-coder-30b-a3b-awq-branching-{p['trajectory_id']}",
                "model_patch": p["patch"],
                "trajectory_id": p["trajectory_id"],
            })
    else:
        predictions.append({
            "instance_id": instance_id,
            "model_name_or_path": "qwen3-coder-30b-a3b-awq-branching",
            "model_patch": "",
        })

    return predictions


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run branching agent on SWE-bench instances")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--instances", nargs="*", default=None,
                        help="Specific instance IDs (default: all 10)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip instances with existing results")
    args = parser.parse_args()

    # Clear any stale cached httpx clients from prior runs
    reset_litellm_clients()

    config = load_config(args.config)
    results_dir = os.path.join(PROJECT_ROOT, config["paths"]["results_dir"])
    setup_logging(results_dir)

    instance_ids = args.instances or TARGET_INSTANCE_IDS
    print(f"Loading SWE-bench instances: {instance_ids}")
    instances = load_swebench_instances(instance_ids=instance_ids)
    print(f"Loaded {len(instances)} instances")

    # Connect to NLI server (DeBERTa runs in a separate process)
    nli_config = config.get("nli", {})
    nli_server_url = nli_config.get("server_url", "http://localhost:8100")
    print(f"Connecting to NLI server at {nli_server_url}...")
    nli_model = NLIClient(server_url=nli_server_url)
    print("Connected to NLI server")

    # Check for existing predictions
    predictions_path = os.path.join(PROJECT_ROOT, config["paths"]["predictions_file"])
    existing_ids = set()
    all_predictions = []
    if args.skip_existing and os.path.exists(predictions_path):
        with open(predictions_path, "r") as f:
            for line in f:
                pred = json.loads(line)
                all_predictions.append(pred)
                existing_ids.add(pred["instance_id"])
        print(f"Found {len(existing_ids)} existing predictions")

    for instance in instances:
        iid = instance["instance_id"]
        if iid in existing_ids:
            print(f"\nSkipping {iid} (already has prediction)")
            continue

        preds = run_single_instance(instance, config, nli_model)

        # Save the primary prediction (first one)
        primary = [p for p in preds if "trajectory_id" not in p]
        all_predictions.extend(primary)

        # Save predictions incrementally
        save_predictions(all_predictions, predictions_path)

        # Also save all trajectory predictions separately
        all_preds_path = predictions_path.replace(".jsonl", "_all_trajectories.jsonl")
        with open(all_preds_path, "a") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  All done! {len(all_predictions)} predictions saved to {predictions_path}")
    print(f"  Run evaluation with: python scripts/run_eval_harness.py --run-id branching_v1")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
