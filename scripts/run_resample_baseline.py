"""Matched-k vanilla resample baseline.

The headline comparison for this project is a DIVERSITY claim: vanilla LLM
resampling mode-collapses to a few semantic forms, whereas semantic branching
spreads across distinct solutions. The fair control is therefore NOT pass@1 — it
is "run the whole trajectory k times, where k = the number of branches the
treatment produced for that instance," at a sampling temperature > 0, and then
compare how many DISTINCT solutions (and whether a passing one) each set
contains at equal trajectory budget.

This driver:
  1. Reads each instance's realized k from a prior treatment run's per-instance
     metadata.json (`total_trajectories`).  -> run branching FIRST, this SECOND.
  2. Runs the SAME phased pipeline with diversity_method="none" (single vanilla
     trajectory, no proposal, no SDLG) k times per instance, with the BASE agent
     sampling at temperature T > 0 (so the k runs actually differ).
  3. Sweeps T (default 0.2/0.7/1.0), writing each temperature to its own results
     dir so the arms never overwrite each other.

Outputs, per temperature, under <results-dir>_t<T>/:
  - predictions.jsonl                  (best-of-k per instance: pass@1-of-vanilla)
  - predictions_all_trajectories.jsonl (all k resamples: diverse-pass@k / diversity)

Usage:
  python scripts/run_resample_baseline.py --treatment-dir results/branching \
      --dataset verified --temperatures 0.2 0.7 1.0
"""

import argparse
import copy
import json
import logging
import os
import sys
import time

os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for run_branching

from src.agent.phased_orchestrator import PhasedOrchestrator
from src.diversity.nli_client import NLIClient
from src.evaluation.dataset import load_swebench_instances

# Reuse the env/image/logging helpers from the branching driver.
from run_branching import (
    build_env_config, load_config, reset_litellm_clients, setup_logging,
)

logger = logging.getLogger(__name__)


def discover_instances_and_k(treatment_dir: str, max_k: int | None) -> dict[str, int]:
    """Map instance_id -> k (number of trajectories the treatment produced).

    k is read from <treatment_dir>/<instance_id>/metadata.json["total_trajectories"].
    Instances without usable metadata are skipped with a warning.
    """
    result: dict[str, int] = {}
    if not os.path.isdir(treatment_dir):
        raise FileNotFoundError(f"Treatment dir not found: {treatment_dir}")

    for name in sorted(os.listdir(treatment_dir)):
        meta_path = os.path.join(treatment_dir, name, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            logger.warning(f"Skipping {name}: unreadable metadata.json ({e})")
            continue
        k = int(meta.get("total_trajectories", 0) or 0)
        if k <= 0:
            logger.warning(f"Skipping {name}: total_trajectories={k}")
            continue
        # Consistency check: on a clean run with the current driver, the
        # per-trajectory patch entries (one per genuine draw, R7.2) equal
        # total_trajectories. A mismatch means an OLD-driver artifact (which
        # dropped failed/patchless draws) or an interrupted run (leftover
        # 'active' trajectories) — either way the treatment's predictions
        # file does not count what total_trajectories counts, and the
        # matched-k comparison would be built on inconsistent denominators.
        n_rows = sum(1 for p in meta.get("patches", []) if p.get("trajectory_id"))
        if n_rows and n_rows != k:
            logger.warning(
                f"{name}: metadata has {n_rows} per-trajectory patch entries but "
                f"total_trajectories={k} — old-driver or interrupted treatment "
                f"artifact; re-run the treatment instance with the current driver "
                f"before using it for the matched-k baseline.")
        if max_k is not None:
            k = min(k, max_k)
        result[name] = k
    return result


def run_one_resample(
    instance: dict,
    config: dict,
    nli_model: NLIClient,
    temperature: float,
    run_results_dir: str,
) -> str:
    """Run ONE vanilla trajectory (diversity_method='none') at `temperature`.

    Returns the trajectory's patch ("" if none). Each call uses a fresh
    container; results land under run_results_dir so concurrent runs of the same
    instance don't collide on disk.
    """
    instance_id = instance["instance_id"]

    agent_config = dict(config["agent"])
    # Base agent must SAMPLE for vanilla diversity — override the model temp
    # (the treatment keeps temp 0 and gets diversity from the proposer instead).
    model_config = {k: v for k, v in config["model"].items() if k != "model_class"}
    model_config = copy.deepcopy(model_config)
    model_config.setdefault("model_kwargs", {})["temperature"] = temperature

    env_config = build_env_config(config, instance_id)

    branching_config = dict(config["branching"])
    branching_config["diversity_method"] = "none"  # vanilla single trajectory
    branching_config["sample_temperature"] = temperature
    branching_config["results_dir"] = run_results_dir

    orchestrator = PhasedOrchestrator(
        instance_id=instance_id,
        problem_statement=instance["problem_statement"],
        agent_config=agent_config,
        model_config=model_config,
        env_config=env_config,
        branching_config=branching_config,
        nli_model=nli_model,
    )
    try:
        results = orchestrator.run()
    finally:
        try:
            orchestrator.manager.cleanup_all()
        except Exception:
            pass

    patches = results.get("patches", [])
    if not patches:
        return ""
    # 'none' arm yields a single trajectory; prefer a submitted patch, else the
    # longest (defensive — there should be exactly one).
    submitted = [p for p in patches if p.get("submitted")]
    best = submitted[0] if submitted else max(patches, key=lambda p: len(p["patch"]))
    return best.get("patch", "")


def run_temperature(
    temperature: float,
    instances_by_id: dict[str, dict],
    k_by_id: dict[str, int],
    config: dict,
    nli_model: NLIClient,
    base_results_dir: str,
    skip_existing: bool,
) -> None:
    """Run the full matched-k baseline for a single temperature."""
    results_dir = f"{base_results_dir}_t{temperature}"
    os.makedirs(results_dir, exist_ok=True)
    primary_path = os.path.join(results_dir, "predictions.jsonl")
    all_path = os.path.join(results_dir, "predictions_all_trajectories.jsonl")

    done_ids: set[str] = set()
    if skip_existing and os.path.exists(primary_path):
        with open(primary_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["instance_id"])
                except Exception:
                    pass

    print(f"\n{'#'*70}\n#  TEMPERATURE {temperature}  ->  {results_dir}\n{'#'*70}")

    for instance_id, k in k_by_id.items():
        if instance_id in done_ids:
            print(f"  Skipping {instance_id} (already done at T={temperature})")
            continue
        instance = instances_by_id.get(instance_id)
        if instance is None:
            logger.warning(f"  {instance_id}: not found in dataset, skipping")
            continue

        print(f"\n=== {instance_id}: {k} vanilla resamples @ T={temperature} ===")
        t0 = time.time()
        run_patches: list[str] = []
        for idx in range(k):
            run_dir = os.path.join(results_dir, instance_id, f"run{idx}")
            try:
                patch = run_one_resample(instance, config, nli_model, temperature, run_dir)
            except Exception as e:
                logger.error(f"  {instance_id} run{idx} failed: {e}", exc_info=True)
                patch = ""
            run_patches.append(patch)
            print(f"    run{idx}: {len(patch)} chars")

        # All resamples -> diversity / coverage file.
        with open(all_path, "a", encoding="utf-8") as f:
            for idx, patch in enumerate(run_patches):
                f.write(json.dumps({
                    "instance_id": instance_id,
                    "model_name_or_path": f"qwen3-coder-resample-t{temperature}-run{idx}",
                    "model_patch": patch,
                    "trajectory_id": f"run{idx}",
                    "temperature": temperature,
                }) + "\n")

        # Best-of-k -> primary file (vanilla pass@1 reference).
        best = max(run_patches, key=len) if run_patches else ""
        primary = []
        if os.path.exists(primary_path):
            with open(primary_path, "r", encoding="utf-8") as f:
                primary = [json.loads(l) for l in f if l.strip()]
        primary = [p for p in primary if p["instance_id"] != instance_id]
        primary.append({
            "instance_id": instance_id,
            "model_name_or_path": f"qwen3-coder-resample-t{temperature}",
            "model_patch": best,
        })
        with open(primary_path, "w", encoding="utf-8") as f:
            for p in primary:
                f.write(json.dumps(p) + "\n")

        print(f"  {instance_id} done in {time.time()-t0:.1f}s "
              f"({sum(1 for p in run_patches if p)}/{k} non-empty)")


def main():
    parser = argparse.ArgumentParser(description="Matched-k vanilla resample baseline")
    parser.add_argument("--config", default=None, help="Path to config YAML (default: branching.yaml)")
    parser.add_argument("--treatment-dir", required=True,
                        help="Treatment results dir to read per-instance k from "
                             "(e.g. results/branching).")
    parser.add_argument("--results-dir", default=os.path.join("results", "resample_baseline"),
                        help="Base output dir; each temperature appends _t<T>.")
    parser.add_argument("--dataset", default="verified",
                        choices=["verified", "lite", "full"],
                        help="Dataset the treatment instances came from.")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.2, 0.7, 1.0],
                        help="Sampling temperatures to sweep (default: 0.2 0.7 1.0).")
    parser.add_argument("--max-k", type=int, default=None,
                        help="Optional cap on resamples per instance (bounds compute).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip instances already present in a temperature's predictions.jsonl")
    args = parser.parse_args()

    reset_litellm_clients()
    config = load_config(args.config)
    base_results_dir = os.path.join(PROJECT_ROOT, args.results_dir)
    setup_logging(base_results_dir)

    treatment_dir = args.treatment_dir
    if not os.path.isabs(treatment_dir):
        treatment_dir = os.path.join(PROJECT_ROOT, treatment_dir)

    k_by_id = discover_instances_and_k(treatment_dir, args.max_k)
    if not k_by_id:
        print(f"No instances with usable metadata found under {treatment_dir}")
        sys.exit(1)
    total_runs = sum(k_by_id.values()) * len(args.temperatures)
    print(f"Discovered {len(k_by_id)} instances from {treatment_dir}")
    print(f"k per instance: {k_by_id}")
    print(f"Total resample runs across {len(args.temperatures)} temperatures: {total_runs}")

    instances = load_swebench_instances(
        dataset_name=args.dataset, instance_ids=list(k_by_id.keys())
    )
    instances_by_id = {inst["instance_id"]: inst for inst in instances}

    nli_config = config.get("nli", {})
    nli_url = nli_config.get("server_url", "http://localhost:8100")
    print(f"Connecting to NLI server at {nli_url}...")
    nli_model = NLIClient(server_url=nli_url)

    for temperature in args.temperatures:
        run_temperature(
            temperature, instances_by_id, k_by_id, config, nli_model,
            base_results_dir, args.skip_existing,
        )

    print(f"\n{'='*70}\n  Matched-k resample baseline complete.\n"
          f"  Compare diverse-pass@k and patch diversity vs {treatment_dir}.\n{'='*70}")


if __name__ == "__main__":
    main()
