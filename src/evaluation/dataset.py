"""Load SWE-bench Verified instances."""

import json
import os

# The 10 target instances from the proposal
TARGET_INSTANCES = {
    "sympy__sympy-12481": {"difficulty": "<15 min"},
    "sympy__sympy-16766": {"difficulty": "<15 min"},
    "sympy__sympy-18189": {"difficulty": "<15 min"},
    "sympy__sympy-12096": {"difficulty": "<15 min"},
    "sympy__sympy-15345": {"difficulty": "<15 min"},
    "sympy__sympy-23534": {"difficulty": "<15 min"},
    "sympy__sympy-22714": {"difficulty": "<15 min"},
    "sympy__sympy-19637": {"difficulty": "<15 min"},
    "sympy__sympy-18763": {"difficulty": "<15 min"},
    "sympy__sympy-19495": {"difficulty": "<15 min"},
}

TARGET_INSTANCE_IDS = list(TARGET_INSTANCES.keys())


def load_swebench_instances(
    dataset_name: str = "SWE-bench/SWE-bench_Verified",
    split: str = "test",
    instance_ids: list[str] | None = None,
) -> list[dict]:
    """Load SWE-bench instances using the datasets library.

    Returns a list of instance dicts with keys like:
    instance_id, repo, base_commit, problem_statement, hints_text, patch, test_patch, etc.
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)

    if instance_ids is None:
        instance_ids = TARGET_INSTANCE_IDS

    instances = []
    for row in dataset:
        if row["instance_id"] in instance_ids:
            instances.append(dict(row))

    # Sort by our target order
    id_order = {iid: i for i, iid in enumerate(instance_ids)}
    instances.sort(key=lambda x: id_order.get(x["instance_id"], 999))

    return instances


def save_predictions(predictions: list[dict], output_path: str) -> None:
    """Save predictions in SWE-bench JSONL format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
