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

# Friendly dataset aliases → HuggingFace dataset names.
DATASET_ALIASES = {
    "verified": "SWE-bench/SWE-bench_Verified",
    "lite": "SWE-bench/SWE-bench_Lite",
    "full": "SWE-bench/SWE-bench",
}


def resolve_dataset_name(dataset: str) -> str:
    """Map a friendly alias (verified/lite/full) to its HF dataset name.

    Passing an already-qualified ``org/name`` string returns it unchanged.
    """
    return DATASET_ALIASES.get(dataset.lower(), dataset)


def load_swebench_instances(
    dataset_name: str = "SWE-bench/SWE-bench_Verified",
    split: str = "test",
    instance_ids: list[str] | None = None,
    all_instances: bool = False,
) -> list[dict]:
    """Load SWE-bench instances using the datasets library.

    Args:
        dataset_name: HF dataset name, or a friendly alias (verified/lite/full).
        split: dataset split (usually "test").
        instance_ids: explicit instance IDs to load. If None and
            ``all_instances`` is False, defaults to the 10 proposal targets.
        all_instances: if True, load EVERY instance in the split (ignores
            instance_ids). Use for full SWE-bench Lite/Verified runs.

    Returns a list of instance dicts with keys like:
    instance_id, repo, base_commit, problem_statement, hints_text, patch, test_patch, etc.
    """
    from datasets import load_dataset

    dataset = load_dataset(resolve_dataset_name(dataset_name), split=split)

    if all_instances:
        # Preserve the dataset's native order.
        return [dict(row) for row in dataset]

    if instance_ids is None:
        instance_ids = TARGET_INSTANCE_IDS

    wanted = set(instance_ids)
    instances = [dict(row) for row in dataset if row["instance_id"] in wanted]

    # Sort by the requested order (unknown IDs sink to the end).
    id_order = {iid: i for i, iid in enumerate(instance_ids)}
    instances.sort(key=lambda x: id_order.get(x["instance_id"], 999))

    return instances


def save_predictions(predictions: list[dict], output_path: str) -> None:
    """Save predictions in SWE-bench JSONL format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
