"""Wrapper around swebench evaluation harness with Windows compatibility."""

import sys
import types
import platform

# Mock the `resource` module on Windows (Unix-only, but swebench imports it)
if platform.system() == "Windows":
    if "resource" not in sys.modules:
        resource_mock = types.ModuleType("resource")
        resource_mock.setrlimit = lambda *args: None
        resource_mock.getrlimit = lambda *args: (1024, 1024)
        resource_mock.RLIMIT_NOFILE = 7
        sys.modules["resource"] = resource_mock

from swebench.harness.run_evaluation import main as swebench_run_eval


def run_evaluation(
    predictions_path: str,
    instance_ids: list[str],
    run_id: str = "baseline_v1",
    max_workers: int = 1,
    timeout: int = 1800,
) -> None:
    """Run SWE-bench evaluation on predictions.

    Args:
        predictions_path: Path to the predictions JSONL file.
        instance_ids: List of instance IDs to evaluate.
        run_id: Identifier for this evaluation run.
        max_workers: Number of parallel workers.
        timeout: Timeout per instance in seconds.
    """
    swebench_run_eval(
        dataset_name="SWE-bench/SWE-bench_Verified",
        split="test",
        instance_ids=instance_ids,
        predictions_path=predictions_path,
        max_workers=max_workers,
        force_rebuild=False,
        cache_level="instance",
        clean=False,
        open_file_limit=4096,
        run_id=run_id,
        timeout=timeout,
        namespace=None,  # Use local image names (sweb.eval.x86_64.*)
        rewrite_reports=False,
        modal=False,
    )
