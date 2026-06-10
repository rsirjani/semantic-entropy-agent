"""Evaluate every trajectory patch from a run, writing an arm-isolated eval record.

Metric-correctness contract (R4.1/R7.2 — this file is the producer of the
`trajectory_eval_<instance>.json` artifacts every metric script consumes):

  1. The eval record contains EXACTLY ONE row per genuine trajectory, so the
     metric-time (n, c) of the Chen estimator counts what the arm actually
     produced. Duplicate patches are evaluated once for compute, but every
     duplicate trajectory inherits its representative's outcome (identical
     patches resolve identically; marked `deduped_from`). Empty patches are
     never sent to Docker but count as `resolved: false` draws — a resample
     that produced nothing still spent budget. Dropping duplicates or empties
     would deflate k for whichever arm produced them — for the vanilla arm,
     duplicates ARE the mode-collapse signal under study.
  2. Output is written into the ARM'S OWN results dir (`--results-dir`), never
     a hardcoded path, so evaluating the control can never overwrite the
     treatment's eval files (R2.5 results isolation).
  3. The best-of "primary" prediction row (null trajectory_id) is normalized to
     trajectory_id "primary" so the metric layer's primary-drop rule sees it.
  4. EVAL-OUTCOME INTEGRITY: every `resolved` is a genuine harness verdict.
     The swebench harness swallows ALL per-instance errors (patch-apply
     failure, test timeout, Docker/build flakes) and simply writes no
     report.json. A missing report is therefore classified from the harness's
     own logs: patch-apply failure and test timeout are attributable to the
     PATCH and recorded as failed draws (with `fail_reason`); any other cause
     is an infrastructure error -> the driver exits nonzero WITHOUT writing
     the eval record, so a Docker flake can never be silently scored as a
     test failure (and then frozen forever by the resume-marker skip).
  5. STALE-REPORT IMMUNITY: the harness returns an existing report.json keyed
     by (run_id, model, instance) WITHOUT re-evaluating — patch content is not
     part of its key. Eval run_ids therefore embed a content hash of the patch
     (`patch_run_id`), so a re-run with a changed patch gets a fresh verdict
     while an identical patch legitimately reuses its cached report (which
     also makes the retry after an infra stop cheap).

Usage:
    python scripts/eval_all_trajectories.py --results-dir results/strategy_t0.7 \
        --instance sympy__sympy-12481
    python scripts/eval_all_trajectories.py --results-dir results/resample_t0.7 \
        --instance sympy__sympy-12481
"""

import hashlib
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

# swebench.harness.constants.APPLY_PATCH_FAIL — inlined so the pure helpers
# stay importable without the harness (which needs the Unix `resource` module).
APPLY_PATCH_FAIL_MARKER = ">>>>> Patch Apply Failed"
# Written by the harness on a test-script timeout (run_evaluation.py appends
# "Timeout error: <N> seconds exceeded." to test_output.txt and raises
# EvaluationError "Test timed out after <N> seconds" into run_instance.log).
TIMEOUT_MARKERS = ("Timeout error:", "Test timed out after")


class EvalOutcomeError(RuntimeError):
    """The harness produced no report for a reason NOT attributable to the patch.

    Recording such a trajectory as `resolved: false` would silently corrupt the
    Chen estimator's (n, c) — an eval-time Docker/build flake is a MEASUREMENT
    failure, not a failed draw (the patch exists and has a definite ground-truth
    verdict). The driver must stop loudly instead of writing the record.
    """


def patch_run_id(arm_slug: str, trajectory_id: str, patch: str) -> str:
    """Harness run_id for one patch — keyed by CONTENT, not just trajectory.

    The swebench harness returns an existing report.json for (run_id, model,
    instance) without re-evaluating, and patch content is not part of its key.
    A patch-blind run_id would therefore let a re-run (treatment re-generated,
    same trajectory_id, different patch) silently inherit the previous patch's
    verdict. Embedding a short content hash makes the cache exact: identical
    patch -> legitimate reuse; changed patch -> fresh evaluation.
    """
    digest = hashlib.sha1((patch or "").encode("utf-8")).hexdigest()[:10]
    return f"{arm_slug}_traj_{trajectory_id}_{digest}"


def classify_missing_report(log_dir: str) -> str:
    """Classify a missing report.json from the harness's own logs.

    Returns a genuine-failure reason ("patch_apply_failed" | "test_timeout")
    when the missing report is attributable to the patch itself, else raises
    EvalOutcomeError (infrastructure: Docker/build/container flake, killed
    run, ...). The two genuine cases are exactly the EvaluationErrors the
    harness raises about the PATCH; everything else it swallows is about the
    ENVIRONMENT and must not be scored.
    """
    def read(name: str) -> str:
        fp = os.path.join(log_dir, name)
        if os.path.isfile(fp):
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
            except OSError:
                pass
        return ""

    instance_log = read("run_instance.log")
    test_output = read("test_output.txt")
    if APPLY_PATCH_FAIL_MARKER in instance_log:
        return "patch_apply_failed"
    if any(m in test_output for m in TIMEOUT_MARKERS) or \
       any(m in instance_log for m in TIMEOUT_MARKERS):
        return "test_timeout"
    raise EvalOutcomeError(
        f"no report.json and no patch-attributable failure marker under "
        f"{log_dir} — eval infrastructure error (Docker daemon/image/container "
        f"flake?). The trajectory's outcome is UNKNOWN, not 'failed'; inspect "
        f"run_instance.log there, fix the environment, and re-run this "
        f"instance (already-completed patches reuse their cached reports).")


def load_latest_trajectories(predictions_path: str, instance_id: str) -> list[dict]:
    """Extract the latest run's trajectory predictions for a given instance.

    Branching runs prepend a best-of "primary" row (null trajectory_id) per run,
    so runs split into batches at those rows and the LAST batch is the latest
    run. The resample driver writes no primary rows — a re-run without
    --skip-existing appends duplicate (instance, trajectory_id) rows into one
    batch — so within the final batch we keep the LAST occurrence per
    trajectory_id (consistent with compute_metrics.load_predictions keep-last).
    """
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

    latest = batches[-1]
    by_tid: dict[str, dict] = {}
    for l in latest:
        by_tid[l.get("trajectory_id") or "primary"] = l
    return [by_tid[t] for t in by_tid]  # insertion order, last occurrence wins


def deduplicate_patches(trajectories: list[dict]) -> list[dict]:
    """Unique non-empty patches, keeping the first trajectory bearing each."""
    seen = set()
    unique = []
    for t in trajectories:
        patch = t["model_patch"]
        if patch and patch not in seen:
            seen.add(patch)
            unique.append(t)
    return unique


def propagate_duplicate_results(
    trajectories: list[dict], evaluated: dict[str, dict]
) -> list[dict]:
    """Build a per-trajectory result row for EVERY trajectory.

    Metric correctness (R4.1 matched-k*): the Chen estimator's (n, c) must count
    every genuine trajectory, so duplicates and empty patches may not silently
    vanish from the eval file. Identical patches resolve identically, so each
    duplicate inherits its evaluated representative's outcome (marked with
    `deduped_from`); empty patches are unconditionally `resolved: false`.
    `evaluated` maps patch text -> result row of the representative.
    """
    results = []
    for t in trajectories:
        # Normalize: a missing or null trajectory_id is the best-of "primary"
        # duplicate row — name it "primary" so the metric layer's primary-drop
        # rule sees it (a null id would slip past and inflate n by one).
        tid = t.get("trajectory_id") or "primary"
        patch = t["model_patch"]
        if not patch:
            results.append({"trajectory_id": tid, "resolved": False,
                            "patch_len": 0, "empty_patch": True})
            continue
        rep = evaluated[patch]
        row = {"trajectory_id": tid, "resolved": rep["resolved"],
               "patch_len": len(patch)}
        if "fail_reason" in rep:
            row["fail_reason"] = rep["fail_reason"]
        if rep["trajectory_id"] != tid:
            row["deduped_from"] = rep["trajectory_id"]
        results.append(row)
    return results


def eval_single_trajectory(
    trajectory: dict,
    instance_id: str,
    run_id: str,
    timeout: int,
    temp_dir: str,
) -> dict:
    """Evaluate a single trajectory patch and return the result."""
    # Heavy import deferred so the pure helpers above stay unit-testable
    # without the swebench harness installed (R8.5 stage tests).
    from src.evaluation.run_eval import run_evaluation

    # Write a single-prediction JSONL to a temp file
    pred = {
        "instance_id": instance_id,
        "model_name_or_path": trajectory["model_name_or_path"],
        "model_patch": trajectory["model_patch"],
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, dir=temp_dir
    ) as f:
        f.write(json.dumps(pred) + "\n")
        temp_path = f.name

    tid = trajectory.get("trajectory_id") or "primary"

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
        # (CWD-relative, matching where the swebench harness writes them; the
        # harness normalizes "/" in the model name to "__", so mirror that —
        # otherwise a slash-bearing model id would make every report lookup
        # miss and silently score the whole arm as failed.)
        log_dir = os.path.join(
            "logs", "run_evaluation", run_id,
            trajectory["model_name_or_path"].replace("/", "__"),
            instance_id,
        )
        report_path = os.path.join(log_dir, "report.json")
        if os.path.exists(report_path):
            with open(report_path) as rf:
                report = json.load(rf)
            resolved = report.get(instance_id, {}).get("resolved", False)
            return {"trajectory_id": tid, "resolved": resolved,
                    "patch_len": len(trajectory["model_patch"])}

        # No report: the harness swallowed an error. Genuine (patch-attributable)
        # failures are recorded as failed draws with their reason; anything else
        # raises EvalOutcomeError so an infra flake is never scored as a failure.
        reason = classify_missing_report(log_dir)
        return {"trajectory_id": tid, "resolved": False,
                "patch_len": len(trajectory["model_patch"]),
                "fail_reason": reason}

    finally:
        os.unlink(temp_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate all trajectory patches from a run")
    parser.add_argument("--results-dir", default="results/branching",
                        help="Arm results dir: predictions are read from and the "
                             "trajectory_eval_<instance>.json is written into THIS dir, "
                             "so each arm's evals stay isolated (R2.5).")
    parser.add_argument("--predictions", default=None,
                        help="Override predictions path (default: <results-dir>/predictions_all_trajectories.jsonl)")
    parser.add_argument("--instance", required=True, help="Instance ID to evaluate")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--include-duplicates", action="store_true",
                        help="Force-evaluate duplicate patches too instead of propagating "
                             "the representative's outcome (slower, same numbers)")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(PROJECT_ROOT, results_dir)
    if args.predictions is None:
        predictions_path = os.path.join(results_dir, "predictions_all_trajectories.jsonl")
    else:
        predictions_path = (args.predictions if os.path.isabs(args.predictions)
                            else os.path.join(PROJECT_ROOT, args.predictions))
    if not os.path.exists(predictions_path):
        print(f"ERROR: {predictions_path} not found")
        sys.exit(1)

    trajectories = load_latest_trajectories(predictions_path, args.instance)
    if not trajectories:
        print(f"ERROR: No trajectories found for {args.instance}")
        sys.exit(1)

    print(f"Found {len(trajectories)} trajectories for {args.instance}")

    if args.include_duplicates:
        to_evaluate = [t for t in trajectories if t["model_patch"]]
    else:
        to_evaluate = deduplicate_patches(trajectories)
        n_dupes = sum(1 for t in trajectories if t["model_patch"]) - len(to_evaluate)
        if n_dupes > 0:
            print(f"Evaluating {len(to_evaluate)} unique patches; {n_dupes} duplicates "
                  f"inherit their representative's outcome (metric n stays exact)")

    for t in to_evaluate:
        tid = t.get("trajectory_id") or "primary"
        print(f"  {tid:30s} {len(t['model_patch']):5d} chars")

    # Evaluate each unique patch; run_id is arm-scoped (no cross-arm log
    # collisions) AND patch-content-keyed (no stale-report reuse after a
    # re-run changes a trajectory's patch — see patch_run_id).
    arm_slug = os.path.basename(os.path.normpath(results_dir))
    evaluated: dict[str, dict] = {}
    for t in to_evaluate:
        tid = t.get("trajectory_id") or "primary"
        run_id = patch_run_id(arm_slug, tid, t["model_patch"])
        try:
            result = eval_single_trajectory(t, args.instance, run_id,
                                            args.timeout, temp_dir=results_dir)
        except EvalOutcomeError as e:
            # Infrastructure failure: the outcome is UNKNOWN. Do NOT write the
            # eval record (a written record would be skipped forever by the
            # resume marker with a fabricated `resolved: false` inside) — exit
            # nonzero so the campaign retries once and then stops loudly.
            print(f"\nEVAL INFRASTRUCTURE ERROR on {args.instance} / {tid}:\n  {e}")
            print("  trajectory_eval record NOT written; re-run after fixing "
                  "the environment (completed patches reuse cached reports).")
            sys.exit(3)
        evaluated[t["model_patch"]] = result
        status = "PASS" if result["resolved"] else "FAIL"
        if result.get("fail_reason"):
            status += f" ({result['fail_reason']})"
        print(f"  → {tid}: {status}")

    # Every genuine trajectory gets a result row (duplicates propagate, empty
    # patches count as failures) so the eval file's n matches the predictions.
    results = propagate_duplicate_results(trajectories, evaluated)

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

    # pass@1 is the GREEDY trajectory (t0/run0), not the best-of "primary" row.
    by_tid = {r["trajectory_id"]: r for r in results}
    greedy = by_tid.get("t0") or by_tid.get("run0")
    greedy_resolved = bool(greedy["resolved"]) if greedy else None
    print(f"{'─'*60}")
    print(f"  pass@1 (greedy t0): "
          f"{'PASS' if greedy_resolved else 'FAIL' if greedy_resolved is not None else 'n/a'}")
    print(f"  diverse-pass@1:    {'PASS' if n_pass > 0 else 'FAIL'} ({n_pass}/{len(results)} trajectories)")
    print(f"{'='*60}")

    # Save results into the arm's own dir (R2.5 isolation)
    results_path = os.path.join(
        results_dir, f"trajectory_eval_{args.instance}.json"
    )
    with open(results_path, "w") as f:
        json.dump({
            "instance_id": args.instance,
            "n_trajectories": len(results),
            "n_resolved": n_pass,
            "pass_at_1": greedy_resolved,
            "diverse_pass_at_1": n_pass > 0,
            "trajectories": results,
        }, f, indent=2)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
