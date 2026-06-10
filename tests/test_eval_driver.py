"""R8.5 stage (d')/R4.1: the eval driver's pure helpers — every genuine
trajectory must get exactly one eval row (duplicates propagate their
representative's outcome, empty patches count as failed draws), and the
latest-run extraction must be idempotent to re-run appends.

No Docker/swebench needed: eval_all_trajectories defers the harness import to
the actual evaluation call, so these helpers import clean.
"""

import json

from eval_all_trajectories import (
    deduplicate_patches, load_latest_trajectories, propagate_duplicate_results,
)


def _row(iid, tid, patch):
    return {"instance_id": iid, "trajectory_id": tid, "model_patch": patch,
            "model_name_or_path": "m"}


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_propagate_gives_every_trajectory_a_row():
    """Duplicates inherit the representative's outcome; empties are failures.

    This is the R4.1 contract: the metric-time (n, c) must count every genuine
    trajectory. The vanilla arm's duplicate patches ARE the mode-collapse
    signal under study — dropping them from the eval record would deflate its
    k, shrink the matched k*, and subsample the treatment's coverage while the
    vanilla arm keeps plain any-pass.
    """
    trajectories = [
        {"trajectory_id": None, "model_patch": "PATCH_A"},   # best-of primary
        {"trajectory_id": "t0", "model_patch": "PATCH_A"},
        {"trajectory_id": "t1", "model_patch": "PATCH_A"},   # duplicate of t0's patch
        {"trajectory_id": "t2", "model_patch": "PATCH_B"},
        {"trajectory_id": "t3", "model_patch": ""},          # empty -> failed draw
    ]
    evaluated = {
        "PATCH_A": {"trajectory_id": None, "resolved": True, "patch_len": 7},
        "PATCH_B": {"trajectory_id": "t2", "resolved": False, "patch_len": 7},
    }
    rows = propagate_duplicate_results(trajectories, evaluated)
    by_tid = {r["trajectory_id"]: r for r in rows}
    # One row per trajectory, null id normalized to "primary".
    assert sorted(by_tid) == ["primary", "t0", "t1", "t2", "t3"]
    # Duplicates inherit the representative's outcome.
    assert by_tid["t0"]["resolved"] is True and by_tid["t1"]["resolved"] is True
    assert by_tid["t2"]["resolved"] is False
    # Empty patch: never evaluated, unconditionally a failed draw.
    assert by_tid["t3"]["resolved"] is False and by_tid["t3"]["empty_patch"] is True
    # Genuine-trajectory n (primary dropped by the metric layer) is 4 == produced.
    assert sum(1 for t in by_tid if t != "primary") == 4


def test_dedup_keeps_first_nonempty_only():
    trajs = [
        {"trajectory_id": "t0", "model_patch": "A"},
        {"trajectory_id": "t1", "model_patch": ""},
        {"trajectory_id": "t2", "model_patch": "A"},
        {"trajectory_id": "t3", "model_patch": "B"},
    ]
    unique = deduplicate_patches(trajs)
    assert [t["trajectory_id"] for t in unique] == ["t0", "t3"]


def test_load_latest_takes_last_batch_after_branching_rerun(tmp_path):
    """Branching re-runs prepend a new primary row -> new batch; only the
    latest batch is evaluated."""
    p = tmp_path / "preds.jsonl"
    _write(p, [
        _row("i1", None, "OLD_BEST"),
        _row("i1", "t0", "OLD_BEST"),
        _row("i1", None, "NEW_BEST"),          # re-run starts here
        _row("i1", "t0", "NEW_BEST"),
        _row("i1", "t0_strategy_1", "ALT"),
    ])
    latest = load_latest_trajectories(str(p), "i1")
    patches = {t.get("trajectory_id") or "primary": t["model_patch"] for t in latest}
    assert patches == {"primary": "NEW_BEST", "t0": "NEW_BEST", "t0_strategy_1": "ALT"}


def test_load_latest_dedupes_resample_rerun_appends(tmp_path):
    """The resample driver writes NO primary rows, so a re-run without
    --skip-existing appends duplicate (iid, tid) rows into one batch; the
    last occurrence must win (consistent with load_predictions keep-last)."""
    p = tmp_path / "preds.jsonl"
    _write(p, [
        _row("i1", "run0", "OLD"),
        _row("i1", "run1", "B"),
        _row("i1", "run0", "NEW"),             # re-run append
    ])
    latest = load_latest_trajectories(str(p), "i1")
    patches = {t["trajectory_id"]: t["model_patch"] for t in latest}
    assert patches == {"run0": "NEW", "run1": "B"}
