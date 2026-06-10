"""R8.5 stage (d')/R4.1: the eval driver's pure helpers — every genuine
trajectory must get exactly one eval row (duplicates propagate their
representative's outcome, empty patches count as failed draws), and the
latest-run extraction must be idempotent to re-run appends.

No Docker/swebench needed: eval_all_trajectories defers the harness import to
the actual evaluation call, so these helpers import clean.
"""

import json
import sys
import types

import pytest

from eval_all_trajectories import (
    EvalOutcomeError, classify_missing_report, deduplicate_patches,
    eval_single_trajectory, load_latest_trajectories, patch_run_id,
    propagate_duplicate_results,
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


# --------------------------------------------------------------------------- #
# Eval-outcome integrity + stale-report immunity (R4.1)
# --------------------------------------------------------------------------- #

def _stub_harness(monkeypatch):
    """Stand in for src.evaluation.run_eval so no Docker/swebench is needed.

    eval_single_trajectory defers `from src.evaluation.run_eval import
    run_evaluation` to call time; planting a stub module makes the call a
    no-op, so the driver's report-reading/classification logic is exercised
    against pre-staged harness log dirs.
    """
    mod = types.ModuleType("src.evaluation.run_eval")
    mod.run_evaluation = lambda **kw: None
    monkeypatch.setitem(sys.modules, "src.evaluation.run_eval", mod)


def _traj(tid, patch, model="m/x"):
    return {"trajectory_id": tid, "model_patch": patch,
            "model_name_or_path": model}


def _log_dir(tmp_path, run_id, iid, model_dir="m__x"):
    d = tmp_path / "logs" / "run_evaluation" / run_id / model_dir / iid
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_patch_run_id_is_content_keyed():
    """The harness reuses an existing report keyed by run_id alone, so the
    run_id must change when the patch content changes (stale-report immunity)
    and stay stable for identical content (cheap retry via cache reuse)."""
    a = patch_run_id("arm", "t0", "PATCH_A")
    assert a == patch_run_id("arm", "t0", "PATCH_A")
    assert a != patch_run_id("arm", "t0", "PATCH_B")
    assert a.startswith("arm_traj_t0_")


def test_report_read_from_model_normalized_path(tmp_path, monkeypatch):
    """A genuine report is read back; the model dir mirrors the harness's
    `/` -> `__` normalization (a raw slash would miss every report and score
    the arm all-failed)."""
    _stub_harness(monkeypatch)
    monkeypatch.chdir(tmp_path)
    run_id = patch_run_id("arm", "t0", "P")
    d = _log_dir(tmp_path, run_id, "i1")
    (d / "report.json").write_text(json.dumps({"i1": {"resolved": True}}))
    res = eval_single_trajectory(_traj("t0", "P"), "i1", run_id, 10,
                                 temp_dir=str(tmp_path))
    assert res["resolved"] is True and "fail_reason" not in res


def test_missing_report_apply_fail_is_a_genuine_failed_draw(tmp_path, monkeypatch):
    _stub_harness(monkeypatch)
    monkeypatch.chdir(tmp_path)
    run_id = patch_run_id("arm", "t1", "BAD")
    d = _log_dir(tmp_path, run_id, "i1")
    (d / "run_instance.log").write_text(">>>>> Patch Apply Failed:\ncorrupt hunk")
    res = eval_single_trajectory(_traj("t1", "BAD"), "i1", run_id, 10,
                                 temp_dir=str(tmp_path))
    assert res["resolved"] is False
    assert res["fail_reason"] == "patch_apply_failed"


def test_missing_report_timeout_is_a_genuine_failed_draw(tmp_path, monkeypatch):
    _stub_harness(monkeypatch)
    monkeypatch.chdir(tmp_path)
    run_id = patch_run_id("arm", "t2", "HANGS")
    d = _log_dir(tmp_path, run_id, "i1")
    (d / "test_output.txt").write_text("...\n\nTimeout error: 1800 seconds exceeded.")
    res = eval_single_trajectory(_traj("t2", "HANGS"), "i1", run_id, 10,
                                 temp_dir=str(tmp_path))
    assert res["resolved"] is False
    assert res["fail_reason"] == "test_timeout"


def test_missing_report_without_markers_raises_not_scores(tmp_path, monkeypatch):
    """An eval-time infra flake must NEVER be recorded as resolved=False —
    the harness swallows Docker/build errors leaving no report and no
    patch-attributable marker; the driver must raise so the record is not
    written (a written record would be frozen by the resume-marker skip)."""
    _stub_harness(monkeypatch)
    monkeypatch.chdir(tmp_path)
    run_id = patch_run_id("arm", "t3", "FINE")
    _log_dir(tmp_path, run_id, "i1")  # log dir exists but is empty
    with pytest.raises(EvalOutcomeError):
        eval_single_trajectory(_traj("t3", "FINE"), "i1", run_id, 10,
                               temp_dir=str(tmp_path))


def test_classify_missing_report_priorities(tmp_path):
    """Apply-fail beats timeout when both markers somehow appear (the apply
    failure happened first — the test script never ran to completion)."""
    d = tmp_path / "ld"
    d.mkdir()
    (d / "run_instance.log").write_text(
        ">>>>> Patch Apply Failed:\nx\nTest timed out after 5 seconds")
    assert classify_missing_report(str(d)) == "patch_apply_failed"
    # Nothing at all -> infra.
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(EvalOutcomeError):
        classify_missing_report(str(empty))


def test_main_exits_3_and_writes_no_record_on_infra_error(tmp_path, monkeypatch):
    """End-to-end driver behavior: on an unclassifiable missing report the
    process exits nonzero (campaign retries once, then stops loudly) and the
    trajectory_eval marker is NOT written (resume must re-evaluate)."""
    import eval_all_trajectories as ead
    _stub_harness(monkeypatch)
    monkeypatch.chdir(tmp_path)
    arm = tmp_path / "armA"
    arm.mkdir()
    (arm / "predictions_all_trajectories.jsonl").write_text(
        json.dumps({"instance_id": "i1", "trajectory_id": "t0",
                    "model_patch": "P", "model_name_or_path": "m"}) + "\n")
    monkeypatch.setattr(sys, "argv", ["eval_all_trajectories.py",
                                      "--results-dir", str(arm),
                                      "--instance", "i1"])
    with pytest.raises(SystemExit) as ei:
        ead.main()
    assert ei.value.code == 3
    assert not (arm / "trajectory_eval_i1.json").exists()
