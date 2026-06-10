"""Mocked end-to-end artifact-chain test (R8.5; queued by scrutiny iteration 6)
plus the iteration-7 draw-accounting-at-fork contract.

Chain under test, with every heavy dependency replaced by its artifact:
synthetic trajectories -> phased_orchestrator.collect_patch_entries ->
run_branching.build_predictions -> predictions JSONL on disk ->
eval_all_trajectories loaders (load_latest_trajectories, deduplicate_patches,
propagate_duplicate_results) -> trajectory_eval JSON on disk ->
compute_metrics loaders -> per_instance_table -> compare. No GPU/Docker/NLI.

Iteration-7 contract additions (R7.2 'draw accounting starts at the fork
decision'): a trajectory that fails at CREATION is a recorded failed draw; an
SDLG alternative that SUBMITS at injection is a recorded completed draw whose
patch is kept; the resample all-trajectories file replaces per-instance rows on
re-run (it has no batch delimiters for the parsers to split on).
"""

import json
import types

from minisweagent.exceptions import Submitted

from src.agent.phased_orchestrator import PhasedOrchestrator, collect_patch_entries
from src.agent.trajectory import Trajectory


def _traj(tid, status, patch, submitted=False, parent=None):
    return types.SimpleNamespace(
        trajectory_id=tid, status=status, patch=patch, submitted=submitted,
        step=5, parent_id=parent, branch_info=None,
    )


PATCH_A = "diff --git a/x.py b/x.py\n@@ -1 +1 @@\n-old\n+aaa\n"
PATCH_B = "diff --git a/x.py b/x.py\n@@ -1 +1 @@\n-old\n+bbb\n"
PATCH_C = "diff --git a/x.py b/x.py\n@@ -1 +1 @@\n-old\n+ccc\n"
PATCH_D = "diff --git a/x.py b/x.py\n@@ -1 +1 @@\n-old\n+ddd\n"


def _write_eval(eval_dir, iid, latest, passing_patches):
    """Mimic the eval driver: dedup -> evaluate reps -> propagate to all rows."""
    from eval_all_trajectories import deduplicate_patches, propagate_duplicate_results
    evaluated = {}
    for t in deduplicate_patches(latest):
        evaluated[t["model_patch"]] = {
            "trajectory_id": t.get("trajectory_id") or "primary",
            "resolved": t["model_patch"] in passing_patches,
            "patch_len": len(t["model_patch"]),
        }
    rows = propagate_duplicate_results(latest, evaluated)
    path = eval_dir / f"trajectory_eval_{iid}.json"
    path.write_text(json.dumps({"instance_id": iid, "trajectories": rows}),
                    encoding="utf-8")
    return rows


def test_mocked_end_to_end_treatment_vs_vanilla(tmp_path):
    """Full chain: producers -> artifacts -> eval -> metrics, both arms."""
    from run_branching import build_predictions
    from eval_all_trajectories import load_latest_trajectories
    from compute_metrics import compare, load_eval, load_predictions, per_instance_table

    iid = "sympy__sympy-9999"

    # --- Treatment arm: 5 genuine draws (submitted A, duplicate A, distinct B,
    # failed-with-captured-patch C, failed empty) + one interrupted 'active'
    # trajectory that must NOT be counted as a draw.
    trajs = [
        _traj("t0", "completed", PATCH_A, submitted=True),
        _traj("t0_strategy_1", "completed", PATCH_A),
        _traj("t0_strategy_2", "completed", PATCH_B),
        _traj("t0_strategy_3", "failed", PATCH_C),
        _traj("t0_strategy_4", "failed", ""),
        _traj("t9", "active", "INTERRUPTED"),
    ]
    entries = collect_patch_entries(trajs, {"t0": "strategy zero"})
    assert [e["trajectory_id"] for e in entries] == [
        "t0", "t0_strategy_1", "t0_strategy_2", "t0_strategy_3", "t0_strategy_4"]

    treat_dir = tmp_path / "treat"
    treat_dir.mkdir()
    tpred = treat_dir / "predictions_all_trajectories.jsonl"
    with open(tpred, "a", encoding="utf-8") as f:
        for row in build_predictions(iid, entries):
            f.write(json.dumps(row) + "\n")

    latest = load_latest_trajectories(str(tpred), iid)
    assert len(latest) == 6                          # primary + 5 genuine draws
    assert latest[0].get("trajectory_id") is None    # best-of primary first
    assert latest[0]["model_patch"] == PATCH_A       # submitted wins best-of
    _write_eval(treat_dir, iid, latest, passing_patches={PATCH_A})

    # --- Vanilla arm: 5 matched resamples, mode-collapsed (D,D,D,D,empty).
    van_dir = tmp_path / "van"
    van_dir.mkdir()
    vpred = van_dir / "predictions_all_trajectories.jsonl"
    vrows = [{"instance_id": iid, "model_name_or_path": f"resample-run{i}",
              "model_patch": p, "trajectory_id": f"run{i}"}
             for i, p in enumerate([PATCH_D, PATCH_D, PATCH_D, PATCH_D, ""])]
    vpred.write_text("".join(json.dumps(r) + "\n" for r in vrows), encoding="utf-8")
    vlatest = load_latest_trajectories(str(vpred), iid)
    assert len(vlatest) == 5
    _write_eval(van_dir, iid, vlatest, passing_patches=set())

    # --- Metric layer.
    preds_a = load_predictions(str(tpred))
    preds_b = load_predictions(str(vpred))
    evals_a = load_eval(str(treat_dir))
    evals_b = load_eval(str(van_dir))
    assert len(preds_a[iid]) == 5 and len(evals_a[iid]) == 5  # primary dropped
    assert len(preds_b[iid]) == 5 and len(evals_b[iid]) == 5

    table_a = per_instance_table(preds_a, evals_a)
    table_b = per_instance_table(preds_b, evals_b)
    assert table_a[iid]["k"] == 5
    assert table_a[iid]["n_resolved"] == 2           # A submitted + duplicate A
    assert table_a[iid]["distinct_patches"] == 3     # A, B, C
    assert table_a[iid]["n_nonempty_patches"] == 4
    assert table_b[iid]["k"] == 5 and table_b[iid]["n_resolved"] == 0
    assert table_b[iid]["distinct_patches"] == 1     # mode collapse

    comp = compare(table_a, table_b, entropy={}, seed=0, split=None,
                   preds_a=preds_a, preds_b=preds_b)
    assert comp["n_compared_instances"] == 1
    assert comp["k_mismatch_instances"] == []        # matched at k=5, both arms
    assert comp["diverse_pass_at_k_gain"]["mean"] == 1.0
    assert comp["rarefied_distinct_gain"]["mean"] == 2.0   # 3 - 1 at k*=5
    assert comp["nonempty_patch_fraction"]["arm_a"] == 0.8
    assert comp["nonempty_patch_fraction"]["arm_b"] == 0.8


def test_mocked_end_to_end_shrinking_rerun_uses_last_batch(tmp_path):
    """A branching re-run that produced FEWER draws supersedes the first run in
    BOTH consumers (eval loader and metric loader) — no orphan rows."""
    from run_branching import build_predictions
    from eval_all_trajectories import load_latest_trajectories
    from compute_metrics import load_predictions

    iid = "sympy__sympy-8888"
    treat_dir = tmp_path / "treat"
    treat_dir.mkdir()
    tpred = treat_dir / "predictions_all_trajectories.jsonl"

    first = collect_patch_entries([
        _traj("t0", "completed", PATCH_A),
        _traj("t0_strategy_1", "completed", PATCH_B),
        _traj("t0_strategy_2", "completed", PATCH_C),
    ], {})
    second = collect_patch_entries([
        _traj("t0", "completed", PATCH_D),
        _traj("t0_strategy_1", "failed", ""),
    ], {})
    with open(tpred, "a", encoding="utf-8") as f:
        for entries in (first, second):
            for row in build_predictions(iid, entries):
                f.write(json.dumps(row) + "\n")

    latest = load_latest_trajectories(str(tpred), iid)
    assert [t.get("trajectory_id") for t in latest] == [None, "t0", "t0_strategy_1"]
    preds = load_predictions(str(tpred))
    assert preds[iid] == [PATCH_D, ""]               # t0_strategy_2 did not survive


def test_placeholder_failed_draw_is_counted_and_safe(tmp_path):
    """A failed-at-creation placeholder (agent=None, env=None) is a genuine
    draw for collect_patch_entries, and save()/cleanup() are no-op safe."""
    t = Trajectory(trajectory_id="t0_strategy_2", parent_id="t0",
                   status="failed", branch_info={"type": "strategy",
                                                 "creation_failed": True})
    assert t.agent is None and t.env is None
    t.cleanup()                                      # must not raise
    assert t.save(str(tmp_path)) is None             # nothing to serialize
    entries = collect_patch_entries([t], {})
    assert len(entries) == 1
    assert entries[0]["patch"] == "" and entries[0]["status"] == "failed"


def test_register_failed_draw_registers_a_failed_placeholder():
    """_register_failed_draw puts a failed empty-patch draw into the manager so
    total_trajectories and the predictions rows count the failed fork (R7.2)."""
    fake_self = types.SimpleNamespace(
        manager=types.SimpleNamespace(trajectories={}))
    traj = PhasedOrchestrator._register_failed_draw(
        fake_self, "t0_strategy_3", "t0", {"type": "strategy"})
    assert fake_self.manager.trajectories["t0_strategy_3"] is traj
    assert traj.status == "failed" and traj.patch == ""
    assert traj.branch_info["creation_failed"] is True
    entries = collect_patch_entries(fake_self.manager.trajectories.values(), {})
    assert [e["trajectory_id"] for e in entries] == ["t0_strategy_3"]


def test_inject_alternative_classifies_draw_outcomes():
    """Submitted at injection = COMPLETED draw with its patch kept (previously
    swallowed by the clone-failure handler = silent data loss); a generic
    execution error = FAILED draw; normal execution = active."""
    class FakeAgent:
        def __init__(self, exc=None):
            self.exc = exc
            self.added = []
        def inject_and_execute(self, content):
            if self.exc is not None:
                raise self.exc
            return []
        def add_messages(self, *messages):
            self.added.extend(messages)

    ok = FakeAgent()
    assert PhasedOrchestrator._inject_alternative(ok, "x") == ("active", False, "")

    sub = FakeAgent(Submitted({"role": "exit",
                               "extra": {"submission": PATCH_A}}))
    status, submitted, patch = PhasedOrchestrator._inject_alternative(sub, "x")
    assert (status, submitted, patch) == ("completed", True, PATCH_A)
    assert sub.added                                  # exit message recorded

    boom = FakeAgent(RuntimeError("container died"))
    assert PhasedOrchestrator._inject_alternative(boom, "x") == ("failed", False, "")


def test_resample_replace_instance_rows_drops_stale_rows(tmp_path):
    """A resample re-run at smaller k replaces the instance's rows instead of
    appending: the file has no batch delimiters, so stale surplus runN rows
    would otherwise resurrect into the vanilla arm's metric-time k."""
    from compute_metrics import load_predictions_by_tid
    from run_resample_baseline import replace_instance_rows

    path = tmp_path / "predictions_all_trajectories.jsonl"
    rows = [{"instance_id": "iidA", "model_patch": PATCH_D,
             "trajectory_id": f"run{i}"} for i in range(5)]
    rows += [{"instance_id": "iidB", "model_patch": PATCH_B,
              "trajectory_id": f"run{i}"} for i in range(2)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    replace_instance_rows(str(path), "iidA", [
        {"instance_id": "iidA", "model_patch": PATCH_A,
         "trajectory_id": f"run{i}"} for i in range(3)])

    by_tid = load_predictions_by_tid(str(path))
    assert sorted(by_tid["iidA"]) == ["run0", "run1", "run2"]   # k=3, no orphans
    assert all(p == PATCH_A for p in by_tid["iidA"].values())
    assert sorted(by_tid["iidB"]) == ["run0", "run1"]           # untouched
