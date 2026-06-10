"""R6.3 budget-audit + R7.3 figure-regeneration wiring (synthetic artifacts, no GPU)."""

import json
import os

import budget_audit as ba
import make_figures as mf


def _write_metadata(d, iid, patches, total_steps):
    inst = os.path.join(d, iid)
    os.makedirs(inst, exist_ok=True)
    with open(os.path.join(inst, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"instance_id": iid, "total_steps": total_steps,
                   "elapsed_seconds": 12.5, "patches": patches}, f)


def _write_eval(d, iid, resolved_by_tid):
    with open(os.path.join(d, f"trajectory_eval_{iid}.json"), "w", encoding="utf-8") as f:
        json.dump({"instance_id": iid,
                   "trajectories": [{"trajectory_id": t, "resolved": r}
                                    for t, r in resolved_by_tid.items()]}, f)


def _write_traj(d, iid, tid, usages):
    """usages: list of (prompt, completion) per assistant call (None -> injected, no usage)."""
    inst = os.path.join(d, iid)
    os.makedirs(inst, exist_ok=True)
    messages = []
    for u in usages:
        if u is None:
            messages.append({"role": "assistant", "content": "x",
                             "extra": {"injected": True}})
        else:
            pt, ct = u
            messages.append({"role": "assistant", "content": "x", "extra": {"response": {
                "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                          "total_tokens": pt + ct}}}})
    with open(os.path.join(inst, f"trajectory_{tid}.traj.json"), "w", encoding="utf-8") as f:
        json.dump({"messages": messages, "trajectory_format": "v1"}, f)


def test_budget_audit_flags_passing_over_cap(tmp_path):
    d = str(tmp_path)
    _write_metadata(d, "i1", [
        {"trajectory_id": "primary", "patch": "x", "submitted": True, "steps": 999},  # ignored
        {"trajectory_id": "t0", "patch": "x", "submitted": True, "steps": 40},   # passes, under cap
        {"trajectory_id": "t1", "patch": "y", "submitted": True, "steps": 280},  # passes, OVER cap
        {"trajectory_id": "t2", "patch": "z", "submitted": False, "steps": 270}, # fails -> not counted
    ], total_steps=350)
    _write_eval(d, "i1", {"primary": True, "t0": True, "t1": True, "t2": False})
    rep = ba.audit(d, d, reference_cap=250)
    assert rep["n_instances"] == 1
    # all = t0,t1,t2 (primary excluded); passing = t0,t1
    assert rep["steps_all_trajectories"]["n"] == 3
    assert rep["steps_passing_trajectories"]["n"] == 2
    over = rep["passing_branches_over_reference_cap"]
    assert len(over) == 1 and over[0]["trajectory_id"] == "t1"


def test_budget_audit_clean_when_under_cap(tmp_path):
    d = str(tmp_path)
    _write_metadata(d, "i1", [
        {"trajectory_id": "t0", "patch": "x", "submitted": True, "steps": 30},
    ], total_steps=30)
    _write_eval(d, "i1", {"t0": True})
    rep = ba.audit(d, d, reference_cap=250)
    assert rep["passing_branches_over_reference_cap"] == []


def test_budget_audit_sums_tokens_from_transcripts(tmp_path):
    d = str(tmp_path)
    _write_metadata(d, "i1", [
        {"trajectory_id": "t0", "patch": "x", "submitted": True, "steps": 40},
        {"trajectory_id": "t1", "patch": "y", "submitted": True, "steps": 50},
        {"trajectory_id": "primary", "patch": "x", "submitted": True, "steps": 40},  # ignored
    ], total_steps=90)
    _write_eval(d, "i1", {"t0": True, "t1": False, "primary": True})
    # t0: two real calls (100+10, 200+20) + one injected (no usage) -> 330 total tokens
    _write_traj(d, "i1", "t0", [(100, 10), (200, 20), None])
    # t1: one real call (50+5) -> 55 total tokens
    _write_traj(d, "i1", "t1", [(50, 5)])
    _write_traj(d, "i1", "primary", [(999, 999)])  # must be excluded
    rep = ba.audit(d, d, reference_cap=250)
    tok = rep["tokens_arm_total"]
    assert tok["prompt_tokens"] == 350 and tok["completion_tokens"] == 35
    assert tok["total_tokens"] == 385
    assert tok["n_trajectories_with_tokens"] == 2  # primary excluded
    # per-trajectory distribution: t0=330, t1=55
    assert rep["tokens_per_trajectory"]["n"] == 2
    assert rep["tokens_per_trajectory"]["max"] == 330.0
    # only t0 passes -> passing token dist has n=1, value 330
    assert rep["tokens_passing_trajectories"]["n"] == 1
    assert rep["tokens_passing_trajectories"]["max"] == 330.0


def _write_control_run(d, iid, run_name, steps, usages, inner_iid=None,
                       write_metadata=True):
    """Control layout: <d>/<iid>/<run_name>/<iid>/{metadata.json, trajectory_t0.traj.json}."""
    inner = os.path.join(d, iid, run_name, inner_iid or iid)
    os.makedirs(inner, exist_ok=True)
    if write_metadata:
        with open(os.path.join(inner, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"instance_id": iid, "total_steps": steps,
                       "elapsed_seconds": 10.0,
                       "patches": [{"trajectory_id": "t0", "patch": "x",
                                    "submitted": True, "steps": steps}]}, f)
    messages = []
    for pt, ct in usages:
        messages.append({"role": "assistant", "content": "x", "extra": {"response": {
            "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                      "total_tokens": pt + ct}}}})
    with open(os.path.join(inner, "trajectory_t0.traj.json"), "w", encoding="utf-8") as f:
        json.dump({"messages": messages, "trajectory_format": "v1"}, f)


def test_budget_audit_reads_control_resample_layout(tmp_path):
    """R6.3: per-ARM accounting must be computable on the vanilla arm's layout.

    The resample arm nests each draw's orchestrator output at
    <dir>/<iid>/run<idx>/<iid>/...; its predictions/eval tid is "run<idx>".
    The audit must map run dirs to those tids so steps/tokens join to
    `resolved` — otherwise the fairness note's cross-arm token comparison is
    unverifiable on the control.
    """
    d = str(tmp_path)
    _write_control_run(d, "i1", "run0", steps=30, usages=[(100, 10)])
    _write_control_run(d, "i1", "run1", steps=300, usages=[(200, 20), (50, 5)])
    # run2 crashed before the orchestrator wrote metadata: still a draw in the
    # eval record (empty patch), but contributes no steps/tokens here.
    _write_control_run(d, "i1", "run2", steps=0, usages=[], write_metadata=False)
    _write_eval(d, "i1", {"run0": False, "run1": True, "run2": False})

    rep = ba.audit(d, d, reference_cap=250)
    assert rep["layout"] == "control"
    assert rep["n_instances"] == 1
    assert rep["n_draws_missing_metadata"] == 1
    # steps: run0=30, run1=300 (run2 missing)
    assert rep["steps_all_trajectories"]["n"] == 2
    assert rep["steps_all_trajectories"]["max"] == 300.0
    # run1 passes AND exceeds the 250-step reference cap
    assert rep["steps_passing_trajectories"]["n"] == 1
    over = rep["passing_branches_over_reference_cap"]
    assert len(over) == 1 and over[0]["trajectory_id"] == "run1"
    # tokens: run0=110, run1=275 -> arm total 385
    tok = rep["tokens_arm_total"]
    assert tok["total_tokens"] == 385 and tok["n_trajectories_with_tokens"] == 2
    assert rep["tokens_passing_trajectories"]["max"] == 275.0
    # per-instance totals are summed across the instance's runs
    assert rep["total_steps_per_instance"]["max"] == 330.0


def test_budget_audit_treatment_layout_detected(tmp_path):
    d = str(tmp_path)
    _write_metadata(d, "i1", [{"trajectory_id": "t0", "patch": "x",
                               "submitted": True, "steps": 30}], total_steps=30)
    _write_eval(d, "i1", {"t0": True})
    rep = ba.audit(d, d, reference_cap=250)
    assert rep["layout"] == "treatment"
    assert rep["n_draws_missing_metadata"] == 0


def test_make_figures_renders_pngs(tmp_path):
    report = {
        "treatment": {"summary": {
            "n_instances": 10,
            "diverse_pass_at_k": {"mean": 0.6, "ci95": [0.4, 0.8]},
            "distinct_patches": {"mean": 3.2, "ci95": [2.1, 4.0]},
            "mean_pairwise_distance": {"mean": 0.5, "ci95": [0.35, 0.65]},
        }},
        "vanilla": {"summary": {
            "n_instances": 10,
            "diverse_pass_at_k": {"mean": 0.4, "ci95": [0.2, 0.6]},
            "distinct_patches": {"mean": 1.4, "ci95": [1.0, 2.0]},
            "mean_pairwise_distance": {"mean": 0.2, "ci95": [0.1, 0.35]},
        }},
        "comparison": {"diverse_pass_at_k_gain": {"mean": 0.2}},  # must be ignored
    }
    out = mf.make_figures(report, str(tmp_path / "figs"))
    assert len(out) == 3
    for p in out:
        assert os.path.isfile(p) and os.path.getsize(p) > 0
