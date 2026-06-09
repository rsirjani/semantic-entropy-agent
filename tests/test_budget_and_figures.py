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
