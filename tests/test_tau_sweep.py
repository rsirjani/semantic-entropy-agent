"""R3.3/R5.5 wiring: post-hoc tau sweep reconstructs the gate from artifacts."""

import json
import math
import os

import tau_sweep as ts


def _write_log(results_dir, iid, entropy, cluster_of_strategy):
    """Write a phased_decisions.log STRATEGY PROPOSAL block in the real format."""
    d = os.path.join(results_dir, iid)
    os.makedirs(d, exist_ok=True)
    n = len(cluster_of_strategy)
    k = len(set(cluster_of_strategy))
    lines = [
        "=" * 70,
        "STRATEGY PROPOSAL",
        f"Proposed: {n} | Clusters: {k} | Unique: {k} | Entropy: {entropy:.3f}",
        "=" * 70,
    ]
    for i, c in enumerate(cluster_of_strategy):
        lines.append(f"  [{i+1}] cluster={c}: strategy text {i}")
    lines.append("")
    lines.append("Unique strategies to fork:")
    with open(os.path.join(d, "phased_decisions.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_eval(eval_dir, iid, resolved_by_tid):
    with open(os.path.join(eval_dir, f"trajectory_eval_{iid}.json"), "w",
              encoding="utf-8") as f:
        json.dump({"instance_id": iid, "trajectories": [
            {"trajectory_id": t, "resolved": r} for t, r in resolved_by_tid.items()
        ]}, f)


def test_partition_entropies_n5_quantization():
    vals = ts.partition_entropies(5)
    # Exactly the 7 partition shapes of 5 -> 7 achievable entropy values.
    expected = [0.0, 0.5004, 0.6730, 0.9503, 1.0549, 1.3322, 1.6094]
    assert len(vals) == 7
    for v, e in zip(vals, expected):
        assert math.isclose(v, e, abs_tol=2e-4)


def test_dominant_trajectory_id_tiebreak_and_sdlg():
    assert ts.dominant_trajectory_id([1, 3, 1]) == "t0_strategy_1"
    assert ts.dominant_trajectory_id([2, 2, 1]) == "t0"      # tie -> lowest index
    assert ts.dominant_trajectory_id(None) == "t0"           # SDLG fallback
    assert ts.dominant_trajectory_id([]) == "t0"


def test_sweep_gates_to_dominant_trajectory(tmp_path):
    rd = str(tmp_path)
    # i1: partition 3+1+1 of 5 -> entropy 0.950; dominant = cluster 0 -> "t0".
    # Dominant t0 FAILS, but branch t0_strategy_2 PASSES: at low tau (branch)
    # the instance passes; past tau=0.950 the gate keeps only t0 -> miss.
    _write_log(rd, "i1", 0.950, [0, 0, 0, 1, 2])
    _write_eval(rd, "i1", {"t0": False, "t0_strategy_1": False,
                           "t0_strategy_2": True})
    report = ts.sweep(rd, rd, taus=[0.0, 0.950, 1.5])
    rows = {r["tau"]: r for r in report["sweep"]}
    assert rows[0.0]["branch_rate"] == 1.0
    assert rows[0.0]["gated_pass_rate"] == 1.0          # superset contains the fix
    # Gate uses strict entropy > tau, so tau=0.950 already declines to branch.
    assert rows[0.950]["branch_rate"] == 0.0
    assert rows[0.950]["gated_pass_rate"] == 0.0        # dominant t0 missed it
    assert rows[0.950]["mean_trajectories_used"] == 1.0
    assert rows[1.5]["gated_pass_rate"] == 0.0


def test_sweep_dominant_in_nonzero_cluster(tmp_path):
    rd = str(tmp_path)
    # Dominant cluster is index 1 -> gated trajectory "t0_strategy_1", which PASSES.
    _write_log(rd, "i1", 0.673, [0, 1, 1, 1, 0])  # sizes [2,3] -> entropy 0.673
    _write_eval(rd, "i1", {"t0": False, "t0_strategy_1": True})
    report = ts.sweep(rd, rd, taus=[1.0])
    row = report["sweep"][0]
    assert row["branch_rate"] == 0.0
    assert row["gated_pass_rate"] == 1.0
    assert row["missing_gated_trajectories"] == []


def test_sweep_sdlg_fallback_uses_branching_log(tmp_path):
    rd = str(tmp_path)
    d = os.path.join(rd, "i1")
    os.makedirs(d)
    with open(os.path.join(d, "branching_log.json"), "w", encoding="utf-8") as f:
        json.dump([{"entropy": 1.05, "step": 12}], f)
    _write_eval(rd, "i1", {"t0": True, "t0_sdlg_1": False})
    report = ts.sweep(rd, rd, taus=[0.5, 1.2])
    rows = {r["tau"]: r for r in report["sweep"]}
    assert report["per_instance"]["i1"]["arm"] == "sdlg"
    assert rows[0.5]["branch_rate"] == 1.0              # 1.05 > 0.5 -> branch
    assert rows[1.2]["branch_rate"] == 0.0              # no-branch keeps t0
    assert rows[1.2]["gated_pass_rate"] == 1.0


def test_sweep_default_grid_is_achievable_set(tmp_path):
    rd = str(tmp_path)
    _write_log(rd, "i1", 1.609, [0, 1, 2, 3, 4])
    _write_eval(rd, "i1", {"t0": True})
    report = ts.sweep(rd, rd, taus=None)
    assert report["n_candidates"] == 5
    assert [r["tau"] for r in report["sweep"]] == report["achievable_entropies"]
    assert len(report["achievable_entropies"]) == 7
