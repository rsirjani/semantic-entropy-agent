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
    # i1: partition 3+1+1 of 5 -> exact entropy 0.950271; dominant = cluster 0
    # -> "t0". Dominant t0 FAILS, but branch t0_strategy_2 PASSES: at low tau
    # (branch) the instance passes; AT the achievable level (strict >) the gate
    # declines and keeps only t0 -> miss. NOTE: tau must be the EXACT level —
    # the 3-decimal logged value 0.950 sits BELOW the true entropy, so a real
    # run at tau=0.950 would still branch (regression: the sweep used to gate
    # there because it trusted the rounded log).
    tau_level = ts.exact_partition_entropy([3, 1, 1])
    _write_log(rd, "i1", 0.950, [0, 0, 0, 1, 2])
    _write_eval(rd, "i1", {"t0": False, "t0_strategy_1": False,
                           "t0_strategy_2": True})
    report = ts.sweep(rd, rd, taus=[0.0, 0.950, tau_level, 1.5])
    rows = {r["tau"]: r for r in report["sweep"]}
    assert rows[0.0]["branch_rate"] == 1.0
    assert rows[0.0]["gated_pass_rate"] == 1.0          # superset contains the fix
    # Entropy is recomputed exactly from the partition (0.950271 > 0.950), so
    # the rounded-log tau STILL branches — matching a real run.
    assert report["per_instance"]["i1"]["entropy_source"] == "recomputed_from_cluster_sizes"
    assert rows[0.950]["branch_rate"] == 1.0
    # At the exact achievable level the strict > gate declines to branch.
    assert rows[tau_level]["branch_rate"] == 0.0
    assert rows[tau_level]["gated_pass_rate"] == 0.0    # dominant t0 missed it
    assert rows[tau_level]["mean_trajectories_used"] == 1.0
    assert rows[1.5]["gated_pass_rate"] == 0.0


def test_sweep_221_boundary_not_flipped_by_log_rounding(tmp_path):
    rd = str(tmp_path)
    # Partition (2,2,1) of 5: H = 1.054920... The 3-decimal log rounds UP to
    # 1.055 > 1.0549, so a sweep trusting the log would BRANCH at the exact
    # achievable level where a real run gates. The exact recompute fixes this.
    tau_level = ts.exact_partition_entropy([2, 2, 1])
    assert 1.0549 < tau_level < 1.05495
    _write_log(rd, "i1", 1.055, [0, 0, 1, 1, 2])  # logged value rounded UP
    _write_eval(rd, "i1", {"t0": True, "t0_strategy_1": False,
                           "t0_strategy_2": False})
    report = ts.sweep(rd, rd, taus=[tau_level])
    row = report["sweep"][0]
    assert report["per_instance"]["i1"]["entropy_source"] == "recomputed_from_cluster_sizes"
    assert row["branch_rate"] == 0.0                    # gated, like a real run
    assert row["gated_pass_rate"] == 1.0                # tie -> dominant = t0, passes


def test_sweep_keeps_logged_entropy_when_partition_disagrees(tmp_path):
    rd = str(tmp_path)
    # Kernel (von Neumann) entropy is NOT a function of cluster sizes; when the
    # logged value disagrees with the partition entropy beyond log-rounding
    # tolerance, the sweep must keep the logged value and say so.
    _write_log(rd, "i1", 0.800, [0, 0, 0, 1, 1])  # partition entropy would be 0.6730
    _write_eval(rd, "i1", {"t0": True})
    report = ts.sweep(rd, rd, taus=[0.7])
    assert report["per_instance"]["i1"]["entropy"] == 0.8
    assert (report["per_instance"]["i1"]["entropy_source"]
            == "parsed_log_disagrees_with_partition")
    assert report["sweep"][0]["branch_rate"] == 1.0     # 0.8 > 0.7


def test_sweep_reports_non_modal_realized_n(tmp_path):
    rd = str(tmp_path)
    # The proposer can under-deliver (<N strategies); realized N must be
    # reported and non-modal instances flagged (different quantization grid).
    _write_log(rd, "i1", 1.609, [0, 1, 2, 3, 4])        # N=5
    _write_log(rd, "i2", 1.609, [0, 1, 2, 3, 4])        # N=5
    _write_log(rd, "i3", 1.099, [0, 1, 2])              # N=3 (under-delivered)
    for iid in ("i1", "i2", "i3"):
        _write_eval(rd, iid, {"t0": True})
    report = ts.sweep(rd, rd, taus=None)
    assert report["n_candidates"] == 5                  # modal N
    assert report["n_candidates_by_instance"] == {"i1": 5, "i2": 5, "i3": 3}
    assert report["non_modal_n_instances"] == ["i3"]
    assert len(report["achievable_entropies"]) == 7     # grid for the MODAL N


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
