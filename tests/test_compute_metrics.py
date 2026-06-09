"""R4/R5 wiring: compute_metrics loaders + arm comparison on synthetic artifacts."""

import json
import os

import compute_metrics as cm


def _write_predictions(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_eval(d, instance_id, resolved):
    with open(os.path.join(d, f"trajectory_eval_{instance_id}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "instance_id": instance_id,
            "n_trajectories": len(resolved),
            "trajectories": [{"trajectory_id": f"t{i}", "resolved": r}
                             for i, r in enumerate(resolved)],
        }, f)


def test_load_predictions_drops_primary_duplicate(tmp_path):
    p = tmp_path / "preds.jsonl"
    _write_predictions(p, [
        {"instance_id": "i1", "model_patch": "BEST"},                       # primary, no traj id
        {"instance_id": "i1", "model_patch": "BEST", "trajectory_id": "t0"},
        {"instance_id": "i1", "model_patch": "ALT", "trajectory_id": "t1"},
    ])
    preds = cm.load_predictions(str(p))
    assert preds["i1"] == ["BEST", "ALT"]        # primary dropped, two trajectories kept


def test_load_eval_and_per_instance_table(tmp_path):
    d = tmp_path / "evalA"
    d.mkdir()
    _write_eval(str(d), "i1", [False, True, False])
    p = tmp_path / "predsA.jsonl"
    _write_predictions(p, [
        {"instance_id": "i1", "model_patch": "@@ -1 +1 @@\n+x=1\n", "trajectory_id": "t0"},
        {"instance_id": "i1", "model_patch": "@@ -9 +9 @@\n+x=1\n", "trajectory_id": "t1"},  # dup edit
        {"instance_id": "i1", "model_patch": "@@ -1 +1 @@\n+y=2\n", "trajectory_id": "t2"},
    ])
    table = cm.per_instance_table(cm.load_predictions(str(p)), cm.load_eval(str(d)))
    row = table["i1"]
    assert row["k"] == 3 and row["n_resolved"] == 1
    assert row["diverse_pass_at_k"] == 1.0                  # one resolved at matched k
    assert row["distinct_patches"] == 2                     # t0/t1 same edit, t2 different


def test_compare_gain_and_off_mode_recovery(tmp_path):
    # Arm A (treatment) resolves i1; arm B (vanilla) resolves neither.
    dA, dB = tmp_path / "eA", tmp_path / "eB"
    dA.mkdir(); dB.mkdir()
    _write_eval(str(dA), "i1", [True, False]); _write_eval(str(dA), "i2", [False, False])
    _write_eval(str(dB), "i1", [False, False]); _write_eval(str(dB), "i2", [False, False])
    pA, pB = tmp_path / "pA.jsonl", tmp_path / "pB.jsonl"
    _write_predictions(pA, [{"instance_id": "i1", "model_patch": "+a", "trajectory_id": "t0"},
                            {"instance_id": "i2", "model_patch": "+b", "trajectory_id": "t0"}])
    _write_predictions(pB, [{"instance_id": "i1", "model_patch": "+a", "trajectory_id": "t0"},
                            {"instance_id": "i2", "model_patch": "+b", "trajectory_id": "t0"}])
    ta = cm.per_instance_table(cm.load_predictions(str(pA)), cm.load_eval(str(dA)))
    tb = cm.per_instance_table(cm.load_predictions(str(pB)), cm.load_eval(str(dB)))
    # i1 has LOW entropy -> off-mode recovery (treatment-only pass at low entropy).
    comp = cm.compare(ta, tb, entropy={"i1": 0.1, "i2": 0.1}, seed=0, split=0.5)
    assert comp["diverse_pass_at_k_gain"]["mean"] == 0.5    # i1 gain 1, i2 gain 0
    omr = [o for o in comp["off_mode_recovery_candidates"] if o["low_entropy"]]
    assert [o["instance_id"] for o in omr] == ["i1"]


def test_load_entropy_from_phased_decisions_log(tmp_path):
    iid = "sympy__sympy-1"
    inst_dir = tmp_path / iid
    inst_dir.mkdir()
    (inst_dir / "phased_decisions.log").write_text(
        "STRATEGY PROPOSAL\nProposed: 5 | Clusters: 3 | Unique: 3 | Entropy: 0.901\n",
        encoding="utf-8",
    )
    ent = cm.load_entropy(str(tmp_path), [iid])
    assert abs(ent[iid] - 0.901) < 1e-9
