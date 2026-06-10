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


def test_load_eval_skips_primary_duplicate(tmp_path):
    """The best-of 'primary' row is excluded so matched-k n counts only genuine
    trajectories (consistent with load_predictions)."""
    d = tmp_path / "evalP"
    d.mkdir()
    with open(os.path.join(str(d), "trajectory_eval_i1.json"), "w", encoding="utf-8") as f:
        json.dump({"instance_id": "i1", "trajectories": [
            {"trajectory_id": "primary", "resolved": True},   # duplicate, must be dropped
            {"trajectory_id": "t0", "resolved": True},
            {"trajectory_id": "t1", "resolved": False},
        ]}, f)
    resolved = cm.load_eval(str(d))["i1"]
    assert resolved == [True, False]   # primary dropped -> n=2, not 3


def test_set_valued_evidence_joins_by_trajectory_id(tmp_path):
    """R5.3: >=2 DISTINCT patches that BOTH pass, joined by trajectory_id (not index)."""
    d = tmp_path / "evalS"
    d.mkdir()
    with open(os.path.join(str(d), "trajectory_eval_i1.json"), "w", encoding="utf-8") as f:
        json.dump({"instance_id": "i1", "trajectories": [
            {"trajectory_id": "primary", "resolved": True},
            {"trajectory_id": "t0", "resolved": True},
            {"trajectory_id": "t1", "resolved": True},
            {"trajectory_id": "t2", "resolved": False},
        ]}, f)
    p = tmp_path / "predsS.jsonl"
    # Deliberately write rows in a DIFFERENT order than the eval to prove the join
    # is by id, not position. t0 and t1 are structurally distinct passing patches.
    _write_predictions(p, [
        {"instance_id": "i1", "model_patch": "BEST", "trajectory_id": "primary"},
        {"instance_id": "i1", "model_patch": "@@ -1 +1 @@\n+y=2\n", "trajectory_id": "t1"},
        {"instance_id": "i1", "model_patch": "@@ -3 +3 @@\n+z=9\n", "trajectory_id": "t2"},  # fails
        {"instance_id": "i1", "model_patch": "@@ -1 +1 @@\n+x=1\n", "trajectory_id": "t0"},
    ])
    sv = cm.set_valued_evidence(str(p), str(d))
    assert len(sv) == 1 and sv[0]["instance_id"] == "i1"
    assert sv[0]["n_distinct_passing_patches"] == 2   # t0,t1 distinct & passing; t2 fails


def test_set_valued_evidence_none_when_single_passing(tmp_path):
    d = tmp_path / "evalS2"
    d.mkdir()
    _write_eval(str(d), "i1", [True, False])  # only one passes
    p = tmp_path / "predsS2.jsonl"
    _write_predictions(p, [
        {"instance_id": "i1", "model_patch": "@@ -1 +1 @@\n+x=1\n", "trajectory_id": "t0"},
        {"instance_id": "i1", "model_patch": "@@ -1 +1 @@\n+y=2\n", "trajectory_id": "t1"},
    ])
    assert cm.set_valued_evidence(str(p), str(d)) == []


def test_compare_enforces_matched_k_at_metric_time(tmp_path):
    """A k mismatch (e.g. a failed resample) must be compared at k*=min(k_a,k_b)
    via the Chen estimator, not silently at each arm's own k."""
    dA, dB = tmp_path / "ekA", tmp_path / "ekB"
    dA.mkdir(); dB.mkdir()
    # Treatment: 4 trajectories, 1 passes. Vanilla: only 2 completed, 1 passes.
    _write_eval(str(dA), "i1", [True, False, False, False])
    _write_eval(str(dB), "i1", [True, False])
    pA, pB = tmp_path / "pkA.jsonl", tmp_path / "pkB.jsonl"
    _write_predictions(pA, [{"instance_id": "i1", "model_patch": f"+a{i}",
                             "trajectory_id": f"t{i}"} for i in range(4)])
    _write_predictions(pB, [{"instance_id": "i1", "model_patch": f"+b{i}",
                             "trajectory_id": f"t{i}"} for i in range(2)])
    ta = cm.per_instance_table(cm.load_predictions(str(pA)), cm.load_eval(str(dA)))
    tb = cm.per_instance_table(cm.load_predictions(str(pB)), cm.load_eval(str(dB)))
    comp = cm.compare(ta, tb, entropy={}, seed=0, split=None)
    # k* = 2: treatment pass@2(n=4,c=1) = 1 - C(3,2)/C(4,2) = 0.5; vanilla
    # pass@2(n=2,c=1) = 1.0. Naive own-k comparison would say 1.0 - 1.0 = 0.
    assert abs(comp["diverse_pass_at_k_gain"]["mean"] - (-0.5)) < 1e-9
    mm = comp["k_mismatch_instances"]
    assert len(mm) == 1 and mm[0]["compared_at_k"] == 2 and not mm[0]["skipped"]


def test_compare_reports_sign_flip_p_and_rarefied_distinct(tmp_path):
    dA, dB = tmp_path / "erA", tmp_path / "erB"
    dA.mkdir(); dB.mkdir()
    for iid in ("i1", "i2"):
        _write_eval(str(dA), iid, [True, False])
        _write_eval(str(dB), iid, [False, False])
    pA, pB = tmp_path / "prA.jsonl", tmp_path / "prB.jsonl"
    # Treatment: 2 distinct patches/instance; vanilla: collapsed (identical twice).
    _write_predictions(pA, [
        {"instance_id": i, "model_patch": f"@@ -1 +1 @@\n+{i}_v{j}=1\n",
         "trajectory_id": f"t{j}"} for i in ("i1", "i2") for j in range(2)])
    _write_predictions(pB, [
        {"instance_id": i, "model_patch": "@@ -1 +1 @@\n+same=1\n",
         "trajectory_id": f"t{j}"} for i in ("i1", "i2") for j in range(2)])
    preds_a, preds_b = cm.load_predictions(str(pA)), cm.load_predictions(str(pB))
    ta = cm.per_instance_table(preds_a, cm.load_eval(str(dA)))
    tb = cm.per_instance_table(preds_b, cm.load_eval(str(dB)))
    comp = cm.compare(ta, tb, entropy={}, seed=0, split=None,
                      preds_a=preds_a, preds_b=preds_b)
    # Two paired gains of +1 -> exact sign-flip p = 2/4 = 0.5.
    assert abs(comp["paired_sign_flip_p"] - 0.5) < 1e-9
    # Power floor: no zero gains among n=2 -> min achievable p = 2^(1-2) = 0.5.
    assert abs(comp["min_achievable_p"] - 0.5) < 1e-9
    # Rarefied distinct gain at k*=2: treatment 2 distinct, vanilla 1 -> +1.
    assert abs(comp["rarefied_distinct_gain"]["mean"] - 1.0) < 1e-9
    # H1 (diversity) endpoint carries its own exact test + power floor + levels.
    assert abs(comp["rarefied_distinct_gain"]["paired_sign_flip_p"] - 0.5) < 1e-9
    assert abs(comp["rarefied_distinct_gain"]["min_achievable_p"] - 0.5) < 1e-9
    assert abs(comp["rarefied_distinct_at_k_star"]["arm_a"]["mean"] - 2.0) < 1e-9
    assert abs(comp["rarefied_distinct_at_k_star"]["arm_b"]["mean"] - 1.0) < 1e-9


def test_load_predictions_dedupes_rerun_appends(tmp_path):
    """Re-running the resample driver appends duplicate (iid, tid) rows; the
    loader must keep the LAST occurrence, never inflate n."""
    p = tmp_path / "dup.jsonl"
    _write_predictions(p, [
        {"instance_id": "i1", "model_patch": "OLD", "trajectory_id": "run0"},
        {"instance_id": "i1", "model_patch": "B", "trajectory_id": "run1"},
        {"instance_id": "i1", "model_patch": "NEW", "trajectory_id": "run0"},  # re-run
    ])
    preds = cm.load_predictions(str(p))
    assert sorted(preds["i1"]) == ["B", "NEW"]   # 2 trajectories, last run0 wins


def test_load_predictions_drops_orphan_tids_from_prior_branching_run(tmp_path):
    """A branching re-run that produced FEWER trajectories (fewer clusters)
    must not leave the old run's orphan tids in the diversity pool: the eval
    driver scores only the LAST primary-delimited batch, so the predictions
    loader must do the same — keep-last-per-tid alone would keep the stale
    t0_strategy_2 patch, inflating n, the rarefaction denominator, and the
    pairwise-distance set with a patch the eval record never scores."""
    p = tmp_path / "orphan.jsonl"
    _write_predictions(p, [
        {"instance_id": "i1", "model_patch": "OLD_BEST"},                     # run 1 primary
        {"instance_id": "i1", "model_patch": "OLD_A", "trajectory_id": "t0"},
        {"instance_id": "i1", "model_patch": "OLD_B", "trajectory_id": "t0_strategy_1"},
        {"instance_id": "i1", "model_patch": "OLD_C", "trajectory_id": "t0_strategy_2"},
        {"instance_id": "i1", "model_patch": "NEW_BEST"},                     # run 2 primary
        {"instance_id": "i1", "model_patch": "NEW_A", "trajectory_id": "t0"},
        {"instance_id": "i1", "model_patch": "NEW_B", "trajectory_id": "t0_strategy_1"},
    ])
    by_tid = cm.load_predictions_by_tid(str(p))
    assert by_tid["i1"] == {"t0": "NEW_A", "t0_strategy_1": "NEW_B"}
    assert sorted(cm.load_predictions(str(p))["i1"]) == ["NEW_A", "NEW_B"]


def test_selected_pass_at_1_flags_degenerate_tiebreak(tmp_path):
    """On the branching arm all signatures are typically unique by construction;
    the selector must DISCLOSE that 'majority' was a pure tie-break there."""
    d = tmp_path / "edeg"
    d.mkdir()
    with open(os.path.join(str(d), "trajectory_eval_i1.json"), "w", encoding="utf-8") as f:
        json.dump({"instance_id": "i1", "trajectories": [
            {"trajectory_id": "t0", "resolved": True},
            {"trajectory_id": "t1", "resolved": False},
        ]}, f)
    p = tmp_path / "pdeg.jsonl"
    _write_predictions(p, [
        {"instance_id": "i1", "model_patch": "@@ -1 +1 @@\n+x=1\n", "trajectory_id": "t0"},
        {"instance_id": "i1", "model_patch": "@@ -1 +1 @@\n+y=2\n", "trajectory_id": "t1"},
    ])
    out = cm.selected_pass_at_1(str(p), str(d))
    row = out["per_instance"]["i1"]
    assert row["majority_multiplicity"] == 1 and row["degenerate_tiebreak"] is True
    assert out["n_degenerate_tiebreak_instances"] == 1


def test_selected_pass_at_1_majority_signature(tmp_path):
    d = tmp_path / "esel"
    d.mkdir()
    # Majority signature (t0,t1 identical edit) FAILS; the distinct t2 passes.
    # Selector must pick the majority (deployable, no oracle) -> miss.
    with open(os.path.join(str(d), "trajectory_eval_i1.json"), "w", encoding="utf-8") as f:
        json.dump({"instance_id": "i1", "trajectories": [
            {"trajectory_id": "t0", "resolved": False},
            {"trajectory_id": "t1", "resolved": False},
            {"trajectory_id": "t2", "resolved": True},
        ]}, f)
    p = tmp_path / "psel.jsonl"
    _write_predictions(p, [
        {"instance_id": "i1", "model_patch": "@@ -1 +1 @@\n+x=1\n", "trajectory_id": "t0"},
        {"instance_id": "i1", "model_patch": "@@ -9 +9 @@\n+x=1\n", "trajectory_id": "t1"},
        {"instance_id": "i1", "model_patch": "@@ -1 +1 @@\n+y=2\n", "trajectory_id": "t2"},
    ])
    out = cm.selected_pass_at_1(str(p), str(d))
    assert out["per_instance"]["i1"]["selected_tid"] == "t0"
    assert out["selected_pass_at_1"] == 0.0   # honest: majority missed the fix


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


def test_load_entropy_uses_last_block_after_rerun(tmp_path):
    """phased_decisions.log is append-mode: a re-run adds a second STRATEGY
    PROPOSAL block while predictions keep-last. Entropy must come from the
    LAST block — the first is the stale run."""
    iid = "sympy__sympy-2"
    inst_dir = tmp_path / iid
    inst_dir.mkdir()
    (inst_dir / "phased_decisions.log").write_text(
        "STRATEGY PROPOSAL\nProposed: 5 | Clusters: 3 | Unique: 3 | Entropy: 0.901\n"
        "...\n"
        "STRATEGY PROPOSAL\nProposed: 5 | Clusters: 2 | Unique: 2 | Entropy: 0.500\n",
        encoding="utf-8",
    )
    ent = cm.load_entropy(str(tmp_path), [iid])
    assert abs(ent[iid] - 0.500) < 1e-9


def test_load_eval_drops_null_trajectory_id(tmp_path):
    """A null trajectory_id is the unnormalized best-of row; load_eval must
    drop it exactly like load_eval_by_tid does, or the coverage table's n
    desyncs from every tid-joined analysis."""
    d = tmp_path / "evalN"
    d.mkdir()
    with open(os.path.join(str(d), "trajectory_eval_i1.json"), "w", encoding="utf-8") as f:
        json.dump({"instance_id": "i1", "trajectories": [
            {"trajectory_id": None, "resolved": True},     # unnormalized primary
            {"trajectory_id": "t0", "resolved": True},
            {"trajectory_id": "t1", "resolved": False},
        ]}, f)
    assert cm.load_eval(str(d))["i1"] == [True, False]     # n=2, not 3


def test_compare_nonempty_robustness_separates_productivity_from_diversity(tmp_path):
    """H1 confound guard: arm A produces 2 distinct patches; arm B produces ONE
    patch and one EMPTY. The raw rarefied gain is positive partly because B
    failed to produce; the non-empty-only robustness row compares at
    k*_ne = 1, where both arms show 1 distinct — gain 0."""
    dA, dB = tmp_path / "neA", tmp_path / "neB"
    dA.mkdir(); dB.mkdir()
    _write_eval(str(dA), "i1", [False, False])
    _write_eval(str(dB), "i1", [False, False])
    pA, pB = tmp_path / "neA.jsonl", tmp_path / "neB.jsonl"
    _write_predictions(pA, [
        {"instance_id": "i1", "model_patch": "@@ -1 +1 @@\n+x=1\n", "trajectory_id": "t0"},
        {"instance_id": "i1", "model_patch": "@@ -1 +1 @@\n+y=2\n", "trajectory_id": "t1"},
    ])
    _write_predictions(pB, [
        {"instance_id": "i1", "model_patch": "@@ -1 +1 @@\n+z=3\n", "trajectory_id": "t0"},
        {"instance_id": "i1", "model_patch": "", "trajectory_id": "t1"},   # empty
    ])
    preds_a, preds_b = cm.load_predictions(str(pA)), cm.load_predictions(str(pB))
    ta = cm.per_instance_table(preds_a, cm.load_eval(str(dA)))
    tb = cm.per_instance_table(preds_b, cm.load_eval(str(dB)))
    comp = cm.compare(ta, tb, entropy={}, seed=0, split=None,
                      preds_a=preds_a, preds_b=preds_b)
    # Raw H1 at k*=2: A has E[distinct]=2; B has 1 (one signature, empty adds none).
    assert abs(comp["rarefied_distinct_gain"]["mean"] - 1.0) < 1e-9
    # Diagnostics expose the production gap and the non-empty-only null.
    assert abs(comp["nonempty_patch_fraction"]["arm_a"] - 1.0) < 1e-9
    assert abs(comp["nonempty_patch_fraction"]["arm_b"] - 0.5) < 1e-9
    assert abs(comp["rarefied_distinct_gain_nonempty"]["mean"] - 0.0) < 1e-9
