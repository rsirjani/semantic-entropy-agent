"""R8.1/R8.3 + driver wiring: write-detection, source-only diff filter, matched-k
discovery, dataset aliases. The orchestrator import is heavy (minisweagent); kept in
its own module so a heavy-import problem can't mask the lighter metric tests."""

import json

from src.agent.phases import is_write_command
from src.evaluation.dataset import resolve_dataset_name


def test_is_write_command_no_false_positives():
    # NOT writes: submit sentinel, bare echo/printf, stderr/null redirects, views.
    assert is_write_command("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /testbed/patch.txt") is False
    assert is_write_command("cd /testbed && cat patch.txt") is False
    assert is_write_command("echo hello world") is False
    assert is_write_command("printf foo") is False
    assert is_write_command("grep -rn foo . 2> /dev/null") is False
    assert is_write_command("find / -name x 2>/dev/null") is False
    assert is_write_command("python repro.py > /dev/null 2>&1") is False


def test_is_write_command_detects_real_writes():
    assert is_write_command("sed -i s/a/b/ f.py") is True
    assert is_write_command("cd /testbed && sed -i s/a/b/ f.py") is True
    assert is_write_command("cat <<'EOF' > /testbed/f.py") is True
    assert is_write_command("echo foo > /testbed/f.py") is True
    assert is_write_command("echo foo >> /testbed/f.py") is True
    assert is_write_command("tee /testbed/f.py") is True
    assert is_write_command("patch -p1 < fix.diff") is True


def test_is_write_command_quoted_comparison_not_a_write():
    """Pilot-measured false-positive class (2/2184 actions): a `>=` inside a
    quoted awk/python program is a comparison, not a redirect. In the SDLG arm
    a false positive BEFORE the first real write corrupts that instance's
    branch point (SDLG fires once, at a view step)."""
    assert is_write_command("awk 'NR>=350 {print NR \": \" $0}' /testbed/sympy/printing/pycode.py | tail -30") is False
    assert is_write_command("cd /testbed && awk 'NR>=495 && NR<=520' /testbed/sympy/core/function.py") is False
    assert is_write_command("awk '/^    >>> kernS/,/^    >>>/ {print NR}' sympy/core/sympify.py") is False
    assert is_write_command("python -c \"print(5 >= 3)\"") is False


def test_is_write_command_chained_write_in_nonfinal_segment():
    """False-negative class of the last-segment-only version: a real write does
    not stop being a write because `&& pytest` follows it."""
    assert is_write_command("sed -i 's/a/b/' f.py && python -m pytest sympy/core/tests") is True
    assert is_write_command("cd /testbed && sed -i 's/a/b/' f.py && python -m pytest") is True
    assert is_write_command("git diff > patch.txt && cat patch.txt") is True
    assert is_write_command("echo fix > /testbed/f.py; ls") is True


def test_is_write_command_heredoc_body_not_inspected():
    """Heredoc BODY lines are file content, not commands — `if x > 0:` inside a
    python heredoc must not read as a redirect. The opening line's redirect
    still counts."""
    assert is_write_command("python - <<'EOF'\nif x > 0:\n    print(x)\nEOF") is False
    assert is_write_command("cat <<'EOF' > /testbed/f.py\nif x > 0:\n    pass\nEOF") is True


def test_search_phase_blocks_writes_and_submission():
    """SEARCH read-only-ness is load-bearing: strategy forks are FRESH
    containers replaying search messages, not filesystem clones, so a SEARCH
    write would desync t0's container state from every fork's. The prefix
    allowlist alone let `echo ... > file` (allowed prefix `echo`) and the
    submit command (allowed prefix `cat` in the last segment) through."""
    from src.agent.phases import Phase, is_command_allowed
    # Reads stay allowed.
    assert is_command_allowed("grep -rn foo sympy/", Phase.SEARCH) is True
    assert is_command_allowed("cd /testbed && cat sympy/core/mul.py", Phase.SEARCH) is True
    assert is_command_allowed("echo checking", Phase.SEARCH) is True
    # Writes hiding behind allowed read prefixes are blocked.
    assert is_command_allowed("echo 'fix' > /testbed/sympy/core/mul.py", Phase.SEARCH) is False
    assert is_command_allowed("cat <<'EOF' > /testbed/repro.py\nx=1\nEOF", Phase.SEARCH) is False
    assert is_command_allowed("sed -i 's/a/b/' f.py && cat f.py", Phase.SEARCH) is False
    # Submission is VERIFY-only — in every phase, regardless of prefix matches.
    submit = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /testbed/patch.txt"
    assert is_command_allowed(submit, Phase.SEARCH) is False
    assert is_command_allowed(submit, Phase.PATCH) is False
    assert is_command_allowed(submit, Phase.VERIFY) is True
    # The VERIFY submit-prep redirect stays allowed (write veto is SEARCH-only).
    assert is_command_allowed("cd /testbed && git diff -- sympy/core/mul.py > patch.txt", Phase.VERIFY) is True


def test_truncate_context_pins_strategy_and_phase_prompts():
    """Mechanism fidelity: 3 pilot trajectories ran past the 80-message
    truncation threshold and silently LOST their assigned-strategy prompt
    (first-4 + last-40 keeps neither). The strategy prompt IS the treatment
    mechanism; truncation must pin '## Current Phase:' user messages."""
    import types
    from src.agent.phases import PATCH_PROMPT_WITH_STRATEGY, VERIFY_PROMPT
    from src.agent.phased_orchestrator import PhasedOrchestrator

    strategy_msg = {"role": "user",
                    "content": PATCH_PROMPT_WITH_STRATEGY.format(strategy="Use a sentinel default")}
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "instance"},
            {"role": "user", "content": "search prompt"},
            {"role": "assistant", "content": "a0"}]
    msgs += [{"role": "user", "content": f"obs {i}"} for i in range(8)]
    msgs.append(strategy_msg)                                   # middle: would be dropped
    msgs.append({"role": "user", "content": VERIFY_PROMPT})     # middle: would be dropped
    msgs += [{"role": "assistant" if i % 2 else "user", "content": f"step {i}"}
             for i in range(80)]
    assert len(msgs) > 80

    captured = {}
    agent = types.SimpleNamespace(messages=msgs,
                                  set_messages=lambda m: captured.update(out=m))
    traj = types.SimpleNamespace(agent=agent, trajectory_id="t0_strategy_1", step=50)
    stub = types.SimpleNamespace(tracer=types.SimpleNamespace(log=lambda *a, **k: None))

    PhasedOrchestrator._truncate_context(stub, traj)
    out = captured["out"]
    assert len(out) < len(msgs)
    contents = [m["content"] for m in out]
    assert strategy_msg["content"] in contents          # pinned, not dropped
    assert VERIFY_PROMPT in contents                    # pinned, not dropped
    assert out[:4] == msgs[:4]                          # head preserved
    assert out[-40:] == msgs[-40:]                      # tail preserved
    # Idempotent under re-truncation: pinned prompts survive a second pass.
    agent2 = types.SimpleNamespace(messages=out + [{"role": "user", "content": f"x{i}"} for i in range(60)],
                                   set_messages=lambda m: captured.update(out2=m))
    traj2 = types.SimpleNamespace(agent=agent2, trajectory_id="t0_strategy_1", step=90)
    PhasedOrchestrator._truncate_context(stub, traj2)
    assert strategy_msg["content"] in [m["content"] for m in captured["out2"]]


def test_resolve_dataset_alias():
    assert resolve_dataset_name("lite") == "SWE-bench/SWE-bench_Lite"
    assert resolve_dataset_name("verified") == "SWE-bench/SWE-bench_Verified"
    assert resolve_dataset_name("org/custom") == "org/custom"   # passthrough


def test_source_only_diff_filter():
    from src.agent.phased_orchestrator import PhasedOrchestrator as PO
    assert PO._is_test_path("a/b/tests/foo.py") is True
    assert PO._is_test_path("a/test_x.py") is True
    assert PO._is_test_path("a/core/mul.py") is False
    diff = (
        "diff --git a/sympy/core/mul.py b/sympy/core/mul.py\n@@ -1 +1 @@\n-old\n+new\n"
        "diff --git a/sympy/core/tests/test_mul.py b/sympy/core/tests/test_mul.py\n@@ -1 +1 @@\n-t\n+t2\n"
    )
    out = PO._filter_diff_to_source(diff)
    assert "core/mul.py" in out and "test_mul.py" not in out


def test_vanilla_none_arm_samples_at_temperature():
    """R2.4: the 'none' arm must decode at sample_temperature>0, not greedy."""
    import run_branching as rb
    base = {"model_kwargs": {"temperature": 0.0, "api_base": "x"}}

    # 'none' arm -> base agent temperature overridden to sample_temperature.
    out = rb.vanilla_samples_at_temperature(
        base, {"diversity_method": "none", "sample_temperature": 0.7})
    assert out["model_kwargs"]["temperature"] == 0.7
    assert base["model_kwargs"]["temperature"] == 0.0  # caller dict untouched

    # Treatment arms keep the base agent greedy (diversity from proposer/SDLG).
    for arm in ("strategy_proposal", "sdlg"):
        out = rb.vanilla_samples_at_temperature(
            base, {"diversity_method": arm, "sample_temperature": 0.7})
        assert out is base  # unchanged, same object
        assert out["model_kwargs"]["temperature"] == 0.0


def test_matched_k_discovery(tmp_path):
    import run_resample_baseline as rb
    for iid, k in [("sympy__sympy-1", 5), ("sympy__sympy-2", 9)]:
        d = tmp_path / iid
        d.mkdir()
        (d / "metadata.json").write_text(json.dumps({"total_trajectories": k}), encoding="utf-8")
    # a dir without metadata.json must be ignored
    (tmp_path / "not_an_instance").mkdir()
    k_by_id = rb.discover_instances_and_k(str(tmp_path), max_k=None)
    assert k_by_id == {"sympy__sympy-1": 5, "sympy__sympy-2": 9}
    capped = rb.discover_instances_and_k(str(tmp_path), max_k=6)
    assert capped == {"sympy__sympy-1": 5, "sympy__sympy-2": 6}


def test_matched_k_discovery_warns_on_patch_row_mismatch(tmp_path, caplog):
    """An old-driver/interrupted treatment artifact (patch entries !=
    total_trajectories) must be flagged: its predictions file does not count
    what total_trajectories counts, so the matched-k denominators disagree."""
    import logging
    import run_resample_baseline as rb
    d = tmp_path / "sympy__sympy-3"
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps({
        "total_trajectories": 5,
        "patches": [{"trajectory_id": f"t{i}", "patch": "x"} for i in range(2)],
    }), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        k_by_id = rb.discover_instances_and_k(str(tmp_path), max_k=None)
    assert k_by_id == {"sympy__sympy-3": 5}      # k source unchanged
    assert any("old-driver or interrupted" in r.message for r in caplog.records)


def _fake_traj(tid, status, patch, submitted=False, step=7, parent=None):
    import types
    return types.SimpleNamespace(
        trajectory_id=tid, status=status, patch=patch, submitted=submitted,
        step=step, parent_id=parent, branch_info=None,
    )


def test_collect_patch_entries_one_entry_per_genuine_draw():
    """R4.1/R7.2 predictions-record completeness at the PRODUCER: failed and
    patchless trajectories are genuine draws (they consumed budget) and must
    appear with patch "" — dropping them deflated the treatment's metric-time
    k while the vanilla driver keeps empty rows for its failed resamples
    (confirmed on the real pilot: 5/10 instances had patches < trajectories)."""
    from src.agent.phased_orchestrator import collect_patch_entries
    trajs = [
        _fake_traj("t0", "completed", "PATCH_A", submitted=True),
        _fake_traj("t0_strategy_1", "completed", ""),       # patchless draw
        _fake_traj("t0_strategy_2", "failed", "PATCH_B"),   # captured on failure
        _fake_traj("t0_strategy_3", "failed", None),        # failed, no patch
        _fake_traj("t9", "active", "X"),                    # interrupted: not a draw
        _fake_traj("t8", "branched", "Y"),                  # legacy parent: excluded
    ]
    entries = collect_patch_entries(trajs, {"t0": "strat A"})
    by_tid = {e["trajectory_id"]: e for e in entries}
    assert sorted(by_tid) == ["t0", "t0_strategy_1", "t0_strategy_2", "t0_strategy_3"]
    assert by_tid["t0"]["patch"] == "PATCH_A" and by_tid["t0"]["strategy"] == "strat A"
    assert by_tid["t0_strategy_1"]["patch"] == ""           # kept, normalized
    assert by_tid["t0_strategy_2"]["patch"] == "PATCH_B"    # failed-but-captured kept
    assert by_tid["t0_strategy_3"]["patch"] == ""           # None -> ""
    assert by_tid["t0_strategy_2"]["status"] == "failed"


def test_build_predictions_writes_a_row_for_every_draw():
    import run_branching as rb
    patches = [
        {"trajectory_id": "t0", "patch": "AAAA", "submitted": False},
        {"trajectory_id": "t0_strategy_1", "patch": "BB", "submitted": True},
        {"trajectory_id": "t0_strategy_2", "patch": "", "submitted": False},
    ]
    preds = rb.build_predictions("i1", patches)
    # Primary first (no trajectory_id), best = the SUBMITTED patch, not the longest.
    assert "trajectory_id" not in preds[0] and preds[0]["model_patch"] == "BB"
    rows = {p["trajectory_id"]: p["model_patch"] for p in preds[1:]}
    assert rows == {"t0": "AAAA", "t0_strategy_1": "BB", "t0_strategy_2": ""}


def test_build_predictions_all_empty_and_none():
    import run_branching as rb
    # All draws patchless: rows still exist (failed draws), primary is "".
    preds = rb.build_predictions("i1", [
        {"trajectory_id": "t0", "patch": "", "submitted": False},
        {"trajectory_id": "t0_strategy_1", "patch": "", "submitted": False},
    ])
    assert preds[0]["model_patch"] == ""
    assert [p["trajectory_id"] for p in preds[1:]] == ["t0", "t0_strategy_1"]
    assert all(p["model_patch"] == "" for p in preds)
    # Zero trajectories (catastrophic instance failure): bare primary only.
    only = rb.build_predictions("i1", [])
    assert len(only) == 1 and only[0]["model_patch"] == ""
