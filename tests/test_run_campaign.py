"""Campaign driver pure logic (R8.5 style: no GPU/Docker/claude needed) +
the eval duplicate-propagation fix that keeps metric-time n exact."""

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import run_campaign as rc
from eval_all_trajectories import propagate_duplicate_results


# --------------------------------------------------------------------------- #
# Decision validation
# --------------------------------------------------------------------------- #

def _state(completed=()):
    return {"started_ts": time.time(), "completed_specs": list(completed),
            "phase_log": []}


def test_validate_decision_accepts_menu_choice():
    choice, why = rc.validate_decision(
        {"choice": "sdlg_t0.7", "rationale": "mechanism contrast is informative"},
        _state(completed=["strategy_t0.7"]),
    )
    assert choice == "sdlg_t0.7" and "contrast" in why


def test_validate_decision_stop():
    choice, why = rc.validate_decision(
        {"choice": "stop", "rationale": "nothing left"}, _state())
    assert choice is None


def test_validate_decision_rejects_unknown_and_dupes_and_empty_rationale():
    with pytest.raises(ValueError):
        rc.validate_decision({"choice": "strategy_t9.9", "rationale": "x"}, _state())
    with pytest.raises(ValueError):
        rc.validate_decision({"choice": "strategy_t0.7", "rationale": "x"},
                             _state(completed=["strategy_t0.7"]))
    with pytest.raises(ValueError):
        rc.validate_decision({"choice": "sdlg_t0.7", "rationale": "  "}, _state())
    with pytest.raises(ValueError):
        rc.validate_decision(["not", "a", "dict"], _state())


def test_remaining_menu_excludes_completed():
    rem = rc.remaining_menu(_state(completed=["strategy_t0.7", "sdlg_t0.7"]))
    assert "strategy_t0.7" not in rem and "sdlg_t0.7" not in rem
    assert "strategy_t1.0" in rem


# --------------------------------------------------------------------------- #
# Step construction — both arms explicitly at the SAME temperature (R2.4)
# --------------------------------------------------------------------------- #

def test_build_steps_temperature_matched_and_isolated():
    steps = {s["name"]: s for s in rc.build_steps("strategy_t0.2")}
    tr = steps["strategy_t0.2/treatment_run"]["cmd"]
    ct = steps["strategy_t0.2/control_run"]["cmd"]
    assert tr[tr.index("--temperature") + 1] == "0.2"
    assert ct[ct.index("--temperatures") + 1] == "0.2"
    # Isolation (R2.5): treatment and control land in spec-specific dirs.
    assert tr[tr.index("--results-dir") + 1].endswith("strategy_t0.2")
    assert ct[ct.index("--results-dir") + 1].endswith("resample_strategy_t0.2")
    # Control reads k from THIS treatment dir (matched-k by construction).
    assert ct[ct.index("--treatment-dir") + 1].endswith("strategy_t0.2")
    # Eval dirs: control eval points at the _t<T> dir the baseline writes.
    assert steps["strategy_t0.2/control_eval"]["eval_dir"].endswith(
        "resample_strategy_t0.2_t0.2")


def test_build_steps_diversity_method_per_arm():
    sdlg = {s["name"]: s for s in rc.build_steps("sdlg_t0.7")}
    cmd = sdlg["sdlg_t0.7/treatment_run"]["cmd"]
    assert cmd[cmd.index("--diversity-method") + 1] == "sdlg"
    kern = {s["name"]: s for s in rc.build_steps("kernel_t0.7")}
    kcmd = kern["kernel_t0.7/treatment_run"]["cmd"]
    assert kcmd[kcmd.index("--clustering-strategy") + 1] == "kernel"


def test_phase_a_is_the_preregistered_primary():
    spec = rc.MENU[rc.PHASE_A_KEY]
    assert spec["temperature"] == rc.PRIMARY_TEMP == 0.7
    assert spec["arm"] == "strategy_proposal" and spec["clustering"] == "greedy"


def test_build_steps_audits_both_arms():
    """R6.3: the fairness comparison needs BOTH arms' token totals."""
    steps = {s["name"]: s for s in rc.build_steps("strategy_t0.7")}
    tcmd = steps["strategy_t0.7/budget_audit_treatment"]["cmd"]
    ccmd = steps["strategy_t0.7/budget_audit_control"]["cmd"]
    assert tcmd[tcmd.index("--results-dir") + 1].endswith("strategy_t0.7")
    assert ccmd[ccmd.index("--results-dir") + 1].endswith(
        "resample_strategy_t0.7_t0.7")
    # Distinct output files — neither audit overwrites the other.
    assert tcmd[tcmd.index("--out") + 1] != ccmd[ccmd.index("--out") + 1]


def _script_long_flags(script_name: str) -> set[str]:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts", script_name)
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    import re
    return set(re.findall(r'add_argument\(\s*"(--[a-z][a-z-]*)"', src))


def test_build_steps_flags_exist_in_target_scripts():
    """Wiring-drift guard: every flag the campaign constructs must exist in the
    target script's argparse. The campaign worktree once diverged from the
    driver fixes for four scrutiny iterations — this makes that class of drift
    a test failure instead of a mid-campaign crash."""
    for key in rc.MENU:
        for step in rc.build_steps(key):
            cmd = step.get("cmd")
            if not cmd:
                continue
            script = os.path.basename(cmd[1])
            known = _script_long_flags(script)
            used = {c for c in cmd if c.startswith("--")}
            assert used <= known, f"{step['name']}: {script} lacks {used - known}"


def test_eval_loop_flags_exist():
    known = _script_long_flags("eval_all_trajectories.py")
    assert {"--results-dir", "--instance"} <= known


def test_analyst_prompt_reflects_current_design():
    """The analyst must be pointed at the post-iteration-7 design, not the
    iteration-3 snapshot: H1 before H2, power floor, gate-saturation check,
    productivity diagnostic, and the first-run-is-confirmatory pin."""
    state = _state(completed=["strategy_t0.7"])
    state["phase_log"] = [{"spec": "strategy_t0.7", "status": "completed",
                           "metrics_path": "results/metrics_x.json"}]
    p = rc.analyst_prompt(state, "campaign_decisions/decision_01.json")
    for needle in ("scrutiny_07", "H1", "H2", "min_achievable_p",
                   "nonempty_patch_fraction", "threat 11",
                   "FIRST completed Phase A", "BOTH arms"):
        assert needle in p, f"analyst prompt missing: {needle}"


# --------------------------------------------------------------------------- #
# vLLM model-identity check (R7.4)
# --------------------------------------------------------------------------- #

def test_expected_model_id_strips_litellm_prefix(tmp_path):
    cfg = tmp_path / "branching.yaml"
    cfg.write_text("model:\n  model_name: \"openai/qwen3-coder\"\n", encoding="utf-8")
    assert rc.expected_model_id(str(cfg)) == "qwen3-coder"
    cfg.write_text("model:\n  model_name: \"qwen3-coder\"\n", encoding="utf-8")
    assert rc.expected_model_id(str(cfg)) == "qwen3-coder"
    assert rc.expected_model_id(str(tmp_path / "missing.yaml")) is None


def test_unexpected_tree_changes_flags_code_not_artifacts():
    before = " M results/old.json\n"
    after = (" M results/old.json\n"
             "?? campaign_decisions/decision_01.json\n"
             "?? results/campaign/campaign.log\n"
             " M scripts/compute_metrics.py\n"
             'R  "configs/branching.yaml" -> "configs/evil.yaml"\n')
    flagged = rc.unexpected_tree_changes(before, after)
    assert len(flagged) == 2
    assert any("compute_metrics" in f for f in flagged)
    assert any("evil" in f for f in flagged)
    # No changes -> nothing flagged
    assert rc.unexpected_tree_changes(before, before) == []


def test_model_mismatch_only_on_demonstrable_mismatch():
    assert rc.model_mismatch_error("qwen3-coder", ["qwen3-coder"]) is None
    assert rc.model_mismatch_error("qwen3-coder", []) is None      # unknown served
    assert rc.model_mismatch_error(None, ["other"]) is None        # unknown expected
    err = rc.model_mismatch_error("qwen3-coder", ["llama-3-8b"])
    assert err and "qwen3-coder" in err and "llama-3-8b" in err


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #

class _Args:
    max_hours = 48.0
    min_disk_gb = 150.0


def test_guardrails_wall_clock(monkeypatch):
    monkeypatch.setattr(rc.os.path, "exists", lambda p: False)
    st = _state()
    st["started_ts"] = time.time() - 49 * 3600
    ok, why = rc.guardrails_ok(st, _Args())
    assert not ok and "wall-clock" in why


def test_guardrails_disk_floor(monkeypatch):
    monkeypatch.setattr(rc.os.path, "exists", lambda p: False)
    monkeypatch.setattr(rc, "disk_free_gb", lambda path=None: 10.0)
    ok, why = rc.guardrails_ok(_state(), _Args())
    assert not ok and "disk floor" in why


def test_guardrails_stop_file(monkeypatch):
    monkeypatch.setattr(rc.os.path, "exists", lambda p: p == rc.STOP_FILE)
    ok, why = rc.guardrails_ok(_state(), _Args())
    assert not ok and "STOP" in why


def test_guardrails_pass(monkeypatch):
    monkeypatch.setattr(rc.os.path, "exists", lambda p: False)
    monkeypatch.setattr(rc, "disk_free_gb", lambda path=None: 500.0)
    ok, _ = rc.guardrails_ok(_state(), _Args())
    assert ok


# --------------------------------------------------------------------------- #
# State round-trip
# --------------------------------------------------------------------------- #

def test_state_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "CAMPAIGN_DIR", str(tmp_path))
    monkeypatch.setattr(rc, "STATE_PATH", str(tmp_path / "campaign_state.json"))
    st = _state(completed=["strategy_t0.7"])
    rc.save_state(st)
    loaded = rc.load_state(resume=True)
    assert loaded["completed_specs"] == ["strategy_t0.7"]
    fresh = rc.load_state(resume=False)
    assert fresh["completed_specs"] == []


# --------------------------------------------------------------------------- #
# Analyst prompt content
# --------------------------------------------------------------------------- #

def test_analyst_prompt_lists_menu_and_completed():
    st = _state(completed=["strategy_t0.7"])
    st["phase_log"] = [{"spec": "strategy_t0.7", "status": "completed",
                        "metrics_path": "results/metrics_strategy_t0.7_vs_vanilla.json"}]
    prompt = rc.analyst_prompt(st, "campaign_decisions/decision_01.json")
    assert "metrics_strategy_t0.7_vs_vanilla.json" in prompt
    assert "sdlg_t0.7" in prompt and "strategy_t0.7:" not in prompt.split("menu")[-1]
    assert "stop" in prompt and "decision_01.json" in prompt
    # The null-is-valid instruction must be present (do not chase a positive).
    assert "null result is a valid outcome" in prompt


# --------------------------------------------------------------------------- #
# Eval duplicate/empty propagation (metric-time n stays exact)
# --------------------------------------------------------------------------- #

def test_propagate_duplicates_and_empty():
    trajs = [
        {"trajectory_id": "t0", "model_patch": "A"},
        {"trajectory_id": "t1", "model_patch": "A"},    # duplicate of t0
        {"trajectory_id": "t2", "model_patch": ""},     # empty -> failed row
        {"trajectory_id": "t3", "model_patch": "B"},
    ]
    evaluated = {
        "A": {"trajectory_id": "t0", "resolved": True, "patch_len": 1},
        "B": {"trajectory_id": "t3", "resolved": False, "patch_len": 1},
    }
    rows = propagate_duplicate_results(trajs, evaluated)
    assert len(rows) == 4  # n preserved
    by_id = {r["trajectory_id"]: r for r in rows}
    assert by_id["t0"]["resolved"] is True and "deduped_from" not in by_id["t0"]
    assert by_id["t1"]["resolved"] is True and by_id["t1"]["deduped_from"] == "t0"
    assert by_id["t2"]["resolved"] is False and by_id["t2"]["empty_patch"]
    assert by_id["t3"]["resolved"] is False


# --------------------------------------------------------------------------- #
# tau pin + server gating (iteration 9)
# --------------------------------------------------------------------------- #

def test_build_steps_pins_tau_superset_explicitly():
    """R2.4-class: the confirmatory cell is defined by (T, tau). tau=0 must be
    on the command line of EVERY treatment run — the post-hoc tau ablation's
    'superset run' premise must not ride on a config default."""
    for key in rc.MENU:
        steps = {s["name"]: s for s in rc.build_steps(key)}
        cmd = steps[f"{key}/treatment_run"]["cmd"]
        assert cmd[cmd.index("--entropy-threshold") + 1] == "0", key


def test_servers_required_only_for_agent_run_steps():
    """vLLM/NLI are needed by the agent runs only; evals need Docker, and
    metrics/audit/sweep are pure post-processing. A dead vLLM container must
    not block metrics computable from artifacts already on disk."""
    for key in rc.MENU:
        for step in rc.build_steps(key):
            expected = step["name"].endswith(("treatment_run", "control_run"))
            assert bool(step.get("needs_servers")) == expected, step["name"]


# --------------------------------------------------------------------------- #
# Campaign loop end-to-end (mocked) — the rehearsal iteration 8 queued
# --------------------------------------------------------------------------- #

def _patch_campaign_paths(monkeypatch, tmp_path):
    results = tmp_path / "results"
    campaign = results / "campaign"
    decisions = tmp_path / "campaign_decisions"
    monkeypatch.setattr(rc, "RESULTS", str(results))
    monkeypatch.setattr(rc, "CAMPAIGN_DIR", str(campaign))
    monkeypatch.setattr(rc, "DECISIONS_DIR", str(decisions))
    monkeypatch.setattr(rc, "STATE_PATH", str(campaign / "campaign_state.json"))
    monkeypatch.setattr(rc, "STOP_FILE", str(decisions / "STOP"))


def test_campaign_loop_end_to_end_mocked(monkeypatch, tmp_path):
    """Execute main()'s full state machine with stubbed steps/analyst: Phase A
    runs first and completely, the analyst is consulted only afterwards, its
    choice runs, 'stop' ends the campaign, and the persisted state records the
    completed specs in order. ensure_servers fires only for the agent runs."""
    _patch_campaign_paths(monkeypatch, tmp_path)
    executed, server_checks = [], []
    monkeypatch.setattr(rc, "ensure_servers",
                        lambda args: server_checks.append(len(executed)))
    monkeypatch.setattr(rc, "run_step",
                        lambda step, args: executed.append(step["name"]))
    decisions = iter([("sdlg_t0.7", "mechanism contrast"), (None, "stop")])
    analyst_calls = []
    def fake_analyst(state, n, args):
        analyst_calls.append((n, list(state["completed_specs"])))
        return next(decisions)
    monkeypatch.setattr(rc, "run_analyst", fake_analyst)
    monkeypatch.setattr(sys, "argv", ["run_campaign.py", "--go", "--max-phases", "4"])

    rc.main()

    a_steps = [n for n in executed if n.startswith("strategy_t0.7/")]
    s_steps = [n for n in executed if n.startswith("sdlg_t0.7/")]
    # Phase A ran first, completely, before anything else.
    assert executed[:len(a_steps)] == a_steps
    assert a_steps == [s["name"] for s in rc.build_steps("strategy_t0.7")]
    # The analyst was first consulted only AFTER Phase A completed.
    assert analyst_calls[0] == (1, ["strategy_t0.7"])
    # Its chosen spec ran fully; the second decision ("stop") ended the loop.
    assert s_steps == [s["name"] for s in rc.build_steps("sdlg_t0.7")]
    assert len(analyst_calls) == 2
    # Server preflight fired once per agent-run step (2 per spec), never for
    # eval/metrics/audit/sweep steps.
    assert len(server_checks) == 4
    # Persisted state records completion in execution order.
    with open(rc.STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)
    assert state["completed_specs"] == ["strategy_t0.7", "sdlg_t0.7"]
    assert [p["spec"] for p in state["phase_log"]] == ["strategy_t0.7", "sdlg_t0.7"]
    assert all(p["status"] == "completed" for p in state["phase_log"])


def test_campaign_resume_skips_completed_phase_a(monkeypatch, tmp_path):
    """--resume with Phase A already complete must not re-run it (the FIRST
    completed Phase A run is the confirmatory dataset, R6.5)."""
    _patch_campaign_paths(monkeypatch, tmp_path)
    os.makedirs(rc.CAMPAIGN_DIR, exist_ok=True)
    with open(rc.STATE_PATH, "w", encoding="utf-8") as f:
        json.dump({"started_ts": time.time(), "started": "x",
                   "completed_specs": ["strategy_t0.7"], "phase_log": []}, f)
    executed = []
    monkeypatch.setattr(rc, "ensure_servers", lambda args: None)
    monkeypatch.setattr(rc, "run_step",
                        lambda step, args: executed.append(step["name"]))
    monkeypatch.setattr(rc, "run_analyst", lambda state, n, args: (None, "stop"))
    monkeypatch.setattr(sys, "argv", ["run_campaign.py", "--go", "--resume"])

    rc.main()

    assert executed == []  # nothing re-ran; analyst said stop immediately


def test_campaign_stop_file_aborts_before_any_step(monkeypatch, tmp_path):
    _patch_campaign_paths(monkeypatch, tmp_path)
    os.makedirs(rc.DECISIONS_DIR, exist_ok=True)
    with open(rc.STOP_FILE, "w", encoding="utf-8") as f:
        f.write("halt")
    executed = []
    monkeypatch.setattr(rc, "ensure_servers", lambda args: None)
    monkeypatch.setattr(rc, "run_step",
                        lambda step, args: executed.append(step["name"]))
    monkeypatch.setattr(sys, "argv", ["run_campaign.py", "--go"])
    with pytest.raises(SystemExit):
        rc.main()
    assert executed == []


def test_stale_decision_file_never_read_as_fresh(monkeypatch, tmp_path):
    """Resume restarts decision numbering at 1; a stale decision_01.json from
    an interrupted campaign must be archived, not validated as if this
    analyst call wrote it (it could re-run a spec nobody just chose)."""
    monkeypatch.setattr(rc, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(rc, "DECISIONS_DIR", str(tmp_path / "campaign_decisions"))
    monkeypatch.setattr(rc, "CAMPAIGN_DIR", str(tmp_path / "results" / "campaign"))
    stale = tmp_path / "campaign_decisions" / "decision_01.json"
    os.makedirs(stale.parent, exist_ok=True)
    stale.write_text('{"choice": "strategy_t1.0", "rationale": "stale"}',
                     encoding="utf-8")

    class _FakeProc:
        returncode = 0
        stdout = ""
    # Analyst subprocess runs but writes NO decision file.
    monkeypatch.setattr(rc.subprocess, "run",
                        lambda *a, **k: _FakeProc())
    monkeypatch.setattr(rc.shutil, "which", lambda name: "claude")

    class _A:
        analyst_model = "claude-fable-5"
    choice, why = rc.run_analyst(_state(completed=["strategy_t0.7"]), 1, _A())
    assert choice is None and "no decision file" in why
    assert not stale.exists()                      # archived, not consumed
    assert (stale.parent / "decision_01.json.superseded").exists()
