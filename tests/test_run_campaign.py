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
