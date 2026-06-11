"""Recursive per-turn SDLG tree — unit/stage tests (no GPU/Docker/NLI server).

Pins the three user decisions:
  1. single highest-score substitution per turn (top_thought_substitution)
  2. distinctness tested against ALL live branches (_sdlg_distinct_from_all)
  3. fork iff distinct, else prune; cap 30 (_sdlg_fork_per_turn gating)
via fake-`self` objects and monkeypatched heavy deps, so the branching LOGIC is
provable without launching a run.
"""
import types

import pytest

from src.agent.phased_orchestrator import PhasedOrchestrator
from src.diversity.sdlg import SDLGGenerator, SubstitutionCandidate


# --------------------------------------------------------------------------- #
# Decision 1: single highest-score substitution
# --------------------------------------------------------------------------- #

def _cand(pos, orig, sub, score):
    return SubstitutionCandidate(
        position=pos, original_token=orig, substitute_token=sub, substitute_id=0,
        attribution=score, substitution=score, importance=score, combined_score=score,
    )


def test_top_substitution_picks_rank1_and_returns_metadata(monkeypatch):
    gen = SDLGGenerator(nli_model=object(), n_candidates=5)
    # extract_thought_text returns (thought, rest); give a long-enough thought.
    monkeypatch.setattr("src.diversity.sdlg.extract_thought_text",
                        lambda r: ("I will rewrite the parser to handle the edge case", ""))
    # ranked desc by score; rank-1 is the 0.9 one.
    ranked = [_cand(3, "parser", "lexer", 0.9), _cand(5, "edge", "corner", 0.4)]
    monkeypatch.setattr(gen, "_rank_substitutions", lambda *a, **k: ranked)
    monkeypatch.setattr(gen, "_generate_thought_alternative",
                        lambda sub, *a, **k: f"ALT using {sub.substitute_token}")
    out = gen.top_thought_substitution("m", {}, [], "greedy response")
    assert out is not None
    assert out["response"] == "ALT using lexer"           # rank-1 substitute used
    assert out["substitution"]["substitute"] == "lexer"
    assert out["substitution"]["score"] == 0.9


def test_top_substitution_none_when_short_or_no_divergence(monkeypatch):
    gen = SDLGGenerator(nli_model=object(), n_candidates=5)
    monkeypatch.setattr("src.diversity.sdlg.extract_thought_text", lambda r: ("too short", ""))
    assert gen.top_thought_substitution("m", {}, [], "x") is None  # <5 words
    monkeypatch.setattr("src.diversity.sdlg.extract_thought_text",
                        lambda r: ("a long enough reasoning trace here now", ""))
    monkeypatch.setattr(gen, "_rank_substitutions", lambda *a, **k: [_cand(1, "a", "b", 0.5)])
    # alternative reproduces greedy -> no real divergence -> None
    monkeypatch.setattr(gen, "_generate_thought_alternative", lambda *a, **k: "GREEDY")
    assert gen.top_thought_substitution("m", {}, [], "GREEDY") is None


# --------------------------------------------------------------------------- #
# Decision 2: distinct vs ALL live branches
# --------------------------------------------------------------------------- #

class _NLI:
    """entailment(a,b) from a dict; default 0 (not entailing)."""
    def __init__(self, pairs):
        self.pairs = pairs
    def classify(self, a, b):
        return {"entailment": self.pairs.get((a, b), 0.0),
                "neutral": 0.0, "contradiction": 0.0}


def _fake_orch(live, nli):
    return types.SimpleNamespace(
        _live_descriptions=live, nli=nli,
        clusterer=types.SimpleNamespace(threshold=0.7),
    )


def test_distinct_when_no_bidirectional_entailment():
    nli = _NLI({})  # nothing entails anything
    orch = _fake_orch({"t0": "fix the parser", "t1": "patch the printer"}, nli)
    d, conflict = PhasedOrchestrator._sdlg_distinct_from_all(orch, "rewrite the evaluator")
    assert d is True and conflict is None


def test_not_distinct_when_bidirectional_entails_a_live_branch():
    # candidate <-> t1 both directions > 0.7 => collapses into t1
    nli = _NLI({("repair the printer", "patch the printer"): 0.95,
                ("patch the printer", "repair the printer"): 0.92})
    orch = _fake_orch({"t0": "fix the parser", "t1": "patch the printer"}, nli)
    d, conflict = PhasedOrchestrator._sdlg_distinct_from_all(orch, "repair the printer")
    assert d is False and conflict == "t1"


def test_one_directional_entailment_is_still_distinct():
    # only forward entails -> NOT bidirectional -> kept distinct
    nli = _NLI({("a", "patch the printer"): 0.95})  # backward missing => 0
    orch = _fake_orch({"t1": "patch the printer"}, nli)
    d, _ = PhasedOrchestrator._sdlg_distinct_from_all(orch, "a")
    assert d is True


def test_empty_registry_is_distinct():
    orch = _fake_orch({}, _NLI({}))
    d, conflict = PhasedOrchestrator._sdlg_distinct_from_all(orch, "first branch")
    assert d is True and conflict is None


# --------------------------------------------------------------------------- #
# Decision 3: fork iff distinct else prune; cap respected
# --------------------------------------------------------------------------- #

def _fork_orch(*, can_branch, distinct, top=True):
    """Fake orchestrator exposing exactly what _sdlg_fork_per_turn touches."""
    log = []
    forks_made = []

    class Mgr:
        branching_log = log
        def can_branch(self, n): return can_branch

    o = types.SimpleNamespace(
        _live_descriptions={"t0": "greedy strategy"},
        _sdlg_pending_forks=[], _sdlg_fork_counter=1,
        manager=Mgr(), model_config={}, tracer=types.SimpleNamespace(log=lambda *a, **k: None),
        sdlg=types.SimpleNamespace(
            top_thought_substitution=lambda *a, **k: (
                {"response": "ALT", "substitution": {"substitute": "x", "score": 0.9}}
                if top else None)),
    )
    o._intent_of = lambda resp, traj: "candidate strategy"
    o._sdlg_distinct_from_all = lambda intent: (distinct, None if distinct else "t0")
    def _clone(parent, content, idx):
        f = types.SimpleNamespace(trajectory_id=f"t0_sdlg_{idx}", status="active")
        forks_made.append(f)
        return f
    o._clone_for_sdlg = _clone
    o._forks_made = forks_made
    return o


def _traj():
    return types.SimpleNamespace(trajectory_id="t0", step=7,
                                 agent=types.SimpleNamespace(messages=["m", "greedy"]))


def test_fork_created_and_queued_when_distinct():
    o = _fork_orch(can_branch=True, distinct=True)
    fork = PhasedOrchestrator._sdlg_fork_per_turn(o, _traj(), {}, "greedy content")
    assert fork is not None and fork.trajectory_id == "t0_sdlg_1"
    assert o._sdlg_pending_forks == [fork]                 # queued for the worklist
    assert o._live_descriptions[fork.trajectory_id] == "candidate strategy"
    assert any(e["event"] == "sdlg_fork" for e in o.manager.branching_log)


def test_pruned_when_not_distinct():
    o = _fork_orch(can_branch=True, distinct=False)
    fork = PhasedOrchestrator._sdlg_fork_per_turn(o, _traj(), {}, "greedy content")
    assert fork is None
    assert o._sdlg_pending_forks == []                     # nothing queued
    assert o._forks_made == []                             # no clone
    pr = [e for e in o.manager.branching_log if e["event"] == "sdlg_prune"]
    assert pr and pr[0]["collapsed_into"] == "t0"


def test_no_fork_when_cap_reached():
    o = _fork_orch(can_branch=False, distinct=True)
    fork = PhasedOrchestrator._sdlg_fork_per_turn(o, _traj(), {}, "greedy content")
    assert fork is None and o._forks_made == []            # cap 30 respected


def test_no_fork_when_no_substitution():
    o = _fork_orch(can_branch=True, distinct=True, top=False)  # top-1 returns None
    fork = PhasedOrchestrator._sdlg_fork_per_turn(o, _traj(), {}, "greedy content")
    assert fork is None and o._forks_made == []
