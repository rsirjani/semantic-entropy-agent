"""R8.5 stage-level pipeline tests (mocked — no GPU/Docker/NLI server).

Covers the stages not already proven in test_pipeline_correctness / the metric
tests, so each part of the pipeline is verified in isolation:

  (a) SEARCH relevance scoring + saturation / step-cap decision.
  (b) strategy-proposal PARSING (clustering+entropy live in test_diversity_clustering).
  (d) source-only fallback patch capture + no-overwrite of a real submission.
  (e) diversity-arm derivation (none / strategy_proposal / sdlg) + SDLG reasoning-only.

The orchestrator __init__ is network/Docker-free (containers are created lazily
in run()), so it can be constructed with a mock NLI against a tmp results dir.
"""

import pytest

from src.agent.branching_defaults import VALID_DIVERSITY_METHODS
from src.agent.phases import should_end_search
from src.diversity.relevance import RelevanceScorer
from src.diversity.sdlg import SDLGGenerator
from src.diversity.strategy_proposer import StrategyProposer


class MockNLI:
    """Entails iff two strings share their first whitespace token."""

    def classify(self, premise, hypothesis):
        e = 0.95 if premise.split()[:1] == hypothesis.split()[:1] else 0.03
        return {"entailment": e, "neutral": 0.0, "contradiction": 1 - e}

    def classify_batch(self, pairs):
        return [self.classify(p, h) for p, h in pairs]


def _make_orchestrator(tmp_path, diversity_method):
    """Construct a PhasedOrchestrator with the mock NLI (no containers yet)."""
    from src.agent.phased_orchestrator import PhasedOrchestrator
    return PhasedOrchestrator(
        instance_id="sympy__sympy-test",
        problem_statement="A bug in Mul.flatten produces wrong output.",
        agent_config={"step_limit": 300},
        model_config={"model_name": "openai/qwen3-coder", "model_kwargs": {}},
        env_config={},
        branching_config={"diversity_method": diversity_method,
                          "results_dir": str(tmp_path)},
        nli_model=MockNLI(),
    )


# ---- (a) SEARCH saturation / step-cap decision -------------------------------

def test_should_end_search_saturation_vs_cap():
    # Not yet past min steps -> keep going even with a low-relevance streak.
    assert should_end_search(step=4, consecutive_low_relevance=5,
                             min_search_steps=8, low_relevance_streak=3,
                             max_search_steps=240) is None
    # Past min steps AND streak reached -> normal saturation exit.
    assert should_end_search(step=10, consecutive_low_relevance=3,
                             min_search_steps=8, low_relevance_streak=3,
                             max_search_steps=240) == "saturated"
    # Streak not reached, under cap -> continue.
    assert should_end_search(step=50, consecutive_low_relevance=1,
                             min_search_steps=8, low_relevance_streak=3,
                             max_search_steps=240) is None
    # Cap reached without saturating (e.g. looping on blocked commands).
    assert should_end_search(step=240, consecutive_low_relevance=0,
                             min_search_steps=8, low_relevance_streak=3,
                             max_search_steps=240) == "step_limit"


def test_relevance_threshold_is_strict():
    """NLI relevance path is pure (no LLM); is_relevant uses a strict > threshold."""
    scorer = RelevanceScorer(nli=MockNLI(), threshold=0.5, use_nli=True)
    # Summary shares the leading token with the problem -> entailment 0.95 > 0.5.
    assert scorer.score("bug everywhere", "bug in flatten") == pytest.approx(0.95)
    # Disjoint leading token -> 0.03, below threshold.
    assert scorer.score("unrelated note", "bug in flatten") == pytest.approx(0.03)
    hit = scorer._score_nli("bug here", "bug in flatten")
    assert (hit > scorer.threshold) is True


# ---- (b) strategy-proposal parsing ------------------------------------------

def test_parse_strategies_extracts_numbered_blocks():
    p = StrategyProposer()
    raw = (
        "STRATEGY 1: Fix in mul.py flatten() — add a guard for the empty-args case.\n"
        "STRATEGY 2 (validation): Fix in core/add.py — normalize inputs before sum.\n"
        "STRATEGY 3: Delegate to a new helper _simplify() in core/operations.py.\n"
    )
    out = p._parse_strategies(raw, expected_n=5)
    assert len(out) == 3
    assert out[0].startswith("Fix in mul.py")
    assert "validation" not in out[1].split("—")[0]  # the (category) tag is stripped


def test_parse_strategies_caps_to_expected_n_and_fallbacks():
    p = StrategyProposer()
    raw = "\n".join(f"STRATEGY {i}: change file_{i}.py to do thing {i}" for i in range(1, 6))
    assert len(p._parse_strategies(raw, expected_n=3)) == 3
    # Unparseable text -> single fallback strategy (never empty).
    assert len(p._parse_strategies("no structure here at all, just prose", 5)) == 1


# ---- (d) source-only fallback capture + no-overwrite -------------------------

class _FakeEnv:
    def __init__(self, diff):
        self.diff = diff
        self.calls = 0

    def execute(self, _):
        self.calls += 1
        return {"output": self.diff}


class _FakeTraj:
    def __init__(self, patch, env):
        self.patch = patch
        self.env = env
        self.trajectory_id = "t0"
        self.step = 7


def test_capture_patch_no_overwrite_of_real_submission(tmp_path):
    orch = _make_orchestrator(tmp_path, "strategy_proposal")
    env = _FakeEnv("diff --git a/x.py b/x.py\n@@ -1 +1 @@\n-a\n+b\n")
    traj = _FakeTraj(patch="REAL SUBMITTED PATCH", env=env)
    orch._capture_patch_if_missing(traj)
    assert traj.patch == "REAL SUBMITTED PATCH"   # untouched
    assert env.calls == 0                          # git diff never run


def test_capture_patch_fills_and_filters_tests(tmp_path):
    orch = _make_orchestrator(tmp_path, "strategy_proposal")
    diff = (
        "diff --git a/sympy/core/mul.py b/sympy/core/mul.py\n@@ -1 +1 @@\n-old\n+new\n"
        "diff --git a/sympy/core/tests/test_mul.py b/sympy/core/tests/test_mul.py\n"
        "@@ -1 +1 @@\n-t\n+t2\n"
    )
    traj = _FakeTraj(patch="", env=_FakeEnv(diff))
    orch._capture_patch_if_missing(traj)
    assert "core/mul.py" in traj.patch          # source kept
    assert "test_mul.py" not in traj.patch      # test edit filtered out


# ---- (e) diversity-arm derivation + SDLG reasoning-only ----------------------

def test_arm_derivation_is_mutually_exclusive(tmp_path):
    sp = _make_orchestrator(tmp_path, "strategy_proposal")
    assert (sp.use_strategy_proposal, sp.sdlg_enabled) == (True, False)
    sd = _make_orchestrator(tmp_path, "sdlg")
    assert (sd.use_strategy_proposal, sd.sdlg_enabled) == (False, True)
    nn = _make_orchestrator(tmp_path, "none")
    assert (nn.use_strategy_proposal, nn.sdlg_enabled) == (False, False)
    assert set(VALID_DIVERSITY_METHODS) == {"strategy_proposal", "sdlg", "none"}


def test_invalid_arm_raises(tmp_path):
    with pytest.raises(ValueError):
        _make_orchestrator(tmp_path, "bogus_method")


def test_sdlg_defaults_to_reasoning_only(tmp_path):
    # Default config -> code-token substitution OFF (faithful to R1.1).
    orch = _make_orchestrator(tmp_path, "sdlg")
    assert orch.sdlg.diversify_code is False
    # The generator honors an explicit opt-in.
    assert SDLGGenerator(nli_model=MockNLI(), diversify_code=True).diversify_code is True
