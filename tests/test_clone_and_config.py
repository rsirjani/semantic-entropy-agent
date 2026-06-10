"""Iteration-15 guards: strict deletion-aware container cloning (the SDLG
fork-state path), search-report elided-output inclusion, and config/defaults
completeness.

Why these matter (R7.2 / R8.2 / mechanism integrity):
- `clone_container_state` is the SDLG arm's fork-state replicator. The old
  implementation silently degraded on every failure path (git-list failure ->
  "nothing to clone"; per-file copy failure -> skipped; deletions ->
  structurally impossible to propagate via docker cp), so a fork could start
  from a state that is NOT the parent's and its results would still be
  attributed to the SDLG mechanism. The strict contract raises
  ContainerCloneError instead, which `_clone_for_sdlg` already converts into
  a failed-at-creation draw (honest R7.2 accounting).
- `build_search_report` matched only the `<output>` tag; long observations are
  rendered as `<output_head>`/`<output_tail>` (3/120 SEARCH-phase pilot
  observations) and were silently invisible to the strategy proposer.
- `configs/branching.yaml` claims (via branching_defaults.py's docstring) to
  set every default explicitly; `clustering_strategy` and `kernel_t` were
  missing, so the shipped config and the defaults file could silently diverge.
"""

import os
import subprocess
import sys
import types

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.docker_helpers import ContainerCloneError, clone_container_state
from src.agent.branching_defaults import BRANCHING_DEFAULTS


# --------------------------------------------------------------------------- #
# clone_container_state — strict, deletion-aware
# --------------------------------------------------------------------------- #

class _FakeRun:
    """Dispatching fake for subprocess.run inside clone_container_state."""

    def __init__(self, tracked="", untracked="", fail_on=None):
        self.tracked = tracked          # `git diff --name-status -z` stdout
        self.untracked = untracked      # `git ls-files ... -z` stdout
        self.fail_on = fail_on or (lambda cmd: False)
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        rc = 1 if self.fail_on(cmd) else 0
        stdout = ""
        if "ls-files" in cmd:
            stdout = self.untracked
        elif "diff" in cmd:
            stdout = self.tracked
        return subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr="boom" if rc else "")


def _patch_run(monkeypatch, fake):
    import src.utils.docker_helpers as dh
    monkeypatch.setattr(dh.subprocess, "run", fake)


def test_clone_noop_on_clean_tree(monkeypatch):
    fake = _FakeRun()
    _patch_run(monkeypatch, fake)
    clone_container_state("src", "dst")
    # Only the two listing calls — no cp, no rm.
    assert len(fake.calls) == 2
    assert not any("cp" in c for c in fake.calls)


def test_clone_propagates_deletions(monkeypatch):
    """A deleted file must be deleted in the fork — docker cp cannot copy a
    file that no longer exists; the old code warned and left the fork's copy
    alive (silent state desync)."""
    fake = _FakeRun(tracked="D\0sympy/old.py\0M\0sympy/core/expr.py\0")
    _patch_run(monkeypatch, fake)
    clone_container_state("src", "dst")
    rm_calls = [c for c in fake.calls if "rm" in c]
    assert len(rm_calls) == 1
    assert "/testbed/sympy/old.py" in rm_calls[0]
    assert "dst" in rm_calls[0]
    cp_calls = [c for c in fake.calls if c[:2] == ["docker", "cp"]]
    # modified file: one cp out of src, one cp into dst
    assert any("src:/testbed/sympy/core/expr.py" in " ".join(c) for c in cp_calls)
    assert any("dst:/testbed/sympy/core/expr.py" in " ".join(c) for c in cp_calls)


def test_clone_copies_untracked(monkeypatch):
    fake = _FakeRun(untracked="repro.py\0")
    _patch_run(monkeypatch, fake)
    clone_container_state("src", "dst")
    assert any("src:/testbed/repro.py" in " ".join(c) for c in fake.calls)


def test_clone_raises_on_git_list_failure(monkeypatch):
    """git-list failure must NOT be treated as 'nothing to clone'."""
    fake = _FakeRun(fail_on=lambda cmd: "diff" in cmd)
    _patch_run(monkeypatch, fake)
    with pytest.raises(ContainerCloneError):
        clone_container_state("src", "dst")


def test_clone_raises_on_copy_failure(monkeypatch):
    """A failed file copy must abort (-> failed draw), never a partial clone."""
    fake = _FakeRun(
        tracked="M\0sympy/core/expr.py\0",
        fail_on=lambda cmd: cmd[:2] == ["docker", "cp"],
    )
    _patch_run(monkeypatch, fake)
    with pytest.raises(ContainerCloneError):
        clone_container_state("src", "dst")


def test_clone_rename_detection_disabled(monkeypatch):
    """Renames must surface as D + A (two handleable entries), so the listing
    call must pin diff.renames=false."""
    fake = _FakeRun()
    _patch_run(monkeypatch, fake)
    clone_container_state("src", "dst")
    diff_call = next(c for c in fake.calls if "diff" in c)
    joined = " ".join(diff_call)
    assert "diff.renames=false" in joined and "-z" in joined


# --------------------------------------------------------------------------- #
# build_search_report — elided long outputs reach the proposer
# --------------------------------------------------------------------------- #

def test_search_report_includes_elided_output_head(monkeypatch):
    from src.diversity import strategy_proposer as sp

    captured = {}

    def fake_completion(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        msg = types.SimpleNamespace(content="ROOT CAUSE: ...")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    monkeypatch.setattr(sp.litellm, "completion", fake_completion)

    proposer = sp.StrategyProposer(model_name="openai/fake")
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "THOUGHT: inspect long file"},
        {"role": "user", "content": "<returncode>0</returncode>\n<warning>too long</warning>"
                                    "<output_head>\ndef _print_Pow(self, expr): MARKER_HEAD\n</output_head>"
                                    "<elided_chars>12000</elided_chars><output_tail>tail</output_tail>"},
        {"role": "assistant", "content": "THOUGHT: inspect short file"},
        {"role": "user", "content": "<returncode>0</returncode><output>\nMARKER_SHORT\n</output>"},
    ]
    report = proposer.build_search_report(messages)
    assert report  # canned response came back
    # Both the normal and the elided observation must appear in the LLM prompt.
    assert "MARKER_SHORT" in captured["prompt"]
    assert "MARKER_HEAD" in captured["prompt"]


# --------------------------------------------------------------------------- #
# config/defaults completeness (R8.2)
# --------------------------------------------------------------------------- #

def test_branching_yaml_sets_every_default_explicitly():
    """branching_defaults.py promises: 'the shipped configs/branching.yaml sets
    all of them explicitly'. Lock that promise so the shipped config and the
    fallback table cannot silently diverge."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "configs", "branching.yaml"), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    branching = config["branching"]
    missing = sorted(set(BRANCHING_DEFAULTS) - set(branching))
    assert not missing, f"configs/branching.yaml branching section missing: {missing}"
