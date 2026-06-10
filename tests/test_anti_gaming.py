"""Anti-gaming guards (R8.x): SWE-bench is gameable via the testbed .git history
(gold patch reachable through `git show`/`git log`/`git blame` past the base
commit) and via the network (eval images have open internet — the upstream
PR/issue is fetchable). These pin the command veto and the container isolation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from src.agent.phases import is_forbidden_command


# --------------------------------------------------------------------------- #
# git history vector
# --------------------------------------------------------------------------- #

def test_blocks_git_history_inspection():
    for cmd in (
        "git log --oneline",
        "git log --all",
        "git show HEAD:sympy/core/sympify.py",
        "git show e92f6f3daf",
        "cd /testbed && git show master:file.py",
        "git blame sympy/core/add.py",
        "git reflog",
        "git rev-list --all",
        "git diff HEAD~5 HEAD",          # diff with a ref argument peeks history
    ):
        forbidden, reason = is_forbidden_command(cmd)
        assert forbidden, f"should block: {cmd}"
        assert "history" in reason


def test_allows_submit_protocol_git():
    # The submit protocol and harmless status checks must still work.
    for cmd in (
        "git diff",
        "git diff > /testbed/patch.txt",
        "cd /testbed && git diff > patch.txt",
        "git status",
        "git stash",
        "git add -A",
    ):
        forbidden, reason = is_forbidden_command(cmd)
        assert not forbidden, f"should allow: {cmd} ({reason})"


def test_blocks_history_hidden_behind_allowed_prefix():
    # A forbidden segment chained after an allowed one must still be caught.
    forbidden, _ = is_forbidden_command("cat README.md && git show HEAD:fix.py")
    assert forbidden
    forbidden, _ = is_forbidden_command("ls; git log")
    assert forbidden
    forbidden, _ = is_forbidden_command("grep -r foo . | git show abc123")
    assert forbidden


# --------------------------------------------------------------------------- #
# network vector
# --------------------------------------------------------------------------- #

def test_blocks_network_fetchers():
    for cmd in (
        "curl https://github.com/sympy/sympy/pull/19495",
        "wget http://example.com/fix.patch",
        "pip install requests",
        "pip download six",
        "cd /testbed && pip install mpmath",
        "git clone https://github.com/sympy/sympy",
        "git fetch origin",
        "nc example.com 443",
    ):
        forbidden, reason = is_forbidden_command(cmd)
        assert forbidden, f"should block: {cmd}"
        assert "network" in reason or "history" in reason


def test_allows_ordinary_investigation():
    for cmd in (
        "grep -rn 'def sympify' sympy/core",
        "cat sympy/core/add.py",
        "python -c \"import sympy; print(sympy.__file__)\"",
        "sed -n '510,520p' sympy/core/sympify.py",
        "pytest sympy/core/tests/test_sympify.py -x",
        "find . -name '*.py' -path '*geometry*'",
    ):
        forbidden, reason = is_forbidden_command(cmd)
        assert not forbidden, f"should allow: {cmd} ({reason})"


# --------------------------------------------------------------------------- #
# container network isolation
# --------------------------------------------------------------------------- #

def test_build_env_config_forces_network_none():
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import run_branching
    cfg = {"environment": {"cwd": "/testbed"}}
    # find_eval_image hits docker; stub it.
    run_branching.find_eval_image = lambda iid: "sweb.eval.x86_64." + iid
    env = run_branching.build_env_config(cfg, "sympy__sympy-19495")
    ra = env["run_args"]
    assert "--network" in ra and ra[ra.index("--network") + 1] == "none"


def test_build_env_config_preserves_existing_run_args():
    import run_branching
    run_branching.find_eval_image = lambda iid: "img"
    cfg = {"environment": {"run_args": ["--memory", "8g"]}}
    env = run_branching.build_env_config(cfg, "x")
    assert "--memory" in env["run_args"] and "--network" in env["run_args"]
