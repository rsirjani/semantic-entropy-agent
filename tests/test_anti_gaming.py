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


def test_blocks_raw_git_internals_reads():
    """The git-CLI veto alone can be bypassed by reading .git's history-bearing
    internals directly: refs/packed-refs name post-fix commits, logs/ is the
    reflog, objects/ holds the gold patch's blobs (zlib-inflatable via
    python -c). Measured 0/1,041 executed pilot actions touch `.git` at all —
    the veto is purely protective."""
    for cmd in [
        "cat .git/packed-refs",
        "cat /testbed/.git/ORIG_HEAD",
        "ls .git/refs/heads",
        "head .git/logs/HEAD",
        'python -c "print(open(\'.git/refs/heads/master\').read())"',
        "python -c \"import zlib; print(zlib.decompress(open('.git/objects/ab/cdef','rb').read()))\"",
        "grep -r fix .git/logs",
        "cat /testbed/.git/FETCH_HEAD",
    ]:
        forbidden, reason = is_forbidden_command(cmd)
        assert forbidden, cmd
        assert ".git internals" in reason


def test_allows_git_exclusion_idioms_and_safe_paths():
    """A blanket `.git` match would break legitimate exclusion idioms; the veto
    targets only history-bearing subpaths."""
    for cmd in [
        "find /testbed -name '*.py' -not -path './.git/*'",
        "grep -rn foo /testbed --exclude-dir=.git",
        "ls -la /testbed",
        "cd /testbed && git diff",
        "cat /testbed/.gitignore",
    ]:
        forbidden, reason = is_forbidden_command(cmd)
        assert not forbidden, (cmd, reason)


def test_heredoc_patch_content_is_not_a_git_invocation():
    """Measured on archived run-2: 163/1,539 actions were agents WRITING patch
    files whose heredoc body contains `diff --git a/... b/...` — file CONTENT,
    not a git invocation. The veto must strip heredoc bodies (and quoted
    spans) exactly like the write detector."""
    cmd = (
        "cd /testbed && cat > fix_patch.txt <<'EOF'\n"
        "diff --git a/sympy/combinatorics/permutations.py b/sympy/combinatorics/permutations.py\n"
        "--- a/sympy/combinatorics/permutations.py\n"
        "+++ b/sympy/combinatorics/permutations.py\n"
        "@@ -900,7 +900,7 @@\n"
        "-        temp = flatten(args)\n"
        "+        temp = [i for sub in args for i in sub]\n"
        "EOF"
    )
    forbidden, reason = is_forbidden_command(cmd)
    assert not forbidden, reason
    # And a quoted mention of git is content, not an invocation:
    forbidden, reason = is_forbidden_command('echo "now run git diff to check"')
    assert not forbidden, reason


def test_git_working_tree_restore_forms():
    """Measured on archived run-2: 14 actions were the agent reverting its OWN
    edits. `git restore <path>` (no --source) and `git checkout -- <path>`
    cannot read other revisions and must be allowed; ref-ambiguous and
    ref-bearing forms stay vetoed."""
    for cmd in (
        "git restore sympy/core/function.py",
        "cd /testbed && git restore sympy/core/function.py",
        "git checkout -- sympy/core/function.py",
        "git stash pop",
        "git apply fix.patch",
        "git add sympy/core/function.py",
        "git status sympy/",
        "git diff -- sympy/core/function.py",
    ):
        forbidden, reason = is_forbidden_command(cmd)
        assert not forbidden, f"should allow: {cmd} ({reason})"
    for cmd in (
        "git restore --source=HEAD~5 sympy/core/function.py",
        "git checkout sympy/core/function.py",   # ref-ambiguous without --
        "git checkout master",
        "git diff HEAD~5",
        "git diff e92f6f3daf -- sympy/core/function.py",
    ):
        forbidden, reason = is_forbidden_command(cmd)
        assert forbidden, f"should block: {cmd}"
    # The ambiguous-form veto teaches the allowed form:
    _, reason = is_forbidden_command("git checkout sympy/core/function.py")
    assert "git checkout -- <path>" in reason


def test_git_head_literal_and_no_index_are_safe():
    """HEAD is pinned at the base commit (`git commit` is itself vetoed), so
    HEAD-relative working-tree operations reveal nothing; --no-index diff
    compares filesystem paths only. Both idioms appeared in archived run-2."""
    for cmd in (
        "cd /testbed && git checkout HEAD -- sympy/geometry/point.py",
        "git diff HEAD",
        "cd /testbed && git diff HEAD -- sympy/core/symbol.py",
        "git diff --no-index /dev/null sympy/core/symbol.py | head -20",
    ):
        forbidden, reason = is_forbidden_command(cmd)
        assert not forbidden, f"should allow: {cmd} ({reason})"
    for cmd in (
        "git diff HEAD~5",
        "git checkout HEAD~1 -- sympy/core/symbol.py",
        "git diff HEAD^",
        "git commit -m wip",
        "git log HEAD",
    ):
        forbidden, reason = is_forbidden_command(cmd)
        assert forbidden, f"should block: {cmd}"
