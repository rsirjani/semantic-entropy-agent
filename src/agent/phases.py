"""Phase definitions for the phased branching agent.

Three phases with different tool access and branching strategies:

Phase 1 — SEARCH: Read-only exploration. Agent searches the codebase to
    understand the bug. Cannot modify files. Branches on relevance of
    findings to the problem statement.

Phase 2 — PATCH: Write access. Agent implements a fix. SDLG operates here
    to produce diverse patches. Branches on semantic diversity of fixes.

Phase 3 — VERIFY: Read-only again. Agent runs tests and reviews changes.
    No branching — just validate the patch.

ENFORCEMENT REALITY (honest contract — what the orchestrator actually checks):

- SEARCH: enforced. `PhasedOrchestrator._step_search` blocks any command that
  fails `is_command_allowed(cmd, Phase.SEARCH)`, which combines the read-only
  prefix allowlist with an `is_write_command` veto. The write veto is
  LOAD-BEARING, not cosmetic: strategy forks are fresh containers that replay
  the search MESSAGES, not clones of the searched container's FILESYSTEM
  (`_create_lazy_trajectory`), so a file written during SEARCH would exist in
  t0's container but not in any fork's — silently desynchronizing the
  branches' starting states. Residual gap (documented, undetectable from the
  command string): `python -c "open('f','w')..."` can still write.
- PATCH / VERIFY: prompt-level guidance only. `_step_patch` / `_step_verify`
  never call `is_command_allowed`; the PATCH/VERIFY entries in
  `PHASE_ALLOWED_COMMANDS` describe the intended envelope and feed the phase
  prompts, but any command the agent emits in those phases executes. VERIFY
  cannot be made strictly read-only anyway — the submit protocol requires
  `git diff > patch.txt`. A VERIFY-phase edit is still the agent's own work
  and lands in its captured patch; this is symmetric across arms.
- Phase transitions: SEARCH→PATCH happens ONLY via `should_end_search`
  (relevance saturation or the hard step cap). The SEARCH_PROMPT tells the
  agent that declaring `STRATEGY:` transitions it — operatively, declaring a
  strategy makes the agent stop searching, its steps score low relevance, and
  saturation fires; the prompt sentence is a behavioral nudge, not a wired
  trigger. PATCH→VERIFY uses `detect_phase_transition`.
"""

import re
from enum import Enum


class Phase(Enum):
    SEARCH = "search"
    PATCH = "patch"
    VERIFY = "verify"


# Commands allowed in each phase
PHASE_ALLOWED_COMMANDS = {
    Phase.SEARCH: {
        "grep", "rg", "find", "cat", "head", "tail", "ls", "tree",
        "wc", "file", "sed -n", "nl", "python -c", "python3 -c",
        "cd", "pwd", "echo",  # echo for debugging, not file writes
    },
    Phase.PATCH: {
        # Everything from search PLUS write operations
        "grep", "rg", "find", "cat", "head", "tail", "ls", "tree",
        "wc", "file", "sed -n", "nl", "python -c", "python3 -c",
        "cd", "pwd", "echo",
        "sed -i", "patch", "cat <<", "tee", "mv", "cp",
        "python",  # full python for scripts that write files
    },
    Phase.VERIFY: {
        "grep", "rg", "find", "cat", "head", "tail", "ls", "tree",
        "wc", "file", "sed -n", "nl", "cd", "pwd",
        "python", "python3", "pytest", "git diff", "git status",
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
    },
}


def is_command_allowed(command: str, phase: Phase) -> bool:
    """Check if a bash command is allowed in the given phase.

    Order matters (each rule closes a bypass of the next):

    1. Submission is decided FIRST and only allowed in VERIFY. (Checked after
       the prefix loop, it was dead code: the submit command starts with
       `echo`/ends with `cat patch.txt`, so a read prefix matched first and a
       SEARCH-phase submission sailed through.)
    2. SEARCH additionally vetoes anything `is_write_command` flags. (The
       prefix allowlist alone lets `echo fix > file.py` or `cat <<EOF > f.py`
       through, because `echo`/`cat` are allowed read prefixes — and SEARCH
       read-only-ness is load-bearing for fork-state consistency; see the
       module docstring.)
    3. Otherwise: prefix allowlist on the last `&&` segment (the actual
       action after `cd ... &&` chains).

    Only SEARCH is enforced by the orchestrator; see the module docstring.
    """
    cmd = command.strip()
    if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in cmd:
        return phase == Phase.VERIFY
    if phase == Phase.SEARCH and is_write_command(cmd):
        return False

    # Strip common prefixes like "cd /testbed && "
    if "&&" in cmd:
        # Check the last command in the chain (the actual action)
        cmd = cmd.split("&&")[-1].strip()

    allowed = PHASE_ALLOWED_COMMANDS[phase]
    for prefix in allowed:
        if cmd.startswith(prefix):
            return True

    return False


# In-place editors / file movers that modify a file by argument (no redirect).
_WRITE_PREFIXES = ("sed -i", "tee ", "patch ", "dd ", "truncate ", "mv ", "cp ")

# A stdout redirect (`>` or `>>`) to a real file. Excludes, via lookarounds:
#   - stderr / fd redirects: 2>, 1>, &>   (digit or & immediately before `>`)
#   - fd duplications:       >&1, >&2     (`&` immediately after the spaces)
#   - the null sink:         > /dev/null
_REDIRECT_RE = re.compile(r"(?<![0-9&])>>?\s*(?!&)(?!/dev/null\b)\S")

# Start of a heredoc on a command line: `<<MARKER`, `<<'MARKER'`, `<<-"MARKER"`.
_HEREDOC_START_RE = re.compile(r"<<-?\s*(['\"]?)(\w+)\1")

# A single- or double-quoted span. The char classes match newlines too, so a
# multi-line quoted program (python -c '...') is blanked as one span.
_QUOTED_RE = re.compile(r"'[^']*'|\"[^\"]*\"")


def _strip_heredoc_bodies(command: str) -> str:
    """Drop heredoc BODY lines (between `<<MARKER` and the closing MARKER line).

    The command line that opens the heredoc is kept — its redirect
    (`cat <<'EOF' > file.py`) is the write signal. The body is file CONTENT,
    not commands; leaving it in produces false redirect matches on code like
    `if x > 0:`.
    """
    out: list[str] = []
    marker: str | None = None
    for line in command.split("\n"):
        if marker is not None:
            if line.strip() == marker:
                marker = None
            continue
        m = _HEREDOC_START_RE.search(line)
        if m:
            marker = m.group(2)
        out.append(line)
    return "\n".join(out)


def _command_segments(command: str) -> list[str]:
    """Top-level command segments, with quoted spans and heredoc bodies blanked.

    Heredoc bodies are dropped first, then quoted spans are replaced by ''
    BEFORE splitting, so a `&&` / `;` inside an awk/python program does not
    create a phantom segment and a `>=` inside quotes cannot read as a
    redirect. Splits on `&&`, `;`, and newlines — every segment is a command
    that executes, so each must be inspected (a write does not stop being a
    write because `&& pytest` follows it).
    """
    cleaned = _QUOTED_RE.sub("''", _strip_heredoc_bodies(command))
    return [seg.strip() for seg in re.split(r"&&|;|\n", cleaned) if seg.strip()]


def is_write_command(command: str) -> bool:
    """Check if a command modifies a file on disk (used to detect patch actions).

    Detects in-place editors (sed -i, tee, patch), file movers (mv/cp/dd/
    truncate), and stdout redirects (`>` / `>>`) to a real file — in ANY
    top-level segment of the command (`&&` / `;` / newline chains), with
    quoted spans and heredoc bodies excluded from inspection.

    Deliberately NOT classified as writes (false-positive classes that could
    mis-trigger SDLG at a non-write step — the first two from the original
    prefix-only heuristic, the last two measured on the pilot logs):
      - the submission command (echoes the sentinel, then cats patch.txt),
      - bare `echo` / `printf` with no redirect,
      - stderr-only redirects (`2>`, `&>`, `2>&1`), fd dups (`>&1`),
        and /dev/null sinks,
      - comparison operators inside quoted programs, e.g.
        `awk 'NR>=350 {print}' file.py` (2 occurrences in 2184 pilot
        actions were misread as writes by the unquoted-regex version),
      - heredoc BODY content (`python - <<'EOF' ... if x > 0 ... EOF`).

    And the converse false-negative class is closed: a real write hidden in a
    non-final segment (`sed -i ... && python -m pytest`,
    `git diff > patch.txt && cat patch.txt`) is detected — the last-segment-
    only version missed it, which in the SDLG arm would skip the genuine
    branch point.

    Known limitations (documented, accepted): a redirect of program output to
    a scratch file (`python repro.py > out.txt`) is treated as a write, and a
    programmatic write (`python -c "open('f','w')..."`) is undetectable from
    the command string. Callers that need precision should diff the tree.
    """
    cmd = command.strip()
    # The submission command is `echo <sentinel> && cat patch.txt` — not a write.
    if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in cmd:
        return False

    for seg in _command_segments(cmd):
        for prefix in _WRITE_PREFIXES:
            if seg.startswith(prefix):
                return True
        if _REDIRECT_RE.search(seg):
            return True

    return False


# Phase-specific system prompts (appended to the base system prompt)

SEARCH_PROMPT = """
## Current Phase: EXPLORATION

You are exploring the codebase to understand the bug. Your goal is to:
1. Find the relevant source files
2. Understand the code structure
3. Reproduce the issue
4. Identify the root cause

**You can only READ files and SEARCH the codebase.** You cannot modify any files yet.
Available commands: grep, find, cat, head, tail, ls, sed -n (view only), python -c (one-liners)

When you believe you understand the root cause and have a fix strategy, say:
STRATEGY: <one sentence describing your fix approach>

This will transition you to the implementation phase.
"""

PATCH_PROMPT = """
## Current Phase: IMPLEMENTATION

You now have write access. Implement your fix based on your investigation.
Available commands: All read commands plus sed -i, cat <<EOF>, patch, python scripts

Make targeted, minimal changes. Do not refactor unrelated code.
When your fix is complete, say:
DONE: <one sentence describing what you changed>

This will transition you to the verification phase.
"""

PATCH_PROMPT_WITH_STRATEGY = """
## Current Phase: IMPLEMENTATION

You have been assigned a specific fix strategy. You MUST implement EXACTLY this approach:

**YOUR ASSIGNED STRATEGY: {strategy}**

CRITICAL RULES:
- You MUST follow the strategy above — it specifies which file, function, and code to change.
- Do NOT simplify to just disabling a check or deleting a block. Implement the strategy as described.
- If the strategy says to add new logic, ADD new logic. If it says to modify an algorithm, MODIFY the algorithm.
- Your patch MUST be structurally different from a simple one-line disable/bypass.

**IMPORTANT: Write your code edit NOW using sed -i or cat <<EOF>. Do NOT spend more time exploring — you already have the information you need from the search phase. Your very next command should be a file edit.**

Available commands: sed -i, cat <<EOF>, patch, python scripts that write files

When your fix is complete, say:
DONE: <one sentence describing what you changed>
"""

PATCH_FORCE_WRITE_MSG = """You have spent too many steps reading files instead of writing your fix. You MUST write your code edit NOW.

Use one of these commands to make your edit:
- `sed -i 's/old/new/' /testbed/path/to/file.py`
- `cat <<'EOF' > /testbed/path/to/file.py` (for multi-line changes)

Do NOT run any more grep, cat, find, or ls commands. Write the edit immediately.
"""

VERIFY_PROMPT = """
## Current Phase: VERIFICATION

Review and test your changes:
1. Run `git diff` to review your changes
2. Run relevant tests to verify the fix
3. Test edge cases

When you are confident the fix is correct, create a patch and submit:
Step 1: cd /testbed && git diff -- <modified files> > patch.txt
Step 2: Verify patch.txt
Step 3: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /testbed/patch.txt
"""


def should_end_search(
    step: int,
    consecutive_low_relevance: int,
    min_search_steps: int,
    low_relevance_streak: int,
    max_search_steps: int,
) -> str | None:
    """Decide whether the SEARCH phase should end, and why.

    Pure decision function (no I/O) extracted from the orchestrator's search
    loop so the saturation / step-cap policy is testable without a GPU run.

    Returns:
        "saturated"  — at least `min_search_steps` taken AND `low_relevance_streak`
                       consecutive low-relevance steps (the normal exit).
        "step_limit" — the hard `max_search_steps` fallback cap was reached
                       (guards against looping on blocked, unscored commands).
        None         — keep searching.

    Saturation is checked first so a well-behaved run exits via relevance, not
    the cap. The cap is a fallback only.
    """
    if step >= min_search_steps and consecutive_low_relevance >= low_relevance_streak:
        return "saturated"
    if step >= max_search_steps:
        return "step_limit"
    return None


def detect_phase_transition(thought: str, action: str, current_phase: Phase) -> Phase | None:
    """Detect if the agent is signaling a PATCH→VERIFY transition.

    Live contract: the orchestrator calls this only with
    `current_phase=Phase.PATCH` (`_step_patch`). SEARCH→PATCH is decided by
    `should_end_search` (relevance saturation / step cap), NOT here — an
    earlier version carried a dead SEARCH branch keyed on "STRATEGY:"
    phrases, which invited the false belief that declaring a strategy ends
    the search phase (it ends it only indirectly, via low-relevance
    saturation; see the module docstring). VERIFY has no outgoing
    transition — it ends with submission.

    Returns the new phase, or None if no transition.
    """
    thought_lower = thought.lower()

    if current_phase == Phase.PATCH:
        # Check if agent is done patching
        if "DONE:" in thought or "done:" in thought_lower:
            return Phase.VERIFY
        # Detect implicit completion
        if any(phrase in thought_lower for phrase in [
            "now let me verify",
            "let me test",
            "let me run the test",
            "let me check if",
            "git diff",
        ]) and is_write_command(action) is False:
            return Phase.VERIFY

    return None
