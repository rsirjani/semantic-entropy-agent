"""Phase definitions for the phased branching agent.

Three phases with different tool access and branching strategies:

Phase 1 — SEARCH: Read-only exploration. Agent searches the codebase to
    understand the bug. Cannot modify files. Branches on relevance of
    findings to the problem statement.

Phase 2 — PATCH: Write access. Agent implements a fix. SDLG operates here
    to produce diverse patches. Branches on semantic diversity of fixes.

Phase 3 — VERIFY: Read-only again. Agent runs tests and reviews changes.
    No branching — just validate the patch.
"""

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

    Uses prefix matching — if any allowed prefix matches the start
    of the command (after stripping cd prefixes), it's allowed.
    """
    # Strip common prefixes like "cd /testbed && "
    cmd = command.strip()
    if "&&" in cmd:
        # Check the last command in the chain (the actual action)
        cmd = cmd.split("&&")[-1].strip()

    allowed = PHASE_ALLOWED_COMMANDS[phase]
    for prefix in allowed:
        if cmd.startswith(prefix):
            return True

    # Special case: submission is only allowed in VERIFY phase
    if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command:
        return phase == Phase.VERIFY

    return False


def is_write_command(command: str) -> bool:
    """Check if a command modifies files (used to detect patch actions)."""
    cmd = command.strip()
    if "&&" in cmd:
        cmd = cmd.split("&&")[-1].strip()

    write_prefixes = {"sed -i", "patch", "cat <<", "tee ", "mv ", "cp ", "echo "}
    # Also catch redirects
    if ">" in cmd and ">>" not in cmd and ">/dev/null" not in cmd:
        return True
    if ">>" in cmd:
        return True

    for prefix in write_prefixes:
        if cmd.startswith(prefix):
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


def detect_phase_transition(thought: str, action: str, current_phase: Phase) -> Phase | None:
    """Detect if the agent is signaling a phase transition.

    Returns the new phase, or None if no transition.
    """
    thought_lower = thought.lower()

    if current_phase == Phase.SEARCH:
        # Check if agent is declaring a strategy
        if "STRATEGY:" in thought or "strategy:" in thought_lower:
            return Phase.PATCH
        # Also detect implicit strategy formation
        if any(phrase in thought_lower for phrase in [
            "i think the fix",
            "the fix should",
            "let me fix",
            "let me modify",
            "i'll change",
            "i need to change",
            "the solution is",
            "to fix this",
        ]):
            return Phase.PATCH

    elif current_phase == Phase.PATCH:
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

    elif current_phase == Phase.VERIFY:
        # No transition from verify — it ends with submission
        pass

    return None
