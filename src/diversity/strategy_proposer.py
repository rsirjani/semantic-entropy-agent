"""Strategy-level diversity via explicit proposal prompting.

Instead of token-level perturbation (SDLG) which doesn't produce diversity
for multi-step agent code generation, we explicitly ask the LLM to propose
multiple different fix strategies after the search phase.

Inspired by Tree of Thoughts (Yao 2023) "propose prompt" which asks the LLM
to generate K different approaches, combined with semantic clustering to
deduplicate semantically equivalent proposals.
"""

import logging
import re

import litellm

logger = logging.getLogger(__name__)


SEARCH_REPORT_PROMPT = """\
An AI coding agent investigated a software bug. Below is its exploration history \
with the actual code it found.

Write a structured report with these sections:
1. ROOT CAUSE: What is the bug and why does it happen? Trace the execution path.
2. RELEVANT CODE: For each relevant location, include the actual code snippet \
(verbatim from the observations below). Tag each with file path, function name, \
and line numbers.
3. FIX POINTS: What are the different points in the code where a fix COULD be \
applied? (e.g., the validation layer, the processing layer, a caller, a helper \
function, etc.)

IMPORTANT: Include the actual code from the observations. Do NOT paraphrase code — \
copy it exactly. The strategies proposed later need to reference real lines.

Exploration history:
{history}

Structured report (include verbatim code snippets from the observations):"""


STRATEGY_PROPOSAL_PROMPT = """\
A software bug needs to be fixed. Here is the problem description and investigation findings.

## Problem
{problem_statement}

## Investigation Findings
{search_report}

## Task
Propose exactly {n} FUNDAMENTALLY DIFFERENT code-level strategies to fix this bug.

CRITICAL RULES:
- Each strategy MUST take a COMPLETELY DIFFERENT APPROACH — not just a different implementation of the same idea
- Each strategy MUST modify DIFFERENT lines, functions, or files
- Each strategy MUST produce a structurally different patch (different diff)
- Do NOT propose strategies that all end up changing the same line of code
- Be SPECIFIC: name the exact function, file, and line range each strategy would change
- Describe the actual code transformation (what gets added/removed/replaced)

Think about fundamentally different approaches:
- Fix at different levels of the call stack (caller vs callee vs helper)
- Fix in different files or modules
- Use different mechanisms (validation vs normalization vs delegation)
- Address different root causes (input handling vs processing logic vs output)
{rejection_clause}
Format your response EXACTLY as:
STRATEGY 1: [Which file and function to modify] — [What specific code change to make]
STRATEGY 2: [Which file and function to modify] — [What specific code change to make]
...and so on for all {n} strategies."""


class StrategyProposer:
    """Propose multiple fix strategies after search phase."""

    def __init__(
        self,
        model_name: str = "openai/qwen3-coder",
        model_kwargs: dict | None = None,
        n_strategies: int = 5,
    ):
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}
        self.n_strategies = n_strategies

    def build_search_report(
        self,
        messages: list[dict],
        relevance_scores: list[dict] | None = None,
        relevance_threshold: float = 0.4,
    ) -> str:
        """Summarize the search phase findings into a concise report.

        Takes the full message history from the search phase and asks the LLM
        to extract the key findings: relevant files, root cause, reproduction.

        If relevance_scores are provided, only includes steps that scored above
        the relevance threshold. This prunes noise from low-value search steps
        (e.g., the trailing low-relevance steps that triggered saturation).
        """
        # Build step-indexed relevance lookup
        # relevance_scores[i] corresponds to the (i+1)th assistant step
        relevant_steps = None
        if relevance_scores:
            relevant_steps = {
                s["step"] for s in relevance_scores if s["relevance"] >= relevance_threshold
            }
            n_total = len(relevance_scores)
            n_relevant = len(relevant_steps)
            logger.info(
                f"Search report: filtering to {n_relevant}/{n_total} relevant steps "
                f"(threshold={relevance_threshold})"
            )

        # Build a condensed history from assistant messages, preserving
        # actual code from observations (not just truncated summaries)
        steps = []
        assistant_step = 0
        for msg in messages:
            if msg.get("role") == "assistant":
                assistant_step += 1
                # If we have relevance data, skip low-relevance steps
                if relevant_steps is not None and assistant_step not in relevant_steps:
                    continue
                content = msg.get("content", "")[:500]
                if content.strip():
                    steps.append(f"[Agent]: {content}")
            elif msg.get("role") == "user" and "<output>" in msg.get("content", ""):
                # Include command outputs — these contain the actual code
                # Use a larger limit (1500 chars) to preserve real code snippets
                if relevant_steps is not None and assistant_step not in relevant_steps:
                    continue
                content = msg.get("content", "")[:1500]
                steps.append(f"[Code output]: {content}")

        # Take last 15 steps to stay within context limits
        recent = steps[-15:]
        history = "\n---\n".join(recent)

        if not history.strip():
            return "(No search steps recorded)"

        prompt = SEARCH_REPORT_PROMPT.format(history=history[:8000])

        try:
            kwargs = {k: v for k, v in self.model_kwargs.items() if k != "temperature"}
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1500,
                **kwargs,
            )
            report = (response.choices[0].message.content or "").strip()
            logger.info(f"Search report: {report[:200]}...")
            return report[:4000]
        except Exception as e:
            logger.warning(f"Failed to build search report: {e}")
            # Fallback: use the last few assistant thoughts directly
            fallback = "\n".join(s[:200] for s in recent[-5:])
            return fallback[:2000]

    def propose(
        self, search_report: str, problem_statement: str, n: int | None = None,
    ) -> list[str]:
        """Ask LLM to propose N different fix strategies with rejection conditioning.

        Uses iterative rejection: after generating strategies, if clustering shows
        they converged, re-prompts with explicit rejection of already-seen approaches.
        This forces the model to explore genuinely different fix strategies.

        Returns a list of strategy descriptions (one sentence each).
        """
        n = n or self.n_strategies

        # First pass: generate all N strategies at once with rejection clause
        rejection_clause = (
            "\nIMPORTANT: Before writing each strategy, explicitly state what makes it "
            "FUNDAMENTALLY DIFFERENT from all previous strategies. If you cannot articulate "
            "the difference, you are proposing the same fix with surface variation.\n"
        )

        prompt = STRATEGY_PROPOSAL_PROMPT.format(
            problem_statement=problem_statement[:2000],
            search_report=search_report[:4000],
            n=n,
            rejection_clause=rejection_clause,
        )

        try:
            kwargs = {k: v for k, v in self.model_kwargs.items() if k != "temperature"}
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=1200,
                **kwargs,
            )
            raw = (response.choices[0].message.content or "").strip()
            strategies = self._parse_strategies(raw, n)
            logger.info(f"Proposed {len(strategies)} strategies (first pass)")
            for i, s in enumerate(strategies):
                logger.info(f"  Strategy {i+1}: {s[:100]}")

            # If we got fewer than requested, try a second pass with explicit rejection
            if len(strategies) < n:
                more = self._propose_with_rejection(
                    search_report, problem_statement,
                    existing_strategies=strategies,
                    n_more=n - len(strategies),
                )
                strategies.extend(more)

            return strategies
        except Exception as e:
            logger.error(f"Strategy proposal failed: {e}")
            return [f"Fix the bug as described in the problem statement."]

    def _propose_with_rejection(
        self,
        search_report: str,
        problem_statement: str,
        existing_strategies: list[str],
        n_more: int,
    ) -> list[str]:
        """Generate additional strategies explicitly rejecting existing ones."""
        existing_list = "\n".join(
            f"- ALREADY PROPOSED (DO NOT REPEAT): {s[:200]}" for s in existing_strategies
        )

        prompt = f"""\
A software bug needs to be fixed. Previous strategies have already been proposed.
You MUST propose {n_more} NEW strategies that are FUNDAMENTALLY DIFFERENT from all existing ones.

## Problem
{problem_statement[:2000]}

## Investigation Findings
{search_report[:3000]}

## Already Proposed Strategies (DO NOT repeat or rephrase these)
{existing_list}

## Task
Propose {n_more} NEW strategies. Each MUST:
- Target DIFFERENT code locations than any existing strategy
- Use a DIFFERENT mechanism (not just rewording the same fix)
- Be specific about file, function, and code change

Format: STRATEGY N: [file and function] — [specific code change]"""

        try:
            kwargs = {k: v for k, v in self.model_kwargs.items() if k != "temperature"}
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=800,
                **kwargs,
            )
            raw = (response.choices[0].message.content or "").strip()
            new_strategies = self._parse_strategies(raw, n_more)
            logger.info(f"Rejection pass: proposed {len(new_strategies)} more strategies")
            return new_strategies
        except Exception as e:
            logger.warning(f"Rejection proposal failed: {e}")
            return []

    def _parse_strategies(self, raw: str, expected_n: int) -> list[str]:
        """Parse STRATEGY N: ... lines from the LLM response."""
        strategies = []

        # Match "STRATEGY N:" or "STRATEGY N (Category):" pattern
        pattern = r"STRATEGY\s+\d+\s*(?:\([^)]*\))?\s*:\s*(.+?)(?=STRATEGY\s+\d+\s*(?:\([^)]*\))?\s*:|$)"
        matches = re.findall(pattern, raw, re.DOTALL | re.IGNORECASE)

        for match in matches:
            text = match.strip()
            # Clean up: remove trailing whitespace, limit length
            text = " ".join(text.split())  # normalize whitespace
            if text and len(text) > 10:
                strategies.append(text[:500])

        # If regex didn't work, try splitting by numbered lines
        if not strategies:
            for line in raw.split("\n"):
                line = line.strip()
                # Match "1.", "2.", etc.
                match = re.match(r"^\d+[.)]\s*(.+)", line)
                if match:
                    text = match.group(1).strip()
                    if text and len(text) > 10:
                        strategies.append(text[:500])

        # If still nothing, use the whole response as a single strategy
        if not strategies:
            logger.warning("Could not parse strategies, using raw response")
            strategies = [raw[:500]]

        return strategies[:expected_n]
