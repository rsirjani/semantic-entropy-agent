"""Intent extraction from agent responses via LLM summarization.

Per the proposal: at each branching decision, summarize the FULL trajectory
(steps 0 through current step) for each candidate, then compute bidirectional
entailment on those summaries.

Key insight: we don't summarize candidate responses in isolation. We summarize
"what is the overall approach of this trajectory if it takes this candidate as
the next step?" Early steps may search in different orders but pursue the same
approach — the trajectory-level summary captures that. Later steps may pursue
genuinely different fix strategies — the summary captures that too.

Two use cases:
1. Branching: for each candidate, summarize (history + candidate) as a full trajectory
2. Pruning: summarize each active trajectory's full history, prune duplicates
"""

import re
import logging
from typing import Callable

import litellm

logger = logging.getLogger(__name__)


# --- Prompt templates ---

TRAJECTORY_WITH_CANDIDATE_PROMPT = """\
An AI coding agent is fixing a bug. Below is its reasoning history so far, \
followed by a proposed next step. Summarize the overall approach and strategy \
in exactly one sentence — what fix is it pursuing and how?

History:
{history}

Proposed next step:
{candidate}

One-sentence summary of the overall approach:"""

TRAJECTORY_INTENT_PROMPT = """\
An AI coding agent is fixing a bug. Below is its reasoning history. \
Summarize the overall approach and current strategy in exactly one sentence — \
what fix is it pursuing and how?

{trajectory_summary}

One-sentence summary of the overall approach:"""


# --- Helpers ---

def extract_thought(response: str) -> str:
    """Extract the THOUGHT section from a THOUGHT/ACTION response."""
    match = re.search(
        r"THOUGHT:\s*\n?(.*?)(?=\n(?:ACTION:|```mswea_bash_command)|\Z)",
        response, re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    match = re.search(r"^(.*?)(?=```)", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def extract_intent_heuristic(response: str) -> str:
    """Fallback: extract the first sentence of THOUGHT. Zero LLM cost."""
    thought = extract_thought(response)
    if not thought:
        return "No clear intent."
    sentences = re.split(r'(?<=[.!?])\s+', thought)
    if sentences:
        first = sentences[0].strip()
        if len(first) < 20 and len(sentences) > 1:
            first = sentences[0].strip() + " " + sentences[1].strip()
        return first
    return thought[:200]


def _build_history_summary(messages: list[dict], max_steps: int = 10) -> str:
    """Build a condensed summary of a trajectory's reasoning steps.

    Extracts the THOUGHT from each assistant message (up to last max_steps),
    formatted as a numbered list.
    """
    thoughts = []
    for msg in messages:
        if msg.get("role") == "assistant":
            thought = extract_thought(msg.get("content", ""))
            if thought:
                thoughts.append(thought)

    recent = thoughts[-max_steps:]
    if not recent:
        return "(no steps yet)"
    lines = [f"Step {i+1}: {t[:300]}" for i, t in enumerate(recent)]
    return "\n".join(lines)


# --- Main class ---

class IntentExtractor:
    """Extract intent summaries via LLM calls (per the proposal).

    For branching: summarize (full history + each candidate) as a trajectory.
    For pruning: summarize each active trajectory's full history.
    """

    def __init__(
        self,
        model_name: str = "openai/qwen3-coder",
        model_kwargs: dict | None = None,
        method: str = "llm",
    ):
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}
        self.method = method

    def _llm_summarize(self, prompt: str) -> str:
        """Call the LLM with a short summarization prompt."""
        messages = [{"role": "user", "content": prompt}]
        kwargs = {k: v for k, v in self.model_kwargs.items() if k != "temperature"}
        response = litellm.completion(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=100,
            **kwargs,
        )
        return (response.choices[0].message.content or "").strip()[:200]

    # --- For branching: summarize (history + candidate) as full trajectory ---

    def extract_with_history(
        self, candidate_response: str, history_messages: list[dict]
    ) -> str:
        """Summarize the full trajectory intent IF this candidate is taken.

        This is the key method for branching decisions. For each candidate:
        - history_messages = steps 0 through S-1 (shared across all candidates)
        - candidate_response = the proposed step S (different per candidate)
        - Summary = one sentence describing the overall approach of this trajectory

        Two candidates that search in different orders but pursue the same
        approach will get the SAME summary → same cluster → no branch.
        Two candidates that pursue genuinely different fix strategies will
        get DIFFERENT summaries → different clusters → branch.
        """
        if self.method == "heuristic":
            return extract_intent_heuristic(candidate_response)

        history = _build_history_summary(history_messages)
        candidate_thought = extract_thought(candidate_response)[:500]

        prompt = TRAJECTORY_WITH_CANDIDATE_PROMPT.format(
            history=history,
            candidate=candidate_thought,
        )
        try:
            return self._llm_summarize(prompt)
        except Exception as e:
            logger.warning(f"LLM intent extraction failed: {e}, using heuristic")
            return extract_intent_heuristic(candidate_response)

    def extract_batch_with_history(
        self, candidates: list[str], history_messages: list[dict]
    ) -> list[str]:
        """Summarize (history + each candidate) for N candidates.

        All candidates share the same history (they diverge at the current step).
        """
        return [
            self.extract_with_history(c, history_messages) for c in candidates
        ]

    # --- For pruning: summarize a trajectory's full history ---

    def extract_trajectory_intent(self, messages: list[dict]) -> str:
        """Summarize a trajectory's overall approach from its full message history.

        Used for cross-trajectory pruning: trajectories with the same overall
        approach (bidirectional entailment on summaries) are redundant.
        """
        summary = _build_history_summary(messages)
        if summary == "(no steps yet)":
            return "No trajectory steps yet."

        if self.method == "heuristic":
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    return extract_intent_heuristic(msg.get("content", ""))
            return "Unknown approach."

        prompt = TRAJECTORY_INTENT_PROMPT.format(trajectory_summary=summary)
        try:
            return self._llm_summarize(prompt)
        except Exception as e:
            logger.warning(f"LLM trajectory summarization failed: {e}")
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    return extract_intent_heuristic(msg.get("content", ""))
            return "Unknown approach."
