"""Relevance scoring for search-phase branching.

Scores how relevant each search finding is to the problem statement.
Uses LLM to summarize what the search result shows, THEN uses DeBERTa
NLI to check entailment between the summary and the problem.

Raw search output (code, grep results) doesn't match the format of a
bug description, so DeBERTa can't compare them directly. The LLM summary
bridges that gap.
"""

import logging

import litellm

from src.diversity.nli import NLIModel

logger = logging.getLogger(__name__)

SUMMARIZE_FINDING_PROMPT = """\
An AI agent is debugging a software issue. It ran a search command and got this result.
In one sentence, what did this search find and why might it be relevant?

Agent's reasoning: {thought}

Search result (truncated):
{observation}

One-sentence summary of what was found:"""


class RelevanceScorer:
    """Score how relevant agent findings are to the problem statement."""

    def __init__(
        self,
        nli: NLIModel,
        threshold: float = 0.5,
        model_name: str = "openai/qwen3-coder",
        model_kwargs: dict | None = None,
    ):
        self.nli = nli
        self.threshold = threshold
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}

    def _summarize_finding(self, thought: str, observation: str) -> str:
        """Use LLM to summarize what a search step found."""
        prompt = SUMMARIZE_FINDING_PROMPT.format(
            thought=thought[:300],
            observation=observation[:500],
        )
        try:
            kwargs = {k: v for k, v in self.model_kwargs.items() if k != "temperature"}
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=80,
                **kwargs,
            )
            return (response.choices[0].message.content or "").strip()[:200]
        except Exception as e:
            logger.debug(f"Summarization failed: {e}")
            # Fallback: use the thought itself
            return thought[:200]

    def score(self, finding_summary: str, problem_statement: str) -> float:
        """Score relevance using LLM judgment.

        DeBERTa NLI doesn't work for code-finding-to-bug relevance because:
        - NLI is trained on natural language sentence pairs, not code/bug pairs
        - Without perfectly matched context, entailment is always near-zero
        - With too much shared context, entailment is always near-one

        Instead, use a cheap LLM call: ask the model to rate relevance 0-10,
        normalize to [0, 1]. This leverages the LLM's code understanding.
        """
        prompt = (
            f"Rate how relevant this investigation finding is to the bug below.\n"
            f"Bug: {problem_statement[:300]}\n"
            f"Finding: {finding_summary}\n"
            f"Rate relevance from 0 (completely irrelevant) to 10 (directly identifies the bug).\n"
            f"Respond with ONLY a single number 0-10."
        )
        try:
            kwargs = {k: v for k, v in self.model_kwargs.items() if k != "temperature"}
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5,
                **kwargs,
            )
            text = (response.choices[0].message.content or "").strip()
            # Extract the first number from the response
            import re
            match = re.search(r"(\d+)", text)
            if match:
                score = int(match.group(1))
                return min(score, 10) / 10.0
            return 0.0
        except Exception as e:
            logger.debug(f"LLM relevance scoring failed: {e}")
            return 0.0

    def score_trajectory_step(
        self,
        thought: str,
        observation: str,
        problem_statement: str,
    ) -> dict:
        """Score a complete agent step.

        1. Summarize what the search found (LLM call)
        2. Check entailment between summary and problem (DeBERTa)
        """
        # Summarize the finding in natural language
        summary = self._summarize_finding(thought, observation)

        # Now DeBERTa can compare: "Found the Permutation constructor..."
        # vs "Permutation constructor fails with non-disjoint cycles"
        relevance = self.score(summary, problem_statement)

        return {
            "summary": summary,
            "relevance": relevance,
            "is_relevant": relevance > self.threshold,
        }

    def has_strategy(self, thought: str) -> bool:
        """Check if the agent's thought contains a fix strategy."""
        thought_lower = thought.lower()
        strategy_signals = [
            "the fix should",
            "the fix is to",
            "i think the bug is",
            "the root cause is",
            "the issue is that",
            "to fix this we need to",
            "the solution is to",
            "i'll modify",
            "i need to modify",
            "i need to change",
            "let me fix",
            "strategy:",
        ]
        return any(signal in thought_lower for signal in strategy_signals)
