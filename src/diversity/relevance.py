"""Relevance scoring for SEARCH-phase saturation detection.

Scores how relevant each search finding is to the problem statement. Two
steps: (1) an LLM call summarizes what the search result shows in one
sentence; (2) the summary is scored against the problem statement by ONE of
two backends, selected by `use_nli`:

  - use_nli=False (the default AND the checked-in config,
    `relevance_use_nli: false`): a temperature-0 LLM call rates relevance
    0-10, normalized to [0, 1]. This is the LIVE configuration — early
    experiments showed DeBERTa entailment is the wrong construct here
    (a finding can be highly relevant without being ENTAILED by the bug
    report; entailment != topical relevance).
  - use_nli=True: P(entailment | premise=problem, hypothesis=summary) under
    DeBERTa-MNLI. Kept as an ablation backend; the LLM summary bridges the
    format gap (raw grep/code output is out of DeBERTa's domain).

Failure semantics (symmetric across arms — every arm runs the same SEARCH
machinery, so none of this is a treatment/control confound): a failed
summary call falls back to the raw thought text; a failed scoring call
scores 0.0, which counts toward the low-relevance saturation streak.
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
        use_nli: bool = False,
    ):
        self.nli = nli
        self.threshold = threshold
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}
        self.use_nli = use_nli

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
        """Score relevance — dispatches to NLI or LLM based on use_nli flag."""
        if self.use_nli:
            return self._score_nli(finding_summary, problem_statement)
        return self._score_llm(finding_summary, problem_statement)

    def _score_nli(self, finding_summary: str, problem_statement: str) -> float:
        """Score relevance using DeBERTa NLI entailment.

        r(o_i) = P(entailment | premise=problem, hypothesis=summary)

        Works because the summary is already natural language (from Step 1),
        so the domain mismatch with DeBERTa is resolved.
        """
        try:
            result = self.nli.classify(
                premise=problem_statement[:500],
                hypothesis=finding_summary[:200],
            )
            score = result["entailment"]
            logger.debug(
                f"NLI relevance: ent={score:.3f} "
                f"neu={result['neutral']:.3f} con={result['contradiction']:.3f}"
            )
            return score
        except Exception as e:
            logger.debug(f"NLI relevance scoring failed: {e}, falling back to LLM")
            return self._score_llm(finding_summary, problem_statement)

    def _score_llm(self, finding_summary: str, problem_statement: str) -> float:
        """Score relevance using LLM judgment.

        Uses a cheap LLM call: ask the model to rate relevance 0-10,
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
