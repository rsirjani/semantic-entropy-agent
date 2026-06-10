"""Regression guard: clustering must NOT prepend a shared context prefix.

2026-06-10 Phase A run-1 instrument defect: prepending problem_statement[:500]
to BOTH sides of every entailment pair saturates DeBERTa (all pairs >=0.94
entailment on 5 structurally distinct strategies -> 1 cluster -> entropy 0 ->
the gate never fires). Empirical A/B on the run-1 artifacts:
scripts/diagnose_context_saturation.py. These tests pin the fix at every
clustering call site and the clusterer's empty-context behavior.
"""

import os
import re

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def _source(rel):
    with open(os.path.join(SRC, rel), "r", encoding="utf-8") as f:
        return f.read()


def test_no_problem_statement_context_at_clustering_call_sites():
    """No analyze()/cluster() call may condition on the problem statement."""
    for rel in ("agent/phased_orchestrator.py", "agent/branching_orchestrator.py"):
        src = _source(rel)
        # Any analyze(/cluster( call whose argument block mentions
        # problem_statement within the context kwarg is the defect returning.
        for m in re.finditer(r"\.(analyze|cluster)\(", src):
            window = src[m.start():m.start() + 400]
            ctx = re.search(r"context\s*=\s*([^,\)]+)", window)
            if ctx:
                assert "problem_statement" not in ctx.group(1), (
                    f"{rel}: clustering context conditions on problem_statement "
                    f"again — this saturates NLI entailment (see "
                    f"tests/test_clustering_context.py docstring)")


def test_pairwise_logging_context_matches_clusterer():
    """The orchestrator's logged pairwise NLI must use the same (empty) context."""
    src = _source("agent/phased_orchestrator.py")
    assert 'cluster_context = ""' in src, (
        "strategy-path cluster_context is no longer empty; the logged pairwise "
        "decisions would desync from the clusterer or re-introduce saturation")


def test_with_context_empty_is_identity():
    from src.diversity.clustering import SemanticClusterer

    c = SemanticClusterer.__new__(SemanticClusterer)  # no NLI needed
    assert c._with_context("modify the parser", "") == "modify the parser"
    assert c._with_context("modify the parser", "ctx") == "ctx modify the parser"
