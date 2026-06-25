"""Behavioral (execution-grounded) meaning equivalence for code-agent patches.

This implements the paper's *primary* meaning relation (Methods, contribution #2):
two final patches mean the same thing iff they make the repository **behave** the
same way -- i.e. they induce the same vector of per-test outcomes over the
instance's ``FAIL_TO_PASS`` + ``PASS_TO_PASS`` tests. Because our outputs are
programs we run, "same meaning" is *checkable* and deterministic; it needs no
learned text-similarity and no model judgment, unlike the NLI / sentence-embedding
clustering of text-only semantic-entropy work (Farquhar 2024; Kuhn 2023).

The signature is obtained essentially for free: the SWE-bench harness already runs
these tests to decide ``resolved`` -- we simply keep the *full per-test vector* it
reports (``tests_status``) instead of collapsing it to the single ``resolved`` bit.

Signature atoms (behaviorally distinct outcomes kept as their own classes):
  ``("NO_PATCH",)``     -- empty / ``None`` patch: no edit was attempted.
  ``("APPLY_FAILED",)`` -- patch does not apply: it breaks the repo before any
                           target test runs, a genuinely different outcome from a
                           patch that applies and fails tests.
Otherwise the signature is ``frozenset{(test_name, "P"|"F"), ...}`` over the
instance's F2P + P2P tests (``"P"`` = passed under this patch, ``"F"`` = failed).
Each test appears exactly once (success xor failure), so for a fixed instance the
frozensets are directly comparable across patches without a separate universe
projection.

Discrete semantic entropy uses the plug-in estimator with an optional Miller--Madow
bias correction ``+ (K-1)/(2M)`` nats (the paper's estimator section), since the
plug-in is biased low at the small ``M`` used per prefix.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Hashable, Iterable, Sequence

NO_PATCH: tuple = ("NO_PATCH",)
APPLY_FAILED: tuple = ("APPLY_FAILED",)

# Categories whose per-test pass/fail defines the instance's behavioral spec.
# FAIL_TO_FAIL / PASS_TO_FAIL are excluded: they are not part of the task's
# acceptance vector (resolved is decided on F2P success + P2P success).
_SPEC_CATEGORIES = ("FAIL_TO_PASS", "PASS_TO_PASS")


def _unwrap(report: dict, instance_id: str | None) -> dict:
    """Accept either the harness's ``{instance_id: {...}}`` wrapper or the inner
    record directly, and return the inner record."""
    if instance_id is not None and instance_id in report:
        return report[instance_id]
    # Single-key wrapper (unknown id) -> unwrap; else assume already-inner.
    if len(report) == 1 and "tests_status" not in report:
        return next(iter(report.values()))
    return report


def behavioral_signature(report: dict, instance_id: str | None = None) -> Hashable:
    """Behavioral meaning class for one patch, from its SWE-bench eval report.

    Returns a hashable signature: ``NO_PATCH`` / ``APPLY_FAILED`` atoms, or a
    ``frozenset`` of ``(test_name, "P"/"F")`` over the F2P + P2P tests. Two patches
    share a class iff their signatures are equal.
    """
    rec = _unwrap(report, instance_id)

    if rec.get("patch_is_None") or not rec.get("patch_exists", True):
        return NO_PATCH
    if not rec.get("patch_successfully_applied", True):
        return APPLY_FAILED

    status = rec.get("tests_status", {})
    items: list[tuple[str, str]] = []
    for cat in _SPEC_CATEGORIES:
        cat_status = status.get(cat, {})
        for t in cat_status.get("success", []):
            items.append((t, "P"))
        for t in cat_status.get("failure", []):
            items.append((t, "F"))
    if not items:
        # Applied patch but no spec tests reported -> cannot be distinguished
        # behaviorally; treat as its own degenerate class rather than guessing.
        return ("APPLIED_NO_TESTS",)
    return frozenset(items)


def is_resolved_signature(sig: Hashable) -> bool:
    """True iff the behavioral signature corresponds to a fully-resolved patch
    (every F2P and P2P test passed). Useful for set-valued / coverage analysis."""
    if not isinstance(sig, frozenset):
        return False
    return all(s == "P" for (_t, s) in sig)


# --------------------------------------------------------------------------- #
# Structural signature (secondary lens) -- files / hunks the patch touches.
# Deterministic, parsed (no model). Separates two *approaches* that behave the
# same (e.g. caller- vs callee-side fix passing identical tests).
# --------------------------------------------------------------------------- #

_DIFF_FILE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)
_HUNK = re.compile(r"^@@ .*@@(.*)$", re.MULTILINE)


def structural_signature(patch: str) -> Hashable:
    """Files + hunk-context headers a unified-diff patch edits. Coarser than AST
    but deterministic and model-free; used as the paper's secondary refinement."""
    if not patch or not patch.strip():
        return NO_PATCH
    files = tuple(sorted(set(_DIFF_FILE.findall(patch))))
    # Hunk @@ ... @@ trailing context (often the enclosing def/class) per file.
    hunks = tuple(sorted(h.strip() for h in _HUNK.findall(patch) if h.strip()))
    return (files, hunks)


# --------------------------------------------------------------------------- #
# Clustering -> discrete semantic entropy
# --------------------------------------------------------------------------- #

def cluster_counts(signatures: Sequence[Hashable]) -> list[int]:
    """Cluster sizes (descending) for a list of per-patch signatures."""
    return sorted(Counter(signatures).values(), reverse=True)


def discrete_entropy(
    counts: Iterable[int],
    miller_madow: bool = True,
) -> dict:
    """Plug-in discrete entropy of a cluster-size distribution, in nats.

    ``H = -sum (n_c/M) log(n_c/M)``; with ``miller_madow`` add the first-order
    bias correction ``(K-1)/(2M)`` (K = #classes, M = #samples). ``H_norm`` is
    divided by ``log M`` (the achievable ceiling), reported only for M > 1.
    """
    counts = [c for c in counts if c > 0]
    M = sum(counts)
    K = len(counts)
    if M == 0:
        return {"H": 0.0, "H_mm": 0.0, "H_norm": 0.0, "K": 0, "M": 0}
    H = -sum((c / M) * math.log(c / M) for c in counts)
    mm = (K - 1) / (2 * M) if miller_madow else 0.0
    H_mm = H + mm
    ceil = math.log(M) if M > 1 else 0.0
    return {
        "H": round(H, 6),
        "H_mm": round(H_mm, 6),
        "H_norm": round(H_mm / ceil, 6) if ceil > 0 else 0.0,
        "K": K,
        "M": M,
        "miller_madow": round(mm, 6),
    }


def behavioral_entropy(
    reports: Sequence[dict],
    instance_id: str | None = None,
    miller_madow: bool = True,
) -> dict:
    """End-to-end: per-patch eval reports -> behavioral signatures -> entropy.

    Returns the ``discrete_entropy`` dict augmented with the signatures and the
    number of fully-resolved classes (for set-valued-evidence analysis).
    """
    sigs = [behavioral_signature(r, instance_id) for r in reports]
    counts = cluster_counts(sigs)
    out = discrete_entropy(counts, miller_madow=miller_madow)
    out["n_distinct_classes"] = out["K"]
    out["n_resolved_classes"] = sum(
        1 for s in set(sigs) if is_resolved_signature(s)
    )
    out["clusters"] = counts
    return out
