"""Publication metrics: unbiased pass@k + INDEPENDENT patch diversity.

Pure, dependency-light functions (numpy + stdlib only) so they are trivially
unit-testable without GPU/Docker. Two families:

1. Coverage — `pass_at_k` (the unbiased Chen et al. 2021 estimator). For the
   matched-k headline we report diverse-pass@k on BOTH arms at the same per-
   instance k.

2. Diversity measured INDEPENDENTLY of the branching signal (rubric R4.2): we do
   NOT reuse the DeBERTa-NLI clustering that *decided* branching (that would be
   circular). Instead we compare the FINAL patches structurally — normalized
   added/removed code lines — via exact-signature distinct counts and graded
   pairwise edit distance (difflib). No model, no NLI.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Coverage: unbiased pass@k (Chen et al. 2021, "Evaluating LLMs Trained on Code")
# --------------------------------------------------------------------------- #

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of pass@k given n samples of which c passed.

    pass@k = 1 - C(n-c, k) / C(n, k), evaluated in the numerically stable product
    form. Requires k <= n. With n == k this reduces to 1.0 iff c >= 1 (the plain
    "any of the k passed"), which is exactly diverse-pass@k for a matched-k set.
    """
    if k <= 0:
        return 0.0
    if k > n:
        raise ValueError(f"pass_at_k requires k <= n (got k={k}, n={n})")
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    # 1 - prod_{i=n-c+1}^{n} (1 - k/i)
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def diverse_pass_at_k(outcomes: Sequence[bool], k: int | None = None) -> float:
    """diverse-pass@k from a per-sample boolean outcome vector.

    k defaults to len(outcomes) (the matched-k "did any of the produced
    trajectories pass"). Pass an explicit k < n to subsample-estimate.
    """
    n = len(outcomes)
    c = int(sum(bool(o) for o in outcomes))
    return pass_at_k(n, c, n if k is None else k)


# --------------------------------------------------------------------------- #
# Independent diversity over final patches (NOT the branching NLI)
# --------------------------------------------------------------------------- #

_DIFF_NOISE = re.compile(r"^(diff --git |index |--- |\+\+\+ |@@ |new file |deleted file |"
                         r"old mode |new mode |similarity |rename |Binary files )")


def normalize_patch(patch: str) -> str:
    """Reduce a unified diff to its semantic content for structural comparison.

    Keeps only the added/removed *code* lines (the `+`/`-` body, excluding the
    `+++`/`---` file headers), strips the leading +/- and surrounding whitespace,
    drops blank lines and hunk/index/header noise. Line numbers and context lines
    — which differ trivially between equivalent patches — are discarded.
    """
    if not patch:
        return ""
    out: list[str] = []
    for line in patch.splitlines():
        if _DIFF_NOISE.match(line):
            continue
        if line[:1] in ("+", "-") and line[:3] not in ("+++", "---"):
            body = line[1:].strip()
            if body:
                out.append(body)
    return "\n".join(out)


def patch_signature(patch: str) -> str:
    """Canonical signature of a patch (normalized body, whitespace-collapsed)."""
    norm = normalize_patch(patch)
    return re.sub(r"\s+", " ", norm).strip()


def distinct_patch_count(patches: Sequence[str]) -> int:
    """Number of DISTINCT non-empty solutions among patches (exact signature)."""
    sigs = {patch_signature(p) for p in patches}
    sigs.discard("")
    return len(sigs)


def mean_pairwise_distance(patches: Sequence[str]) -> float:
    """Mean pairwise structural distance in [0,1] over non-empty patches.

    distance(a,b) = 1 - difflib ratio on the normalized bodies. Graded companion
    to the exact distinct count: 0.0 when all (non-empty) patches are identical,
    →1.0 when they share nothing. Returns 0.0 if fewer than two non-empty patches.
    """
    norms = [n for n in (normalize_patch(p) for p in patches) if n]
    if len(norms) < 2:
        return 0.0
    dists = []
    for i in range(len(norms)):
        for j in range(i + 1, len(norms)):
            ratio = difflib.SequenceMatcher(None, norms[i], norms[j]).ratio()
            dists.append(1.0 - ratio)
    return float(np.mean(dists))


# --------------------------------------------------------------------------- #
# Uncertainty (across-instance spread)
# --------------------------------------------------------------------------- #

def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap (point, lo, hi) for a statistic over per-instance `values`.

    Resampling is seeded (numpy default_rng) so CIs are reproducible. Returns
    (point_estimate, ci_low, ci_high). Empty input → (0,0,0); single value → that
    value for all three (no spread to estimate).
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    point = float(statistic(arr))
    if arr.size == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = np.array([statistic(arr[row]) for row in idx])
    lo = float(np.percentile(boots, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boots, 100 * (1 + ci) / 2))
    return point, lo, hi
