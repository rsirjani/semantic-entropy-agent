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


def expected_distinct_at_k(patches: Sequence[str], k: int) -> float:
    """Rarefaction: expected #distinct non-empty signatures in a random k-subset.

    When two arms produced different numbers of trajectories, comparing raw
    distinct counts is biased toward the larger arm (distinct count rises
    mechanically with sample size). The unbiased fix is the classic rarefaction
    estimator: for a uniform random k-subset of the n trajectories,
    E[#distinct] = sum_sig P(>=1 trajectory with that signature is drawn)
                 = sum_sig pass_at_k(n, m_sig, k)
    using the same hypergeometric identity as the Chen estimator, where m_sig is
    the signature's multiplicity. Empty patches stay in n (they are draws that
    contribute no signature). With k == n this equals distinct_patch_count.
    """
    n = len(patches)
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)
    counts: dict[str, int] = {}
    for p in patches:
        sig = patch_signature(p)
        if sig:
            counts[sig] = counts.get(sig, 0) + 1
    return float(sum(pass_at_k(n, m, k) for m in counts.values()))


def select_majority_patch(patches: Sequence[str]) -> int | None:
    """Deployable selector (R4.4): index of the majority-signature patch.

    Self-consistency over FINAL patches: pick the normalized signature with the
    highest multiplicity (the modal solution), and return the index of its first
    occurrence. Ties break to the signature seen earliest (deterministic).
    Empty patches never win. Returns None if every patch is empty.

    Uses only the predictions artifacts — no NLI, no test execution — so it is
    a genuinely deployable selection rule, not an oracle.
    """
    counts: dict[str, int] = {}
    first_idx: dict[str, int] = {}
    for i, p in enumerate(patches):
        sig = patch_signature(p)
        if not sig:
            continue
        counts[sig] = counts.get(sig, 0) + 1
        first_idx.setdefault(sig, i)
    if not counts:
        return None
    best = max(counts, key=lambda s: (counts[s], -first_idx[s]))
    return first_idx[best]


def mean_pairwise_distance(patches: Sequence[str]) -> float:
    """Mean pairwise structural distance in [0,1] over non-empty patches.

    distance(a,b) = 1 - difflib ratio on the normalized bodies. Graded companion
    to the exact distinct count: 0.0 when all (non-empty) patches are identical,
    →1.0 when they share nothing. Returns 0.0 if fewer than two non-empty patches.

    Unlike the distinct count, this needs NO rarefaction correction at unequal
    k: every pair is equally likely to appear in a uniform random k-subset, so
    the expected subset mean equals the full-sample mean (linearity). Exactness
    caveat: that identity is exact when all patches are non-empty; with empty
    patches in the pool the number of non-empty pairs varies per subset and the
    expected subset MEAN (a ratio of random sums) need not equal the full mean.
    This metric is a descriptive companion (never a confirmatory endpoint), and
    the comparison reports each arm's FULL-sample mean, not a subset estimate.
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

def paired_permutation_pvalue(
    gains: Sequence[float],
    n_resamples: int = 20000,
    seed: int = 0,
) -> float | None:
    """Two-sided paired sign-flip (permutation) test on per-instance gains.

    H0: the per-instance gain distribution is symmetric about 0 (no arm
    difference). Test statistic: |mean(gain)|. For n <= 20 the test is EXACT —
    all 2^n sign assignments are enumerated — which matters at this project's
    n = 10, where a percentile bootstrap over lumpy 0/1 gains is unreliable.
    Larger n falls back to seeded Monte Carlo sign-flips.

    Returns the p-value, or None for empty input. All-zero gains return 1.0
    (no evidence of any difference, trivially).
    """
    arr = np.asarray(gains, dtype=float)
    n = arr.size
    if n == 0:
        return None
    observed = abs(arr.mean())
    if n <= 20:
        # Exact enumeration of all sign patterns via bit masks.
        count = 0
        total = 1 << n
        for mask in range(total):
            signs = np.fromiter(
                ((1.0 if mask >> i & 1 else -1.0) for i in range(n)),
                dtype=float, count=n,
            )
            if abs((signs * arr).mean()) >= observed - 1e-12:
                count += 1
        return count / total
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, n))
    stats = np.abs((signs * arr).mean(axis=1))
    # +1 correction keeps the Monte Carlo p-value valid (never exactly 0).
    return float((np.sum(stats >= observed - 1e-12) + 1) / (n_resamples + 1))


def min_achievable_sign_flip_p(gains: Sequence[float], tol: float = 1e-12) -> float | None:
    """Lower bound on the exact sign-flip p-value given the zero pattern.

    Sign flips on zero gains never change the |mean| statistic, so with z zeros
    among n gains every sign pattern's statistic is duplicated 2^z times, and
    the observed statistic is attained by at least the two global sign choices
    on the nonzero entries: p >= 2 * 2^z / 2^n = 2^(1+z-n). Equivalently, with
    m = n - z nonzero gains the test can NEVER report p < 2^(1-m) — at n = 10
    that means p < 0.05 requires at least m = 6 instances with a nonzero,
    consistently-signed difference. Reporting this alongside the p-value keeps
    a null honest: "p = 0.25" may mean "underpowered given 8 ties", not
    "evidence of no effect". Returns None for empty input.
    """
    arr = np.asarray(gains, dtype=float)
    if arr.size == 0:
        return None
    z = int(np.sum(np.abs(arr) <= tol))
    return float(min(1.0, 2.0 ** (1 + z - arr.size)))


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
