"""R4 metrics: unbiased pass@k + independent patch diversity + bootstrap CI."""

import math

import pytest

from src.evaluation.metrics import (
    bootstrap_ci, distinct_patch_count, diverse_pass_at_k, mean_pairwise_distance,
    normalize_patch, pass_at_k, patch_signature,
)


def test_pass_at_k_known_values():
    assert pass_at_k(5, 0, 5) == 0.0          # none passed
    assert pass_at_k(5, 1, 5) == 1.0          # n-c < k -> certain
    assert math.isclose(pass_at_k(5, 1, 1), 0.2)   # single draw, 1/5
    assert math.isclose(pass_at_k(10, 1, 1), 0.1)
    assert math.isclose(pass_at_k(2, 1, 1), 0.5)


def test_pass_at_k_monotonic_in_k_and_c():
    assert pass_at_k(10, 2, 1) < pass_at_k(10, 2, 5)      # more draws -> higher
    assert pass_at_k(10, 1, 3) < pass_at_k(10, 4, 3)      # more correct -> higher


def test_pass_at_k_requires_k_le_n():
    with pytest.raises(ValueError):
        pass_at_k(3, 1, 5)


def test_diverse_pass_at_k_matched_k_is_any_pass():
    assert diverse_pass_at_k([False, False, True]) == 1.0
    assert diverse_pass_at_k([False, False, False]) == 0.0
    assert diverse_pass_at_k([]) == 0.0


def test_normalize_patch_strips_headers_and_keeps_code():
    patch = (
        "diff --git a/f.py b/f.py\n"
        "index 111..222 100644\n"
        "--- a/f.py\n+++ b/f.py\n"
        "@@ -1,2 +1,2 @@\n"
        " context_line\n"
        "-old_code = 1\n"
        "+new_code = 2\n"
    )
    norm = normalize_patch(patch)
    assert "old_code = 1" in norm and "new_code = 2" in norm
    assert "context_line" not in norm           # context dropped
    assert "@@" not in norm and "diff --git" not in norm and "+++" not in norm


def test_distinct_patch_count_ignores_lineno_and_empty():
    a = "@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n"
    a_shifted = "@@ -50,1 +50,1 @@\n-x = 1\n+x = 2\n"   # same edit, different line nums
    b = "@@ -1,1 +1,1 @@\n-y = 1\n+y = 3\n"
    assert distinct_patch_count([a, a_shifted]) == 1     # structurally identical
    assert distinct_patch_count([a, b]) == 2
    assert distinct_patch_count(["", "   ", a]) == 1     # empties don't count


def test_patch_signature_equal_for_equivalent_patches():
    a = "@@ -1 +1 @@\n+foo = 1\n"
    b = "@@ -99 +99 @@\n+foo = 1\n"
    assert patch_signature(a) == patch_signature(b)


def test_mean_pairwise_distance_bounds():
    same = "@@ -1 +1 @@\n+foo = 1\n"
    assert mean_pairwise_distance([same, same]) == 0.0           # identical -> 0
    assert mean_pairwise_distance([same]) == 0.0                 # <2 non-empty -> 0
    d = mean_pairwise_distance([
        "@@ -1 +1 @@\n+alpha = 1\n",
        "@@ -1 +1 @@\n+zeta_completely_different()\n",
    ])
    assert 0.0 < d <= 1.0


def test_bootstrap_ci_deterministic_and_edge_cases():
    vals = [0.0, 1.0, 0.0, 1.0, 1.0]
    r1 = bootstrap_ci(vals, seed=42)
    r2 = bootstrap_ci(vals, seed=42)
    assert r1 == r2                                   # seeded -> reproducible
    pt, lo, hi = r1
    assert lo <= pt <= hi
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)
    assert bootstrap_ci([0.7]) == (0.7, 0.7, 0.7)     # single value -> no spread
