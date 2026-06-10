"""R4 metrics: unbiased pass@k + independent patch diversity + bootstrap CI."""

import math

import pytest

from src.evaluation.metrics import (
    bootstrap_ci, distinct_patch_count, diverse_pass_at_k, expected_distinct_at_k,
    mean_pairwise_distance, normalize_patch, paired_permutation_pvalue, pass_at_k,
    patch_signature, select_majority_patch,
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


PATCH_A = "@@ -1 +1 @@\n+x = 1\n"
PATCH_B = "@@ -1 +1 @@\n+y = 2\n"
PATCH_C = "@@ -1 +1 @@\n+z = 3\n"


def test_expected_distinct_at_k_full_k_equals_exact_count():
    patches = [PATCH_A, PATCH_A, PATCH_B, "", PATCH_C]
    assert math.isclose(expected_distinct_at_k(patches, len(patches)),
                        distinct_patch_count(patches))


def test_expected_distinct_at_k_all_unique_equals_k():
    # All multiplicities 1 -> E[distinct in k draws] = sum k/n over sigs = k.
    patches = [PATCH_A, PATCH_B, PATCH_C]
    assert math.isclose(expected_distinct_at_k(patches, 2), 2.0)
    assert math.isclose(expected_distinct_at_k(patches, 1), 1.0)


def test_expected_distinct_at_k_monotone_and_below_exact():
    patches = [PATCH_A, PATCH_A, PATCH_A, PATCH_B]   # collapsed arm: 2 distinct
    e1 = expected_distinct_at_k(patches, 1)
    e2 = expected_distinct_at_k(patches, 2)
    e4 = expected_distinct_at_k(patches, 4)
    assert e1 < e2 < e4
    assert math.isclose(e4, 2.0)                     # full sample = exact count
    assert expected_distinct_at_k([], 3) == 0.0


def test_paired_permutation_pvalue_known_cases():
    # All-zero gains: no difference, p = 1.
    assert paired_permutation_pvalue([0.0, 0.0, 0.0]) == 1.0
    # n=2, gains (1,1): patterns (+,+),(+,-),(-,+),(-,-) -> |mean| in {1,0,0,1};
    # |stat|>=1 for 2 of 4 -> p=0.5.
    assert math.isclose(paired_permutation_pvalue([1.0, 1.0]), 0.5)
    # Consistent positive gains at n=10 -> smallest achievable two-sided p = 2/1024.
    p = paired_permutation_pvalue([1.0] * 10)
    assert math.isclose(p, 2 / 1024)
    assert paired_permutation_pvalue([]) is None


def test_select_majority_patch_majority_ties_and_empties():
    # Majority signature wins regardless of position.
    assert select_majority_patch([PATCH_B, PATCH_A, PATCH_A]) == 1
    # Tie -> earliest-seen signature's first occurrence.
    assert select_majority_patch([PATCH_B, PATCH_A]) == 0
    # Empty patches never win; all-empty -> None.
    assert select_majority_patch(["", PATCH_A, ""]) == 1
    assert select_majority_patch(["", "   "]) is None


def test_bootstrap_ci_deterministic_and_edge_cases():
    vals = [0.0, 1.0, 0.0, 1.0, 1.0]
    r1 = bootstrap_ci(vals, seed=42)
    r2 = bootstrap_ci(vals, seed=42)
    assert r1 == r2                                   # seeded -> reproducible
    pt, lo, hi = r1
    assert lo <= pt <= hi
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)
    assert bootstrap_ci([0.7]) == (0.7, 0.7, 0.7)     # single value -> no spread
