"""Unit tests for the execution-grounded behavioral meaning relation.

Pure: no GPU, no Docker, no swebench. Exercises the paper's primary clustering
instrument (per-test outcome vector) and the Miller-Madow-corrected entropy.
"""
import math

import pytest

from src.evaluation.behavioral_signature import (
    behavioral_signature, behavioral_entropy, structural_signature,
    cluster_counts, discrete_entropy, is_resolved_signature,
    NO_PATCH, APPLY_FAILED,
)


def _report(f2p_pass, f2p_fail, p2p_pass, p2p_fail, **flags):
    inner = {
        "patch_is_None": False, "patch_exists": True,
        "patch_successfully_applied": True,
        "tests_status": {
            "FAIL_TO_PASS": {"success": list(f2p_pass), "failure": list(f2p_fail)},
            "PASS_TO_PASS": {"success": list(p2p_pass), "failure": list(p2p_fail)},
            "FAIL_TO_FAIL": {"success": [], "failure": []},
            "PASS_TO_FAIL": {"success": [], "failure": []},
        },
    }
    inner.update(flags)
    return inner


# --------------------------------------------------------------------------- #
# Signature semantics
# --------------------------------------------------------------------------- #

def test_same_test_vector_same_class_regardless_of_listing_order():
    a = _report(["t1"], ["t2"], ["t3"], [])
    b = _report(["t1"], ["t2"], ["t3"], [])  # identical behavior
    assert behavioral_signature(a) == behavioral_signature(b)


def test_one_flipped_test_is_a_different_class():
    a = _report(["t1"], ["t2"], ["t3"], [])
    b = _report(["t1", "t2"], [], ["t3"], [])  # t2 now passes -> different outcome
    assert behavioral_signature(a) != behavioral_signature(b)


def test_resolved_signature_detection():
    resolved = _report(["t1", "t2"], [], ["t3", "t4"], [])   # all pass
    unresolved = _report(["t1"], ["t2"], ["t3", "t4"], [])
    assert is_resolved_signature(behavioral_signature(resolved))
    assert not is_resolved_signature(behavioral_signature(unresolved))


def test_empty_patch_maps_to_no_patch_atom():
    assert behavioral_signature({"patch_is_None": True, "patch_exists": False}) == NO_PATCH


def test_apply_failed_is_its_own_class_distinct_from_test_failure():
    apply_fail = {"patch_exists": True, "patch_successfully_applied": False}
    test_fail = _report([], ["t1"], ["t2"], [])
    assert behavioral_signature(apply_fail) == APPLY_FAILED
    assert behavioral_signature(apply_fail) != behavioral_signature(test_fail)


def test_unwrap_accepts_harness_instance_keyed_wrapper():
    inner = _report(["t1"], [], ["t2"], [])
    wrapped = {"sympy__sympy-12345": inner}
    assert behavioral_signature(wrapped, "sympy__sympy-12345") == behavioral_signature(inner)


# --------------------------------------------------------------------------- #
# Entropy + Miller-Madow
# --------------------------------------------------------------------------- #

def test_collapsed_outcomes_give_zero_entropy():
    reports = [_report(["t1"], ["t2"], ["t3"], []) for _ in range(5)]
    out = behavioral_entropy(reports, miller_madow=False)
    assert out["K"] == 1
    assert out["H"] == 0.0
    assert out["n_resolved_classes"] == 0


def test_three_singleton_classes_entropy_and_miller_madow():
    reports = [
        _report(["t1"], ["t2"], ["t3"], []),          # class A
        _report(["t1", "t2"], [], ["t3"], []),         # class B
        _report([], ["t1", "t2"], ["t3"], []),         # class C
    ]
    out = behavioral_entropy(reports, miller_madow=True)
    assert out["K"] == 3 and out["M"] == 3
    assert out["H"] == pytest.approx(math.log(3), abs=1e-6)         # ln 3
    assert out["miller_madow"] == pytest.approx((3 - 1) / (2 * 3), abs=1e-6)
    assert out["H_mm"] == pytest.approx(math.log(3) + 1 / 3, abs=1e-6)


def test_resolved_class_count_for_set_valued_evidence():
    # Two distinct *passing* patches + one failing -> 2 resolved classes (case 1).
    reports = [
        _report(["t1"], [], ["t2"], []),               # resolved, vector A
        _report(["t1"], [], ["t2", "t3"], []),         # resolved, vector B (extra test)
        _report([], ["t1"], ["t2"], []),               # unresolved
    ]
    out = behavioral_entropy(reports)
    assert out["n_resolved_classes"] == 2


def test_discrete_entropy_empty_is_safe():
    assert discrete_entropy([])["H"] == 0.0


# --------------------------------------------------------------------------- #
# Structural signature (secondary lens)
# --------------------------------------------------------------------------- #

def test_structural_signature_keys_on_files_touched():
    diff_a = (
        "diff --git a/sympy/core/x.py b/sympy/core/x.py\n"
        "--- a/sympy/core/x.py\n+++ b/sympy/core/x.py\n"
        "@@ -1,3 +1,3 @@ def foo():\n-    a\n+    b\n"
    )
    diff_b = (  # same file+hunk context -> same structural class
        "diff --git a/sympy/core/x.py b/sympy/core/x.py\n"
        "--- a/sympy/core/x.py\n+++ b/sympy/core/x.py\n"
        "@@ -1,3 +1,3 @@ def foo():\n-    a\n+    c\n"
    )
    diff_c = (  # different file -> different class
        "diff --git a/sympy/core/y.py b/sympy/core/y.py\n"
        "--- a/sympy/core/y.py\n+++ b/sympy/core/y.py\n"
        "@@ -9,2 +9,2 @@ def bar():\n-    a\n+    b\n"
    )
    sa, sb, sc = (structural_signature(d) for d in (diff_a, diff_b, diff_c))
    assert sa == sb
    assert sa != sc
    assert structural_signature("") == NO_PATCH


def test_cluster_counts_descending():
    assert cluster_counts(["a", "a", "b", "a", "c", "b"]) == [3, 2, 1]
