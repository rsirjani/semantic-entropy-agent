"""R8.1/R8.3 + driver wiring: write-detection, source-only diff filter, matched-k
discovery, dataset aliases. The orchestrator import is heavy (minisweagent); kept in
its own module so a heavy-import problem can't mask the lighter metric tests."""

import json

from src.agent.phases import is_write_command
from src.evaluation.dataset import resolve_dataset_name


def test_is_write_command_no_false_positives():
    # NOT writes: submit sentinel, bare echo/printf, stderr/null redirects, views.
    assert is_write_command("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /testbed/patch.txt") is False
    assert is_write_command("cd /testbed && cat patch.txt") is False
    assert is_write_command("echo hello world") is False
    assert is_write_command("printf foo") is False
    assert is_write_command("grep -rn foo . 2> /dev/null") is False
    assert is_write_command("find / -name x 2>/dev/null") is False
    assert is_write_command("python repro.py > /dev/null 2>&1") is False


def test_is_write_command_detects_real_writes():
    assert is_write_command("sed -i s/a/b/ f.py") is True
    assert is_write_command("cd /testbed && sed -i s/a/b/ f.py") is True
    assert is_write_command("cat <<'EOF' > /testbed/f.py") is True
    assert is_write_command("echo foo > /testbed/f.py") is True
    assert is_write_command("echo foo >> /testbed/f.py") is True
    assert is_write_command("tee /testbed/f.py") is True
    assert is_write_command("patch -p1 < fix.diff") is True


def test_resolve_dataset_alias():
    assert resolve_dataset_name("lite") == "SWE-bench/SWE-bench_Lite"
    assert resolve_dataset_name("verified") == "SWE-bench/SWE-bench_Verified"
    assert resolve_dataset_name("org/custom") == "org/custom"   # passthrough


def test_source_only_diff_filter():
    from src.agent.phased_orchestrator import PhasedOrchestrator as PO
    assert PO._is_test_path("a/b/tests/foo.py") is True
    assert PO._is_test_path("a/test_x.py") is True
    assert PO._is_test_path("a/core/mul.py") is False
    diff = (
        "diff --git a/sympy/core/mul.py b/sympy/core/mul.py\n@@ -1 +1 @@\n-old\n+new\n"
        "diff --git a/sympy/core/tests/test_mul.py b/sympy/core/tests/test_mul.py\n@@ -1 +1 @@\n-t\n+t2\n"
    )
    out = PO._filter_diff_to_source(diff)
    assert "core/mul.py" in out and "test_mul.py" not in out


def test_matched_k_discovery(tmp_path):
    import run_resample_baseline as rb
    for iid, k in [("sympy__sympy-1", 5), ("sympy__sympy-2", 9)]:
        d = tmp_path / iid
        d.mkdir()
        (d / "metadata.json").write_text(json.dumps({"total_trajectories": k}), encoding="utf-8")
    # a dir without metadata.json must be ignored
    (tmp_path / "not_an_instance").mkdir()
    k_by_id = rb.discover_instances_and_k(str(tmp_path), max_k=None)
    assert k_by_id == {"sympy__sympy-1": 5, "sympy__sympy-2": 9}
    capped = rb.discover_instances_and_k(str(tmp_path), max_k=6)
    assert capped == {"sympy__sympy-1": 5, "sympy__sympy-2": 6}
