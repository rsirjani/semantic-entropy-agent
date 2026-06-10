"""Diagnose the gate over-merge: replay tonight's sympy-12096 strategies through
the live NLI server WITH and WITHOUT the 500-char problem-statement prefix.

Hypothesis: the shared context prefix saturates DeBERTa entailment, merging
structurally distinct strategies (different files/mechanisms) into one cluster.
"""
import itertools
import json
import re
import urllib.request

NLI = "http://localhost:8100/classify"
ENTAIL_THR = 0.7  # the clusterer's threshold

log = open("results/strategy_t0.7/sympy__sympy-12096/phased_decisions.log",
           encoding="utf-8").read()
strats = [m.group(2).strip() for m in
          re.finditer(r"^\s*\[(\d)\] cluster=\d+: (.+)$", log, re.MULTILINE)][:5]
assert len(strats) == 5, f"expected 5 strategies, got {len(strats)}"

# The same context the orchestrator prepended (problem_statement[:500]) — pull
# it from the trace's pipeline.init for this instance if present, else from the
# trajectory's first user message.
context = None
try:
    traj = json.load(open("results/strategy_t0.7/sympy__sympy-12096/trajectory_t0.traj.json",
                          encoding="utf-8"))
    msgs = traj.get("messages") or traj.get("history") or []
    for m in msgs:
        if m.get("role") == "user":
            context = m.get("content", "")[:500]
            break
except Exception as e:
    print("trajectory read failed:", e)
if not context:
    raise SystemExit("no context found")
print(f"context head: {context[:80]!r}\n")


def classify(premise: str, hypothesis: str) -> float:
    body = json.dumps({"premise": premise, "hypothesis": hypothesis}).encode()
    req = urllib.request.Request(NLI, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["entailment"]


def cluster_count(texts, with_context):
    """Union-find single-link clustering, same rule as the orchestrator."""
    n = len(texts)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merges = []
    for i, j in itertools.combinations(range(n), 2):
        a = f"{context} {texts[i]}" if with_context else texts[i]
        b = f"{context} {texts[j]}" if with_context else texts[j]
        fwd, bwd = classify(a, b), classify(b, a)
        same = fwd > ENTAIL_THR and bwd > ENTAIL_THR
        merges.append((i, j, round(fwd, 3), round(bwd, 3), same))
        if same:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    k = len({find(i) for i in range(n)})
    return k, merges


for label, wc in (("WITH 500-char context prefix", True),
                  ("WITHOUT context (raw strategies)", False)):
    k, merges = cluster_count(strats, wc)
    print(f"== {label}: {k} cluster(s) ==")
    for i, j, fwd, bwd, same in merges:
        print(f"  [{i}]vs[{j}] fwd={fwd} bwd={bwd} -> {'SAME' if same else 'DIFF'}")
    print()
