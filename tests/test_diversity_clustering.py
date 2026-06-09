"""R1.2/R1.3/R8.4: clustering strategies + the real heat-kernel entropy."""

import itertools
import math

import pytest

from src.diversity.clustering import SemanticClusterer


class MockNLI:
    """Entails iff two strings share their first whitespace token (group label)."""

    def _e(self, a, b):
        return 0.95 if a.split()[0] == b.split()[0] else 0.03

    def classify(self, premise, hypothesis):
        e = self._e(premise, hypothesis)
        return {"entailment": e, "neutral": 0.0, "contradiction": 1 - e}

    def classify_batch(self, pairs):
        return [self.classify(p, h) for p, h in pairs]


def test_invalid_strategy_and_kernel_t_guards():
    with pytest.raises(ValueError):
        SemanticClusterer(MockNLI(), strategy="nope")
    with pytest.raises(ValueError):
        SemanticClusterer(MockNLI(), strategy="kernel", kernel_t=0.0)


def test_connected_is_order_independent():
    c = SemanticClusterer(MockNLI(), entailment_threshold=0.5, strategy="connected")
    base = ["A x", "A y", "B z", "C w"]
    sizes = {
        tuple(sorted(len(cl.indices) for cl in c.cluster(list(perm))))
        for perm in itertools.permutations(base)
    }
    assert sizes == {(1, 1, 2)}     # one size-multiset across all permutations


def test_kernel_entropy_orders_and_limits():
    k = SemanticClusterer(MockNLI(), entailment_threshold=0.5, strategy="kernel", kernel_t=5.0)

    def ent(intents):
        _, sym = k._pairwise_entailment(intents, "")
        return k.compute_kernel_entropy(sym)

    identical = ent(["A a", "A b", "A c", "A d"])     # 1 component -> ~0
    two_group = ent(["A a", "A b", "B c", "B d"])     # 2 components -> -> log 2
    four = ent(["A a", "B b", "C c", "D d"])          # 4 components -> -> log 4
    assert identical < two_group < four
    assert identical < 0.05                            # collapses toward 0
    assert two_group < math.log(2) + 0.1               # below/approaching log 2
    assert four < math.log(4) + 0.1


def test_greedy_default_analyze_shape():
    g = SemanticClusterer(MockNLI(), entailment_threshold=0.5, strategy="greedy")
    res = g.analyze(["A x", "A y", "B z"], tau=0.0)
    assert set(res) == {"clusters", "entropy", "should_branch", "n_clusters", "strategy"}
    assert res["strategy"] == "greedy"
    assert res["n_clusters"] == 2 and res["entropy"] > 0.0
