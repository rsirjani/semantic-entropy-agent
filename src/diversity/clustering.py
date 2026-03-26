"""Semantic clustering via bidirectional entailment + entropy computation.

Implements Algorithm 1 from Farquhar et al. 2024 (Semantic Entropy).
"""

import logging
import math
from dataclasses import dataclass, field

from src.diversity.nli import NLIModel

logger = logging.getLogger(__name__)


@dataclass
class SemanticCluster:
    """A cluster of semantically equivalent candidate intents."""

    indices: list[int] = field(default_factory=list)
    """Indices into the original candidates list."""

    representative_idx: int = 0
    """Index of the cluster representative (first element added)."""

    intents: list[str] = field(default_factory=list)
    """The intent strings in this cluster."""


class SemanticClusterer:
    """Bidirectional entailment clustering + semantic entropy.

    Algorithm (Farquhar et al. 2024, Algorithm 1 / Extended Data Fig. 1):
    1. Initialize: first intent → first cluster
    2. For each subsequent intent:
       a. Compare to representative of each existing cluster
       b. If bidirectional entailment with any cluster rep → add to that cluster
       c. Otherwise → create new cluster
    3. Return list of clusters
    """

    def __init__(self, nli: NLIModel, entailment_threshold: float = 0.5):
        self.nli = nli
        self.threshold = entailment_threshold

    def cluster(self, intents: list[str], context: str = "") -> list[SemanticCluster]:
        """Cluster intents by bidirectional entailment.

        Per Algorithm 1 of Kuhn et al. 2023: the NLI classifier receives
        the CONTEXT concatenated with each sequence, not the raw sequences
        alone. This is critical because meaning depends on context — e.g.
        "modify the constructor" and "fix the disjointness check" are the
        same approach when the context is "Permutation bug with overlapping cycles."

        Args:
            intents: List of N intent strings (one per candidate response).
            context: Problem context (e.g. issue description). Concatenated
                     with each intent for NLI, per the paper's Algorithm 1.

        Returns:
            List of SemanticCluster objects.
        """
        if not intents:
            return []

        # Per Algorithm 1: NLI input is cat(context, sequence)
        # If context is provided, prepend it to each intent
        def _with_context(intent: str) -> str:
            if context:
                return f"{context} {intent}"
            return intent

        clusters: list[SemanticCluster] = []

        for i, intent in enumerate(intents):
            assigned = False
            for cluster in clusters:
                rep_intent = intents[cluster.representative_idx]
                rep_ctx = _with_context(rep_intent)
                intent_ctx = _with_context(intent)

                # Explicit scoring with diagnostic logging (replaces opaque bidirectional_entailment)
                fwd = self.nli.classify(rep_ctx, intent_ctx)
                bwd = self.nli.classify(intent_ctx, rep_ctx)
                fwd_ent = fwd["entailment"]
                bwd_ent = bwd["entailment"]
                is_match = fwd_ent > self.threshold and bwd_ent > self.threshold

                logger.info(
                    f"  NLI [{i}] vs cluster_rep[{cluster.representative_idx}]: "
                    f"fwd={fwd_ent:.3f} bwd={bwd_ent:.3f} thr={self.threshold} -> {'SAME' if is_match else 'DIFF'}"
                )

                if is_match:
                    cluster.indices.append(i)
                    cluster.intents.append(intent)
                    assigned = True
                    break

            if not assigned:
                clusters.append(
                    SemanticCluster(
                        indices=[i],
                        representative_idx=i,
                        intents=[intent],
                    )
                )

        logger.info(
            f"Clustered {len(intents)} intents into {len(clusters)} clusters: "
            f"{[len(c.indices) for c in clusters]}"
        )
        return clusters

    @staticmethod
    def compute_entropy(clusters: list[SemanticCluster], n_total: int) -> float:
        """Compute semantic entropy over the cluster distribution.

        H = -Σ p(c) * log(p(c))

        where p(c) = |cluster_c| / N (count-based, discrete variant).

        Args:
            clusters: List of semantic clusters.
            n_total: Total number of candidates (N).

        Returns:
            Semantic entropy value. 0.0 means all in one cluster (no diversity).
            log(K) is maximum for K equally-sized clusters.
        """
        if n_total <= 0 or not clusters:
            return 0.0

        entropy = 0.0
        for c in clusters:
            p = len(c.indices) / n_total
            if p > 0:
                entropy -= p * math.log(p)

        return entropy

    @staticmethod
    def should_branch(entropy: float, tau: float = 0.5) -> bool:
        """Decide whether to branch based on semantic entropy.

        Args:
            entropy: Computed semantic entropy.
            tau: Threshold. Branch if entropy > tau.

        Returns:
            True if entropy exceeds threshold (diverse enough to branch).
        """
        return entropy > tau

    def analyze(
        self, intents: list[str], tau: float = 0.5, context: str = ""
    ) -> dict:
        """Full clustering analysis: cluster, compute entropy, decide branching.

        Args:
            intents: List of N intent strings.
            tau: Entropy threshold for branching.
            context: Problem context for NLI (per Algorithm 1 of Kuhn et al.).

        Returns dict with:
            clusters: list of SemanticCluster
            entropy: float
            should_branch: bool
            n_clusters: int
        """
        clusters = self.cluster(intents, context=context)
        entropy = self.compute_entropy(clusters, len(intents))
        branch = self.should_branch(entropy, tau)

        logger.info(
            f"Semantic analysis: {len(clusters)} clusters, "
            f"entropy={entropy:.3f}, tau={tau}, branch={branch}"
        )

        return {
            "clusters": clusters,
            "entropy": entropy,
            "should_branch": branch,
            "n_clusters": len(clusters),
        }
