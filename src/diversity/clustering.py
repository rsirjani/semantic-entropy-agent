"""Semantic clustering via bidirectional entailment + entropy computation.

Provides several selectable clustering / branching strategies:

- "greedy"  (A): online single-pass clustering against each cluster's first
                 element. Algorithm 1 from Farquhar et al. 2024 / Kuhn et al.
                 2023. Order-dependent, representative-only, hard threshold.
- "connected" (B): full pairwise bidirectional-entailment graph + connected
                   components (single-link / transitive closure). Order-
                   independent; uses all pairs. Recommended default.
- "kernel" (G): connected-components for the discrete cluster structure, but
               the branching diversity signal is the von Neumann entropy of the
               graph heat kernel exp(-t L) built on the semantic-similarity graph
               (Kernel Language Entropy, Nikitin et al. 2024, arXiv:2405.20003).
               Threshold-free, graded; handles partial overlap. In the long-time
               / well-separated limit it recovers log(#connected-components).
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from src.diversity.nli import NLIModel

logger = logging.getLogger(__name__)

VALID_STRATEGIES = ("greedy", "connected", "kernel")


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
    """Semantic clustering + semantic entropy with selectable strategies.

    Args:
        nli: NLI model / client exposing classify() and classify_batch().
        entailment_threshold: bidirectional-entailment cutoff for "same meaning".
        strategy: one of VALID_STRATEGIES ("greedy", "connected", "kernel").
        kernel_t: heat-kernel diffusion time for the "kernel" strategy.
    """

    def __init__(
        self,
        nli: NLIModel,
        entailment_threshold: float = 0.5,
        strategy: str = "greedy",
        kernel_t: float = 1.0,
    ):
        self.nli = nli
        self.threshold = entailment_threshold
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown clustering strategy {strategy!r}; "
                f"expected one of {VALID_STRATEGIES}"
            )
        self.strategy = strategy
        if kernel_t <= 0:
            raise ValueError(f"kernel_t must be > 0, got {kernel_t}")
        self.kernel_t = kernel_t
        logger.info(
            f"SemanticClusterer: strategy={strategy} thr={entailment_threshold} "
            f"kernel_t={kernel_t}"
        )

    # ------------------------------------------------------------------ #
    # Public clustering entry point (dispatches on strategy)
    # ------------------------------------------------------------------ #

    def cluster(self, intents: list[str], context: str = "") -> list[SemanticCluster]:
        """Cluster intents into semantic equivalence classes.

        Per Algorithm 1 of Kuhn et al. 2023, the NLI classifier receives the
        CONTEXT concatenated with each sequence, not the raw sequences alone:
        meaning depends on context.

        Args:
            intents: List of N intent strings (one per candidate response).
            context: Problem context, concatenated with each intent for NLI.

        Returns:
            List of SemanticCluster objects.
        """
        if not intents:
            return []
        if len(intents) == 1:
            return [SemanticCluster(indices=[0], representative_idx=0, intents=[intents[0]])]

        if self.strategy == "greedy":
            return self._cluster_greedy(intents, context)
        # both "connected" and "kernel" use the pairwise graph for the
        # discrete cluster structure; "kernel" differs only in the entropy.
        fwd_ent, _ = self._pairwise_entailment(intents, context)
        return self._cluster_connected(intents, fwd_ent)

    def _with_context(self, intent: str, context: str) -> str:
        return f"{context} {intent}" if context else intent

    # ------------------------------------------------------------------ #
    # Strategy A: online greedy (Farquhar/Kuhn Algorithm 1) — the baseline
    # ------------------------------------------------------------------ #

    def _cluster_greedy(
        self, intents: list[str], context: str
    ) -> list[SemanticCluster]:
        clusters: list[SemanticCluster] = []

        for i, intent in enumerate(intents):
            assigned = False
            for cluster in clusters:
                rep_intent = intents[cluster.representative_idx]
                rep_ctx = self._with_context(rep_intent, context)
                intent_ctx = self._with_context(intent, context)

                fwd = self.nli.classify(rep_ctx, intent_ctx)
                bwd = self.nli.classify(intent_ctx, rep_ctx)
                fwd_ent = fwd["entailment"]
                bwd_ent = bwd["entailment"]
                is_match = fwd_ent > self.threshold and bwd_ent > self.threshold

                logger.info(
                    f"  NLI [{i}] vs cluster_rep[{cluster.representative_idx}]: "
                    f"fwd={fwd_ent:.3f} bwd={bwd_ent:.3f} thr={self.threshold} "
                    f"-> {'SAME' if is_match else 'DIFF'}"
                )

                if is_match:
                    cluster.indices.append(i)
                    cluster.intents.append(intent)
                    assigned = True
                    break

            if not assigned:
                clusters.append(
                    SemanticCluster(
                        indices=[i], representative_idx=i, intents=[intent]
                    )
                )

        logger.info(
            f"[greedy] Clustered {len(intents)} intents into {len(clusters)} "
            f"clusters: {[len(c.indices) for c in clusters]}"
        )
        return clusters

    # ------------------------------------------------------------------ #
    # Pairwise entailment matrix (shared by B and G)
    # ------------------------------------------------------------------ #

    def _pairwise_entailment(
        self, intents: list[str], context: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the full N x N entailment-probability matrices.

        Returns (fwd_ent, sym_aff) where:
            fwd_ent[i][j] = P(entailment | premise=intent_i, hypothesis=intent_j)
            sym_aff[i][j] = min(fwd_ent[i][j], fwd_ent[j][i])  (symmetric affinity)
        Diagonal of fwd_ent is 1.0; diagonal of sym_aff is 0.0 (no self-loop).
        """
        n = len(intents)
        ctx = [self._with_context(s, context) for s in intents]

        pairs: list[tuple[str, str]] = []
        idx: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    pairs.append((ctx[i], ctx[j]))
                    idx.append((i, j))

        results = self.nli.classify_batch(pairs)

        fwd_ent = np.eye(n, dtype=float)  # diagonal = 1.0 (self-entailment)
        for (i, j), res in zip(idx, results):
            fwd_ent[i, j] = res["entailment"]

        sym_aff = np.minimum(fwd_ent, fwd_ent.T)
        np.fill_diagonal(sym_aff, 0.0)
        return fwd_ent, sym_aff

    # ------------------------------------------------------------------ #
    # Strategy B: connected components on the bidirectional-entailment graph
    # ------------------------------------------------------------------ #

    def _cluster_connected(
        self, intents: list[str], fwd_ent: np.ndarray
    ) -> list[SemanticCluster]:
        """Single-link clustering: union i,j iff they bidirectionally entail.

        Order-independent (transitive closure of the entailment relation).
        """
        n = len(intents)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)  # keep lowest index as root

        for i in range(n):
            for j in range(i + 1, n):
                if fwd_ent[i, j] > self.threshold and fwd_ent[j, i] > self.threshold:
                    union(i, j)

        # Group by root, preserving index order; root (min index) = representative.
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        clusters = [
            SemanticCluster(
                indices=members,
                representative_idx=root,
                intents=[intents[k] for k in members],
            )
            for root, members in sorted(groups.items())
        ]

        logger.info(
            f"[connected] Clustered {len(intents)} intents into {len(clusters)} "
            f"clusters: {[len(c.indices) for c in clusters]}"
        )
        return clusters

    # ------------------------------------------------------------------ #
    # Entropy computations
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_entropy(clusters: list[SemanticCluster], n_total: int) -> float:
        """Discrete semantic entropy over the cluster distribution.

        H = -Σ p(c) log p(c),  p(c) = |cluster_c| / N  (count-based variant).
        0.0 = all in one cluster; log(K) is max for K equal clusters.
        """
        if n_total <= 0 or not clusters:
            return 0.0
        entropy = 0.0
        for c in clusters:
            p = len(c.indices) / n_total
            if p > 0:
                entropy -= p * math.log(p)
        return entropy

    def compute_kernel_entropy(self, sym_aff: np.ndarray) -> float:
        """Von Neumann entropy of the graph heat kernel (Kernel Language Entropy).

        Kernel Language Entropy (Nikitin et al. 2024, arXiv:2405.20003): build a
        semantic-similarity graph over the candidates, form the heat kernel
        K_t = exp(-t L) from the graph Laplacian L, treat the normalized kernel
        rho = K_t / tr(K_t) as a density matrix (PSD, trace 1), and take its von
        Neumann entropy S = -tr(rho log rho) = -Σ p_i log p_i.

        Construction:
        - Affinity W = sym_aff (symmetric entailment affinity, zero diagonal).
        - Unnormalized Laplacian L = D - W, D = diag(row sums). L is symmetric
          PSD; the multiplicity of its zero eigenvalue equals the number of
          connected components of the semantic-similarity graph.
        - Heat-kernel eigenvalues are exp(-t * mu_i) for the Laplacian
          eigenvalues mu_i; rho's eigenvalues are exp(-t mu_i) / Σ_j exp(-t mu_j).

        Limiting behaviour (this is the honest KLE relaxation — NOT the
        count-weighted variant the previous Gram-matrix code claimed):
        - K well-separated clusters -> L has K zero eigenvalues -> as t grows the
          spectrum is dominated by those K flat modes and S -> log K (the number
          of semantic clusters), exactly the discrete semantic entropy of a
          uniform K-cluster distribution.
        - All-identical candidates -> single connected component -> one zero mode
          -> S -> 0 (no diversity).
        - Partial / fuzzy overlap spreads the Laplacian spectrum so the heat
          kernel mixes modes -> S interpolates smoothly, threshold-free and
          order-independent.

        kernel_t is the genuine heat-kernel diffusion time t: larger t diffuses
        further and sharpens toward the connected-component count; smaller t
        keeps fine-grained spectral structure.
        """
        n = sym_aff.shape[0]
        if n <= 1:
            return 0.0

        w = 0.5 * (sym_aff + sym_aff.T)  # symmetrize against FP drift
        np.fill_diagonal(w, 0.0)  # no self-loops in the affinity graph

        deg = w.sum(axis=1)
        laplacian = np.diag(deg) - w  # unnormalized graph Laplacian (PSD)

        mu = np.linalg.eigvalsh(laplacian)  # real, ascending; mu >= 0
        mu = np.clip(mu, 0.0, None)  # guard tiny negative FP eigenvalues

        # Heat-kernel spectrum exp(-t * mu); shift by the minimum exponent for
        # numerical stability before normalizing (cancels in the ratio).
        log_w = -self.kernel_t * mu
        log_w -= log_w.max()
        weights = np.exp(log_w)

        p = weights / weights.sum()
        p = p[p > 1e-12]
        return float(-(p * np.log(p)).sum())

    @staticmethod
    def should_branch(entropy: float, tau: float = 0.5) -> bool:
        """Branch iff semantic entropy exceeds the threshold tau."""
        return entropy > tau

    # ------------------------------------------------------------------ #
    # Full analysis (dispatches entropy by strategy)
    # ------------------------------------------------------------------ #

    def analyze(
        self, intents: list[str], tau: float = 0.5, context: str = ""
    ) -> dict:
        """Cluster, compute the branching-diversity signal, decide branching.

        For "greedy" and "connected" the signal is the count-based discrete
        semantic entropy. For "kernel" it is the von Neumann (heat-kernel)
        entropy, which lives on the same ~[0, log N] nat scale, so tau is
        comparable across strategies (though it may warrant recalibration).

        Returns dict with: clusters, entropy, should_branch, n_clusters, strategy.
        """
        if not intents:
            return {
                "clusters": [],
                "entropy": 0.0,
                "should_branch": False,
                "n_clusters": 0,
                "strategy": self.strategy,
            }

        if self.strategy == "kernel":
            fwd_ent, sym_aff = self._pairwise_entailment(intents, context)
            clusters = self._cluster_connected(intents, fwd_ent)
            entropy = self.compute_kernel_entropy(sym_aff)
        else:
            clusters = self.cluster(intents, context=context)
            entropy = self.compute_entropy(clusters, len(intents))

        branch = self.should_branch(entropy, tau)

        logger.info(
            f"Semantic analysis [{self.strategy}]: {len(clusters)} clusters, "
            f"entropy={entropy:.3f}, tau={tau}, branch={branch}"
        )

        return {
            "clusters": clusters,
            "entropy": entropy,
            "should_branch": branch,
            "n_clusters": len(clusters),
            "strategy": self.strategy,
        }
