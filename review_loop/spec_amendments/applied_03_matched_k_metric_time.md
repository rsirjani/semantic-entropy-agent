# Applied amendment (iteration 3): metric-time matched-k, rarefied diversity, deployable selector

**Rubric items amended:** R4.1, R4.2, R4.4
**Channel:** 1 (derivable correction) for R4.1/R4.2; 2 (value-level specification) for R4.4.

## R4.1 — matched k enforced at metric time

**Derivation.** The spec required matched k but only the *run-time* driver enforced
it (`run_resample_baseline.py` reads k from treatment metadata). At metric time,
`compare()` differenced `diverse_pass_at_k` computed at *each arm's own* k. Whenever
k_A ≠ k_B (a failed resample, a `--max-k` cap, a patch-capture loss), the arm with
more trajectories gets a mechanically higher any-pass probability — the comparison
is biased by sample count, which is exactly the confound the matched-k design
exists to remove. The unbiased fix is standard: evaluate BOTH arms at the common
k\* = min(k_A, k_B) with the Chen et al. (2021) estimator pass@k\*(n, c), which uses
all n samples of each arm and is unbiased for the k\*-subset probability. At equal
k this reduces to the previous behavior, so it strictly dominates.

**Demonstration (test):** `tests/test_compute_metrics.py::test_compare_enforces_matched_k_at_metric_time` —
treatment 4 trajectories (1 pass) vs vanilla 2 trajectories (1 pass): own-k
comparison says gain 0; matched-k\*=2 correctly says −0.5.

## R4.2 — rarefaction for distinct counts at unequal k

**Derivation.** E[#distinct signatures] is monotone increasing in sample count, so
raw distinct-count differences at unequal k are biased toward the larger arm. The
rarefaction estimator E[#distinct in a uniform random k\*-subset] =
Σ_sig (1 − C(n−m_sig, k\*)/C(n, k\*)) = Σ_sig pass@k\*(n, m_sig) is the classical
unbiased correction (same hypergeometric identity as Chen). Mean pairwise distance
needs no correction (every pair equally likely in a random subset ⇒ expected subset
mean = full mean) — stated in the metric docstring so nobody "corrects" it wrongly.

**Implemented:** `src/evaluation/metrics.py::expected_distinct_at_k`,
used by `compare()`; property tests (k=n ⇒ exact count; all-unique ⇒ =k; monotone).

## R4.4 — selector specified as *deployable*

**Rationale.** The previous wording ("majority cluster / regression tests") was
ambiguous and had NO implementation — RESULTS.md §3 listed selected-pass@1 as a
metric while no selector existed (an overclaim this iteration removes). The
amendment pins the requirement to a selector computable from run artifacts alone.
Implemented as majority vote over normalized final-patch signatures
(self-consistency; deterministic tie-breaks; empty patches never win; all-empty =
miss, not skip), `scripts/compute_metrics.py::selected_pass_at_1`, on both arms.

**Rejected alternatives.**
- *NLI-majority-cluster selector:* needs the DeBERTa server at analysis time and
  reuses the branching signal; the signature-majority selector is artifact-only and
  mechanism-independent, so it runs identically on the vanilla arm. (NLI selection
  would not be circular for *selection* — R4.2 only forbids it for the diversity
  metric — but the simpler selector is strictly more reproducible.)
- *Regression-test selector:* requires executing tests at selection time; that is a
  different (and stronger) deployment assumption — left as roadmap, not required.

**Tripwire check.** No item flips to pass by these edits as of the edit time: the
requirements were simultaneously implemented and tested this iteration (the edits
*add* obligations relative to the previous spec; they do not relax any), and the
headline evidence (GPU runs) remains as ungathered as before.
