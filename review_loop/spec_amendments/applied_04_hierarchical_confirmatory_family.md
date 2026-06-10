# Applied amendment (iteration 4): hierarchical confirmatory family (R6.5) + selector-degeneracy disclosure (R4.4)

**Channel:** value-level design amendment (scrutiny charter, ratchet v2).
**Spec sections edited:** R6.5 (rewritten as a fixed-sequence two-endpoint family +
required power-floor disclosure), R4.4 (added degeneracy-disclosure requirement).

## What changed

R6.5 previously named ONE confirmatory endpoint: the matched-k\* diverse-pass@k
(coverage) gain at T=0.7, exact sign-flip test. It now names the same single
confirmatory **cell**, but within it a **fixed-sequence (gatekeeping) family** at
family-wise α=0.05:

- **H1 — diversity:** rarefied distinct-patch gain at matched k\* (exact sign-flip).
- **H2 — coverage:** matched-k\* diverse-pass@k gain (exact sign-flip), confirmatory
  **only if H1 rejects**; otherwise descriptive.

It also now requires reporting the tie-imposed power floor `min_achievable_p =
2^(1+z−n)` beside every sign-flip p.

R4.4 now requires disclosing how often the majority-signature selector's pick was a
pure tie-break (typical on the branching arm, whose patches are one-per-cluster by
construction).

## Scientific rationale

1. **Coherence with §0.** The spec's own §0 says the headline is "a diversity
   claim, not a leaderboard claim," and the §0/scrutiny_03 claim sentence has two
   halves — (a) more structurally distinct patches, (b) higher coverage. Under the
   old R6.5 the *diversity* half had **no confirmatory test at all** (descriptive
   only) while the *coverage* half — the leaderboard-flavored half — was the sole
   primary. The confirmatory machinery tested the secondary half of the paper's
   own thesis. The fixed-sequence family fixes that incoherence.
2. **The order is causal, not opportunistic.** Branching can only raise coverage
   *through* producing distinct solutions; a significant H2 with a null H1 would be
   uninterpretable (and likely noise). Fixed-sequence gatekeeping with the upstream
   mechanism first is the textbook design (e.g. Maurer & Bretz fixed-sequence
   procedures); FWER stays at 0.05 with no alpha splitting.
3. **Power facts argued *before* any data exist.** Derived this iteration: the
   exact sign-flip p is floored at 2^(1+z−n) by z tied (zero-gain) instances; at
   n=10, p<0.05 requires ≥6 same-direction nonzero gains. The 0/1 coverage gain on
   easy instances will tie often (both arms pass or both fail); the rarefied
   distinct gain is near-continuous and ties rarely. Making the only confirmatory
   endpoint the worst-powered one would have predictably produced an
   uninterpretable null and invited misreading "p=0.25" as "no effect." The
   `min_achievable_p` disclosure makes that failure mode visible in print.

## Why this is not self-serving (the hostile read, answered)

*Hostile read:* "you added the endpoint you are likelier to win, before the runs."
Answers: (i) the GPU runs have not been executed — there is no artifact this edit
grades into a pass, and no result this edit was fitted to; (ii) the coverage claim
became **strictly harder** (it now needs its own p<0.05 *and* H1 upstream — under
the old spec it needed only itself); (iii) the diversity claim's bar went from
"descriptive narrative" to "exact test at α=0.05" — a raise, not a cut; (iv) FWER
is controlled, so the family does not buy extra rejection probability under the
global null; (v) the asymmetric-power fact is disclosed in RESULTS §2.2 so a
reviewer can see exactly why H1 leads. A null H1 now *kills* the headline claim
outright — the old spec had no confirmatory way to kill the diversity half.

## Rejected alternatives

- **Keep single-endpoint coverage primary (status quo).** Rejected: tests the
  wrong half of the §0 thesis and is predictably underpowered (floor analysis);
  a paper whose only confirmatory number is a power-doomed null while its title
  claim rides on descriptive statistics is weaker science, not more conservative
  science.
- **Make diversity the lone primary, coverage descriptive.** Rejected: that WOULD
  be self-serving (drops the harder claim from the confirmatory record entirely).
- **Bonferroni-split α=0.025 each, no hierarchy.** Rejected: ignores the causal
  ordering, halves the power of both tests, and permits the incoherent outcome
  "coverage confirmed, diversity not."
- **Co-primary with Hochberg step-up.** Workable but needlessly opaque at two
  endpoints; fixed-sequence is simpler, equally valid, and encodes the science.
- **Inventing a cluster-aware selector for the treatment arm (R4.4).** Rejected:
  post-hoc selector proliferation is a forking path; instead the degeneracy is
  measured and disclosed, and the NLI-side alternative (dominant-cluster pick)
  already exists as the largest-τ row of the τ sweep.

## Implementation (same iteration — obligations added AND met)

- `src/evaluation/metrics.py::min_achievable_sign_flip_p` (+ unit test incl. the
  m=6 boundary where the floor is attained).
- `scripts/compute_metrics.py::compare`: H1 gets its own exact sign-flip p and
  floor; per-arm rarefied levels at k\*; H2 gets `min_achievable_p`.
- `scripts/compute_metrics.py::selected_pass_at_1`: per-instance
  `majority_multiplicity` / `degenerate_tiebreak` + arm-level count (+ test).
- RESULTS.md §2.2 (family + power disclosure), §3 (selector asymmetry), §5 (table
  rows), §6 threat 4 (power floor).
