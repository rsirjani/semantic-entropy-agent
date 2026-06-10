# Applied amendment 05 — H1 productivity-confound diagnostics (R4.2)

**Channel:** value-level design amendment (scrutiny charter, ratchet v2).

**Spec edit:** R4.2 gains a required diagnostic: per-arm non-empty patch
fraction + a descriptive non-empty-only rarefied-gain robustness row (at
k\*_ne = min non-empty count), fixed pre-data; plus a required disclosure of the
exact-signature granularity and its bias direction.

## The gap

H1 (the confirmatory diversity endpoint) is the rarefied distinct-patch gain at
matched k\*. `expected_distinct_at_k` counts non-empty signatures while empty
patches stay in n as draws contributing nothing — the right convention for "how
many distinct solutions does a k-budget buy you." But it conflates two causal
stories: (a) the arm explores distinct solutions (mode-collapse escape — the
title claim), and (b) the arm simply *finishes* more often (its patches are
non-empty). If vanilla at T=0.7 fails to produce a patch on some resamples, the
treatment could reject H1 purely on production rate, and the paper would
over-read it as diversity. The reverse risk also exists: a real diversity gain
could be dismissed as "just productivity" without the diagnostic to separate
them.

Secondly, the exact signature counts lexical variants (renamed variable, moved
hunk) as distinct in both arms. Direction: the whole-agent-sampling control is,
if anything, the noisier producer of trivial variants (post-branch execution in
the treatment is greedy), so granularity inflates the *control's* distinct count
more — biasing H1 toward the null, i.e. conservative for the diversity claim.
This direction was nowhere stated; a hostile reviewer should find it in print,
not in a rebuttal.

## The fix (implemented + tested this iteration)

`compute_metrics.compare()` now emits `nonempty_patch_fraction` (per arm) and
`rarefied_distinct_gain_nonempty` (descriptive; same rarefaction identity over
non-empty patches at k\*_ne, with sign-flip p and power floor for completeness,
explicitly labeled NOT the confirmatory endpoint). RESULTS §3 and threat 7 state
both properties and the reading rule: if H1 rejects but the non-empty-only row
is ≈0 alongside a production-rate gap, the win is reported as productivity, not
mode-collapse escape.

## Why this is not self-serving

The confirmatory endpoint is unchanged (no forking paths); the amendment adds a
way for the *treatment's own win to be downgraded* in the writeup. It can only
make the claim harder to over-read. Nothing flips to pass: the diagnostic is
implemented and tested in the same iteration, pre-data.

## Rejected alternatives

- **Redefine H1 as the non-empty-only gain.** Conditions the confirmatory
  endpoint on an outcome (producing a patch) that is itself part of each arm's
  behavior — post-hoc subsetting inside the primary test is worse than
  disclosing a companion row. Also, "failed to produce anything" is genuinely a
  failure to explore; excluding it from the headline would flatter whichever
  arm aborts more.
- **Redefine H1 as mean pairwise distance gain.** Graded and lexically robust,
  but it measures spread among produced patches only (empty patches vanish
  entirely), has a less interpretable null (0 distance when <2 non-empty), and
  swapping the pre-registered endpoint for the second time in two iterations is
  its own forking-paths smell. It stays as a reported companion metric.
- **Cluster patches at a similarity threshold (fuzzy distinct).** Introduces a
  free parameter chosen by us (circularity-adjacent: we'd tune what "distinct"
  means for the metric our title depends on). Exact signatures + graded distance
  + disclosed direction beats a tunable middle ground for auditability.
