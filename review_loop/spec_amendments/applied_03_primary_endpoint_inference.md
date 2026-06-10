# Applied amendment (iteration 3): pre-registered primary endpoint, exact small-n inference, explicit temperature matching

**Rubric items amended:** R6.1 (strengthened), R6.5 (new), R2.4 (strengthened).
**Channel:** 2 (value-level, argued from the science).

## R6.5 — pre-registered primary endpoint

**Problem.** The design sweeps 3 temperatures × ≥2 treatment arms × 3 clustering
strategies × a τ grid, each yielding a treatment−vanilla gain. With n=10 instances,
running ~dozens of comparisons and narrating whichever is largest is textbook
garden-of-forking-paths; a hostile reviewer's first question. The spec previously
required CIs (R6.1) but said nothing about which comparison carries the claim.

**Amendment.** Exactly ONE comparison is confirmatory, fixed before any GPU run:
strategy-proposal (greedy clustering, τ=0 superset) vs matched-k vanilla at T=0.7,
metric = per-instance matched-k\* diverse-pass@k gain, exact paired sign-flip test.
All other cells are exploratory/descriptive and must be labeled so in the writeup.
Changing the primary after seeing results is forbidden (post-hoc findings reported
as such).

**Why T=0.7 (and the config default change 1.0 → 0.7).** The vanilla arm decodes
its *entire* agent at T (its only diversity source); the treatment arms keep
post-branch execution greedy and inject T only into the diversity source. EntroPO
(Fig. 4) shows SWE precision degrades past T≈0.9 for whole-agent sampling. A T=1.0
primary therefore risks manufacturing a treatment win out of vanilla
formatting/precision degradation rather than mode collapse — the exact confound the
"no strawman control" rule exists to prevent. 0.7 sits below that knee while still
sampling well above determinism; 0.2 and 1.0 remain as exploratory sweep arms. The
amendment also requires one robustness row: treatment vs vanilla at *vanilla's
best* sweep temperature, so the claim cannot rest on an unfavorable control T.

## R6.1 — exact paired sign-flip test

**Derivation.** At n=10 the per-instance matched-k gains are lumpy (mostly −1/0/+1
after the any-pass reduction); a percentile bootstrap of the mean over 10 such
values has known undercoverage and granularity problems. The paired sign-flip
permutation test is exact here: under H0 (gain distribution symmetric about 0) all
2^10 = 1024 sign patterns are equally likely and fully enumerable —
`src/evaluation/metrics.py::paired_permutation_pvalue` (exact at n ≤ 20, seeded
Monte Carlo beyond). The bootstrap CI stays as a companion interval; the exact test
is the decision rule.

## R2.4 — explicit temperature on both arms

**Problem found.** RESULTS.md §5's documented headline commands ran the treatment
WITHOUT `--temperature` (silently inheriting the config default, then 1.0) while
running vanilla at 0.7 — a temperature-mismatched comparison violating the matching
the spec already required. The amendment makes the failure mode structural: the
documented commands must pass the temperature explicitly on both arms, and arms are
compared only at equal T.

**Rejected alternatives.**
- *Bonferroni over all cells instead of a primary endpoint:* at n=10 the power cost
  of correcting over ~20 cells makes every cell uninterpretable; one confirmatory
  endpoint + labeled exploration is the standard, honest design at this n.
- *Wilcoxon signed-rank instead of sign-flip:* near-equivalent here, but the
  sign-flip statistic (|mean|) matches the reported quantity exactly, has no
  tie-handling ambiguity with many zero gains, and is exact at this n.
- *Keeping config default T=1.0:* maximizes vanilla's nominal diversity
  (NoveltyBench's "best case") but conflates diversity with degradation per the
  EntroPO knee; the robustness-row requirement preserves the strongest-vanilla
  comparison anyway, without making the confounded cell the headline.

**Tripwire check.** These edits add constraints on the *future* runs and analysis;
nothing about the current artifact newly passes (the runs do not exist), and the
evidence bar for the headline is raised (exact test + robustness row), not lowered.
