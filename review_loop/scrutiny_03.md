# Scrutiny record — iteration 3 (first-principles design review)

Reviewer charter: principal-scientist review BEFORE the headline GPU runs — the
last cheap moment to change the design. Prior iterations (00–02) audited rubric
compliance and implementation faithfulness; this pass interrogates the design
itself: is the claim worth defending, is the math right, is the framing coherent,
and which comparisons would a hostile reviewer actually press?

---

## 1. The big picture

**The claim the paper should defend (one falsifiable sentence):**

> On n=10 easy SymPy SWE-bench-Verified instances, at matched per-instance
> trajectory count and matched sampling temperature, semantic-entropy-gated
> branching produces (a) more structurally distinct final patches and (b) a
> positive diverse-pass@k\* gain over resampling the identical phased agent —
> falsified if the matched-k vanilla control matches branching's rarefied
> distinct-patch counts or its coverage (exact paired sign-flip test, primary
> endpoint at T=0.7).

**Ideal evidence:** one confirmatory paired comparison at a pre-registered
temperature with exact small-n inference; the diversity (mode-collapse) half
measured by a mechanism-independent metric; the coverage half at metric-time
matched k; every other cell (temperatures, second generator, clustering variants,
τ) labeled exploratory; budget accounting showing the control was not starved.

**Minimal sufficient experiment set:** (1) strategy arm at T=0.7 (τ=0 superset,
greedy clustering); (2) matched-k vanilla at T=0.7; (3) the exploratory sweep
T∈{0.2, 1.0} both arms; (4) SDLG arm at T=0.7 (exploratory, mechanism contrast for
R5.4); τ ablation needs NO runs (post-hoc, see §3.3 below); kernel/connected
clustering ablations optional (descriptive). Anything beyond this is nice-to-have.

**Does the repo serve this claim or a weaker one?** Mostly this claim — the
matched-k control, scaffold matching, and independent diversity metric are real
and correctly aimed. But four places served a *weaker or vaguer* claim before this
iteration: matched-k was a run-time promise, not a metric-time guarantee; the
documented headline commands were temperature-mismatched (silently favoring
nobody in particular, but unfaithful to the design); inference rested on a
percentile bootstrap that is not decision-grade at n=10; and selected-pass@1 was
claimed in the writeup with no selector in existence.

---

## 2. Findings (with evidence pointers)

### TRUTHFULLY

- **T1 (major, fixed).** RESULTS.md §5's reproduction commands ran the treatment
  without `--temperature` while the vanilla control ran at 0.7 — with the then
  config default `sample_temperature: 1.0`, anyone following the documented
  commands produced a temperature-confounded headline. Violates the R2.4 matching
  the spec itself demands. *Fix:* explicit `--temperature 0.7` in the commands;
  R2.4 amended to require explicit temperature on both arms.
- **T2 (major, fixed).** `selected-pass@1` appeared in RESULTS.md §3 and the
  results table while NO selector existed anywhere in the codebase (grep:
  zero implementations). An overclaim-in-waiting. *Fix:* implemented the
  majority-signature (self-consistency) selector — artifact-only, deployable,
  mechanism-independent — `scripts/compute_metrics.py::selected_pass_at_1`, tested.
- **T3 (moderate, fixed).** R5.4 "off-mode recovery" records could be read as
  confirmed mode-collapse signatures when, at k≈5, "vanilla 0/k passed" has
  substantial probability under pure sampling noise (P = (1−p)^k). *Fix:* records
  now carry both arms' (k, n_resolved) and a chance-level caveat; RESULTS threat
  10 makes strata/off-mode descriptive-only at n=10.
- **T4 (disclosure, fixed).** The "temperature-matched" knob touches different
  amounts of text per arm (whole agent in vanilla vs proposer-only in treatment).
  Not a flaw — it's inherent to comparing sampling-driven vs mechanism-driven
  diversity — but it was undisclosed. Now stated in RESULTS §2.2 with the
  robustness-row requirement (treatment vs vanilla's best T).

### MATHEMATICALLY

- **M1 (major, fixed).** *Matched-k at metric time.* `compare()` differenced each
  arm's `diverse_pass_at_k` at its OWN k. Any k divergence (failed resample,
  `--max-k`, capture loss) biases the gain toward the larger arm. Re-derivation:
  the correct paired quantity at common k\* = min(k_A, k_B) is
  pass@k\*(n_A, c_A) − pass@k\*(n_B, c_B) with the Chen estimator (unbiased for the
  k\*-subset probability, uses all n samples). Implemented + tested
  (`test_compare_enforces_matched_k_at_metric_time`: own-k says 0, matched-k\*
  correctly says −0.5 on the constructed example).
- **M2 (major, fixed).** *Distinct counts are sample-size-biased.* E[#distinct]
  grows with k, so cross-arm distinct-patch comparisons at unequal k were biased
  the same way. Fix: rarefaction estimator E[#distinct in a random k\*-subset] =
  Σ_sig pass@k\*(n, m_sig) (`expected_distinct_at_k`, property-tested). Derived,
  not assumed: mean pairwise distance needs NO correction (pair-inclusion
  symmetry ⇒ expected subset mean = full mean) — documented so nobody
  "corrects" it into a bias.
- **M3 (major, fixed).** *Inference at n=10.* Percentile bootstrap of a mean of
  ten lumpy values (gains are mostly −1/0/+1) has well-known undercoverage and
  granularity pathologies. The paired sign-flip permutation test is EXACT here
  (2^10 = 1024 enumerable sign patterns). Implemented
  (`paired_permutation_pvalue`, exact ≤ n=20, seeded MC beyond), wired into
  `compare()`, and made the decision rule for the (new) pre-registered primary
  endpoint; bootstrap CI demoted to companion interval. Multiple-comparison
  exposure across the sweep grid addressed by R6.5 (one confirmatory cell,
  everything else labeled exploratory).
- **M4 (stated plainly, per charter).** *Entropy quantization:* with N=5
  candidates, discrete SE takes exactly 7 values (partitions of 5: 0, 0.500,
  0.673, 0.950, 1.055, 1.332, 1.609 nats). **τ is a cluster-partition-shape rule
  at this N, not a continuous dial; τ=0 ≡ "branch iff ≥2 clusters."** Also the
  plug-in estimator is biased low (Miller–Madow ≈ (K−1)/2N, up to ~0.4 nats at
  K=5); harmless within-experiment because N is fixed, fatal if anyone compares
  entropies across different N — now a stated requirement (R3.3, RESULTS threat 9).
- **M5 (design insight → implemented).** *The τ sweep needs zero extra GPU runs.*
  The τ=0 run is the superset of every τ>0 run: the gate's no-branch action keeps
  exactly the dominant-cluster representative trajectory, which exists in the
  superset run (trajectory i = cluster i's rep, greedy post-branch execution;
  tie-break `max(key=len)` = lowest index — replicated exactly). So every τ is a
  deterministic subset selection over artifacts. `scripts/tau_sweep.py` implements
  it (R3.3 + feeds R5.5), tested on synthetic logs, smoke-run on the real
  `results/branching` artifacts (which confirm the known degeneracy: all
  instances at entropy 1.609 pre-gate-fix).
- **M6 (resolved in treatment's favor — direction matters).** *Trajectory-matched
  vs token-matched.* Branched trajectories share the SEARCH prefix; vanilla
  resamples pay full search each time. At matched trajectory count the CONTROL
  receives at least as much total compute, so trajectory-matching is the
  conservative match for a treatment win (a win cannot be a compute artifact) and
  the stricter test under a null. Token-matching would hand vanilla fewer
  trajectories and flatter the treatment — correctly NOT used. Now stated in
  RESULTS §2.2; `budget_audit.py` quantifies the realized asymmetry per arm.
- **M7 (checked, no action).** KLE implementation: von Neumann entropy of
  exp(−tL)/tr — verified the limits (K zero modes ⇒ log K as t→∞; S→log N as
  t→0 regardless of structure, hence the documented per-strategy τ
  recalibration). Chen estimator product form re-derived (1 − Π(1 − k/i));
  pass@k=n reduces to any-pass; bootstrap seeded. Bidirectional-entailment
  clustering context-conditioned at every call site. No defects found.

### PHILOSOPHICALLY

- **P1 (coherent, non-circular — verified).** "Diversity" in the claim is defined
  by structural patch distance (difflib over normalized diffs), independent of the
  NLI machinery that *produces* the branching. The mode/diversity framing of §0.1
  survives the circularity probe: the gate uses NLI entropy, the measurement does
  not.
- **P2 (weakest joint a hostile reviewer would press).** Not the framing — the
  *inference*: 20+ comparison cells at n=10 with bootstrap CIs invited a
  forking-paths attack, and the τ=0 headline invited "your title's gate never
  fires." Both now answered structurally (R6.5 primary endpoint + exact test;
  τ-quantization disclosure + post-hoc sweep that makes the gate claims
  empirical rather than vestigial). The remaining honest weakness is n=10
  easy-SymPy scope — already disclosed (threats 1/4/8) and not repairable without
  GPU budget.
- **P3 (falsifiability at this n).** The §0.1 falsifiable predictions
  (gain concentrates off-mode; ≈0 on-mode) are NOT confirmable at n=10 strata of
  ~5 — and the docs now say so: strata and off-mode records are descriptive; the
  only confirmatory statement is the pooled primary endpoint. This is the honest
  scope, and a null is publishable under it.

---

## 3. Steelmanned alternatives (decisions + reasons)

| Design choice | Strongest alternative | Decision |
|---|---|---|
| Clustering substrate = intent summaries | Cluster raw code/diff text | **Keep.** Wei et al. 2026 independently find NLI weak on raw code text; intent summaries are the defensible substrate. Diversity *measurement* is on final patches anyway (R4.2/R4.3), so the substrate choice cannot contaminate the metric. |
| Gate signal = discrete SE | KLE everywhere (graded, threshold-free) | **Keep discrete as primary.** KLE's scale is t-dependent and non-transferable (S→log N as t→0); discrete SE matches Farquhar/Kuhn and is what the quantization disclosure makes honest. KLE stays as ablation. |
| Matched-k definition | Token-matched budget | **Keep trajectory-matched** — it is the conservative direction (M6); token-matching favors the treatment. Per-arm token accounting reported so reviewers can verify. |
| τ default = 0 for the headline | τ>0 so the title's gate visibly fires | **Keep τ=0, add disclosure + post-hoc sweep** (M5). The superset run strictly dominates: it contains every τ>0 outcome. Changing the default would discard the branches the coverage claim needs. |
| Two diversity arms (strategy/SDLG) | Drop SDLG (complexity) or in-context "give me a different answer" arm | **Keep both, exploratory.** SDLG is the only mechanism that does not depend on model confidence (the case-3 story needs it). The regeneration rival stays a disclosed future-work arm (RESULTS threat 11) — running it well needs its own matching design. |
| Independent diversity metric = difflib structural | AST-diff or behavioral (test-vector) diversity | **Keep for v1, roadmap behavioral.** AST diffing on unified diffs of arbitrary hunks is brittle; the structural metric is already independent of the branching signal, and rarefaction (M2) fixed its real bias. |
| Instance set = 10 easy SymPy | Proposal's difficulty-spanning Appendix C set | **Keep, disclosed.** ~0% base resolve on hard band ⇒ no contrast at this compute; deviation already documented (CLAUDE.md, RESULTS threat 8). |
| Selector = majority signature | NLI-majority cluster; regression-test selector | **Majority signature** — artifact-only, identical on both arms, no NLI server at analysis time (amendment record `applied_03_matched_k_metric_time.md`). |

---

## 4. Actions taken this iteration

Code (all tested; 63 pytest pass, py_compile clean, all driver `--help` smokes OK):

1. `src/evaluation/metrics.py`: `expected_distinct_at_k` (rarefaction),
   `paired_permutation_pvalue` (exact sign-flip ≤ n=20), `select_majority_patch`
   (deployable selector); no-correction-needed note on `mean_pairwise_distance`.
2. `scripts/compute_metrics.py`: metric-time matched-k\* comparison with k-mismatch
   reporting; exact sign-flip p in `compare()`; rarefied distinct gain; strata
   reuse the same matched gains; off-mode records carry both arms' (k, c) +
   chance-level note; `selected_pass_at_1` wired for both arms.
3. `scripts/tau_sweep.py` (new): post-hoc τ sweep from artifacts (R3.3/R5.5),
   achievable-entropy grid, dominant-trajectory reconstruction with the
   orchestrator's exact tie-break; smoke-verified on real artifacts.
4. `scripts/run_branching.py`: `sample_temperature` fallback routed through
   `cfg()` (R8.2 — was a raw `.get(..., 1.0)`).
5. `configs/branching.yaml` + `branching_defaults.py`: `sample_temperature`
   1.0 → 0.7 (pre-registered primary endpoint temperature; EntroPO Fig. 4 knee).
6. Tests: +14 (`test_metrics.py` ×5, `test_compute_metrics.py` ×3,
   `test_tau_sweep.py` ×6).
7. `RESULTS.md`: §2.2 matching-direction + knob disclosure + pre-registered
   primary endpoint; §2.3 τ post-hoc + quantization; §3 metric upgrades; §5
   temperature-explicit commands + tau_sweep command + table rows; §6 threats
   9 (quantization/bias) and 10 (descriptive strata/off-mode).
8. `GOLD_STANDARD.md` amendments (R2.4, R3.3, R4.1, R4.2, R4.4, R6.1, R6.5) —
   rationale + rejected alternatives in `spec_amendments/applied_03_*.md` (×3).

## 5. What still requires a human / GPU

Unchanged from iteration 2, sharpened: the headline matched-k runs (now with
explicit `--temperature 0.7` primary + 0.2/1.0 exploratory), SDLG arm, then
`compute_metrics.py` / `tau_sweep.py` / `budget_audit.py` over the artifacts.
The τ ablation no longer needs separate runs (M5).

## 6. Verdict logic

Real, substantive design flaws were found and fixed this iteration (T1–T3,
M1–M3). Per the scrutiny charter, `gold_standard_met` is therefore **false** —
the loop should run at least once more over the post-amendment design to confirm
nothing further confesses.
