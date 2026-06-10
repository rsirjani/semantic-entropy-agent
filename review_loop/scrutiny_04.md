# Scrutiny record — iteration 4 (first-principles design review)

Charter: principal-scientist review BEFORE the headline GPU runs. Iteration 3
re-derived the estimators and pre-registered an endpoint; this pass asks whether
the post-amendment design is internally coherent, whether the confirmatory
machinery actually tests the thesis the paper states, and whether the post-hoc
analyses faithfully reconstruct what a real run would do.

---

## 1. The big picture

**The claim the paper should defend (one falsifiable sentence):**

> On n=10 easy SymPy SWE-bench-Verified instances, at matched per-instance
> trajectory count and matched sampling temperature (T=0.7), semantic-entropy-gated
> branching (a) produces more structurally distinct final patches than resampling
> the identical phased agent (rarefied distinct gain at matched k\*, exact paired
> sign-flip test), and (b) given (a), is more likely to contain a passing fix
> (matched-k\* diverse-pass@k gain, exact test) — falsified if the matched-k
> vanilla control matches branching's rarefied diversity, or matches its coverage.

**Ideal evidence:** one confirmatory cell whose internal structure mirrors the
causal chain (diversity first, coverage through diversity), exact small-n
inference with its power limits printed beside it, mechanism-independent diversity
measurement, every other cell labeled exploratory, and budget accounting showing
the control was not starved.

**Minimal sufficient experiment set:** unchanged from iteration 3 — (1) strategy
arm T=0.7 τ=0 superset; (2) matched-k vanilla T=0.7; (3) exploratory sweep
T∈{0.2,1.0}; (4) SDLG arm T=0.7 (exploratory, R5.4 mechanism contrast). τ ablation
post-hoc, zero GPU.

**Did the repo serve this claim or a weaker one?** A subtly weaker one, in one
load-bearing place: the spec's §0 declares the headline "a diversity claim, not a
leaderboard claim," yet R6.5 made the *coverage* gain the sole confirmatory
endpoint and left the diversity half descriptive. The confirmatory machinery
tested the secondary half of the paper's own thesis — and, as the new power
analysis shows, the worse-powered half. Fixed this iteration (finding P1/M3).

---

## 2. Findings (with evidence pointers)

### MATHEMATICALLY

- **M1 (major, fixed) — the post-hoc τ sweep could disagree with a real run at
  exactly the grid τ values it sweeps.** The orchestrator gates on full-precision
  `entropy > τ`; the sweep parsed the 3-decimal `Entropy:` log line. Partition
  (2,2,1) of 5 has H = 1.054920…, logged as **1.055 > 1.0549** → the sweep
  branched at the achievable-grid τ where a real run gates; (3,1,1) = 0.950271
  logs as 0.950 → the sweep gated at τ=0.950 where a real run branches (the old
  test `test_sweep_gates_to_dominant_trajectory` actually asserted this
  unfaithful behavior). *Fix:* discrete SE is a deterministic function of the
  cluster partition, which is in the log — `tau_sweep.py` now recomputes it
  exactly (`exact_partition_entropy`), keeps the logged value (flagged) when they
  disagree beyond log-rounding tolerance (kernel runs — von Neumann entropy is
  NOT a partition function and must not be "corrected"), uses a full-precision
  grid with a 1e-9 boundary epsilon, and the orchestrator now logs 6 decimals
  (`phased_orchestrator.py::_log_strategy_proposal`). Regression test pins the
  (2,2,1) boundary (`test_sweep_221_boundary_not_flipped_by_log_rounding`); the
  old test corrected to orchestrator-faithful semantics.
- **M2 (major, disclosed + instrumented) — the exact sign-flip test has a hard
  tie-imposed power floor, and nobody had computed it.** Sign flips on zero gains
  never change |mean|, so with z zeros among n gains p ≥ 2·2^z/2^n = 2^(1+z−n).
  At n=10: **p < 0.05 requires ≥6 instances with a nonzero, same-direction
  difference** (m=6 gives exactly 0.03125, attained; m=5 floors at 0.0625). The
  0/1 coverage gain on easy instances (both arms often pass or both fail) will
  tie frequently — the pre-registered primary was predictably at risk of an
  *uninterpretable* null that would read as "no effect." *Fix:*
  `min_achievable_sign_flip_p` implemented (+ boundary unit test), reported
  beside every sign-flip p in `compare()`, and stated pre-run in RESULTS §2.2/§6
  threat 4. This is a disclosure, not a design escape: the floor is a property of
  exact small-n inference, and printing it is what keeps a null honest.
- **M3 (major, fixed) — confirmatory machinery vs the actual thesis (see P1).**
  The fix is statistical: a fixed-sequence (gatekeeping) family, H1 = rarefied
  distinct gain (near-continuous, rarely ties → not power-doomed), H2 = coverage
  gain tested only if H1 rejects. FWER = 0.05 with no alpha splitting; the
  coverage claim is now strictly harder than before (needs its own p<0.05 AND H1
  upstream); the diversity claim moves from descriptive to confirmatory. H1 wired
  with its own exact test + floor + per-arm levels in `compare()`.
- **M4 (moderate, fixed) — realized N is an unverified premise.** R3.3's
  quantization/comparability argument assumes N candidates per instance is fixed
  at 5, but `StrategyProposer.propose` can under-deliver (parse failure → 1
  generic fallback strategy; rejection pass can return fewer than requested).
  Entropies from different N sit on different quantization grids. Nothing
  reported realized N anywhere. *Fix:* τ sweep reports `n_candidates_by_instance`,
  grids on the modal N (ties → larger), flags `non_modal_n_instances`
  (+ test); spec/RESULTS now require excluding flagged instances from pooled
  τ/strata analyses. Run-time hard enforcement rejected (unbounded retries change
  the proposal distribution itself — selection pressure toward parse-friendly
  outputs).
- **M5 (minor, fixed) — duplicate prediction rows silently inflate n.** The
  resample driver appends to `predictions_all_trajectories.jsonl`; a re-run
  without `--skip-existing` duplicates (iid, tid) rows, inflating per-instance k,
  the rarefaction denominator, and the pairwise-distance set. *Fix:*
  `load_predictions` now dedupes by (instance_id, trajectory_id), keep-last
  (+ test). Eval loading was already dict-keyed and safe.
- **M6 (checked, no defect).** Re-verified: the Chen estimator at k\* needs no
  i.i.d. assumption in the matched-k\* use (it is the finite-population
  expectation over uniform k\*-subsets — exactly the hypergeometric identity —
  so applying it to the deliberately-non-i.i.d. branching arm is valid for the
  "random k\*-subset of what the arm produced" question both arms are asked);
  rarefaction inherits the same identity; the fixed-sequence procedure controls
  FWER by standard gatekeeping logic; `paired_permutation_pvalue` handles
  fractional gains (mean statistic, generic).

### TRUTHFULLY

- **T1 (minor, fixed) — the trace artifact lied about the proposer temperature.**
  `phased_orchestrator.py` logged a hardcoded `"temperature": 1.0` in the
  `phase2.propose_strategies.input` trace while the proposer actually runs at
  `sample_temperature` (now 0.7). A misrecorded determinism knob (R7.4) sitting
  in the exact artifact a reproducer would consult. Fixed to log
  `self.proposer.temperature`.
- **T2 (minor, fixed) — stale "known gap" in RESULTS §7** claimed no figure
  script exists; `scripts/make_figures.py` exists and is render-tested
  (`test_make_figures_renders_pngs`). An understatement rather than an overclaim,
  but the doc must match the artifacts in both directions.
- **T3 (moderate, disclosed) — selected-pass@1 is structurally asymmetric across
  arms.** On the branching arm the final patches are one-per-semantic-cluster by
  construction, so the majority-signature selector usually finds all
  multiplicities = 1 and its "majority" pick is pure earliest-seen tie-break —
  closer to first-trajectory-pass@1 than to self-consistency. The vanilla arm's
  resamples carry real multiplicity. Presenting the two numbers side by side
  without this caveat would quietly favor whichever arm the tie-break happens to
  bless. *Fix:* `selected_pass_at_1` now reports per-instance
  `majority_multiplicity`/`degenerate_tiebreak` and an arm-level count (+ test);
  RESULTS §3 and R4.4 disclose it, and note that the NLI-side alternative
  (dominant-cluster pick) already exists as the largest-τ row of the τ sweep — no
  new selector invented post-hoc (forking-paths discipline).

### PHILOSOPHICALLY

- **P1 (the central finding) — §0 and R6.5 disagreed about what the thesis is.**
  The spec calls the headline a *diversity* claim; the confirmatory record
  defended only the *coverage* corollary. A hostile reviewer reads that as: "the
  title claim is carried entirely by descriptive statistics; the one
  pre-registered test is the leaderboard number — and it is underpowered by the
  authors' own arithmetic." The fixed-sequence family (M3) closes the joint: H1
  is the title claim, H2 is the payoff, the order is the causal chain, and a null
  H1 now *kills* the headline outright — the old design had no confirmatory way
  to kill the diversity half. The hostile read of the change itself ("you added
  the endpoint you're likelier to win") is answered in
  `applied_04_hierarchical_confirmatory_family.md`: no data exist yet, FWER is
  controlled, the coverage bar went UP, and the power asymmetry is disclosed in
  print.
- **P2 (coherence probe, passed).** "Diversity" remains defined by structural
  patch distance, independent of the NLI gate; the H1 endpoint inherits that
  independence, so making it confirmatory introduces no circularity.
- **P3 (falsifiability at this n, sharpened).** With the floor analysis, the
  falsifiability claim is now quantitative: H1 is falsifiable at n=10 (continuous
  gains, floor 2^-9); H2's confirmatory power is honestly limited (needs ≥6
  non-tied instances) and that limit is printed, not hidden. Strata/off-mode
  remain descriptive (unchanged from iteration 3).

---

## 3. Steelmanned alternatives (this iteration's decisions)

| Design choice | Strongest alternative | Decision |
|---|---|---|
| Confirmatory = coverage only (status quo) | Fixed-sequence H1 diversity → H2 coverage | **Change to fixed-sequence.** Status quo tests the wrong half of §0 and is power-doomed (M2); Bonferroni co-primaries ignore the causal order and halve power; diversity-only-primary would drop the harder claim (self-serving). Full argument + hostile read in the amendment record. |
| τ sweep trusts the logged entropy | Recompute exactly from the partition | **Recompute with tolerance + flag.** Pure snap-to-grid would corrupt kernel sweeps; trusting the log provably flips gate decisions at grid boundaries (M1). |
| Assume N=5 fixed (config) | Hard-enforce N at run time; or drop under-N instances | **Measure and disclose.** Enforcement changes the proposal distribution (retry selection pressure); dropping loses valid coverage data. Non-modal-N instances are flagged and excluded only from τ/strata pooling. |
| Majority selector reported as-is | Add a cluster-size-weighted selector for the treatment arm | **Disclose degeneracy, add nothing.** New selectors post-hoc are forking paths; the dominant-cluster rule already appears as the largest-τ row of the τ sweep, pre-registered by construction. |
| Append-mode predictions trusted | Loader dedupe by (iid, tid) | **Dedupe keep-last** — re-runs are a documented workflow (`--skip-existing` exists), so the metric layer must be idempotent to them. |

Iteration-3 decisions re-examined and left standing: trajectory-matched budget
(conservative direction unchanged); τ=0 superset headline (still strictly
dominant); intent-summary clustering substrate; difflib structural diversity
metric; 10-easy-SymPy scope (disclosed).

---

## 4. Actions taken this iteration

All verified: 69 pytest pass (63 → 69; one test corrected to
orchestrator-faithful semantics, 7 added), py_compile clean, driver `--help`
smokes pass, `compute_metrics.py` and `tau_sweep.py` re-smoke-run end-to-end on
the real `results/branching` artifacts.

1. `scripts/tau_sweep.py`: `exact_partition_entropy` + recompute-with-tolerance
   (`entropy_source` per instance), full-precision grid + 1e-9 boundary epsilon,
   modal-N grid, `n_candidates_by_instance` + `non_modal_n_instances` + console
   warning.
2. `src/agent/phased_orchestrator.py`: tracer logs the real proposer temperature
   (was hardcoded 1.0); decision log now records entropy at 6 decimals.
3. `src/evaluation/metrics.py`: `min_achievable_sign_flip_p` (derivation in
   docstring).
4. `scripts/compute_metrics.py`: H1 endpoint instrumented (sign-flip p + floor +
   per-arm rarefied levels at k\*); H2 floor; predictions dedupe; selector
   degeneracy reporting; H1/H2-labeled console output.
5. Tests: +7 (−0): sign-flip floor boundary (m=6 attains 0.03125), (2,2,1)
   τ-boundary regression, kernel-disagreement fallback, non-modal-N reporting,
   dedupe-on-rerun, H1 wiring, selector degeneracy; corrected
   `test_sweep_gates_to_dominant_trajectory`.
6. `RESULTS.md`: §2.2 confirmatory family + power disclosure; §2.3 gate precision
   + realized N; §3 selector asymmetry; §5 H1/H2 table rows; §6 threats 4 and 9
   extended; §7 stale figure-gap corrected.
7. `GOLD_STANDARD.md`: R6.5 rewritten (fixed-sequence family + floor reporting),
   R3.3 strengthened (realized N + full-precision reconstruction), R4.4
   strengthened (degeneracy disclosure). Records:
   `spec_amendments/applied_04_hierarchical_confirmatory_family.md`,
   `spec_amendments/applied_04_realized_n_and_gate_precision.md`.

## 5. What still requires a human / GPU

Unchanged in shape from iteration 3: the pre-registered cell (strategy T=0.7 τ=0
superset + matched-k vanilla T=0.7), exploratory sweep, SDLG arm; then
`compute_metrics.py` (now emits the full H1→H2 record), `tau_sweep.py`,
`budget_audit.py`. Plus human ratification of the two iteration-4 spec amendments.

## 6. Verdict logic

This iteration found and fixed substantive issues — a numerically unfaithful
post-hoc gate reconstruction (M1), a confirmatory design misaligned with the
paper's own thesis and power-doomed at the floor (M2/M3/P1), and an unverified
fixed-N premise (M4). Per the charter, `gold_standard_met` = **false**; the loop
should make one more pass over the post-amendment design.
