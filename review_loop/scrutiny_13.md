# Scrutiny record — iteration 13 (first-principles design review)

Charter: fresh-eyes principal-scientist pass over the campaign branch.
Iteration 12 closed the SDLG arm's mechanism-fidelity seams and left a named
worklist: `src/diversity/relevance.py` scoring internals,
`scripts/make_figures.py` on partial artifacts, the echo-scoring contract
documentation, and a final end-to-end doc-chain pass. This iteration executed
that worklist and re-derived the estimator stack once more. It found **one
genuine spec/artifact mismatch** (the R3.3 quantization-grid exclusion was
documented but not implemented at either consumer) plus four smaller honesty
defects, all fixed and tested. No spec edits were needed — the spec already
demanded the missing behavior; the artifact was brought up to it.

---

## 1. The big picture

**The claim the paper should defend (one falsifiable sentence):**

> On n=10 easy SymPy SWE-bench-Verified instances, at matched per-instance
> trajectory count and matched sampling temperature (T=0.7), semantic-entropy-
> gated branching (a) produces more structurally distinct final patches than
> resampling the identical phased agent (H1: rarefied distinct gain at matched
> k\*, exact paired sign-flip test), and (b) given (a), is more likely to
> contain a passing fix (H2: matched-k\* diverse-pass@k gain, exact test) —
> falsified if the matched-k vanilla control matches branching's rarefied
> diversity, or matches its coverage.

Unchanged from iterations 4–12; re-examined from scratch and still the right
claim. The repo serves this claim, not a weaker substitute: the matched-k
control is the same scaffold at the same temperature; the diversity metric is
mechanism-independent (structural patch signatures, not the branching NLI);
the confirmatory family is fixed-sequence with the diversity claim (the title
claim) leading; the τ=0 superset + post-hoc sweep answers "the headline never
exercises the gate"; and a null result is a pre-declared publishable outcome.

**Ideal evidence, extended by this iteration:** every analysis the paper
*describes as guarded* must actually run the guard. RESULTS.md and R3.3
promised that instances whose realized candidate count N deviates from the
modal N — whose entropies therefore sit on a *different quantization grid* —
are "excluded from τ/strata pooling." The τ sweep only flagged them while
still pooling them into `branch_rate`/`gated_pass_rate`, and the R5.2 strata
in `compute_metrics.py` had no realized-N awareness at all. A reader checking
the docs against the artifacts would have caught the project promising a
control it did not run — the same doc/artifact-mismatch class as iteration
12's vocabulary bridge, one layer further downstream (analysis, not
mechanism).

**Minimal sufficient experiment set:** unchanged — (1) strategy arm T=0.7 τ=0
superset; (2) matched-k vanilla T=0.7; (3) exploratory T∈{0.2, 1.0}; (4) SDLG
arm T=0.7; τ ablation post-hoc at zero GPU. Campaign dry-run re-verified
after this iteration's edits (full pinned command sequence, T and τ explicit
on both arms).

---

## 2. Findings (with evidence pointers)

### TRUTHFULLY

- **T1 (major, fixed) — the R3.3 quantization-grid exclusion was documented
  but not implemented.** GOLD_STANDARD R3.3: non-modal-N instances must be
  "flagged and **excluded from pooled τ/strata analyses**, never silently
  mixed across quantization grids." RESULTS.md threat 9 claimed "any such
  instance is excluded from τ/strata pooling and disclosed." Reality:
  `tau_sweep.py::sweep` pooled every usable instance into the sweep rows
  (`for iid in usable`) and only *listed* `non_modal_n_instances`;
  `compute_metrics.py`'s R5.2 strata and median threshold had no realized-N
  input at all, and the R5.4 off-mode `low_entropy` flag would label an
  off-grid instance against a mixed-grid median. Why it matters: discrete
  entropy's range is [0, ln N] — an all-singleton N=3 instance (maximally
  diverse signal, H=1.099) reads as *lower* entropy than an all-singleton N=5
  instance (1.609), so grid mixing systematically mislabels under-delivered
  instances as "low entropy" and distorts both the strata and the sweep's
  branch rates. *Fix:* `tau_sweep.py` pools only modal-N instances into the
  rows (`n_pooled_instances`, `excluded_from_pooled_rows`; full reporting
  stays in `per_instance`); `compute_metrics.py` gains `load_realized_n`
  (parses the same last-block member lines tau_sweep sums, so the two scripts
  agree on what realized N means), excludes off-modal-grid instances from the
  strata pool and the median threshold (`strata_modal_n`,
  `strata_grid_excluded`, `strata_grid_note`), and off-mode records for such
  instances carry `low_entropy: null` + `low_entropy_reason:
  "realized_n_off_modal_grid"` instead of a wrong-grid label. SDLG-arm
  caveat handled explicitly: the branching log carries no partition, so no
  realized-N guard is possible there — the outputs *say so* rather than
  implying the guard ran. Tests:
  `test_sweep_reports_non_modal_realized_n` (extended: exclusion + a
  discriminating τ=1.2 row that would read 2/3 under pooling),
  `test_compare_strata_exclude_off_modal_grid`,
  `test_compare_strata_no_realized_n_info_says_so`,
  `test_load_realized_n_counts_last_block_members`.
- **T2 (minor, fixed) — `make_figures.py` rendered missing data as zero.**
  A partial/legacy metrics JSON (missing metric cell, or `ci95: null`)
  rendered a silent 0-height bar — in a paper figure, indistinguishable from
  a true zero — and the `ci95` unpack crashed on an explicit null. *Fix:*
  missing/None cells draw nothing, annotate "n/a" at the bar position, and
  print a warning; both the per-arm charts and the H1 comparison figure.
  Test: `test_make_figures_partial_artifact_renders_na_not_zero`.
- **T3 (minor, fixed) — `relevance.py` described a method the live config
  disables.** The module docstring presented the DeBERTa-NLI entailment path
  as *the* scoring method, while the checked-in config sets
  `relevance_use_nli: false` (comment: "entailment ≠ topical relevance") and
  the class default is the LLM 0–10 scorer; `configs/branching.yaml` line 237
  labeled the threshold "context-conditioned entailment per Kuhn et al."
  though it applies to the LLM score in the live config; and `has_strategy`
  (a keyword-heuristic method a reader might assume drives phase transitions)
  was dead code — defined, never called. *Fix:* docstring rewritten to state
  the dispatch, the live default, and the failure semantics (summary failure
  → raw thought; scoring failure → 0.0, counted toward saturation —
  symmetric across arms since every arm runs the same SEARCH machinery, so
  not a treatment/control confound); config comment corrected; dead method
  removed.
- **T4 (minor, fixed) — the echo-scoring degradation contract lived only in
  code docstrings.** RESULTS.md §2.4 deviation 1 documented the text-level
  vocabulary bridge but not what happens if the pinned vLLM rejects the
  `max_tokens=0, echo=True` prompt-logprobs shape (out-of-top-k substitutes
  score I_ij = 0.0; ranking degrades toward (A+S)-dominated — visible in
  logs, no silent mechanism swap). *Fix:* disclosure added to §2.4, AND the
  risk converted from "verified only at run time" to a pre-launch check:
  `scripts/smoke_test.py` now probes both raw `/v1/completions` request
  shapes `_get_importance_scores` depends on (top-k logprobs; echo
  prompt-logprobs) with the exact JSON sdlg.py sends.
- **T5 (minor, fixed) — `smoke_test.py` probed the wrong port.** It hardcoded
  `http://localhost:8000/v1`; the live config and `start_vllm.sh` pin host
  port **8001** (8000 is owned by the wslrelay→pdf-reader backend — the same
  machine fact behind iteration 12's T1). A pre-launch smoke test failing
  against a healthy server invites exactly the wrong "fix." Corrected to
  8001 with the reason in a comment.

### Seams audited CLEAN (the iteration-12 worklist, discharged)

- **`relevance.py` internals:** the 0–10 regex parse clamps to [0,1]; all
  instrument calls are temperature-0 with the temperature key stripped from
  model_kwargs; `score_trajectory_step` output schema matches every consumer
  in `phased_orchestrator.py` (saturation streak, pruning, decision log).
  Behavioral content unchanged by T3 (docs/dead code only).
- **`make_figures.py` semantics:** the per-arm `distinct_patches` bars are
  correctly labeled own-k descriptives; the only cross-arm diversity figure
  is the rarefied @k\* chart annotated with the H1 sign-flip p, its power
  floor, and the artifact-encoded H2 gate status — figure and JSON state the
  same inference rule.
- **Estimator stack, fourth spot-check** (two full re-derivations stand from
  iterations 9–10): `pass_at_k` product form ≡ 1−C(n−c,k)/C(n,k) with
  correct edge guards (c=0→0, n−c<k→1, k>n raises) — matches the Chen
  reference implementation; `expected_distinct_at_k` is the standard
  rarefaction identity Σ_sig pass@k(n,m_sig,k) with empty patches in n but
  never contributing a signature; the exact sign-flip enumerates all 2^n
  masks (identity included), all-zero gains → p=1.0, MC fallback carries the
  +1 correction; `min_achievable_sign_flip_p` = 2^(1+z−n) re-derived (sign
  flips on zeros never move |mean|; the two global sign choices on nonzeros
  always attain the observed statistic); `bootstrap_ci` seeded percentile
  with sane n=0/n=1 degenerate returns; `mean_pairwise_distance`'s
  no-rarefaction claim is exact for all-non-empty pools and the docstring
  states the empty-patch caveat and descriptive-only status.
- **`compute_metrics.compare` k\* discipline:** H2 gains use the eval-table
  k\* = min(k_a, k_b) via Chen on both arms; H1 rarefied additionally bounds
  k\* by prediction counts, with `pred_eval_count_mismatch` naming any
  desync instead of silently min()-ing; the confirmatory-family block derives
  H2's status from H1's p in the artifact itself.
- **`budget_audit.py`:** both arm layouts auto-detected and stage-tested;
  injected/branched messages carry no usage so only real model calls count;
  `primary` excluded everywhere; the fairness note names the known
  undercount and which arm it works against.
- **Campaign driver:** dry-run re-verified post-edit — confirmatory cell
  first, T and τ=0 explicit on the treatment command, per-temperature
  results dirs, both-arm budget audits, tau sweep step present. No code in
  `run_campaign.py` parses the tau-sweep/metrics JSON schemas (the analyst
  reads them as text), so this iteration's additive keys break nothing.
- **Doc chain:** CLAUDE.md instance table ↔ `dataset.py` target list
  (verified iteration 12, unchanged); RESULTS §5 command sequence ↔ driver
  step list (dir suffix `_t0.7` consistent); §2.3/§3/§6 wording now matches
  the implemented guard behavior (that was T1's doc half);
  `collect_results.py` confirmed legacy (`run_id="baseline_v1"`, outside the
  campaign path, like `run_baseline.py` per R2.3).

### MATHEMATICALLY

- **M1 — the T1 fix touches no confirmatory number.** H1/H2 use patch
  signatures and eval outcomes only — entropy and realized N enter only the
  descriptive strata/off-mode/τ-sweep layers. The confirmatory cell is
  bit-identical before/after this iteration.
- **M2 — direction of the grid-mixing bias (why T1 matters even
  descriptively):** entropy's ceiling is ln N, so under-delivered instances'
  entropies are deflated relative to the modal grid; a mixed-grid median
  splits low, pushing modal instances into "high" and off-grid instances
  into "low" — which would have *inflated* the R5.4 off-mode (case-3,
  mode-collapse signature) candidate list with artifacts of proposer
  under-delivery. The exclusion is therefore conservative for the §0.1
  narrative, consistent with the standing rule that fixes must never
  manufacture treatment-favorable readings.
- **M3 — N=5 achievable-entropy grid re-confirmed** against the 7 partitions
  of 5 ({0, .5004, .6730, .9503, 1.0549, 1.3322, 1.6094}); the sweep's
  strict-> gate with 1e-9 float-noise tolerance and the exact-recompute-at-
  boundary behavior re-checked via the existing (2,2,1) boundary test.

### PHILOSOPHICALLY

- **P1 — framing re-audited, still coherent and non-circular.** H1's
  diversity metric is mechanism-independent; the §0.1 five-mechanism family
  is carried into R5's decomposition; the entropy blind spot has both a
  theory citation chain and a measuring instrument (R5.4). Nothing new to
  confess at the framing layer.
- **P2 — the weakest joint** remains gate saturation (threat 11: the pilot's
  all-singleton partitions make τ degenerate at this substrate/threshold) and
  n=10 power — both pre-answered with pre-committed readings (saturation is
  reported as a negative finding about the gate, not retuned away; the power
  floor is printed beside every p). After T1, a hostile reviewer probing
  "your τ analysis pools entropies from different grids" has an
  artifact-level answer too.
- **P3 — the generalizing lesson, continuing the 5→12 series:** iteration 12
  taught that fallbacks must fail closed; iteration 13's instance is that
  **a disclosed limitation is not yet a guarded one** — three of this pass's
  five findings (T1, T4, T5) were risks the docs *described* correctly but
  no artifact enforced. The audit chain extends: … → own-code failure
  semantics → **documented-guard implementation (every "we exclude/flag/
  degrade gracefully" sentence in the writeup must name the code path and
  test that does it)**.

---

## 3. Steelmanned alternatives (this iteration's decisions)

| Design choice | Strongest alternative | Decision |
|---|---|---|
| Exclude non-modal-N instances from pooled τ/strata rows | Rescale entropies onto a common grid (e.g. normalize by ln N) and pool everything | **Exclude.** Normalized entropy H/ln N changes the gate semantics (the orchestrator gates on raw nats, so the sweep must too — R3.3's "replicate the real run" premise), and partition shapes at different N still aren't comparable events (a (2,2) at N=4 and (3,2) at N=5 have different cluster-count implications for branching). At the expected under-delivery rate (rare), excluding loses almost nothing; rescaling buys pooled n at the cost of construct fidelity. Per-instance records keep the excluded data visible. |
| Parse realized N in `compute_metrics.py` with its own regexes | Import `tau_sweep.parse_instance` (single parser) | **Own minimal parser, same regex semantics.** `tau_sweep` already imports from `compute_metrics`; importing back would be circular. The member-line regex is identical in both and the new `test_load_realized_n_counts_last_block_members` pins last-block behavior to the same convention tau_sweep tests pin. A shared `src/evaluation/artifacts.py` refactor was considered and rejected as churn this close to launch — noted as cleanup, not a correctness need. |
| Figures: render missing cells as annotated "n/a" | Raise on partial artifacts (fail loud) | **Annotate.** The figures script legitimately runs on single-arm and partial-campaign artifacts (R7.3 regenerability during the campaign); failing loud would block regeneration of the cells that DO exist. The dishonest failure mode was the silent zero, which is now impossible; the warning is printed besides. |
| smoke_test gains the SDLG contract probe | Leave the contract run-time-verified (disclosed) | **Probe pre-launch.** Zero cost (the smoke test already requires a live server), uses the exact request JSON from `sdlg.py`, and converts a disclosed silent degradation into a checked precondition — the same direction as iteration 11's classify-or-refuse eval verdicts. |
| Remove `has_strategy` dead code | Keep it for a future heuristic | **Remove.** Dead code in the relevance scorer invites the false belief that keyword matching gates phase transitions (it never did); git history preserves it. |

Standing decisions re-examined and left in place: trajectory-matched budget
(conservative direction, quantified by the audit); τ=0 superset headline +
command-pinned post-hoc sweep; intent-summary clustering substrate;
fixed-sequence H1→H2 family (untouched — still no data); majority-signature
selector with degeneracy disclosure; 10-easy-SymPy scope with the Appendix-C
deviation disclosed; no-retune rule under gate saturation; R6.5 adaptive
boundary + both-plane integrity guard; classify-or-refuse eval verdicts; arm
purity at the fallback layer.

Known limitations recorded, not fixed (disclosed): harness log-marker
constants remain inlined strings coupled to the pinned swebench version
(verified iteration 12); the iteration-10 porcelain caveat (campaign must
start from a clean tree) stands; the SDLG arm has no realized-N guard because
its branching log carries no partition (now stated in the artifacts
themselves); `mean_pairwise_distance`'s subset-mean identity is inexact with
empty patches in the pool (descriptive metric only, documented in the
docstring).

---

## 4. Actions taken this iteration

All verified: **136 pytest pass** (132 → 136; +4 new, 0 removed),
`py_compile` clean on every touched file, campaign dry-run prints the full
pinned plan post-edit.

1. `scripts/tau_sweep.py` — pooled sweep rows now exclude off-modal-grid
   instances (`n_pooled_instances`, `excluded_from_pooled_rows`); validity
   note and CLI warning updated to state the implemented behavior (T1).
2. `scripts/compute_metrics.py` — `load_realized_n` (last-block member-line
   parse, consistent with tau_sweep); `compare(..., realized_n=)` excludes
   off-modal-grid instances from the R5.2 strata pool and median threshold
   (`strata_modal_n`, `strata_grid_excluded`, `strata_grid_note`); R5.4
   off-mode records carry `realized_n` and a None `low_entropy` +
   `low_entropy_reason` when off-grid; CLI warning added (T1).
3. `scripts/make_figures.py` — missing/None metric cells render as annotated
   "n/a" (never a silent 0 bar), tolerant `ci95` handling, printed warning;
   both per-arm charts and the H1 comparison figure (T2).
4. `src/diversity/relevance.py` — module docstring rewritten (live backend =
   LLM 0–10 scorer; NLI kept as ablation backend; failure semantics stated,
   symmetric across arms); dead `has_strategy` removed (T3).
5. `configs/branching.yaml` — `relevance_threshold` comment corrected (T3).
6. `RESULTS.md` — §2.4 deviation 1: echo-scoring graceful-degradation
   contract + smoke-test probe sentence (T4); §2.3: τ-sweep/strata exclusion
   wording now describes the implemented guard, including the SDLG-arm
   caveat (T1 doc half).
7. `scripts/smoke_test.py` — port corrected to the live 8001 (T5); new
   `test_sdlg_importance_contract` probing both raw completions shapes the
   importance scorer depends on (T4).
8. Tests — `tests/test_tau_sweep.py` (exclusion + discriminating-τ
   assertions), `tests/test_compute_metrics.py` (+3: realized-N loader,
   strata grid exclusion, no-info disclosure),
   `tests/test_budget_and_figures.py` (+1: partial-artifact n/a rendering).
9. `review_loop/scrutiny_13.md` — this record.

No GOLD_STANDARD.md edits: the spec already required the T1 behavior (R3.3);
this iteration brought the artifact up to the spec, so no amendment record is
needed.

## 5. What still requires a human / GPU

1. Ratify the iteration-12 spec amendment
   (`applied_12_arm_purity_no_silent_fallback.md`) and any earlier unratified
   ones — independent spec-critic review per ratchet v2. (No new amendments
   from iteration 13.)
2. **Pre-launch:** start the vLLM server (`scripts/start_vllm.sh`) and run
   `python scripts/smoke_test.py` — it now verifies the SDLG importance
   contract (top-k logprobs + echo prompt-logprobs) before any GPU hours are
   spent.
3. **Launch the campaign from a CLEAN working tree:**
   `python scripts/run_campaign.py --go`.
4. After the runs: check threat 11 first (realized entropy distribution and
   `non_modal_n_instances` in `tau_sweep_*.json`); confirm
   `pred_eval_count_mismatch` / `k_mismatch_instances` /
   `strata_grid_excluded` empty or explained; read H1 against
   `nonempty_patch_fraction`; read H2 through `confirmatory_family`; verify
   both-arm budget audits; fill RESULTS §5 from script output only.
5. Git hygiene: decide whether this campaign branch becomes mainline.

## 6. Verdict logic

This iteration discharged the iteration-12 worklist and found one substantive
defect in it: (T1, major) a guard that both the spec and the writeup claimed
— exclusion of off-quantization-grid instances from pooled τ/strata analyses
— existed only as a flag in one consumer and not at all in the other, plus
four smaller honesty fixes (silent-zero figures, a docstring selling a
disabled backend, an undocumented degradation contract, a wrong-port smoke
test). Per the charter, finding real fixable issues means `gold_standard_met`
= **false**; the loop should pass once more. The audit surface is now thin:
every named seam from twelve iterations has had a dedicated pass, and this
pass's findings were all in the descriptive/tooling layer — none touched the
confirmatory cell. If the next fresh pass over the remaining corners (the
`phases.py` transition machinery details, `nli_server.py` device handling,
the `run_branching.py`/`run_resample_baseline.py` CLI surface against the
documented commands one final time) surfaces nothing substantive, the design
has nothing left to confess and the remaining actions are human-only
(ratification + smoke test + GPU launch).
