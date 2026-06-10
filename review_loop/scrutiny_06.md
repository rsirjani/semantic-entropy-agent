# Scrutiny record — iteration 6 (first-principles design review)

Charter: principal-scientist review BEFORE the headline GPU runs. Iteration 5
rewrote the eval driver and asked this pass for fresh eyes on the post-fix
pipeline, plus a check that nothing else consumes the legacy eval files. This
pass found that the iteration-5 completeness contract — "the eval record counts
what the arm produced" — was enforced at the eval layer but **violated one
layer up, at the predictions producer, in the pro-treatment direction**, and
confirmed it on the real pilot artifacts (5 of 10 instances affected). Fixed,
tested, and the contract extended to the producer in the spec.

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

**Ideal evidence:** unchanged from iterations 4–5 (one confirmatory cell
mirroring the causal chain, exact small-n inference with printed power floors,
mechanism-independent diversity measurement, budget accounting showing the
control was not starved) **plus the clause iteration 5 added — an artifact
pipeline that cannot redefine k between run and metric — now enforced at BOTH
layers that define k: the eval record AND the predictions record it is built
from.** That second layer is what iteration 6 had to earn.

**Minimal sufficient experiment set:** unchanged — (1) strategy arm T=0.7 τ=0
superset; (2) matched-k vanilla T=0.7; (3) exploratory sweep T∈{0.2,1.0};
(4) SDLG arm T=0.7. τ ablation post-hoc, zero GPU.

**Did the repo serve this claim or a weaker one?** The metric and eval layers
served it; the *run drivers* did not. The treatment driver recorded only
trajectories that finished with a non-empty patch, while the control driver
recorded every resample including failures. "Matched trajectory budget" was
therefore true at run time and false in the artifacts — the treatment's
unproductive draws were paid for but never counted against it. The design
documents promised a conservative comparison; the bookkeeping delivered a
pro-treatment one.

---

## 2. Findings (with evidence pointers)

### TRUTHFULLY

- **T1 (blocker, fixed) — the treatment arm did not record its own
  unproductive draws; the control did.** Old
  `phased_orchestrator._collect_results` kept only `status=="completed"`
  trajectories with non-empty `traj.patch`; `run_branching.run_single_instance`
  wrote per-trajectory prediction rows only for those. The vanilla driver
  (`run_resample_baseline.run_temperature`) appends an empty-patch row for
  every resample, including exceptions. **Confirmed on the real pilot run:**
  5/10 instances have fewer patch entries than `total_trajectories`
  (sympy-12096 4/5, sympy-12481 6/9, sympy-15345 8/11, sympy-16766 4/5,
  sympy-18189 2/5, sympy-23534 3/5). *Fix:* `collect_patch_entries` (one entry
  per completed-or-failed draw, patch normalized to "") +
  `build_predictions` (one row per draw incl. empty; best-of over non-empty
  only), both pure and stage-tested.
- **T2 (major, consequence of T1, fixed) — iteration 5's productivity
  diagnostic could not fire on the arm it was built for.** `nonempty_patch_fraction`
  reads the predictions; the treatment's empty draws never reached that
  artifact, so the treatment's fraction was structurally ≈1.0 and the
  H1-productivity-confound instrument was inert exactly where the confound
  favored the treatment. Truthful in both arms post-fix.
- **T3 (moderate, fixed) — silent data loss on failed trajectories.**
  `_capture_patch_if_missing` runs for failed trajectories too (the run loop
  calls it after the status loop exits), but `_collect_results` then discarded
  the captured patch because the status was `failed`, not `completed`. A real,
  possibly passing treatment patch could vanish (this direction hurts the
  treatment; both directions are now closed).
- **T4 (checked, clean) — legacy eval files have no remaining consumers.**
  Verified by grep: every consumer of `trajectory_eval_*` /
  `predictions_all_trajectories` is the current driver/metric/test set;
  `results/branching/figures/generate_pub_figures.py` is a one-off pilot
  visualization, `collect_results.py` serves the legacy ReactAgent summary
  only. The pilot eval/predictions stay smoke-only (now doubly so: the
  resample driver WARNS on the pilot metadata's row mismatch).

### MATHEMATICALLY

- **M1 (the T1 bias, quantified).** Let the treatment produce k draws, c of
  them passing, but record only r ≤ k rows (duds dropped). Vanilla records all
  k. Metric-time k\* = min(r, k) = r, and the comparison becomes
  pass@r(r, c) vs pass@r(k, c_v): the treatment is scored as if it had
  produced only its productive draws while vanilla is subsampled at the
  treatment's shrunken r. Worked example at the pilot's worst case
  (sympy-18189, r=2, k=5): treatment 1 pass in 5 draws → recorded
  pass@2(2,1)=1.0; vanilla 1 pass in 5 → pass@2(5,1)=0.4; manufactured gain
  +0.6 where the truth is a tie at matched k=5 (pass@5 = 1.0 both, and at any
  common k the true gain is 0). H1 inherits the same direction: treatment
  empties leave both its signature pool and its n, vanilla's empties stay in
  its n and depress its rarefied distinct count. Both confirmatory endpoints
  were biased toward the treatment; the bias is now structurally removed
  rather than estimated.
- **M2 (major, fixed) — stale-orphan prediction rows on shrinking re-runs.**
  `load_predictions_by_tid` kept last-occurrence-per-tid across ALL run
  batches, while the eval driver scores only the LAST primary-delimited
  batch. A branching re-run that produced fewer clusters (cluster count is
  run-dependent) left the old run's `t0_strategy_2` row in the diversity
  pool: `per_instance_table.distinct_patches`, `n_nonempty_patches`, and
  `compare()`'s rarefaction (via `len(pa)` and the signature pool) counted a
  patch the eval record never scored. Tid-joined analyses
  (`set_valued_evidence`, `selected_pass_at_1`, τ-sweep) were safe — they key
  on the eval record. *Fix:* the predictions loader now batch-splits per
  instance exactly like `eval_all_trajectories.load_latest_trajectories`
  (tested: `test_load_predictions_drops_orphan_tids_from_prior_branching_run`).
- **M3 (checked, no defect) — standing re-derivations.** pass@k product form
  (reduces to any-pass at k=n; hypergeometric identity needs no i.i.d. for
  the random-subset question); rarefaction as Σ_sig pass@k(n, m_sig, k);
  exact sign-flip enumeration incl. identity mask (p never 0); floor
  2^(1+z−n) (all-zero → 1.0); seeded bootstrap. `compare()`'s rarefaction
  k\* = min(k_a, k_b, len(pa), len(pb)) — the len() terms and the eval ks can
  no longer disagree post-T1/M2, removing a silent inconsistency channel
  rather than adding a new rule.
- **M4 (observation, instrumented + disclosed, no code change) — the entropy
  signal SATURATED in the pilot.** Every pilot instance's 5 proposals
  clustered all-singleton → entropy = ln 5 ≈ 1.609, the grid maximum; the τ
  sweep on the pilot shows branch_rate 1.0 for all τ < 1.6094 and 0.0 at
  1.6094. If the real runs reproduce this, the gate is *uninformative at this
  substrate/threshold*: "branch iff entropy > τ" degenerates to "always
  branch," and the entropy-stratified analysis (R5.2) has a single stratum.
  This is now threat 11 in RESULTS §6 with an explicit no-post-hoc-retuning
  rule (a retuned entailment threshold or τ is a new exploratory cell, never
  a quiet recalibration of the confirmatory one). The artifacts already
  expose it (per-instance partitions + realized N in the sweep output).

### PHILOSOPHICALLY

- **P1 — the lesson that generalizes iteration 5's:** pre-registration is only
  as strong as artifact semantics, and artifact semantics are only as strong
  as their *weakest producer*. Iteration 5 hardened the eval layer; the
  producer one layer up could still redefine k, and did, in the favorable
  direction, on half the real pilot. "Matched budget" is a property of the
  recording convention, not just the run design: both arms must count
  unproductive draws under the same rule. The spec now states the completeness
  contract at both layers (R4.1 eval, R7.2 predictions), each carried by stage
  tests.
- **P2 — does the title survive M4?** "Semantic-entropy-gated branching" is
  honest only if the paper reports the realized entropy distribution. If
  entropy ≡ ln K on every instance, the gate is a cluster-count rule that
  never says no, and the paper must say so — the contribution then rests on
  branching-vs-resampling (H1/H2), with the gate reported as a null mechanism
  at this substrate. That is a publishable negative finding under §0's own
  charter; what is NOT acceptable is retuning the entailment threshold after
  seeing saturation and presenting the retuned gate as the pre-registered one.
  Threat 11 now pins this in print before any data exist.
- **P3 — weakest joint a hostile reviewer would press now:** "your clustering
  substrate may be too granular (everything distinct → gate saturates), your
  n is 10, one repo, easy band." The first is now measured-and-disclosed with
  a pre-committed reading rule (threat 11) and was already partially defended
  by the substrate choice (intent summaries, per Wei et al. 2026's
  NLI-weak-on-raw-code finding); the rest are disclosed scope limits movable
  only by more GPU/replication. No design change can spin them, and none was
  attempted.

---

## 3. Steelmanned alternatives (this iteration's decisions)

| Design choice | Strongest alternative | Decision |
|---|---|---|
| Fix predictions completeness at the producer | Synthesize missing rows at eval time from metadata.json | **Producer.** Eval-time synthesis leaves the predictions artifact lying and guesses tids for rows that never existed; iteration 5 already established artifacts-truthful-at-source when it rejected metric-side k patching. |
| k source = `total_trajectories` + consistency warning | Switch k source to len(patches) post-fix | **Keep + warn.** The two agree on clean current-driver runs; switching silently changes semantics for legacy/interrupted artifacts. The warning makes any disagreement loud instead of redefining the source. |
| Exclude interrupted (`active`) trajectories from draws | Count partial draws | **Exclude.** An interrupted run is not a usable arm artifact; counting partials would let a Ctrl-C'd run pass as complete. The metadata mismatch warning surfaces exactly this case. |
| Run-loop failures = failed draws in BOTH arms | Filter infra failures out of the vanilla record | **Symmetric counting.** Infra-vs-model failure is not machine-distinguishable from artifacts; both arms now use one rule, `nonempty_patch_fraction` (now truthful in both) exposes asymmetric failure rates, and the runbook says re-run infra-failed instances before metrics. |
| Report gate saturation; never retune post-hoc | Retune entailment threshold (0.5) now so the gate is informative | **Report, don't retune.** A pre-data retune targeted at making the gate "do something" is gate engineering toward the title claim; the pilot is one (old-driver) run and the threshold is part of the pre-registered configuration. If saturation replicates, it is a reported negative finding about the gate (threat 11), and any retuned configuration is an explicitly exploratory new cell. |
| Loader batch semantics mirror the eval driver | Rotate/truncate predictions per run | **Mirror.** Same reasoning as iteration 5's last-block rule: rotation orphans existing artifacts and adds a run-side failure mode; the parser-side rule is testable and now identical in both consumers. |

Standing decisions re-examined and left in place: trajectory-matched budget
(the conservative direction is *finally true in the artifacts*, not just the
design); τ=0 superset headline; intent-summary clustering substrate;
exact-signature H1 with disclosed direction; fixed-sequence H1→H2 family
(untouched — still no data); 10-easy-SymPy scope (disclosed).

---

## 4. Actions taken this iteration

All verified: **82 pytest pass** (77 → 82; +5 new, 0 removed), `py_compile`
clean on all touched files, `compute_metrics.py` and `tau_sweep.py` re-run
end-to-end on the real `results/branching` artifacts, driver smoke checks
pass.

1. `src/agent/phased_orchestrator.py` — `collect_patch_entries` (new pure
   helper): one patch entry per genuine draw (completed OR failed; patch may
   be ""; `status` recorded), `active`/`branched` excluded with rationale;
   `_collect_results` rewired onto it.
2. `scripts/run_branching.py` — `build_predictions` (new pure helper): primary
   best-of (submitted first, then longest non-empty) + one prediction row per
   draw including empty-patch rows; `run_single_instance` rewired.
3. `scripts/compute_metrics.py` — `load_predictions_by_tid` is run-batch
   aware (last primary-delimited batch per instance, keep-last per tid within
   it), mirroring the eval driver.
4. `scripts/run_resample_baseline.py` — `discover_instances_and_k` warns when
   metadata patch entries disagree with `total_trajectories` (old-driver or
   interrupted artifact).
5. Tests: +5 — `test_collect_patch_entries_one_entry_per_genuine_draw`,
   `test_build_predictions_writes_a_row_for_every_draw`,
   `test_build_predictions_all_empty_and_none`,
   `test_matched_k_discovery_warns_on_patch_row_mismatch`,
   `test_load_predictions_drops_orphan_tids_from_prior_branching_run`.
6. `RESULTS.md` — §3 new "Predictions-record completeness" bullet (contract,
   worked bias example, loader batch rule, diagnostic-now-truthful note);
   §6 new threat 11 (gate-signal saturation, with the no-post-hoc-retuning
   reading rule); threat renumbering (rival baselines → 12).
7. `GOLD_STANDARD.md` — R7.2 extended with predictions-record completeness +
   batch-aware loaders + matched-k metadata warning. Record:
   `spec_amendments/applied_06_predictions_record_completeness.md` (includes
   tripwire check: nothing flips to pass; the removed bias favored the
   treatment, so the evidence bar went up).

Note: the existing `results/branching` artifacts were produced by the
pre-fix driver — already documented smoke-only since iteration 5, and the
resample driver now warns on them mechanically. The pre-registered runs
regenerate predictions, eval, and metrics with the fixed pipeline.

## 5. What still requires a human / GPU

Unchanged in shape: ratify the iteration-6 amendment (and the standing
iteration-3/4/5 ones if not yet ratified); run the pre-registered cell with
the FIXED drivers (`run_branching.py --temperature 0.7 --results-dir
results/strategy_t0.7`, then `run_resample_baseline.py --treatment-dir
results/strategy_t0.7 --temperatures 0.7`), evaluate per arm per instance
with `eval_all_trajectories.py --results-dir <arm_dir>`, then
`compute_metrics.py` / `tau_sweep.py` / `budget_audit.py`. Before launch,
rebase or regenerate the campaign worktree's `run_campaign.py` against the
current main-tree drivers (it predates both the iteration-5 eval-driver
rewrite ratification and this iteration's producer fix — verify it does not
vendor old copies).

## 6. Verdict logic

This iteration found and fixed substantive evidence-pipeline flaws — a
pro-treatment k-deflation at the predictions producer confirmed on half the
real pilot instances (T1/M1), silent loss of failed-trajectory patches (T3),
an inert productivity diagnostic (T2), and stale-orphan diversity rows on
shrinking re-runs (M2) — and surfaced a new disclosed risk (gate saturation,
M4/threat 11). Per the charter, `gold_standard_met` = **false**; the loop
should make one more pass over the post-fix producers (fresh eyes on
`collect_patch_entries`/`build_predictions` wiring through a mocked
end-to-end instance, and on whether the SDLG arm's child-trajectory statuses
flow through the same draw-accounting correctly).
