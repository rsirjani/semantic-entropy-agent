# Scrutiny record — iteration 7 (first-principles design review)

Charter: principal-scientist review BEFORE the headline GPU runs. Iteration 6
queued two checks for this pass: a mocked end-to-end run through the rewired
producers (`collect_patch_entries → build_predictions → eval driver →
compute_metrics`), and verification that SDLG-arm child trajectories flow
through the new draw accounting. Both were done. The end-to-end chain is
sound; the SDLG check found that the iteration-6 contract — every draw the
arm paid for appears in the record — was still violated at the **fork
creation** step, in both treatment arms, with one sub-case (an alternative
that submits during injection) silently **discarding a completed, possibly
passing treatment patch**. A symmetric artifact-hygiene gap was found in the
resample arm (orphan rows on shrinking-k re-runs, unfixable parser-side
because the file has no batch delimiters). All fixed at the producers, tested
(82 → 88 passing), and the spec extended (applied_07).

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

**Ideal evidence:** unchanged from iterations 4–6 (one confirmatory cell
mirroring the causal chain, exact small-n inference with printed power floors,
mechanism-independent diversity measurement, budget accounting showing the
control was not starved, artifacts that cannot redefine k between run and
metric) **plus this iteration's closing clause: the artifact record must begin
at the *decision to draw*, not at the first trajectory object that survives
construction.** Iteration 5 hardened the eval layer, iteration 6 the
predictions producer; iteration 7 closes the layer above both — the fork
itself.

**Minimal sufficient experiment set:** unchanged — (1) strategy arm T=0.7 τ=0
superset; (2) matched-k vanilla T=0.7; (3) exploratory sweep T∈{0.2, 1.0};
(4) SDLG arm T=0.7. τ ablation post-hoc, zero GPU.

**Did the repo serve this claim or a weaker one?** The metric, eval, and
predictions layers served it. The fork paths did not: a strategy fork whose
container failed, or an SDLG child whose clone/injection failed, vanished
from `manager.trajectories` — hence from `total_trajectories`, the
predictions rows, and the metric-time k — while the vanilla driver records
every crashed resample as an empty-patch draw. "Matched trajectory budget"
was again true of the design and not guaranteed by the bookkeeping, this time
at the fork boundary.

---

## 2. Findings (with evidence pointers)

### TRUTHFULLY

- **T1 (major, fixed) — fork-creation failures were invisible draws in both
  treatment arms.** `_create_lazy_trajectory` returned `None` on any exception
  and the run loop `continue`d (old `phased_orchestrator.py` — strategy arm);
  `_clone_for_sdlg` returned `None` from a blanket `except Exception` (SDLG
  arm). Neither registered anything in `manager.trajectories`, so the failed
  draw was absent from `total_trajectories`, the metadata patches list, and
  the predictions file — while `run_resample_baseline.run_temperature`
  records an empty-patch row for every failed resample, including exceptions.
  This is iteration 6's asymmetry one layer earlier. *Fix:*
  `_register_failed_draw` registers a failed empty-patch placeholder
  trajectory (agent=None; container reaped immediately; `Trajectory.save`
  no-ops on placeholders), wired into both fork paths.
- **T2 (major, fixed) — an SDLG alternative that SUBMITS during injection was
  silently discarded.** `inject_and_execute → execute_actions` raises
  `Submitted` when the injected response contains the submit command;
  `_clone_for_sdlg`'s blanket handler logged it as "Failed to clone," dropped
  the child (a *completed draw with a real, possibly passing patch* — silent
  data loss against the treatment, R7.2), and leaked its container
  (`cleanup_all` only iterates registered trajectories). *Fix:*
  `_inject_alternative` classifies the injection outcome — active /
  completed-with-submission / failed — and `_clone_for_sdlg` registers the
  child in every case, capturing any working-tree diff on failed injections
  before reaping. The SDLG fork index now advances on every attempt so a
  registered failed draw's id is never overwritten by the next cluster
  (`_apply_sdlg`).
- **T3 (minor, fixed) — RESULTS.md threat 11 misstated the pre-registered
  entailment threshold.** The text said "the configured entailment threshold
  (0.5)"; `configs/branching.yaml` sets `entailment_threshold: 0.7` (raised
  pre-pilot from 0.5 "to prevent over-merging"). A methods section quoting
  0.5 while the runs use 0.7 would be a factual error in print; worse, the
  direction matters — a *higher* threshold merges less, making the observed
  all-singleton saturation *more* likely, and the no-post-hoc-retune rule
  must therefore cover this knob too. RESULTS threat 11 now states 0.7, the
  raise's provenance, the direction, and extends the no-retune rule to it.
- **T4 (checked, clean) — the mocked end-to-end chain (iteration 6's queued
  check).** Synthetic trajectories (submitted + duplicate + distinct +
  failed-with-patch + failed-empty + interrupted-active) flow through
  `collect_patch_entries → build_predictions → load_latest_trajectories →
  deduplicate/propagate → trajectory_eval JSON → load_predictions/load_eval →
  per_instance_table → compare` and produce exactly the right numbers: k=5
  both arms, primary dropped everywhere, duplicate inherits its
  representative's outcome, interrupted draw excluded, rarefied distinct gain
  3−1=2 at k\*=5, non-empty fraction 0.8/0.8, no k mismatch
  (`tests/test_end_to_end_mocked.py::test_mocked_end_to_end_treatment_vs_vanilla`).
  A second test drives a shrinking re-run through BOTH consumers and confirms
  no orphan survives.

### MATHEMATICALLY

- **M1 (T1/T2 bias, quantified).** Same mechanism as iteration 6's M1, one
  layer earlier: a dropped failed fork shrinks the treatment's recorded r
  below its true k, and the comparison becomes pass@r(r, c) vs vanilla's
  pass@r(k, c_v) — the treatment is scored only on draws that survived
  construction while vanilla carries its crashes. The Submitted-at-injection
  sub-case points the other way (a passing treatment patch deleted lowers c).
  Both directions violate the completeness contract; both are now closed
  structurally, not estimated.
- **M2 (moderate, fixed) — resample-arm orphan rows on shrinking-k re-runs.**
  Iteration 6 made the loaders batch-aware using the branching file's
  best-of "primary" rows as delimiters. The resample arm's
  `predictions_all_trajectories.jsonl` has **no** delimiter rows, so the
  last-batch rule is vacuous there: after a treatment re-run lowers k for an
  instance, a resample re-run (without `--skip-existing`) appends k′ < k new
  rows and keep-last-per-tid resurrects the old run's surplus `runN` rows
  into the vanilla arm's metric-time k, rarefaction denominator, and
  diversity pool. No parser can distinguish those batches; the fix must live
  at the producer. *Fix:* `replace_instance_rows` — per-instance row
  replacement mirroring the driver's own primary-file semantics (tested:
  `test_resample_replace_instance_rows_drops_stale_rows`).
- **M3 (checked, no defect) — standing re-derivations.** pass@k product form
  ↔ 1 − C(n−c,k)/C(n,k) via the C(n−k,c)/C(n,c) symmetry, with correct
  guards (c≤0 → 0, n−c<k → 1); rarefaction = Σ_sig pass@k(n, m_sig, k)
  (hypergeometric inclusion); exact sign-flip enumeration at n≤20 includes
  the identity mask (p ≥ 2^−n, never 0); tie floor 2^(1+z−n) is a valid
  lower bound (the two global sign choices on nonzero entries × 2^z zero
  patterns); seeded bootstrap. τ-sweep gate `entropy > τ + 1e-9` matches the
  orchestrator's strict `>` at achievable-grid τ values, and
  `dominant_trajectory_id`'s tie-break (lowest index) matches
  `max(clusters, key=len)`.
- **M4 (minor, docstring tightened) — the no-rarefaction claim for mean
  pairwise distance is exact only without empty patches.** With empties in
  the pool, the number of non-empty pairs varies per subset, so the expected
  subset *mean* (a ratio of random sums) need not equal the full mean. The
  code never computes subset means — each arm reports its full-sample mean —
  and the metric is descriptive, never confirmatory; the docstring now states
  the caveat instead of overclaiming linearity
  (`src/evaluation/metrics.py::mean_pairwise_distance`).

### PHILOSOPHICALLY

- **P1 — the generalizing lesson.** Three iterations have now moved the same
  invariant up the stack: iteration 5 (the eval record counts what the arm
  produced), iteration 6 (the predictions record counts what the arm
  produced), iteration 7 (the *run record itself* counts every draw the arm
  decided to make). The stable formulation, now in R7.2: **draw accounting
  starts at the fork decision.** Any accounting that begins later — at object
  construction, at completion, at patch capture — is an opportunity for
  asymmetric attrition, and attrition is never direction-neutral when only
  one arm's failures are recorded.
- **P2 — does anything still bound k off-record?** Checked: the trajectory
  cap (`can_branch`) is applied *before* a draw is decided (clusters beyond
  the cap are never drawn — correctly not failed draws, and the cap event is
  traced); the `--max-k` resample cap is reported at metric time as a k
  mismatch; instance-level crashes leave no metadata and are loudly excluded
  (runbook: re-run before metrics). The remaining attrition channels are all
  either pre-decision or disclosed.
- **P3 — weakest joint a hostile reviewer would press now:** unchanged in
  kind from iteration 6 — gate saturation (threat 11, now with the correct
  threshold value and the no-retune rule extended to the entailment knob),
  n=10 / one repo / easy band (disclosed scope), and the τ=0 headline (the
  gate's adaptivity is a separate, post-hoc-evaluable claim via the sweep).
  No design change can spin these; none was attempted. The smoke-run of the
  τ sweep on the pilot again shows branch_rate 1.0 below 1.6094 — if the
  pre-registered runs reproduce it, the gate is reported as uninformative at
  this substrate per threat 11.

---

## 3. Steelmanned alternatives (this iteration's decisions)

| Design choice | Strongest alternative | Decision |
|---|---|---|
| Register failed forks as placeholder trajectories at the producer | Synthesize missing rows at eval/metric time from fork logs | **Producer.** Same artifacts-truthful-at-source principle as iterations 5–6; synthesized tids would be guesses and the predictions file would keep lying. |
| Placeholder = `Trajectory(agent=None)` | A parallel `failed_draws` list in metadata | **Placeholder.** One code path: `collect_patch_entries`, `total_trajectories`, the mismatch warning, and `cleanup_all` all see the draw with zero new plumbing; `save()` no-ops, so no schema change. |
| `Submitted`-at-injection child recorded completed, container reaped | Let the child re-enter the run loop | **Record + reap.** The agent submitted — the draw is finished by definition; re-running it would double-spend budget and desync step accounting. |
| Resample file: per-instance replace at the producer | Parser-side batch rule (mirror iteration 6) or synthetic delimiter rows | **Producer replace.** The file has no delimiters and never did; synthetic delimiters change the schema every consumer parses; replacement matches the driver's own primary-file semantics and is loss-free for other instances. |
| Failed forks do NOT count against `max_trajectories` (`can_branch` counts active+completed only, unchanged) | Count them against the cap | **Keep.** The cap bounds *concurrent real compute*; a failed creation consumed no run budget beyond its crash, and there is no retry logic that could exploit the headroom. Noted, not changed. |
| Fix `mean_pairwise_distance` docstring only | Add a rarefaction-style correction for the pairwise metric | **Docstring.** The metric reports full-sample means per arm (no subsetting happens), is descriptive-only, and inventing a correction for a non-confirmatory companion adds analyst degrees of freedom for no inferential gain. |
| Threat 11 states the real threshold (0.7) + provenance + direction | Quietly fix the number | **Full disclosure.** The raise (0.5→0.7) happened pre-pilot and is part of the frozen configuration; saying so, plus the saturation-direction implication, is what makes the no-retune rule auditable. |

Standing decisions re-examined and left in place: trajectory-matched budget
(conservative direction, now enforced from the fork decision down);
τ=0 superset headline with post-hoc sweep; intent-summary clustering
substrate; fixed-sequence H1→H2 family (untouched — still no data);
majority-signature selector with degeneracy disclosure; 10-easy-SymPy scope.
Legacy `TrajectoryManager._create_branch` shares T2's blanket-except pattern
but belongs to the legacy `branching_orchestrator` path, which no
pre-registered arm uses — noted, not modified (changing dead code would only
blur the audit trail).

---

## 4. Actions taken this iteration

All verified: **88 pytest pass** (82 → 88; +6 new, 0 removed), `py_compile`
clean on all touched files, `compute_metrics.py`, `tau_sweep.py`, and
`budget_audit.py` re-run end-to-end on the real `results/branching` pilot
artifacts.

1. `src/agent/trajectory.py` — `agent`/`env` optional (None only for
   failed-at-creation placeholders, documented); `save()` no-ops on
   placeholders.
2. `src/agent/phased_orchestrator.py` — `_register_failed_draw` (new),
   `_inject_alternative` (new, classifies injection outcome incl. the
   Submitted case), `_create_lazy_trajectory` and `_clone_for_sdlg` rewired
   to always leave a draw record; `_apply_sdlg` advances the fork index on
   every attempt.
3. `scripts/run_resample_baseline.py` — `replace_instance_rows` (new);
   `run_temperature` replaces an instance's all-trajectories rows on re-run
   instead of appending.
4. `tests/test_end_to_end_mocked.py` (new file, 6 tests) — mocked end-to-end
   treatment-vs-vanilla chain; shrinking-re-run last-batch propagation;
   placeholder safety + counting; `_register_failed_draw`;
   `_inject_alternative` outcome classification (incl. Submitted);
   resample per-instance replacement.
5. `src/evaluation/metrics.py` — `mean_pairwise_distance` docstring caveat
   (M4).
6. `RESULTS.md` — §3 new "Draw accounting starts at the fork decision"
   bullet; §6 threat 11 corrected to the real entailment threshold (0.7)
   with provenance, direction, and the no-retune rule extended to that knob.
7. `GOLD_STANDARD.md` — R7.2 extended with the fork-decision draw-accounting
   clause and the producer-side re-run-safety rule for delimiter-less
   artifacts. Record: `spec_amendments/applied_07_draw_accounting_at_fork.md`
   (derivation, rejected alternatives, tripwire check: nothing flips to pass;
   the bar rose — the treatment now carries failed fork attempts in its own
   denominator and a silent treatment-patch-loss channel is closed).

Note: the existing `results/branching` artifacts remain smoke-only
(pre-iteration-6/7 drivers; they mechanically trigger the resample driver's
mismatch warning). The pre-registered runs regenerate everything with the
fixed pipeline.

## 5. What still requires a human / GPU

Unchanged in shape: ratify the iteration-7 amendment (and any unratified
3–6 ones); run the pre-registered cell with the FIXED drivers
(`run_branching.py --temperature 0.7 --results-dir results/strategy_t0.7`,
then `run_resample_baseline.py --treatment-dir results/strategy_t0.7
--temperatures 0.7`), evaluate per arm per instance
(`eval_all_trajectories.py --results-dir <arm_dir>`), then
`compute_metrics.py` / `tau_sweep.py` / `budget_audit.py`. Before launch,
rebase/regenerate the campaign worktree's `run_campaign.py` against the
current main-tree drivers (it now predates the iteration-5, -6, AND -7
fixes). After the runs: check the realized entropy distribution first
(threat 11), read H1 against `nonempty_patch_fraction`, report
`min_achievable_p` beside both sign-flip p-values.

## 6. Verdict logic

This iteration again found and fixed substantive evidence-pipeline flaws —
invisible failed-fork draws in both treatment arms (T1), silent discard of a
completed submitted SDLG child plus a container leak (T2), resample-arm
orphan rows unfixable parser-side (M2), and a factual threshold misstatement
in the threats section (T3). Per the charter, `gold_standard_met` =
**false**; the loop should make one more pass with fresh eyes on the new
fork-path code (`_register_failed_draw` / `_inject_alternative` wiring under
real exception flows, and whether any remaining lifecycle event — e.g. the
trajectory cap, `KeyboardInterrupt` mid-fork — can still create an
unrecorded asymmetry between the arms).
