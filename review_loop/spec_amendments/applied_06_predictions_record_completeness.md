# Applied amendment 06 — predictions-record completeness (R7.2 extension)

**Iteration:** 6
**Channel:** auto-applicable correction (channel 1), with value-level framing
recorded for completeness. The amendment is *derivable* from the spec's own
R4.1: "the Chen estimator's (n, c) must count what the arm *produced*" and
"every genuine trajectory of the run appears in the eval record." The eval
record is constructed from `predictions_all_trajectories.jsonl`; therefore the
identical completeness requirement necessarily holds one layer up, at the
predictions producer. A spec that requires eval-record completeness while
permitting the predictions producer to drop draws is internally inconsistent —
the iteration-5 amendment closed the eval layer and left its sole input open.

## The defect this closes (found this iteration, confirmed on real artifacts)

`phased_orchestrator._collect_results` emitted patch entries only for
trajectories with `status == "completed"` AND a non-empty patch;
`run_branching.run_single_instance` wrote per-trajectory prediction rows only
for those entries. Consequences:

1. **Pro-treatment k deflation.** A treatment trajectory that failed or
   produced no diff vanished from the predictions → eval record → Chen (n, c),
   while the vanilla resample driver records an empty-patch row for every
   unproductive resample (`run_resample_baseline.run_temperature` appends ""
   on exception). On the real pilot run, **5 of 10 instances** had fewer
   prediction rows than trajectories (e.g. sympy-18189: 5 trajectories, 2
   rows). Worked bias example: treatment 5 draws, 1 pass, 2 recorded →
   pass@k\*(2,1) = 1.0 vs vanilla pass@2(5,1) = 0.4 — a +0.6 headline "gain"
   manufactured by dropping the treatment's own duds. The H1 rarefied-distinct
   comparison inherits the same bias (treatment empties leave its n; vanilla
   empties stay in its n).
2. **Silent data loss (R7.2 violation in the other direction).** A patch
   captured by `_capture_patch_if_missing` on a trajectory later marked
   `failed` was discarded at collection — a real, possibly passing patch
   dropped from the treatment's record.
3. **Diagnostic corruption.** Iteration 5's `nonempty_patch_fraction`
   productivity diagnostic was structurally ≈1.0 for the treatment arm — its
   empty draws never reached the artifact the diagnostic reads — so the
   confound instrument added in iteration 5 could not fire on the arm it was
   most needed for.
4. **Stale-orphan rows on shrinking re-runs (companion fix).** The metric
   loaders kept last-occurrence-per-tid across ALL run batches; a branching
   re-run with fewer clusters left the prior run's orphan tids in the
   diversity pool while the eval driver (correctly) scored only the last
   batch. Loaders are now batch-aware, mirroring
   `eval_all_trajectories.load_latest_trajectories`.

## Spec edits

- **R7.2**: added the predictions-record completeness clause (one prediction
  row per genuine draw in every arm, empty-patch rows for failed/patchless
  draws, failed-trajectory patches not discarded), the run-batch-aware loader
  rule, and the matched-k driver's metadata-consistency warning.

## Implementation (same iteration — nothing is left as an unfunded mandate)

- `src/agent/phased_orchestrator.py::collect_patch_entries` (new, pure,
  stage-tested): one entry per trajectory with status completed OR failed,
  patch normalized to ""; statuses `active` (interrupted run) and `branched`
  (legacy parent) excluded with rationale in the docstring.
- `scripts/run_branching.py::build_predictions` (new, pure, stage-tested):
  primary best-of row (submitted first, then longest **non-empty**) + one row
  per draw including empty ones.
- `scripts/compute_metrics.py::load_predictions_by_tid`: last
  primary-delimited batch per instance, keep-last per tid within it.
- `scripts/run_resample_baseline.py::discover_instances_and_k`: warns on
  patch-entries vs total_trajectories mismatch (k source unchanged:
  total_trajectories).
- Tests: +5 (`test_collect_patch_entries_one_entry_per_genuine_draw`,
  `test_build_predictions_writes_a_row_for_every_draw`,
  `test_build_predictions_all_empty_and_none`,
  `test_matched_k_discovery_warns_on_patch_row_mismatch`,
  `test_load_predictions_drops_orphan_tids_from_prior_branching_run`).
  82 pass total.

## Tripwire check (ratchet v2 clause ii)

No rubric item flips to `pass` for the artifact as it existed at edit time —
the opposite: the existing `results/branching` predictions are now explicitly
non-compliant with R7.2 (they remain smoke-only, as documented since
iteration 5) and the resample driver will WARN on them. The evidence bar for
the headline claim goes up (the treatment must now carry its unproductive
draws in its own denominator). The bias being removed favored the treatment,
so this amendment can only reduce a future headline gain, never inflate it.

## Rejected alternatives

- **Fix at eval time (synthesize missing rows from metadata.json):** leaves
  the predictions artifact lying, requires the eval driver to guess tids for
  rows that never existed, and breaks the "artifacts truthful at the source"
  principle established in iteration 5 (same reasoning as rejecting
  metric-side k patching there).
- **Read k from len(patches) instead of total_trajectories in the resample
  driver:** post-fix the two agree on clean runs; switching the k source
  would silently change semantics for legacy artifacts and interrupted runs.
  Kept total_trajectories + a loud consistency warning instead.
- **Count interrupted (`active`) trajectories as draws:** an interrupted run
  is not a usable arm artifact at all (its instance must be re-run); counting
  partial draws would let a Ctrl-C'd run masquerade as a complete one. The
  consistency warning surfaces exactly this case.
- **Exclude infra-failure resamples from the vanilla record (symmetric
  concern):** distinguishing infra failures from model failures automatically
  is not possible from the artifacts; both arms now count run-loop failures
  as failed draws identically, and `nonempty_patch_fraction` (now truthful in
  both arms) plus the run logs expose any asymmetric failure-rate so the
  analyst can re-run affected instances before metrics.
