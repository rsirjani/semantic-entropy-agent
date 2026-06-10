# Applied amendment (iteration 11): eval-outcome integrity + stale-report immunity (R4.1)

**Channel:** value-level design amendment (scrutiny charter, ratchet policy v2) —
arguably channel 1 (a logically derivable correctness hole), applied as channel 2
with the full record either way.

## What changed in GOLD_STANDARD.md

R4.1 ("eval-record completeness") gains two clauses:

1. **Eval-outcome integrity.** Every `resolved` in the eval record must be a
   genuine harness verdict. A missing harness report must be *classified*:
   patch-apply failure and test timeout are patch-attributable failed draws
   (recorded with their reason); any other cause is an eval-infrastructure error
   that must abort the eval step loudly **without writing the eval record**.
2. **Stale-report immunity.** Eval run ids must embed a content hash of the
   patch, because the harness returns an existing `report.json` keyed by
   (run_id, model, instance) *without re-evaluating*.

## The science (why this is required, not nice-to-have)

Verified in the installed harness source
(`swebench/harness/run_evaluation.py`):

- Lines 253–273: `run_instance` catches `EvaluationError`, `BuildImageError`,
  **and bare `Exception`**, logs, and returns `completed: False` — no report is
  written, the eval run continues. The error never reaches our driver's exit
  code.
- Line 118–123: if `report_path.exists()`, the harness **returns the existing
  report** without running anything. Patch content is not part of the key.

Our eval driver (`scripts/eval_all_trajectories.py`) previously mapped "no
report.json" to `resolved: false` with exit code 0 and wrote the
`trajectory_eval_<iid>.json` resume marker. Consequences:

- An eval-time Docker/build/container flake was **silently scored as a test
  failure**. This corrupts the Chen estimator's (n, c) for whichever arm the
  flake hits — it can flip a per-instance any-pass, fabricate or destroy an
  off-mode-recovery candidate (R5.4), and shift the H2 gain — and the resume
  marker freezes the fabricated verdict forever.
- After a treatment re-run changed a trajectory's patch (same trajectory id),
  re-evaluation **inherited the previous patch's verdict** from the stale
  cached report, because the run id `{arm}_traj_{tid}` was patch-blind.

This is distinct from the iteration-6/7 decision that *generation-time*
infra failures count as failed draws in both arms (a crashed draw consumed
budget; symmetric counting is correct there, and infra-vs-model is not
distinguishable from those artifacts). At the **evaluation** layer the budget
is already spent, the patch exists, and its ground-truth verdict is definite —
recording `false` on a measurement failure is fabrication, not accounting.
And unlike the generation layer, the eval layer largely IS machine-
distinguishable: report present = verdict; `>>>>> Patch Apply Failed` marker
= genuinely unappliable patch (failed draw); timeout marker = hanging patch
(failed draw per SWE-bench convention); anything else = environment.

## Implementation (discharged same-iteration)

- `scripts/eval_all_trajectories.py`: `patch_run_id` (SHA-1 content hash in the
  run id), `classify_missing_report` (apply-fail / timeout / raise
  `EvalOutcomeError`), model-name `/`→`__` normalization mirroring the harness,
  `fail_reason` propagated to duplicate rows, `main` exits 3 without writing the
  record on infra errors.
- `scripts/compute_metrics.py`: `pred_eval_count_mismatch` diagnostic (named,
  never silently `min()`-ed).
- Tests: 7 new in `tests/test_eval_driver.py` (content-keyed run id, normalized
  report path, apply-fail and timeout classified as genuine failed draws, infra
  error raises, classification priority, `main` exits 3 with no record written),
  1 in `tests/test_compute_metrics.py`, 1 in `tests/test_budget_and_figures.py`
  (rarefied comparison figure). 130 pass (121 → 130).

## Rejected alternatives

- **Keep `resolved: false` but add an `eval_error` flag and let the metrics
  skip flagged rows.** Rejected: a skipped row changes the metric-time k
  silently (the very class of bug R4.1 exists to prevent); and a flag in a
  written record still gets frozen by the resume-marker skip. Stop-loudly is
  the established failure philosophy of every other step.
- **Preflight Docker before each eval step.** Rejected as the primary fix: a
  preflight catches a dead daemon but not per-instance build/container/network
  flakes mid-run — exactly the errors the harness swallows. (A preflight may
  still be added as a convenience; it is not sufficient.)
- **Re-derive the verdict from the harness's top-level summary JSON
  (`error_ids` / `resolved_ids`).** Rejected: the summary file is written per
  harness invocation in CWD with a model-derived name, is itself overwritten
  across invocations, and pools apply-fail with infra errors in `error_ids` —
  strictly less information than the per-instance logs already on disk.
- **Force re-evaluation by deleting old report dirs instead of content-keyed
  run ids.** Rejected: deletes the legitimate cache (identical patches would
  re-run for hours after every interruption) and requires remembering to
  delete — the content key is self-enforcing.
- **Treat timeouts as infra errors.** Rejected: a patch that makes the test
  suite hang is unresolved under SWE-bench convention; classifying it as infra
  would block the campaign on a genuinely bad patch.

## Tripwire check (the one forbidden move)

This amendment **adds** obligations to R4.1; it flips no rubric item to pass
for the pre-existing artifact (the obligations were implemented and tested in
the same iteration), and it does not reduce the evidence required for the
headline claim — it protects that evidence from fabrication. No GPU-run
evidence is graded as gathered by this edit.
