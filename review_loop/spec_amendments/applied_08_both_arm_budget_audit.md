# Applied amendment (iteration 8): R6.3 — per-arm accounting must mean BOTH arms

**Channel:** value-level design amendment (scrutiny charter, ratchet v2 channel 2),
with a strong derivable component.

## What changed

R6.3 now states that the budget-fairness audit tool must read **each arm's actual
artifact layout** — treatment (`<iid>/metadata.json`) AND control
(`<iid>/run<idx>/<iid>/metadata.json`, draws keyed by the predictions trajectory id
`run<idx>`) — stage-tested on both, and that the documented workflow must audit the
control alongside the treatment.

## Why (argued from the science)

The design's central fairness claim (RESULTS §2.2 "matching direction") is:
*at matched trajectory count the control receives at least as much total compute,
so a treatment win cannot be a compute artifact.* R6.3 exists to turn that claim
from an assertion into a measurement, and `budget_audit.py`'s own fairness note
instructs the reader to "compare it across arms at matched k."

But the audit tool's discovery globs (`<results_dir>/*/metadata.json`,
`<results_dir>/*/trajectory_*.traj.json`) only matched the **treatment** layout.
The control arm nests each resample's orchestrator output one level deeper
(`<iid>/run<idx>/<iid>/...`), so running the audit on the control returned
`n_instances=0` — the cross-arm comparison the spec demands was *physically
impossible* with the shipped tooling, and the campaign driver accordingly audited
only the treatment (while its docstring claimed "both arms"). A fairness audit
that can only see one arm is not an audit.

This is the same class of defect iterations 5–7 chased through the eval and
predictions layers: a contract ("per-arm accounting") enforced at the claim level
but violated by a producer/consumer's layout assumptions one layer down.

## Implementation (same iteration — nothing flips to pass without work)

- `scripts/budget_audit.py`: layout auto-detection (`detect_layout`) +
  `collect_draw_records` normalizing both layouts into per-instance draw records;
  control draws keyed `run<idx>` so steps/tokens join the eval record's tids;
  crashed resamples (no metadata) counted in `n_draws_missing_metadata` (their
  empty-patch draws still exist in the eval record); per-instance totals summed
  across runs.
- `scripts/run_campaign.py`: new `budget_audit_control` step per spec, distinct
  output file.
- `RESULTS.md` §5: both-arm audit commands documented.
- Tests: `test_budget_audit_reads_control_resample_layout` (synthetic control
  tree incl. a metadata-less crashed run, over-cap passing resample, token sums),
  `test_budget_audit_treatment_layout_detected`,
  `test_build_steps_audits_both_arms`.
- Treatment-path behavior verified unchanged: smoke on `results/branching`
  reproduces the documented 53-trajectory / 12,882,383-token numbers exactly.

## Tripwire check (the one forbidden move)

No rubric item flips to `pass` for the pre-existing artifact: R6.3 gains an
obligation that did not exist before, and the obligation is discharged by new
work (code + tests) in the same iteration. The evidence bar for the headline
claim goes UP — the fairness direction claimed in §2.2 is now *checkable* on the
arm where it could fail.

## Rejected alternatives

- **Leave the tool treatment-only and compute the control's tokens ad hoc after
  the runs.** Rejected: ad-hoc analysis of the fairness-critical number is
  exactly the unauditable path the rubric exists to forbid; it also would have
  left the campaign driver's "both arms" docstring false.
- **Change the resample driver to write a flat treatment-style layout.** Rejected:
  touches the GPU-run code path right before the campaign for a consumer-side
  problem; per-run orchestrator dirs exist so concurrent resamples don't collide
  on disk (documented in `run_one_resample`), and the audit can read the existing
  truth without changing what producers write.
- **Have the audit read tokens from the predictions/eval records instead of
  transcripts.** Rejected: those records do not carry token usage; the
  transcripts are the only truthful token source (R6.3's accounting basis).
