# Applied amendment (iteration 10): analyst data-plane integrity + artifact-encoded H1→H2 gate

**Channel:** value-level design amendment (scrutiny charter, ratchet policy v2).
**Spec edit:** `GOLD_STANDARD.md` R6.5, adaptive-execution boundary — two added
obligations:

1. The campaign driver must **verify, not assume**, that the analyst process
   modified no existing results/decision artifact: an integrity fingerprint of
   the data plane (results artifacts + prior decision files) taken before and
   checked after each analyst invocation, with any change stopping the
   campaign loudly.
2. The fixed-sequence H1→H2 gate must be **encoded in the metrics artifact
   itself** (H2's confirmatory-vs-descriptive status derived from H1's p in
   the output JSON), not carried solely in prose.

## Rationale (argued from the science)

R6.5's pre-registration boundary is only as strong as its enforcement. The
campaign already guards the **code plane** (git-porcelain diff across the
analyst window, `unexpected_tree_changes`) — but that check deliberately skips
`results/` and `campaign_decisions/`, because the campaign's own logs write
there. Those trees are the **data plane**: the metrics/eval/predictions
artifacts every subsequent scheduling decision reads, the numbers RESULTS §5
is filled from, and the decision files that *are* the R6.5 audit chain. The
driver's own docstring states the design's standard: the analyst runs with
permissions skipped, and "prompts are not enforcement." By that standard, an
analyst edit to a metrics JSON (steering every later phase toward a chosen
narrative) or to a *prior* decision file (rewriting the audit chain) was
undetectable. The amendment closes the asymmetry: a guard that protects the
experiment code but not the experiment *data* protects the wrong half.

The second clause closes the analogous reading-side hole: the gatekeeping
family (H2 confirmatory only if H1 rejects) existed only in prose (RESULTS
§2.2, the analyst prompt). The artifact printed two flat p-values; a reader —
including the campaign's own analyst, whose decisions are data-dependent —
could treat an H2 p < 0.05 as confirmatory when the gate never opened. A
multiple-comparison control that depends on every reader remembering a prose
rule is weaker than one the artifact itself states.

## Implementation discharged same-iteration

- `scripts/run_campaign.py::artifact_fingerprint` / `changed_artifacts` —
  SHA-256 over everything under `results/` (excluding `results/campaign/`,
  the campaign's own mutable log/state area) and `campaign_decisions/`
  (excluding the one decision file the current call is expected to write);
  snapshot taken after stale-decision archiving, verified after the analyst
  subprocess; any delta returns a stop with the changed paths named.
- `scripts/compute_metrics.py::compare` — `confirmatory_family` block:
  H1 p + rejects flag at family-wise α=0.05, H2 p + derived status, and a
  note that confirmatory status applies only in the pre-registered cell.
- Tests: `test_analyst_artifact_tamper_stops_campaign` (metrics tamper AND
  prior-decision-file tamper both stop), `test_analyst_clean_run_passes_artifact_guard`
  (decision file + campaign's own log area are legitimate),
  `test_confirmatory_family_gate_open_and_closed`.
- Disclosures: RESULTS.md §2.2 consequence (iv) + gate-encoding sentence.

## Rejected alternatives

- **Narrow the porcelain exclusion (skip only `results/campaign/`) instead of
  hashing.** Porcelain shows *untracked* new files only as a directory entry
  and shows nothing content-level for files inside an untracked dir; the real
  campaign's results dirs are untracked at creation, so tampering inside them
  would be invisible to any porcelain-based rule. Content hashes see both
  tracked and untracked files identically.
- **Recompute metrics after each analyst call and diff.** Detects metrics
  tampering but not predictions/eval tampering (recomputing from tampered
  inputs reproduces the tampered output, self-consistently). The hash check
  dominates it at lower cost.
- **mtime/size snapshot instead of content hashes.** Cheaper, but a rewrite
  preserving length defeats size, and editors/tools can preserve or reset
  mtimes; SHA-256 over a few hundred MB costs seconds per analyst call (≤5
  calls per campaign) — negligible against multi-hour phases.
- **Trust the analyst prompt ("write EXACTLY one file").** Rejected by the
  design's own stated standard: prompts are not enforcement.
- **Hard-fail (raise) instead of graceful stop on tamper.** The graceful stop
  matches every other analyst-failure mode (timeout, no file, invalid JSON):
  state is saved, Phase A results are kept, the human inspects and resumes.

## Tripwire check (the one forbidden effect)

This amendment **adds** obligations; it does not flip any rubric item to pass
for the pre-existing artifact, and it does not reduce the evidence required
for the headline claim (which still awaits the GPU runs). Both added
obligations are discharged by same-iteration code + tests, so the artifact
satisfies the strengthened spec by doing the work, not by moving the line.
