# Applied amendment 05 — Eval-record completeness & arm-isolated eval outputs

**Channel:** value-level design amendment (scrutiny charter, ratchet v2), with a
strong corrective component — most of this is *derivable* estimator correctness.

**Spec edits:** R4.1 (eval-record completeness clause + loader-consistency +
last-block parsing rule), R7.2 (arm-isolated eval outputs), R6.3 (disclosed
token-accounting exclusions).

## The defect this fixes (found iteration 5; flagged for audit by verdict_04)

`scripts/eval_all_trajectories.py` — the *producer* of every
`trajectory_eval_<iid>.json` the metric layer consumes — had three
evidence-corrupting behaviors, none visible to the metric scripts downstream:

1. **Hardcoded output dir.** It always wrote into `results/branching/`, so
   evaluating the vanilla control would have *overwritten the treatment's eval
   files* (R2.5 violation). The documented §5 workflow
   (`--compare-eval results/resample_t0.7`) was not actually producible.
2. **Default dedup deleted the signal under study.** `deduplicate_patches`
   silently dropped duplicate-patch and empty-patch trajectories from the eval
   record, so metric-time k = #unique non-empty patches, not #trajectories
   produced. The vanilla arm's duplicates ARE the mode-collapse phenomenon; the
   thesis predicts the control collapses to few forms. Quantitatively: vanilla
   produces 5 trajectories — 4 copies of failing patch F, 1 passing P. True
   trajectory-level pass@1 = 0.2; over the deduped record, pass@1 = 0.5. At
   metric-time matched k\*, the vanilla arm's deflated k (e.g. 2) drags
   k\* = min(k_T, k_V) down, *subsampling the treatment's coverage via the Chen
   estimator while the vanilla arm keeps plain any-pass* — and compressing the
   H1 rarefaction comparison into a regime with almost no room for a diversity
   difference. Direction: anti-treatment for H1/H2 point estimates, but more
   importantly *wrong* — the comparison would no longer answer the matched-
   trajectory-budget question either way.
3. **The best-of "primary" row leaked as a null id.** Dedup kept the first
   bearer of each patch — usually the null-id primary row — and dropped the
   genuine trajectory carrying the same patch. `load_eval` counted null-id rows
   (`!= "primary"`) while `load_eval_by_tid` dropped them, so the coverage
   table's n disagreed with every tid-joined analysis (selector, set-valued,
   τ-sweep, budget audit) on the same file.

Related re-run staleness defect fixed under the same rule: `phased_decisions.log`
is append-mode, while predictions loaders keep-last and `metadata.json` is
overwritten; `load_entropy` and `tau_sweep.parse_instance` read the FIRST
`STRATEGY PROPOSAL` block — a re-run instance would have its *old* run's
entropy/partition joined to its *new* run's trajectories. Both parsers now read
the LAST block (tested).

## The fix (implemented this iteration, all tested)

- `eval_all_trajectories.py` rewritten: `--results-dir` (arm-isolated output,
  arm-scoped eval run_ids), `propagate_duplicate_results` (one row per genuine
  trajectory; duplicates inherit the representative's outcome marked
  `deduped_from`; empty patches `resolved: false` without a container run),
  primary-row normalization to `"primary"`, keep-last per (iid, tid) within the
  latest batch (resample re-run appends), greedy `pass@1` keyed to t0/run0
  rather than the best-of row. Heavy swebench import deferred so the helpers
  are stage-testable (R8.5).
- `compute_metrics.load_eval` drops null trajectory ids (loader consistency).
- Both entropy/partition parsers read the last block.
- RESULTS §3 (contract stated), §5 (eval commands now explicit per arm).

Identical patches resolve identically under the deterministic SWE-bench harness,
so outcome propagation is exact, not an approximation; it is also marked in the
artifact (`deduped_from`) so a reader can re-derive the unique-patch view.

## Why this is not self-serving

No rubric item flips to pass *for the artifact as it existed*: the GPU runs have
not happened, the new spec text adds obligations (completeness, isolation,
disclosure) that were implemented and tested in the same iteration, and the
amendment makes the headline comparison *harder to corrupt*, not easier to win.
The existing pilot eval files in `results/branching` were produced by the old
driver and are now documented as superseded for headline use (the pre-registered
runs regenerate them with the fixed driver).

## Rejected alternatives

- **Evaluate every trajectory independently (no dedup at all).** Scientifically
  equivalent record, but multiplies Docker eval cost by the duplication factor —
  on the vanilla arm exactly where duplication is predicted to be highest. The
  propagation design keeps the compute saving and the metric correctness;
  `--include-duplicates` remains for spot-validation of the determinism premise.
- **Fix k at metric time instead (read k from predictions, treat missing eval
  rows as failures).** Repairs (n, c) but leaves the eval artifact lying about
  what was evaluated, breaks tid joins (τ-sweep dominant-trajectory lookup,
  budget audit pass joins), and silently mislabels a *passing duplicate* as a
  failure. The artifact itself must be truthful.
- **Keep the shared eval dir and disambiguate by filename suffix.** Fragile
  (glob-based loaders), and still allows two arms' files to interleave in one
  dir; per-arm dirs match the R2.5 isolation rule every other artifact follows.
