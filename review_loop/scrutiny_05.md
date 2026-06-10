# Scrutiny record — iteration 5 (first-principles design review)

Charter: principal-scientist review BEFORE the headline GPU runs. Iteration 4
fixed the confirmatory family and the τ-sweep numerics and explicitly left two
audits for this pass: `eval_all_trajectories.py` / `collect_results.py` (never
re-read since the metric layer was built), the H1 difflib-signature granularity,
and the 6e-4 recompute tolerance. This pass did those audits. The headline
finding is that the **evaluation driver — the producer of every eval artifact
the metric layer consumes — would have corrupted the headline comparison in
three independent ways**. All fixed and tested this iteration.

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

**Ideal evidence:** one confirmatory cell mirroring the causal chain
(diversity → coverage), exact small-n inference with its power floor printed,
mechanism-independent diversity measurement, every other cell exploratory,
budget accounting showing the control was not starved — **and an artifact
pipeline that cannot silently change what "k", "n", or "c" mean between the run
and the metric.** That last clause is what iteration 5 had to earn.

**Minimal sufficient experiment set:** unchanged from iterations 3–4 —
(1) strategy arm T=0.7 τ=0 superset; (2) matched-k vanilla T=0.7; (3)
exploratory sweep T∈{0.2,1.0}; (4) SDLG arm T=0.7. τ ablation post-hoc, zero
GPU.

**Did the repo serve this claim or a weaker one?** The *design documents*
served it; the *eval wiring* did not. RESULTS §5 documented a comparison
(`--eval results/strategy_t0.7 --compare-eval results/resample_t0.7`) that the
eval driver was physically incapable of producing (it hardcoded
`results/branching` as its output dir), and the eval record it did produce
counted unique non-empty patches, not produced trajectories — redefining k at
exactly the layer the matched-k machinery trusts. The most carefully
pre-registered endpoint is worthless if the artifact feeding it measures a
different denominator.

---

## 2. Findings (with evidence pointers)

### TRUTHFULLY

- **T1 (blocker, fixed) — the documented workflow was not executable: the eval
  driver wrote every arm's eval into `results/branching/`.** Old
  `eval_all_trajectories.py:181-183` hardcoded the output path; there is no
  other producer of `trajectory_eval_<iid>.json`. Evaluating the vanilla
  control would have *overwritten the treatment's eval files* (R2.5), and the
  §5 commands referencing `results/resample_t0.7` eval files could never have
  been satisfied. *Fix:* `--results-dir` end-to-end (predictions default, eval
  output, temp files, arm-scoped eval run_ids). The campaign worktree
  (`.claude/worktrees/campaign`) had independently drafted this fix; the main
  branch now carries it (same `propagate_duplicate_results` API its tests
  import), plus further corrections the draft missed (T3, M2 below).
- **T2 (minor, fixed) — the eval summary's "pass@1 (greedy)" was the best-of
  row.** `results[0]` after batch-splitting is the *primary* (best-of-k)
  duplicate, not the greedy trajectory; the saved `pass_at_1` field lied in
  the artifact. Now keyed to `t0`/`run0` explicitly.
- **T3 (moderate, fixed) — metric loaders disagreed about what a null
  trajectory id means.** `load_eval` kept null-id rows (`!= "primary"`),
  `load_eval_by_tid` dropped them — the coverage table's n could differ by 1
  from every tid-joined analysis (selector, set-valued, τ-sweep, budget audit)
  *reading the same file*. Now both drop null ids, and the eval driver
  normalizes the row to `"primary"` at write time (tested both layers).
- **T4 (disclosure, added) — budget accounting silently excluded the
  treatment's proposer/intent/NLI calls** (they are not in `.traj.json`
  transcripts). Bounded (one proposer call + 10 NLI pair passes of a 0.4B
  model per instance vs k full trajectories) and direction-disclosed in
  `budget_audit.py::fairness_note` + RESULTS §5 + R6.3.

### MATHEMATICALLY

- **M1 (blocker, fixed) — default eval dedup redefined the Chen estimator's
  sample space and deleted the phenomenon under study.** The old driver
  dropped duplicate AND empty patches from the eval record, so metric-time
  k = #unique non-empty patches. Derivation of the damage: vanilla produces 5
  trajectories, 4 copies of failing F + 1 passing P → true pass@1 = 0.2, but
  over the deduped record pass@1 = 0.5; worse, at metric time
  k\* = min(k_T, k_V) inherits vanilla's deflated k, so the treatment's
  coverage is *subsampled via pass@k\*(n_T, c_T)* while vanilla keeps plain
  any-pass — and the H1 rarefaction is compressed to a k\* with little room
  for any diversity difference. The mode-collapse thesis predicts vanilla
  duplicates; the eval step erased them. *Fix:* evaluate unique patches once
  (compute), then `propagate_duplicate_results` writes one row per genuine
  trajectory — duplicates inherit the representative's outcome (deterministic
  harness ⇒ exact, marked `deduped_from`), empty patches are `resolved: false`
  draws. Eval-file n now equals produced trajectories (tested:
  `test_propagate_gives_every_trajectory_a_row`).
- **M2 (major, fixed) — re-run staleness: every post-hoc parser read the FIRST
  strategy-proposal block; every other artifact reflects the LAST run.**
  `phased_decisions.log` is append-mode (`phased_orchestrator.py:1431`),
  predictions dedupe keep-last, `metadata.json` is overwritten — but
  `load_entropy` and `tau_sweep.parse_instance` searched from the top. A
  re-run instance would get run-1's entropy/partition joined to run-2's
  trajectories: wrong strata, wrong gate reconstruction, wrong dominant
  trajectory. Both now read the last block/match (tested:
  `test_sweep_parses_last_block_after_rerun`,
  `test_load_entropy_uses_last_block_after_rerun`). Same rule for the resample
  predictions file, which has no primary-row batch separators: the eval driver
  now keeps the last occurrence per tid (tested).
- **M3 (moderate, instrumented) — H1 conflates diversity with patch-production
  rate.** `expected_distinct_at_k` counts non-empty signatures with empties
  remaining in n — correct for "distinct solutions per k-budget," but an arm
  that merely *finishes more often* gains rarefied distinct count without
  exploring anything. At T=0.7 vanilla can fail to produce patches; H1 could
  reject on productivity alone and be over-read as mode-collapse escape.
  *Fix (diagnostics, endpoint unchanged):* `nonempty_patch_fraction` per arm +
  `rarefied_distinct_gain_nonempty` (descriptive, k\*_ne = min non-empty
  count), fixed pre-data, with the reading rule in RESULTS §3
  (tested: `test_compare_nonempty_robustness_separates_productivity_from_diversity`).
- **M4 (checked, affirmed with direction stated) — exact-signature granularity
  is conservative for H1.** Lexical variants count as distinct in both arms;
  the whole-agent-sampling control produces trivial variants at least as
  readily as greedy post-branch execution, so the inflation favors the
  *control's* distinct count — biasing H1 toward the null. Defensible for a
  confirmatory endpoint (a win survives a conservative metric); now stated in
  print (RESULTS §3/threat 7) instead of implied.
- **M5 (checked, no defect) — the 6e-4 recompute tolerance interacts safely
  with 6-decimal logs** (|exact−logged| ≤ 5e-7 ≪ 6e-4 going forward; legacy
  3-decimal logs ≤ 5e-4 < 6e-4; kernel values disagree ≫ 6e-4 and stay
  flagged). A coincidental kernel value within 6e-4 of a partition entropy
  would be relabeled, shifting it < 6e-4 — immaterial except exactly at a grid
  boundary of a grid that is wrong for kernel runs anyway (explicit `--taus`
  documented for kernel). Accepted.
- **M6 (checked, no defect) — re-derivations standing:** Chen at matched k\*
  is the finite-population hypergeometric identity (no i.i.d. needed for the
  "random k\*-subset of what the arm produced" question); rarefaction is the
  same identity per signature; the pair-inclusion symmetry claim for mean
  pairwise distance holds (U-statistic linearity), with the <2-non-empty
  convention noted; sign-flip floor 2^(1+z−n) re-verified; fixed-sequence
  gatekeeping controls FWER at 0.05 without splitting.

### PHILOSOPHICALLY

- **P1 — the deepest lesson of this pass: pre-registration is only as strong
  as artifact semantics.** Iterations 3–4 hardened the *statistics* (exact
  tests, power floors, gatekeeping). None of it would have mattered: the
  artifact feeding those tests redefined k below the metric layer. The
  pipeline's correctness now has the same standing in the spec as the
  estimator's (R4.1 eval-record completeness, R7.2 isolation), each clause
  carried by a stage test rather than a promise.
- **P2 — is the §0.1 framing still coherent and non-circular after the
  changes?** Yes: diversity remains defined by structural patch distance,
  independent of the NLI gate; the new diagnostics make the *interpretation*
  of H1 more honest without touching its definition. The falsifiable
  predictions remain falsifiable: H1 at n=10 has floor 2^-9; H2's tie floor is
  printed; the productivity row adds a way for a *treatment win to be
  downgraded*, which is the opposite of bending toward a win.
- **P3 — weakest joint a hostile reviewer would press now:** "your distinct
  count is lexical, your n is 10, one repo, easy band." The first is answered
  with the conservativeness direction + graded companion + pre-registered
  robustness row; the rest are disclosed scope limits (threats 1/2/4) that
  only more GPU/replication can move — correctly listed as roadmap, not
  spinnable by design changes.

---

## 3. Steelmanned alternatives (this iteration's decisions)

| Design choice | Strongest alternative | Decision |
|---|---|---|
| Propagate duplicate outcomes in eval | Evaluate every trajectory independently | **Propagate.** The harness is deterministic given a patch; propagation is exact, marked in the artifact, and avoids multiplying Docker cost by the duplication factor exactly where duplication is predicted highest (vanilla). `--include-duplicates` retained for spot-validation. |
| Fix the eval record (producer) | Fix k at metric time from predictions, treat missing rows as failures | **Fix the producer.** Metric-side patching leaves the artifact lying, breaks tid joins (τ-sweep dominant lookup, budget pass joins), and mislabels passing duplicates as failures. Artifacts must be truthful at the source. |
| H1 endpoint unchanged + productivity diagnostics | Redefine H1 over non-empty patches only, or as mean pairwise distance | **Keep endpoint, add diagnostics.** Conditioning the primary test on patch production is post-hoc subsetting inside the endpoint; swapping endpoints a second iteration running is forking paths. The non-empty row is descriptive and can only *downgrade* a treatment win. Full argument in `applied_05_h1_productivity_diagnostics.md`. |
| Exact signature for distinct | Fuzzy/threshold clustering of patches | **Exact, with direction disclosed.** A tunable similarity threshold reintroduces a researcher degree of freedom on the title metric; exact + graded companion + stated conservative direction is more auditable. |
| Last-block parsing for append-mode logs | Truncate/rotate the log per run | **Parse last.** Rotating logs would orphan the existing pilot artifacts and add a run-side failure mode; the parser-side rule is testable and matches the keep-last semantics every other loader already has. |
| Token accounting from `.traj.json` only | Instrument proposer/NLI token counting | **Disclose the exclusion.** Run-loop instrumentation touches the GPU-run code path right before the campaign (risk) for a correction bounded well under 1% of arm totals; the disclosure names the direction (against the fairness margin, dominated by trajectory sums). Roadmap if reviewers demand exact totals. |

Standing decisions re-examined and left in place: trajectory-matched budget
(conservative direction re-affirmed, now with the exclusion disclosure);
τ=0 superset headline; intent-summary clustering substrate (Wei et al. 2026's
NLI-weak-on-raw-code finding still supports it); 10-easy-SymPy scope
(disclosed); fixed-sequence H1→H2 family (untouched — no data yet, no reason).

---

## 4. Actions taken this iteration

All verified: **77 pytest pass** (69 → 77; +8 new, 0 removed), `py_compile`
clean on all touched scripts, `compute_metrics.py` and `tau_sweep.py` re-run
end-to-end on the real `results/branching` artifacts, driver `--help` smokes
pass.

1. `scripts/eval_all_trajectories.py` — rewritten (see T1/T2/M1/M2): arm-
   isolated output, one eval row per genuine trajectory, primary normalization,
   keep-last per tid, greedy-keyed pass@1, lazy swebench import (stage-testable).
2. `scripts/compute_metrics.py` — `load_eval` drops null tids; `load_entropy`
   reads the last block; `compare()` emits `nonempty_patch_fraction` +
   `rarefied_distinct_gain_nonempty` (+ console output).
3. `scripts/tau_sweep.py` — parses the LAST strategy-proposal block.
4. `scripts/budget_audit.py` — fairness note discloses the proposer/NLI
   exclusion and its direction.
5. Tests: `tests/test_eval_driver.py` (4 new — propagation, dedup, batch
   split, resample re-run); `test_compute_metrics.py` +3 (null-tid drop,
   last-block entropy, productivity robustness); `test_tau_sweep.py` +1
   (last-block parse).
6. `RESULTS.md` — §3 eval-record contract + H1 granularity/productivity
   disclosures; §5 explicit per-arm eval commands + token-exclusion note;
   §6 threat 7 extended.
7. `GOLD_STANDARD.md` — R4.1 eval-record completeness + loader consistency +
   last-block rule; R4.2 productivity diagnostics + granularity disclosure;
   R7.2 arm-isolated eval outputs; R6.3 disclosed accounting exclusions.
   Records: `spec_amendments/applied_05_eval_record_completeness.md`,
   `spec_amendments/applied_05_h1_productivity_diagnostics.md`.

Note: `results/branching`'s existing `trajectory_eval_*.json` were produced by
the old driver (deduped record) — usable for smoke tests only; the
pre-registered runs regenerate all eval records with the fixed driver.
`collect_results.py` was audited and left alone: it serves the legacy baseline
summary only and feeds no headline metric.

## 5. What still requires a human / GPU

Unchanged in shape: ratify the two iteration-5 amendments; run the
pre-registered cell + exploratory cells; **evaluate with the FIXED driver**
(`eval_all_trajectories.py --results-dir <arm_dir> --instance <iid>` per
instance per arm); then `compute_metrics.py` / `tau_sweep.py` /
`budget_audit.py`. The campaign worktree's `run_campaign.py` already invokes
the driver with `--results-dir` and stays compatible.

## 6. Verdict logic

This iteration found and fixed substantive evidence-pipeline flaws — an
unproducible documented workflow with cross-arm overwrite (T1), an eval record
that redefined the Chen estimator's denominator and deleted the vanilla arm's
mode-collapse signal (M1), re-run staleness in every post-hoc parser (M2), and
an uninstrumented productivity confound on the confirmatory endpoint (M3). Per
the charter, `gold_standard_met` = **false**; the loop should make one more
pass over the post-fix design (fresh eyes on the rewritten eval driver and the
new diagnostics, plus the standing question of whether anything else consumes
the legacy deduped eval files).
