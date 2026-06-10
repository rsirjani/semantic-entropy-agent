# Scrutiny record — iteration 11 (first-principles design review)

Charter: fresh-eyes principal-scientist pass over the campaign branch whose
iteration-10 deltas touched the launcher's data-plane guard and the metrics
artifact's self-description. The statistics stack has been independently
re-derived twice (iterations 9 and 10) with no defect, so this pass went where
scrutiny had been thinnest: the **evaluation layer** — the layer that turns
patches into the `resolved` booleans every estimator consumes — plus the figure
generator and the metric loaders' input-consistency assumptions. That choice
paid off: the design's chain of custody was guarded from spec to predictions,
but the step that converts predictions into *verdicts* trusted the SWE-bench
harness in two ways the harness does not deserve.

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

Unchanged from iterations 4–10; re-examined and still the right claim — a
diversity/mode-collapse claim with a coverage consequence, scoped to the easy
band, null explicitly publishable. The repo serves this claim, not a weaker
substitute.

**Ideal evidence:** everything the standing machinery already provides, PLUS
this iteration's closing clause: **the verdicts themselves must be genuine** —
an eval pipeline that can silently transmute an infrastructure flake into a
`resolved: false`, or serve a cached verdict for a patch that no longer exists,
poisons every estimator downstream no matter how carefully those estimators
were derived. Measurement integrity is upstream of statistical rigor.

**Minimal sufficient experiment set:** unchanged — (1) strategy arm T=0.7 τ=0
superset; (2) matched-k vanilla T=0.7; (3) exploratory T∈{0.2, 1.0}; (4) SDLG
arm T=0.7; τ ablation post-hoc at zero GPU. The campaign driver encodes exactly
this (dry-run re-verified this iteration).

---

## 2. Findings (with evidence pointers)

### TRUTHFULLY

- **T1 (major, fixed) — eval-time infrastructure errors were silently scored
  as test failures.** Verified in the installed harness source
  (`swebench/harness/run_evaluation.py:253-273`): `run_instance` catches
  `EvaluationError`, `BuildImageError`, **and bare `Exception`**, logs, writes
  no report, and the run continues — the error never reaches our driver's exit
  code. `scripts/eval_all_trajectories.py` then mapped "no `report.json`" to
  `resolved: false`, exit 0, and wrote the `trajectory_eval_<iid>.json` resume
  marker — freezing the fabricated verdict forever (the campaign's eval loop
  and `--resume` both skip on that marker). A Docker/build/container flake was
  therefore indistinguishable from a genuine test failure in the artifacts: it
  corrupts the Chen (n, c) in whichever arm it hits, can flip a per-instance
  any-pass (H2 gain), and can fabricate or destroy an off-mode-recovery
  candidate (R5.4). Iteration 10's record even asserted the opposite ("eval
  steps fail loudly in-step") — true only for a dead Docker daemon (which
  raises before the harness loop), false for every per-instance error the
  harness swallows. *Why this does not contradict the iteration-6/7
  symmetric-counting decision:* that decision is about **generation-time**
  failures, where the draw consumed budget and infra-vs-model is not
  machine-distinguishable from the artifacts; at the **evaluation** layer the
  budget is already spent, the patch exists with a definite ground-truth
  verdict, and the cause largely IS distinguishable from the harness's own
  logs. *Fix:* a missing report is now **classified**
  (`classify_missing_report`): `>>>>> Patch Apply Failed` → genuine failed
  draw (`fail_reason: patch_apply_failed`); timeout markers → genuine failed
  draw (`fail_reason: test_timeout`, SWE-bench convention for hanging
  patches); anything else → `EvalOutcomeError` → exit 3 **without writing the
  record** (campaign retries once, then stops loudly — the established
  failure philosophy). R4.1 amended
  (`spec_amendments/applied_11_eval_outcome_integrity.md`).
- **T2 (major, fixed) — stale-verdict reuse across re-runs.** Verified at
  `swebench/harness/run_evaluation.py:118-123`: if `report.json` exists for
  (run_id, model, instance), the harness **returns the existing report without
  evaluating**. Our run id was `{arm_slug}_traj_{tid}` — patch-content-blind —
  so re-evaluating after a treatment re-run (same trajectory id, different
  patch) silently inherited the *previous* patch's verdict. *Fix:*
  `patch_run_id` embeds a SHA-1 of the patch content: changed patch → fresh
  verdict; identical patch → legitimate cache reuse, which also makes the
  post-stop retry after T1's loud exit cheap (completed patches re-resolve
  instantly from cache). Also mirrored the harness's `/`→`__` model-name
  normalization in the report path — with a slash-bearing model id (R9.1
  allows one), every report lookup would have missed and scored the whole arm
  failed, the same silent-False failure mode as T1.

### MATHEMATICALLY (re-derived / re-checked, not trusted)

- **M1 (no defect) — the estimator stack is untouched and was not re-derived a
  third time** (two independent re-derivations stand: scrutiny_09 §M, _10 §M1).
  Spot-checks done where this iteration's changes touch the math: (a) the H1
  rarefied loop's `k* = min(k_a, k_b, len(pa), len(pb))` silently absorbed a
  predictions/eval count disagreement that H2's `k* = min(k_a, k_b)` would not
  — the two endpoints could have been computed at *different* k\* on a
  desynced instance with no trace in the artifact. Now named:
  `pred_eval_count_mismatch` lists every instance where the predictions file
  and the eval record disagree on draw count, in the comparison JSON and as a
  console warning (the loaders are designed to agree — eval is *built from*
  predictions — so any entry means desynced artifacts and the fix is re-running
  the eval, which T2's content-keyed cache makes cheap). (b) Verified `t3`-class
  rows (`fail_reason` present) flow through `load_eval` as ordinary `False`
  draws — the classification annotates, never changes, the metric contract.
- **M2 (checked, clean) — figure-layer consistency with the rarefaction
  requirement.** `make_figures.py` plotted each arm's **raw** distinct-patch
  count side by side — exactly the "raw distinct counts rise mechanically with
  sample size" comparison R4.2 forbids cross-arm (mostly harmless at run-time-
  matched k, but k mismatches from failed draws are expected and the H1
  endpoint lives at k\*). The figure set now includes
  `fig_rarefied_distinct_at_k_star.png` (per-arm rarefied levels + CIs at the
  common k\*, annotated with the H1 sign-flip p, its power floor, and the
  H2 gate status from `confirmatory_family` — the figure a reader grabs states
  the same inference rule as the JSON), and the raw distinct chart is labeled
  "(own k)".

### PHILOSOPHICALLY

- **P1 — framing re-audited, still coherent and non-circular.** H1's diversity
  metric remains mechanism-independent (structural signatures, no NLI);
  "diversity" is defined without reference to what produces it; the τ
  quantization is stated plainly (a partition-shape rule at N=5, grid = the
  7 achievable values); the τ=0 headline's "gate never gates" objection is
  answered by the post-hoc sweep being the gate's evaluation. Nothing new to
  confess on the framing.
- **P2 — the weakest joint a hostile reviewer would press** is now, as before,
  gate saturation (threat 11) and n=10 power — both pre-answered with
  pre-committed readings. After this iteration they would NOT find the eval
  layer: previously a referee asking "how do you know a `resolved: false` is a
  failed patch and not a failed evaluation?" had no answer in the artifacts;
  now the record carries `fail_reason` for every report-less failure and the
  pipeline refuses to write unverifiable verdicts.
- **P3 — the generalizing lesson, continuing the 5→10 series:** every
  iteration has moved one asserted invariant into something executable. The
  iteration-11 instance: **a pipeline that guards its own data plane can still
  be poisoned by trusting a third-party tool's silent-failure semantics.** The
  harness's contract ("report missing" ⇒ ???) was assumed benign in two
  distinct ways (error swallowing, cache-by-name); both are now either
  classified or keyed away. The spec → metrics → producers → artifacts →
  branch → commands → state machine → data plane chain now extends one link
  further: → **third-party verdict semantics**.

---

## 3. Steelmanned alternatives (this iteration's decisions)

| Design choice | Strongest alternative | Decision |
|---|---|---|
| Stop-loudly (exit 3, no record) on unclassifiable missing report | Record `resolved: false` + `eval_error: true` and have metrics skip flagged rows | **Stop.** A skipped row silently changes metric-time k (the bug class R4.1 exists to prevent), and a written record is frozen by the resume-marker skip; stop-and-fix matches every other step's failure philosophy, and the content-keyed cache makes the retry cheap. |
| Classify apply-fail and timeout as genuine failed draws | Treat ALL missing reports as infra (maximally strict) | **Classify.** An unappliable patch and a test-hanging patch are properties of the *patch* (SWE-bench scores both unresolved); strict-everything would let one genuinely bad patch block the campaign permanently. The two genuine cases are exactly the two `EvaluationError`s the harness raises *about the patch*; everything else it swallows is about the environment. |
| Content hash in the eval run id | Delete stale report dirs before re-eval | **Hash.** Deletion destroys the legitimate cache (identical patches would re-run for hours after every interruption) and relies on someone remembering; the content key is self-enforcing in both directions (stale → miss, identical → hit). |
| Read per-instance harness logs to classify | Read the harness's top-level summary JSON (`error_ids`/`resolved_ids`) | **Per-instance logs.** The summary is per-invocation, CWD-located, overwritten across invocations, and pools apply-fail with infra in `error_ids` — strictly less information than `run_instance.log`/`test_output.txt`, which are already keyed correctly. |
| Name pred/eval count disagreements (`pred_eval_count_mismatch`) | Hard-fail `compare()` on any disagreement | **Name, don't fail.** The metrics script is also the post-hoc explorer of partial/legacy artifacts; fabricating nothing while *reporting* the desync (console WARNING + JSON field) keeps it usable for diagnosis, and the campaign's metrics step output is read by the analyst, who is instructed to read the comparison block. The confirmatory cell runs the full chain in order, where desync cannot occur without an interruption that already stops the campaign. |
| Add the rarefied @k\* comparison figure | Drop the raw distinct figure entirely | **Add, keep both.** The raw per-arm count at own k is a legitimate descriptive (it is the quantity rarefaction corrects); deleting it hides the size of the correction. Labeling ("own k") + the new figure give the honest pair. |

Standing decisions re-examined and left in place: trajectory-matched budget
(conservative direction); τ=0 superset headline + command-pinned post-hoc
sweep; intent-summary clustering substrate; fixed-sequence H1→H2 family
(untouched — still no data); majority-signature selector with degeneracy
disclosure; 10-easy-SymPy scope; no-retune rule under gate saturation; the
R6.5 adaptive boundary and both-plane integrity guard; stop-don't-retry on
invalid analyst decisions; generation-layer symmetric failure counting
(expressly NOT extended to the eval layer — see T1 rationale).

Known limitations recorded, not fixed (disclosed): the genuine-fail
classification trusts the harness's log markers (`>>>>> Patch Apply Failed`,
timeout strings) — a harness version change could rename them; the markers are
inlined as constants at the top of the driver where a version bump audit will
find them. The infra-stop is per-instance, so a flaky-but-recovering Docker
daemon could stop the campaign mid-eval repeatedly; that is the intended
trade (loud beats fabricated), and resume + report cache make recovery cheap.
The porcelain code-plane caveat from iteration 10 (files already dirty at
campaign start) stands unchanged.

---

## 4. Actions taken this iteration

All verified: **130 pytest pass** (121 → 130; +9 new, 0 removed), `py_compile`
clean on every touched script, campaign dry-run prints the full correct plan.

1. `scripts/eval_all_trajectories.py` — `classify_missing_report` +
   `EvalOutcomeError` (T1); `patch_run_id` content-keyed run ids + model-name
   normalization (T2); `fail_reason` propagated to duplicate rows; `main`
   exits 3 without writing the record on infra errors; docstring contract
   extended (items 4–5).
2. `scripts/compute_metrics.py` — `pred_eval_count_mismatch` diagnostic in
   `compare()` + console warning (M1a).
3. `scripts/make_figures.py` — `make_comparison_figure` (rarefied distinct
   @k\* per arm, H1 p + power floor + H2 gate status on the figure); raw
   distinct chart labeled "(own k)" (M2).
4. `GOLD_STANDARD.md` R4.1 — eval-outcome integrity + stale-report immunity
   clauses (amendment record:
   `spec_amendments/applied_11_eval_outcome_integrity.md`; tripwire check
   inside — adds obligations discharged same-iteration, flips nothing).
5. `RESULTS.md` — §3 eval-record completeness items (iii)/(iv) + the
   mismatch diagnostic; §7 figure-regeneration paragraph (rarefied figure is
   the cross-arm one).
6. `tests/test_eval_driver.py` — +7: content-keyed run id; normalized report
   path read-back; apply-fail and timeout classified as genuine failed draws;
   infra error raises instead of scoring; classification priority; `main`
   exits 3 with no record written.
7. `tests/test_compute_metrics.py` — +1: mismatch named with counts, empty
   when consistent. `tests/test_budget_and_figures.py` — +1: rarefied
   comparison figure rendered (and not rendered without the comparison data).

## 5. What still requires a human / GPU

1. Ratify the iteration-11 amendment
   (`applied_11_eval_outcome_integrity.md`) and any earlier unratified ones —
   independent spec-critic review per ratchet v2.
2. **Launch the campaign from this branch from a CLEAN working tree**:
   `python scripts/run_campaign.py --go`. Phase A is the pre-registered
   confirmatory cell; T and τ are pinned in the launched commands; the analyst
   window is integrity-guarded on both planes; eval verdicts are now
   classified-or-refused.
3. After the runs: check threat 11 first (realized entropy distribution in
   `tau_sweep_*.json`); read H1 against `nonempty_patch_fraction`; read H2
   through `confirmatory_family`; check `pred_eval_count_mismatch` and
   `k_mismatch_instances` are empty (or explained); verify both-arm
   `budget_audit_*.json`; fill RESULTS §5 from script output only.
4. Git hygiene: decide whether this campaign branch becomes mainline.

## 6. Verdict logic

This iteration found and fixed two major measurement-integrity holes in the
evaluation layer — the one layer between the guarded run artifacts and the
twice-re-derived estimators that had never been audited against the harness's
actual source: (T1) every per-instance harness error was silently scored as a
test failure and frozen by the resume marker; (T2) the harness's
report-by-name cache let a re-run inherit a stale verdict for a different
patch. Plus two consistency gaps (figure layer showing the forbidden raw
cross-arm comparison; pred/eval count desync silently absorbed by `min()`).
Per the charter, finding real fixable issues means `gold_standard_met` =
**false**; the loop should pass once more. The next iteration arrives at a
branch whose verdict-producing layer is now classified-or-refuse: fresh eyes
should re-check the remaining unaudited third-party seams (the swebench
dataset loader, the vLLM/litellm response handling already covered by stage
tests, the NLI server protocol) and the doc chain end-to-end — if nothing
substantive surfaces, the design has nothing left to confess and the
remaining actions are human-only (ratification + GPU launch).
