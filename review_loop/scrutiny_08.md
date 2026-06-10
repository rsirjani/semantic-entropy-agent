# Scrutiny record — iteration 8 (first-principles design review)

Charter: principal-scientist review BEFORE the headline GPU runs, executed in
the **campaign worktree** — the lineage that will actually launch those runs.
That vantage point produced this iteration's central finding before any code
was read: the experiment infrastructure had **forked**, and the branch holding
the launch button was four scrutiny iterations behind the design.

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

(Unchanged from iterations 4–7 — re-examined, still the right claim.)

**Ideal evidence:** everything iterations 4–7 established (one confirmatory
cell mirroring the causal chain, exact small-n inference with printed power
floors, mechanism-independent diversity measurement, artifacts that cannot
redefine k between run and metric, draw accounting from the fork decision)
**plus this iteration's closing clause: the code that LAUNCHES the experiments
must be the code the scrutiny record describes.** A pre-registered design is a
property of a specific artifact lineage; if the runner forks from the design,
the pre-registration silently stops referring to anything.

**Minimal sufficient experiment set:** unchanged — (1) strategy arm T=0.7 τ=0
superset; (2) matched-k vanilla T=0.7; (3) exploratory sweep T∈{0.2, 1.0};
(4) SDLG arm T=0.7. τ ablation post-hoc, zero GPU. The campaign driver encodes
exactly this (Phase A = the confirmatory cell; the rest are analyst-ordered
exploratory cells).

**Did the repo serve this claim or a weaker one?** The *main branch* served it;
the *campaign branch* — the one with `run_campaign.py`, the one that exists to
launch the runs — did not. It diverged at iteration 3 and therefore lacked: the
H1→H2 hierarchical confirmatory family and power floor (iter 4), the eval-record
completeness rewrite (iter 5 — the campaign had only its own earlier draft,
missing the greedy-keyed pass@1, keep-last-per-tid, and loader-consistency
fixes), the predictions-producer completeness fix whose pro-treatment bias was
*confirmed on 5/10 real pilot instances* (iter 6), and the fork-decision draw
accounting incl. the Submitted-at-injection data-loss fix (iter 7). Both
scrutiny_06 §5 and scrutiny_07 §5 explicitly listed "rebase/regenerate the
campaign worktree before launch" as a required human action; it had not
happened. Had the campaign launched from this branch, every documented
guarantee about the headline numbers would have been false in the artifacts —
with the iteration-6 bias direction favoring the treatment.

---

## 2. Findings (with evidence pointers)

### TRUTHFULLY

- **T1 (blocker, fixed) — the launch lineage did not contain the design.**
  `git merge-base` showed the campaign branch rooted at d8e6b79 (iter 3) with
  only `run_campaign.py` + a draft eval fix on top; iterations 4–7 (69598ac..
  6f1651c) lived on a separate branch. *Fix:* merged `review-loop/20260610-013030`
  into this branch; the single conflict (`scripts/eval_all_trajectories.py`)
  resolved by taking the iteration-5+ rewrite, which is a strict superset of
  the campaign draft (verified: same `propagate_duplicate_results` API +
  null-tid normalization the draft had, plus the corrections it missed). All
  102 merged tests passed immediately; the campaign's own 14 tests pass against
  the merged drivers.
- **T2 (major, fixed) — the campaign's "budget_audit (both arms)" was false.**
  The docstring promised both arms; `build_steps` audited only the treatment —
  and could not have done otherwise, because `budget_audit.py` physically could
  not read the control layout (see M1). Docstring, steps, and tool all now
  agree (both arms, distinct output files, `test_build_steps_audits_both_arms`).
- **T3 (major, fixed) — the campaign analyst was pinned to the iteration-3
  design.** Its prompt cited `scrutiny_03.md` as "the design" and described the
  metrics output in pre-iteration-4 vocabulary (single primary endpoint, no
  H1/H2, no power floor, no productivity diagnostic, no saturation check). An
  analyst steering exploratory GPU spend while misreading the confirmatory
  structure would make systematically wrong calls (e.g. reading an H2 tie-floor
  null as "no effect" and stopping). *Fix:* prompt re-pointed at
  scrutiny_07/04 + RESULTS §2.2/§6; explicit reading rules for
  `min_achievable_p`, `nonempty_patch_fraction`, threat-11 saturation, both-arm
  audits (`test_analyst_prompt_reflects_current_design` pins every needle).
- **T4 (moderate, fixed + amendment) — adaptive data collection was
  undisclosed.** The campaign lets an LLM analyst choose which exploratory
  cells run, and when to stop, *after reading interim results* — adaptive
  collection that the pre-registration machinery never contemplated. For the
  confirmatory cell it is structurally harmless (Phase A runs first, exactly
  once, before any analyst is consulted — enforced in code and tested); but the
  repeat cell (`strategy_t0.7_seed2`) invited optional-stopping abuse ("the
  second draw looked better"), and the data-dependence of the exploratory cell
  set was nowhere disclosed. *Fix:* R6.5 adaptive-execution-boundary clause
  (first completed Phase A run IS the confirmatory dataset; repeats are
  variance-only; decisions are checked-in artifacts; data-dependence disclosed),
  RESULTS §2.2 disclosure paragraph, campaign docstring + analyst prompt pins.
  Amendment record: `applied_08_adaptive_campaign_boundary.md`.

### MATHEMATICALLY

- **M1 (major, fixed) — R6.3's cross-arm token comparison was incomputable on
  the control.** The fairness argument (RESULTS §2.2): at matched trajectory
  count the control pays ≥ the treatment's compute (k full SEARCHes vs one
  shared SEARCH), so a treatment win cannot be a compute artifact. The audit
  exists to *measure* that. But `budget_audit.py`'s discovery globbed
  `<dir>/*/metadata.json` and `<dir>/*/trajectory_*.traj.json` — the treatment
  layout — while the control nests each resample at
  `<dir>/<iid>/run<idx>/<iid>/...` (each resample is its own orchestrator run;
  its predictions/eval tid is `run<idx>`). On the control the audit returned
  n_instances=0: the inequality the design *leans on* was checkable only on the
  side where it cannot fail. *Fix:* layout auto-detection +
  `collect_draw_records` normalizing both layouts (control draws keyed
  `run<idx>` so steps/tokens join the eval record; crashed metadata-less runs
  counted as `n_draws_missing_metadata`, their empty-patch draws still in the
  eval record; per-instance totals summed across runs). Treatment path verified
  byte-identical on the pilot (53 trajectories / 12,882,383 tokens — matches
  RESULTS §5). Tests: synthetic control tree incl. crashed run, over-cap
  passing resample, token joins. Amendment record:
  `applied_08_both_arm_budget_audit.md`.
- **M2 (checked, no defect) — campaign↔driver wiring re-derived flag by flag.**
  Every flag `build_steps` constructs exists in the target script's argparse;
  the control-dir naming convention (`{base}_t{float}`) matches
  `run_temperature`'s f-string on both sides for all menu temperatures; the
  eval loop's fallback discovery (predictions file) is required and correct for
  the control layout (no top-level `<iid>/metadata.json`); `--skip-existing`
  resume semantics are safe on both runners (primary files written only on
  instance completion; batch-aware loaders + per-instance replacement handle
  re-runs). Now enforced continuously by
  `test_build_steps_flags_exist_in_target_scripts` — the wiring-drift class of
  defect (T1's mechanism) is a test failure rather than a mid-campaign crash.
- **M3 (checked, no defect) — iteration 7's queued fork-path questions.**
  (a) KeyboardInterrupt mid-fork: not an `Exception`, so it propagates past
  `_create_lazy_trajectory`/`_clone_for_sdlg` handlers to the run loop's
  KeyboardInterrupt handler; the interrupted trajectory stays `active`,
  excluded from draws, and the resample driver's metadata-mismatch warning
  flags the instance — symmetric with the control (an interrupted control
  instance writes no predictions rows at all). (b) Trajectory cap: applied
  before the draw decision, traced, correctly not a failed draw. (c)
  `_register_failed_draw` overwrite path: when `_inject_strategy_prompt` fails
  after registration, the placeholder replaces the active object under the same
  tid and reaps the container — one draw, one record. (d) Submitted-at-injection
  children: recorded completed with the submission patch; fork index advances on
  every attempt. No remaining unrecorded asymmetry found between the arms'
  draw accounting.
- **M4 (minor, fixed) — the standing vLLM container was an unverified
  determinism knob.** `ensure_servers` only checked HTTP liveness;
  `docker start vllm-server` resurrects whatever model that container was
  built for. A wrong model would fail loudly (404s) but only after burning the
  campaign's retry budget, and an *aliased* served name could in principle run
  the wrong weights silently. *Fix:* `expected_model_id` (config, litellm
  `openai/` prefix stripped) vs `/v1/models` ids; demonstrable mismatch raises
  before any run step; unknown sides never block (no false positives). Tested.

### PHILOSOPHICALLY

- **P1 — the generalizing lesson, one level above iteration 7's.** Iterations
  5–7 established: pre-registration is only as strong as artifact semantics,
  and artifact semantics are only as strong as their weakest producer. This
  iteration adds: **and all of it is only as strong as the lineage that
  launches the runs.** A design that lives on a branch the runner does not run
  is documentation, not design. The fix has the same shape as iterations 5–7's
  (move the invariant up the stack and pin it with a test): the merge is done,
  and the wiring-drift test makes silent divergence between the campaign and
  its drivers a red suite.
- **P2 — is the adaptive campaign compatible with the §0.1 framing and the
  no-result-chasing charter?** Yes, once bounded (T4): the analyst optimizes
  *information per GPU-hour over exploratory cells*, which is exactly what a
  human experimenter would do, and the charter's "do not chase a positive" is
  written into its decision principles. The confirmatory cell is immune by
  construction. The genuinely new risk was optional stopping via the repeat
  cell, which the first-run pin closes *before any data exist* — the cheapest
  possible time.
- **P3 — weakest joint a hostile reviewer would press now:** unchanged in kind
  — gate saturation (threat 11; the pilot τ-sweep still shows branch_rate 1.0
  below 1.6094), n=10 / one repo / easy band, and the τ=0 headline. All are
  disclosed with pre-committed reading rules; none is spinnable by design
  changes; none was spun. The new joint this iteration *added and answered* is
  "your scheduler reads the data" — answered structurally (boundary clause)
  rather than by promise.

---

## 3. Steelmanned alternatives (this iteration's decisions)

| Design choice | Strongest alternative | Decision |
|---|---|---|
| Merge iter-4–7 into the campaign branch | Rebase the 3 campaign commits onto the iter-7 head | **Merge.** Preserves both lineages' audit trails (the scrutiny record IS the artifact); the conflict surface was one file with a strict-superset resolution; a rebase would rewrite published history the verdicts reference. |
| Resolve eval conflict with the iter-5+ rewrite | Keep the campaign draft + cherry-pick fixes | **Take theirs wholesale.** Iteration 5 explicitly audited the draft and carried its API forward plus corrections (greedy-keyed pass@1, keep-last per tid, lazy import); the draft has no unique content. Verified by grep before resolving, and by the draft's own test passing against the merged file. |
| budget_audit supports both layouts at the consumer | Flatten the control's output layout at the producer | **Consumer.** The nested layout exists so concurrent resamples don't collide on disk; changing the GPU-run producer right before the campaign re-opens run-path risk for a read-side problem (same call as iteration 5's token-accounting decision). |
| Adaptive analyst bounded to exploratory cells | Forbid adaptivity (fixed cell order) | **Bound, don't forbid.** Exploratory cells are descriptive by design; spending fixed GPU budget on the most informative ones is pure efficiency once the confirmatory boundary is pinned. A fixed order burns budget regardless of what the primary shows. |
| First completed Phase A run = confirmatory dataset | Allow pre-registered pooling of repeat draws | **First run.** A pooling rule invented now would still be chosen by the same agent that later sees the data (the loop); fixed-n replication with a combining rule is legitimate future work, not a scheduler option. |
| Tree-integrity check after each analyst call | Trust the prompt ("make no other changes") | **Check.** The analyst runs with permissions skipped; prompts are not enforcement. A modified `compute_metrics.py` between phases would corrupt every later cell silently. Porcelain-diff allowing only `campaign_decisions/` + `results/` is cheap, tested, and loud. |
| Model-identity check: fail only on demonstrable mismatch | Hard-fail when /v1/models is unreadable | **Demonstrable only.** An unreadable model list at preflight would block a healthy campaign on a transient; a real mismatch still fails at the first run step. The check exists to convert a confusing late failure into a clear early one, not to add a new flake source. |

Standing decisions re-examined and left in place: trajectory-matched budget
(conservative direction — now *measurable on both arms*, closing the loop the
direction-claim opened); τ=0 superset headline + post-hoc sweep; intent-summary
clustering substrate; fixed-sequence H1→H2 family (untouched — still no data);
majority-signature selector with degeneracy disclosure; 10-easy-SymPy scope;
no-retune rule for the saturated gate (threat 11).

---

## 4. Actions taken this iteration

All verified: **111 pytest pass** (102 post-merge → 111; +9 new, 0 removed),
`py_compile` clean on every touched file, `compute_metrics.py`, `tau_sweep.py`,
and `budget_audit.py` re-run end-to-end on the real `results/branching` pilot
artifacts (numbers unchanged), campaign dry-run prints the correct full plan.

1. **Merged** `review-loop/20260610-013030` (iterations 4–7: producers, metrics,
   spec, scrutiny records, amendment records) into the campaign branch; conflict
   in `scripts/eval_all_trajectories.py` resolved by superset (d6f9db9).
2. `scripts/budget_audit.py` — `detect_layout` + `collect_draw_records`: both
   arms' layouts; control draws keyed by predictions tid `run<idx>`;
   `n_draws_missing_metadata`; treatment path provably unchanged.
3. `scripts/run_campaign.py` — `budget_audit_control` step; analyst prompt
   re-pointed at the current design with explicit reading rules; docstring
   pre-registration boundary; vLLM model-identity preflight
   (`expected_model_id` / `model_mismatch_error` / `served_model_ids`);
   analyst working-tree integrity check (`unexpected_tree_changes`).
4. Tests: +9 — control-layout audit (incl. crashed-run accounting), treatment
   layout detection, both-arm audit steps, wiring-drift guard over the whole
   menu, eval-loop flags, analyst-prompt needles, model-id prefix stripping,
   demonstrable-mismatch-only, tree-change flagging.
5. `RESULTS.md` — §2.2 adaptive-execution disclosure (three pinned
   consequences); §5 both-arm audit commands with the layout note.
6. `GOLD_STANDARD.md` — R6.3 both-arm layout clause; R6.5 adaptive-execution
   boundary. Records: `spec_amendments/applied_08_both_arm_budget_audit.md`,
   `spec_amendments/applied_08_adaptive_campaign_boundary.md` (tripwire checks
   inside each: nothing flips to pass; both obligations discharged by
   same-iteration work).

## 5. What still requires a human / GPU

1. Ratify the iteration-8 amendments (and any unratified 3–7 ones) —
   independent spec-critic review per ratchet v2.
2. **Launch the campaign from THIS branch** (it now carries iterations 4–7 +
   the campaign driver): `python scripts/run_campaign.py --go` (vLLM container
   + disk preflights are in the driver; NLI on CPU per the measured VRAM
   headroom). Phase A is the pre-registered confirmatory cell; analyst phases
   follow.
3. After the runs: check threat 11 first (realized entropy distribution in
   `tau_sweep_*.json`), read H1 against `nonempty_patch_fraction`, verify the
   both-arm `budget_audit_*.json` token totals confirm the conservative
   matching direction, fill RESULTS §5 from script output only.
4. Decide whether the merged campaign branch becomes the new mainline (the
   iter-4–7 branch and master are both behind it now) — a human git-hygiene
   call, not a design one.

## 6. Verdict logic

This iteration found and fixed a blocker-class infrastructure flaw — the
launch lineage lacked four iterations of design and bias fixes (T1) — plus a
structurally incomputable fairness audit on the control arm (M1/T2), a
stale-design analyst (T3), undisclosed adaptive data collection (T4), and two
smaller integrity holes (M4, tree check). Per the charter, `gold_standard_met`
= **false**; the loop should pass once more over the unified branch (fresh
eyes on the merge result as a whole — especially whether any main-branch test
or doc still references campaign-absent paths, and whether the analyst-driven
campaign needs a dry-run rehearsal mode before the real launch).
