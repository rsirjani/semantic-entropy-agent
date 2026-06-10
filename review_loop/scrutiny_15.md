# Scrutiny record — iteration 15 (first-principles design review)

Charter: fresh-eyes principal-scientist pass. Iteration 14 left a named
worklist (the clone/injection layer, `TrajectoryManager.save_all`,
`strategy_proposer` parse robustness, a config-vs-defaults sweep) and
predicted that if it surfaced nothing substantive the design would have
nothing left to confess. It surfaced plenty — including the single largest
protocol finding of the loop so far, which was not on any worklist: **the
confirmatory treatment had already been re-run (run-2) at a code revision two
guard-commits older than the one its control would have used, with open
container network**, and the just-committed anti-gaming veto carried a
measured 10.6%-of-actions false-positive rate that would have crippled both
future arms.

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

Re-derived from scratch; unchanged from iterations 4–14 and still the right
claim. The design serves it: same-scaffold matched-k control, mechanism-
independent diversity metric, fixed-sequence H1→H2 confirmatory family with
the power floor disclosed, τ=0 superset + zero-GPU post-hoc gate sweep,
pre-declared publishable null. **Ideal evidence requires one more property the
repo had only implicitly: the two arms of a cell must be produced by the same
experimental apparatus *over time* — same code revision, same guard envelope,
same network posture.** Run-2 violated exactly that, silently.

**Minimal sufficient experiment set:** unchanged — (1) strategy arm T=0.7 τ=0
superset; (2) matched-k vanilla T=0.7; (3) exploratory T∈{0.2, 1.0}; (4) SDLG
arm T=0.7; τ ablation post-hoc. Phase A must now be re-run (third attempt)
from one pinned revision; the campaign driver enforces the pin.

## 2. Findings (with evidence)

### TRUTHFULLY — protocol and guard integrity

- **T1 (blocker-class, resolved by archival + enforcement) — confirmatory-cell
  code-revision asymmetry.** Reconstructed timeline from git + campaign logs:
  iteration-14 guards committed 07:07–07:09; run-1 archived and campaign
  relaunched 07:13 (run-2); treatment completed 09:01; eval 3/10 instances
  done; campaign process stopped; **anti-gaming guard set committed 09:52**
  (`564b411`: `is_forbidden_command` veto wired at all three step sites AND
  `--network none` in `build_env_config`) — *after* the treatment ran,
  *before* the control ever ran. Consequences had the campaign resumed: the
  control would have faced a command veto and a closed network the treatment
  never did. Measured from the archived artifacts: 209/1,539 treatment actions
  (13.6%) would have been vetoed under the 09:52 guard; the treatment's 22
  `pip install mpmath` attempts had genuine internet access. **Decision:
  archive run-2** (`results/archive_run2_preguard/`, git-preserved) — made
  while NO treatment-vs-control comparison existed (the control never ran;
  the primary endpoints are paired gains, uncomputable without it), so the
  archival cannot be outcome-chasing; I did not read run-2's per-instance
  resolve outcomes. **Enforcement:** the campaign driver now pins HEAD at
  start (`code_revision` in state), refuses any step after a commit/checkout
  (`guardrails_ok`) or with tracked files modified, and refuses to *start*
  dirty. Spec: R2.3 amended ("scaffold-matched includes
  code-revision-matched", applied_15). Tests:
  `test_guardrails_enforce_code_revision_symmetry`,
  `test_fresh_state_pins_code_revision` (+ a hermetic-git autouse fixture so
  the campaign tests no longer depend on the live repo state).
- **T2 (major, fixed) — the 09:52 anti-gaming veto had a measured 10.6%
  false-positive rate.** Auditing run-2 under the veto (the validation the
  veto itself never got): 209 hits, of which **163 were heredoc patch-file
  CONTENT** (`cat > fix.patch <<'EOF'` … `diff --git a/…` — the body text
  matched `\bgit\b\s+`), **14 were the agent reverting its own edits**
  (`git restore <file>`, `git checkout <file>`), 6 were `git diff <path>`
  pathspec forms, 22 were genuine network attempts, and 4 were
  `git checkout HEAD -- <file>` / `--no-index` forms that reveal nothing
  (HEAD is pinned at the base commit because `git commit` is itself vetoed).
  A veto loop on patch-file writing would have burned steps and broken
  legitimate workflows in BOTH arms. *Fix:* `is_forbidden_command` now scans
  segments with heredoc bodies stripped and quoted spans blanked (the same
  `_command_segments` treatment the write detector got in iteration 14 —
  the lesson generalized one guard too late), and the git rules are
  semantics-aware: path-only subcommands allowed (`status`/`add`/`apply`/
  `stash`, `restore` without `--source`), ref-ambiguous subcommands
  (`diff`/`checkout`) allowed only in `--`-pathspec, literal-`HEAD`, or
  `--no-index` forms, with an instructive veto reason naming the allowed
  form; everything else (log/show/blame/reflog/…) forbidden. Re-audit:
  **33/1,528 (2.2%)** — 22 network (correct) + 11 ref-ambiguous nudges; zero
  genuine history access in run-2. Residual disclosed: quoted-program
  smuggling (`bash -c 'git log'`) is invisible after blanking — same class as
  the `python -c` write channel; the network half is backstopped by
  `--network none`. Tests: 3 new test functions, 12 total anti-gaming.
- **T3 (major, fixed) — `.git`-internals read bypass.** The git-CLI veto did
  not cover direct reads of `.git`'s history-bearing files: `packed-refs`/
  `refs/` name the post-fix commits, `logs/` is the reflog, `objects/` holds
  the gold blobs (zlib-inflatable via `python -c`). Now vetoed on the RAW
  command text (so quoted paths match), targeted to history-bearing subpaths
  only — `find … -not -path './.git/*'` exclusion idioms stay legal. Measured:
  0/1,041 executed pilot actions referenced `.git` at all — purely protective.
- **T4 (major, fixed) — `clone_container_state` (SDLG fork-state) silently
  degraded on every failure path.** A git-listing failure logged a warning and
  returned ("nothing to clone"); a per-file `docker cp` failure was skipped;
  and **deletions were structurally unpropagatable** (docker cp of a deleted
  file fails → the fork keeps the file). Fork-state consistency is
  load-bearing (iteration 14's T2 argument); a desynced fork's results would
  still be attributed to the SDLG mechanism. *Fix:* strict, deletion-aware
  contract — `git -c diff.renames=false diff --name-status -z` + untracked
  `-z` listing, deletions executed in the target, every subprocess checked,
  any failure raises `ContainerCloneError`, which `_clone_for_sdlg`'s
  existing handler converts to a failed-at-creation draw (R7.2 accounting —
  conservative against the treatment). Mitigating context, stated honestly:
  SDLG forks at the first *detected* write *before* it executes, so the
  parent tree is typically pristine; the strict path matters for the
  VERIFY-phase fallback and undetected `python -c` writes. 6 tests.
- **T5 (minor, fixed) — the strategy proposer was blind to elided
  observations.** `build_search_report` included an observation only if
  `<output>` appeared in it; observations >10k chars render as
  `<output_head>/<output_tail>` and were silently dropped — measured 3/120
  SEARCH-phase pilot observations, exactly the longest (often most code-rich)
  outputs. Treatment-mechanism-input fix (the control has no proposer),
  disclosed in §2.1; fixed before any confirmatory run.
- **T6 (minor, fixed) — spec/artifact contradiction in R1.2.** The rubric
  still *mandated* context-conditioned clustering ("problem statement
  prepended … at every call site") after run-1 was archived precisely because
  that conditioning saturates DeBERTa entailment (≥0.94 for all distinct-
  strategy pairs WITH the prefix vs ≤0.55 without). R1.2 now requires a
  *consistent context policy* and documents the measured empty-context
  deviation (applied_15, channel 1 — the old wording graded the defective
  instrument compliant and the corrected one deviant).
- **T7 (minor, fixed) — config/defaults drift.** `configs/branching.yaml`
  lacked `clustering_strategy` and `kernel_t`, contradicting
  `branching_defaults.py`'s "the shipped config sets all of them explicitly";
  both added, plus a completeness test
  (`test_branching_yaml_sets_every_default_explicitly`) so the contract can't
  silently rot. Stale `(250)` step-limit comment corrected;
  `vanilla_samples_at_temperature` now reads `diversity_method` via `cfg()`.

### Seams audited CLEAN (worklist discharged)

- **`branching_agent.py`:** `query_only` mirrors installed
  `DefaultAgent.query` line-for-line (limits → n_calls → query → cost →
  add_messages); `execute_response` = `execute_actions`;
  `inject_and_execute`'s `parse_regex_actions(content, *, action_regex,
  format_error_template)` call matches the installed minisweagent 2.2.7
  signature, and `LitellmTextbasedModel.config` carries both fields. The
  mocked e2e test fakes injection, so this signature check against the real
  package was the missing verification.
- **`TrajectoryManager.save_all`:** re-saves all trajectories (idempotent,
  placeholder draws skipped with their record in metadata.json), called after
  every trajectory and in `finally`. `Trajectory.save` → `agent.save(path,
  *extra_dicts)` matches the installed signature.
- **`strategy_proposer._parse_strategies`:** regex handles "STRATEGY N:",
  parenthesized variants, case-insensitivity; numbered-list fallback; final
  whole-response fallback yields one strategy → single trajectory, flagged by
  realized-N reporting (never a silent grid mix — R3.3 guard). Catastrophic
  `propose()` failure returns the generic strategy → 1 trajectory, same
  honest degradation.
- **Legacy-only code confirmed out of the run path:** `SWEBenchContainer`
  (no `--network none`) → only `react_agent.py`/`run_baseline.py` (excluded
  by R2.3); `TrajectoryManager.branch/_create_branch` → only
  `branching_orchestrator.py` (legacy).

### MATHEMATICALLY

- **M1 — sixth estimator verification, this time by MY OWN brute force (not
  trusting prior iterations):** `pass_at_k` ≡ exhaustive k-subset enumeration
  for all (n≤8, c, k); `expected_distinct_at_k` ≡ exhaustive subset
  enumeration over signature pools with duplicates AND empty patches;
  `paired_permutation_pvalue` ≡ full 2^10 sign enumeration on lumpy vectors;
  floor 2^(1+z−n) respected and attained at exactly 6 same-sign nonzeros
  (p = 0.03125). My first harness run "found" mismatches — caused by my own
  API misuse (passing bare strings where unified diffs are expected; passing
  (n,z) where the gains vector is expected) — a useful reminder that the
  metric functions are typed for real artifacts; the corrected harness is
  exact to 1e-12.
- **M2 — direction analysis of every iteration-15 change:** veto fixes are
  symmetric (both arms share the phase machinery) and measured against run-2;
  clone strictness can only convert silent desyncs into failed draws *against*
  the treatment; the search-report fix restores the documented mechanism
  (treatment-only input, disclosed pre-data, touches no metric/matching/
  endpoint); revision pinning constrains the *operator*, not the data. No
  change touches an estimator, a matching rule, or an endpoint.
- **M3 — re-audit numbers:** veto FP 209/1,539 → 33/1,528 (2.2%), all
  remaining hits either genuine network (22) or instructive ref-ambiguity
  nudges (11); `.git` references 0/1,041 executed actions; elided
  observations 3/120 SEARCH-phase.

### PHILOSOPHICALLY

- **P1 — framing intact.** Nothing this iteration touched §0.1, the H1→H2
  gate, matched-k definitions, or τ machinery. The diversity metric remains
  mechanism-independent.
- **P2 — the weakest joint has moved.** After 15 iterations the residual
  hostile-reviewer pressure points are operational, not conceptual: (a) gate
  saturation at the new entailment instrument (threat 11 — will the fixed
  clusterer produce non-degenerate partitions? answerable only by the run,
  pre-committed reading either way); (b) n=10 power (threat 4, floor
  disclosed); (c) the third Phase A attempt must actually run start-to-finish
  under the pinned revision. The protocol now has the property that the only
  way to violate arm symmetry is to defeat an enforced guard, not to forget
  a convention.
- **P3 — the generalizing lesson, continuing the series:** iteration 13 —
  a disclosed limitation is not a guarded one; iteration 14 — an assumed
  invariant is not a guarded one; **iteration 15 — a guard is itself an
  instrument and needs the same measured validation as the thing it guards**
  (the anti-gaming veto shipped with tests but without a pilot-log audit, and
  carried a 10.6% FP rate), **and the experimental apparatus includes its own
  revision history** (two arms at different commits are different scaffolds
  no matter what the diff "should" do).

## 3. Steelmanned alternatives (this iteration's decisions)

| Decision | Strongest alternative | Why rejected |
|---|---|---|
| Archive run-2 | Salvage with disclosure (run the control under the new guards, footnote the asymmetry) | The control cannot honestly run WITHOUT the guards (reintroducing a measured leak channel to match a flawed treatment), and 13.6% vetoed-action envelope difference + open-network treatment is not footnote-sized. No paired outcome existed when decided — same legitimate-exception class as run-1. |
| Pin HEAD for the whole campaign | Pin per-cell only | Within-cell symmetry carries the inference in EVERY cell; whole-campaign pinning costs nothing (don't edit code mid-campaign) and keeps provenance one SHA. |
| Heredoc/quote-blanked veto scanning | Keep raw-text scanning, add patch-content special cases | Special-casing `diff --git` invites the next FP class; the write detector already proved the blanking approach on 2,184 actions; one shared `_command_segments` = one behavior to test. Residual quoted-program smuggling disclosed (same class as `python -c`). |
| Git-semantics subcommand rules (path-only vs ref-ambiguous) | Flags-only-for-everything (the 09:52 rule) | Measured: vetoes the agent reverting its own edits (14×) and pathspec diffs (6×) — burns steps and teaches the agent nothing. The semantic rules are derived from what a positional CAN mean to git, each form tested. |
| Allow literal `HEAD` positional | Veto all refs uniformly | HEAD is provably pinned at base (`git commit` vetoed), `git diff HEAD`/`checkout HEAD -- f` are natural idioms (4 run-2 uses) that reveal nothing; uniform veto is friction without protection. |
| Strict clone with deletion propagation | Clone the full container filesystem (docker commit/run) | Full-image clone per fork adds minutes + disk per fork for a state that is typically pristine; the strict file-level clone is exact for the cases that occur and *fails loudly* into the existing failed-draw accounting otherwise. |
| Fix the search-report tag match | Leave it (3/120 is small) | The dropped observations are precisely the longest/most code-rich ones feeding the proposer — a mechanism-fidelity bug of the iteration-14 T1 class; the fix is one condition + test, disclosed. |

Standing decisions re-examined and left in place: trajectory-matched budget
(conservative direction re-verified); τ=0 superset + post-hoc sweep;
intent-summary clustering substrate; H1→H2 fixed sequence (no data yet);
majority-signature selector with degeneracy disclosure; 10-easy-SymPy scope;
no-retune rule; classify-or-refuse eval verdicts; arm purity at the fallback
layer.

## 4. Actions taken (all verified)

**166 pytest pass** (151 → 166: +15, 0 removed), `py_compile` clean on every
touched file, campaign dry-run prints the identical pinned Phase A plan.

1. `results/strategy_t0.7` + `results/campaign` → `results/archive_run2_preguard/` (T1).
2. `scripts/run_campaign.py` — `_git_head`/`_tracked_modifications`,
   revision pin in state, `guardrails_ok` revision+dirty checks,
   dirty-start refusal (T1).
3. `src/agent/phases.py` — `is_forbidden_command` v2: heredoc/quote-blanked
   segment scanning, git-semantics rules with instructive reasons,
   `.git`-internals raw-text veto (T2, T3).
4. `src/utils/docker_helpers.py` — `clone_container_state` strict +
   deletion-aware, `ContainerCloneError` (T4).
5. `src/diversity/strategy_proposer.py` — `<output_head>` inclusion (T5).
6. `configs/branching.yaml` — `clustering_strategy`, `kernel_t`, comment
   fixes; `scripts/run_branching.py` — `cfg()` consistency (T7).
7. `GOLD_STANDARD.md` — R1.2 context-policy correction, R2.3
   revision-symmetry strengthening (T6, T1; applied_15 record).
8. `RESULTS.md` — run-2 protocol-deviation disclosure (threat 11), veto
   correction + run-2 audit numbers (threat 13), clone/search-report
   mechanism notes (§2.1), SDLG-call budget-undercount mention (§5).
9. Tests: `tests/test_clone_and_config.py` (8), `tests/test_anti_gaming.py`
   (+3 incl. measured FP classes), `tests/test_run_campaign.py` (+2 +
   hermetic-git fixture).

## 5. What still requires a human / GPU

1. Ratify applied_12, applied_14, applied_15 spec amendments.
2. Commit this iteration, then launch Phase A run-3 from the clean tree:
   `python scripts/run_campaign.py --go` (the driver now refuses dirty/moved
   revisions; do NOT commit anything mid-campaign).
3. Pre-launch smoke: `scripts/start_vllm.sh` + `python scripts/smoke_test.py`.
4. Post-run reading order unchanged (threat-11 saturation check first, then
   completeness diagnostics, then H1/H2 through `confirmatory_family`).

## 6. Verdict logic

This iteration found one blocker-class protocol defect (run-2 revision/
network asymmetry — archived, now structurally prevented), two major guard
defects (veto FP rate, unguarded `.git` internals), one major mechanism-
integrity defect (silent clone degradation), and three minor honesty/
hygiene defects — all fixed, measured against the archived pilots, and
stage-tested. Per the charter, finding real issues ⇒ `gold_standard_met:
false`; the loop must pass again. Iteration 16's fresh-eyes corners: the
eval driver's interaction with archived dirs (stale `logs/run_evaluation`
reports under content-hashed run ids are safe by design — verify),
`relevance.py`/`intent.py` (the last unaudited diversity modules),
`make_figures.py` against the current metrics schema, and a re-audit that
run-3's launch preconditions (clean tree, pinned SHA, servers) are all
mechanically checkable. If that pass finds nothing substantive, the
remaining actions are human-only.
