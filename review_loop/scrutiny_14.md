# Scrutiny record — iteration 14 (first-principles design review)

Charter: fresh-eyes principal-scientist pass over the campaign branch.
Iteration 13 closed the documented-but-unimplemented-guard class and left a
named worklist: `src/agent/phases.py` transition machinery,
`scripts/nli_server.py` device/dtype handling, and a final
CLI-surface-vs-documented-commands check on `run_branching.py` /
`run_resample_baseline.py` / `eval_all_trajectories.py`. This iteration
executed that worklist with measurement (every claim below about the pilot is
counted from the 2,184 logged actions / 79 trajectory transcripts, not
eyeballed) and found **three genuine mechanism-integrity defects** in the
phase machinery — all in the run path, all fixed and stage-tested — plus two
smaller honesty/hygiene defects. One spec amendment applied (R8.1
strengthened to name the new guard classes).

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

Unchanged from iterations 4–13; re-examined from scratch and still the right
claim, and the repo serves it (matched-k same-scaffold control, mechanism-
independent diversity metric, fixed-sequence confirmatory family, τ=0
superset + post-hoc gate sweep, pre-declared publishable null).

**Ideal evidence, extended by this iteration:** the paper's mechanism story
("each branch independently implements its assigned strategy"; "SDLG forks at
the first implementation commitment"; "all forks start from the same
post-search state") must be true of the *run mechanics*, not just the
prompts. This pass found all three of those sentences were violable:

1. the assigned-strategy prompt could silently fall out of context on long
   trajectories (observed on 3 of the pilot's 79 transcripts);
2. the "first write command" branch-point detector misread quoted comparison
   operators as writes (2 of 2,184 pilot actions) and missed writes chained
   before `&& pytest`;
3. the read-only SEARCH phase — whose read-only-ness the fork design *assumes*
   (forks replay messages into fresh containers; they do not clone the
   searched filesystem) — was enforced only by a prefix allowlist that
   `echo … > file` and even the submission command sailed through.

**Minimal sufficient experiment set:** unchanged — (1) strategy arm T=0.7 τ=0
superset; (2) matched-k vanilla T=0.7; (3) exploratory T∈{0.2, 1.0}; (4) SDLG
arm T=0.7; τ ablation post-hoc at zero GPU. Campaign dry-run re-verified
after this iteration's edits (full pinned command plan unchanged).

---

## 2. Findings (with evidence pointers)

### TRUTHFULLY / mechanism integrity

- **T1 (major, fixed) — context truncation silently dropped the assigned-
  strategy prompt.** `_truncate_context` kept the first 4 + last 40 messages
  once a trajectory exceeded 80; the `PATCH_PROMPT_WITH_STRATEGY` message sits
  after the replayed search messages — squarely in the dropped middle.
  Measured on the pilot transcripts: 3 trajectories
  (sympy-12481 `t0_strategy_1`, sympy-15345 `t0_strategy_3`, sympy-23534
  `t0`) ran past the threshold and their final message lists no longer
  contain "YOUR ASSIGNED STRATEGY". Consequence: exactly the long/hard
  trajectories — where mode-collapse pressure matters most — lost the one
  instruction that differentiates them and could drift back to the model's
  mode, *diluting the treatment mechanism the paper describes*. The VERIFY
  prompt (the submit protocol) was droppable the same way. *Fix:*
  `_truncate_context` pins every `"## Current Phase:"` user message found in
  the dropped middle (≤2 messages of cost); idempotent under repeated
  truncation. Test:
  `test_truncate_context_pins_strategy_and_phase_prompts` (incl. re-truncation
  pass). Direction note (M2): this restores the *documented* mechanism — it
  does not touch metrics, matching, or endpoints, and the identical machinery
  (with its generic "fix the bug" patch prompt) runs in the vanilla arm.
- **T2 (major, fixed) — the SEARCH read-only boundary was unenforced against
  redirect writes and submission.** `is_command_allowed` prefix-matched the
  last `&&` segment: `echo fix > /testbed/f.py` passed (allowed prefix
  `echo`), `cat <<EOF > repro.py` passed (`cat`), and the submission command
  passed in ANY phase (prefix `echo`, or `cat patch.txt` as last segment) —
  the function's own "submission is only allowed in VERIFY" special case was
  dead code below the prefix loop. Why it is load-bearing:
  `_create_lazy_trajectory` builds each strategy fork as a *fresh container*
  replaying the pruned search messages ("search phase doesn't modify files"
  — its own comment), so a SEARCH-phase write would exist in t0's container
  but in none of the forks' — silently desynchronizing the branches' starting
  states (and an early submission would end the instance with 1 draw).
  Measured: 0 of 189 pilot SEARCH actions were writes, so enforcement changes
  nothing on the observed distribution — it converts an *assumed* invariant
  into a guarded one. *Fix:* submission decided first (VERIFY-only), then a
  SEARCH-phase `is_write_command` veto, then the prefix allowlist; the
  orchestrator's existing blocked-command handler (error message +
  low-relevance scoring, so blocked loops still saturate) covers the new
  vetoes unchanged. Residual, documented: `python -c "open('f','w')…"` is
  undetectable from the command string. Test:
  `test_search_phase_blocks_writes_and_submission`.
- **T3 (major, fixed) — the write detector (the SDLG branch-point trigger)
  had measured false positives and a structural false-negative class.**
  Old behavior: split on `&&`, inspect only the LAST segment, regex the rest.
  Measured against all 2,184 pilot actions: 2 false positives —
  `awk 'NR>=350 {print …}' file | tail` (the `>=` inside single quotes reads
  as a redirect) — and a false-negative class: a real write in a non-final
  segment (`sed -i … && pytest`; `git diff > patch.txt && cat patch.txt`).
  Why it matters: SDLG fires **once**, at the first detected write; a false
  positive before the first real write moves that instance's branch point to
  a *view* step, corrupting the mechanism attribution for the whole instance
  (R1.1's "the branch point is a genuine write step"). The SDLG arm has not
  run yet — this was the last cheap moment. *Fix:* `is_write_command` now
  strips heredoc bodies, blanks quoted spans, splits on `&&`/`;`/newlines,
  and checks every top-level segment (prefixes + redirect regex). Validation:
  re-ran both detectors over all 2,184 pilot actions — exactly 9 verdicts
  change: the 2 false positives (now False) and 7 chained patch-prep
  redirects (now True, consistent with their already-True unchained forms —
  the documented scratch-file limitation applied uniformly). Tests:
  `test_is_write_command_quoted_comparison_not_a_write`,
  `…_chained_write_in_nonfinal_segment`, `…_heredoc_body_not_inspected`.
- **T4 (minor, fixed) — dead/misleading transition machinery and untrue
  enforcement docs.** `detect_phase_transition` carried a SEARCH branch keyed
  on "STRATEGY:" phrases that no caller can reach (`_step_search` never calls
  it; SEARCH→PATCH is decided solely by `should_end_search` saturation/cap) —
  inviting the false belief that declaring a strategy ends the search phase
  (the SEARCH_PROMPT even tells the agent it does; operatively the
  declaration only stops useful searching, relevance drops, and saturation
  fires). The `phases.py` module docstring also implied three enforced
  read-only/write boundaries when only SEARCH is enforced
  (`_step_patch`/`_step_verify` never consult the allowlists). *Fix:* dead
  branch removed (same class as iteration 13's `has_strategy`); module
  docstring rewritten with the honest enforcement contract (SEARCH enforced;
  PATCH/VERIFY are prompt-level guidance; VERIFY cannot be read-only because
  the submit protocol requires `git diff > patch.txt`; the STRATEGY: sentence
  is a nudge, not a wired trigger).
- **T5 (minor, fixed) — `nli.py` degenerate-input early return broke its own
  schema.** `compute_sdlg_scores` on `< 2` SEP tokens returned only
  `{tokens, attributions}`; `nli_server.py`'s `/sdlg_scores` endpoint indexes
  `scores["gradients"]`/`["embeddings"]` directly → KeyError → HTTP 500.
  Dead in the live path (the SDLG generator uses `/sdlg_rank`, which guards
  on empty tokens; the local fallback at `sdlg.py:314` runs only without a
  server and also guards), but a latent contract violation. *Fix:* the early
  return now carries the full schema.

### Seams audited CLEAN (the iteration-13 worklist, discharged)

- **`nli_server.py` device/dtype:** device is CLI/env-driven end-to-end
  (`--device` → `NLI_DEVICE` → `NLIModel(device=…)`; the campaign driver
  passes `--nli-device` explicitly), fp32 throughout, `torch.no_grad()` on
  all classify paths, and the gradient path matches Aichberger Alg. 2:
  self-entailment input (text as premise AND hypothesis), loss toward
  contradiction, `A_i = ‖z_i ⊙ ∇z_i L‖₂` on the hypothesis span only,
  word-initial substitution positions, self-substitution excluded
  (`s_ij[token_ids[i]] = -1`), trivial case-variants filtered, multiple
  replacements per position kept with exact-pair dedup, top-50 cap. The
  `attr_max > 0` normalization guard is safe (attributions are norms, ≥ 0).
  `/sdlg_rank` guards both empty-tokens and empty-word-starts.
- **CLI surface vs documented commands (final pass):** every flag in RESULTS
  §5 and in the campaign dry-run plan exists with the documented semantics —
  `run_branching.py` (`--config/--results-dir/--clustering-strategy/
  --temperature/--entropy-threshold/--diversity-method/--skip-existing`),
  `run_resample_baseline.py` (`--treatment-dir/--results-dir/--temperatures/
  --max-k/--skip-existing`; per-temperature dir suffix `_t<T>` matches the
  eval commands' `resample…_t0.7`), `eval_all_trajectories.py`
  (`--results-dir/--predictions/--instance/--timeout/--include-duplicates`).
  Dry-run re-verified post-edit; T and τ remain explicit on the treatment
  command (R2.4).
- **`eval_all_trajectories.py` re-read:** batch splitting on primary rows,
  keep-last within batch, dedup-with-propagation, empty-patch failed draws,
  classify-or-refuse missing-report handling, content-hashed `patch_run_id`,
  arm-scoped run ids, results into the arm's own dir — all consistent with
  the §3 contracts.
- **`run_resample_baseline.py` re-read:** per-instance row replacement,
  metadata k discovery with old-driver mismatch warning, full-agent
  temperature override for the none arm only — clean.
- **Estimator stack, fifth spot-check (this time brute force, not
  re-derivation):** `pass_at_k` ≡ exhaustive hypergeometric enumeration for
  all (n ≤ 8, c, k); `expected_distinct_at_k` ≡ exhaustive subset enumeration
  on a pool with duplicates AND empty patches (empties dilute n, contribute
  no signature; k = n recovers the raw distinct count);
  `paired_permutation_pvalue` ≡ exhaustive 2^10 sign enumeration on a lumpy
  0/±1 gain vector; `min_achievable_sign_flip_p` = 2^(1+z−n) confirmed and
  *attained* by an all-same-direction-nonzeros vector. All exact to 1e-12.

### MATHEMATICALLY

- **M1 — direction analysis of the detector change.** Both arms run the same
  phase machinery, so T2/T3 guards are symmetric by construction. On the
  pilot distribution the SEARCH veto is a no-op (0/189) and the detector
  changes touch 9/2,184 actions, none in SEARCH. The false-positive fix
  protects the *SDLG arm's* branch-point attribution (mechanism fidelity);
  no metric, matching rule, or endpoint consumes `is_write_command` output.
- **M2 — direction analysis of the truncation fix.** Pinning the strategy
  prompt can only *increase* realized treatment diversity on long
  trajectories — superficially treatment-favorable. It is nevertheless
  correct, for three reasons: (i) the paper's mechanism description ("each
  fork independently implements its assigned strategy") was false of the
  artifact for 3/79 pilot trajectories — the old behavior was a bug diluting
  the *documented* design, not a conservative choice; (ii) the comparison's
  fairness rests on matched trajectory counts, temperature, and scaffold —
  none touched; (iii) the same pinning applies to the vanilla arm's own
  phase prompts (generic patch prompt, VERIFY submit protocol), whose loss
  hurt it identically (a vanilla trajectory that loses the submit protocol
  produces fewer submitted patches). Disclosed in RESULTS §2.1.
- **M3 — estimator brute-force results** (see clean-seams list): the
  confirmatory pipeline's three estimators are exactly the combinatorial
  quantities the writeup claims, including the empty-patch and tie edge
  cases the prior derivational spot-checks reasoned about.

### PHILOSOPHICALLY

- **P1 — framing re-audited, still coherent and non-circular.** Nothing this
  iteration touched the §0.1 five-mechanism family, the entropy blind spot,
  the H1→H2 gate, or the matched-k definition. H1's diversity metric remains
  mechanism-independent (patch signatures, not the branching NLI).
- **P2 — the weakest joint** remains gate saturation (threat 11) and n=10
  power (threat 4), both pre-answered with pre-committed readings. After this
  pass, a hostile reviewer probing "does each branch actually carry its
  strategy to completion?" has an artifact-level answer (pinned prompts +
  test) instead of a prompt-level hope.
- **P3 — the generalizing lesson, continuing the 5→13 series:** iteration 13
  taught that a *disclosed* limitation is not yet a guarded one; iteration
  14's instance is one layer deeper — **an *assumed* invariant is not yet a
  guarded one.** The fork design's correctness rested on "SEARCH doesn't
  modify files" (a code comment), the mechanism's potency rested on "the
  strategy prompt stays in context" (nobody had claimed otherwise), and the
  branch-point detector's precision rested on "commands look like their last
  segment". None had a guard or a measurement until now. Audit rule going
  forward: every sentence of the mechanism story in §2 of RESULTS must name
  the code path that makes it true and the test that pins it.

---

## 3. Steelmanned alternatives (this iteration's decisions)

| Design choice | Strongest alternative | Decision |
|---|---|---|
| Pin phase prompts across truncation | Re-inject the strategy prompt after every truncation (always-latest position) | **Pin.** Re-injection changes the message *order* the model sees every time truncation fires (the instruction would leapfrog recent context repeatedly), is harder to make idempotent, and complicates the transcripts; pinning preserves one stable copy at a stable position for ≤2 messages of budget. |
| Enforce SEARCH read-only with `is_write_command` | Clone the root container's filesystem into each fork (make writes harmless instead of forbidden) | **Enforce.** Filesystem cloning per fork was the original design the lazy-template approach deliberately replaced (one live container at a time fits 32 GB VRAM; clone adds per-fork latency and a new failure surface at every fork creation). Forbidding writes in a phase that is documented read-only is the cheaper invariant — and measured at 0 occurrences, it forbids almost nothing. |
| Quote/heredoc-stripping segment parser | Real shell tokenizer (bashlex) | **Strip.** A grammar-true parser adds a dependency and still cannot catch the actual residual class (programmatic writes); the pure-string version is fully stage-testable and was validated against every action the pilot ever produced. |
| Apply the scratch-file limitation uniformly (chained `git diff > patch.txt` = write) | Special-case `> patch.txt` as a non-write | **Uniform.** Filename special cases invite false negatives (`> patch2.txt`?); the limitation is documented with its mitigation, and the only consumer that cares (SDLG trigger) already faced it in the unchained form. |
| Document PATCH/VERIFY allowlists as unenforced | Enforce them | **Document.** VERIFY *requires* a redirect for the submit protocol, so a write-block there breaks every submission; a PATCH allowlist block adds risk mid-campaign for no scientific gain (a PATCH-phase command of any kind is the treatment's own work, symmetric across arms). Honesty fix, not behavior fix. |
| Strengthen R8.1 in the spec now | Leave the spec, rely on tests | **Strengthen** (applied_14): the iteration-13 lesson cuts both ways — guard behavior the project relies on must be a named spec requirement so regression is a violation, not drift. Tripwire-clean: R8.1 was pass and the strengthened version is met by this same iteration's work. |

Standing decisions re-examined and left in place: trajectory-matched budget
(conservative, audited); τ=0 superset headline + post-hoc gate sweep;
intent-summary clustering substrate; fixed-sequence H1→H2 family (untouched —
still no data); majority-signature selector with degeneracy disclosure;
10-easy-SymPy scope with the Appendix-C deviation disclosed; no-retune rule
under gate saturation; R6.5 adaptive boundary + both-plane integrity guard;
classify-or-refuse eval verdicts; arm purity at the fallback layer; R3.3
quantization-grid exclusion at both consumers.

Known limitations recorded, not fixed (disclosed): `python -c` programmatic
writes are undetectable from the command string (documented in
`is_write_command` and the phases module docstring); harness log-marker
constants remain inlined and coupled to the pinned swebench version; the
porcelain caveat (campaign must start from a clean tree) stands; the SDLG arm
has no realized-N guard (no partition logged — stated in the artifacts);
`mean_pairwise_distance`'s subset-mean identity is inexact with empty patches
(descriptive only).

---

## 4. Actions taken this iteration

All verified: **144 pytest pass** (139 → 144; +5 new, 0 removed),
`py_compile` clean on every touched file, campaign dry-run prints the full
pinned plan unchanged, and the new detector was validated against all 2,184
pilot actions (9 verdict changes, every one inspected and correct).

1. `src/agent/phases.py` — `is_write_command` v2 (heredoc-body strip, quoted-
   span blanking, all-top-level-segment inspection; T3); `is_command_allowed`
   reordered (submission first → VERIFY-only; SEARCH write veto; T2); dead
   SEARCH branch of `detect_phase_transition` removed and live contract
   documented; module docstring states the enforcement reality (T4).
2. `src/agent/phased_orchestrator.py` — `_truncate_context` pins
   `"## Current Phase:"` user messages across truncation (T1).
3. `src/diversity/nli.py` — degenerate-input early return carries the full
   schema (T5).
4. `tests/test_pipeline_correctness.py` — +5 tests: quoted-comparison FPs
   (pilot-measured), chained-write FNs, heredoc-body exclusion, SEARCH
   write/submission blocking (+ VERIFY submit-prep stays allowed), truncation
   pinning incl. idempotence.
5. `RESULTS.md` §2.1 — "Mechanism integrity at the run-mechanics layer"
   paragraph disclosing both guards with the pilot measurements.
6. `GOLD_STANDARD.md` R8.1 — strengthened (segment-aware detection classes,
   enforced SEARCH boundary, submission-first, unenforced-boundaries-
   documented); amendment record
   `review_loop/spec_amendments/applied_14_write_detector_and_phase_enforcement.md`.
7. `review_loop/scrutiny_14.md` — this record.

## 5. What still requires a human / GPU

1. Ratify the iteration-12 and iteration-14 spec amendments
   (`applied_12_arm_purity_no_silent_fallback.md`,
   `applied_14_write_detector_and_phase_enforcement.md`) — independent
   spec-critic review per ratchet v2.
2. **Pre-launch:** start vLLM (`scripts/start_vllm.sh`) and run
   `python scripts/smoke_test.py` (verifies the SDLG importance contract on
   the live port 8001).
3. **Launch the campaign from a CLEAN working tree:**
   `python scripts/run_campaign.py --go`.
4. After the runs: threat-11 check first (realized entropy distribution +
   `non_modal_n_instances`); confirm `pred_eval_count_mismatch` /
   `k_mismatch_instances` / `strata_grid_excluded` empty or explained; read
   H1 against `nonempty_patch_fraction`; read H2 through
   `confirmatory_family`; verify both-arm budget audits; fill RESULTS §5
   from script output only.
5. Git hygiene: decide whether this campaign branch becomes mainline.

## 6. Verdict logic

This iteration discharged the iteration-13 worklist and found that the
"remaining corners" were not corners: three mechanism-integrity defects sat
in the run path (strategy-prompt loss under truncation — 3 pilot
trajectories; an unenforced load-bearing SEARCH invariant; measured branch-
point detector errors), any of which would have quietly weakened or
mis-attributed the treatment mechanism in the not-yet-run confirmatory and
SDLG cells. All are fixed, stage-tested, measured against the pilot logs, and
disclosed; none touches a metric, a matching rule, or an endpoint. Per the
charter, finding real fixable issues means `gold_standard_met` = **false**;
the loop should pass once more. Iteration 15's fresh-eyes corners: the
clone/injection layer (`src/agent/branching_agent.py::inject_and_execute`,
`src/utils/docker_helpers.py::clone_container_state`,
`trajectory.py::TrajectoryManager.save_all`), `strategy_proposer.py` parse
robustness against malformed proposals, and a last
`configs/branching.yaml`-vs-`branching_defaults.py` defaults sweep. If that
pass surfaces nothing substantive, the design has nothing left to confess
and the remaining actions are human-only (ratification + smoke test + GPU
launch).
