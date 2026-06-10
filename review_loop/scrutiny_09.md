# Scrutiny record — iteration 9 (first-principles design review)

Charter: principal-scientist review of the **unified campaign branch** (the
lineage iteration 8 merged and that will launch the GPU runs), before any
headline run executes. Iteration 8 queued three fresh-eyes questions for this
pass: (a) does anything still reference campaign-absent paths, (b) does the
analyst-driven campaign need a rehearsal mode before the real launch, and
(c) is the merged RESULTS.md internally consistent end-to-end. All three were
worked; the review also re-derived the statistics stack from scratch rather
than trusting prior verdicts.

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

(Unchanged from iterations 4–8; re-examined and still the right claim — a
diversity/mode-collapse claim with a coverage consequence, not a leaderboard
claim, with the null explicitly publishable.)

**Ideal evidence:** the iteration 4–8 machinery (one confirmatory cell
mirroring the causal chain; exact small-n inference with printed power floors;
mechanism-independent diversity metric; draw accounting from the fork decision;
both-arm budget audit; a launch lineage that contains the design) **plus this
iteration's closing clause: every parameter that DEFINES the pre-registered
cell must be visible in the command that launches it, and the launcher's own
state machine must be execution-tested, not just unit-tested.**

**Minimal sufficient experiment set:** unchanged — (1) strategy arm T=0.7 τ=0
superset; (2) matched-k vanilla T=0.7; (3) exploratory T∈{0.2, 1.0}; (4) SDLG
arm T=0.7; τ ablation post-hoc at zero GPU. The campaign driver encodes exactly
this (Phase A = the confirmatory cell, analyst-ordered exploratory cells after,
bounded by the R6.5 adaptive-execution clause).

**Does the repo serve this claim or a weaker one?** This claim. No framing
drift found this pass; the remaining gaps were operational (below), not
conceptual.

---

## 2. Findings (with evidence pointers)

### TRUTHFULLY

- **T1 (major, fixed) — the pre-registered τ=0 was pinned nowhere in the
  launch commands.** The confirmatory cell is defined by (arm, clustering,
  T, τ). R2.4 required T to be explicit on both arms — and the campaign obeys
  (`--temperature 0.7` / `--temperatures 0.7`) — but τ rode silently on
  `configs/branching.yaml` `entropy_threshold: 0.0` in BOTH the campaign's
  treatment commands (`run_campaign.py::build_steps`) and the documented
  reproduction command (RESULTS.md §5). The entire post-hoc τ ablation (R3.3)
  and the "superset run" framing rest on that value; a silent config change
  between cells would corrupt the sweep with no loud failure. *Fix:* explicit
  `--entropy-threshold 0` on every treatment run in `build_steps` and in the
  RESULTS §5 command; R2.4 amended to require τ explicitness (parallel to its
  temperature clause); pinned by
  `test_build_steps_pins_tau_superset_explicitly` over the whole menu.
  Amendment record: `spec_amendments/applied_09_explicit_tau_pin.md`.
- **T2 (minor, fixed) — stale doc pointer.** The campaign docstring said the
  adaptive-execution disclosure lives in RESULTS.md §6; it lives in §2.2.
  Fixed. (The iteration-8 queued sweep for campaign-absent paths found nothing
  else: the §5 command chain, control-dir naming (`resample_<key>_t<T>`), and
  port wiring (8001 in both `configs/branching.yaml` and `VLLM_URL`) are
  consistent end-to-end.)

### MATHEMATICALLY (re-derived, not trusted)

- **M1 (no defect) — estimator stack re-derivation.** (a) `pass_at_k`'s
  product form: C(n−c,k)/C(n,k) = Π_{i=n−c+1}^{n}(1−k/i) — algebra checks;
  edge cases (c=0, n−c<k) correct. (b) Rarefaction
  `expected_distinct_at_k` = Σ_sig P(signature drawn) via the same
  hypergeometric identity — correct, empty patches correctly stay in n while
  contributing no signature. (c) The exact sign-flip test enumerates all 2^n
  sign patterns with statistic |mean|, includes the identity pattern (valid
  p ≥ 2^−n·count), two-sided by construction; symmetry of per-instance gains
  under the strong null follows from arm exchangeability — the test is valid
  for both H1 (near-continuous gains) and H2 (lumpy 0/1 gains). (d) The tie
  floor `min_achievable_sign_flip_p` = 2^(1+z−n): the two global sign choices
  on the m=n−z nonzero entries always attain the observed |mean|, each
  duplicated 2^z times — bound confirmed. (e) `compare()` enforces
  k\* = min(k_a, k_b) at metric time for H2 and
  min(k_a, k_b, |preds_a|, |preds_b|) for H1; the R5.2 stratification and the
  R5.4 low-entropy flag share one threshold so they cannot disagree. (f) The
  mean-pairwise-distance docstring honestly scopes its no-correction claim to
  all-non-empty pools. (g) `tau_sweep`'s achievable grid enumerates integer
  partitions exactly, recomputes entropies at full precision from the logged
  partition (with the 3-decimal-rounding boundary case handled and
  kernel-arm disagreement flagged, never overwritten), reports realized N
  with non-modal flags. No defect found anywhere in this stack.
- **M2 (no defect) — budget-match direction re-checked.** Matched trajectory
  count gives the control ≥ the treatment's compute (k full SEARCHes vs one
  shared SEARCH), so a treatment win cannot be a compute artifact; the
  treatment's failed creation-time draws also count into k, pushing the same
  conservative direction. The both-arm `budget_audit` measures rather than
  assumes this. Token-matching would favor the treatment and is correctly not
  used; the disclosure (RESULTS §2.2) says exactly this.

### OPERATIONALLY (the launcher itself — this iteration's focus)

- **O1 (moderate, fixed) — stale analyst decision file on resume.**
  `run_analyst` numbers decision files per analyst phase starting at 1; on
  `--resume` after an interruption, a leftover `decision_01.json` could be
  READ AS THE FRESH ANALYST'S OUTPUT if the new subprocess failed to write —
  executing a spec nobody just chose (or mis-stopping), and breaking the R6.5
  "every scheduling decision is a checked-in artifact" audit chain (one file,
  two meanings). *Fix:* archive any pre-existing file to `*.superseded`
  before invoking the analyst; only a freshly written file is ever validated.
  Test: `test_stale_decision_file_never_read_as_fresh`.
- **O2 (moderate, fixed) — vLLM/NLI required before steps that don't use
  them.** `run_spec` called `ensure_servers` before EVERY step, including
  eval (Docker only) and metrics/audit/sweep (pure post-processing). After a
  ~30 h run phase, a vLLM container that fails to restart would abort the
  campaign before computing any metrics from artifacts already on disk — and
  on resume, before every remaining post-processing step. *Fix:*
  `needs_servers` is set only on the two agent-run steps; `run_spec` checks
  it. Tests: `test_servers_required_only_for_agent_run_steps` + the e2e test
  counts exactly 2 preflights per spec.
- **O3 (verification gap, closed) — the campaign state machine had never
  executed end-to-end.** Unit tests covered every component, but `main()`'s
  assembled loop (Phase A ordering, analyst gating, stop semantics, state
  persistence, resume, STOP file) had only ever printed a dry-run. This is
  the "rehearsal mode" question iteration 8 queued. *Decision:* a mocked
  end-to-end pytest of `main()` itself (stubbed `run_step`/`ensure_servers`/
  `run_analyst`, tmp-dir state), NOT a `--rehearse` CLI flag — the pytest
  executes the identical orchestration code on every suite run, while a
  rehearse flag would add new production branches right before launch for
  marginal extra realism. Tests: `test_campaign_loop_end_to_end_mocked`
  (Phase A first and complete, analyst consulted only after, chosen spec runs
  fully, stop ends the loop, state records completion order),
  `test_campaign_resume_skips_completed_phase_a` (the R6.5 first-run pin
  survives resume), `test_campaign_stop_file_aborts_before_any_step`.

### PHILOSOPHICALLY

- **P1 — framing re-audited, still coherent and non-circular.** "Diversity"
  is measured by structural patch signatures (rarefied at matched k\*), fully
  independent of the NLI machinery that decides branching; H1 is therefore a
  genuine test of the mode-collapse premise, not an echo of the mechanism.
  The §0.1 five-mechanism family, the entropy blind spot, and the
  branch-don't-abstain defense (Tomov et al.) remain the strongest honest
  version of the story. The falsifiable predictions are testable at n=10 for
  H1 (near-continuous gains, floor 2^(1−m) reachable); H2's tie floor is
  disclosed with `min_achievable_p` printed beside every p.
- **P2 — weakest joints a hostile reviewer would press, unchanged in kind:**
  gate saturation (threat 11 — pilot branch_rate 1.0 below ln 5; if the real
  runs reproduce it, the gate is reported uninformative at this substrate, a
  pre-committed negative reading), n=10 / one repo / easy band (threats 1/2/4,
  scoped claims), and the τ=0 headline ("the title's gate never gates") —
  answered by the post-hoc sweep being the gate's evaluation, plus the plain
  quantization statement that τ at N=5 is a partition-shape rule. None of
  these is spinnable by design edits; none was spun.
- **P3 — the generalizing lesson of 5→9:** iteration 5: artifact semantics;
  6: producer completeness; 7: draw accounting at the fork; 8: the launch
  lineage; 9: **the launch *commands* and the launcher's *control flow***.
  Each layer closed by moving the invariant into something executable (a test
  or a pinned flag) rather than a promise. After this pass, every layer
  between the design document and the GPU — spec → metrics → producers →
  artifacts → branch → commands → state machine — has an executable guard.

---

## 3. Steelmanned alternatives (this iteration's decisions)

| Design choice | Strongest alternative | Decision |
|---|---|---|
| Pin τ=0 on the command line of every treatment run | Assert the config default equals 0 in a test | **Pin.** A test protects the value at test time; the campaign runs for days and the YAML is editable in between (a pre-launch human edit would evade even the analyst tree-integrity check). Command-line pinning makes the cell definition part of the launched command — R2.4's own logic for temperature. |
| Mocked e2e pytest of `main()` | A `--rehearse` CLI flag with stub steps | **Pytest.** Identical orchestration code is exercised on every suite run with zero new production branches; a rehearse flag adds code paths (stub plumbing, flag interactions) to the launcher right before launch, the exact moment new code is most dangerous, for marginal realism (the real-machine specifics — Docker, vLLM — are exactly what a rehearsal must stub anyway). |
| Archive stale decision files to `*.superseded` | Number decisions by counting existing files | **Archive.** Keeps the 1:1 mapping between analyst phase n and `decision_n.json` that the R6.5 "decisions are checked-in artifacts" disclosure relies on; count-based numbering desyncs file names from phase numbers and silently legitimizes the stale file instead of quarantining it. |
| `needs_servers` only on agent-run steps | Keep ensure-everywhere (defense in depth) | **Per-step.** The "depth" defended nothing: eval/metrics never contact vLLM/NLI, while the cost was a real campaign-abort mode on dead-container-after-runs. Run steps still preflight on every attempt. |
| Stop campaign on invalid analyst decision | Retry the analyst once | **Stop (unchanged).** A stop is safe and resumable; a retry loop lets a confused analyst burn turns and invites prompt-drift. The human reads the log and resumes. |

Standing decisions re-examined and left in place: trajectory-matched budget
(conservative direction, measured on both arms); τ=0 superset headline + post-hoc
sweep (now command-pinned); intent-summary clustering substrate; fixed-sequence
H1→H2 family (untouched — still no data); majority-signature selector with
degeneracy disclosure; 10-easy-SymPy scope; no-retune rule under gate
saturation; the R6.5 adaptive-execution boundary.

Known limitation recorded, not fixed (disclosed): `unexpected_tree_changes`
diffs porcelain lines, so an analyst edit to a file ALREADY dirty at campaign
start is invisible — mitigated by launching from a clean tree (currently only
the loop's own `history.jsonl` is dirty) and by the campaign log capturing the
pre-launch fingerprint. Eval steps do not preflight Docker availability; a
Docker outage fails loudly in the step itself and is retried/resumable — an
explicit preflight is optional polish, not a correctness hole.

---

## 4. Actions taken this iteration

All verified: **117 pytest pass** (111 → 117; +6 new, 0 removed), `py_compile`
clean on every touched script, campaign dry-run prints the full correct plan
including the new `--entropy-threshold 0` pin.

1. `scripts/run_campaign.py` — explicit `--entropy-threshold 0` on every
   treatment run (T1); `needs_servers` step gating (O2); stale-decision-file
   archiving in `run_analyst` (O1); docstring pointer §6→§2.2 and the (T, τ)
   cell-definition note (T2).
2. `RESULTS.md` §5 — documented treatment command pins `--entropy-threshold 0`
   with the rationale comment.
3. `GOLD_STANDARD.md` R2.4 — τ-explicitness clause (amendment record:
   `spec_amendments/applied_09_explicit_tau_pin.md`; tripwire check inside —
   nothing flips to pass; the added obligation is discharged by
   same-iteration work).
4. `tests/test_run_campaign.py` — +6: τ pin over the whole menu; server
   gating over the whole menu; mocked end-to-end campaign loop (ordering,
   analyst gating, state persistence, server-preflight counts); resume skips
   completed Phase A; STOP file aborts before any step; stale decision file
   archived, never validated as fresh.

## 5. What still requires a human / GPU

1. Ratify the iteration-9 amendment (`applied_09_explicit_tau_pin.md`) and any
   earlier unratified ones — independent spec-critic review per ratchet v2.
2. **Launch the campaign from this branch**: `python scripts/run_campaign.py
   --go` (Phase A = the pre-registered confirmatory cell; vLLM model-identity,
   disk, and NLI preflights are in the driver; NLI on CPU per the measured
   VRAM headroom). Launch from a clean working tree so the analyst
   tree-integrity check has a complete baseline.
3. After the runs: check threat 11 first (realized entropy distribution in
   `tau_sweep_*.json`), read H1 against `nonempty_patch_fraction`, verify the
   both-arm `budget_audit_*.json` token totals confirm the conservative
   matching direction, fill RESULTS §5 from script output only.
4. Git hygiene: decide whether this campaign branch becomes mainline
   (unchanged from iteration 8).

## 6. Verdict logic

This iteration found and fixed one major pre-registration explicitness gap
(the τ=0 definition of the confirmatory cell was not in any launch command),
two moderate operational integrity holes in the campaign driver (stale
decision file on resume; server preflight blocking post-processing), and
closed the queued rehearsal question with an executed end-to-end test of the
campaign state machine. The statistics stack was re-derived from first
principles with no defect found. Per the charter, finding real fixable issues
means `gold_standard_met` = **false**; the loop should pass once more. The
next iteration should arrive with fresh eyes on a branch whose last pass
changed only the launcher — if it too finds nothing substantive in the design
or the math, the design has nothing left to confess and the remaining actions
are human-only (ratification + GPU launch).
