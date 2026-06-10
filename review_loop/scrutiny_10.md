# Scrutiny record — iteration 10 (first-principles design review)

Charter: fresh-eyes principal-scientist pass over the campaign branch whose
last delta (iteration 9) touched only the launcher. Iteration 9 queued: re-check
the analyst prompt against the final design docs and confirm no new
inconsistency; beyond that, this pass re-derived the statistics from scratch
(again, independently) and hunted for anything the nine prior passes had not
confessed.

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

Unchanged from iterations 4–9; re-examined and still the right claim. It is a
diversity/mode-collapse claim with a coverage consequence, scoped to the easy
band, with the null explicitly publishable. The repo serves THIS claim — no
weaker substitute, no framing drift found this pass.

**Ideal evidence:** the standing machinery (confirmatory cell mirroring the
causal chain; exact small-n inference with printed power floors;
mechanism-independent diversity metric; draw accounting from the fork decision;
both-arm budget audit; command-pinned cell definition; execution-tested
launcher) **plus this iteration's closing clause: every integrity property the
design asserts about its adaptive executor must be checked by the executor, on
BOTH planes — code AND data — and every inference rule the analysis states in
prose must be encoded in the artifact a reader actually opens.**

**Minimal sufficient experiment set:** unchanged — (1) strategy arm T=0.7 τ=0
superset; (2) matched-k vanilla T=0.7; (3) exploratory T∈{0.2, 1.0}; (4) SDLG
arm T=0.7; τ ablation post-hoc at zero GPU. The campaign driver encodes exactly
this.

---

## 2. Findings (with evidence pointers)

### TRUTHFULLY

- **T1 (moderate, fixed) — the analyst integrity guard protected the code
  plane but not the data plane.** `unexpected_tree_changes`
  (`run_campaign.py`) deliberately skips `results/` and `campaign_decisions/`
  because the campaign's own logs write there during the analyst window. But
  those trees hold exactly the data every later scheduling decision reads, the
  numbers RESULTS §5 will be filled from, and the decision files that ARE the
  R6.5 audit chain. The driver's own docstring states the standard the design
  holds itself to — the analyst runs with permissions skipped and "prompts are
  not enforcement" — and by that standard, an analyst edit to a metrics JSON
  (steering every later phase) or to a *prior* decision file (rewriting the
  audit trail) was undetectable: tracked files were explicitly skipped, and
  the real campaign's results dirs are untracked, where porcelain shows no
  content-level change at all. *Fix:* `artifact_fingerprint` — SHA-256 of
  every file under `results/` (excluding `results/campaign/`, the campaign's
  own mutable area) and `campaign_decisions/` (excluding the current decision
  file), snapshotted after stale-decision archiving and verified after the
  analyst subprocess; any delta stops the campaign with the changed paths
  named. R6.5 amended to require this
  (`spec_amendments/applied_10_analyst_artifact_integrity.md`). Tests: tamper
  with a metrics file → stop; tamper with a prior decision file → stop; clean
  run (decision file + campaign log area changes) → passes.
- **T2 (checked, clean) — iteration 9's queued analyst-prompt audit.** The
  prompt's design pointers (scrutiny_07/04/03, RESULTS §2.2 + §6 threat 11,
  both-arm budget audit, first-run-is-confirmatory pin, null-is-valid) match
  the current design; iterations 8–9 changed launcher mechanics, not anything
  the analyst decides on. One improvement applied: the prompt now points at
  the new `confirmatory_family` field (see M2) instead of restating the gate
  rule only in prose.

### MATHEMATICALLY (re-derived independently, not trusted)

- **M1 (no defect) — estimator stack re-derivation, second independent pass.**
  (a) `pass_at_k` product form: Π_{i=n−c+1}^{n}(1−k/i) =
  [(n−k)!(n−c)!]/[n!(n−c−k)!] = C(n−c,k)/C(n,k) — re-derived from the
  factorial identity; edge cases (c=0 → 0; n−c<k → 1; k>n → error) correct.
  (b) Rarefaction: E[#distinct at k] = Σ_sig P(≥1 of its m copies drawn) =
  Σ_sig [1 − C(n−m,k)/C(n,k)] — same identity; empties stay in n, contribute
  no signature; equals the raw distinct count at k=n. (c) Exact sign-flip
  test: enumerates all 2^n masks including identity, statistic |mean|,
  two-sided by construction, valid under arm-exchangeability symmetry of the
  paired null. (d) Tie floor: z zeros duplicate every statistic 2^z times and
  the two global sign choices on the m=n−z nonzero entries attain the observed
  |mean| → p ≥ 2^(1+z−n); at n=10, rejection needs ≥6 same-direction nonzero
  gains — matches the printed disclosure. (e) The achievable-entropy grid for
  N=5 recomputed by hand: partitions [5],[4,1],[3,2],[3,1,1],[2,2,1],
  [2,1,1,1],[1⁵] → {0, 0.5004, 0.6730, 0.9503, 1.0549, 1.3322, 1.6094} nats —
  matches `partition_entropies(5)` and the documented 7-value grid, including
  the (2,2,1) boundary case (1.054920… vs the 3-decimal log's 1.055) the
  sweep's full-precision recomputation exists for. (f) Matching direction
  re-checked: at matched trajectory count the control pays k full SEARCHes vs
  the treatment's one shared SEARCH, and the treatment's failed creation-time
  draws also count into k — both push conservative; token-matching would
  favor the treatment and is correctly not used. (g) KLE τ=0 semantics on the
  kernel ablation cells: the von Neumann entropy of ρ = K_t/tr(K_t) is
  strictly positive even for a single merged cluster (ρ has full support over
  N nodes), so `--entropy-threshold 0` on a kernel cell means always-branch —
  the superset premise holds there too, and `tau_sweep` correctly refuses to
  overwrite a logged kernel entropy with the partition recomputation
  (`entropy_source` flag). No defect found anywhere in this stack.
- **M2 (minor, fixed) — the fixed-sequence gate existed only in prose.** The
  multiple-comparison control (H2 confirmatory iff H1 rejects, family-wise
  α=0.05) lived in RESULTS §2.2 and the analyst prompt; the metrics JSON
  printed two flat p-values. A reader — including the campaign's own
  data-driven analyst — could read an H2 p<0.05 as confirmatory with the gate
  closed. *Fix:* `compare()` now emits a `confirmatory_family` block (H1 p +
  rejects flag, H2 p + derived status, note scoping confirmatory status to
  the pre-registered cell), printed in the console summary; tested gate-open
  (10 instances, treatment all-distinct vs vanilla collapsed → exact p =
  2/1024 → H2 confirmatory) and gate-closed (identical pools → p=1 → H2
  descriptive). R6.5's amendment covers this clause too.

### OPERATIONALLY

- **O1 (minor, fixed) — `--max-phases` was a per-invocation budget and resume
  collided decision numbering.** On `--resume` the analyst loop restarted at
  n=1: each resume refilled the analyst-phase budget (bounded only by the
  menu), and decision_01.json from the completed first phase would be
  archived to `.superseded` and re-numbered — breaking the 1:1
  analyst-phase ↔ decision-file mapping the R6.5 audit-chain disclosure
  relies on, and overloading `.superseded` (meant for *interrupted*
  decisions) with completed ones. *Fix:* the loop counter continues from the
  count of completed analyst-chosen specs (`completed_specs` minus Phase A),
  making the cap global and the numbering stable; an interrupted phase
  correctly reuses its n (stale file archived, decision re-made). Test:
  resume with 2 analyst phases done and `--max-phases 4` → analyst called
  with n=3, 4 only.

### PHILOSOPHICALLY

- **P1 — framing re-audited, still coherent and non-circular.** H1's
  diversity metric (structural patch signatures, rarefied at matched k\*) is
  independent of the NLI machinery that decides branching; "diversity" is
  defined without reference to the mechanism. The falsifiable predictions
  remain falsifiable at n=10 for H1; H2's tie floor is printed beside every p.
  The §0.1 five-mechanism family and the entropy blind spot remain the
  strongest honest version of the story.
- **P2 — weakest joints, unchanged in kind, all pre-answered:** gate
  saturation (threat 11, pre-committed negative reading), n=10/one-repo/easy
  band (scoped claims, threats 1/2/4), τ=0 headline ("the title's gate never
  gates" — answered by the post-hoc sweep being the gate's evaluation plus
  the plain partition-quantization statement). Nothing new to confess here.
- **P3 — the generalizing lesson of 5→10:** each iteration moved one more
  asserted invariant into something executable. Iteration 10's instance: an
  integrity guard that covers half the attack surface (code but not data) is
  an assertion about the other half; and an inference rule carried only in
  prose is an assertion about every future reader. Both are now checked or
  encoded. After this pass, the spec → metrics → producers → artifacts →
  branch → commands → state machine → **adaptive-executor data plane** chain
  has an executable guard at every layer this review could identify.

---

## 3. Steelmanned alternatives (this iteration's decisions)

| Design choice | Strongest alternative | Decision |
|---|---|---|
| Content-hash the data plane across the analyst window | Narrow the porcelain exclusion to `results/campaign/` only | **Hash.** Porcelain shows untracked dirs as a single entry and no content-level change for files inside them; the real campaign's results dirs are untracked at creation, so porcelain-based rules are structurally blind exactly where the data lives. Hashes see tracked and untracked alike. |
| Hash everything under results/ + campaign_decisions/ | Recompute metrics after each analyst call and diff | **Hash.** Recomputation cannot detect predictions/eval tampering (recomputing from tampered inputs is self-consistent); hashing detects any byte change at lower cost (seconds per analyst call, ≤5 calls). |
| SHA-256 content hash | size+mtime snapshot | **SHA-256.** Length-preserving rewrites defeat size; tools can preserve mtimes; the cost difference is negligible at this scale. |
| Graceful stop on tamper (campaign keeps results, human resumes) | Raise/abort hard | **Stop.** Matches every other analyst-failure mode (timeout, no file, invalid JSON); the artifacts are regenerable from predictions, and the stop message says so. |
| Global `--max-phases` via completed-spec count | Persist a phase counter in state | **Count from completed_specs.** No new state field to desync; an interrupted phase naturally reuses its number (and the stale-archive path handles its half-written decision); the cap's meaning ("at most N analyst-chosen cells") falls out directly. |
| Encode the H1→H2 gate in the metrics JSON | Keep the rule in RESULTS §2.2 + analyst prompt only | **Encode.** A multiple-comparison control that depends on every reader remembering a prose rule is weaker than one the artifact states; the block also scopes confirmatory status to the pre-registered cell so exploratory cells cannot borrow it. |

Standing decisions re-examined and left in place: trajectory-matched budget
(conservative direction, measured both arms); τ=0 superset headline +
command-pinned post-hoc sweep; intent-summary clustering substrate;
fixed-sequence H1→H2 family (untouched — still no data); majority-signature
selector with degeneracy disclosure; 10-easy-SymPy scope; no-retune rule under
gate saturation; the R6.5 adaptive boundary; stop-don't-retry on invalid
analyst decisions.

Known limitations recorded, not fixed (disclosed): the porcelain check still
cannot see analyst edits to files already dirty at campaign start (mitigated:
launch from a clean tree; the data plane is now hash-guarded regardless). Eval
steps still do not preflight Docker (fails loudly in-step, retried/resumable).
The artifact guard assumes a cooperative-but-fallible analyst, not a motivated
adversary with shell access — an adversarial analyst could in principle attack
the driver process itself; that threat class is out of scope for a
self-hosted research campaign and noted here for honesty.

---

## 4. Actions taken this iteration

All verified: **121 pytest pass** (117 → 121; +4 new, 0 removed), `py_compile`
clean on every touched script, campaign dry-run prints the full correct plan.

1. `scripts/run_campaign.py` — `artifact_fingerprint`/`changed_artifacts` +
   integrity check across the analyst window (T1); global resume-stable
   `--max-phases` counter and decision numbering (O1); analyst prompt points
   at `confirmatory_family`; docstring updated (guard planes, global cap).
2. `scripts/compute_metrics.py` — `confirmatory_family` block in `compare()`
   + console line (M2).
3. `GOLD_STANDARD.md` R6.5 — data-plane integrity + artifact-encoded gate
   clauses (amendment record:
   `spec_amendments/applied_10_analyst_artifact_integrity.md`; tripwire
   check inside — adds obligations discharged same-iteration, flips nothing).
4. `RESULTS.md` §2.2 — adaptive-execution consequence (iv) (integrity guard)
   and the gate-encoding sentence.
5. `tests/test_run_campaign.py` — +3: metrics/prior-decision tamper stops the
   campaign; clean analyst run passes the guard; resume continues phase count
   and numbering. (Stale-decision test also patches RESULTS so the
   fingerprint walks tmp dirs.)
6. `tests/test_compute_metrics.py` — +1: gate open (exact p = 2/1024 →
   H2 confirmatory) and gate closed (p = 1 → H2 descriptive).

## 5. What still requires a human / GPU

1. Ratify the iteration-10 amendment
   (`applied_10_analyst_artifact_integrity.md`) and any earlier unratified
   ones — independent spec-critic review per ratchet v2.
2. **Launch the campaign from this branch from a CLEAN working tree**:
   `python scripts/run_campaign.py --go`. Phase A is the pre-registered
   confirmatory cell; T and τ are pinned in the launched commands; the
   analyst window is now integrity-guarded on both planes.
3. After the runs: check threat 11 first (realized entropy distribution in
   `tau_sweep_*.json`), read H1 against `nonempty_patch_fraction`, read H2
   through the `confirmatory_family` block, verify both-arm
   `budget_audit_*.json` token totals confirm the conservative matching
   direction, fill RESULTS §5 from script output only.
4. Git hygiene: decide whether this campaign branch becomes mainline.

## 6. Verdict logic

This iteration found and fixed one moderate integrity hole in the adaptive
executor (the data plane the analyst reads — and every later phase trusts —
was unguarded while the code plane was guarded), one minor inference-integrity
gap (the H1→H2 gate existed only in prose, not in the artifact), and one minor
guardrail drift (per-invocation `--max-phases`, colliding decision numbering
on resume). The statistics stack was re-derived independently a second time
with no defect found; iteration 9's queued analyst-prompt audit came back
clean. Per the charter, finding real fixable issues means `gold_standard_met`
= **false**; the loop should pass once more. The next iteration arrives at a
branch whose last two deltas touched only the launcher and the artifact
self-description — if fresh eyes find nothing substantive there or anywhere
else, the design has nothing left to confess and the remaining actions are
human-only (ratification + GPU launch).
