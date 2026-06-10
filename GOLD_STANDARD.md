# Gold Standard — Definition of "Publication-Ready" for this project

This document is the **objective rubric** the autonomous review/improve loop
(`scripts/review_improve_loop.py`) scores the repository against. It defines what
a high-quality, publishable version of *Semantic Entropy Clustering for Diverse
Agentic Code Generation* looks like, so that "ready for the paper" is a checklist
verdict, not a vibe. Every iteration must score each rubric item as
`pass | partial | fail` **with an evidence pointer** (file path, command, or
results artifact), and may only declare `gold_standard_met: true` when every
**[BLOCKER]** item is `pass` and no `blocking_gaps` remain.

**Spec evolution — the ratchet** *(policy v2, human-ratified 2026-06-09: value-level
design judgment delegated to the scrutiny charter; the anti-self-serving tripwire
retained).* This rubric may evolve under the agent's own research judgment, with one
inviolable constraint: never in a way the optimizer can use to grade the
**already-existing artifact** into a pass without doing the work. Three channels:

- **Auto-applicable corrections.** The loop MAY directly edit this file to apply a
  *correction* — an amendment that is mathematically or logically **derivable** (a
  biased estimator, an inconsistent definition, a broken identity) AND that
  **increases** rigor. It must record the derivation + classification in
  `review_loop/spec_amendments/`.
- **Value-level design amendments (scrutiny charter).** The loop MAY directly apply
  amendments that rest on **research judgment** — redefining a comparison it argues
  is fairer, recalibrating a default, adding/strengthening a requirement, or
  explicitly de-scoping a claim — PROVIDED the amendment (i) is argued from the
  science (papers, math, the end-goal claim), not from convenience; (ii) does NOT
  have the effect of flipping any rubric item to `pass` for the artifact as it
  exists at the time of the edit, nor of reducing the evidence required for the
  headline claim while the corresponding work remains undone; (iii) any de-scoping
  is disclosed as such in the writeup's threats/limitations; and (iv) the full
  rationale + rejected alternatives are recorded in
  `review_loop/spec_amendments/applied_<NN>_<slug>.md`.
- **Quarantined proposals.** Any amendment whose effect is that the current
  artifact newly passes, or that trades away evidence for the headline claim to
  avoid work, MUST NOT be applied — it is written as a proposal in
  `review_loop/spec_amendments/` for human ratification.

Enforcement is external, not trusted: after every iteration an independent
adversarial **spec-critic** reviews any change to this file from the lens of a
senior researcher. It APPROVES corrections and scientifically-argued value
amendments (channels 1–2), and the wrapper **reverts** any edit whose effect is
self-serving — making the existing artifact pass, or weakening the evidence bar for
convenience (channel 3 violations). Reverted edits are preserved as diffs in
`review_loop/spec_amendments/` so the idea is never lost, only the autonomous
application. The agent therefore has full design authority EXCEPT the one power no
self-grading agent can safely hold: moving the finish line onto its own current
position.

**Section order:** §0/§0.1 are framing; §1–§9 are the rubric (all BLOCKERs come
first, §1–§8; §9 is a tracked roadmap dimension); §10 (writeup) is last because it
*consumes* everything above and is the only non-blocker rubric section.

---

## 0. The claim we are defending

**Headline (a diversity claim, not a leaderboard claim):** at a *matched
trajectory budget and matched sampling temperature*, vanilla LLM resampling
mode-collapses to a few semantic forms, whereas semantic-entropy-guided branching
explores meaningfully distinct solutions — and is therefore more likely to
contain a passing fix. The contribution is the **semantic-entropy-gated branching
core**; the candidate generator (strategy-proposal vs SDLG) is a pluggable,
ablated input.

A **rigorous null/negative result is still publishable** — if branching does *not*
beat vanilla, the paper reports that honestly with the same statistical care. The
loop must NOT optimize toward a positive result; it optimizes toward *trustworthy
evidence either way*.

---

## 0.1 Why diversity helps — the holistic framing  `[narrative spine]`

**Unifying principle:** greedy / low-temperature decoding returns the *mode* of the
model's solution distribution; diversity is valuable precisely when the correct
solution is **not that mode**. Do NOT reduce this to any single mechanism (in
particular, not to "the answer is a set"). The mismatch between where the model
concentrates probability mass and where a correct solution lies arises for several
distinct reasons, and the paper must treat them as a holistic family:

1. **Multiplicity (aleatoric).** The problem is genuinely multi-valued —
   under-specified PRs, preference-laden choices, several structurally different
   patches that all pass. The answer *is a set*; diversity covers it and pass@k
   rises mechanically with set size.
2. **Residual epistemic uncertainty (post-search).** There is a single right fix,
   but after SEARCH the model still spreads belief over candidates and the correct
   one is not the argmax. Diversity surfaces the non-argmax candidate. SEARCH
   exists to *reduce* this; what remains is what branching exploits.
3. **Distributional / mode-collapse bias — EVEN WHEN THE ANSWER IS A SINGLE
   POINT.** This is the core motivation and the one the earlier framing missed.
   Pretraining frequency bias and RLHF sharpening concentrate sampling on a few
   "typical"/safe modes; a *better, correct* trajectory can exist in the model's
   support but in a low-probability region it rarely samples. Diversity (higher
   temperature, semantic branching, SDLG's targeted off-mode token substitution,
   cross-model pooling) deliberately explores *off the dominant mode* to reach it.
   Here diversity helps not because the answer is a set, but because the model's
   default concentration is a biased artifact of training that under-weights the
   true answer (knowledge collapse — Wright 2025; reduced variety — NoveltyBench,
   Zhang 2025).
4. **Multi-step compounding.** In an agent, an early commitment (a search finding,
   a reasoning step) determines which region of solution space is even reachable;
   a single trajectory locks in and cannot recover. Trajectory-level diversity
   hedges against early lock-in — distinct from token-level diversity.
5. **Model-space bias (§9).** Any one model is a single biased draw from "model
   space"; some correct solutions lie outside its well-sampled support entirely.
   Cross-model / ensemble diversity reaches solutions no single model covers — the
   limiting case of (3), where resampling one model cannot add mass that is not
   there.

**Semantic entropy as a proxy — and its blind spot.** High post-search semantic
entropy signals a *spread* distribution (cases 1–2), so the adaptive gate (branch
iff entropy > τ) spends diversity where the distribution is visibly multi-modal and
saves it where the mode is already decisive. But entropy measures *spread, not
correctness*: under case 3, mode collapse can make the model **confidently wrong**
— low entropy over a biased mode — which the entropy gate will NOT flag. That blind
spot is exactly why the paper carries mechanisms that do not depend on the model's
own confidence: SDLG forces off-mode exploration by construction, and cross-model
diversity (§9) escapes a single model's bias. The arms are therefore *complementary,
not redundant*, and the ablations must be read in that light.

Falsifiable predictions (test, do not assume):
- Branching's benefit over matched-k vanilla is **largest where the correct
  solution is off the model's mode** — partly predicted by post-search entropy
  (cases 1–2), but ALSO appearing on *low-entropy* instances where vanilla
  collapsed to a confident wrong mode and a diverse mechanism recovered the fix
  (case 3). The analysis must therefore look *beyond entropy alone*.
- It is ≈0 on instances where greedy already sits on the correct mode
  (straightforward, low-uncertainty, unbiased) — diversity then only costs compute.

Consequences this rubric enforces: (i) entropy is measured *after* SEARCH (R1.5,
§2); (ii) §5 stratifies the diversity benefit by uncertainty AND separately examines
off-mode recovery on low-entropy instances, so the narrative is never reduced to
"set vs point"; (iii) model identity is itself a diversity/uncertainty axis (§9).

---

## 1. Faithfulness to the reference methods  `[BLOCKER]`

The implementation must match (or deliberately, documentedly improve on) the
methods in `PDFs/`. Each sub-item names the source of truth.

- **R1.1 — SDLG (Aichberger 2025, `PDFs/Aichberger_2025_SDLG.pdf`, Alg. 1 & 2).**
  Gradient-based token attribution through a DeBERTa NLI model; rank
  (position, substitute) pairs by attribution × substitution × importance;
  substitute the high-impact token and let the LLM complete from the substitution
  point. Substitutions apply to **reasoning, not action tokens**. Verify
  `src/diversity/sdlg.py` implements this and that the branch point is a genuine
  write step (`phases.is_write_command`), not `echo`/submit/stderr redirects.
- **R1.2 — Semantic entropy & bidirectional-entailment clustering**
  (Farquhar 2024 `PDFs/farquhar_nature.pdf`; Kuhn 2023
  `PDFs/Farquhar_2024_Semantic_Entropy.pdf`, Alg. 1). The `greedy` clusterer must
  reproduce Algorithm 1; clustering applies a **consistent context policy at
  every call site** — for this project the NLI context is deliberately **empty**,
  a documented deviation from Kuhn's QA-style context-conditioning: prepending a
  shared problem-statement prefix to both sides saturates DeBERTa entailment on
  self-contained strategy/intent sentences (measured on the archived Phase A
  run-1: all pairs of five structurally distinct strategies score ≥0.94
  entailment WITH the prefix vs ≤0.55 without — the gate was measuring the
  prefix, not the strategies; reproduce with
  `scripts/diagnose_context_saturation.py`, pinned by
  `tests/test_clustering_context.py`). Discrete SE = `-Σ p_c log p_c`.
- **R1.3 — Clustering ablation variants.** `connected` (order-independent
  transitive closure) and `kernel` (Kernel Language Entropy, Nikitin 2024 — a
  genuine graph heat kernel `exp(-tL)`, von Neumann entropy of `ρ=K_t/tr K_t`,
  recovering `log K` in the well-separated limit) are correctly implemented and
  *correctly cited*. No method is described as something it is not.
- **R1.4 — Agent loop (ReAct, Yao 2023).** Phased SEARCH→PATCH→VERIFY with the
  documented tool-access boundaries per phase.
- **R1.5 — Adaptive branching.** Branch iff semantic entropy `> τ`; the τ gate is
  read from the *same* config key in every arm (no hardcoded divergence).

*Pass:* every sub-item verified against code with a file:line pointer, and any
deviation from a paper is intentional and documented (not a bug).

---

## 2. Experimental arms — scaffold- and temperature-matched  `[BLOCKER]`

- **R2.1 — Treatment arm(s) present and runnable:** `strategy_proposal` (and, if
  claimed, `sdlg`).
- **R2.2 — Matched-k vanilla control present and runnable:** `diversity_method:
  "none"` run **k times per instance**, where `k = #trajectories the treatment
  produced` (`scripts/run_resample_baseline.py`).
- **R2.3 — Scaffold-matched:** the control is the *same phased agent* with
  branching disabled — NOT the legacy `ReactAgent` baseline (that confounds
  branching with a different harness). Confirm the comparison used in the paper
  uses the phased `none` arm. **Scaffold-matched includes code-revision-matched:**
  every arm of a compared cell must be produced by the SAME committed code
  revision — the campaign pins HEAD at start (`code_revision` in the campaign
  state, stamped into the artifacts' provenance) and refuses to run any step
  after a commit/checkout or with tracked files modified. The need is measured,
  not hypothetical: Phase A run-2's treatment ran hours before the anti-gaming
  veto and container network isolation were committed, so the matched-k control
  would have faced a behavioral envelope (and a closed network) the treatment
  never did — 13.6% of the treatment's actions would have been vetoed under the
  guard the control would run with; the cell was archived as a protocol
  deviation rather than completed asymmetrically.
- **R2.4 — Temperature-matched:** one `sample_temperature` knob drives both the
  proposer and the vanilla baseline; the headline runs a sweep (0.2/0.7/1.0).
  Vanilla MUST sample at T>0 (temp=0 = deterministic = a strawman). The documented
  reproduction commands must set the temperature **explicitly on both arms** (a
  treatment command that silently inherits a different config default than the
  control's CLI temperature is an R2.4 violation). The same explicitness applies
  to the entropy-gate τ on the treatment arms: the confirmatory cell is defined
  by (T, τ), and the τ=0 superset premise underlying the post-hoc τ ablation
  (R3.3) must be pinned explicitly in the documented and campaign-driver
  commands (`--entropy-threshold 0`), never inherited from a config default.
  Arms are compared only at equal
  T; additionally report ONE robustness row comparing the treatment against
  vanilla's *best* sweep temperature, so a win cannot be an artifact of comparing
  against vanilla at an unfavorable T (the knob touches the whole agent in the
  vanilla arm but only the diversity source in the treatment arms — disclosed).
- **R2.5 — Results isolation:** each arm/temperature/clustering-strategy writes a
  **separate results dir**; no run overwrites another's predictions.

---

## 3. Ablations  `[BLOCKER for the ones claimed; otherwise scope explicitly]`

- **R3.1 — Diversity generator:** strategy_proposal vs sdlg, isolated (never
  stacked), each attributable — **including at the fallback layer**: an arm
  that cannot produce candidates at a decision point must produce NO branch
  there, never silently substitute a different generator. (A temperature-
  sampling fallback inside the sdlg arm mis-attributes the mechanism in the
  artifacts — the orchestrator records every fork as `sdlg_fork` — and a
  hardcoded fallback temperature breaks the R2.4 match in the sweep cells.)
- **R3.2 — Clustering strategy:** greedy vs connected vs kernel, each into its own
  results dir; τ recalibrated for kernel (non-transferable scale, documented).
- **R3.3 — τ / entropy-gate sensitivity:** evaluated **post-hoc from the τ=0
  superset run** by a runnable script (no separate GPU runs needed): the gate's
  no-branch action keeps exactly the dominant-cluster representative trajectory,
  which exists in the superset run and executed greedily, so every τ is evaluable
  by trajectory subsetting (branch-rate, trajectories used, gated pass-rate per τ).
  The analysis MUST disclose the entropy quantization plainly: at N candidates the
  discrete entropy takes only partition-of-N values (7 values at N=5), so τ is a
  cluster-partition-shape rule at small N, and the admissible τ grid is the
  achievable-entropy set, not a continuous dial. N must be held fixed across arms
  and instances for τ/strata comparability (plug-in entropy bias varies with K, N)
  — and because the proposer can under-deliver (<N parsed strategies), the
  **realized** N must be reported per instance with non-modal-N instances flagged
  and excluded from pooled τ/strata analyses, never silently mixed across
  quantization grids. Gate reconstruction must compare at full precision
  (recompute entropy from the logged cluster partition; a rounded log value can
  cross a gate boundary).

*Pass:* each ablation either has results, or is explicitly de-scoped in the
writeup with justification.

---

## 4. Metrics — independent and unbiased  `[BLOCKER]`

- **R4.1 — Coverage:** `diverse-pass@k` computed with the **unbiased Chen et al.
  (2021) estimator** on *both* arms, at matched k. Oracle/coverage framing stated
  honestly (it is an upper bound, not deployable accuracy). **Matched k must be
  enforced at METRIC time, not only at run time:** the two-arm comparison evaluates
  both arms at the common per-instance k\* = min(k_A, k_B) via the Chen estimator
  and reports every k-mismatched instance — failed resamples, `--max-k` caps, or
  capture losses must never silently hand the larger arm a mechanical any-pass
  advantage. **Eval-record completeness:** the per-arm eval artifacts must contain
  exactly ONE row per genuine trajectory — duplicate patches may be evaluated once
  for compute, but every duplicate trajectory inherits its representative's
  outcome (marked as propagated), and empty patches count as failed draws without
  a container run — because the Chen estimator's (n, c) must count what the arm
  *produced*. The vanilla arm's duplicate patches are the mode-collapse signal
  itself; an eval step that drops them deflates vanilla's k and silently
  subsamples the treatment's coverage at the shrunken k\*. Metric loaders must
  drop the best-of duplicate row consistently whether its trajectory id is
  `"primary"` or null, and all post-hoc parsers of append-mode run logs must read
  the LAST run's block (re-runs append; predictions/metadata reflect the last run).
  **Eval-outcome integrity:** every `resolved` in the eval record must be a genuine
  harness verdict. The SWE-bench harness swallows ALL per-instance errors (it
  writes no report and continues), so a missing report must be **classified** from
  the harness's own logs: patch-apply failure and test timeout are attributable to
  the patch and are recorded as failed draws with their reason; ANY other cause is
  an eval-infrastructure error that must abort the eval step loudly **without
  writing the eval record** — a Docker flake silently scored as `resolved: false`
  corrupts the Chen (n, c) in whichever arm it hits, and a written record would be
  frozen forever by the resume-marker skip. **Stale-report immunity:** the harness
  returns an existing report keyed by (run_id, model, instance) *without
  re-evaluating* — patch content is not in its key — so eval run ids must embed a
  content hash of the patch; a patch-blind run id lets a re-run inherit a stale
  verdict for a different patch (and the content key makes post-stop retries
  cheap: identical patches legitimately reuse their cached reports).
- **R4.2 — Diversity measured INDEPENDENTLY of the branching signal.** The
  diversity of the final outputs must NOT be measured with the same DeBERTa-NLI
  clustering used to *decide* branching (circular). Use an independent metric over
  **final patches** — structural/AST or normalized edit distance, and/or
  behavioral diversity — reported as distinct-solution counts per arm. This metric
  must exist as a runnable script over the predictions artifacts. Cross-arm
  distinct-count differences at unequal sample counts must use a rarefaction
  estimator (expected distinct in a random k\*-subset) — raw distinct counts rise
  mechanically with sample size. **Productivity-confound diagnostics:** an empty
  patch lowers the rarefied distinct count exactly like a duplicate, so a
  diversity "gain" can masquerade for a patch-production-rate gap; the comparison
  must report each arm's non-empty patch fraction and a descriptive non-empty-only
  rarefied-gain robustness row (computed at k\*_ne = min non-empty count), fixed
  before the runs, and the writeup must state the exact-signature granularity
  (lexical variants count as distinct in both arms) with its bias direction.
- **R4.3 — Diversity measured on FINAL patches, not proposal-time branches**
  (branches can converge downstream).
- **R4.4 (strengthening, not blocker) — Selection-aware accuracy:** a *deployable*
  selector — one computable from the run artifacts alone (e.g. majority vote over
  normalized final-patch signatures with deterministic tie-breaks), no hidden
  tests, no oracle — → `selected-pass@1` on both arms, so the paper does not
  overclaim the oracle number. The selector's arm-asymmetry must be disclosed:
  on the branching arm patches are one-per-cluster by construction, so a
  majority-signature vote typically degenerates to its tie-break — the analysis
  must report how often (degenerate-tiebreak count per arm), and must not invent
  additional selectors after seeing results.

---

## 5. Diversity-benefit analysis (uncertainty AND mode-collapse)  `[BLOCKER for the narrative]`

The headline thesis (§0.1) is that diversity helps when the correct solution is off
the model's mode — driven by *several* mechanisms, not just uncertainty. A pooled
average hides this; the analysis must decompose the benefit by mechanism. (This
sits right after the metrics it consumes — it turns coverage/diversity numbers into
the paper's central claim.) The analysis must NOT reduce the story to "set vs
point"; it must cover at least the entropy-driven and the mode-collapse-driven
cases separately.

- **R5.1** The full §0.1 framing (all five mechanisms + the entropy blind spot) is
  reflected in the writeup (see R10.3) — not just the set-valued case.
- **R5.2 — Stratified benefit (cases 1–2).** A runnable script buckets instances by
  *post-search semantic entropy* (per instance: `phased_decisions.log`
  `Entropy:` line for the strategy arm; `branching_log.json` `entropy` for the SDLG
  arm; or `trace.jsonl`) and reports `diverse-pass@k` gain (treatment − matched-k
  vanilla) and the independent diversity metric *per stratum*. Hypothesis to test,
  not assume: part of the gain concentrates in high-entropy strata.
- **R5.3 — Set-valued evidence (case 1).** Identify and report instances where ≥2
  structurally distinct patches both pass the hidden tests — direct evidence the
  solution is a *set*.
- **R5.4 — Off-mode recovery (case 3) — the mode-collapse signature.** Identify
  instances where a diverse mechanism produced a *passing* fix that matched-k
  vanilla did NOT, **while post-search entropy was LOW** (the model was confidently
  on a wrong mode). These are the cases the entropy gate cannot predict and are the
  direct evidence that diversity helps by escaping training bias, not only by
  covering a set. Report SDLG vs strategy-proposal vs (where available) cross-model
  separately here, since they address this case differently.
- **R5.5 — Honest counter-analysis.** Show diversity does NOT help (and may waste
  budget) on low-uncertainty/unbiased instances where greedy already sits on the
  correct mode, and that the adaptive τ gate would have skipped branching there.
  Guards against a "diversity always helps" overclaim and motivates the gate — while
  acknowledging the gate's blind spot from R5.4.

*Pass:* the framing is present and the analysis script (entropy stratification +
off-mode-recovery detection) is implemented and runnable over the artifacts. The
*executed* numbers may be a known gap pending the GPU runs — record that in
`next_actions`; do not fabricate them.

---

## 6. Statistical rigor & honest scope  `[BLOCKER]`

- **R6.1 — Uncertainty:** every headline number carries a confidence interval
  (bootstrap over instances acceptable); no point estimates without spread. The
  primary arm-vs-arm gain additionally carries an **exact paired sign-flip
  (permutation) p-value** — at n ≤ 20 all 2^n sign patterns are enumerable, and at
  this project's n=10 the exact test, not a percentile bootstrap over lumpy 0/1
  gains, is the decision-grade inference.
- **R6.2 — Scope claims match the data:** claims are scoped to the instance set
  actually run (currently 10 easy SymPy; goal: full SWE-bench Lite). No
  generalization beyond what was measured.
- **R6.5 — Pre-registered confirmatory family (multiple-comparison control):** the
  sweep × arms × ablations grid is many cells; exactly ONE comparison **cell** is
  named confirmatory *before* the GPU runs (currently: strategy-proposal, greedy,
  τ=0 superset vs matched-k vanilla at T=0.7). Within that cell, the two halves of
  the §0 headline form a **fixed-sequence (gatekeeping) family** at family-wise
  α=0.05: **H1 = rarefied distinct-patch gain at matched k\*** (the diversity /
  mode-collapse half — the title claim), then **H2 = matched-k\* diverse-pass@k
  gain**, each with the exact sign-flip test, H2 confirmatory **only if H1
  rejects** (otherwise H2 is descriptive). The order is fixed by the causal chain
  (coverage can only move through diversity), not by the data; it makes the
  coverage claim strictly harder than a lone H2 endpoint while giving the §0
  diversity claim — previously descriptive-only — a confirmatory test. Every other
  cell — temperatures, SDLG, clustering/τ ablations, entropy strata, off-mode
  candidates — is labeled exploratory/descriptive in the writeup. Changing the
  family or its order after seeing results is forbidden; if the runs motivate a
  different endpoint, that is reported as a post-hoc finding. **Power floor
  disclosure required:** the exact sign-flip p has a tie-imposed floor
  p ≥ 2^(1+z−n) (z = zero gains); the analysis must report this
  (`min_achievable_p`) beside every sign-flip p so a null is never presented as
  evidence of no effect when the test could not have rejected. **Adaptive
  execution boundary:** if run scheduling is automated with a data-reading
  agent (the campaign driver), its adaptivity may touch **exploratory cells
  only** — the confirmatory cell runs first, exactly once, by a fixed command
  sequence; the FIRST completed confirmatory run is the confirmatory dataset
  (a repeat draw is variance estimation, never a replacement or pooling
  partner); every scheduling decision is a checked-in artifact with a written
  rationale; and the writeup discloses that the exploratory cell set is
  data-dependent. **Adaptivity reads measured data, never alters it:** the
  driver must verify — not assume — that the analyst process modified no
  existing results or decision artifact (an integrity fingerprint of the data
  plane taken before and checked after each analyst invocation; any change
  stops the campaign loudly), since prompts alone are not enforcement and a
  guard that covers only code/config leaves the very numbers later decisions
  and the writeup consume unprotected. The fixed-sequence H1→H2 gate must
  also be encoded in the metrics artifact itself (H2's confirmatory vs
  descriptive status derived from H1's p in the output), not carried solely
  in prose a reader can miss.
- **R6.3 — Budget-fairness audit:** per-trajectory step distributions reported for
  passing branches (the `step_limit` 250→300 asymmetry must be shown not to
  manufacture wins), and per-arm token/compute accounting reported — with any
  systematic exclusions (calls not stored in the per-trajectory transcripts, e.g.
  the strategy proposer and NLI passes) disclosed alongside, including which arm
  the exclusion favors. **Per-arm means BOTH arms:** the audit tool must read
  each arm's actual artifact layout (treatment `<iid>/metadata.json`; control
  `<iid>/run<idx>/<iid>/metadata.json`, draws keyed by the predictions tid
  `run<idx>`), stage-tested on both — a fairness comparison whose tool can only
  read the treatment arm is an assertion, not an audit — and the documented
  workflow must audit the control alongside the treatment.
- **R6.4 — Threats to validity** enumerated and either addressed or acknowledged
  (cherry-picked difficulty band, single repo, oracle selection, n).

---

## 7. Reproducibility & artifact quality  `[BLOCKER]`

- **R7.1** One documented command per arm reproduces its predictions; configs are
  checked in; the NLI/vLLM/Docker prerequisites are documented.
- **R7.2** No silent data loss: every completed trajectory's patch is captured
  before container teardown; predictions JSONL schema is documented and stable;
  the evaluation step writes its per-instance eval records into the **arm's own
  results dir** (no hardcoded shared default — evaluating one arm must not be
  able to overwrite another arm's eval files), and every genuine trajectory of
  the run appears in the eval record (see R4.1 eval-record completeness).
  **Predictions-record completeness (one layer up):** the eval record is built
  from the per-trajectory predictions file, so the run drivers in EVERY arm
  must write **one prediction row per genuine draw** — a trajectory that
  failed or produced no diff still consumed budget and must appear as an
  empty-patch row, exactly as the resample driver records its unproductive
  resamples (a patch captured on a *failed* trajectory must likewise not be
  discarded). Dropping an arm's own unproductive draws deflates that arm's
  metric-time k and inflates its diverse-pass@k\* and rarefied-distinct levels
  against the other arm. Post-hoc loaders must be **run-batch aware**: on a
  re-run that produced fewer trajectories, rows from the superseded run must
  not survive into the diversity pool (score only the last primary-delimited
  batch, matching the eval driver), and the matched-k driver must warn when a
  treatment metadata's per-trajectory entries disagree with its
  `total_trajectories` (old-driver or interrupted artifact).
  **Draw accounting starts at the fork decision:** a fork that fails at
  *creation* (container/clone/injection error) is still a genuine draw and
  must be recorded as a failed empty-patch trajectory — exactly as the
  resample driver records a crashed resample — never silently skipped; a
  branch whose injected response *submits* during creation is a completed
  draw whose patch must be captured, never discarded. Artifacts without
  batch delimiters (the resample arm's all-trajectories file has no
  best-of rows to split on) must be made re-run-safe at the **producer** —
  per-instance row replacement — since no parser-side last-batch rule can
  isolate a smaller-k re-run there.
- **R7.3** Figures/tables are regenerable from the predictions artifacts by a
  checked-in script.
- **R7.4** Determinism knobs (seeds where applicable, model/temperature, package
  versions) are recorded with the results.

---

## 8. Internal correctness (no latent bugs that corrupt evidence)  `[BLOCKER]`

- **R8.1** Write-command / branch-point detection is correct so SDLG and
  read-budget logic fire at the right step: no false positives from `echo`/
  submit/stderr redirects, from comparison operators inside quoted programs
  (`awk 'NR>=350 …'` — measured on pilot logs), or from heredoc body content;
  and no false negatives for writes in non-final `&&`/`;`/newline segments
  (`sed -i … && pytest`) — detection inspects every top-level command segment
  with quoted spans and heredoc bodies excluded. The SEARCH phase must
  **enforce** its read-only boundary with this same detector (a prefix
  allowlist alone admits `echo … > file` under the allowed `echo` prefix):
  fork-state consistency is load-bearing because strategy forks replay search
  *messages* into fresh containers rather than cloning the searched
  container's *filesystem*, so a SEARCH-phase write would silently desync the
  root trajectory's starting state from every fork's. Submission is
  VERIFY-only and must be decided before any prefix match. Phase boundaries
  that are NOT enforced (PATCH/VERIFY allowlists are prompt-level guidance)
  must be documented as such, not implied to be checked.
- **R8.2** Config defaults are centralized and consistent across call sites.
- **R8.3** Captured fallback patches are source-only (comparable to the curated
  submit path), not raw diffs that include test edits.
- **R8.4** `py_compile` clean; key modules import; clustering smoke checks pass
  (greedy/connected agree on separable inputs; kernel orders identical<grouped<
  distinct; order-independence of connected).
- **R8.5 — Stage-level pipeline tests `[BLOCKER]`.** The pipeline must be testable
  *part by part*, not only end-to-end (the full run needs GPU/Docker). Each stage
  has a runnable check that uses mocks/fakes for the heavy dependencies (no vLLM,
  no Docker, no real NLI server): (a) SEARCH relevance scoring + saturation/step
  caps; (b) strategy proposal parsing + clustering + entropy/branch decision;
  (c) SDLG write-point trigger fires on real writes and NOT on echo/submit/stderr;
  (d) source-only patch capture + no-overwrite of a real submission; (e) matched-k
  resample driver wiring (k read from metadata, per-temperature dirs, none-arm
  selected); (f) the metric scripts (§4) on tiny synthetic predictions. Prefer a
  `tests/` suite (pytest) the loop can run; if absent, the loop should add it. This
  is how the loop "tests the pipeline at different parts" without launching runs.

---

## 9. Model diversity & ensembles — uncertainty over the model itself  `[TRACKED]`

A single model's pretraining + RLHF impose systematic biases (knowledge collapse —
`PDFs/Wright_2025_Epistemic_Diversity_Knowledge_Collapse.pdf`; reduced variety —
`PDFs/Zhang_2025_NoveltyBench.pdf`). Within-model sampling cannot escape those
biases; different model families have different epistemic blind spots. So model
choice is itself an uncertainty axis.

- **R9.1 — Model-swappable now `[BLOCKER]`.** No hardcoded model assumptions in
  the pipeline; the base model is selected by config / CLI end-to-end. Verify a
  different model id can be plugged without code edits (e.g. via
  `model.model_name` and the vLLM endpoint), and that intent/strategy/SDLG/relevance
  sub-calls all honor it.
- **R9.2 — Roadmap analysis (tracked, not required for v1 unless claimed):**
  (a) reproduce the mode-collapse finding on ≥2 distinct model families to show it
  is not Qwen-specific (strengthens the claim and aligns with the knowledge-collapse
  literature); (b) an *ensemble* arm that pools branches across models, measuring
  whether cross-model coverage / diversity exceeds any single model at matched
  budget. Both reuse the matched-budget, independent-metric machinery.
- **R9.3** If the paper makes ANY cross-model or ensemble claim, R9.2 is promoted
  to `[BLOCKER]` and must be run with the same rigor (matched budget, independent
  diversity metric, CIs).

---

## 10. Writeup scaffolding (need not be the final paper)  `[NON-BLOCKER but tracked]`

Last because it consumes everything above.

- **R10.1** A `RESULTS.md` / methods+results draft that states the design choices
  (matched-k, temperature sweep, scaffold match, independent diversity metric) and
  the threats-to-validity section.
- **R10.2** A results table/figure stub wired to the metric scripts.
- **R10.3** The uncertainty framing of §0.1 (epistemic vs aleatoric, the SEARCH
  step as epistemic-uncertainty reduction, semantic entropy as post-search
  uncertainty, diversity for set-valued solutions) is stated explicitly as the
  paper's conceptual spine and connected to the empirical design.

---

## DONE — evidence sufficiency bar

The loop may STOP and declare success only when ALL of the following hold and are
**stable across `--require-stable` consecutive iterations**:

1. Every `[BLOCKER]` rubric item (sections 1, 2, 3-claimed, 4, 5, 6, 7, 8 —
   including R8.5 stage tests — and R9.1 model-swappability) is `pass` with an
   evidence pointer.
2. The pipeline can, by a single documented command per arm, produce the
   predictions needed to compute the headline metrics — i.e. the *evidence is
   gatherable* (the loop need not execute the multi-hour GPU runs itself; it must
   prove they are wired, correct, and that the metric scripts consume their output).
   Stage-level tests (R8.5) pass, proving each part works in isolation.
3. `diverse-pass@k` (R4.1), the independent diversity metric (R4.2), and the
   uncertainty-stratification script (R5.2) are implemented and runnable over the
   predictions schema. Executed numbers may legitimately await the GPU runs; the
   *wiring and correctness* must be proven and any pending run listed in
   `next_actions`.
4. Threats to validity (R6.4) are enumerated and addressed/acknowledged, and the
   §0.1 uncertainty framing (R10.3/R5.1) is stated as the conceptual spine.
5. `blocking_gaps` is empty.

If any BLOCKER is `fail`/`partial`, the loop must record the gap, implement the
highest-value fix(es) this iteration, and continue.

**Anti-stall:** if an iteration makes zero `changes_made` and the gold standard is
still unmet, that is a *stuck* signal — the verdict must explain why (e.g.
"requires a GPU run I cannot launch here") and put the concrete human action in
`next_actions`. The wrapper treats repeated stalls as a stop condition.

---

## Out of scope / non-goals (do NOT do these to "pass")

- Do not launch the full multi-hour SWE-bench/vLLM runs from inside the loop
  unless explicitly enabled; wiring + smoke verification is sufficient for the
  readiness verdict.
- Do not fabricate, hardcode, or cherry-pick results to make a metric look good.
- Do not weaken a BLOCKER, delete a test, or narrow the instance set to force a
  pass. Propose spec changes openly instead.
- Do not expand scope into new research directions not in the proposal; fix and
  complete the claimed contribution.

---

## Verdict schema (the loop writes `review_loop/verdict_<NN>.json` matching this)

```json
{
  "iteration": 0,
  "timestamp": "ISO-8601",
  "summary": "one paragraph",
  "rubric": {
    "R1_faithfulness":         {"status": "pass|partial|fail", "evidence": "path:line or cmd", "notes": ""},
    "R2_arms_matched":         {"status": "...", "evidence": "", "notes": ""},
    "R3_ablations":            {"status": "...", "evidence": "", "notes": ""},
    "R4_metrics_independent":  {"status": "...", "evidence": "", "notes": ""},
    "R5_uncertainty_analysis": {"status": "...", "evidence": "", "notes": ""},
    "R6_statistical_rigor":    {"status": "...", "evidence": "", "notes": ""},
    "R7_reproducibility":      {"status": "...", "evidence": "", "notes": ""},
    "R8_internal_correctness": {"status": "...", "evidence": "", "notes": ""},
    "R9_model_diversity":      {"status": "...", "evidence": "", "notes": ""},
    "R10_writeup":             {"status": "...", "evidence": "", "notes": ""}
  },
  "issues_found": [
    {"title": "", "severity": "blocker|major|minor", "rubric_key": "R4_metrics_independent",
     "decision": "fix-now|defer|wontfix", "rationale": ""}
  ],
  "changes_made": [{"file": "", "what": "", "why": ""}],
  "evidence_gatherable": false,
  "gold_standard_met": false,
  "blocking_gaps": ["concrete gap 1"],
  "next_actions": ["concrete action (incl. human-only steps like GPU runs)"],
  "confidence": 0.0
}
```
