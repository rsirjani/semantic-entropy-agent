# Results & Methods Draft — Semantic-Entropy-Gated Diverse Agentic Code Generation

> **Status.** This is the publication scaffolding (rubric R10). The pipeline is
> wired and unit-/stage-tested (`tests/`, run `python -m pytest tests/ -q`); the **executed headline
> numbers await the GPU/vLLM + SWE-bench runs**, which cannot be launched from the
> review loop. Every results cell below is therefore a **command + a pending
> placeholder**, never a fabricated value. When a run completes, `scripts/compute_metrics.py`
> fills the table directly from the predictions artifacts.

---

## 1. The claim (conceptual spine — §0.1 of `GOLD_STANDARD.md`)

**Headline (a diversity claim, not a leaderboard claim).** At a *matched trajectory
budget and matched sampling temperature*, vanilla LLM resampling **concentrates on a
few semantic forms with diminishing returns** — not zero diversity (EntroPO's
baselines still improve with rollout count; SWE-agent App. B.5 shows pure resampling
pass@6 nearly doubles pass@1 on Lite, which is exactly why the matched-k control is
necessary) — whereas semantic-entropy-gated branching explores meaningfully distinct
solutions and is therefore more likely to contain a passing fix. The diverse-pass@k
number is reported as an **oracle/coverage row** (mirroring Tree of Thoughts' "+best
state" convention), never as deployable accuracy. The contribution is the
**semantic-entropy-gated branching core**; the candidate generator (strategy-proposal
vs SDLG) is a pluggable, ablated input. **A rigorous null/negative result is
publishable** — if branching does not beat matched-k vanilla, we report that with the
same statistical care.

**Why diversity helps — the unifying principle.** Greedy / low-temperature decoding
returns the *mode* of the model's solution distribution; diversity is valuable
precisely when the correct solution is **not that mode**. We do **not** reduce this
to a single mechanism (in particular, not to "the answer is a set"). The mismatch
between where the model concentrates probability mass and where a correct fix lies
arises for several distinct reasons, treated as a holistic family:

1. **Multiplicity (aleatoric).** The problem is genuinely multi-valued — the answer
   *is a set*; diversity covers it and pass@k rises mechanically with set size.
2. **Residual epistemic uncertainty (post-search).** A single right fix exists, but
   after SEARCH the model still spreads belief and the correct candidate is not the
   argmax. SEARCH exists to *reduce* this; what remains is what branching exploits.
3. **Distributional / mode-collapse bias — even when the answer is a single point.**
   Pretraining frequency bias and RLHF sharpening concentrate sampling on a few
   "typical"/safe modes; a *better, correct* trajectory can sit in the model's support
   but in a low-probability region it rarely samples. Diversity deliberately explores
   *off the dominant mode* to reach it. **Load-bearing citations:** NoveltyBench
   (Zhang et al. 2025 — <3 functionally distinct outputs per 10 samples at temperature
   1.0, their "best-case" setting; and the OLMo-2 staged analysis showing each
   alignment stage SFT→DPO→RLVR reduces diversity, biggest drop at DPO) and EntroPO
   (Prop. 3.4 — standard DPO preserves the reference policy's likelihood ratios,
   perpetuating "rare correct trajectories"; their less-diverse policies scale worse
   with rollout count on SWE-bench with our exact model). Wright et al. 2025 is cited
   for *concentrated* diversity (LLMs less diverse than web search; Qwen family
   stagnant) — NOT for "knowledge collapse is happening": their own conclusion is that
   models are not locked into narrow frames. Moore et al. 2024 is NOT cited for this
   mechanism (its temp-0 paraphrase-consistency protocol measures a different
   construct and superficially points the other way).
4. **Multi-step compounding.** An early agent commitment determines which region of
   solution space is even reachable; a single trajectory locks in and cannot recover.
   Trajectory-level diversity hedges against early lock-in — distinct from token-level.
5. **Model-space bias (§9).** Any one model is a single biased draw from "model space";
   cross-model / ensemble diversity reaches solutions no single model covers — the
   limiting case of (3).

**Semantic entropy as a proxy — and its blind spot.** High post-search semantic
entropy signals a *spread* distribution (cases 1–2), so the adaptive gate (branch iff
entropy `> τ`) spends diversity where the distribution is visibly multi-modal and
saves it where the mode is already decisive. But entropy measures *spread, not
correctness*: under case 3, mode collapse can make the model **confidently wrong** —
low entropy over a biased mode — which the entropy gate will **not** flag. This blind
spot is **documented in print**: Farquhar et al. 2024 (Nature, p. 629) explicitly
scope semantic entropy away from "situations in which LLMs are confidently wrong,"
and Tomov et al. 2026 (`PDFs/2511.04418v2.pdf`) prove consistency-based UQ tracks
epistemic error only when aleatoric uncertainty is zero. The same theory supplies our
defense: entropy is a *multiplicity* signal, so the correct response to high entropy
is **branching (exploration), not abstention** — and the τ-gate is claimed only for
the multiplicity/residual-uncertainty mechanisms (cases 1–2). That is why the paper
carries mechanisms that do **not** depend on the model's own confidence: SDLG forces
off-mode exploration by construction, and cross-model diversity (§9) escapes a single
model's bias. The arms are therefore *complementary, not redundant*.

**Falsifiable predictions (tested by §4, not assumed):**
- Branching's benefit over matched-k vanilla is **largest where the correct solution
  is off the model's mode** — partly predicted by post-search entropy (cases 1–2), but
  **also** on *low-entropy* instances where vanilla collapsed to a confident wrong mode
  and a diverse mechanism recovered the fix (case 3). The analysis looks *beyond
  entropy alone*.
- The benefit is ≈0 where greedy already sits on the correct mode — diversity then only
  costs compute, and the adaptive τ gate would have skipped branching there.

---

## 2. Experimental design

### 2.1 Arms (scaffold- and temperature-matched)

| Arm | `diversity_method` | Mechanism | Branching core |
|---|---|---|---|
| **Treatment: strategy-proposal** | `strategy_proposal` | LLM proposes K strategies → cluster → fork one trajectory per cluster | shared SEARCH→cluster→entropy→branch |
| **Treatment: SDLG** | `sdlg` | single trajectory; SDLG forks at the first *write* command via DeBERTa token attribution | shared SEARCH→cluster→entropy→branch |
| **Control: matched-k vanilla** | `none` | the **same phased agent**, branching disabled, run **k times per instance** at the same temperature | none (resample) |

The control is the **same scaffold** with branching disabled — *not* the legacy
`ReactAgent` (that would confound branching with a different harness). The matched-k
control reads `k = #trajectories the treatment produced` per instance from the
treatment metadata (`scripts/run_resample_baseline.py --treatment-dir …`).

### 2.2 Matched budget & temperature

One `sample_temperature` knob (`configs/branching.yaml`) drives **both** the proposer
and the vanilla baseline, so the arms differ only in the branching mechanism. The
headline sweeps temperature **0.2 / 0.7 / 1.0**; vanilla samples at **T > 0** (T = 0
would be a deterministic strawman). Each (arm × temperature × clustering-strategy)
writes a **separate results dir** so no run overwrites another's predictions.

**Disclosure — what the knob touches in each arm.** In the treatment arms the
temperature applies to the *diversity source* (the strategy proposer; the SDLG arm
perturbs tokens directly) while post-branch execution stays greedy; in the vanilla
arm the *whole agent* decodes at T (its only diversity source is base-agent
sampling). The knob is matched, but it injects randomness into different amounts of
text — that is inherent to comparing mechanism-driven vs sampling-driven diversity,
and it is why the sweep includes vanilla's most favorable temperature (see the
robustness row below) rather than trusting any single T.

**Matching direction (which budget match, for which claim).** Branched trajectories
share the SEARCH prefix (search runs once, then forks); vanilla resamples each pay
the full search cost. At matched *trajectory* count the control therefore receives
**at least as much** total compute as the treatment, so a treatment win at matched k
cannot be attributed to a compute advantage — trajectory-matching is the
*conservative* match for the headline coverage/diversity claim. A *token*-matched
comparison (fewer vanilla trajectories) would favor the treatment and is **not**
used. `scripts/budget_audit.py` reports per-arm token totals so the realized
asymmetry is quantified, not assumed.

**Pre-registered confirmatory family (multiple-comparison control).** The sweep ×
arms × ablations grid has many cells; exactly ONE comparison cell is confirmatory,
fixed before the GPU runs: **strategy-proposal (greedy clustering, τ=0 superset
run) vs matched-k vanilla at T = 0.7**. Within that cell the headline claim has two
halves, tested as a **fixed-sequence (hierarchical, gatekeeping) family** that
mirrors the causal chain and controls family-wise error at α = 0.05 without
splitting alpha:

- **H1 — diversity (the title claim, mode collapse):** per-instance rarefied
  distinct-patch gain at matched k\* (treatment − vanilla), exact paired sign-flip
  test. If H1 is not significant, the mode-collapse premise is not confirmed and
  H2 is reported as descriptive only.
- **H2 — coverage (tested only if H1 rejects):** per-instance matched-k\*
  `diverse-pass@k` gain, exact paired sign-flip test (n=10 → all 1024 sign
  patterns).

The order is fixed by the science, not by the data: branching can only raise
coverage *through* producing distinct solutions, so confirming coverage without
confirming diversity would be uninterpretable. The hierarchy makes the coverage
claim *strictly harder* than under a single-endpoint design (it now needs its own
p < 0.05 **and** H1 upstream), while giving the diversity claim — which §1 says is
the headline — a confirmatory test it previously lacked. The gate is encoded **in
the artifact itself**, not only in this prose: the metrics JSON's
`comparison.confirmatory_family` block derives H2's status (confirmatory vs
descriptive) from H1's exact sign-flip p at α=0.05, so neither the campaign
analyst nor a reader filling §5 can mistake an H2 p < 0.05 for a confirmatory
result when H1 did not reject (the block also states it applies as confirmatory
only in this pre-registered cell).

**Power disclosure (decided before the runs).** The exact sign-flip p-value has a
hard floor set by ties: with z zero gains among n instances, p ≥ 2^(1+z−n). At
n = 10, **p < 0.05 requires at least 6 instances with a nonzero, same-direction
difference**; e.g. a "treatment wins on 4, ties on 6" outcome bottoms out at
p = 0.125 *no matter how clean the wins are*. `compute_metrics.py` reports this
floor (`min_achievable_p`) next to every p-value so an insignificant result is
read correctly: it may reflect ties/power, not evidence of no effect. The 0/1
coverage gain is expected to produce many ties on easy instances; the
near-continuous H1 diversity gain is not — another reason H1 leads the sequence.

T = 0.7 sits below the T≈0.9 knee where whole-agent decoding precision
degrades (EntroPO Fig. 4), so the primary cannot manufacture a win out of vanilla
decoding degradation at T = 1.0. All other cells (T = 0.2/1.0, SDLG arm, clustering
and τ ablations, entropy strata) are **exploratory/descriptive**. One required
robustness row: treatment (T = 0.7) vs vanilla at *its best* sweep temperature — if
the headline gain survives only against vanilla's worst temperature, that is
reported, not hidden.

**Adaptive execution of the exploratory cells (disclosed).** The runs are executed
by an autonomous campaign driver (`scripts/run_campaign.py`): the confirmatory cell
always runs **first, exactly once**, with the deterministic command sequence above;
which *exploratory* cells run afterwards — and when the campaign stops — is chosen
adaptively by an LLM analyst reading the completed cells' metric outputs, from a
fixed pre-declared menu (SDLG arm, T = 0.2/1.0, clustering variants, one repeat
draw). Three consequences are pinned in advance: (i) the confirmatory dataset is
the **first** completed Phase A run — the repeat draw (`strategy_t0.7_seed2`)
estimates sampling variance and can never replace, pool into, or re-litigate the
primary; (ii) adaptivity cannot affect any confirmatory number, only which
exploratory/descriptive cells exist to report — the set of exploratory cells in
the paper is therefore data-dependent and is reported as such; (iii) every
decision is a checked-in artifact (`campaign_decisions/decision_*.json`) with the
analyst's written rationale, so the selection path is auditable; (iv) the analyst
window is integrity-guarded on **both planes** — git porcelain for code/config,
and a content-hash fingerprint of all results/decision artifacts taken before and
verified after every analyst invocation — so adaptivity can *read* measured data
and order future cells, but any modification of existing artifacts (metrics,
predictions, eval records, or prior decision files) stops the campaign loudly.

### 2.3 Ablations

- **Generator:** strategy-proposal vs SDLG, isolated (`diversity_method` is a single
  mutually-exclusive switch — the two generators can never stack).
- **Clustering:** `greedy` (Farquhar/Kuhn Alg. 1) vs `connected` (order-independent
  transitive closure) vs `kernel` (Kernel Language Entropy, Nikitin 2024). τ is
  **not transferable** to `kernel` (different scale) and is recalibrated per strategy.
- **τ / entropy-gate sensitivity (R3.3):** computed **post-hoc, at zero extra GPU
  cost**, by `scripts/tau_sweep.py`. The τ=0 headline run produces the *superset* of
  trajectories any τ>0 run would produce; the gate's no-branch action keeps exactly
  the dominant-cluster representative, which already exists in the superset run
  (trajectory *i* executes cluster *i*'s representative; post-branch execution is
  greedy), so every τ is evaluable by trajectory subsetting. Both arms read the
  **same** `entropy_threshold` key *and act on it identically* — `strategy_proposal`
  collapses to the single dominant cluster when `entropy ≤ τ`, mirroring the SDLG
  arm's early return (see `_propose_strategies`, `phased_orchestrator.py`).
  **Stated plainly:** with N=5 candidates, discrete semantic entropy is
  *partition-quantized* — it takes exactly 7 values {0, 0.500, 0.673, 0.950, 1.055,
  1.332, 1.609} nats — so τ at this N is a **cluster-partition-shape rule**, not a
  continuous dial (τ=0 ≡ "branch iff ≥2 clusters"). The sweep grid is exactly the
  achievable set; the paper says this rather than implying a smooth threshold.
  Two precision/validity details the sweep enforces: (i) entropies are recomputed
  **exactly from the logged cluster partition** (the rounded `Entropy:` log line
  can cross a gate boundary — e.g. partition (2,2,1) is 1.054920…, which a
  3-decimal log rounds to 1.055 > 1.0549), falling back to the logged value, with
  the source flagged, when the two disagree (kernel runs); (ii) the **realized**
  candidate count N is reported per instance — the proposer can return fewer than
  the configured 5 strategies — and instances whose N deviates from the modal N
  are flagged (`non_modal_n_instances`) because their entropies sit on a
  different quantization grid and must not be pooled silently.

### 2.4 Documented deviations from the reference methods (R1.1)

These are deliberate, disclosed deviations — not bugs:

1. **SDLG vocabulary bridge (Aichberger 2025 App. D).** The paper's setup relied on
   the generator (OPT) and the NLI model (DeBERTa) sharing a vocabulary; Qwen3's
   ~151k BPE and DeBERTa's vocabulary do not align. We unify at the **text level**:
   (i) substitution candidates are proposed and scored (attribution `A_i`,
   substitution `S_ij`) in DeBERTa's embedding space server-side; (ii) the importance
   term `I_ij = p_LLM(v_j | y_<i)` is computed under the **generator's own
   tokenization** by converting each substitute to its surface string and matching it
   against the generator's top-k next-token strings, with an exact echo-scored
   prompt-logprob query for substitutes outside the top-k
   (`src/diversity/sdlg.py::_get_importance_scores`, unit-tested in
   `tests/test_sdlg_importance.py`); (iii) the chosen substitute is spliced into the
   reasoning text as a string (first occurrence of the original token's surface form)
   and the generator re-tokenizes and completes from the splice point. No bilingual
   embedding mapping is used.
2. **Score combination.** Candidates are ranked by the arithmetic mean
   `(A_i + S_ij + I_ij)/3` rather than the paper's product form: the mean keeps a
   candidate rankable on attribution+substitution when the generator assigns it
   negligible mass, where a product would zero the score.
3. **Importance conditioning.** `I_ij` conditions on the generated text prefix only
   (not the full re-encoded conversation context) — matching the splice used at
   generation time.
4. **SDLG trigger scope.** The proposal (p. 2) applies diverse generation "at each
   agent step"; for compute reasons SDLG fires at the **first write command** (the
   first state-changing action, `phases.is_write_command`), where the trajectory
   first commits to an implementation. Branching frequency is therefore a per-run
   statistic, not per-step.
5. **Instance set.** Deviates from proposal Appendix C — see §6, threat 8.

---

## 3. Metrics (independent and unbiased — R4)

All metrics are pure post-processing over the predictions/eval artifacts
(`src/evaluation/metrics.py`, no GPU/NLI/Docker):

- **Eval-record completeness (the metric inputs' contract).** The per-arm
  `trajectory_eval_<iid>.json` files are produced by
  `scripts/eval_all_trajectories.py`, which (i) writes into the **arm's own
  results dir** (`--results-dir`), so evaluating the control can never
  overwrite the treatment's eval files; (ii) emits **exactly one row per
  genuine trajectory** — duplicate patches are evaluated once for compute but
  every duplicate inherits its representative's outcome (identical patches
  resolve identically; marked `deduped_from`), and **empty patches count as
  `resolved: false` draws** without a Docker run. This is load-bearing for the
  Chen estimator: its (n, c) must count what the arm actually *produced*. The
  vanilla arm's duplicate patches ARE the mode-collapse signal under study —
  an eval record that deduplicated them would deflate vanilla's k, shrink the
  matched k\*, and silently subsample the treatment's coverage while the
  vanilla arm kept plain any-pass. Re-runs are safe end-to-end: the eval
  driver and the metric loaders both keep the **last** occurrence per
  (instance, trajectory), and entropy/partition parsers read the **last**
  `STRATEGY PROPOSAL` block of the append-mode decisions log. Two
  eval-outcome guarantees (iteration-11): (iii) **every `resolved` is a
  genuine harness verdict** — the SWE-bench harness swallows all per-instance
  errors (no report written, run continues), so a missing report is
  classified from the harness logs: patch-apply failure and test timeout are
  patch-attributable failed draws (recorded with `fail_reason`), while any
  other cause (Docker/build/container flake) aborts the eval step **without
  writing the record** (exit 3 → the campaign retries once, then stops
  loudly) — an eval-time flake scored as a failure would silently corrupt the
  Chen (n, c) and the resume marker would freeze it forever; (iv)
  **stale-report immunity** — the harness reuses an existing `report.json`
  keyed by (run_id, model, instance) *without re-evaluating*, so the eval
  run id embeds a SHA-1 of the patch content (`patch_run_id`): a re-run with
  a changed patch gets a fresh verdict, an identical patch legitimately
  reuses its cached report (making post-stop retries cheap). The comparison
  additionally reports `pred_eval_count_mismatch` — instances where the
  predictions file and eval record disagree on the draw count (desynced
  artifacts) — instead of silently `min()`-ing inconsistent denominators.
- **Predictions-record completeness (the contract one layer up).** The eval
  record is built *from* `predictions_all_trajectories.jsonl`, so completeness
  must hold there too: the branching driver writes **one prediction row per
  genuine trajectory — including failed and patchless draws as empty-patch
  rows** (`run_branching.build_predictions`,
  `phased_orchestrator.collect_patch_entries`), exactly as the resample driver
  writes an empty row for every unproductive resample. Before this fix the
  treatment driver silently dropped trajectories that failed or produced no
  diff (on the real pilot, 5 of 10 instances had fewer prediction rows than
  trajectories), which would have deflated the treatment's metric-time k —
  e.g. a treatment with 1 pass among 5 draws but only 2 recorded rows scores
  pass@k\*(2,1)=1.0 against vanilla's pass@2(5,1)=0.4, a +0.6 "gain"
  manufactured entirely by dropping the treatment's own duds — and made the
  `nonempty_patch_fraction` diagnostic structurally ≈1.0 for the treatment arm
  (its empty draws never reached the artifact). Both arms now count
  unproductive draws identically; run-loop failures are failed draws in both.
  The metric loaders are run-batch aware: a branching **re-run that produces
  fewer trajectories** cannot leave the prior run's orphan rows in the
  diversity pool (`load_predictions_by_tid` scores only the last
  primary-delimited batch, mirroring the eval driver), and the resample driver
  warns when a treatment `metadata.json`'s patch entries disagree with
  `total_trajectories` (old-driver or interrupted artifact).
- **Draw accounting starts at the fork decision (iteration-7 extension).** The
  contract above covered trajectories that *exist* in the run record; the fork
  paths could still lose draws **at creation**. (i) A strategy or SDLG fork
  whose container/clone/injection fails is registered as a **failed
  empty-patch draw** (`phased_orchestrator._register_failed_draw`) — exactly
  as the resample driver records a crashed resample — instead of being
  silently skipped, which would deflate the treatment's metric-time k (the
  same pro-treatment direction as the predictions-record fix, one layer
  earlier). (ii) An SDLG alternative that **submits during injection** raises
  `Submitted` inside the clone step; previously the blanket clone-failure
  handler swallowed it, discarding a completed — possibly passing — child and
  leaking its container. It is now recorded as a completed, submitted draw
  with its patch kept (`_inject_alternative`). (iii) The **resample**
  all-trajectories file has no batch delimiters (no "primary" rows), so the
  parsers' last-batch rule cannot isolate a re-run there; the driver therefore
  **replaces the instance's rows on re-run** (`replace_instance_rows`,
  mirroring its primary file) so a smaller-k re-run cannot leave stale surplus
  resamples in the vanilla arm's k and diversity pool. All three are
  stage-tested (`tests/test_end_to_end_mocked.py`), alongside a mocked
  end-to-end chain test from synthetic trajectories through
  `collect_patch_entries → build_predictions → eval loaders →
  compute_metrics.compare`.

- **Coverage — `diverse-pass@k`** via the **unbiased Chen et al. (2021)** estimator on
  *both* arms at matched k. Framed honestly as an **oracle upper bound**, not
  deployable accuracy. **Matched k is enforced at metric time, not only at run
  time:** `compare()` evaluates both arms at the common per-instance
  k\* = min(k_treatment, k_vanilla) via the Chen estimator and reports any
  k-mismatched instances — a failed resample or `--max-k` cap can therefore never
  silently hand the larger arm a mechanical any-pass advantage.
- **Diversity measured INDEPENDENTLY of the branching signal (R4.2).** We do **not**
  reuse the DeBERTa-NLI clustering that *decided* branching (circular). Diversity of the
  **final patches** is structural: exact-signature **distinct-patch count** + graded
  **mean pairwise edit distance** (`difflib`) over normalized diff bodies. Measured on
  **final** patches, not proposal-time branches (which can converge downstream, R4.3).
  Cross-arm distinct-count differences at unequal k use the **rarefaction estimator**
  `expected_distinct_at_k` (expected #distinct in a random k\*-subset — the same
  hypergeometric identity as Chen) because raw distinct counts rise mechanically with
  sample size; mean pairwise distance needs no correction (expected subset mean =
  full mean, by pair-inclusion symmetry). **Two known properties of the H1 endpoint,
  disclosed before the runs:** (i) *granularity* — the exact signature counts
  lexical variants (e.g. a renamed variable) as distinct in BOTH arms; since the
  whole-agent-sampling vanilla arm is, if anything, the noisier producer of trivial
  variants, this inflates the *control's* distinct count more and biases H1 toward
  the null (conservative for the diversity claim; the graded pairwise distance and
  the roadmap behavioral metric complement it). (ii) *productivity confound* — an
  empty patch lowers the rarefied distinct count exactly like a duplicate, so an H1
  win could in principle reflect a patch-*production*-rate gap rather than
  diversity; `compute_metrics.py` therefore reports each arm's
  `nonempty_patch_fraction` and a **descriptive robustness row**
  (`rarefied_distinct_gain_nonempty`: the same rarefied gain over non-empty patches
  only, at k\*_ne = min non-empty count). If the confirmatory H1 rejects but the
  non-empty-only row shows ≈0 gain alongside a large production-rate gap, the paper
  reports the win as productivity, not mode-collapse escape. This row is fixed now,
  pre-data, and does not alter the confirmatory endpoint.
- **Selection-aware accuracy (R4.4) — implemented:** `selected-pass@1` under the
  **majority normalized-patch-signature selector** (self-consistency over final
  patches; ties → earliest seen; empty patches never win; all-empty counts as a
  miss). Deployable by construction — it reads only the predictions artifacts, no
  hidden tests, no NLI — so the oracle row is never presented alone
  (`scripts/compute_metrics.py::selected_pass_at_1`). **Known asymmetry,
  disclosed:** on the branching arm the trajectories are one-per-semantic-cluster
  *by construction*, so final patches are typically all-distinct and "majority"
  degenerates to the earliest-seen tie-break — closer to first-trajectory-pass@1
  than to true self-consistency; the vanilla arm's resamples carry real
  multiplicity. The script reports `degenerate_tiebreak` per instance and
  `n_degenerate_tiebreak_instances` per arm so the two selected-pass@1 numbers are
  read in that light, and the NLI-side alternative (keep only the dominant
  cluster's trajectory) is already reported as the largest-τ row of the τ sweep —
  no extra selector is invented post-hoc.
- **Uncertainty (R6.1):** every headline number carries a **bootstrap CI** over
  instances (`bootstrap_ci`, seeded for reproducibility), and the primary endpoint
  additionally carries the **exact paired sign-flip p-value**
  (`paired_permutation_pvalue`: all 2^n sign patterns at n ≤ 20) — at n = 10 the
  exact test is the trustworthy inference; the percentile bootstrap over lumpy 0/1
  gains is reported as a companion interval, not the decision rule.

---

## 4. Diversity-benefit analysis (R5) — uncertainty **and** mode-collapse

A pooled average hides the mechanism. `scripts/compute_metrics.py` decomposes the
benefit (treatment − matched-k vanilla):

- **R5.2 Stratified benefit (cases 1–2).** Bucket instances by *post-search semantic
  entropy* (extracted from `phased_decisions.log` `Entropy:` for the strategy arm, or
  `branching_log.json` `entropy` for SDLG) and report the `diverse-pass@k` gain per
  stratum. *Hypothesis to test:* part of the gain concentrates in high-entropy strata.
- **R5.3 Set-valued evidence (case 1).** Report instances where ≥2 structurally
  distinct patches both pass the hidden tests — direct evidence the solution is a *set*.
- **R5.4 Off-mode recovery (case 3) — the mode-collapse signature.** Flag instances
  where a diverse mechanism produced a *passing* fix that matched-k vanilla did NOT,
  **while post-search entropy was LOW**. These are exactly the cases the entropy gate
  cannot predict — direct evidence diversity helps by escaping training bias, not only
  by covering a set. The low-entropy flag uses the **same** split as R5.2 (consistency
  fix this iteration). Reported separately for SDLG vs strategy-proposal.
- **R5.5 Honest counter-analysis.** Show diversity does NOT help (and wastes budget)
  on low-uncertainty/unbiased instances where greedy already sits on the correct mode,
  and that the adaptive τ gate would have skipped branching there.

---

## 5. Headline results table (wired; numbers pending GPU runs)

Reproduce per arm, then compute. **No numbers are filled in until the runs execute** —
do not infer values from these placeholders.

```bash
# Treatment (strategy-proposal), T=0.7 (the pre-registered primary T — passed
# EXPLICITLY so the proposer temperature provably matches the vanilla arm, R2.4).
# tau=0 is pinned EXPLICITLY for the same reason: the confirmatory cell is
# defined by (T, tau), and the tau ablation's "superset run" premise (§2.3)
# must not ride on a config default an edit could silently change:
python scripts/run_branching.py --config configs/branching.yaml \
    --results-dir results/strategy_t0.7 --clustering-strategy greedy \
    --temperature 0.7 --entropy-threshold 0
# Matched-k vanilla control at the SAME temperature:
python scripts/run_resample_baseline.py --treatment-dir results/strategy_t0.7 \
    --results-dir results/resample --temperatures 0.7
# Evaluate EVERY trajectory of each arm into the arm's OWN dir (one row per
# genuine trajectory: duplicates propagate, empty patches count as failures):
for IID in $(jq -r .instance_id results/strategy_t0.7/predictions.jsonl); do
  python scripts/eval_all_trajectories.py --results-dir results/strategy_t0.7 --instance $IID
  python scripts/eval_all_trajectories.py --results-dir results/resample_t0.7 --instance $IID
done
# Metrics + R5 analysis:
python scripts/compute_metrics.py \
    --predictions results/strategy_t0.7/predictions_all_trajectories.jsonl \
    --eval results/strategy_t0.7 --results-dir results/strategy_t0.7 \
    --compare-predictions results/resample_t0.7/predictions_all_trajectories.jsonl \
    --compare-eval results/resample_t0.7 \
    --out results/metrics_strategy_vs_vanilla_t0.7.json
# Budget-fairness + per-arm token/compute accounting (R6.3) — BOTH arms; the
# audit auto-detects each arm's layout (treatment: <iid>/metadata.json;
# control: <iid>/run<idx>/<iid>/metadata.json with tid = "run<idx>"):
python scripts/budget_audit.py --results-dir results/strategy_t0.7 \
    --eval results/strategy_t0.7 --reference-cap 250 \
    --out results/budget_audit_strategy_t0.7.json
python scripts/budget_audit.py --results-dir results/resample_t0.7 \
    --eval results/resample_t0.7 --reference-cap 250 \
    --out results/budget_audit_resample_t0.7.json
# Post-hoc tau sweep (R3.3/R5.5) — zero extra GPU runs, from the same artifacts:
python scripts/tau_sweep.py --results-dir results/strategy_t0.7 \
    --eval results/strategy_t0.7 --out results/tau_sweep_strategy_t0.7.json
```

Per-arm token totals are summed from each trajectory's stored litellm response
(`extra.response.usage` in the `.traj.json` transcripts) — no run-loop
instrumentation is needed, so `budget_audit.py` reports `tokens_arm_total`,
`tokens_per_trajectory`, and `tokens_passing_trajectories` directly from the
artifacts. On the existing `results/branching` run this already yields ~12.9 M total
tokens over 53 trajectories with **0** passing branches above the 250-step baseline
cap (max passing = 164 steps), so the step-limit asymmetry did not manufacture wins.
Cost in $ is omitted only because the local vLLM model is unregistered for litellm
cost calculation; tokens and steps are the compute proxies. **Known undercount,
disclosed:** the treatment arm's strategy-proposer call, intent-extraction
sub-calls, and DeBERTa-NLI forward passes are not stored in the per-trajectory
`.traj.json` transcripts and are excluded from these sums — the exclusion is
bounded (one proposer call and O(N²)=10 NLI pair passes of a 0.4B model per
instance, vs k full agent trajectories) and works *against* the fairness claim's
margin rather than for it, since the structural argument (vanilla pays k full
SEARCHes, the treatment one shared SEARCH) rests on the trajectory sums, which
dominate by orders of magnitude.

| Metric (n=10 easy SymPy) | Matched-k vanilla | Strategy-proposal | SDLG |
|---|---|---|---|
| diverse-pass@k\* (Chen, oracle) ± CI | _pending_ | _pending_ | _pending_ |
| distinct final patches (rarefied @k\*) ± CI | _pending_ | _pending_ | _pending_ |
| mean pairwise patch distance ± CI | _pending_ | _pending_ | _pending_ |
| selected-pass@1 (majority signature; degenerate-tiebreak count disclosed) | _pending_ | _pending_ | _pending_ |
| H1 (confirmatory): rarefied distinct gain @k\*, sign-flip p + min-achievable p (T=0.7) | — | _pending_ | _exploratory_ |
| H2 (confirmatory iff H1 rejects): diverse-pass@k\* gain, sign-flip p + min-achievable p (T=0.7) | — | _pending_ | _exploratory_ |

Gain (treatment − vanilla) with bootstrap CI, and the per-entropy-stratum / off-mode
breakdown, are emitted by the same command into the `comparison` block of the output JSON.

---

## 6. Threats to validity (R6.4) — enumerated and addressed/acknowledged

1. **Cherry-picked difficulty band.** The 10 instances are all "<15 min" SymPy
   (proposal Appendix C). Coverage gains on easy instances may not transfer to harder
   ones. *Addressed:* scope claims are restricted to this band (R6.2); the roadmap is
   full SWE-bench Lite, and the same machinery runs unchanged at larger n.
2. **Single repository.** All instances are SymPy; repo-specific structure could inflate
   strategy diversity. *Acknowledged:* a cross-repo replication is listed in
   `next_actions`.
3. **Oracle selection.** `diverse-pass@k` is an upper bound (any branch passing counts),
   not deployable accuracy. *Addressed:* framed explicitly as oracle/coverage, and the
   selection-aware `selected-pass@1` (R4.4) is reported alongside so we never present the
   oracle number as accuracy.
4. **Small n / power floor.** n = 10 → wide CIs, and the exact sign-flip test has a
   hard tie-imposed floor p ≥ 2^(1+z−n): with z zero gains, p < 0.05 needs ≥6
   same-direction nonzero gains. On easy instances the 0/1 coverage gain (H2) will
   often tie, so an insignificant H2 is *expected under low power*, not evidence of
   no effect. *Addressed:* every number carries a bootstrap CI (R6.1);
   `min_achievable_p` is reported next to every sign-flip p; the better-powered,
   near-continuous diversity gain (H1) leads the confirmatory sequence; we do not
   over-read point estimates.
5. **Budget-fairness / step-limit asymmetry.** `max_search_steps` (240) sits below the
   per-trajectory `step_limit` (300), and lazy strategy trajectories reset to step 0, so
   they can in principle use more patch steps than the baseline's 250 total. *Addressed:*
   per-trajectory step distributions for passing branches and per-arm token/compute are
   reported (R6.3) to show wins are not manufactured by the asymmetry; in practice
   trajectories submit far earlier than the cap.
6. **Entropy-gate blind spot.** The adaptive gate cannot flag confidently-wrong
   low-entropy modes (case 3, §1). *Addressed:* SDLG and cross-model diversity do not
   depend on the model's confidence; R5.4 explicitly measures off-mode recovery the gate
   would miss.
7. **Independent-diversity proxy.** Structural (AST-free, line-level) diversity may over-
   or under-count behaviorally-equivalent patches. *Acknowledged:* it is deliberately
   **not** the branching NLI (avoids circularity, R4.2); a behavioral-diversity check is
   a roadmap item. Two sub-risks are measured rather than assumed (§3): the exact
   signature's lexical granularity (bias direction: toward the null — the
   whole-agent-sampling control produces trivial variants at least as readily), and
   the productivity confound (an H1 win that is really a patch-production-rate gap
   is exposed by `nonempty_patch_fraction` + the non-empty-only robustness row).
8. **Instance-set deviation from the proposal (disclosure).** Proposal Appendix C
   committed to a *difficulty-spanning* SymPy set (sympy-12481 plus nine instances
   rated <15 min up to >4 hr) to test whether branching helps more on harder
   problems. The set actually run is **all-easy (<15 min)** and shares only
   sympy-12481 with Appendix C. *Justification:* pilot runs showed the local 4-bit
   30B model has a ~0% base resolve rate on the 1 hr+ band, yielding no
   treatment-vs-control contrast at our compute budget; the easy band gives nonzero
   base rates where a coverage difference is measurable. *Consequence:* the
   proposal's "helps more on harder problems" hypothesis is **untested** and is
   replaced by the entropy-stratification analysis (§4) within the easy band; all
   claims are scoped accordingly (threat 1).
9. **Entropy quantization & estimator bias at N=5.** With 5 candidates the discrete
   semantic entropy takes only 7 partition-quantized values, the plug-in estimator is
   biased low (Miller–Madow ≈ (K−1)/2N nats, up to ~0.4 nats at K=5), and the τ gate
   is effectively a cluster-partition-shape rule. *Addressed:* the gate and strata
   are defined on the plug-in value with N **fixed by config** (n_strategies =
   sdlg_n_alternatives = 5) across arms and instances, so within-experiment
   comparisons are consistent; absolute entropy values are never interpreted across
   different N; `scripts/tau_sweep.py` reports the achievable-τ grid explicitly.
   *Residual risk, measured not assumed:* the proposer can under-deliver (<5
   parsed strategies), silently putting that instance on a different quantization
   grid — the τ sweep therefore reports the **realized** N per instance and flags
   `non_modal_n_instances`; any such instance is excluded from τ/strata pooling and
   disclosed.
10. **Strata and off-mode candidates are descriptive at n=10.** A median split
    leaves ~5 instances per entropy stratum, and a single "treatment passed, vanilla
    0/k" instance can be sampling noise (P(0 of k) = (1−p)^k). *Addressed:* only the
    pre-registered primary endpoint is confirmatory (exact sign-flip test); strata
    (R5.2) and off-mode records (R5.4) are reported as descriptive evidence with both
    arms' (k, n_resolved) attached, and off-mode claims require replication across
    temperatures before being asserted.
11. **Gate-signal saturation.** In the pilot run every instance's five strategy
   proposals clustered **all-singleton** (entropy = ln 5 ≈ 1.609, the maximum of
   the achievable grid), so the τ gate was degenerate there: any τ < ln 5 means
   always-branch, and the entropy signal carries information only if the NLI
   clustering actually merges some intent summaries at the configured
   entailment threshold (**0.7**, `configs/branching.yaml` — raised pre-pilot
   from 0.5 to prevent over-merging, which cuts the other way: a higher
   threshold merges *less*, making all-singleton saturation *more* likely;
   the no-retune rule below applies to this knob too). *Measured, not
   assumed:* the τ sweep reports
   every instance's realized partition and the achievable grid, so saturation
   is visible in the artifacts. If the pre-registered runs reproduce it, the
   paper reports the gate as uninformative at this substrate/threshold — a
   negative finding about the adaptive gate, not a license to retune τ or the
   entailment threshold post-hoc (any retuned configuration is a new,
   exploratory cell).
12. **Uncontrolled rival baselines.** (a) *In-context regeneration:* NoveltyBench
   (Fig. 5) shows prompting "give me a different answer" with prior answers in
   context recovers much diversity in open-ended NL. We do not run this arm;
   defenses: our setting is multi-turn and action-constrained, the strategy-proposal
   arm *is* a structured version of this idea inside the agent loop, and EntroPO
   (Fig. 4) shows unprincipled sampling randomness degrades SWE precision past
   T≈0.9. (b) *Sequential revision:* Snell et al. show sequential self-revision can
   beat parallel sampling at matched budget on MATH; we do not run a revision arm
   because their revision model required fine-tuning ("simply prompting existing
   LLMs to correct their own mistakes tends to be largely ineffective"), which is
   out of scope for an inference-time method. Both are stated as future-work arms,
   not silently omitted.

---

## 7. Reproducibility (R7)

- One documented command per arm reproduces its predictions (§5); configs are checked in
  (`configs/branching.yaml`); NLI/vLLM/Docker prerequisites are in `CLAUDE.md`.
- **No silent data loss (R7.2):** every completed trajectory's patch is captured before
  container teardown (`_capture_patch_if_missing`, source-only, no-overwrite of a real
  submission) — covering the three completion paths that previously dropped diffs.
- Tables are regenerable from the predictions artifacts by `scripts/compute_metrics.py`,
  and figures by `scripts/make_figures.py` from the same JSON (R7.3; rendering
  covered by `tests/test_budget_and_figures.py`). The cross-arm diversity figure
  is the **rarefied distinct @k\*** chart (per-arm levels + CIs at the common k\*,
  annotated with the H1 sign-flip p and the H2 gate status); the per-arm raw
  distinct-patch bars are own-k descriptives and are labeled as such (comparing
  raw distinct counts across arms is the mechanical sample-size bias R4.2 forbids).
- Determinism knobs (model id, temperature, bootstrap seed, package versions) are
  recorded with the results; the base model is config-selected end-to-end (R9.1), so a
  different model family can be plugged with no code edits.

---

## 8. Model diversity & ensembles (§9)

The base model is selected by config/CLI end-to-end (`model.model_name`), and the
intent/strategy/SDLG/relevance sub-calls all honor it (R9.1, verified by
`tests/test_pipeline_stages.py::test_model_name_is_config_driven_end_to_end`). The
roadmap (R9.2, tracked — not claimed in v1): (a) reproduce the mode-collapse finding on
≥2 model families to show it is not Qwen-specific; (b) an ensemble arm pooling branches
across models at matched budget. If the paper makes any cross-model claim, R9.2 is
promoted to a blocker and run with the same rigor (matched budget, independent diversity
metric, CIs).
