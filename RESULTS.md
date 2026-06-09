# Results & Methods Draft — Semantic-Entropy-Gated Diverse Agentic Code Generation

> **Status.** This is the publication scaffolding (rubric R10). The pipeline is
> wired and unit-/stage-tested (`tests/`, 33 passing); the **executed headline
> numbers await the GPU/vLLM + SWE-bench runs**, which cannot be launched from the
> review loop. Every results cell below is therefore a **command + a pending
> placeholder**, never a fabricated value. When a run completes, `scripts/compute_metrics.py`
> fills the table directly from the predictions artifacts.

---

## 1. The claim (conceptual spine — §0.1 of `GOLD_STANDARD.md`)

**Headline (a diversity claim, not a leaderboard claim).** At a *matched trajectory
budget and matched sampling temperature*, vanilla LLM resampling mode-collapses to a
few semantic forms, whereas semantic-entropy-gated branching explores meaningfully
distinct solutions and is therefore more likely to contain a passing fix. The
contribution is the **semantic-entropy-gated branching core**; the candidate
generator (strategy-proposal vs SDLG) is a pluggable, ablated input. **A rigorous
null/negative result is publishable** — if branching does not beat matched-k vanilla,
we report that with the same statistical care.

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
   but in a low-probability region it rarely samples (knowledge collapse — Wright 2025;
   reduced variety — NoveltyBench, Zhang 2025). Diversity deliberately explores *off
   the dominant mode* to reach it.
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
low entropy over a biased mode — which the entropy gate will **not** flag. That is why
the paper carries mechanisms that do **not** depend on the model's own confidence:
SDLG forces off-mode exploration by construction, and cross-model diversity (§9)
escapes a single model's bias. The arms are therefore *complementary, not redundant*.

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

### 2.3 Ablations

- **Generator:** strategy-proposal vs SDLG, isolated (`diversity_method` is a single
  mutually-exclusive switch — the two generators can never stack).
- **Clustering:** `greedy` (Farquhar/Kuhn Alg. 1) vs `connected` (order-independent
  transitive closure) vs `kernel` (Kernel Language Entropy, Nikitin 2024). τ is
  **not transferable** to `kernel` (different scale) and is recalibrated per strategy.
- **τ / entropy-gate sensitivity (R3.3):** sweep `--tau`; report branch-rate vs τ.
  Both arms now read the **same** `entropy_threshold` key *and act on it identically*
  — `strategy_proposal` collapses to the single dominant cluster when `entropy ≤ τ`,
  mirroring the SDLG arm's early return (see `_propose_strategies`,
  `phased_orchestrator.py`).

---

## 3. Metrics (independent and unbiased — R4)

All metrics are pure post-processing over the predictions/eval artifacts
(`src/evaluation/metrics.py`, no GPU/NLI/Docker):

- **Coverage — `diverse-pass@k`** via the **unbiased Chen et al. (2021)** estimator on
  *both* arms at matched k. Framed honestly as an **oracle upper bound**, not
  deployable accuracy.
- **Diversity measured INDEPENDENTLY of the branching signal (R4.2).** We do **not**
  reuse the DeBERTa-NLI clustering that *decided* branching (circular). Diversity of the
  **final patches** is structural: exact-signature **distinct-patch count** + graded
  **mean pairwise edit distance** (`difflib`) over normalized diff bodies. Measured on
  **final** patches, not proposal-time branches (which can converge downstream, R4.3).
- **Selection-aware accuracy (R4.4, strengthening):** a selector (majority cluster /
  regression tests) → `selected-pass@1`, so we do not overclaim the oracle number.
- **Uncertainty (R6.1):** every headline number carries a **bootstrap CI** over
  instances (`bootstrap_ci`, seeded for reproducibility).

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
# Treatment (strategy-proposal), T=0.7, greedy clustering:
python scripts/run_branching.py --config configs/branching.yaml \
    --results-dir results/strategy_t0.7 --clustering-strategy greedy
# Matched-k vanilla control at the same temperature:
python scripts/run_resample_baseline.py --treatment-dir results/strategy_t0.7 \
    --results-dir results/resample --temperatures 0.7
# Metrics + R5 analysis:
python scripts/compute_metrics.py \
    --predictions results/strategy_t0.7/predictions_all_trajectories.jsonl \
    --eval results/strategy_t0.7 --results-dir results/strategy_t0.7 \
    --compare-predictions results/resample_t0.7/predictions_all_trajectories.jsonl \
    --compare-eval results/resample_t0.7 \
    --out results/metrics_strategy_vs_vanilla_t0.7.json
```

| Metric (n=10 easy SymPy) | Matched-k vanilla | Strategy-proposal | SDLG |
|---|---|---|---|
| diverse-pass@k (Chen, oracle) ± CI | _pending_ | _pending_ | _pending_ |
| distinct final patches ± CI | _pending_ | _pending_ | _pending_ |
| mean pairwise patch distance ± CI | _pending_ | _pending_ | _pending_ |
| selected-pass@1 (majority cluster) | _pending_ | _pending_ | _pending_ |

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
4. **Small n.** n = 10 → wide CIs. *Addressed:* every number carries a bootstrap CI
   (R6.1); we do not over-read point estimates.
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
   a roadmap item.

---

## 7. Reproducibility (R7)

- One documented command per arm reproduces its predictions (§5); configs are checked in
  (`configs/branching.yaml`); NLI/vLLM/Docker prerequisites are in `CLAUDE.md`.
- **No silent data loss (R7.2):** every completed trajectory's patch is captured before
  container teardown (`_capture_patch_if_missing`, source-only, no-overwrite of a real
  submission) — covering the three completion paths that previously dropped diffs.
- Tables are regenerable from the predictions artifacts by `scripts/compute_metrics.py`
  (R7.3). *Known gap:* a checked-in **figure** script (plots from the same JSON) is a
  `next_actions` item.
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
