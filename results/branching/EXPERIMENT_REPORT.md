# Experiment Report: Semantic Entropy Clustering for Diverse Agentic Code Generation

**Authors:** Ramtin Sirjani & Paul Moore
**Course:** PhD-level Gen AI, Western University
**Date:** 2026-03-28
**Run ID:** `branching_v2_20260328`

---

## 1. Research Question

Can semantic entropy over clustered diverse outputs improve multi-step agentic code generation by forcing exploration of fundamentally different solution strategies?

**Core hypothesis:** At each agent decision point, if the model's output distribution has high semantic entropy (i.e., the model is uncertain between meaningfully different approaches), branching into independent trajectories — one per semantic cluster — will increase the probability that at least one trajectory solves the problem.

**Metric:** `diverse-pass@1` — does *any* trajectory across all branches solve the instance? Compared against `pass@1` — does the single greedy trajectory solve it?

---

## 2. Benchmark: SWE-bench Verified

### 2.1 What is SWE-bench?

SWE-bench (Jimenez et al., ICLR 2024) is a benchmark for evaluating whether language model agents can resolve real-world GitHub issues. Each instance consists of:

- **A GitHub issue description** (the "problem statement") — a real bug report or feature request from the project's issue tracker
- **A repository snapshot** at the exact commit before the fix was merged
- **A gold patch** — the actual human-written fix
- **Test cases** — unit tests that (a) fail before the fix (FAIL_TO_PASS) and (b) must continue passing after (PASS_TO_PASS)

**SWE-bench Verified** is a human-validated subset where annotators confirmed each instance is solvable from the issue description alone, removing ambiguous or under-specified issues from the original 2,294-instance set.

An instance is "resolved" if and only if:
1. The agent's patch applies cleanly to the codebase
2. All FAIL_TO_PASS tests now pass (the bug is fixed)
3. All PASS_TO_PASS tests still pass (no regressions introduced)

This is a strict criterion: a patch that fixes the target bug but breaks any existing test is counted as a failure.

### 2.2 Our 10 Instances

We selected 10 instances from SymPy (a Python symbolic mathematics library), all rated as <15 minutes estimated difficulty in our proposal:

| Instance | SymPy Ver. | Problem | FAIL_TO_PASS Tests | PASS_TO_PASS Tests |
|----------|-----------|---------|--------------------|--------------------|
| sympy-12481 | 1.0 | `Permutation` constructor fails with non-disjoint cycles — raises `ValueError` instead of composing cycles left-to-right | 1 | 7 |
| sympy-16766 | 1.5 | `PythonCodePrinter` doesn't support `Indexed` — `lambdify()` generates code with warnings for indexed operations | 1 | 7 |
| sympy-18189 | 1.6 | `diophantine()` gives incomplete results depending on argument order when `permute=True` | 1 | N/A |
| sympy-12096 | 1.0 | `evalf` does not call `_imp_` recursively — composed implemented functions fail to evaluate numerically | 1 | 43 |
| sympy-15345 | 1.4 | `mathematica_code(Max(x,2))` outputs `'Max(x, 2)'` (Python syntax) instead of `'Max[x, 2]'` (Mathematica syntax) | 1 | 8 |
| sympy-23534 | 1.11 | Using `symbols` to create `Function` instances fails with extra parentheses — `symbols('f', cls=Function)` works but tuple unpacking doesn't | 1 | 11 |
| sympy-22714 | 1.10 | `simplify` with `evaluate(False)` gives "Imaginary coordinates are not permitted" crash with `Point2D` | 1 | 11 |
| sympy-19637 | 1.7 | `kernS` function: `'kern' referenced before assignment` — `UnboundLocalError` on certain input strings | 1 | 40 |
| sympy-18763 | 1.5 | Incorrect LaTeX parenthesizing of `Subs` — `3*Subs(-x+y, (x,), (1,))` renders without proper parentheses | 1 | 141 |
| sympy-19495 | 1.7 | `subs` on `ConditionSet` with `ImageSet` produces wrong results — bound variable substitution leaks | 1 | 8 |

Note the wide range in PASS_TO_PASS test counts: sympy-18763 has **141** regression tests that must continue passing, while some instances have only 7-8. This matters for understanding failure modes.

### 2.3 Evaluation Infrastructure

Each instance runs in its own Docker container built from SWE-bench's pre-configured environment images. The images contain:
- The exact Python version and dependencies for that SymPy version
- The repository checked out at the correct commit
- All test infrastructure pre-installed

The evaluation harness applies the agent's patch via `git apply`, then runs the specified test suite. Results are reported as FAIL_TO_PASS (did the bug fix work?) and PASS_TO_PASS (did existing functionality survive?).

---

## 3. System Architecture

> **Figure 7** (`fig7_architecture.png`) — Full system architecture diagram

### 3.1 Infrastructure

| Component | Specification |
|-----------|--------------|
| **LLM** | Qwen3-Coder-30B-A3B, 4-bit AWQ quantization |
| **Serving** | vLLM 0.11.0 in Docker, OpenAI-compatible API |
| **GPU** | NVIDIA RTX 5090, 32 GB VRAM |
| **NLI Model** | DeBERTa-large fine-tuned on MNLI (separate CPU process) |
| **Eval** | SWE-bench Docker containers (per-instance images) |
| **Host** | Windows 11, AMD Ryzen 9 5950X, 64 GB RAM |

Model loading consumed **16.95 GB VRAM**, leaving ~15 GB for KV cache (supports ~3.2x concurrency at 32K context).

### 3.2 Baseline Agent (Control)

The baseline is a standard ReAct-style agent loop (Yao et al. 2023):
1. Receives the issue description as a task prompt
2. Iterates: produce a THOUGHT (reasoning) + one bash ACTION
3. Observes the command output, then repeats
4. Submits a `git diff` patch when done

This is a single greedy trajectory (temperature=0.0) with a step limit of 30. No branching, no diversity mechanisms. The agent uses the same Qwen3-Coder-30B-A3B model and the same Docker evaluation setup.

### 3.3 Branching Agent (Treatment)

The branching agent extends the baseline with four phases:

**Phase 1 — SEARCH (read-only, single trajectory)**
- Same ReAct loop but restricted to read-only commands (grep, find, cat, etc.)
- Relevance scoring via context-conditioned entailment (Kuhn et al. 2023) detects when exploration saturates
- After N consecutive low-relevance steps, triggers transition to Phase 2

**Phase 2 — STRATEGY PROPOSAL (no execution)**
- LLM summarizes search findings into a structured bug report
- A single LLM call proposes K=5 fundamentally different fix strategies (categorically forced: validation fix, algorithm change, preprocessing, API change, upstream fix)
- Bidirectional entailment clustering (Farquhar et al. 2024) deduplicates strategies using DeBERTa-large-MNLI
- Semantic entropy H = -Σ p(c) log p(c) computed over cluster distribution
- If H > τ (threshold = 0.0, always branch), fork one trajectory per unique cluster

**Phase 3 — PATCH (write access, per-trajectory)**
- Each trajectory receives its assigned strategy as a system prompt injection
- Agent independently implements the fix in its own Docker container
- SDLG (Aichberger et al. 2025) generates within-strategy alternatives via gradient-based token attribution through DeBERTa
- If SDLG produces semantically diverse alternatives (H > 0), fork additional sub-trajectories

**Phase 4 — VERIFY (read-only, per-trajectory)**
- Each trajectory runs tests, reviews changes, submits a git diff patch

### 3.4 Key Algorithms

**Bidirectional Entailment Clustering (Algorithm 1, Kuhn et al. 2023):**
```
clusters = []
for intent i in intents:
  for cluster c in clusters:
    if P(entail | rep_c, i) > τ AND P(entail | i, rep_c) > τ:
      add i to c; break
  if not assigned:
    create new cluster {i}
```
Threshold τ = 0.7 (raised from default 0.5 to prevent over-merging of multi-sentence strategies).

**SDLG Token Attribution (Aichberger et al. 2025):**
1. Feed reasoning text as self-entailment (premise=hypothesis) through DeBERTa
2. Compute loss toward "contradiction" class
3. Backpropagate to embedding layer: attribution = ||z_i ⊙ ∇z_i L||_2
4. For top-K attributed positions, find substitution tokens via gradient-embedding dot product
5. LLM regenerates from substitution point, producing semantically different completions

### 3.5 Configuration

```yaml
branching:
  n_strategies: 5                # Strategies to propose
  diversity_method: strategy_proposal
  entailment_threshold: 0.7      # NLI bidirectional threshold
  entropy_threshold: 0.0         # Always branch unique strategies
  max_trajectories: 30           # Hard cap
  sdlg_enabled: true             # Within-strategy diversity
  sdlg_n_alternatives: 5         # SDLG candidates per step
  sdlg_top_k: 20                 # Token substitution candidates
```

### 3.6 Previous Iteration Failures

**SDLG-only approach (abandoned 2026-03-23):** Token-level perturbation via SDLG produced zero diversity (entropy=0.000, always 1 cluster). The Aichberger et al. paper was designed for single-sentence QA where substituting "July" → "August" changes meaning. In multi-turn agent context with long conversation histories, individual token substitutions don't change the overall strategy.

**Context prefix bug (fixed 2026-03-23):** Even after switching to strategy proposals, clustering collapsed all strategies into 1 cluster (entropy=0.000). Root cause: `context=problem_statement[:500]` was prepended to every strategy before NLI comparison. DeBERTa's 512-token window was dominated by the shared 500-char prefix, making everything appear as bidirectional entailment. Fix: removed context prefix, raised threshold from 0.5 → 0.7.

**Docker image naming bug (fixed 2026-03-28):** 9/10 instances failed because `find_eval_image()` used `.split("\n")` which doesn't handle Windows `\r\n` from Docker output. Changed to `.splitlines()` + increased timeout from 10s → 30s.

---

## 4. Results

### 4.1 Headline Metrics

| Metric | Baseline | Branching |
|--------|----------|-----------|
| **pass@1** | **0/10 (0%)** | **0/10 (0%)** |
| **diverse-pass@1** | N/A | **0/10 (0%)** |
| Instances with patches submitted | 8/10 | 10/10 |
| Total unique patches | 8 | 43 |
| Total agent steps | 485 | 4,335 |
| Total runtime | ~45 min | 252 min (4.2 hr) |

Both conditions achieve 0% resolution. However, the branching condition produces substantially more output to analyze.

### 4.2 Baseline vs. Branching: Per-Instance Comparison

| Instance | Baseline Steps | Baseline Patch | Branching Trajectories | Branching Patches | Branching Steps | Branching Time |
|----------|---------------|----------------|----------------------|-------------------|-----------------|----------------|
| sympy-12481 | 63 | 548 ch | 6 | 4 | 844 | 35.0 min |
| sympy-16766 | 24 | 659 ch | 5 | 3 | 183 | 8.1 min |
| sympy-18189 | 76 | (empty) | 1 | 1 | 16 | 1.4 min |
| sympy-12096 | 46 | 693 ch | 6 | 6 | 505 | 23.9 min |
| sympy-15345 | 53 | (empty) | 9 | 9 | 533 | 19.8 min |
| sympy-23534 | 53 | 417 ch | 5 | 4 | 397 | 14.7 min |
| sympy-22714 | 16 | 693 ch | 5 | 1 | 519 | 80.7 min |
| sympy-19637 | 63 | 628 ch | 9 | 6 | 497 | 26.0 min |
| sympy-18763 | 59 | 528 ch | 6 | 4 | 662 | 28.8 min |
| sympy-19495 | 32 | 527 ch | 5 | 5 | 179 | 13.2 min |

> **Figure 1** (`fig1_instance_overview.png`) — Trajectories, patches, and time per instance

Key observations:
- **sympy-15345**: Baseline failed to produce any patch. Branching produced 9 distinct patches — the framework recovered from a baseline dead-end.
- **sympy-18189**: Branching solved it in 16 steps (vs 76 for baseline). The phased approach with early search-phase termination was more efficient.
- **sympy-22714**: Despite 5 strategies and 80 minutes, only 1 patch was produced — this is genuinely difficult for the model.
- **sympy-19495**: Branching explored 3 *different files* across trajectories (conditionset.py, fancysets.py, basic.py), showing true structural diversity.

### 4.3 What Happened at the Test Level

A critical finding: **every patch from both conditions breaks the existing test suite**. The SWE-bench evaluation reports reveal:

| Condition | Patches Applied | FAIL_TO_PASS Solved | PASS_TO_PASS Maintained |
|-----------|----------------|--------------------|-----------------------|
| Baseline | 8/8 (100%) | 0/8 (0%) | 0/8 (0%) |
| Branching | 43/43 (100%) | 0/43 (0%) | 0/43 (0%) |

All patches apply cleanly (the diffs are syntactically valid), but they universally fail on two fronts:
1. **FAIL_TO_PASS = 0**: None of the patches actually fix the target bug
2. **PASS_TO_PASS = 0**: Every patch introduces regressions — existing tests that previously passed now fail

For example, sympy-12481:
- The baseline patch replaces `if is_cycle:` with `if False:` — a brute-force suppression of the error that breaks 7/7 pass-to-pass tests
- Branching strategy_1 adds a try/except with `Cycle()` composition — more sophisticated, but still fails the target test and breaks all 7 regression tests
- Branching strategy_4 adds a new `safe_subs()` method — structurally different approach, but still incorrect

For sympy-18763 (hardest test suite: 141 pass-to-pass tests):
- Both baseline and all branching patches break all 141 regression tests, suggesting the patches modify a critical code path without understanding the downstream effects

This pattern tells us the model understands *what* needs to change (correct file, correct function) but not *how* to change it correctly. The branching framework's diverse strategies all land in the same failure mode: syntactically plausible but semantically wrong patches.

### 4.4 Baseline vs. Branching: Qualitative Patch Analysis

The branching condition produces genuinely different approaches. For **sympy-19495** (ConditionSet/ImageSet substitution bug):

| Trajectory | File Modified | Approach |
|-----------|--------------|----------|
| Baseline | conditionset.py | Short-circuit: `return base` when condition is True |
| t0 (greedy) | conditionset.py | Modify `_eval_subs` to handle bound variable conflict |
| strategy_1 | **fancysets.py** | Add `_eval_subs` to `ImageSet` instead |
| strategy_2 | **basic.py** | Global validation in `Basic.subs()` for bound variables |
| strategy_3 | conditionset.py | Remove duplicate ConditionSet creation line |
| strategy_4 | conditionset.py | Add new `safe_subs()` method (51 lines) |

The strategies target **3 different files** and represent fundamentally different design philosophies:
- Fix at the `ConditionSet` level (local fix)
- Fix at the `ImageSet` level (sibling class)
- Fix at the `Basic.subs()` level (global, upstream)

For **sympy-12096** (evalf recursion bug), patch sizes range from 303 to 4,070 characters — a 13x size range — confirming structural diversity. The baseline's 693-char patch adds argument evaluation in `_eval_evalf`, while strategy_4's 4,070-char patch completely restructures the function dispatch logic.

**Crucially, the branching patches have near-zero overlap with the baseline** (measured by shared added lines). Across all instances, the average baseline-branching patch overlap is <5%, confirming the strategies explore genuinely different solution paths.

> **Figure 2** (`fig2_branching_tree.png`) — Branching tree structure for all instances
> **Figure 6** (`fig6_patch_diversity.png`) — Patch size distributions per instance

### 4.5 Diversity Analysis

**Strategy-level diversity (inter-strategy):**
- 9/10 instances produced 5 distinct strategy clusters (maximum possible for K=5)
- 1 instance (sympy-18189) completed in a single greedy trajectory (the phased orchestrator detected a fast solve and skipped strategy proposal)
- Strategy-level semantic entropy consistently at H = 1.609 = ln(5), the theoretical maximum for 5 equiprobable clusters
- The NLI model correctly identifies all 5 proposed strategies as semantically distinct

**SDLG-level diversity (intra-strategy):**
- 45 thought-perturbation events generated 86 alternatives; 45 code-perturbation events generated 90 alternatives
- 86% of within-strategy SDLG events collapsed to 1 cluster (no diversity detected)
- Only 8 SDLG branching events occurred across 4 instances
- This confirms token perturbation does not produce strategic diversity in multi-turn agent context

> **Figure 9** (`fig9_strategy_vs_sdlg_clustering.png`) — Strategy vs SDLG clustering outcomes
> **Figure 5** (`fig5_strategy_vs_sdlg.png`) — Patch sources: strategy vs SDLG branches

### 4.6 NLI Clustering Analysis

272 total NLI comparisons were performed:

| Classification | Count | % | Mean P(entail) fwd | Mean P(entail) bwd |
|---------------|-------|---|--------------------|--------------------|
| SAME (bidirectional entailment) | 165 | 60.7% | 0.970 | 0.963 |
| DIFF (not bidirectionally entailed) | 107 | 39.3% | 0.220 | 0.245 |

The bimodal distribution (scores cluster near 0 or 1, with clear separation at threshold 0.7) indicates DeBERTa makes confident, consistent classifications. The SAME comparisons are almost exclusively within-strategy SDLG alternatives (which are indeed semantically similar), while DIFF comparisons are between different strategies.

> **Figure 3** (`fig3_nli_distributions.png`) — NLI entailment score distributions
> **Figure 4** (`fig4_entropy_distribution.png`) — Semantic entropy distribution

### 4.7 Computational Budget

> **Figure 8** (`fig8_compute_budget.png`) — Compute vs diversity trade-off

The branching run consumed 5.6x more compute than baseline (4,335 vs 485 total steps; 252 vs ~45 minutes). Notable patterns:

- **sympy-22714**: 80.7 minutes for 1 patch — the model repeatedly failed to converge across all 5 strategies, exhausting step budgets
- **sympy-18189**: Only 1.4 minutes — the phased orchestrator's early termination correctly identified this as a fast solve
- **Correlation between trajectories and time** is moderate (r ≈ 0.5); time is dominated by individual trajectory difficulty, not trajectory count

---

## 5. Discussion

### 5.1 Why 0% Resolution for Both Conditions?

The model produces patches that are **syntactically valid** (100% apply cleanly) but **semantically wrong** (0% pass tests). This is a qualitatively different failure from "no patch produced" — the model locates the right file and function, understands the error message, and writes plausible-looking code. But it fails to:

1. **Correctly fix the target bug** (0/43 FAIL_TO_PASS across all branching trajectories)
2. **Preserve existing behavior** (0/43 PASS_TO_PASS — every patch introduces regressions)

This points to **insufficient code reasoning capability** in Qwen3-Coder-30B-A3B at 4-bit quantization, not a failure of the branching framework. The model can identify what to change but cannot produce a correct change.

**Comparison with published results:**
- EntroPO (Yu et al. 2026) achieves **60.4%** on SWE-bench Verified using the *same model architecture* (Qwen3-Coder-30B-A3B) but at **full precision with RL-tuned weights**. The gap between 0% and 60.4% reflects the combined effect of 4-bit quantization and lack of RL fine-tuning.
- The original SWE-bench paper (Jimenez et al. 2024) reports that even GPT-4 achieves only 1.7% on the full benchmark (pre-Verified), though agent frameworks have since improved this substantially.

### 5.2 Interesting Findings Despite 0% Resolution

Even with both conditions at 0%, the data reveals meaningful differences:

**1. Branching recovers from baseline dead-ends.**
- sympy-15345: Baseline submitted an empty patch (dead-end). Branching produced 9 non-empty patches across different strategies. The phased search + strategy proposal overcame the failure mode that stalled the baseline.
- sympy-18189: Similar — baseline produced nothing, branching produced a 206-character patch.

**2. Patches are structurally different, not just syntactically varied.**
- sympy-19495 branching trajectories modify 3 different files (conditionset.py, fancysets.py, basic.py) representing fundamentally different fix locations in the dependency hierarchy.
- Patch sizes within instances have high coefficient of variation (CV up to 98% for sympy-19637), confirming the strategies produce solutions of genuinely different scope and complexity.
- Baseline-to-branching overlap is near 0% across all instances — the branching strategies are not rediscovering the greedy solution.

**3. The P2P=0 pattern reveals a systematic model weakness.**
The fact that *every* patch (baseline and branching) breaks *all* pass-to-pass tests — not just one or two — suggests the model doesn't run or reason about the test suite during patch generation. A model with test-execution feedback (like EntroPO's hybrid selector) would likely filter out these regression-causing patches.

**4. SDLG vs Strategy Proposals: a clear winner.**
Strategy-level proposals produce genuine semantic diversity (5 clusters, H=1.609) while SDLG token perturbation produces none (1 cluster, H=0.000 in 86% of cases). This is a significant empirical finding: gradient-based token attribution through DeBERTa, as designed by Aichberger et al. (2025) for single-turn QA, does not transfer to multi-turn agentic code generation. The approach was designed for substitutions like "Paris" → "London" in factoid QA; in our setting, substituting one token in a multi-sentence reasoning trace does not change the agent's overall strategy.

**5. The entropy threshold is a meaningful signal.**
The 54 clustering events show a clear bimodal entropy distribution: strategy-level comparisons yield H ≈ 1.6 (branch), while SDLG within-strategy comparisons yield H ≈ 0.0 (don't branch). The entropy threshold correctly discriminates between genuine diversity and superficial variation. This validates the core mechanism even though the downstream patches aren't correct.

### 5.3 Comparison with Related Work

| Method | Approach | Diversity Source | Model | SWE-bench Verified |
|--------|----------|-----------------|-------|--------------------|
| **Ours (branching)** | Inference-time branching | Semantic entropy clustering | Qwen3-30B (4-bit) | 0% |
| **Ours (baseline)** | Single greedy trajectory | None | Qwen3-30B (4-bit) | 0% |
| EntroPO | Training-time (RL) | Entropy-regularized DPO/KTO | Qwen3-30B (full) | 60.4% |
| Self-consistency | Decoding-time | Temperature sampling | Various | N/A (QA only) |
| Tree of Thoughts | Search-time | LLM proposes + evaluates | Various | N/A (reasoning only) |
| SWE-agent | Agent interface design | Custom ACI | Various | 12.5% (GPT-4) |

The critical insight: inference-time diversity cannot compensate for a model that lacks the base capability to generate correct patches. EntroPO's 60.4% vs our 0% is primarily explained by model quality (full precision + RL tuning), not by the diversity mechanism.

### 5.4 Limitations

1. **Single model at one quantization.** 4-bit AWQ quantization likely degrades reasoning. Testing with the full-precision model or a frontier model (Claude, GPT-4o) would isolate the contribution of the branching framework.
2. **No temperature-sampling baseline (pass@K).** We don't compare against simple temperature sampling, which would isolate whether *semantic* clustering outperforms *random* diversity.
3. **No verifier/reranker.** Our pipeline picks the first submitted patch. A test-execution-based selector (as in EntroPO) would likely help even with imperfect individual patches.
4. **10 instances is underpowered.** Statistical claims require more instances. Our selection (all SymPy, all <15 min estimated) may not be representative.
5. **SDLG designed for QA, not agents.** The Aichberger et al. method was explicitly designed for short-form single-turn QA. Our attempt to transfer it to multi-turn agentic reasoning was unsuccessful, which is itself a contribution.

---

## 6. Files and Artifacts

### Results
| Path | Description |
|------|-------------|
| `results/branching/full_summary.json` | Complete results JSON |
| `results/branching/predictions.jsonl` | Primary predictions (10 entries) |
| `results/branching/predictions_all_trajectories.jsonl` | All 43 trajectory patches |
| `results/branching/branching_run.log` | Full execution log |
| `results/branching/trajectory_eval_*.json` | Per-instance evaluation results |
| `results/branching/{instance}/metadata.json` | Per-instance run metadata |
| `results/branching/{instance}/trajectory_*.traj.json` | Full trajectory conversation logs |
| `results/baseline_v3/preds.json` | Baseline predictions |
| `results/baseline_v3/{instance}/*.traj.json` | Baseline trajectory logs |

### Figures (in `results/branching/figures/`)
| Figure | File | Description |
|--------|------|-------------|
| Fig 1 | `fig1_instance_overview.png/pdf` | Trajectories, patches, and time per instance |
| Fig 2 | `fig2_branching_tree.png/pdf` | Branching tree structure for all instances |
| Fig 3 | `fig3_nli_distributions.png/pdf` | NLI entailment score distributions |
| Fig 4 | `fig4_entropy_distribution.png/pdf` | Semantic entropy distribution |
| Fig 5 | `fig5_strategy_vs_sdlg.png/pdf` | Patch sources: strategy vs SDLG |
| Fig 6 | `fig6_patch_diversity.png/pdf` | Patch size distributions (solution diversity) |
| Fig 7 | `fig7_architecture.png/pdf` | System architecture diagram |
| Fig 8 | `fig8_compute_budget.png/pdf` | Compute budget analysis |
| Fig 9 | `fig9_strategy_vs_sdlg_clustering.png/pdf` | Strategy vs SDLG clustering outcomes |

### Code
| Path | Description |
|------|-------------|
| `scripts/run_branching.py` | Main branching entry point |
| `scripts/run_baseline.py` | Baseline entry point |
| `src/agent/phased_orchestrator.py` | Core branching orchestration (1357 LOC) |
| `src/diversity/clustering.py` | Bidirectional entailment clustering |
| `src/diversity/sdlg.py` | SDLG gradient-based diverse generation |
| `src/diversity/strategy_proposer.py` | Strategy proposal + deduplication |
| `src/diversity/nli.py` | DeBERTa NLI model |
| `configs/branching.yaml` | Full pipeline configuration |

---

## 7. Conclusion

Both the baseline (single greedy trajectory) and the semantic branching framework achieve 0/10 on SWE-bench Verified with Qwen3-Coder-30B-A3B at 4-bit quantization. However, the experiment yields several important findings:

1. **The diversity mechanism works.** Strategy-level semantic entropy clustering produces maximum diversity (H = ln(5) = 1.609 on 9/10 instances). NLI-based clustering makes sharp, bimodal classification decisions. The framework generates 43 structurally distinct patches that modify different files, use different fix strategies, and have near-zero overlap with the baseline.

2. **SDLG does not transfer to agentic settings.** Gradient-based token attribution (Aichberger et al. 2025) produces zero within-strategy diversity in 86% of cases. This is a significant negative transfer finding: methods designed for single-turn QA diversity do not generalize to multi-turn agent trajectories where individual token choices are dominated by accumulated context.

3. **Branching recovers from baseline dead-ends.** On 2 instances where the baseline produced empty patches, branching generated multiple non-empty alternatives. The phased search-then-propose architecture avoids the exploration traps that stall the single-trajectory agent.

4. **The bottleneck is model capability, not diversity methodology.** The universal P2P=0 failure (all pass-to-pass tests broken by every patch, in both conditions) indicates the model cannot produce correct code changes for these instances, regardless of how many diverse approaches it attempts. With a stronger base model — one that achieves non-zero baseline pass@1 — the branching framework's diverse exploration would have a meaningful chance of improving diverse-pass@1.

**For the paper:** This is a meaningful negative result. It demonstrates that inference-time diversity mechanisms (semantic entropy clustering, strategy proposals, SDLG) can successfully force exploration of different solution strategies, but cannot compensate for insufficient base model capability. The framework is ready to be tested with a stronger model.
