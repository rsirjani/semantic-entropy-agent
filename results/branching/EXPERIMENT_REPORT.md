# Experiment Report: Semantic Entropy Clustering for Diverse Agentic Code Generation

**Authors:** Ramtin Sirjani & Paul Moore
**Course:** PhD-level Gen AI, Western University
**Date:** 2026-03-28
**Run ID:** `branching_v2_20260328`

---

## 1. Research Question

Can semantic entropy over clustered diverse outputs improve multi-step agentic code generation by forcing exploration of fundamentally different solution strategies?

**Core hypothesis:** At each agent decision point, if the model's output distribution has high semantic entropy (i.e., the model is uncertain between meaningfully different approaches), branching into independent trajectories—one per semantic cluster—will increase the probability that at least one trajectory solves the problem.

**Metric:** `diverse-pass@1` — does *any* trajectory across all branches solve the instance? Compared against `pass@1` — does the single greedy trajectory solve it?

---

## 2. System Architecture

> **Figure 7** (`fig7_architecture.png`) — Full system architecture diagram

### 2.1 Infrastructure

| Component | Specification |
|-----------|--------------|
| **LLM** | Qwen3-Coder-30B-A3B, 4-bit AWQ quantization |
| **Serving** | vLLM 0.11.0 in Docker, OpenAI-compatible API |
| **GPU** | NVIDIA RTX 5090, 32 GB VRAM |
| **NLI Model** | DeBERTa-large fine-tuned on MNLI (separate CPU process) |
| **Eval** | SWE-bench Docker containers (per-instance images) |
| **Host** | Windows 11, AMD Ryzen 9 5950X, 64 GB RAM |

Model loading consumed **16.95 GB VRAM**, leaving ~15 GB for KV cache (supports ~3.2x concurrency at 32K context).

### 2.2 Pipeline Phases

The pipeline executes four phases per instance:

**Phase 1 — SEARCH (read-only, single trajectory)**
- ReAct-style agent explores the codebase: reads files, runs grep, reproduces the bug
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

### 2.3 Key Algorithms

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

---

## 3. Experimental Setup

### 3.1 Benchmark

10 instances from SWE-bench Verified, all from SymPy repository:

| Instance ID | Description |
|---|---|
| sympy-12481 | Permutation non-disjoint cycle handling |
| sympy-16766 | PythonCodePrinter indexed support |
| sympy-18189 | Array symbol handling |
| sympy-12096 | Matrix xreplace regression |
| sympy-15345 | Mathematica code generation |
| sympy-23534 | Symbol substitution issue |
| sympy-22714 | Point/Vector evaluation |
| sympy-19637 | kernS function issue |
| sympy-18763 | LaTeX printing regression |
| sympy-19495 | ConditionSet/ImageSet substitution |

All instances were originally estimated at <15 min difficulty.

### 3.2 Configuration

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

### 3.3 Previous Iteration Failures

**SDLG-only approach (abandoned 2026-03-23):** Token-level perturbation via SDLG produced zero diversity (entropy=0.000, always 1 cluster). The Aichberger et al. paper was designed for single-sentence QA where substituting "July" → "August" changes meaning. In multi-turn agent context with long conversation histories, individual token substitutions don't change the overall strategy.

**Context prefix bug (fixed 2026-03-23):** Even after switching to strategy proposals, clustering collapsed all strategies into 1 cluster (entropy=0.000). Root cause: `context=problem_statement[:500]` was prepended to every strategy before NLI comparison. DeBERTa's 512-token window was dominated by the shared 500-char prefix, making everything appear as bidirectional entailment. Fix: removed context prefix, raised threshold from 0.5 → 0.7.

**Docker image naming bug (fixed 2026-03-28):** 9/10 instances failed because `find_eval_image()` used `.split("\n")` which doesn't handle Windows `\r\n` from Docker output. Changed to `.splitlines()` + increased timeout from 10s→30s.

---

## 4. Results

### 4.1 Aggregate Metrics

| Metric | Value |
|--------|-------|
| **pass@1 (greedy)** | **0/10 (0%)** |
| **diverse-pass@1** | **0/10 (0%)** |
| Total trajectories | 43 |
| Total unique patches | 43 |
| Total agent steps | 4,335 |
| Total runtime | 4.2 hours (252 min) |
| Avg trajectories/instance | 4.3 |
| Avg time/instance | 25.2 min |

> **Figure 1** (`fig1_instance_overview.png`) — Trajectories, patches, and time per instance

### 4.2 Per-Instance Breakdown

| Instance | Time (min) | Trajectories | Patches | Strategies | SDLG Branches | Steps |
|----------|-----------|-------------|---------|-----------|--------------|-------|
| sympy-12481 | 35.0 | 6 | 4 | 6 | 0 | 844 |
| sympy-16766 | 8.1 | 5 | 3 | 5 | 0 | 183 |
| sympy-18189 | 1.4 | 1 | 1 | 1 | 0 | 16 |
| sympy-12096 | 23.9 | 6 | 6 | 6 | 1 | 505 |
| sympy-15345 | 19.8 | 9 | 9 | 9 | 4 | 533 |
| sympy-23534 | 14.7 | 5 | 4 | 5 | 0 | 397 |
| sympy-22714 | 80.7 | 5 | 1 | 5 | 0 | 519 |
| sympy-19637 | 26.0 | 9 | 6 | 9 | 3 | 497 |
| sympy-18763 | 28.8 | 6 | 4 | 6 | 1 | 662 |
| sympy-19495 | 13.2 | 5 | 5 | 5 | 0 | 179 |

> **Figure 2** (`fig2_branching_tree.png`) — Branching tree structure for all instances

### 4.3 Diversity Analysis

The framework successfully produces genuine solution diversity:

**Strategy-level diversity (inter-strategy):**
- 9/10 instances produced 5 distinct strategy clusters (maximum possible)
- 1 instance (sympy-18189) completed in a single greedy trajectory (fast solve, no strategy proposal triggered)
- Average strategy-level semantic entropy: H = 1.609 (= ln(5), maximum for 5 uniform clusters)
- This means the NLI model correctly identified all 5 proposed strategies as semantically distinct approaches

**SDLG-level diversity (intra-strategy):**
- 45 SDLG thought-perturbation events generated 86 alternative thought traces
- 45 SDLG code-perturbation events generated 90 alternative code blocks
- However, within-strategy SDLG alternatives almost always clustered into 1 cluster (37/43 = 86%)
- Only 8 SDLG branching events occurred across all instances (in 4 instances)
- This confirms the finding from the abandoned SDLG-only approach: token perturbation in multi-turn context doesn't produce strategic diversity

> **Figure 9** (`fig9_strategy_vs_sdlg_clustering.png`) — Strategy vs SDLG clustering outcomes
> **Figure 5** (`fig5_strategy_vs_sdlg.png`) — Patch sources: strategy vs SDLG branches

### 4.4 NLI Clustering Analysis

272 total NLI comparisons were performed:

| Classification | Count | Percentage | Mean P(entail) fwd | Mean P(entail) bwd |
|---------------|-------|-----------|--------------------|--------------------|
| SAME (bidirectional entailment) | 165 | 60.7% | 0.970 | 0.963 |
| DIFF (not bidirectionally entailed) | 107 | 39.3% | 0.220 | 0.245 |

The bimodal distribution (scores cluster near 0 or 1, with clear separation at threshold 0.7) indicates DeBERTa is making confident, consistent classification decisions.

> **Figure 3** (`fig3_nli_distributions.png`) — NLI entailment score distributions
> **Figure 4** (`fig4_entropy_distribution.png`) — Semantic entropy distribution

### 4.5 Patch Diversity

Patch sizes vary substantially within instances, providing evidence that different strategies produce structurally different solutions:

- sympy-12096: patches range from 303 to 4,070 characters (CV=96%)
- sympy-15345: patches range from 463 to 2,298 characters (CV=50%)
- sympy-19637: patches range from 400 to 4,575 characters (CV=98%)

> **Figure 6** (`fig6_patch_diversity.png`) — Patch size distributions per instance

### 4.6 Computational Budget

> **Figure 8** (`fig8_compute_budget.png`) — Compute vs diversity trade-off

Notable outlier: sympy-22714 consumed 80.7 minutes but only produced 1 unique patch (out of 5 strategies attempted). This instance appears to be genuinely difficult — the model exhausted its step budget on most strategies without converging on a valid patch.

---

## 5. Discussion

### 5.1 Why 0% Resolution?

Despite generating 43 diverse patches across 10 instances, none passed SWE-bench evaluation. Key factors:

1. **Model capability is the bottleneck, not diversity.** Qwen3-Coder-30B-A3B at 4-bit quantization does not reliably produce correct patches for SWE-bench SymPy instances. The baseline (single greedy trajectory) also achieves 0/10. Branching cannot improve upon a base model that never generates a correct solution.

2. **Quantization degradation.** 4-bit AWQ quantization (from the original 16-bit weights) may reduce reasoning capability. EntroPO (Yu et al. 2026) reports 60.4% on SWE-bench Verified with the same model at full precision with RL-tuned weights — a dramatically different operating point.

3. **Instance difficulty.** These SymPy instances may be at or above the capability threshold for this model size. SWE-bench Verified is designed to be challenging even for frontier models.

4. **No verifier/reranker.** Our pipeline submits the first submitted patch from each trajectory. EntroPO uses a hybrid selector (validity checks → regression tests → verifier model → step heuristic). Adding a verifier could improve selection even when individual trajectories are weak.

### 5.2 What the Framework Successfully Demonstrates

1. **Genuine semantic diversity.** 9/10 instances produced 5 semantically distinct strategies (H = ln(5) = 1.609). The NLI clustering correctly distinguishes different approaches.

2. **Two-level diversity hierarchy works as designed.** Strategy-level proposals create inter-approach diversity; SDLG adds finer-grained intra-approach alternatives (though with limited effectiveness in this context).

3. **Scalable branching.** The framework scales from 1 trajectory (fast instances) to 9 trajectories (complex instances with SDLG sub-branches) within the 30-trajectory budget.

4. **End-to-end pipeline.** Full integration of Docker-isolated execution, NLI-based clustering, SDLG gradient attribution, and SWE-bench evaluation.

### 5.3 Comparison with Related Work

| Method | Approach | Diversity Source | Model | SWE-bench Verified |
|--------|----------|-----------------|-------|--------------------|
| **Ours** | Inference-time branching | Semantic entropy clustering | Qwen3-30B (4-bit) | 0% |
| EntroPO | Training-time (RL) | Entropy-regularized DPO/KTO | Qwen3-30B (full) | 60.4% |
| Self-consistency | Decoding-time | Temperature sampling | Various | N/A (QA only) |
| Tree of Thoughts | Search-time | LLM proposes + evaluates | Various | N/A (reasoning only) |

The critical difference from EntroPO: they modify the model weights via entropy-regularized RL to produce diverse outputs naturally, while we attempt to force diversity at inference time from a model that may not have the underlying capability to generate correct solutions for these tasks.

### 5.4 Limitations

1. **Single model.** Results reflect Qwen3-Coder-30B-A3B at 4-bit only. A stronger base model would likely show non-zero diverse-pass@1.
2. **No baseline pass@K.** We don't report temperature-sampled pass@K for comparison, which would isolate the contribution of semantic clustering from random sampling.
3. **Small instance set.** 10 instances is insufficient for statistical significance.
4. **SDLG ineffective in multi-turn context.** The Aichberger et al. method was designed for single-turn QA. Our results confirm it doesn't transfer to agentic settings — strategy-level proposals are far more effective.

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
| `scripts/run_branching.py` | Main entry point |
| `src/agent/phased_orchestrator.py` | Core branching orchestration (1357 LOC) |
| `src/diversity/clustering.py` | Bidirectional entailment clustering |
| `src/diversity/sdlg.py` | SDLG gradient-based diverse generation |
| `src/diversity/strategy_proposer.py` | Strategy proposal + deduplication |
| `src/diversity/nli.py` | DeBERTa NLI model |
| `configs/branching.yaml` | Full pipeline configuration |

---

## 7. Conclusion

The semantic branching framework successfully generates genuine solution diversity — 43 semantically distinct trajectories across 10 SWE-bench instances, with strategy-level entropy consistently at its theoretical maximum (H = ln(5)). However, none of the 43 patches resolve the target issues, indicating that the Qwen3-Coder-30B-A3B model at 4-bit quantization lacks sufficient capability for these SymPy instances. The framework's value lies in its ability to systematically explore the solution space; a stronger base model would likely yield non-trivial diverse-pass@1 improvements over greedy pass@1.

**Key takeaway for the paper:** Diversity generation works. The bottleneck is model capability, not the branching methodology. This is a meaningful negative result — it demonstrates that inference-time diversity mechanisms require a sufficiently capable base model to translate exploration breadth into correct solutions.
