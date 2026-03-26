# Phase 2: Semantic Branching Implementation

Read `CLAUDE.md` first — it has the full project context, system specs, reference PDFs, and all background. Then come back here.

## Current State (Phase 1 Complete)

Phase 1 baseline is done. We used **mini-swe-agent v2** (the standard SWE-bench agent framework) with our locally-served **Qwen3-Coder-30B-A3B** (4-bit AWQ) via vLLM.

### Infrastructure (all working)
- **vLLM server**: Docker container `vllm-server` serving `qwen3-coder` on `http://localhost:8000/v1` (OpenAI-compatible API). Start script: `scripts/start_vllm.sh`. Model at `D:/models/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit/` (~17 GB, 16.95 GiB VRAM).
- **mini-swe-agent v2**: Installed (`pip install mini-swe-agent`). Config at `configs/swebench_local.yaml`. Uses text-based model class (`litellm_textbased`), 250 step limit, Docker environment.
- **SWE-bench eval images**: All 10 sympy instances tagged as both `sweb.eval.x86_64.sympy__sympy-{id}:latest` and `docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-{id}:latest`.
- **Evaluation harness**: `scripts/run_eval_harness.py` wraps swebench with a `resource` mock for Windows.

### Baseline Results: pass@1 = 0/10 (0%)

| Instance | Difficulty | Patch | Applied | Result | Steps |
|---|---|---|---|---|---|
| sympy-12481 | <15 min | 1109 ch | Yes | FAIL | 65 |
| sympy-13974 | 15m-1hr | 1401 ch | Yes | FAIL | 36 |
| sympy-13877 | 15m-1hr | 995 ch | Yes | FAIL | 51 |
| sympy-14248 | 1-4hr | 619 ch | Error | Malformed | 43 |
| sympy-13852 | 1-4hr | 832 ch | Yes | FAIL | 68 |
| sympy-12489 | 1-4hr | 2259 ch | Yes | FAIL | 46 |
| sympy-18199 | 1-4hr | 881 ch | Yes | FAIL | 49 |
| sympy-17630 | 1-4hr | 0 ch | N/A | Empty | 38 |
| sympy-16597 | 1-4hr | 662 ch | Yes | FAIL | 54 |
| sympy-13878 | >4hr | 0 ch | N/A | Empty | 57 |

Results in `results/baseline/`. Trajectories at `results/baseline/{instance_id}/{instance_id}.traj.json`. Predictions at `results/baseline/preds.json`.

### Key observations from baseline
- Model generates plausible patches but none pass tests — the fixes are wrong, not just formatting issues
- 2/10 submitted empty patches (model failed to follow submission protocol)
- 1/10 had malformed patch (trailing CR characters)
- Average ~50 steps per instance with 250 step limit — plenty of room
- This 0% baseline means even 1 solve from semantic branching demonstrates improvement

## Phase 2: What to Build

Implement **semantic branching** on top of mini-swe-agent v2. At each agent step, instead of taking one greedy action, we:

1. **Generate N diverse candidates** using SDLG
2. **Cluster them by meaning** using bidirectional entailment
3. **Compute semantic entropy** over the clusters
4. **Branch if entropy > τ** — each cluster becomes an independent trajectory

### The three core components

#### 1. SDLG Diverse Generation
**Reference**: `PDFs/Aichberger_2025_SDLG.pdf` (Appendix A of proposal has summary)

At each agent step, generate N=5 candidate responses:
- Take the greedy response from the model
- Feed it through DeBERTa-NLI to compute attribution scores (gradient of cross-entropy loss w.r.t. token embeddings, targeting "contradiction")
- For each candidate: find the highest-scoring token substitution (using substitution score × importance score), replace it, and have the LLM complete from the substitution point
- **Critical**: substitutions apply ONLY to reasoning traces (THOUGHT), NOT to action tokens (bash commands) — to avoid syntactically invalid commands
- Extract a one-sentence intent summary from each candidate via a separate model call

**DeBERTa model needed**: `microsoft/deberta-large-mnli` (~600 MB). This needs to run on GPU alongside the main model. With 32 GB VRAM and ~17 GB for Qwen3, there's ~15 GB headroom — but KV cache uses ~10 GB, leaving ~5 GB. DeBERTa at fp16 is ~1.3 GB — it fits, but may need to run on CPU or use careful memory management.

**Practical concern**: SDLG requires gradient-based attribution through DeBERTa, which means we need `torch` with CUDA. The Windows host has CPU-only torch. Options:
- Run DeBERTa in a separate Docker container with GPU
- Run DeBERTa in the vLLM container (has CUDA torch)
- Install CUDA-enabled torch on Windows via conda
- Run DeBERTa on CPU (slower but simpler — attribution is on short text, not massive batches)

#### 2. Semantic Clustering
**Reference**: `PDFs/farquhar_nature.pdf` (Extended Data Fig. 1) and `PDFs/Farquhar_2024_Semantic_Entropy.pdf` (Algorithm 1)

Cluster the N intent summaries by bidirectional entailment:
- For each pair of summaries (i, j), check if i entails j AND j entails i using DeBERTa-large-MNLI
- If bidirectional entailment holds, they share a cluster
- This is the same DeBERTa model used for SDLG attribution — dual purpose
- Each cluster = a meaningfully different course of action (different diagnosis, target file, or repair strategy)

#### 3. Adaptive Branching
**Reference**: Proposal Section 2.3

- Compute semantic entropy H over the cluster probability distribution: H = -Σ p(c) log p(c) where p(c) = |cluster c| / N
- If H > τ (threshold), branch: each cluster spawns an independent trajectory with its own conversation history
- If H ≤ τ, take the greedy action (no branching)
- Hard cap B=30 trajectories per problem (proposal Appendix B budget analysis)
- τ needs to be tuned — start with τ = 0.5 (moderately selective)

### Architecture approach

The simplest path: **wrap mini-swe-agent's `DefaultAgent.step()` method**.

```python
class BranchingAgent(DefaultAgent):
    def step(self):
        # 1. Get greedy response
        greedy_response = self.model.query(self.messages)

        # 2. Generate N-1 diverse alternatives via SDLG
        alternatives = sdlg_generate(greedy_response, self.messages, n=4)
        candidates = [greedy_response] + alternatives

        # 3. Extract intent summaries
        intents = [extract_intent(c) for c in candidates]

        # 4. Cluster by bidirectional entailment
        clusters = semantic_cluster(intents)

        # 5. Compute semantic entropy
        entropy = compute_semantic_entropy(clusters)

        # 6. Branch or proceed
        if entropy > self.tau and len(self.active_trajectories) < self.B:
            return self.branch(clusters)
        else:
            return self.execute_greedy(greedy_response)
```

Each branched trajectory is a **full copy** of the conversation history up to the branching point, plus the cluster's representative response. Branched trajectories continue independently to completion.

### Output structure

```
results/branching/
    {instance_id}/
        trajectory_0.traj.json      # Main trajectory
        trajectory_1.traj.json      # Branch from step X, cluster 1
        trajectory_2.traj.json      # Branch from step X, cluster 2
        ...
        branching_log.json          # When/where branching happened, entropy values
    preds.json                      # Best patch per instance (or all patches)
    predictions.jsonl               # For swebench eval
    summary.json                    # Results + comparison to baseline
```

### Evaluation

- **diverse-pass@1**: Did ANY branched trajectory produce a passing patch?
- Compare against baseline pass@1
- Report: average trajectories spawned, branching events per problem, entropy distribution

## Implementation Order

1. **Set up DeBERTa** — get `microsoft/deberta-large-mnli` running (CPU or GPU). Write a simple NLI inference wrapper. Test bidirectional entailment on example sentence pairs.

2. **Implement intent extraction** — given a candidate response (THOUGHT + ACTION), extract a one-sentence intent summary. This can be done via the LLM itself (separate call with a short prompt) or a heuristic (extract the first sentence of THOUGHT).

3. **Implement semantic clustering** — given N intent summaries, cluster by bidirectional entailment. Test on example intents.

4. **Implement SDLG token substitution** — this is the hardest part. Need gradient-based attribution through DeBERTa. Given a response, identify high-impact tokens and substitute them. The LLM then completes from the substitution point.

5. **Implement branching agent** — wrap mini-swe-agent's DefaultAgent with branching logic. Handle trajectory management (copying conversation history, managing B=30 cap).

6. **Run on all 10 instances** — log everything. Each instance may produce multiple trajectories.

7. **Evaluate** — run swebench eval on all trajectory patches. Compute diverse-pass@1.

## Important notes

- **Read the PDFs** when you need implementation details — especially `Aichberger_2025_SDLG.pdf` for SDLG and `farquhar_nature.pdf` / `Farquhar_2024_Semantic_Entropy.pdf` for clustering
- **VRAM is tight**: 17 GB model + 10 GB KV cache = 27 GB of 32 GB. DeBERTa needs to fit in remaining ~5 GB or run on CPU
- **Start simple**: get clustering + branching working first with temperature-based diversity (sample N responses at temperature > 0) before implementing full SDLG. This lets you test the branching framework independently from the SDLG generation.
- **Budget**: Proposal Appendix B estimates 10-15 hours total runtime for Phase 2 with B=30 cap
- **Work incrementally** — get each component working before integrating
- **Ask me before making architectural decisions that are ambiguous**
