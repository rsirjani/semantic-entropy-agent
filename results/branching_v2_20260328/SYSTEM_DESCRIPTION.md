# End-to-End System Description: Semantic Entropy Clustering for Diverse Agentic Code Generation

This document describes every component of the system in the order that data flows through it, from a GitHub issue entering the pipeline to a set of evaluated patches coming out. Written for a PhD presentation audience.

---

## 1. The Problem We Are Solving

When an LLM agent tackles a software bug, it follows a single trajectory: one chain of reasoning steps, one sequence of code explorations, one fix attempt. If the model makes a wrong turn early — say, misidentifying the root cause — it commits to that path and never considers alternatives.

**Our core insight**: We can detect when the model is *uncertain between meaningfully different approaches* using **semantic entropy** (Farquhar et al. 2024). When entropy is high, we **branch** the agent into multiple independent trajectories — one per semantically distinct approach — and let them all run to completion. If any trajectory produces a correct fix, we succeed.

This is different from temperature sampling (which produces syntactically varied but strategically identical outputs) and from Tree of Thoughts (which uses the LLM itself to evaluate options). We use an **external NLI classifier** (DeBERTa) to measure semantic similarity, making the diversity assessment independent of the generation model.

---

## 2. Infrastructure: What Runs Where

Three services run simultaneously during execution:

### 2.1 vLLM Server (GPU — Docker container)

**What**: Qwen3-Coder-30B-A3B quantized to 4-bit AWQ, served via vLLM 0.11.0 with an OpenAI-compatible API.

**Why this model**: Qwen3-Coder is an MoE (Mixture of Experts) coding model. The 30B parameter variant with 3B active parameters fits in 16.95 GB at 4-bit precision, leaving room on the RTX 5090's 32 GB VRAM for KV cache. Full weight access is required for SDLG (gradient-based token attribution needs logprobs).

**How it runs**: Docker container with GPU passthrough (`--gpus all`). Accepts HTTP requests at `localhost:8000/v1/completions`. All agent LLM calls route through this single endpoint — the search phase, strategy proposals, intent extraction, and patch generation all use the same model.

```
docker run --gpus all -v D:/models/Qwen3-Coder:/models/qwen3-coder \
  vllm/vllm-openai:latest --model /models/qwen3-coder \
  --served-model-name qwen3-coder --max-model-len 32768 \
  --gpu-memory-utilization 0.85
```

### 2.2 NLI Server (CPU — Python process)

**What**: DeBERTa-large fine-tuned on MNLI (Microsoft Natural Language Inference), running as a FastAPI HTTP server.

**Why separate**: DeBERTa is a 350M-parameter transformer. Running it in the same process as the orchestrator would compete with vLLM for GPU memory. By running it on CPU in a separate process, both can operate simultaneously without memory contention.

**What it provides**: Five endpoints:
- `POST /classify` — Given (premise, hypothesis), returns P(entailment), P(neutral), P(contradiction). This is the core operation used for semantic clustering.
- `POST /classify_batch` — Batch version for efficiency.
- `POST /sdlg_scores` — Given text, computes self-entailment gradients: feeds text as both premise and hypothesis, backpropagates toward the "contradiction" class, returns per-token attribution scores. This identifies which tokens most impact semantic meaning.
- `POST /sdlg_rank` — Full SDLG substitution ranking. For each high-attribution token position, computes substitution scores for the entire vocabulary using gradient-embedding dot products. Returns ranked (position, replacement) candidates.
- `GET /health` — Health check.

**How DeBERTa classification works**: The model receives two texts (premise, hypothesis) concatenated with a `[SEP]` token. Its output is a 3-way softmax: P(entailment), P(neutral), P(contradiction). "Entailment" means "if the premise is true, the hypothesis must also be true." We use bidirectional entailment — both directions must exceed the threshold — to determine if two texts are semantically equivalent.

### 2.3 Orchestrator (CPU — Python process)

**What**: The `PhasedOrchestrator` class (~1,400 lines) that coordinates everything. It calls the vLLM server for LLM completions, the NLI server for clustering decisions, and manages Docker containers for each agent trajectory.

**What it manages**:
- The phased execution flow (SEARCH → PROPOSAL → PATCH → VERIFY)
- Trajectory creation, forking, and cleanup
- Docker containers (one per trajectory, created lazily, destroyed after use)
- The complete audit trail (step logs, trajectory logs, pipeline traces)

---

## 3. The Pipeline: Step by Step

### 3.0 Input: A SWE-bench Instance

Each instance provides:
- `instance_id`: e.g., `sympy__sympy-12481`
- `problem_statement`: The GitHub issue text — a bug report written by a real developer
- A Docker image: Contains the exact repository state at the commit before the fix, with all dependencies

The orchestrator receives these and begins Phase 1.

---

### 3.1 Phase 1: SEARCH (Read-Only Exploration)

**Goal**: Understand the bug by exploring the codebase. Build enough context for strategy proposal.

**What happens**:

A single trajectory (t0) is created with a fresh Docker container from the SWE-bench eval image. The agent operates in a ReAct loop:

```
LOOP:
  1. Agent receives: system prompt + issue description + conversation history
  2. Agent produces: THOUGHT (reasoning text) + ACTION (one bash command)
  3. Orchestrator checks: Is this command read-only? (grep, find, cat, head, ls, etc.)
     - If write command attempted → BLOCK, send error message, agent retries
     - If read command → EXECUTE in Docker container
  4. Agent receives: command output (observation)
  5. Orchestrator scores: How relevant was this finding?
```

**Relevance Scoring** (the novel part of search phase):

After each search step, the orchestrator evaluates whether the finding is relevant to the bug:

1. **LLM summarization**: The raw search output (grep results, code listings) doesn't match the format of a bug description. So we ask the LLM: "In one sentence, what did this search find and why might it be relevant?" This produces a natural-language summary.

2. **NLI relevance check**: We pass (problem_statement, finding_summary) to DeBERTa. If P(entailment) > 0.5, the finding is relevant — it provides information that supports understanding the bug.

3. **Saturation detection**: We track consecutive low-relevance steps. After N consecutive irrelevant findings (default N=3), the search is "saturated" — the agent is going in circles. Transition to Phase 2.

This prevents the agent from wasting steps on unproductive exploration while ensuring it does enough investigation to understand the codebase.

**Context Pruning**:

Before Phase 2, the orchestrator prunes irrelevant search steps from the conversation history. Each irrelevant assistant message (thought + action) and its observation are removed. This frees context window space for the strategy proposal and patch phases. Only the system prompt, instance prompt, and relevant search findings survive.

---

### 3.2 Phase 2: STRATEGY PROPOSAL (No Execution)

**Goal**: Propose K fundamentally different fix strategies, then determine which ones are semantically distinct.

This phase has three sub-steps:

#### Step 2a: Build Search Report

The `StrategyProposer` takes the agent's search history and asks the LLM to write a structured bug report with three sections:

1. **ROOT CAUSE**: What is the bug and why does it happen? (Traces the execution path)
2. **RELEVANT CODE**: Actual code snippets from the search (verbatim, with file paths and line numbers)
3. **FIX POINTS**: Different points in the code where a fix *could* be applied

This report is the input to strategy proposal. The code snippets are critical — they let the strategy proposer reference real code, not hallucinated code.

Only search steps marked as relevant (by the Phase 1 scoring) are included. This focuses the report on productive findings.

#### Step 2b: Propose K Strategies

A single LLM call with a carefully constructed prompt asks for K=5 **fundamentally different** code-level strategies. The prompt includes:

- The problem statement (original issue)
- The search report (with real code snippets)
- Explicit constraints: "Each strategy MUST modify DIFFERENT lines, functions, or files", "MUST take a COMPLETELY DIFFERENT APPROACH"
- Categorical forcing: different mechanisms (validation vs normalization vs delegation), different levels (caller vs callee vs helper), different files/modules

The prompt requires a specific format:
```
STRATEGY 1: [file and function] — [specific code change]
STRATEGY 2: [file and function] — [specific code change]
...
```

This is inspired by Tree of Thoughts (Yao et al. 2023) "propose prompt" which asks the LLM to generate multiple approaches, but we add semantic clustering to verify they're actually different.

#### Step 2c: Cluster Strategies via Bidirectional Entailment

This is the core of our method — Algorithm 1 from Kuhn et al. (2023) / Farquhar et al. (2024):

```
clusters = [Cluster({strategy_0})]

for each strategy_i (i = 1, 2, ..., K-1):
    for each existing cluster c:
        rep = c.representative  (first strategy added to c)

        fwd = DeBERTa(premise=rep, hypothesis=strategy_i)  → P(entailment)
        bwd = DeBERTa(premise=strategy_i, hypothesis=rep)  → P(entailment)

        if fwd > 0.7 AND bwd > 0.7:
            # Bidirectional entailment: these mean the same thing
            c.add(strategy_i)
            break

    if strategy_i not assigned to any cluster:
        clusters.append(Cluster({strategy_i}))
```

**Why bidirectional?** Unidirectional entailment is asymmetric. "Fix the constructor" entails "Modify permutations.py" (fixing the constructor requires modifying that file), but not vice versa (you could modify permutations.py for other reasons). Bidirectional entailment means the two strategies are truly semantically equivalent — they describe the same approach using different words.

**Threshold = 0.7**: We raised this from the default 0.5 because multi-sentence strategy descriptions have more surface-level overlap than single-word QA answers. At 0.5, genuinely different strategies were being merged.

**Semantic Entropy Computation**:

After clustering, we compute:

```
H = -Σ p(c) * log(p(c))
```

where p(c) = |cluster_c| / K (fraction of strategies in each cluster).

- If H = 0: all strategies collapsed to 1 cluster → no diversity → don't branch
- If H = ln(K): each strategy is its own cluster → maximum diversity → branch all
- If H > τ (threshold = 0.0 in our config): branch

With τ=0.0, we always branch whenever there's more than one cluster. In our results, 9/10 instances produced 5 clusters (H=1.609=ln(5)), meaning all 5 proposed strategies were genuinely different according to DeBERTa.

**Output**: A list of unique strategy strings, one per cluster (using the cluster representative).

---

### 3.3 Phase 3: PATCH (Per-Strategy Execution)

**Goal**: Each unique strategy independently implements its fix in an isolated Docker environment.

**How trajectories are created**:

Strategies execute **lazily and sequentially** — only one Docker container exists at a time:

1. **Strategy 0** (t0): Reuses the root trajectory's container (which has the search-phase state). Receives a strategy-specific prompt injected into its conversation history.

2. **Strategies 1-4** (t0_strategy_1, t0_strategy_2, ...): For each subsequent strategy, the orchestrator:
   - Creates a fresh Docker container from the SWE-bench eval image
   - Creates a new BranchingAgent with a fresh LiteLLM model connection
   - Loads the **pruned search messages** (the template saved after Phase 1 context pruning) into the agent's conversation history — this gives each trajectory the same search context
   - Injects the strategy-specific prompt: "You should implement the following fix strategy: [strategy text]"
   - Runs the agent to completion
   - Saves results and **destroys the container** before starting the next strategy

This lazy-sequential approach means only 1 SWE-bench container exists alongside vLLM at any point, keeping memory usage manageable.

**What the agent does within each trajectory**:

The agent runs a normal ReAct loop, but now with write access (sed, patch, cat <<EOF, etc.):

```
LOOP:
  1. Agent produces: THOUGHT + ACTION (now allowed to write files)
  2. Orchestrator executes command in this trajectory's Docker container
  3. Agent observes output
  4. If agent runs "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /testbed/patch.txt":
     → Trajectory is marked "submitted", patch is captured, loop ends
  5. If step limit hit → trajectory marked "completed" (without submission)
```

**SDLG Within-Strategy Diversity**:

On the first write step of each trajectory, the orchestrator optionally runs SDLG to check for within-strategy diversity:

1. **Extract reasoning**: Split the agent's response into THOUGHT (natural language reasoning) and CODE (bash command).

2. **Token attribution via DeBERTa**: Feed the THOUGHT text as self-entailment (premise=hypothesis=thought) through DeBERTa. Backpropagate toward the "contradiction" class. The gradient ∇L at each token embedding position reveals how much that token contributes to the overall semantic meaning. High-attribution tokens are the ones whose presence/absence most changes what the text means.

   Formally: attribution_i = ||z_i ⊙ ∇_{z_i} L_{contradiction}||_2

3. **Substitution scoring**: For each high-attribution token position i, compute substitution scores for all vocabulary tokens j:

   S_ij = (z_i - e_j) · ∇_{z_i} L / (||z_i - e_j|| · ||∇_{z_i} L||)

   This measures how much replacing token i with token j would shift the semantic meaning in the "contradiction" direction — i.e., how much the substitution changes what the text means.

4. **Importance scoring**: For each candidate substitution, query the LLM for the substitute token's probability at that position (via logprobs API). High probability = the substitution is a likely alternative, not a random word.

5. **Combined ranking**: Score = (attribution + substitution + importance) / 3. Select the top candidate substitution.

6. **Generate alternative**: Replace the token in the THOUGHT, truncate the response at that point, and let the LLM complete the rest. This forces the model to generate a new reasoning chain (and consequently a new code action) from the point of divergence.

7. **Cluster alternatives**: Extract intent summaries from the greedy response and each SDLG alternative. Cluster them via bidirectional entailment. If multiple clusters emerge (entropy > 0), fork sub-trajectories (e.g., t0_strategy_1_sdlg_1).

**In practice**: SDLG produces very little within-strategy diversity (86% of events result in 1 cluster). Token-level perturbation in reasoning text doesn't change the overall approach when the strategy has already been fixed by the strategy prompt. This is a key empirical finding.

---

### 3.4 Phase 4: VERIFY (Per-Trajectory)

**Goal**: Test the fix and submit a patch.

The agent transitions to VERIFY phase when it attempts to run tests or submit. In this phase, it can:
- Run `python -m pytest` or specific test commands
- Run `git diff` to review changes
- Submit via `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /testbed/patch.txt`

When the agent submits, the orchestrator captures the patch (the git diff output) and marks the trajectory as submitted.

---

### 3.5 Output Collection

After all strategies (and their SDLG children) complete, the orchestrator collects:

**Per trajectory**:
- `trajectory_{id}.traj.json` — Full conversation history (every message, thought, action, observation)
- Patch text (git diff)
- Submission status (submitted or not)
- Step count, parent/child relationships

**Per instance**:
- `metadata.json` — Summary: elapsed time, trajectory count, strategies proposed, branching events, all patches with their trajectory IDs and strategy descriptions
- `step_log.json` — Per-step details for every trajectory
- `phased_decisions.log` — Human-readable log of phase transitions and branching decisions

**Across all instances**:
- `predictions.jsonl` — One primary prediction per instance (best submitted patch) in SWE-bench format
- `predictions_all_trajectories.jsonl` — ALL trajectory patches for diversity analysis
- `branching_run.log` — Full execution log
- `trace.jsonl` — Meticulous I/O trace of every pipeline operation

---

## 4. Evaluation

### 4.1 Per-Trajectory Evaluation

The `eval_all_trajectories.py` script evaluates each trajectory's patch individually against SWE-bench:

For each trajectory:
1. Write the patch to a temporary single-prediction JSONL file
2. Call `swebench.harness.run_evaluation` which:
   - Spins up a fresh Docker container from the eval image
   - Applies the patch via `git apply`
   - Runs the FAIL_TO_PASS tests (should now pass if the bug is fixed)
   - Runs the PASS_TO_PASS tests (should still pass — no regressions)
3. Report: `resolved = (all F2P pass) AND (all P2P pass)`

### 4.2 Metrics

- **pass@1 (baseline)**: Does the single greedy trajectory resolve the instance?
- **pass@1 (branching)**: Does the primary (first submitted) trajectory resolve it?
- **diverse-pass@1**: Does *any* trajectory across all branches resolve it?

diverse-pass@1 is the key metric. If diverse-pass@1 > pass@1, branching helped. If diverse-pass@1 = pass@1, the diverse trajectories didn't find anything the greedy path missed.

---

## 5. What Makes This System Novel

### 5.1 vs. Temperature Sampling (Self-Consistency, Wang et al. 2023)

Temperature sampling generates syntactically varied outputs from the same token distribution. Two temperature samples often represent the same strategy with different variable names. Our approach uses an **external semantic classifier** (DeBERTa NLI) to verify that branches represent genuinely different approaches.

### 5.2 vs. Tree of Thoughts (Yao et al. 2023)

ToT uses the LLM itself to evaluate which branches to pursue (self-evaluation). This is unreliable because the model may systematically prefer one type of approach. We use a **separate NLI model** for evaluation, decoupling generation from assessment.

### 5.3 vs. EntroPO (Yu et al. 2026)

EntroPO achieves diversity by **modifying the training loss** — adding entropy regularization to DPO/KTO so the model naturally produces diverse outputs. This is a training-time method requiring access to training infrastructure. Our approach works at **inference time** with any frozen model — no fine-tuning required.

### 5.4 vs. LATS (Zhou et al. 2023)

LATS uses Monte Carlo Tree Search with LLM-generated value estimates. Our approach uses **information-theoretic criteria** (semantic entropy) rather than learned value functions to decide when and where to branch.

### 5.5 Our Specific Contribution

We combine:
1. **Semantic entropy** (Farquhar et al. 2024) as the branching criterion
2. **SDLG** (Aichberger et al. 2025) for gradient-based diverse generation
3. **Strategy proposals** (ToT-inspired) as the primary diversity source
4. **Phased execution** (SEARCH → PROPOSE → PATCH → VERIFY) to structure the agent loop
5. **Docker isolation** per trajectory for safe parallel exploration

The system also demonstrates empirically that:
- Strategy-level proposals produce genuine diversity (H = ln(5) consistently)
- SDLG token perturbation does NOT transfer from QA to agentic settings
- NLI-based bidirectional entailment clustering makes sharp, bimodal decisions (scores near 0 or 1)

---

## 6. Code Map

| Component | File | LOC | What It Does |
|-----------|------|-----|-------------|
| **Orchestrator** | `src/agent/phased_orchestrator.py` | 1,357 | Manages all 4 phases, trajectory creation/destruction, SDLG triggers |
| **Trajectory** | `src/agent/trajectory.py` | 369 | Trajectory dataclass, TrajectoryManager, container cloning |
| **Agent** | `src/agent/branching_agent.py` | 149 | Extends mini-swe-agent's DefaultAgent with `query_only()`, `execute_response()`, `inject_and_execute()` |
| **Phases** | `src/agent/phases.py` | ~200 | Phase definitions, command allowlists (SEARCH=read-only, PATCH=write, VERIFY=test) |
| **Clustering** | `src/diversity/clustering.py` | 185 | Algorithm 1 (bidirectional entailment clustering) + entropy computation |
| **SDLG** | `src/diversity/sdlg.py` | 549 | Gradient-based token attribution + substitution + LLM completion |
| **Strategy Proposer** | `src/diversity/strategy_proposer.py` | 301 | Search report building + K-strategy proposal + prompt engineering |
| **Intent Extraction** | `src/diversity/intent.py` | 204 | One-sentence intent summaries (LLM or heuristic) |
| **NLI Model** | `src/diversity/nli.py` | 195 | DeBERTa wrapper: classify, entails, bidirectional_entailment, compute_sdlg_scores |
| **NLI Client** | `src/diversity/nli_client.py` | ~100 | HTTP client for the separate NLI server process |
| **Relevance** | `src/diversity/relevance.py` | ~150 | LLM-summarize-then-NLI relevance scoring for search phase |
| **NLI Server** | `scripts/nli_server.py` | 190 | FastAPI server wrapping DeBERTa (classify, batch, SDLG endpoints) |
| **Run Script** | `scripts/run_branching.py` | 268 | Entry point: load config, find Docker images, run all instances |
| **Evaluation** | `scripts/eval_all_trajectories.py` | 194 | Per-trajectory SWE-bench evaluation |
| **Config** | `configs/branching.yaml` | 224 | All hyperparameters: strategies, thresholds, step limits, SDLG settings |

**Total**: ~5,976 lines of implementation code.

---

## 7. Data Flow Diagram (for a single instance)

```
                    GitHub Issue (problem_statement)
                              │
                              ▼
                    ┌─────────────────┐
                    │  PHASE 1: SEARCH │  Single trajectory (t0)
                    │  Read-only ReAct │  Docker container #1
                    └────────┬────────┘
                             │ Each step:
                             │  1. LLM → THOUGHT + grep/cat/find
                             │  2. Execute in container
                             │  3. Score relevance via NLI
                             │  4. Track consecutive low-relevance
                             │
                             │ Saturated? (N consecutive irrelevant)
                             ▼
                    ┌─────────────────┐
                    │   Context Prune  │  Remove irrelevant search steps
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  PHASE 2: PROPOSE│
                    │                  │  1. LLM builds search report
                    │  Strategy        │  2. LLM proposes K=5 strategies
                    │  Proposal        │  3. DeBERTa clusters via bidir. entailment
                    │                  │  4. Compute H = -Σ p(c) log p(c)
                    └────────┬────────┘
                             │
                    (5 unique strategies, H=1.609)
                             │
              ┌──────────────┼──────────────┐──────────┐──────────┐
              ▼              ▼              ▼          ▼          ▼
         ┌────────┐    ┌────────┐    ┌────────┐  ┌────────┐  ┌────────┐
         │  t0    │    │ t0_s1  │    │ t0_s2  │  │ t0_s3  │  │ t0_s4  │
         │ Strat 0│    │ Strat 1│    │ Strat 2│  │ Strat 3│  │ Strat 4│
         │        │    │        │    │        │  │        │  │        │
         │ Docker │    │ Docker │    │ Docker │  │ Docker │  │ Docker │
         │ cont.  │    │ cont.  │    │ cont.  │  │ cont.  │  │ cont.  │
         │ #1     │    │ #2     │    │ #3     │  │ #4     │  │ #5     │
         └───┬────┘    └───┬────┘    └───┬────┘  └───┬────┘  └───┬────┘
             │             │             │           │           │
             │  PHASE 3:   │             │           │           │
             │  PATCH      │   (run sequentially, 1 container at a time)
             │  Write cmds │             │           │           │
             │  + SDLG     │             │           │           │
             │             │             │           │           │
             ▼             ▼             ▼           ▼           ▼
         ┌────────┐    ┌────────┐    ┌────────┐  ┌────────┐  ┌────────┐
         │ PHASE 4│    │ PHASE 4│    │ PHASE 4│  │ PHASE 4│  │ PHASE 4│
         │ VERIFY │    │ VERIFY │    │ VERIFY │  │ VERIFY │  │ VERIFY │
         │ Test & │    │ Test & │    │ Test & │  │ Test & │  │ Test & │
         │ Submit │    │ Submit │    │ Submit │  │ Submit │  │ Submit │
         └───┬────┘    └───┬────┘    └───┬────┘  └───┬────┘  └───┬────┘
             │             │             │           │           │
             ▼             ▼             ▼           ▼           ▼
        ┌─────────────────────────────────────────────────────────────┐
        │              SWE-bench Evaluation (per trajectory)          │
        │  Apply patch → Run F2P tests → Run P2P tests → Resolved?   │
        └─────────────────────────────────────────────────────────────┘
             │             │             │           │           │
             ▼             ▼             ▼           ▼           ▼
        diverse-pass@1 = (any trajectory resolved?)
```

**Key constraint**: Docker containers are created and destroyed sequentially. Only 1 SWE-bench container exists at any time alongside the vLLM server, keeping total VRAM + RAM within limits (16.95 GB model + ~3 GB container).
