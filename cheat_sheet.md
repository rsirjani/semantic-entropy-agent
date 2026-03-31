# Presentation Cheat Sheet

---
---

## 1. NLI Relevance Scoring in the Search Phase

---

### What It Does

During the **search phase**, the agent runs bash commands (`grep`, `find`, `cat`) to explore the
codebase. After each step, we score how relevant the finding is to the bug. When N consecutive
steps score low, we declare **search saturation** and transition to the strategy phase.

---

### Why Not Pure DeBERTa NLI?

DeBERTa NLI is trained on **natural language sentence pairs**, not code-to-bug-description pairs.

| Problem | What happens |
|---------|-------------|
| No shared context between code output and bug description | Entailment score always ~0 |
| Too much shared context (copy-pasted keywords) | Entailment score always ~1 |

**Solution**: Use the LLM itself (Qwen3-Coder) to bridge the domain gap.

---

### The Two-Step Process

```
┌──────────────────────┐      ┌──────────────────────┐      ┌─────────────┐
│  Raw search output   │ ──→  │  LLM summarizes to   │ ──→  │ LLM rates   │
│  (grep, code, etc.)  │      │  natural language     │      │ relevance   │
│                      │      │  one-sentence summary │      │ 0–10        │
└──────────────────────┘      └──────────────────────┘      └─────────────┘
```

#### Step 1 — LLM Summarization

Converts raw search output (grep results, code snippets) into a natural language sentence.

**Prompt:**
```
An AI agent is debugging a software issue. It ran a search command and got this result.
In one sentence, what did this search find and why might it be relevant?

Agent's reasoning: {thought}
Search result (truncated): {observation}

One-sentence summary of what was found:
```

| Parameter | Value |
|-----------|-------|
| `{thought}` | The agent's reasoning for running the command (truncated to 300 chars) |
| `{observation}` | The bash command output (truncated to 500 chars) |
| temperature | 0.0 |
| max_tokens | 80 |

**Output example**: *"Found the Permutation constructor in sympy/combinatorics/permutations.py
which handles cycle notation input"*

---

#### Step 2 — LLM Relevance Rating

The summary from Step 1 is compared to the problem statement.

**Prompt:**
```
Rate how relevant this investigation finding is to the bug below.
Bug: {problem_statement}
Finding: {finding_summary}
Rate relevance from 0 (completely irrelevant) to 10 (directly identifies the bug).
Respond with ONLY a single number 0-10.
```

| Parameter | Value |
|-----------|-------|
| `{problem_statement}` | First 300 chars of the GitHub issue |
| `{finding_summary}` | The one-sentence summary from Step 1 |
| temperature | 0.0 |
| max_tokens | 5 |

**Output processing:**
1. Regex extracts first integer from response
2. Clamp to max 10
3. Normalize: `relevance = score / 10.0` → value in **[0, 1]**

---

### Saturation Detection

```
if consecutive_low_relevance_steps >= N:
    → transition to STRATEGY PHASE

where "low relevance" = score <= threshold (default threshold = 0.5)
```

The orchestrator tracks how many search steps in a row scored below threshold.
Once that count hits the cap, the search phase ends.

---
---

## 2. The Prompt for Generating 5 Diverse Strategies

---

### Context

After search saturation, the pipeline does two things in sequence:
1. **Build a search report** — distill the exploration history into a structured summary
2. **Propose strategies** — ask the LLM to propose N fundamentally different fix approaches

---

### Step 1: Search Report Generation

The search history is condensed into a structured report. Only steps that scored **above the
relevance threshold** are included (low-value steps are pruned).

**Prompt:**
```
An AI coding agent investigated a software bug. Below is its exploration history
with the actual code it found.

Write a structured report with these sections:
1. ROOT CAUSE: What is the bug and why does it happen? Trace the execution path.
2. RELEVANT CODE: For each relevant location, include the actual code snippet
   (verbatim from the observations below). Tag each with file path, function name,
   and line numbers.
3. FIX POINTS: What are the different points in the code where a fix COULD be
   applied? (e.g., the validation layer, the processing layer, a caller, a helper
   function, etc.)

IMPORTANT: Include the actual code from the observations. Do NOT paraphrase code —
copy it exactly. The strategies proposed later need to reference real lines.

Exploration history:
{history}

Structured report (include verbatim code snippets from the observations):
```

| Parameter | Value |
|-----------|-------|
| `{history}` | Last 15 search steps, filtered to relevant-only, max 8000 chars |
| temperature | 0.0 |
| max_tokens | 1500 |

---

### Step 2: The Strategy Proposal Prompt (verbatim)

```
A software bug needs to be fixed. Here is the problem description and
investigation findings.

## Problem
{problem_statement}

## Investigation Findings
{search_report}

## Task
Propose exactly {n} FUNDAMENTALLY DIFFERENT code-level strategies to fix this bug.

CRITICAL RULES:
- Each strategy MUST take a COMPLETELY DIFFERENT APPROACH — not just a different
  implementation of the same idea
- Each strategy MUST modify DIFFERENT lines, functions, or files
- Each strategy MUST produce a structurally different patch (different diff)
- Do NOT propose strategies that all end up changing the same line of code
- Be SPECIFIC: name the exact function, file, and line range each strategy
  would change
- Describe the actual code transformation (what gets added/removed/replaced)

Think about fundamentally different approaches:
- Fix at different levels of the call stack (caller vs callee vs helper)
- Fix in different files or modules
- Use different mechanisms (validation vs normalization vs delegation)
- Address different root causes (input handling vs processing logic vs output)

IMPORTANT: Before writing each strategy, explicitly state what makes it
FUNDAMENTALLY DIFFERENT from all previous strategies. If you cannot articulate
the difference, you are proposing the same fix with surface variation.

Format your response EXACTLY as:
STRATEGY 1: [Which file and function to modify] — [What specific code change]
STRATEGY 2: [Which file and function to modify] — [What specific code change]
...and so on for all {n} strategies.
```

| Parameter | Value |
|-----------|-------|
| `{problem_statement}` | First 2000 chars of the GitHub issue |
| `{search_report}` | The structured report from Step 1 (max 4000 chars) |
| `{n}` | 5 (default) |
| temperature | **1.0** (high, to encourage diversity) |
| max_tokens | 1200 |

---

### Iterative Rejection (if first pass yields < 5)

If parsing yields fewer than 5 strategies, a **second pass** explicitly rejects already-seen
approaches:

```
## Already Proposed Strategies (DO NOT repeat or rephrase these)
- ALREADY PROPOSED (DO NOT REPEAT): {strategy_1}
- ALREADY PROPOSED (DO NOT REPEAT): {strategy_2}
...

Propose {n_more} NEW strategies. Each MUST:
- Target DIFFERENT code locations than any existing strategy
- Use a DIFFERENT mechanism (not just rewording the same fix)
- Be specific about file, function, and code change
```

---

### Parsing

Strategies are extracted via regex:

```
STRATEGY\s+\d+\s*(?:\([^)]*\))?\s*:\s*(.+?)
```

Matches `STRATEGY N:` or `STRATEGY N (Category):`.
Falls back to numbered lines (`1.`, `2.`), then raw response as single strategy.

---
---

## 3. Bidirectional Entailment, Clustering, and Semantic Entropy

---

### 3a. DeBERTa NLI Classification

**Model**: DeBERTa-large fine-tuned on MNLI (Multi-Genre Natural Language Inference).

**Input**: A pair of natural language sentences — (premise, hypothesis).

**Tokenization**:
```
[CLS]  premise tokens  [SEP]  [SEP]  hypothesis tokens  [SEP]
```

**Forward pass**:
```
                                    ┌─────────────┐
[CLS] premise [SEP][SEP] hyp [SEP] │             │
         ──────────────────────→    │  DeBERTa    │ ──→ logits ∈ ℝ³
                                    │  (large)    │        │
                                    └─────────────┘        ↓
                                                      softmax
                                                         │
                                                         ↓
                                               { p₀,  p₁,  p₂ }
```

**Output**: Three probabilities that sum to 1:

| Index | Label | Symbol | Meaning |
|:-----:|:-----:|:------:|---------|
| 0 | contradiction | p₀ | Premise **contradicts** hypothesis |
| 1 | neutral | p₁ | Premise **neither entails nor contradicts** hypothesis |
| 2 | entailment | p₂ | Premise **logically implies** hypothesis |

**Math**:

```
logits = DeBERTa(premise, hypothesis)      logits = [l₀, l₁, l₂]

              exp(lₖ)
p_k = ─────────────────────                for k ∈ {0, 1, 2}
       exp(l₀) + exp(l₁) + exp(l₂)
```

---

### 3b. Entailment Check

Given two texts A and B, and a threshold **θ** (default = 0.5):

```
                   ┌ True    if  p₂(premise=A, hypothesis=B)  >  θ
entails(A, B) =   │
                   └ False   otherwise
```

In plain English: **"Does A logically imply B?"** — yes if the entailment probability exceeds θ.

---

### 3c. Bidirectional Entailment

Two texts are **semantically equivalent** if and only if **each entails the other**:

```
bidir_entail(A, B)  =  entails(A, B)  ∧  entails(B, A)
```

Expanded:

```
bidir_entail(A, B) = True
    iff   P(entailment | premise=A, hypothesis=B) > θ
    AND   P(entailment | premise=B, hypothesis=A) > θ
```

**Why bidirectional?** One-way entailment is too loose:

| A | B | A→B? | B→A? | Bidir? | Correct? |
|---|---|:----:|:----:|:------:|:--------:|
| "Fix the constructor" | "Modify the code" | Yes | No | **No** | Different clusters |
| "Remove the duplicate check" | "Delete the redundant validation" | Yes | Yes | **Yes** | Same cluster |
| "Add input normalization" | "Fix the constructor" | No | No | **No** | Different clusters |

---

### 3d. Clustering Algorithm (Algorithm 1, Farquhar et al. 2024)

**Inputs**:

| Input | Type | Description |
|-------|------|-------------|
| intents | list of N strings | One-sentence intent summaries, one per candidate |
| context | string | The problem statement (GitHub issue description) |
| θ | float (default 0.5) | Entailment threshold |

**Context concatenation** (per Algorithm 1 of Kuhn et al. 2023):

Before any NLI comparison, each intent is prepended with the problem context:

```
nli_input_i  =  context  +  " "  +  intent_i
```

**Why?** Meaning depends on context. "Modify the constructor" and "fix the disjointness check"
might describe the **same** approach when the context is *"Permutation constructor fails with
non-disjoint cycles."*

**The Algorithm** (greedy, sequential assignment):

```
FUNCTION cluster(intents[0..N-1], context, θ):

    clusters ← []

    FOR i = 0 TO N-1:
        assigned ← False

        FOR EACH cluster C IN clusters:

            rep ← C.representative_idx
            a   ← context + " " + intents[rep]     // representative with context
            b   ← context + " " + intents[i]       // current intent with context

            fwd ← P(entailment | premise=a, hypothesis=b)    // forward NLI
            bwd ← P(entailment | premise=b, hypothesis=a)    // backward NLI

            IF fwd > θ  AND  bwd > θ:
                C.indices.append(i)
                assigned ← True
                BREAK                              // assign to FIRST match

        IF NOT assigned:
            clusters.append(
                new Cluster(representative=i, indices=[i])
            )

    RETURN clusters
```

**Key properties**:

| Property | Detail |
|----------|--------|
| Order-dependent | First intent seen becomes cluster representative |
| Greedy | Assigns to first matching cluster, not best match |
| Single representative | Only compares new intents to the cluster's first member |
| Linear in clusters | Each intent compared to at most K existing representatives |
| Worst case | O(N × K) NLI calls, where K ≤ N |

**Output**: List of `SemanticCluster` objects:

```
SemanticCluster:
    indices:            [0, 3]          ← which candidate indices belong here
    representative_idx: 0               ← index of the cluster representative
    intents:            ["Fix the...",  ← the intent strings
                         "Repair the..."]
```

---

### 3e. Semantic Entropy

**Purpose**: Quantify how diverse the candidate set is.
- High entropy → many meaningfully different approaches → branch
- Low entropy → all candidates say the same thing → don't branch

**Definition** (discrete variant, Farquhar et al. 2024):

```
              K
    H  =  −  Σ   p(cₖ) · ln( p(cₖ) )
             k=1
```

**Every variable explained**:

| Symbol | Name | Definition | Type |
|:------:|------|-----------|:----:|
| H | Semantic entropy | The output — diversity measure in nats | scalar ≥ 0 |
| K | Number of clusters | How many semantic clusters were found | integer ≥ 1 |
| k | Cluster index | Iterates from 1 to K | integer |
| cₖ | Cluster k | The k-th semantic cluster | set of indices |
| \|cₖ\| | Cluster size | Number of candidates in cluster k | integer ≥ 1 |
| N | Total candidates | Total number of intent summaries | integer ≥ 1 |
| p(cₖ) | Cluster probability | = \|cₖ\| / N (fraction of candidates in cluster k) | float ∈ (0,1] |
| ln | Natural logarithm | Base-e logarithm | function |
| − | Negation | Makes H positive (since ln of fractions is negative) | — |

**Bounds**:

| Scenario | Cluster sizes | H value | Interpretation |
|----------|:------------:|:-------:|---------------|
| All same | [5] | **0** | Zero diversity — all candidates equivalent |
| Two equal groups | [3, 2] (approx) | **0.673** | Moderate diversity |
| All different | [1, 1, 1, 1, 1] | **ln(5) ≈ 1.609** | Maximum diversity |

**Derivation of bounds**:
- **Minimum**: If K=1, then p(c₁)=1, and −1·ln(1) = 0
- **Maximum**: If all K clusters have size N/K, then p(cₖ) = 1/K for all k,
  and H = −K · (1/K) · ln(1/K) = ln(K)

---

### 3f. Branching Decision

```
              ┌ True   (BRANCH)        if  H > τ
should_branch │
              └ False  (DON'T BRANCH)  if  H ≤ τ
```

Where **τ** (tau) is the entropy threshold (default = **0.5**).

- **H > τ** → candidates are diverse → each cluster spawns its own independent trajectory
- **H ≤ τ** → candidates converged → take the greedy action, no branching

---

### 3g. Worked Example (end to end)

**Setup**: 5 strategy proposals for a SymPy Permutation bug.

```
Strategy 0: "Normalize overlapping cycles in the Permutation constructor"
Strategy 1: "Raise ValueError when cycles share elements"
Strategy 2: "Canonicalize cycle input in the from_sequence method"
Strategy 3: "Add validation in the Permutation constructor"
Strategy 4: "Fix the _af_new helper to detect overlapping cycles"
```

**Step 1 — Clustering** (θ = 0.5):

```
i=0: No clusters yet → Create Cluster A = {0} (rep: Strategy 0)

i=1: Compare to rep of A (Strategy 0):
     fwd = P(ent | "Normalize..." → "Raise ValueError...") = 0.12  < 0.5
     → NOT same. Create Cluster B = {1}

i=2: Compare to rep of A (Strategy 0):
     fwd = P(ent | "Normalize..." → "Canonicalize...") = 0.78
     bwd = P(ent | "Canonicalize..." → "Normalize...") = 0.65
     → Both > 0.5 → SAME. Add to Cluster A = {0, 2}

i=3: Compare to rep of A (Strategy 0):
     fwd = P(ent | "Normalize..." → "Add validation...") = 0.72
     bwd = P(ent | "Add validation..." → "Normalize...") = 0.58
     → Both > 0.5 → SAME. Add to Cluster A = {0, 2, 3}

i=4: Compare to rep of A (Strategy 0):
     fwd = P(ent | "Normalize..." → "Fix _af_new...") = 0.21  < 0.5
     → NOT same. Compare to rep of B (Strategy 1):
     fwd = P(ent | "Raise ValueError..." → "Fix _af_new...") = 0.15  < 0.5
     → NOT same. Create Cluster C = {4}
```

**Result**: 3 clusters with sizes [3, 1, 1]

**Step 2 — Semantic entropy**:

```
p(A) = 3/5 = 0.6       p(B) = 1/5 = 0.2       p(C) = 1/5 = 0.2

H = −[ 0.6·ln(0.6)  +  0.2·ln(0.2)  +  0.2·ln(0.2) ]

  = −[ 0.6·(−0.5108)  +  0.2·(−1.6094)  +  0.2·(−1.6094) ]

  = −[ −0.3065  +  (−0.3219)  +  (−0.3219) ]

  = −(−0.9503)

  = 0.950
```

**Step 3 — Branching decision**:

```
H = 0.950  >  τ = 0.5   →   BRANCH

→ Spawn 3 independent trajectories:
    Trajectory A: pursues Strategy 0 ("Normalize overlapping cycles...")
    Trajectory B: pursues Strategy 1 ("Raise ValueError...")
    Trajectory C: pursues Strategy 4 ("Fix _af_new helper...")
```

---
---

## 4. SDLG (Semantically Diverse Language Generation) — Full Detail

---

### Overview

SDLG (Aichberger et al., 2025) generates diverse alternative responses by:
1. Finding which tokens most impact semantic meaning (via NLI gradients)
2. Identifying substitutes that shift meaning while remaining plausible
3. Replacing the top-ranked token and letting the LLM complete from there

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────┐
│ Greedy       │    │ DeBERTa gradient │    │ Rank (pos, sub) │    │ Substitute   │
│ response y¹  │ →  │ attribution on   │ →  │ pairs by        │ →  │ top token,   │
│              │    │ each token       │    │ A + S + I       │    │ LLM completes│
└─────────────┘    └──────────────────┘    └─────────────────┘    └──────────────┘
```

---

### Step 0: Get Greedy Response

Generate the baseline response at **temperature = 0**. This is **y¹**, the greedy completion.

Split it into two parts:

```
┌─────────────────────────────────────────────────────────────────┐
│  THOUGHT (natural language reasoning)                           │
│  "The bug is in the Permutation constructor. When cycles        │
│   overlap, the constructor silently merges them. I should       │
│   remove the duplicate check and add normalization..."          │
├─────────────────────────────────────────────────────────────────┤
│  CODE (bash command inside ```mswea_bash_command ... ```)       │
│  sed -i 's/old_code/new_code/' sympy/combinatorics/perms.py    │
└─────────────────────────────────────────────────────────────────┘
```

---

### Step 1: Gradient-Based Token Attribution — Score A_i

**Goal**: Find which tokens in the text carry the most semantic weight.

**Process**:

```
1. Self-entailment setup:
   Input to DeBERTa:  [CLS]  text  [SEP]  [SEP]  text  [SEP]
                       ^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^
                            premise              hypothesis
                                                 (same text)

2. Forward pass:
   DeBERTa(input) → logits = [l₀, l₁, l₂]    (contradiction, neutral, entailment)

3. Loss toward CONTRADICTION (class 0):
   L = CrossEntropy(logits, target=0)
   L = −log( softmax(logits)[0] )
   L = −log( exp(l₀) / (exp(l₀) + exp(l₁) + exp(l₂)) )

4. Backpropagate through the embedding layer:
   ∇_{z_i} L  for every token embedding z_i

5. Extract hypothesis portion only:
   Tokens after the [SEP][SEP] delimiter (not the premise copy)
```

**Why target contradiction?** The text entails itself (trivially). We ask: *"Which tokens, if
changed, would flip the meaning from self-entailment to self-contradiction?"* The gradient
tells us exactly which token embeddings the loss is most sensitive to.

**Attribution formula**:

```
    A_i  =  ‖ z_i ⊙ ∇_{z_i} L ‖₂
```

**Every symbol**:

| Symbol | Name | What it is | Shape |
|:------:|------|-----------|:-----:|
| i | Position index | Which token in the hypothesis | integer |
| z_i | Token embedding | The learned embedding vector for token i | [D] where D=1024 |
| L | Contradiction loss | CrossEntropy(logits, target=contradiction) | scalar |
| ∇_{z_i} L | Gradient | How L changes when z_i changes (partial derivatives) | [D] |
| ⊙ | Hadamard product | Element-wise multiplication of two vectors | [D] |
| ‖·‖₂ | L2 norm | sqrt( Σ_d  (z_i[d] · ∇L[d])² ) | scalar |
| A_i | Attribution | Importance of token i to the meaning | scalar ≥ 0 |

**Intuition for each piece**:

| Component | What it captures |
|-----------|-----------------|
| z_i (embedding magnitude) | Large embedding = token has a strong "signal" in the model |
| ∇_{z_i} L (gradient magnitude) | Large gradient = loss is very sensitive to this token |
| z_i ⊙ ∇_{z_i} L (product) | Both conditions simultaneously: strong token AND loss cares |
| ‖·‖₂ (L2 norm) | Collapses the D-dimensional product into a single scalar score |

**Normalization** (server-side):

```
    A_i  ←  A_i / max(A₁, A₂, ..., A_T)

    so all attributions are in [0, 1], with the most important token at 1.0
```

**Word boundary filter**: Only tokens at word boundaries are eligible for substitution:
- Tokens starting with `Ġ` (GPT-style BPE word-initial marker)
- Tokens starting with `▁` (SentencePiece word-initial marker)
- The very first token

This avoids substituting sub-word fragments like `##tion` or `ruct`.

---

### Step 2: Substitution Score — Score S_ij

**Goal**: For each high-attribution position i, find which replacement tokens j from the
vocabulary would shift meaning most in the direction the gradient points.

**Formula**:

```
              (z_i − z_j)  ·  ∇_{z_i} L
    S_ij  =  ─────────────────────────────
              ‖z_i − z_j‖₂  ·  ‖∇_{z_i} L‖₂
```

**Every symbol**:

| Symbol | Name | What it is | Shape |
|:------:|------|-----------|:-----:|
| i | Original position | Token position being substituted | integer |
| j | Replacement token | A candidate from the full vocabulary V | integer |
| z_i | Original embedding | Embedding of the token currently at position i | [D] |
| z_j | Replacement embedding | Embedding of candidate replacement token j | [D] |
| z_i − z_j | Difference vector | How much the embedding changes if we substitute | [D] |
| ∇_{z_i} L | Gradient at position i | Direction that increases contradiction loss | [D] |
| · | Dot product | Σ_d (z_i−z_j)[d] · ∇L[d] | scalar |
| ‖z_i − z_j‖₂ | Norm of difference | Magnitude of the embedding change | scalar |
| ‖∇_{z_i} L‖₂ | Norm of gradient | Magnitude of the gradient | scalar |
| S_ij | Substitution score | Cosine similarity between change direction and gradient | scalar ∈ [-1, 1] |

**Interpretation step by step**:

```
  Numerator:   (z_i − z_j) · ∇_{z_i} L
               ─────────────────────────
               "How much does replacing token i with token j
                move the embedding in the direction that
                INCREASES the contradiction loss?"

  Denominator: ‖z_i − z_j‖₂ · ‖∇_{z_i} L‖₂
               ────────────────────────────────
               Normalizes to [-1, 1] (cosine similarity)
               so the score is independent of vector magnitudes

  S_ij ≈ +1:  Substitution moves perfectly along the gradient
               → maximally changes the meaning

  S_ij ≈  0:  Substitution is orthogonal to the gradient
               → changes something, but not the meaning

  S_ij ≈ -1:  Substitution moves against the gradient
               → reinforces the original meaning (bad substitute)
```

**Computed server-side** (to avoid transferring the [V × D] embedding matrix):

```python
emb_matrix = nli_model.get_embedding_matrix()    # [V, D]  (V ≈ 128,000 for DeBERTa)

diff       = z_i.unsqueeze(0) - emb_matrix       # [V, D]  difference for every vocab token
diff_norms = diff.norm(dim=-1).clamp(min=1e-8)   # [V]     norm of each difference
s_ij       = (diff @ grad_i) / (diff_norms * grad_norm)  # [V]  substitution score for all j

s_ij[original_token_id] = -1.0                    # exclude self-substitution
```

Then take the **top-K** (default K=20) replacements ranked by S_ij.

---

### Step 3: Importance Score — Score I_ij

**Goal**: Ensure the substitution is **likely under the LLM** — semantically disruptive
AND contextually plausible.

**Formula**:

```
    I_ij  =  p( v_j  |  y_{<i},  x,  w )
```

**Every symbol**:

| Symbol | Name | What it is |
|:------:|------|-----------|
| v_j | Replacement token | The candidate substitute token j |
| y_{<i} | Preceding text | All tokens before position i in the current text |
| x | Conversation context | The full message history (system prompt, prior turns) |
| w | LLM weights | The parameters of Qwen3-Coder-30B-A3B |
| p(·) | LLM probability | The model's next-token probability distribution |
| I_ij | Importance score | How likely the LLM thinks token j is at position i |

**How it's computed**:

```
1. Build the prefix: all text up to (but not including) position i

2. Send to vLLM completions endpoint:
   POST /v1/completions
   {
     "prompt": prefix,
     "max_tokens": 1,
     "logprobs": 20,        ← return top-20 tokens with log-probabilities
     "temperature": 0
   }

3. Response includes:
   { "the": -0.52, "a": -1.23, "compose": -2.10, "remove": -0.15, ... }

4. Convert log-probabilities to probabilities:
   I_ij = exp(logprob_j)

   e.g., logprob = -2.10  →  I_ij = exp(-2.10) = 0.122
```

**Interpretation**:

| I_ij value | Meaning |
|:----------:|---------|
| ~0.5–1.0 | LLM strongly expects this token here — very plausible substitute |
| ~0.05–0.2 | LLM considers it a reasonable alternative |
| ~0.001 | LLM thinks this token is very unlikely here — poor substitute |

**Why it matters**: Without I_ij, SDLG might substitute "remove" with "xylophone" (high S_ij
because the embeddings are far apart, but nonsensical in context). I_ij filters for tokens
the LLM would actually generate.

---

### Step 4: Combined Score and Ranking

The three scores are combined as a **simple average**:

```
                    A_i  +  S_ij  +  I_ij
    Combined_ij  =  ─────────────────────
                             3
```

**Summary of the three scores**:

| Score | Range | What it measures | Source |
|:-----:|:-----:|-----------------|--------|
| A_i | [0, 1] | How important is position i to the meaning? | DeBERTa gradients (normalized) |
| S_ij | [-1, 1] | How much does replacing with j shift meaning toward contradiction? | DeBERTa embeddings + gradients |
| I_ij | [0, 1] | How likely is token j at this position under the LLM? | vLLM logprobs |

**Ranking**:

All (position i, replacement j) pairs are sorted by Combined_ij **descending**.

```
Rank 1: position=7  "remove" → "compose"   A=0.92  S=0.81  I=0.12  Combined=0.617
Rank 2: position=7  "remove" → "extend"    A=0.92  S=0.73  I=0.08  Combined=0.577
Rank 3: position=3  "check"  → "merge"     A=0.78  S=0.65  I=0.15  Combined=0.527
...
```

Note: multiple substitutions at the **same position** are allowed (different replacements at
a high-attribution position produce genuinely different completions).

---

### Step 5: Generate Alternatives via Prefix Truncation

For each of the top-ranked substitutions, generate an alternative response:

---

#### THOUGHT-Level Substitution (strategic diversity)

Produces **fundamentally different approaches** by changing the reasoning.

```
GREEDY RESPONSE:
┌─────────────────────────────────────────────────────────────┐
│ "The bug is in the constructor. I should remove the         │
│  duplicate check and instead validate at the call site..."  │
│                                           ↑                 │
│                                      position i             │
├─────────────────────────────────────────────────────────────┤
│ ```mswea_bash_command                                       │
│ sed -i 's/old/new/' file.py                                 │
│ ```                                                         │
└─────────────────────────────────────────────────────────────┘

STEP 1 — Substitute at position i:
    "remove" → "compose"

STEP 2 — Truncate everything after substitution point:
    prefix = "The bug is in the constructor. I should compose"

STEP 3 — Send as assistant prefix, LLM completes:
    messages = [...history..., {"role": "assistant", "content": prefix}]
    → LLM generates: "...the cycles before validation, ensuring
       disjointness is checked during composition rather than..."
    → LLM also generates a NEW code block (different sed command)

RESULT:
┌─────────────────────────────────────────────────────────────┐
│ "The bug is in the constructor. I should compose the cycles │
│  before validation, ensuring disjointness is checked..."    │
├─────────────────────────────────────────────────────────────┤
│ ```mswea_bash_command                                       │
│ sed -i 's/different_old/different_new/' file.py             │  ← DIFFERENT code
│ ```                                                         │
└─────────────────────────────────────────────────────────────┘
```

Key: **both the reasoning AND the code change**, because the LLM regenerates everything
from the substitution point onward.

---

#### CODE-Level Substitution (tactical diversity)

Produces **different implementations of the same approach**.

```
GREEDY RESPONSE:
┌─────────────────────────────────────────────────────────────┐
│ "The bug is in the constructor. I should remove the         │
│  duplicate check..."    (KEPT UNCHANGED)                    │
├─────────────────────────────────────────────────────────────┤
│ ```mswea_bash_command                                       │
│ sed -i 's/if not is_disjoint/if len(set...' file.py        │
│                    ↑                                        │
│               position i                                    │
│ ```                                                         │
└─────────────────────────────────────────────────────────────┘

STEP 1 — Substitute at position i in the CODE:
    "is_disjoint" → "overlap"

STEP 2 — Truncate code after substitution, keep thought prefix:
    prefix = thought + "```mswea_bash_command\nsed -i 's/if not overlap"

STEP 3 — LLM completes the code:
    → "...in cycles: raise ValueError(...)' file.py\n```"

RESULT: Same reasoning, different code patch.
```

---

#### Budget Split

```
N candidates total (default N=5):
    - 1 is the greedy response (always kept)
    - ceil((N-1)/2) from THOUGHT-level SDLG    (e.g., 2 for N=5)
    - floor((N-1)/2) from CODE-level SDLG      (e.g., 2 for N=5)
```

This ensures both:
- **Strategic diversity**: different reasoning → different fix approaches
- **Tactical diversity**: same reasoning → different implementations

---

### Fallback

If SDLG produces zero alternatives (text too short, NLI server down, no valid substitutions),
falls back to **temperature sampling** at T=0.7 — simply re-queries the LLM multiple times
with randomness.
