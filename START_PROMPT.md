# Project Kickoff Prompt

Paste everything below the line into a new Claude Code chat.

---

I'm implementing a research project described in `CLAUDE.md` — read it first, thoroughly. It has everything: the project overview, system details, all reference PDFs and what each contains, hardware specs, Docker/vLLM setup status, and the 10 SWE-bench instances we're targeting.

The project has two phases: **Phase 1 (baseline)** and **Phase 2 (semantic branching)**. We do Phase 1 first, end to end, before touching Phase 2.

## Phase 1: Baseline (single-trajectory, minimal tooling)

### Step 1 — Download the model

Download **Qwen3-Coder-30B-A3B** quantized to **4-bit (AWQ or GPTQ — whichever vLLM supports best)**. The model needs to be stored on D: drive (C: only has 52 GB free). We have a working `vllm/vllm-openai:latest` Docker image (vLLM 0.11.0, PyTorch 2.8.0+cu128) with GPU passthrough confirmed on our RTX 5090 (32 GB VRAM).

Figure out: where to download the quantized model from (HuggingFace), how to mount it into the vLLM Docker container, and how to serve it with the OpenAI-compatible API.

### Step 2 — Verify inference

Write a quick smoke test that hits the vLLM server with a simple coding prompt and confirms we get reasonable output, token probabilities are accessible, and VRAM usage is within budget.

### Step 3 — Set up SWE-bench evaluation

We need to run our 10 SymPy instances from SWE-bench Verified. The instances are listed in the table at the bottom of `CLAUDE.md`. We already have 75 sympy Docker eval images pulled (`ghcr.io/epoch-research/swe-bench.eval.x86_64.sympy__sympy-*`). We also have `sweb.base.py.x86_64` and `sweb.env.py.x86_64.*` images.

Set up the SWE-bench harness to run instances using the **minimal-tooling setting used on the SWE-bench leaderboard** (bash-only tool interface — the agent gets the issue text, can run bash commands to explore the repo, edit files, and submit a patch). This matches the setup described in Section 2.1 of our proposal (`PDFs/proposal.pdf`).

The agent should use our locally-served Qwen3-Coder-30B-A3B via the vLLM OpenAI-compatible API (not a cloud API).

### Step 4 — Run baseline and log everything

For each of the 10 instances:
1. Run the agent in a single trajectory (pass@1) with the minimal tooling
2. **Log the full trajectory** — every agent message, every tool call, every observation — to a structured file. Organize as `results/baseline/{instance_id}/trajectory.jsonl` (or similar structured format)
3. After the trajectory completes, evaluate the patch against the instance's unit tests
4. Record **fail-to-pass (f2p)** and **pass-to-pass (p2p)** results per instance
5. Write a summary to `results/baseline/summary.json` with per-instance pass/fail and aggregate pass@1

### Step 5 — Baseline results

Produce a clean results table showing:
- Instance ID, difficulty estimate, pass/fail, f2p test results, p2p test results, number of agent steps, tokens used

This is our baseline. Save everything — we need to compare against Phase 2.

## Phase 2: Semantic Branching (diverse-pass@1)

Only start this after Phase 1 is fully working and we have baseline results.

Implement the semantic branching method from the proposal on top of the same minimal-tooling agent. The three core components, with their reference papers (all in `PDFs/`):

1. **SDLG diverse generation** — Reference: `PDFs/Aichberger_2025_SDLG.pdf`. At each agent step, generate N candidate responses. Use gradient-based attribution through DeBERTa NLI to identify high-impact tokens in reasoning traces, substitute them, and have the LLM complete from the substitution point. Substitutions apply ONLY to reasoning traces, NOT action tokens (bash commands). Extract a one-sentence intent summary from each candidate via a separate model call.

2. **Semantic clustering** — Reference: `PDFs/farquhar_nature.pdf` (Extended Data Fig. 1) and `PDFs/Farquhar_2024_Semantic_Entropy.pdf` (Algorithm 1). Cluster the N candidates by shared meaning using bidirectional entailment on the extracted intent summaries. Use DeBERTa-large fine-tuned on MNLI. Two candidates share a cluster if their summaries mutually entail each other.

3. **Adaptive branching** — Compute semantic entropy over the cluster distribution. If entropy > threshold τ, branch: each cluster spawns an independent trajectory with its own conversation history. If entropy ≤ τ, take the greedy action (no branching). Hard cap B=30 trajectories per problem.

For reference on how a similar system works at training time (not directly implemented, but useful for the hybrid selector and trajectory evaluation patterns), see `PDFs/EntroPO.pdf`.

### Phase 2 evaluation

Run the same 10 instances with semantic branching. Log every trajectory (there will be multiple per instance now). Report **diverse-pass@1** (did ANY branched trajectory produce a passing patch?). Compare against baseline pass@1. Report average trajectories spawned and branching events per problem.

Save results to `results/branching/` with the same structure.

## Important notes

- **Read `CLAUDE.md` first** — it has system specs, Docker status, model details, and all PDF locations
- **Read the PDFs** using the `mcp__file-system-windows-python__read-file` tool when you need implementation details from a specific paper
- **Docker is the runtime** for both vLLM inference and SWE-bench evaluation
- **D: drive** for all large files (models, results, logs). Project is at `d:/Projects/PhD_Class_2026/Gen AI/Project/`
- Work incrementally — get each step working before moving to the next
- Ask me before making architectural decisions that are ambiguous
