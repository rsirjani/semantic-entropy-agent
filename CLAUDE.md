# Semantic Entropy Clustering for Diverse Agentic Code Generation

## Project Overview

This is a research project for a PhD-level Gen AI course at Western University (Ramtin Sirjani & Paul Moore). We are implementing a **semantic branching** framework for multi-step agentic code generation that forces the model to explore diverse solutions at each agent step, evaluated on SWE-bench Verified.

### Core Idea

At each agent step:
1. **Diverse generation** — Generate N candidate responses using SDLG (Semantically Diverse Language Generation), which uses gradient-based attribution through a DeBERTa NLI model to find and replace high-impact tokens, producing alternatives that are both likely and meaningfully different.
2. **Semantic clustering** — Cluster candidates by meaning via bidirectional entailment on extracted intent summaries (using DeBERTa-large fine-tuned on MNLI). Each cluster = a meaningfully different course of action.
3. **Adaptive branching** — Compute semantic entropy over the cluster distribution. If entropy > threshold τ, branch each cluster into its own independent trajectory. Otherwise, take the greedy action. Hard cap of B=30 trajectories per problem.

### Target Model & Evaluation

- **Model**: Qwen3-Coder-30B-A3B, quantized to 4-bit, run locally
- Full weight access is required for token-level probabilities and gradient-based attribution
- **Evaluation**: 10 instances from SWE-bench Verified, drawn from SymPy (see Appendix C of proposal for instance list)
- **Metrics**: pass@1 (baseline) vs diverse-pass@1 (whether any branched trajectory passes)

## System Details

### Host Machine (Windows)
- **OS**: Windows 11 Home, Build 26200
- **CPU**: AMD Ryzen 9 5950X, 16 cores / 32 threads @ ~4.0 GHz
- **RAM**: 64 GB DDR4
- **GPU**: NVIDIA GeForce RTX 5090, 32 GB VRAM
- **GPU Driver**: 581.57, CUDA 13.0
- **Storage**: D: drive — 3.7 TB total, 2.1 TB free (project lives here); C: drive — 52 GB free (tight)
- **Python (Windows)**: 3.13.11 via Miniconda (conda 26.1.1)
- **Conda envs**: base, locagent, mlagents, py39, swe-py310, swe-py311, swe-py35–py38

### WSL (Ubuntu, WSL 2)
- **Status**: Running
- **GPU passthrough**: Working (nvidia-smi sees RTX 5090 from WSL)
- **CUDA toolkit**: NOT installed in WSL (nvcc not found) — will need to install or use Docker
- **Python**: 3.12.3

### Docker Desktop
- **Version**: Docker 29.2.1, Docker Desktop
- **Resources**: 32 CPUs allocated, 43.9 GB RAM allocated
- **GPU passthrough**: Working (`--gpus all` tested, nvidia-smi sees RTX 5090)
- **Existing images**:
  - `vllm/vllm-openai:latest` (38.5 GB) — vLLM 0.11.0, PyTorch 2.8.0+cu128, CUDA available, GPU detected
  - `sweb.base.py.x86_64` + multiple SWE-bench eval images (75 sympy-specific images already pulled)
  - `sweb.env.py.x86_64.*` environment images

### Inference Strategy: Docker + vLLM
- **vLLM requires Linux** — cannot run natively on Windows
- **Docker is the recommended path** (already have `vllm/vllm-openai:latest` image, GPU passthrough confirmed working)
- The vllm image runs PyTorch 2.8.0+cu128 and detects the RTX 5090 correctly
- Alternative: WSL would work but needs CUDA toolkit installed; Docker is simpler since the vllm image is self-contained
- **SWE-bench evaluation** also uses Docker (eval images already pulled), so Docker is the unified runtime for both model serving and benchmark evaluation

### Key Constraints
- **VRAM**: 32 GB — Qwen3-Coder-30B-A3B at 4-bit quantization should fit (~16-18 GB), leaving room for DeBERTa NLI model (~600 MB) to run concurrently for SDLG/clustering
- **C: drive space**: Only 52 GB free — avoid placing large models or Docker volumes on C:, use D: for everything
- **Docker image sizes**: vLLM image alone is 38.5 GB; SWE-bench images are ~3 GB each. Monitor D: space

## Reading PDFs

This project has a `file-system-windows-python` MCP server configured that can read PDFs with rendered page images + extracted text. To read any PDF:

```
Use the mcp__file-system-windows-python__read-file tool with the full path to the PDF.
```

All reference PDFs are stored in `PDFs/` subdirectory.

## Reference Papers (in PDFs/)

### Our Proposal
- **PDFs/proposal.pdf** — "Semantic Entropy Clustering of Diverse Outputs to Explore Solution Trajectories in Large Language Model Agents" (Sirjani & Moore). This IS the project spec — all methodology, evaluation plan, and instance selection is defined here.

### Core Implementation References

These are the papers whose methods we directly implement or build upon:

- **PDFs/Aichberger_2025_SDLG.pdf** — Aichberger et al., 2025. "Improving uncertainty estimation through semantically diverse language generation" (ICLR 2025, arXiv:2406.04306). **THIS IS THE KEY PAPER.** Defines SDLG: gradient-based token attribution through a DeBERTa NLI model to identify high-impact tokens, then deliberately substitutes them to produce semantically diverse yet likely responses. We use this for our diverse generation step. Details of the method are in Appendix A of the proposal.

- **PDFs/farquhar_nature.pdf** — Farquhar, Kossen, Kuhn & Gal, 2024. "Detecting hallucinations in large language models using semantic entropy" (Nature, 630:625–630). **This is the paper cited in the proposal.** Defines semantic entropy, bidirectional entailment clustering, and applies it to confabulation detection across LLaMA 2, Falcon, Mistral, and GPT-4. Introduces discrete semantic entropy (works without token probabilities). Extended Data Fig. 1 has the clustering algorithm pseudocode. Also covers paragraph-level detection via factoid decomposition. We use this for both semantic clustering and the branching decision (entropy threshold).

- **PDFs/Farquhar_2024_Semantic_Entropy.pdf** — Kuhn, Gal & Farquhar, 2023. "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation" (ICLR 2023, arXiv:2302.09664). **Companion paper** — the earlier conference version with more pedagogical detail in the main text. Algorithm 1 here is the same bidirectional entailment clustering pseudocode. Useful for understanding the method derivation.

- **PDFs/Yao_2023_ReAct.pdf** — Yao et al., 2023b. "ReAct: Synergizing reasoning and acting in language models" (ICLR 2023). The agent loop paradigm we follow — each response may contain reasoning traces, actions (bash commands), or both.

### Evaluation Framework References

- **PDFs/Jimenez_2024_SWE_Bench.pdf** — Jimenez et al., 2024. "SWE-bench: Can language models resolve real-world GitHub issues?" (ICLR 2024). Defines the benchmark we evaluate on (SWE-bench Verified subset).

- **PDFs/Yang_2024_SWE_Agent.pdf** — Yang et al., 2024. "SWE-agent: Agent-computer interfaces enable automated software engineering" (NeurIPS 2024). Demonstrates custom agent-computer interfaces for SWE-bench.

### Related Approaches (context, not directly implemented)

- **PDFs/Wang_2023_Self_Consistency.pdf** — Wang et al., 2023. "Self-consistency improves chain of thought reasoning in language models" (ICLR 2023). Self-consistency decoding — we extend these ideas to multi-step agentic trajectories.

- **PDFs/Snell_2024_Scaling_LLM_Test_Time_Compute.pdf** — Snell et al., 2024. "Scaling LLM test-time compute optimally can be more effective than scaling model parameters" (arXiv:2408.03314). Test-time compute scaling theory — motivates why diverse generation matters.

- **PDFs/Yao_2023_Tree_of_Thoughts.pdf** — Yao et al., 2023a. "Tree of Thoughts: Deliberate problem solving with large language models" (NeurIPS 2023). Tree search over reasoning paths — our approach differs by branching over semantic clusters using entropy as the criterion.

- **PDFs/Zhou_2023_LATS.pdf** — Zhou et al., 2023. "Language agent tree search unifies reasoning, acting, and planning in language models" (arXiv:2310.04406). LATS — combines planning with MCTS. Our approach uses semantic entropy instead of value-function estimates.

### Diversity & Knowledge Collapse References

- **PDFs/Wright_2025_Epistemic_Diversity_Knowledge_Collapse.pdf** — Wright et al., 2025. "Epistemic diversity and knowledge collapse in large language models" (arXiv:2510.04226). Documents knowledge collapse in LLMs — motivates our work.

- **PDFs/Wright_2024_LLM_Tropes.pdf** — Wright et al., 2024. "LLM tropes: Revealing fine-grained values and opinions in large language models" (EMNLP 2024). Documents recurring patterns in LLM outputs.

- **PDFs/Zhang_2025_NoveltyBench.pdf** — Zhang et al., 2025. "NoveltyBench: Evaluating language models for human-like diversity" (COLM 2025). Shows models generate less variety than human writers.

- **PDFs/Moore_2024_LLM_Consistency_Values.pdf** — Moore et al., 2024. "Are large language models consistent over value-laden questions?" (EMNLP 2024). Related work on value consistency and output homogeneity.

### Additional Reference (NOT in proposal)

- **PDFs/EntroPO.pdf** — "Building Coding Agents via Entropy-Enhanced Multi-Turn Preference Optimization" (Yu, Cheng, Wu, Xing; arXiv:2509.12434v3, Feb 2026). **An RL-based approach to the same problem we tackle at inference time.** EntroPO adds an entropy regularization term to DPO/KTO to prevent diversity collapse during preference learning. Achieves SOTA among open-weight models on SWE-bench (60.4% Verified, 49.7% Lite with 30B model). Key differences from our approach:
  - EntroPO is a **training-time** method (modifies the loss function); ours is an **inference-time** method (modifies the agent loop)
  - EntroPO uses parallel rollouts + hybrid best-trajectory selection; we use semantic clustering + branching
  - EntroPO formulates diversity as an entropy-regularized MDP (Eq. 2 in the paper); we measure diversity via semantic entropy over clustered outputs
  - Both address the same core problem: diversity collapse limits test-time scaling effectiveness
  - Their hybrid selector (validity checks → regression tests → verifier model → step heuristic) is a useful reference for our trajectory evaluation
  - They use Qwen3-Coder-30B-A3B — the same model we plan to use

## Key Implementation Components

1. **Agent loop** (ReAct-style): issue → reasoning + bash action → observation → repeat
2. **SDLG diverse generation**: DeBERTa NLI for gradient attribution → token substitution → LLM completes from substitution point. Substitutions apply only to reasoning traces, NOT action tokens.
3. **Intent extraction**: For each candidate, extract a one-sentence intent summary via a separate model call
4. **Semantic clustering**: Bidirectional entailment on intent summaries using DeBERTa-large (MNLI)
5. **Semantic entropy**: Compute entropy over cluster distribution → compare to threshold τ
6. **Branching**: If entropy > τ, each cluster spawns an independent trajectory with its own conversation history
7. **Evaluation**: Run each completed trajectory's patch against SWE-bench unit tests

## SWE-bench Instances (from proposal Appendix C)

| Instance ID | Est. Difficulty |
|---|---|
| sympy__sympy-12481 | <15 min |
| sympy__sympy-16766 | <15 min |
| sympy__sympy-18189 | <15 min |
| sympy__sympy-12096 | <15 min |
| sympy__sympy-15345 | <15 min |
| sympy__sympy-23534 | <15 min |
| sympy__sympy-22714 | <15 min |
| sympy__sympy-19637 | <15 min |
| sympy__sympy-18763 | <15 min |
| sympy__sympy-19495 | <15 min |
