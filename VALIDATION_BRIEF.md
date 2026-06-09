# Validation Brief: Search-step + Trajectory-capture + Diversity-arm + Clustering-strategy changes

A self-contained validation brief you can hand to another model. It assumes no access to the originating conversation.

## Repository context

Research project implementing semantic branching for agentic code generation on SWE-bench Verified (SymPy instances). The active orchestrator is `src/agent/phased_orchestrator.py` (`PhasedOrchestrator`), which runs phases: SEARCH (read-only exploration, single trajectory) → STRATEGY PROPOSAL (cluster K proposed strategies, fork one trajectory per cluster) → PATCH → VERIFY. SDLG is an alternative diversity generator that forks child trajectories at the first write command. Trajectories run sequentially (one Docker container at a time); each container is destroyed (`cleanup()`) after its trajectory finishes.

Config: `configs/branching.yaml`. Eval driver: `scripts/run_branching.py`.

## Change 1 — Raise the search-phase step cap

File: `configs/branching.yaml`, key `branching.max_search_steps`: 30 → 240.

Why: The search phase is normally ended early by a saturation detector (N consecutive low-relevance steps). `max_search_steps` is only a hard fallback cap for when saturation never trips (e.g. the agent loops on blocked write commands during SEARCH, which aren't relevance-scored). The previous value of 30 cut off exploration prematurely in those cases. The mini-SWE-agent baseline uses a total `step_limit` of 250; the new search cap (240) sits just below that, so the search fallback is generous but still bounded under the baseline's whole-task budget.

Validate:

- `branching.max_search_steps == 240` and is read at `phased_orchestrator.py:108` into `self.max_search_steps`, enforced at the SEARCH loop hard cap (search loop checks `root.step >= self.max_search_steps`).
- This is a fallback; normal runs still exit search via saturation (`min_search_steps`, `low_relevance_streak`). So well-behaved runs are unaffected; only pathological/long searches get more room.

## Change 2 — Capture trajectory patches before container teardown (data-loss bug fix)

File: `src/agent/phased_orchestrator.py`.

Problem found: A trajectory's `traj.patch` was only populated on two completion paths:

- `Submitted` exception → `traj.patch = exception submission`
- `LimitsExceeded` exception → `traj.patch = git diff`

But the run loop has a third completion path: `if traj.step > self.max_steps: traj.status = "completed"` — which sets status to completed without capturing any diff. Trajectories that instead stop on a format error, or write a "## Summary" rather than the exact `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` command, also end with an empty `traj.patch`. Immediately after, `cleanup()` destroys the container, so the working-tree diff is permanently lost.

Downstream impact: `_collect_results()` builds the eval prediction set from completed trajectories with a non-empty `traj.patch`. So these branches were saved to disk as `.traj.json` (transcripts) but silently dropped from `predictions_all_trajectories.jsonl`, undercounting diverse-pass@1.

Empirical confirmation (instance `sympy__sympy-15345`): 11 trajectories saved, but only 8 patches in the prediction set. The 3 missing branches (`t0_strategy_1`, `t0_strategy_3_sdlg_2`, `t0_strategy_4_sdlg_1`) each had transcripts showing a verified-working fix (e.g. "square brackets: True, No longer uses parentheses: True"), yet contributed no patch. Same shape on `sympy__sympy-12481` (6 patches from 9 branches).

Fix: New method `_capture_patch_if_missing(traj)` — if `traj.patch` is empty, run `cd /testbed && git diff --no-color` while the container is still alive and store the result on `traj.patch` (no-op if a patch already exists; logged via the tracer as `trajectory.patch_captured_fallback`). It is called at both trajectory-teardown sites (the strategy loop and the SDLG-child loop), positioned before `save_all()` and before `cleanup()`. This makes container teardown the single chokepoint where a diff is always captured, covering all three completion paths at once.

Validate:

- `_capture_patch_if_missing` is invoked before every `traj.cleanup()` / `sdlg_traj.cleanup()` in `run()`, and before `save_all()` so per-instance `metadata.json` reflects the captured patch.
- It is a no-op when `traj.patch` is already set → existing `Submitted`/`LimitsExceeded` behavior is unchanged.
- `cleanup()` only does `docker rm` + nulls `container_id`; the in-memory `Trajectory` object (and its now-populated `.patch`) persists in `manager.trajectories`, so `_collect_results()` (run at the end) sees it.
- Failure-safe: git-diff capture is wrapped in try/except and logs a warning on error; it never raises into the run loop.

## Change 3 — Raise per-trajectory step cap (consequence of Change 1)

File: `configs/branching.yaml`, key `agent.step_limit`: 250 → 300.

Why: `self.max_steps` (= `agent.step_limit`) bounds each trajectory's total `traj.step`. The root trajectory `t0` and any SDLG children inherit the search step count (SDLG children are created with `step=parent.step`; `t0` continues counting from search). Lazy `t0_strategy_N` trajectories reset to `step=0`. With `max_search_steps` now 240 and `step_limit` still 250, `t0` and its SDLG children would enter PATCH already near the cap and trip the run-loop step limit (the no-capture path from Change 2) before making edits. Raising `step_limit` to 300 leaves them ~60 real patch/verify steps. This change and Change 2 are complementary: Change 1 made the latent leak bite much harder, Change 3 prevents it from biting, Change 2 makes any remaining trips non-destructive.

Validate:

- `agent.step_limit == 300`, read at `phased_orchestrator.py:107` into `self.max_steps`.
- Asymmetry is intended: lazy strategy trajectories (start at step 0) can now use up to 300 patch steps vs. the baseline's 250 total. In practice they submit far earlier, so diverse-pass@1 is unlikely to be skewed — but this is the one fairness caveat a reviewer should sanity-check (consider noting it in the writeup).

## Change 4 — Make `diversity_method` a single mutually-exclusive switch (de-confound the two generators)

Files: `src/agent/phased_orchestrator.py`, `configs/branching.yaml`.

Problem found: The two diversity generators were not alternatives — they were **stacked**. Phase 2 (STRATEGY PROPOSAL) forked one trajectory per strategy cluster, and then SDLG independently forked child trajectories at the first write command *inside each of those strategy trajectories*. A single run produced a tree of `strategy_i × sdlg_j` branches from two different generators feeding the same clustering core. Worse, the config exposed `diversity_method: "strategy_proposal"` and `sdlg_enabled: true` as two independent keys, so the "active arm" was ambiguous and a stale config could silently co-activate both. This makes diverse-pass@1 unattributable: you cannot tell whether a passing branch came from SDLG (the claimed token-attribution contribution) or from simply prompting the model to list K strategies. For publication, the generator must be isolated per run.

Fix: `diversity_method` is now the **single source of truth**, read once in `PhasedOrchestrator.__init__`. `sdlg_enabled` and `use_strategy_proposal` are **derived** from it (mutually exclusive), validated against the allowed set, and logged at startup as the active arm. The standalone `sdlg_enabled` config key was removed so it can never be set independently. Phase 2 strategy proposal in `run()` is now gated on `self.use_strategy_proposal`; in the `sdlg` arm it is skipped and the single root trajectory falls through to PATCH, where SDLG forks at the first write (unchanged mechanism). The two arms now share the same SEARCH → cluster → entropy → branch core but differ only in the candidate generator.

- `"strategy_proposal"` arm: SEARCH → propose K strategies → cluster → fork per cluster. SDLG OFF.
- `"sdlg"` arm: SEARCH → single trajectory → SDLG forks at first write command. No strategy proposal.

Validate:

- `__init__` reads `branching_config["diversity_method"]` (default `"strategy_proposal"`), raises `ValueError` on any value outside `{"strategy_proposal", "sdlg"}`, and sets `self.use_strategy_proposal = (method == "strategy_proposal")` and `self.sdlg_enabled = (method == "sdlg")`. The two booleans are always complementary — there is no code path where both are true.
- In `run()`, the STRATEGY PROPOSAL block (`_propose_strategies`) executes **only** under `if root.status == "active" and self.use_strategy_proposal:`. The `elif` branch logs the SDLG-arm single-trajectory path and proposes nothing, so `unique_strategies` stays empty and hits the existing single-strategy fallback (`["Fix the bug as described..."]`).
- All three `self.sdlg_enabled` guard sites (`_step_patch` SDLG check, the PATCH-phase fork trigger, and the VERIFY-phase fallback fork trigger) are now driven by the derived flag, so SDLG cannot fire in the `strategy_proposal` arm.
- `configs/branching.yaml` no longer contains a standalone `sdlg_enabled:` key; `grep -n sdlg_enabled` over `src/**/*.py` returns only the derivation site and the three guard reads in `phased_orchestrator.py` (no config read).
- Arm independence from Changes 1–3: the search-step cap, step cap, and `_capture_patch_if_missing` teardown capture all operate per trajectory regardless of which generator produced it, so this change does not interact with them.

## Change 5 — Make the semantic clustering method a selectable strategy

Files: `src/diversity/clustering.py`, `src/agent/phased_orchestrator.py`, `src/agent/branching_orchestrator.py`, `scripts/run_branching.py`.

Problem found: Clustering was hard-coded to a single algorithm — online greedy single-pass clustering against each cluster's *first* element (Farquhar et al. 2024 / Kuhn et al. 2023, Algorithm 1). This has three structural weaknesses that directly affect the branch decision: (1) **order-dependence** — permuting the candidates can change the clusters, hence the entropy, hence branch/no-branch; (2) **representative-only comparison** — each new intent is compared only to a cluster's first element and assigned on the first match (`break`), so it neither uses best-match nor checks agreement with the rest of the cluster; (3) **hard binarization** — `fwd > thr and bwd > thr` discards the continuous NLI signal, so a 0.51/0.51 match and a 0.99/0.99 match are identical and a 0.49 near-miss spawns a whole new branch. None of these were configurable, so the method couldn't be ablated.

Fix: `SemanticClusterer` now takes a `strategy` argument (`"greedy" | "connected" | "kernel"`, default `"greedy"`) plus `kernel_t` (default 1.0), validated against `VALID_STRATEGIES` in `__init__`.

- `"greedy"` (A): the original Algorithm 1, refactored into `_cluster_greedy` with identical behavior. The baseline.
- `"connected"` (B): `_pairwise_entailment` computes the full N×N entailment matrix in one `classify_batch` call; `_cluster_connected` unions i,j (union-find) iff they bidirectionally entail (`fwd_ent[i,j] > thr and fwd_ent[j,i] > thr`) and returns connected components. Order-independent (transitive closure), uses all pairs, lowest index kept as representative. Reduces to the greedy result when the entailment relation is cleanly transitive.
- `"kernel"` (G): clusters via connected components (so branching still has discrete groups to fork from), but the branch-decision signal is `compute_kernel_entropy` — the von Neumann entropy of the symmetric entailment-affinity Gram matrix (Kernel Language Entropy, Nikitin et al. 2024). The kernel is symmetrized, diagonal set to 1.0, projected to PSD by clipping negative eigenvalues, normalized by trace; `S = -Σ λ_i log λ_i`. `kernel_t` is a temperature on the spectrum (`λ^(1/t)`). In the clean block-diagonal limit this *exactly* recovers the count-based semantic entropy; it degrades gracefully under fuzzy overlap and stays near zero for near-duplicates.

`analyze()` dispatches: `"greedy"`/`"connected"` use the count-based `compute_entropy`; `"kernel"` uses `compute_kernel_entropy`. Return dict gained a `"strategy"` key. Both orchestrators construct the clusterer with `strategy=branching_config.get("clustering_strategy", "greedy")` and `kernel_t=branching_config.get("kernel_t", 1.0)`. `scripts/run_branching.py` adds `--clustering-strategy {greedy,connected,kernel}` and `--kernel-t`, which override the config by injecting into `config["branching"]`. Default stays `"greedy"`, so existing runs are byte-for-byte unaffected.

Validate:

- `SemanticClusterer.__init__` raises `ValueError` for any `strategy` outside `{"greedy","connected","kernel"}`; `cluster()` returns the greedy result only when `strategy == "greedy"`, else routes through `_pairwise_entailment` + `_cluster_connected`.
- `_pairwise_entailment` builds an N×N matrix with diagonal 1.0 from a single `classify_batch` over all ordered i≠j pairs; `sym_aff = min(fwd, fwd.T)` with zero diagonal. NLI input is `f"{context} {intent}"` per Algorithm 1 — same context-prepending as the greedy path.
- Order-independence of `"connected"`: shuffling the input intents yields the same clusters (up to cluster ordering) and the same entropy. The greedy path does **not** have this property — that asymmetry is the point of the ablation.
- `compute_kernel_entropy` ordering sanity (mock NLI, threshold 0.5): all-identical < grouped < all-distinct (e.g. ≈0.39 < 1.21 < 1.37). It is on the same ~[0, log N] nat scale as the count-based entropy but **not numerically identical** for non-block-diagonal affinities.
- Defaults preserved: with no `clustering_strategy` set anywhere, both orchestrators run `"greedy"`; the CLI flag and config key both override, CLI winning.

Caveat for τ-sweeps: the kernel entropy is on a comparable scale to the count-based value but not equal, so the branching threshold `tau` is **not directly transferable** between `"kernel"` and the other two arms and must be recalibrated per strategy before comparing branch rates. (This compounds the existing tau-unification nit below.)

## Things a validator should specifically check

- **No double-capture / no overwrite:** confirm `_capture_patch_if_missing` never overwrites a real `Submitted` patch (guarded by `if traj.patch: return`).
- **Ordering:** capture happens before both `save_all()` and `cleanup()` at both call sites.
- **Both arms:** the patch-capture fix is independent of the diversity arm (`diversity_method` `strategy_proposal` vs `sdlg`) — it operates on every trajectory at teardown regardless.
- **No backfill:** the fix only affects future runs. Patches already lost on existing results are unrecoverable (containers gone); those instances must be re-run to capture them. The `.traj.json` transcripts show the edits but contain no clean git diff.
- **Bounds:** `max_search_steps` (240) < `step_limit` (300) must hold; if someone later raises `max_search_steps` again, `step_limit` must stay above it by a meaningful patch budget.
- **Single active generator per run:** confirm a `strategy_proposal` run logs `sdlg=False` at init and produces zero `phase3.sdlg.*` / `phase4.sdlg_trigger` tracer events; confirm an `sdlg` run logs `strategy_proposal=False` and emits no `phase2.*` strategy-proposal events. No run should show both.
- **Fairness nit (entropy gate, not yet unified):** the strategy-proposal path calls the clusterer with a hardcoded `tau=0.0`, while the SDLG path passes the configured `entropy_threshold`. Identical today because `entropy_threshold: 0.0` (both always branch), but if τ is ever raised to exercise the adaptive gate, only the SDLG arm responds. Unify both to read the same config value before reporting any τ-sweep results.
- **Results collision:** the two arms write to the same `paths.results_dir` / `predictions_file`, so running them back-to-back overwrites the first arm's predictions. Run each arm into a separate results dir (e.g. duplicate the config with distinct `paths:`) before comparing diverse-pass@1. The same applies to clustering-strategy comparisons (Change 5): each `--clustering-strategy` run must write to its own results dir.
- **Clustering-strategy default unchanged:** confirm that with no `clustering_strategy` set, runs use `"greedy"` and reproduce prior results; the `"connected"`/`"kernel"` paths add an N×N `classify_batch` per clustering call (negligible for the small N here, but it is extra NLI traffic vs. the greedy path's lazy per-pair calls).
- **Kernel τ not transferable:** if any τ-sweep compares `"kernel"` against `"greedy"`/`"connected"`, confirm τ was recalibrated per strategy — the kernel entropy is on a comparable but non-identical scale.

## Files changed

- `configs/branching.yaml` — `max_search_steps`: 30→240; `step_limit`: 250→300 (both commented); removed the standalone `sdlg_enabled` key (now derived from `diversity_method`); documented `diversity_method` as the single arm switch and annotated `n_strategies` / `sdlg_*` as arm-specific.
- `src/agent/phased_orchestrator.py` — added `_capture_patch_if_missing()`, called before `cleanup()` at the strategy-loop and SDLG-child-loop teardown sites; `diversity_method` parsed/validated in `__init__` with `sdlg_enabled` + `use_strategy_proposal` derived (mutually exclusive) and logged; Phase 2 strategy proposal in `run()` gated on `use_strategy_proposal`; clusterer constructed with `strategy` + `kernel_t` from config (Change 5).
- `src/diversity/clustering.py` — added selectable `strategy` (`greedy`/`connected`/`kernel`) and `kernel_t` to `SemanticClusterer`; new `_pairwise_entailment`, `_cluster_connected` (union-find connected components), and `compute_kernel_entropy` (von Neumann entropy of the affinity Gram kernel); `analyze()` dispatches entropy by strategy and returns a `"strategy"` key. Greedy path behavior unchanged.
- `src/agent/branching_orchestrator.py` — clusterer constructed with `strategy` + `kernel_t` from config (parity with the phased orchestrator).
- `scripts/run_branching.py` — added `--clustering-strategy {greedy,connected,kernel}` and `--kernel-t` CLI flags that override `config["branching"]`.

Syntax/config validated: `phased_orchestrator.py`, `branching_orchestrator.py`, `clustering.py`, and `run_branching.py` all parse (`py_compile`); `branching.yaml` loads with the expected values (`max_search_steps: 240`, `step_limit: 300`, `diversity_method: "strategy_proposal"`, no `sdlg_enabled` key). Clustering strategies smoke-tested with a mock NLI: `greedy`/`connected` agree on cluster structure for cleanly-separable inputs; `kernel` entropy orders all-identical < grouped < all-distinct.
