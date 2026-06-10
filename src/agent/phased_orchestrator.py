"""Phased Branching Orchestrator — Strategy-Level Diversity.

Manages three phases:

Phase 1 — SEARCH: Read-only exploration (single trajectory).
    Agent explores codebase, finds relevant files, reproduces bug.

Phase 2 — STRATEGY PROPOSAL: After search, propose K different fix
    strategies, cluster them, fork one trajectory per unique strategy.

Phase 3 — PATCH: Write access, no further branching. Each trajectory
    independently implements its assigned strategy.

Phase 4 — VERIFY: Read-only, no branching. Test and submit.

Key insight: diversity comes from explicit strategy proposals (ToT-inspired),
not token-level perturbation (SDLG), which doesn't work for multi-step agents.
"""

import copy
import json
import logging
import os
import time

from minisweagent.exceptions import (
    InterruptAgentFlow, Submitted, LimitsExceeded, FormatError,
)
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

from src.agent.branching_agent import BranchingAgent
from src.agent.branching_defaults import VALID_DIVERSITY_METHODS, cfg
from src.agent.phases import (
    Phase, SEARCH_PROMPT, PATCH_PROMPT, PATCH_PROMPT_WITH_STRATEGY,
    PATCH_FORCE_WRITE_MSG, VERIFY_PROMPT,
    detect_phase_transition, is_command_allowed, is_write_command,
    should_end_search,
)
from src.agent.trajectory import Trajectory, TrajectoryManager
from src.diversity.clustering import SemanticClusterer
from src.diversity.intent import IntentExtractor, extract_thought
from src.diversity.nli import NLIModel
from src.diversity.relevance import RelevanceScorer
from src.diversity.sdlg import SDLGGenerator
from src.diversity.strategy_proposer import StrategyProposer
from src.utils.tracer import PipelineTracer

logger = logging.getLogger(__name__)


def collect_patch_entries(trajectories, strategies_by_tid: dict) -> list[dict]:
    """One patch entry per GENUINE DRAW — every trajectory that consumed agent
    budget (status 'completed' OR 'failed'), in creation order, with patch ""
    when none was captured.

    Metric-correctness contract (R4.1/R7.2, predictions-record completeness):
    the per-trajectory predictions file is built from these entries, and the
    eval record's (n, c) is built from the predictions — so a draw that is
    dropped HERE vanishes from the arm's matched-k denominator. The vanilla
    resample driver writes an empty-patch row for every failed/unproductive
    resample; dropping the treatment's failed/patchless trajectories (the old
    behavior: only completed trajectories with non-empty patches) would
    asymmetrically deflate the treatment's k and inflate its diverse-pass@k*
    and rarefied-distinct numbers at metric time. It also silently discarded
    real patches captured on 'failed' trajectories (silent data loss, R7.2).

    Excluded statuses: 'active' (an interrupted run — not a finished draw; the
    run is incomplete and must be re-run, not scored) and 'branched' (a legacy
    branching_orchestrator parent whose budget continues in its children).
    """
    entries = []
    for traj in trajectories:
        if traj.status not in ("completed", "failed"):
            continue
        entries.append({
            "trajectory_id": traj.trajectory_id,
            "patch": traj.patch or "",
            "submitted": traj.submitted,
            "status": traj.status,
            "steps": traj.step,
            "parent_id": traj.parent_id,
            "strategy": strategies_by_tid.get(traj.trajectory_id, ""),
            "branch_info": traj.branch_info,
        })
    return entries


class PhasedOrchestrator:
    """Orchestrates multi-trajectory agent with strategy-level branching."""

    def __init__(
        self,
        instance_id: str,
        problem_statement: str,
        agent_config: dict,
        model_config: dict,
        env_config: dict,
        branching_config: dict,
        nli_model: NLIModel,
    ):
        self.instance_id = instance_id
        self.problem_statement = problem_statement
        self.agent_config = agent_config
        self.model_config = model_config
        self.env_config = env_config
        self.branching_config = branching_config

        # Shared components
        self.nli = nli_model
        self.clusterer = SemanticClusterer(
            nli=nli_model,
            entailment_threshold=cfg(branching_config, "entailment_threshold"),
            strategy=cfg(branching_config, "clustering_strategy"),
            kernel_t=cfg(branching_config, "kernel_t"),
        )
        self.intent_extractor = IntentExtractor(
            model_name=model_config.get("model_name", "openai/qwen3-coder"),
            model_kwargs=model_config.get("model_kwargs", {}),
            method="llm",
        )
        self.relevance_scorer = RelevanceScorer(
            nli=nli_model,
            threshold=cfg(branching_config, "relevance_threshold"),
            model_name=model_config.get("model_name", "openai/qwen3-coder"),
            model_kwargs=model_config.get("model_kwargs", {}),
            use_nli=cfg(branching_config, "relevance_use_nli"),
        )

        # Shared sampling temperature (proposer + vanilla resample baseline use
        # the SAME value, so the arms differ only in the branching mechanism).
        self.sample_temperature = cfg(branching_config, "sample_temperature")

        # Strategy proposer
        self.proposer = StrategyProposer(
            model_name=model_config.get("model_name", "openai/qwen3-coder"),
            model_kwargs=model_config.get("model_kwargs", {}),
            n_strategies=cfg(branching_config, "n_strategies"),
            temperature=self.sample_temperature,
        )

        # SDLG generator for diverse implementations within each strategy
        self.sdlg = SDLGGenerator(
            nli_model=nli_model,
            n_candidates=cfg(branching_config, "sdlg_n_alternatives"),
            top_k_substitutes=cfg(branching_config, "sdlg_top_k"),
            # Reasoning-only by default (R1.1); code-level substitution is an
            # explicit, ablatable opt-in.
            diversify_code=cfg(branching_config, "sdlg_diversify_code"),
        )

        # Parameters
        self.max_trajectories = cfg(branching_config, "max_trajectories")
        self.max_steps = agent_config.get("step_limit", 250)
        self.max_search_steps = cfg(branching_config, "max_search_steps")
        self.min_search_steps = cfg(branching_config, "min_search_steps")
        self.patch_read_budget = cfg(branching_config, "patch_read_budget")

        # Diversity method is a SINGLE switch — the arms are mutually exclusive
        # so results can be attributed to exactly one generator:
        #   "strategy_proposal": Phase 2 proposes K strategies and branches over
        #       their clusters. SDLG is OFF.
        #   "sdlg": single trajectory after search; SDLG branches at the first
        #       write command. No strategy proposal.
        #   "none": single trajectory, no proposal, no SDLG. This is the vanilla
        #       agent used by the matched-k resample baseline — run k times at
        #       sample_temperature to measure mode collapse against branching.
        # sdlg_enabled is DERIVED from this, never set independently, so a run
        # can never stack both generators.
        self.diversity_method = cfg(branching_config, "diversity_method")
        if self.diversity_method not in VALID_DIVERSITY_METHODS:
            raise ValueError(
                f"diversity_method must be one of {VALID_DIVERSITY_METHODS}, "
                f"got {self.diversity_method!r}"
            )
        self.use_strategy_proposal = self.diversity_method == "strategy_proposal"
        self.sdlg_enabled = self.diversity_method == "sdlg"
        logger.info(
            f"Diversity arm: {self.diversity_method} "
            f"(strategy_proposal={self.use_strategy_proposal}, "
            f"sdlg={self.sdlg_enabled}, sample_temperature={self.sample_temperature})"
        )

        # Trajectory manager
        self.manager = TrajectoryManager(
            instance_id=instance_id,
            max_trajectories=self.max_trajectories,
            results_dir=branching_config.get("results_dir", "results/branching"),
        )

        # Pipeline tracer — meticulous I/O logging for every operation
        trace_path = os.path.join(
            branching_config.get("results_dir", "results/branching"),
            "trace.jsonl",
        )
        self.tracer = PipelineTracer(trace_path)
        self.tracer.log(
            "pipeline.init",
            input={
                "instance_id": instance_id,
                "problem_statement": problem_statement[:500],
                "agent_config_keys": list(agent_config.keys()),
                "model_name": model_config.get("model_name"),
                "branching_config": branching_config,
            },
            output={"status": "initialized"},
        )

        # Per-trajectory phase tracking
        self.trajectory_phases: dict[str, Phase] = {}
        # Strategy assigned to each trajectory (for logging)
        self.trajectory_strategies: dict[str, str] = {}

        # Search phase: track relevance scores for saturation detection
        self.search_relevance_scores: list[dict] = []
        self.consecutive_low_relevance = 0
        self.low_relevance_streak_threshold = cfg(
            branching_config, "low_relevance_streak"
        )

        # Logging
        self.step_log: list[dict] = []

    def run(self) -> dict:
        """Run the strategy-level branching agent.

        Flow:
        1. Shared SEARCH phase (single trajectory)
        2. Strategy proposal + clustering → fork per unique strategy
        3. PATCH + VERIFY for each trajectory (round-robin)
        """
        start_time = time.time()

        root = self.manager.create_root(
            agent_config=self.agent_config,
            model_config=self.model_config,
            env_config=self.env_config,
            task=self.problem_statement,
        )
        self.trajectory_phases["t0"] = Phase.SEARCH

        total_steps = 0

        try:
            # === Phase 1: SEARCH (single trajectory) ===
            logger.info(f"=== SEARCH PHASE ===")
            while root.status == "active" and self.trajectory_phases.get("t0") == Phase.SEARCH:
                try:
                    self._step_search(root)
                    total_steps += 1
                except Exception as e:
                    logger.error(f"Error in search step {root.step}: {e}", exc_info=True)
                    break

                # Saturation (normal exit) is checked before the hard cap
                # (fallback) by should_end_search. The cap prevents infinite
                # loops when the agent repeatedly attempts blocked commands,
                # which don't get relevance-scored and so never saturate.
                end_reason = should_end_search(
                    step=root.step,
                    consecutive_low_relevance=self.consecutive_low_relevance,
                    min_search_steps=self.min_search_steps,
                    low_relevance_streak=self.low_relevance_streak_threshold,
                    max_search_steps=self.max_search_steps,
                )
                if end_reason == "saturated":
                    self.tracer.log(
                        "phase1.search_saturated",
                        scores={"consecutive_low": self.consecutive_low_relevance,
                                "threshold": self.low_relevance_streak_threshold,
                                "total_search_steps": root.step},
                        decision="TRANSITION_TO_PHASE2",
                        phase="SEARCH", trajectory_id="t0", step=root.step,
                    )
                    self.trajectory_phases["t0"] = Phase.PATCH
                    break
                if end_reason == "step_limit":
                    self.tracer.log(
                        "phase1.search_step_limit",
                        scores={"total_search_steps": root.step,
                                "max_search_steps": self.max_search_steps},
                        decision="TRANSITION_TO_PHASE2",
                        phase="SEARCH", trajectory_id="t0", step=root.step,
                    )
                    logger.info(f"Search hit step limit ({self.max_search_steps}), transitioning to STRATEGY PROPOSAL")
                    self.trajectory_phases["t0"] = Phase.PATCH
                    break

            # === Phase 2: STRATEGY PROPOSAL (no containers created yet) ===
            # Build search report BEFORE pruning so relevant messages are visible.
            unique_strategies = []
            if root.status == "active" and self.use_strategy_proposal:
                logger.info(f"=== STRATEGY PROPOSAL ===")
                unique_strategies = self._propose_strategies(root)
            elif root.status == "active" and self.sdlg_enabled:
                logger.info(
                    "=== SDLG ARM: single trajectory, no strategy proposal "
                    "(SDLG branches at first write) ==="
                )
            elif root.status == "active":
                logger.info(
                    "=== NONE ARM: single vanilla trajectory, no proposal, no "
                    "SDLG (matched-k resample baseline) ==="
                )

            # === Prune low-relevance steps from root's context ===
            if root.status == "active":
                msgs_before = len(root.agent.messages)
                self._prune_irrelevant_steps(root)
                msgs_after = len(root.agent.messages)
                self.tracer.log(
                    "phase1.context_pruning",
                    input={"messages_before": msgs_before,
                           "total_search_steps": len(self.search_relevance_scores),
                           "relevant_steps": sum(1 for s in self.search_relevance_scores if s["is_relevant"]),
                           "irrelevant_steps": sum(1 for s in self.search_relevance_scores if not s["is_relevant"])},
                    output={"messages_after": msgs_after,
                            "messages_removed": msgs_before - msgs_after},
                    phase="SEARCH→PHASE2", trajectory_id="t0",
                )

            # Save pruned search messages as template for cloned trajectories
            self._search_messages_template = copy.deepcopy(root.agent.messages)

            # NLI model runs in a separate server process — no memory to free here
            import gc
            gc.collect()

            # === Phase 3+4: PATCH + VERIFY (LAZY SEQUENTIAL) ===
            # Create ONE container at a time. Run to completion. Destroy. Next.
            # Only 1 SWE-bench container exists alongside vLLM at any point.
            if not unique_strategies:
                unique_strategies = ["Fix the bug as described in the problem statement."]

            logger.info(f"=== PATCH/VERIFY PHASE ({len(unique_strategies)} strategies, lazy sequential) ===")

            for i, strategy in enumerate(unique_strategies):
                traj_id = "t0" if i == 0 else f"t0_strategy_{i}"

                if i == 0:
                    # First strategy: reuse root trajectory (container already exists)
                    traj = root
                    self.trajectory_strategies[traj_id] = strategy
                    self._inject_strategy_prompt(traj, strategy)
                    self.trajectory_phases[traj_id] = Phase.PATCH
                else:
                    # Create fresh container just-in-time
                    traj = self._create_lazy_trajectory(strategy, i)
                    if not traj:
                        continue

                logger.info(f"--- Running {traj.trajectory_id}: {strategy[:80]} ---")
                self.tracer.log(
                    "phase3.trajectory_start",
                    input={"trajectory_id": traj.trajectory_id,
                           "strategy": strategy[:200],
                           "strategy_index": i},
                    phase="PATCH", trajectory_id=traj.trajectory_id,
                )

                while traj.status == "active":
                    phase = self.trajectory_phases.get(traj.trajectory_id, Phase.PATCH)

                    try:
                        if phase == Phase.PATCH:
                            self._step_patch(traj)
                        elif phase == Phase.VERIFY:
                            self._step_verify(traj)
                        total_steps += 1
                    except Exception as e:
                        logger.error(f"Error in {traj.trajectory_id} step {traj.step}: {e}")
                        traj.status = "failed"

                    if traj.step > self.max_steps:
                        logger.warning(f"{traj.trajectory_id} hit step limit ({self.max_steps})")
                        traj.status = "completed"
                        break

                # Capture any uncommitted diff BEFORE destroying the container —
                # the step-limit/format-error completion paths don't populate
                # traj.patch, and cleanup() makes the diff unrecoverable.
                self._capture_patch_if_missing(traj)

                # Save results and DESTROY container before next strategy
                self.manager.save_all()
                try:
                    traj.cleanup()
                except Exception:
                    pass

                logger.info(
                    f"--- Finished {traj.trajectory_id}: status={traj.status} "
                    f"submitted={traj.submitted} patch={len(traj.patch or '')}ch ---"
                )

                # === Run SDLG children of this trajectory ===
                sdlg_children = [
                    t for t in self.manager.trajectories.values()
                    if t.parent_id == traj.trajectory_id and t.status == "active"
                ]
                if sdlg_children:
                    logger.info(f"--- Running {len(sdlg_children)} SDLG children of {traj.trajectory_id} ---")

                for sdlg_traj in sdlg_children:
                    logger.info(f"--- Running SDLG child {sdlg_traj.trajectory_id} ---")
                    self.tracer.log(
                        "phase3.sdlg_child_start",
                        input={"trajectory_id": sdlg_traj.trajectory_id,
                               "parent_id": traj.trajectory_id},
                        phase="SDLG_CHILD", trajectory_id=sdlg_traj.trajectory_id,
                    )

                    while sdlg_traj.status == "active":
                        sdlg_phase = self.trajectory_phases.get(sdlg_traj.trajectory_id, Phase.PATCH)
                        try:
                            if sdlg_phase == Phase.PATCH:
                                self._step_patch(sdlg_traj)
                            elif sdlg_phase == Phase.VERIFY:
                                self._step_verify(sdlg_traj)
                            total_steps += 1
                        except Exception as e:
                            logger.error(f"Error in SDLG child {sdlg_traj.trajectory_id} step {sdlg_traj.step}: {e}")
                            sdlg_traj.status = "failed"

                        if sdlg_traj.step > self.max_steps:
                            sdlg_traj.status = "completed"
                            break

                    self._capture_patch_if_missing(sdlg_traj)
                    self.manager.save_all()
                    try:
                        sdlg_traj.cleanup()
                    except Exception:
                        pass

                    logger.info(
                        f"--- Finished SDLG child {sdlg_traj.trajectory_id}: status={sdlg_traj.status} "
                        f"submitted={sdlg_traj.submitted} patch={len(sdlg_traj.patch or '')}ch ---"
                    )

                gc.collect()

        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            self.manager.save_all()
            self._save_logs()

        elapsed = time.time() - start_time
        results = self._collect_results(elapsed, total_steps)
        self._print_tree()
        return results

    # ---- Phase 1: SEARCH ----

    def _step_search(self, traj: Trajectory) -> None:
        """Execute one search step. Read-only."""
        traj.step += 1

        self._ensure_phase_prompt(traj, Phase.SEARCH)

        # --- LLM query ---
        self.tracer.log(
            "phase1.llm_query",
            input={"messages_count": len(traj.agent.messages),
                   "last_message": traj.agent.messages[-1].get("content", "")[:500] if traj.agent.messages else ""},
            phase="SEARCH", trajectory_id=traj.trajectory_id, step=traj.step,
        )

        try:
            message = traj.agent.query_only()
        except (LimitsExceeded, Submitted) as e:
            traj.agent.add_messages(*e.messages)
            self.tracer.log("phase1.llm_query_exception", output={"exception": type(e).__name__},
                            phase="SEARCH", trajectory_id=traj.trajectory_id, step=traj.step)
            self._finish_trajectory(traj, e)
            return
        except FormatError as e:
            traj.agent.add_messages(*e.messages)
            self.tracer.log("phase1.llm_query_format_error", output={"error": str(e)[:200]},
                            phase="SEARCH", trajectory_id=traj.trajectory_id, step=traj.step)
            return

        content = message.get("content", "")
        thought = extract_thought(content)
        actions = message.get("extra", {}).get("actions", [])
        action_cmd = actions[0].get("command", "") if actions else ""

        self.tracer.log(
            "phase1.llm_response",
            output={"thought": thought, "action": action_cmd, "full_content": content},
            phase="SEARCH", trajectory_id=traj.trajectory_id, step=traj.step,
        )

        # --- Command allowlist check ---
        if action_cmd and not is_command_allowed(action_cmd, Phase.SEARCH):
            self.tracer.log(
                "phase1.command_blocked",
                input={"command": action_cmd},
                output={"allowed": False, "reason": "write command in SEARCH phase"},
                decision="BLOCK",
                phase="SEARCH", trajectory_id=traj.trajectory_id, step=traj.step,
            )
            error_msg = {
                "role": "user",
                "content": (
                    "You are in the EXPLORATION phase — you cannot modify files yet. "
                    "Continue investigating the codebase. Use grep, cat, find, etc. "
                    "When you have a fix strategy, state it with STRATEGY: <your approach>"
                ),
            }
            if traj.agent.messages and traj.agent.messages[-1].get("role") == "assistant":
                traj.agent.messages.pop()
            traj.agent.add_messages(error_msg)
            self._log_step(traj, phase="SEARCH", blocked=action_cmd[:100])

            # Blocked commands count as low-relevance for saturation detection.
            # Without this, the agent can loop forever attempting write commands
            # that always get blocked, never triggering the saturation threshold.
            self.consecutive_low_relevance += 1
            self.search_relevance_scores.append({
                "step": traj.step,
                "relevance": 0.0,
                "is_relevant": False,
                "summary": f"[BLOCKED] {action_cmd[:80]}",
            })
            return

        # --- Execute search command ---
        self.tracer.log(
            "phase1.execute_command",
            input={"command": action_cmd},
            phase="SEARCH", trajectory_id=traj.trajectory_id, step=traj.step,
        )

        try:
            obs_messages = traj.agent.execute_response(message)
        except Submitted as e:
            traj.agent.add_messages(*e.messages)
            self._finish_trajectory(traj, e)
            return
        except InterruptAgentFlow as e:
            traj.agent.add_messages(*e.messages)
            obs_messages = []

        observation = ""
        if obs_messages:
            observation = obs_messages[0].get("content", "")[:2000]

        self.tracer.log(
            "phase1.command_output",
            output={"observation": observation},
            phase="SEARCH", trajectory_id=traj.trajectory_id, step=traj.step,
        )

        # --- Relevance scoring ---
        self.tracer.log(
            "phase1.relevance_scoring.input",
            input={"thought": thought, "observation": observation[:500],
                   "problem_statement": self.problem_statement[:500]},
            phase="SEARCH", trajectory_id=traj.trajectory_id, step=traj.step,
        )

        relevance = self.relevance_scorer.score_trajectory_step(
            thought, observation[:500], self.problem_statement
        )

        self.tracer.log(
            "phase1.relevance_scoring.output",
            output={"summary": relevance.get("summary", ""),
                    "is_relevant": relevance["is_relevant"]},
            scores={"relevance": relevance["relevance"],
                    "threshold": self.relevance_scorer.threshold},
            phase="SEARCH", trajectory_id=traj.trajectory_id, step=traj.step,
        )

        # --- Saturation tracking ---
        self.search_relevance_scores.append({
            "step": traj.step,
            "relevance": relevance["relevance"],
            "is_relevant": relevance["is_relevant"],
            "summary": relevance.get("summary", ""),
        })
        if relevance["is_relevant"]:
            self.consecutive_low_relevance = 0
        else:
            self.consecutive_low_relevance += 1

        self.tracer.log(
            "phase1.saturation_check",
            scores={"consecutive_low_relevance": self.consecutive_low_relevance,
                    "streak_threshold": self.low_relevance_streak_threshold,
                    "total_steps": traj.step,
                    "total_relevant": sum(1 for s in self.search_relevance_scores if s["is_relevant"]),
                    "total_irrelevant": sum(1 for s in self.search_relevance_scores if not s["is_relevant"])},
            decision="SATURATED" if self.consecutive_low_relevance >= self.low_relevance_streak_threshold else "CONTINUE",
            phase="SEARCH", trajectory_id=traj.trajectory_id, step=traj.step,
        )

        self._log_step(
            traj,
            phase="SEARCH",
            thought=thought[:200],
            action=action_cmd[:100],
            relevance=relevance["relevance"],
            summary=relevance.get("summary", ""),
            is_relevant=relevance["is_relevant"],
        )

        self._check_finished(traj)

    def _prune_irrelevant_steps(self, traj: Trajectory) -> None:
        """Remove low-relevance search steps from the agent's message history.

        Each search step produces an assistant message (thought + action) followed
        by a user message (observation). We identify which assistant messages
        correspond to low-relevance steps and remove both the assistant message
        and its observation, freeing context for the patch phase.

        The system message and the first user message (instance prompt) are always kept.
        """
        if not self.search_relevance_scores:
            return

        relevant_steps = {
            s["step"] for s in self.search_relevance_scores
            if s["relevance"] >= self.relevance_scorer.threshold
        }
        n_total = len(self.search_relevance_scores)
        n_relevant = len(relevant_steps)

        if n_relevant >= n_total:
            return  # Nothing to prune

        messages = traj.agent.messages
        pruned = []
        assistant_step = 0

        for msg in messages:
            if msg.get("role") == "assistant":
                assistant_step += 1
                if assistant_step not in relevant_steps:
                    continue  # Drop this assistant message
                pruned.append(msg)
            elif msg.get("role") == "user" and assistant_step > 0:
                # This is an observation following an assistant message
                if assistant_step not in relevant_steps:
                    continue  # Drop the observation too
                pruned.append(msg)
            else:
                # System message, instance prompt, or phase prompts — always keep
                pruned.append(msg)

        n_removed = len(messages) - len(pruned)
        logger.info(
            f"Pruned {n_removed} messages from context "
            f"({n_relevant}/{n_total} relevant steps kept)"
        )
        traj.agent.set_messages(pruned)

    # ---- Strategy Proposal + Fork ----

    def _propose_strategies(self, root: Trajectory) -> list[str]:
        """Propose strategies, cluster them, return unique strategy strings.

        No containers or trajectories are created here — just strategy text.
        Container creation happens lazily in the Phase 3 sequential loop.
        """

        # 1. Build search report with code
        self.tracer.log(
            "phase2.build_search_report.input",
            input={"messages_count": len(root.agent.messages),
                   "relevance_scores_count": len(self.search_relevance_scores),
                   "relevant_steps": sum(1 for s in self.search_relevance_scores if s["is_relevant"]),
                   "relevance_threshold": self.relevance_scorer.threshold},
            phase="STRATEGY_PROPOSAL", trajectory_id="t0",
        )

        search_report = self.proposer.build_search_report(
            root.agent.messages,
            relevance_scores=self.search_relevance_scores,
            relevance_threshold=self.relevance_scorer.threshold,
        )

        self.tracer.log(
            "phase2.build_search_report.output",
            output={"search_report": search_report, "length": len(search_report)},
            phase="STRATEGY_PROPOSAL", trajectory_id="t0",
        )

        # 2. Propose K strategies
        self.tracer.log(
            "phase2.propose_strategies.input",
            input={"search_report": search_report,
                   "problem_statement": self.problem_statement[:500],
                   "n_strategies": self.proposer.n_strategies,
                   "temperature": self.proposer.temperature},
            phase="STRATEGY_PROPOSAL", trajectory_id="t0",
        )

        strategies = self.proposer.propose(search_report, self.problem_statement)

        self.tracer.log(
            "phase2.propose_strategies.output",
            output={"n_proposed": len(strategies),
                    "strategies": strategies},
            phase="STRATEGY_PROPOSAL", trajectory_id="t0",
        )

        if not strategies:
            return []

        # 3. Cluster strategies — pairwise NLI + bidirectional entailment.
        # Use the SAME context and threshold as the clusterer / the SDLG path so
        # the logged pairwise decisions cannot desync from the actual clustering.
        #
        # DELIBERATE DEVIATION from Kuhn et al.'s context-conditioning: NO shared
        # problem-statement prefix. Kuhn conditions entailment on the QA question
        # to disambiguate context-DEPENDENT short answers ("Paris" vs "France");
        # our strategies/intents are self-contained imperatives, and prepending a
        # 500-char shared prefix to both sides SATURATES DeBERTa entailment —
        # measured on the 2026-06-10 Phase A run-1 artifacts (sympy-12096): all
        # 10 pairs of 5 structurally distinct strategies score >=0.94 WITH the
        # prefix (-> 1 cluster, entropy 0, gate never fires) vs <=0.55 WITHOUT
        # (-> 5 clusters). The gate was measuring the prefix, not the strategies.
        # Reproduce with scripts/diagnose_context_saturation.py.
        cluster_context = ""
        entail_thr = self.clusterer.threshold
        tau = cfg(self.branching_config, "entropy_threshold")
        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                a = f"{cluster_context} {strategies[i]}".strip()
                b = f"{cluster_context} {strategies[j]}".strip()
                fwd = self.nli.classify(a, b)
                bwd = self.nli.classify(b, a)
                same = fwd["entailment"] > entail_thr and bwd["entailment"] > entail_thr
                self.tracer.log(
                    "phase2.pairwise_nli",
                    input={"strategy_a_idx": i, "strategy_a": strategies[i][:300],
                           "strategy_b_idx": j, "strategy_b": strategies[j][:300]},
                    scores={"forward_entailment": round(fwd["entailment"], 4),
                            "forward_neutral": round(fwd["neutral"], 4),
                            "forward_contradiction": round(fwd["contradiction"], 4),
                            "backward_entailment": round(bwd["entailment"], 4),
                            "backward_neutral": round(bwd["neutral"], 4),
                            "backward_contradiction": round(bwd["contradiction"], 4),
                            "entailment_threshold": entail_thr},
                    decision="SAME_CLUSTER" if same else "DIFFERENT_CLUSTERS",
                    phase="STRATEGY_PROPOSAL",
                )

        analysis = self.clusterer.analyze(
            strategies,
            tau=tau,
            context=cluster_context,
        )
        clusters = analysis["clusters"]
        entropy = analysis["entropy"]
        should_branch = analysis["should_branch"]

        all_reps = [strategies[c.representative_idx] for c in clusters]

        # Adaptive gate (R1.5): branch iff entropy > tau. When entropy <= tau the
        # proposed strategies are not meaningfully multi-modal, so take the single
        # dominant (largest) cluster's representative — the greedy/mode action —
        # exactly as the SDLG arm returns early without forking. Both arms read the
        # SAME entropy_threshold config key and ACT on it identically; only the
        # candidate generator differs. (With the default tau=0.0 every non-trivial
        # entropy branches, so the main run is unaffected; this matters under the
        # R3.3 tau sweep, where the strategy arm previously branched regardless.)
        if should_branch:
            unique_strategies = all_reps
        else:
            dominant = max(clusters, key=lambda c: len(c.indices))
            unique_strategies = [strategies[dominant.representative_idx]]

        self.tracer.log(
            "phase2.clustering_result",
            input={"n_strategies": len(strategies)},
            output={"n_clusters": len(clusters),
                    "n_unique": len(unique_strategies),
                    "cluster_sizes": [len(c.indices) for c in clusters],
                    "cluster_members": [list(c.indices) for c in clusters],
                    "unique_strategies": unique_strategies},
            scores={"semantic_entropy": round(entropy, 4),
                    "entropy_threshold": tau},
            decision="BRANCH_ALL" if should_branch else "SINGLE_STRATEGY",
            phase="STRATEGY_PROPOSAL",
        )

        self._log_strategy_proposal(strategies, clusters, entropy, unique_strategies)
        return unique_strategies

    def _register_failed_draw(
        self, traj_id: str, parent_id: str | None, branch_info: dict,
        env=None,
    ) -> Trajectory:
        """Record a draw that failed at CREATION as a failed placeholder (R7.2).

        A fork the orchestrator decided to make is a genuine draw even if the
        container / clone / injection failed before it could run: the vanilla
        resample driver records its run-loop failures as empty-patch rows, so
        the treatment must count its failed creations under the same rule —
        dropping them would deflate the treatment's metric-time k (the same
        pro-treatment direction as the iteration-6 producer fix, one layer
        earlier). The placeholder has no agent; any half-created container is
        reaped immediately.
        """
        traj = Trajectory(
            trajectory_id=traj_id, agent=None, env=env, parent_id=parent_id,
            status="failed", branch_info=dict(branch_info, creation_failed=True),
        )
        self.manager.trajectories[traj_id] = traj
        try:
            traj.cleanup()
        except Exception:
            pass
        logger.warning(f"Registered failed-at-creation draw {traj_id} (R7.2)")
        return traj

    def _create_lazy_trajectory(
        self, strategy: str, index: int,
    ) -> Trajectory | None:
        """Create a fresh trajectory for a strategy just-in-time.

        Creates a new Docker container, model, and agent from scratch.
        Uses the saved search messages template (no need to clone from a
        running container — search phase doesn't modify files).
        Only 1 container exists at a time = safe within 32GB VRAM.

        On failure the strategy's draw is registered as a failed placeholder
        (see _register_failed_draw) — never silently skipped.
        """
        traj_id = f"t0_strategy_{index}"
        new_env = None
        try:
            # Build env config — image is already set correctly by run_branching.py
            env_config = dict(self.env_config)

            # Use the same constructor pattern as TrajectoryManager.create_root
            new_env = DockerEnvironment(**env_config)
            new_model = LitellmTextbasedModel(**self.model_config)
            new_agent = BranchingAgent(model=new_model, env=new_env, **self.agent_config)

            # Load the saved search messages template (pruned)
            messages = copy.deepcopy(self._search_messages_template)
            new_agent.set_messages(messages)

            traj = Trajectory(
                trajectory_id=traj_id,
                agent=new_agent,
                env=new_env,
                parent_id="t0",
                branch_step=0,
                status="active",
                step=0,
                last_branch_step=0,
                branch_info={"type": "strategy", "strategy": strategy[:200]},
            )

            self.manager.trajectories[traj_id] = traj
            self.trajectory_phases[traj_id] = Phase.PATCH
            self.trajectory_strategies[traj_id] = strategy
            self._inject_strategy_prompt(traj, strategy)

            logger.info(f"Created lazy trajectory {traj_id} for strategy: {strategy[:100]}")
            return traj

        except Exception as e:
            logger.error(f"Failed to create trajectory for strategy {index}: {e}", exc_info=True)
            self.trajectory_strategies[traj_id] = strategy
            self._register_failed_draw(
                traj_id, "t0",
                {"type": "strategy", "strategy": strategy[:200]},
                env=new_env,
            )
            return None

    def _inject_strategy_prompt(self, traj: Trajectory, strategy: str) -> None:
        """Inject the strategy-specific patch prompt into the trajectory."""
        prompt = PATCH_PROMPT_WITH_STRATEGY.format(strategy=strategy)
        traj.agent.add_messages({"role": "user", "content": prompt})
        setattr(traj, "_phase_injected_patch", True)

    # ---- Phase 3: PATCH ----

    def _step_patch(self, traj: Trajectory) -> None:
        """Execute one patch step, applying SDLG on the first write command."""
        traj.step += 1
        self._truncate_context(traj)

        # --- LLM query ---
        self.tracer.log(
            "phase3.llm_query",
            input={"messages_count": len(traj.agent.messages)},
            phase="PATCH", trajectory_id=traj.trajectory_id, step=traj.step,
        )

        try:
            message = traj.agent.query_only()
        except (LimitsExceeded, Submitted) as e:
            traj.agent.add_messages(*e.messages)
            self.tracer.log("phase3.llm_query_exception", output={"exception": type(e).__name__},
                            phase="PATCH", trajectory_id=traj.trajectory_id, step=traj.step)
            self._finish_trajectory(traj, e)
            return
        except FormatError as e:
            traj.agent.add_messages(*e.messages)
            return

        content = message.get("content", "")
        thought = extract_thought(content)
        actions = message.get("extra", {}).get("actions", [])
        action_cmd = actions[0].get("command", "") if actions else ""

        self.tracer.log(
            "phase3.llm_response",
            output={"thought": thought, "action": action_cmd, "full_content": content},
            phase="PATCH", trajectory_id=traj.trajectory_id, step=traj.step,
        )

        # --- Phase transition check ---
        new_phase = detect_phase_transition(thought, action_cmd, Phase.PATCH)
        if new_phase == Phase.VERIFY:
            self.tracer.log(
                "phase3.phase_transition",
                decision="PATCH→VERIFY",
                phase="PATCH", trajectory_id=traj.trajectory_id, step=traj.step,
            )
            self.trajectory_phases[traj.trajectory_id] = Phase.VERIFY

        # --- Read-only budget tracking ---
        # Track consecutive read-only steps in PATCH phase per trajectory.
        # After budget exhausted, let command through but nudge agent to write.
        is_write_cmd = action_cmd and self._is_write_command(action_cmd)
        patch_reads = getattr(traj, "_patch_read_steps", 0)

        if not is_write_cmd and action_cmd:
            patch_reads += 1
            setattr(traj, "_patch_read_steps", patch_reads)
        elif is_write_cmd:
            setattr(traj, "_patch_read_steps", 0)  # Reset on write

        # --- SDLG diversification (if enabled) ---
        if self.sdlg_enabled:
            sdlg_already_applied = getattr(traj, "_sdlg_applied", False)

            self.tracer.log(
                "phase3.sdlg_check",
                input={"action": action_cmd[:200], "is_write_command": is_write_cmd,
                       "sdlg_already_applied": sdlg_already_applied,
                       "can_branch": self.manager.can_branch(1)},
                decision="TRIGGER_SDLG" if (is_write_cmd and not sdlg_already_applied and self.manager.can_branch(1)) else "SKIP_SDLG",
                phase="PATCH", trajectory_id=traj.trajectory_id, step=traj.step,
            )

        if self.sdlg_enabled and is_write_cmd and not getattr(traj, "_sdlg_applied", False) and self.manager.can_branch(1):
            setattr(traj, "_sdlg_applied", True)
            sdlg_forks = self._apply_sdlg(traj, message, content)

        # --- Execute greedy response ---
        self.tracer.log(
            "phase3.execute_command",
            input={"command": action_cmd},
            phase="PATCH", trajectory_id=traj.trajectory_id, step=traj.step,
        )

        try:
            obs_messages = traj.agent.execute_response(message)
        except Submitted as e:
            traj.agent.add_messages(*e.messages)
            self._finish_trajectory(traj, e)
            return
        except InterruptAgentFlow as e:
            traj.agent.add_messages(*e.messages)
            obs_messages = []

        observation = ""
        if obs_messages:
            observation = obs_messages[0].get("content", "")[:2000] if isinstance(obs_messages, list) and obs_messages else ""

        self.tracer.log(
            "phase3.command_output",
            output={"observation": observation},
            phase="PATCH", trajectory_id=traj.trajectory_id, step=traj.step,
        )

        # Nudge agent if read budget exceeded (but don't block)
        if not is_write_cmd and patch_reads > self.patch_read_budget:
            self.tracer.log(
                "phase3.read_budget_nudge",
                input={"read_steps": patch_reads, "budget": self.patch_read_budget},
                decision="NUDGE_WRITE",
                phase="PATCH", trajectory_id=traj.trajectory_id, step=traj.step,
            )
            traj.agent.add_messages({
                "role": "user",
                "content": (
                    "You've been reading files for several steps. Please write your code edit now "
                    "using sed -i or cat <<'EOF' >. You have enough information from the search phase."
                ),
            })

        strategy = self.trajectory_strategies.get(traj.trajectory_id, "")
        self._log_step(
            traj, phase="PATCH", thought=thought[:200],
            action=action_cmd[:100], strategy=strategy[:100],
        )
        self._check_finished(traj)

    def _is_write_command(self, cmd: str) -> bool:
        """Check if a command modifies files (triggers SDLG diversification).

        Delegates to the single hardened detector in phases.is_write_command so
        the SDLG trigger and the read-budget tracker agree, and so the submit
        command / bare echo / stderr redirects are not misclassified as writes
        (which previously fired SDLG at the wrong point in the sdlg arm).
        """
        return is_write_command(cmd)

    def _apply_sdlg(
        self, traj: Trajectory, message: dict, greedy_content: str,
    ) -> list[Trajectory]:
        """Apply SDLG to generate diverse alternatives, cluster them, fork unique ones.

        Pipeline:
        1. Generate N alternatives via SDLG (thought-level + code-level)
        2. Extract intent summaries for each alternative
        3. Cluster intents via bidirectional NLI entailment
        4. Compute semantic entropy over clusters
        5. If entropy > τ: fork ONE trajectory per unique cluster (representative)
           If entropy ≤ τ: no forking (alternatives aren't diverse enough)
        """
        pre_messages = traj.agent.messages[:-1]

        model_name = self.model_config.get("model_name", "openai/qwen3-coder")
        model_kwargs = self.model_config.get("model_kwargs", {})

        self.tracer.log(
            "phase3.sdlg.generate.input",
            input={"greedy_content": greedy_content,
                   "pre_messages_count": len(pre_messages),
                   "n_alternatives": self.sdlg.n_candidates},
            phase="SDLG", trajectory_id=traj.trajectory_id, step=traj.step,
        )

        # Step 1: Generate N alternatives (thought-level + code-level SDLG)
        try:
            alternatives = self.sdlg.generate(
                model_name=model_name,
                model_kwargs=model_kwargs,
                messages=pre_messages,
                greedy_response=greedy_content,
            )
        except Exception as e:
            self.tracer.log(
                "phase3.sdlg.generate.error",
                output={"error": str(e)},
                phase="SDLG", trajectory_id=traj.trajectory_id, step=traj.step,
            )
            return []

        # Filter out duplicates of greedy
        unique_alts = [alternatives[0]]  # greedy is always first
        for alt in alternatives[1:]:
            if alt != greedy_content:
                unique_alts.append(alt)

        self.tracer.log(
            "phase3.sdlg.generate.output",
            output={"n_alternatives": len(alternatives),
                    "n_unique": len(unique_alts),
                    "alternatives": [alt[:500] for alt in unique_alts]},
            phase="SDLG", trajectory_id=traj.trajectory_id, step=traj.step,
        )

        if len(unique_alts) <= 1:
            logger.info("SDLG: no unique alternatives generated, skipping forking")
            return []

        # Step 2: Extract intent summaries for clustering
        intent_extractor = IntentExtractor(
            model_name=model_name,
            model_kwargs=model_kwargs,
            method="llm",
        )
        intents = intent_extractor.extract_batch_with_history(unique_alts, pre_messages)

        self.tracer.log(
            "phase3.sdlg.intents",
            output={"intents": intents},
            phase="SDLG", trajectory_id=traj.trajectory_id, step=traj.step,
        )

        # Step 3: Cluster via bidirectional entailment + compute entropy.
        # context="" — same deliberate deviation as the strategy path: a shared
        # problem-statement prefix saturates DeBERTa entailment and merges
        # distinct intents into one cluster (see _propose_strategies and
        # scripts/diagnose_context_saturation.py). Intents are prompted to be
        # self-contained one-sentence summaries, so no conditioning is needed.
        entropy_threshold = cfg(self.branching_config, "entropy_threshold")
        analysis = self.clusterer.analyze(
            intents,
            tau=entropy_threshold,
            context="",
        )
        clusters = analysis["clusters"]
        entropy = analysis["entropy"]
        should_branch = analysis["should_branch"]

        self.tracer.log(
            "phase3.sdlg.clustering",
            output={"n_clusters": len(clusters),
                    "cluster_sizes": [len(c.indices) for c in clusters],
                    "cluster_intents": [[intents[i] for i in c.indices] for c in clusters]},
            scores={"semantic_entropy": round(entropy, 4),
                    "entropy_threshold": entropy_threshold},
            decision="BRANCH" if should_branch else "NO_BRANCH_LOW_ENTROPY",
            phase="SDLG", trajectory_id=traj.trajectory_id, step=traj.step,
        )

        if not should_branch:
            logger.info(
                f"SDLG: entropy {entropy:.3f} ≤ τ={entropy_threshold}, "
                f"alternatives not diverse enough — no forking"
            )
            return []

        # Step 4: Fork ONE trajectory per unique cluster (skip cluster containing greedy)
        # Greedy (index 0) is always in unique_alts[0] — find its cluster
        greedy_cluster_idx = None
        for ci, cluster in enumerate(clusters):
            if 0 in cluster.indices:
                greedy_cluster_idx = ci
                break

        forked = []
        fork_index = 1
        for ci, cluster in enumerate(clusters):
            if ci == greedy_cluster_idx:
                continue  # Greedy trajectory already runs on the parent
            if not self.manager.can_branch(1):
                self.tracer.log(
                    "phase3.sdlg.fork_skip",
                    input={"cluster_index": ci},
                    decision="TRAJECTORY_CAP_REACHED",
                    phase="SDLG", trajectory_id=traj.trajectory_id, step=traj.step,
                )
                break

            # Use the cluster representative's alternative
            rep_idx = cluster.representative_idx
            alt_content = unique_alts[rep_idx]

            new_traj = self._clone_for_sdlg(traj, alt_content, fork_index)
            # The fork index advances on EVERY attempt: _clone_for_sdlg always
            # registers a draw under this index (active, completed-at-injection,
            # or failed placeholder — R7.2), so reusing the index would
            # overwrite that record.
            fork_index += 1
            if new_traj:
                self.tracer.log(
                    "phase3.sdlg.fork_created",
                    input={"cluster_index": ci,
                           "representative_idx": rep_idx,
                           "intent": intents[rep_idx],
                           "alternative_content": alt_content[:500]},
                    output={"trajectory_id": new_traj.trajectory_id,
                            "parent_id": traj.trajectory_id},
                    phase="SDLG", trajectory_id=new_traj.trajectory_id, step=traj.step,
                )
                forked.append(new_traj)

        # Log the SDLG event
        self.manager.branching_log.append({
            "timestamp": time.time(),
            "event": "sdlg_fork",
            "parent": traj.trajectory_id,
            "step": traj.step,
            "n_alternatives": len(alternatives),
            "n_unique": len(unique_alts),
            "n_clusters": len(clusters),
            "entropy": round(entropy, 4),
            "n_forked": len(forked),
            "forked_ids": [t.trajectory_id for t in forked],
        })

        return forked

    @staticmethod
    def _inject_alternative(agent, alt_content: str) -> tuple[str, bool, str]:
        """Execute an injected SDLG alternative; classify the draw's outcome.

        Returns (status, submitted, patch). The alternative response can itself
        contain the submit command — execute_actions then raises Submitted, and
        that is a COMPLETED draw with a real (possibly passing) patch which must
        be kept. Before this helper, a blanket clone-failure handler swallowed
        the exception and the child vanished from the record entirely (silent
        data loss against the treatment) while its container leaked. A
        non-Submitted execution error is a FAILED draw — still a draw (R7.2).
        """
        try:
            agent.inject_and_execute(alt_content)
            return "active", False, ""
        except Submitted as e:
            agent.add_messages(*e.messages)
            patch = e.messages[0].get("extra", {}).get("submission", "")
            return "completed", True, patch
        except Exception as e:
            logger.error(f"Injected SDLG alternative failed to execute: {e}")
            return "failed", False, ""

    def _clone_for_sdlg(
        self, parent: Trajectory, alt_content: str, index: int,
    ) -> Trajectory | None:
        """Clone a strategy trajectory for an SDLG alternative response.

        ALWAYS leaves a draw record under t<parent>_sdlg_<index> (R7.2): an
        'active' child for the run loop, a 'completed' child if the injected
        alternative submitted immediately, or a 'failed' placeholder/child if
        cloning or injection failed. Returns the trajectory only when it still
        needs to run (status 'active'); the caller must advance its fork index
        regardless, since the id is taken in every case.
        """
        traj_id = f"{parent.trajectory_id}_sdlg_{index}"
        new_env = None
        try:
            from src.utils.docker_helpers import clone_container_state

            env_config = parent.env.config.model_dump()
            new_env = DockerEnvironment(**env_config)
            clone_container_state(parent.env.container_id, new_env.container_id)

            model_config = parent.agent.model.config.model_dump()
            new_model = LitellmTextbasedModel(**model_config)

            agent_config = parent.agent.config.model_dump()
            new_agent = BranchingAgent(model=new_model, env=new_env, **agent_config)

            # Copy messages UP TO (but not including) the greedy response
            messages = copy.deepcopy(parent.agent.messages[:-1])
            new_agent.set_messages(messages)
            new_agent.n_calls = parent.agent.n_calls
            new_agent.cost = parent.agent.cost
            new_agent.extra_template_vars = copy.deepcopy(parent.agent.extra_template_vars)
        except Exception as e:
            logger.error(f"Failed to clone for SDLG {index}: {e}", exc_info=True)
            self._register_failed_draw(
                traj_id, parent.trajectory_id, {"type": "sdlg", "rank": index},
                env=new_env,
            )
            return None

        # Inject the SDLG alternative response and execute it (may submit/fail)
        status, submitted, patch = self._inject_alternative(new_agent, alt_content)

        traj = Trajectory(
            trajectory_id=traj_id,
            agent=new_agent,
            env=new_env,
            parent_id=parent.trajectory_id,
            branch_step=parent.step,
            status=status,
            step=parent.step,
            last_branch_step=parent.step,
            submitted=submitted,
            patch=patch,
            branch_info={"type": "sdlg", "rank": index},
        )

        self.manager.trajectories[traj_id] = traj
        self.trajectory_phases[traj_id] = self.trajectory_phases.get(
            parent.trajectory_id, Phase.PATCH
        )
        # Inherit the parent's strategy assignment
        parent_strategy = self.trajectory_strategies.get(parent.trajectory_id, "")
        self.trajectory_strategies[traj_id] = parent_strategy
        # Mark as already SDLG'd so we don't re-apply
        setattr(traj, "_sdlg_applied", True)

        if status != "active":
            # Finished (or died) at injection: capture any working-tree diff a
            # failed injection left behind, then reap the container now — the
            # run loop only iterates 'active' children.
            if status == "failed":
                self._capture_patch_if_missing(traj)
            try:
                traj.cleanup()
            except Exception:
                pass
            logger.info(
                f"SDLG sub-trajectory {traj_id} finished at injection: "
                f"status={status} submitted={submitted} patch={len(traj.patch or '')}ch"
            )
            return None

        logger.info(f"Created SDLG sub-trajectory {traj_id} from {parent.trajectory_id}")
        return traj

    # ---- Phase 4: VERIFY ----

    def _step_verify(self, traj: Trajectory) -> None:
        """Execute one verify step. No branching."""
        traj.step += 1
        self._truncate_context(traj)

        self._ensure_phase_prompt(traj, Phase.VERIFY)

        self.tracer.log(
            "phase4.llm_query",
            input={"messages_count": len(traj.agent.messages)},
            phase="VERIFY", trajectory_id=traj.trajectory_id, step=traj.step,
        )

        try:
            message = traj.agent.query_only()
        except (LimitsExceeded, Submitted) as e:
            traj.agent.add_messages(*e.messages)
            self.tracer.log("phase4.llm_query_exception", output={"exception": type(e).__name__},
                            phase="VERIFY", trajectory_id=traj.trajectory_id, step=traj.step)
            self._finish_trajectory(traj, e)
            return
        except FormatError as e:
            traj.agent.add_messages(*e.messages)
            return

        content = message.get("content", "")
        thought = extract_thought(content)
        actions = message.get("extra", {}).get("actions", [])
        action_cmd = actions[0].get("command", "") if actions else ""

        self.tracer.log(
            "phase4.llm_response",
            output={"thought": thought, "action": action_cmd, "full_content": content},
            phase="VERIFY", trajectory_id=traj.trajectory_id, step=traj.step,
        )

        # --- SDLG in VERIFY (fallback, if enabled) ---
        is_write_cmd = action_cmd and self._is_write_command(action_cmd)

        if self.sdlg_enabled and is_write_cmd and not getattr(traj, "_sdlg_applied", False) and self.manager.can_branch(1):
            self.tracer.log(
                "phase4.sdlg_trigger",
                input={"action": action_cmd[:200], "is_write_command": True},
                decision="TRIGGER_SDLG_IN_VERIFY",
                phase="VERIFY", trajectory_id=traj.trajectory_id, step=traj.step,
            )
            setattr(traj, "_sdlg_applied", True)
            self._apply_sdlg(traj, message, content)

        self.tracer.log(
            "phase4.execute_command",
            input={"command": action_cmd},
            phase="VERIFY", trajectory_id=traj.trajectory_id, step=traj.step,
        )

        try:
            obs_messages = traj.agent.execute_response(message)
        except Submitted as e:
            traj.agent.add_messages(*e.messages)
            self.tracer.log(
                "phase4.submitted",
                output={"patch_length": len(e.messages[0].get("extra", {}).get("submission", "")),
                        "patch_preview": e.messages[0].get("extra", {}).get("submission", "")[:500]},
                decision="SUBMITTED",
                phase="VERIFY", trajectory_id=traj.trajectory_id, step=traj.step,
            )
            self._finish_trajectory(traj, e)
            return
        except InterruptAgentFlow as e:
            traj.agent.add_messages(*e.messages)
            obs_messages = []

        observation = ""
        if isinstance(obs_messages, list) and obs_messages:
            observation = obs_messages[0].get("content", "")[:2000]

        self.tracer.log(
            "phase4.command_output",
            output={"observation": observation},
            phase="VERIFY", trajectory_id=traj.trajectory_id, step=traj.step,
        )

        self._log_step(traj, phase="VERIFY", thought=thought[:200])
        self._check_finished(traj)

    # ---- Utilities ----

    def _ensure_phase_prompt(self, traj: Trajectory, phase: Phase) -> None:
        """Inject the phase-specific prompt if not already present."""
        phase_prompts = {
            Phase.SEARCH: SEARCH_PROMPT,
            Phase.PATCH: PATCH_PROMPT,
            Phase.VERIFY: VERIFY_PROMPT,
        }
        # PATCH prompt is injected by _inject_strategy_prompt, so skip here
        if phase == Phase.PATCH:
            return

        traj_phase_key = f"_phase_injected_{phase.value}"
        if not hasattr(traj, traj_phase_key):
            setattr(traj, traj_phase_key, True)
            traj.agent.add_messages({
                "role": "user",
                "content": phase_prompts[phase],
            })

    def _truncate_context(self, traj: Trajectory, max_messages: int = 80) -> None:
        """Truncate old observation messages to prevent context window overflow.

        Keeps the first 4 messages (system + instance prompt + phase prompts)
        and the last max_messages/2 messages. Drops observations from the middle.
        """
        messages = traj.agent.messages
        if len(messages) <= max_messages:
            return

        keep_start = 4  # System prompt, instance, phase prompt, etc.
        keep_end = max_messages // 2

        # Keep first few + last half, drop middle observations
        truncated = messages[:keep_start] + messages[-keep_end:]

        n_dropped = len(messages) - len(truncated)
        self.tracer.log(
            "context_truncation",
            input={"messages_before": len(messages), "max_messages": max_messages},
            output={"messages_after": len(truncated), "dropped": n_dropped},
            trajectory_id=traj.trajectory_id, step=traj.step,
        )

        traj.agent.set_messages(truncated)
        logger.info(f"Truncated {traj.trajectory_id}: {n_dropped} messages dropped ({len(truncated)} remaining)")

    def _finish_trajectory(self, traj: Trajectory, exception: InterruptAgentFlow):
        """Handle trajectory completion."""
        traj.status = "completed"
        if isinstance(exception, Submitted):
            traj.submitted = True
            traj.patch = exception.messages[0].get("extra", {}).get("submission", "")
            self.tracer.log(
                "trajectory.finished",
                output={"status": "submitted", "patch_length": len(traj.patch),
                        "patch": traj.patch},
                trajectory_id=traj.trajectory_id, step=traj.step,
            )
        elif isinstance(exception, LimitsExceeded):
            # The working-tree diff is captured uniformly at container teardown
            # by _capture_patch_if_missing (the single source-filtered chokepoint
            # for all non-Submitted paths), so we don't grab it here — that
            # avoids a second, divergent git-diff path.
            self.tracer.log(
                "trajectory.finished",
                output={"status": "limits_exceeded"},
                trajectory_id=traj.trajectory_id, step=traj.step,
            )

    def _check_finished(self, traj: Trajectory):
        if traj.agent.is_finished():
            traj.status = "completed"
            traj.submitted = traj.agent.get_submission() != ""
            traj.patch = traj.agent.get_submission()

    @staticmethod
    def _is_test_path(path: str) -> bool:
        """True if a repo path is a test/conftest file (excluded from patches)."""
        p = path.strip()
        base = p.rsplit("/", 1)[-1]
        return (
            "/tests/" in p or "/test/" in p
            or base.startswith("test_") or base.endswith("_test.py")
            or base == "conftest.py"
        )

    @classmethod
    def _filter_diff_to_source(cls, diff_text: str) -> str:
        """Drop test-file sections from a unified git diff.

        The submit path produces a CURATED `git diff -- <source files>`, but the
        fallback capture runs a blanket `git diff`, which can include edits to
        tracked test files. Filtering those out keeps fallback-captured patches
        comparable to submitted ones (SWE-bench re-applies the gold test_patch
        regardless, so test edits in a prediction are noise at best).
        """
        if not diff_text:
            return diff_text
        lines = diff_text.splitlines(keepends=True)
        sections: list[tuple[str, list[str]]] = []
        current_path = ""
        buf: list[str] = []
        for line in lines:
            if line.startswith("diff --git "):
                if buf:
                    sections.append((current_path, buf))
                buf = [line]
                # "diff --git a/<path> b/<path>" → take the b/ path
                parts = line.split()
                current_path = parts[3][2:] if len(parts) >= 4 else ""
            else:
                buf.append(line)
        if buf:
            sections.append((current_path, buf))
        kept = [
            "".join(body) for path, body in sections
            if not (path and cls._is_test_path(path))
        ]
        return "".join(kept).strip()

    def _git_diff_source_only(self, traj: Trajectory) -> str:
        """Run `git diff` in the live container and strip test-file sections."""
        result = traj.env.execute({"command": "cd /testbed && git diff --no-color"})
        return self._filter_diff_to_source(result.get("output", "").strip())

    def _capture_patch_if_missing(self, traj: Trajectory) -> None:
        """Capture the working-tree diff before a container is destroyed.

        The run loop can mark a trajectory 'completed' via the step-limit cap,
        and a trajectory can also stop on a format error or write a summary
        instead of the exact submit command — none of which raise Submitted /
        LimitsExceeded, the two paths that normally populate traj.patch. In
        those cases the agent may still have made real edits. Grab the git diff
        now, while the container is alive, so the branch isn't silently dropped
        from diverse-pass@1. No-op if a patch was already captured. The diff is
        filtered to source files to match the curated submit path.
        """
        if traj.patch:
            return
        try:
            patch = self._git_diff_source_only(traj)
        except Exception as e:
            logger.warning(f"Failed to capture fallback patch for {traj.trajectory_id}: {e}")
            return
        if patch:
            traj.patch = patch
            self.tracer.log(
                "trajectory.patch_captured_fallback",
                output={"patch_length": len(patch), "patch": patch},
                trajectory_id=traj.trajectory_id, step=traj.step,
            )
            logger.info(
                f"Captured fallback patch for {traj.trajectory_id} "
                f"({len(patch)}ch) before cleanup"
            )

    def _log_step(self, traj: Trajectory, **kwargs):
        entry = {
            "timestamp": time.time(),
            "trajectory_id": traj.trajectory_id,
            "step": traj.step,
            **kwargs,
        }
        self.step_log.append(entry)

        log_path = os.path.join(self.manager.results_dir, "phased_decisions.log")
        with open(log_path, "a", encoding="utf-8") as f:
            phase = kwargs.get("phase", "?")
            thought = kwargs.get("thought", "")
            action = kwargs.get("action", "")
            strategy = kwargs.get("strategy", "")
            f.write(f"[{traj.trajectory_id}] step {traj.step} ({phase})")
            if kwargs.get("blocked"):
                f.write(f" BLOCKED: {kwargs['blocked']}")
            if kwargs.get("relevance"):
                f.write(f" relevance={kwargs['relevance']:.3f}")
            f.write(f"\n  THOUGHT: {thought}\n")
            if kwargs.get("summary"):
                f.write(f"  SUMMARY: {kwargs['summary']}\n")
            if action:
                f.write(f"  ACTION: {action}\n")
            if strategy:
                f.write(f"  STRATEGY: {strategy}\n")
            f.write("\n")

    def _log_strategy_proposal(self, strategies, clusters, entropy, unique_strategies):
        log_path = os.path.join(self.manager.results_dir, "phased_decisions.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"STRATEGY PROPOSAL\n")
            f.write(f"Proposed: {len(strategies)} | Clusters: {len(clusters)} | ")
            # 6 decimals: the post-hoc tau sweep compares this against achievable-
            # entropy grid values at the gate boundary; 3 decimals can round
            # ACROSS the boundary (e.g. partition (2,2,1) of 5: 1.054920 -> 1.055
            # > 1.0549). tau_sweep.py additionally recomputes the exact value
            # from the cluster partition, but the logged value should not lie.
            f.write(f"Unique: {len(unique_strategies)} | Entropy: {entropy:.6f}\n")
            f.write(f"{'='*70}\n")
            for i, s in enumerate(strategies):
                cluster_id = "?"
                for ci, cl in enumerate(clusters):
                    if i in cl.indices:
                        cluster_id = str(ci)
                f.write(f"  [{i+1}] cluster={cluster_id}: {s[:200]}\n")
            f.write(f"\nUnique strategies to fork:\n")
            for i, s in enumerate(unique_strategies):
                f.write(f"  → Strategy {i}: {s[:200]}\n")
            f.write("\n")

    def _save_logs(self):
        log_path = os.path.join(self.manager.results_dir, "step_log.json")
        with open(log_path, "w") as f:
            json.dump(self.step_log, f, indent=2)

    def _collect_results(self, elapsed: float, total_steps: int) -> dict:
        completed = self.manager.completed_trajectories
        # One entry per genuine draw (completed OR failed; patch may be "") —
        # see collect_patch_entries for the metric-correctness contract.
        patches = collect_patch_entries(
            self.manager.trajectories.values(), self.trajectory_strategies,
        )

        results = {
            "instance_id": self.instance_id,
            "elapsed_seconds": elapsed,
            "total_steps": total_steps,
            "total_trajectories": len(self.manager.trajectories),
            "completed_trajectories": len(completed),
            "submitted_trajectories": sum(1 for t in completed if t.submitted),
            "patches": patches,
            "branching_events": len(self.manager.branching_log),
            "strategies_proposed": len(self.trajectory_strategies),
        }

        # Log final results summary
        self.tracer.log(
            "pipeline.results",
            output={
                "instance_id": self.instance_id,
                "elapsed_seconds": round(elapsed, 1),
                "total_steps": total_steps,
                "total_trajectories": results["total_trajectories"],
                "completed_trajectories": results["completed_trajectories"],
                "submitted_trajectories": results["submitted_trajectories"],
                "n_patches": len(patches),
                "patches_summary": [
                    {"trajectory_id": p["trajectory_id"],
                     "submitted": p["submitted"],
                     "patch_length": len(p["patch"]),
                     "parent_id": p["parent_id"],
                     "branch_type": (p.get("branch_info") or {}).get("type", "root"),
                     "strategy": p["strategy"][:200]}
                    for p in patches
                ],
                "branching_tree": [
                    {"id": t.trajectory_id, "parent": t.parent_id,
                     "status": t.status, "steps": t.step,
                     "submitted": t.submitted,
                     "branch_info": t.branch_info,
                     "patch_length": len(t.patch) if t.patch else 0}
                    for t in sorted(self.manager.trajectories.values(),
                                    key=lambda t: t.trajectory_id)
                ],
            },
        )

        summary_path = os.path.join(self.manager.results_dir, "metadata.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def _print_tree(self):
        """Print the branching tree with phase and strategy annotations."""
        print(f"\n{'='*70}")
        print(f"  Strategy Branching Tree: {self.instance_id}")
        print(f"{'='*70}")

        for traj in sorted(self.manager.trajectories.values(), key=lambda t: t.trajectory_id):
            phase = self.trajectory_phases.get(traj.trajectory_id, Phase.SEARCH)
            strategy = self.trajectory_strategies.get(traj.trajectory_id, "")
            icon = {"completed": "✓" if traj.submitted else "○",
                    "branched": "⑂", "pruned": "✂",
                    "failed": "✗", "active": "…"}.get(traj.status, "?")
            patch_info = f" patch={len(traj.patch)}ch" if traj.patch else ""
            parent = f" ← {traj.parent_id}" if traj.parent_id else ""
            strat_info = f" [{strategy[:60]}]" if strategy else ""
            print(f"  {icon} {traj.trajectory_id} [{phase.value}] {traj.step}steps{patch_info}{parent}{strat_info}")

        print()
