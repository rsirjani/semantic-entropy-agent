"""BranchingOrchestrator: multi-trajectory agent with semantic branching.

Manages multiple BranchingAgent trajectories for one problem instance.
At each step, generates diverse candidates, clusters by semantic meaning,
and branches when semantic entropy exceeds a threshold.
"""

import json
import logging
import os
import time
from typing import Any

from minisweagent.exceptions import InterruptAgentFlow, Submitted, LimitsExceeded, FormatError

from src.agent.trajectory import Trajectory, TrajectoryManager
from src.diversity.clustering import SemanticClusterer
from src.diversity.intent import IntentExtractor
from src.diversity.nli import NLIModel
from src.diversity.temperature_sampler import TemperatureSampler

logger = logging.getLogger(__name__)


class BranchingOrchestrator:
    """Orchestrates multi-trajectory branching agent for one problem instance.

    Main loop:
    1. Create root trajectory
    2. Round-robin across active trajectories, one step at a time
    3. At each step: generate diverse candidates, cluster, maybe branch
    4. Collect patches from all completed trajectories
    """

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

        # Diversity components
        self.nli = nli_model
        self.clusterer = SemanticClusterer(
            nli=nli_model,
            entailment_threshold=branching_config.get("entailment_threshold", 0.5),
        )
        self.intent_extractor = IntentExtractor(
            model_name=model_config.get("model_name", "openai/qwen3-coder"),
            model_kwargs=model_config.get("model_kwargs", {}),
            method=branching_config.get("intent_method", "llm"),
        )
        # Diversity generator: SDLG (gradient-based) or temperature sampling
        diversity_method = branching_config.get("diversity_method", "sdlg")
        n_candidates = branching_config.get("n_candidates", 5)
        if diversity_method == "sdlg":
            from src.diversity.sdlg import SDLGGenerator
            self.sampler = SDLGGenerator(
                nli_model=nli_model,
                n_candidates=n_candidates,
            )
            logger.info("Using SDLG for diverse generation")
        else:
            self.sampler = TemperatureSampler(
                n_candidates=n_candidates,
                temperature=branching_config.get("sample_temperature", 0.7),
            )
            logger.info("Using temperature sampling for diverse generation")

        # Branching parameters
        self.tau = branching_config.get("entropy_threshold", 0.5)
        self.max_trajectories = branching_config.get("max_trajectories", 30)
        self.branch_after_step = branching_config.get("branch_after_step", 3)
        self.min_steps_between_branches = branching_config.get("min_steps_between_branches", 2)
        self.max_steps = agent_config.get("step_limit", 30)

        # Trajectory manager
        self.manager = TrajectoryManager(
            instance_id=instance_id,
            max_trajectories=self.max_trajectories,
            results_dir=branching_config.get("results_dir", "results/branching"),
        )

        # Step-level log for tree visualization
        self.step_log: list[dict] = []

    def run(self) -> dict:
        """Run the branching agent to completion.

        Returns dict with patches, branching info, and metadata.
        """
        start_time = time.time()

        # Create root trajectory
        root = self.manager.create_root(
            agent_config=self.agent_config,
            model_config=self.model_config,
            env_config=self.env_config,
            task=self.problem_statement,
        )

        total_steps = 0

        try:
            while self.manager.active_trajectories:
                # Round-robin: process each active trajectory once
                active = list(self.manager.active_trajectories)

                for traj in active:
                    if traj.status != "active":
                        continue  # May have been branched by another trajectory

                    try:
                        self._step_trajectory(traj)
                        total_steps += 1
                    except Exception as e:
                        logger.error(
                            f"Error in trajectory {traj.trajectory_id} "
                            f"step {traj.step}: {e}"
                        )
                        traj.status = "failed"

                # After each round: prune redundant trajectories
                self._prune_redundant_trajectories()

                # Safety: check if we're making progress
                if total_steps > self.max_steps * self.max_trajectories:
                    logger.warning("Global step limit reached, stopping all trajectories")
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            # Save all trajectories
            self.manager.save_all()

            # Save step log for tree visualization
            self._save_step_log()

        elapsed = time.time() - start_time

        # Collect results
        results = self._collect_results(elapsed, total_steps)

        # Print tree visualization
        self._print_tree()

        return results

    def _step_trajectory(self, traj: Trajectory) -> None:
        """Execute one step on a single trajectory.

        1. Query model (greedy)
        2. If branching eligible: generate diverse candidates, cluster, maybe branch
        3. Otherwise: execute greedy response
        """
        traj.step += 1

        # Step 1: Get greedy response
        try:
            greedy_message = traj.agent.query_only()
        except (LimitsExceeded, Submitted) as e:
            # Agent finished (hit limit or submitted)
            traj.agent.add_messages(*e.messages)
            self._finish_trajectory(traj, e)
            return
        except FormatError as e:
            # Model output was malformed — add error message and continue
            traj.agent.add_messages(*e.messages)
            self._log_step(traj, greedy_content="[FORMAT_ERROR]", branched=False)
            return

        greedy_content = greedy_message.get("content", "")

        # Step 2: Check if branching is eligible
        branching_eligible = (
            traj.step > self.branch_after_step
            and (traj.step - traj.last_branch_step) >= self.min_steps_between_branches
            and self.manager.can_branch(2)  # Need at least 2 new trajectories
        )

        if not branching_eligible:
            # Just execute the greedy response
            try:
                traj.agent.execute_response(greedy_message)
            except Submitted as e:
                traj.agent.add_messages(*e.messages)
                self._finish_trajectory(traj, e)
                return
            except InterruptAgentFlow as e:
                traj.agent.add_messages(*e.messages)

            self._log_step(traj, greedy_content=greedy_content, branched=False)
            self._check_finished(traj)
            return

        # Step 3: Generate diverse candidates
        candidates = self.sampler.generate(
            model_name=self.model_config["model_name"],
            model_kwargs=self.model_config.get("model_kwargs", {}),
            messages=traj.agent.messages[:-1],  # Messages before greedy was appended
            greedy_response=greedy_content,
        )

        # Step 4: Extract intents — summarize (full history + each candidate)
        # Messages before greedy was appended = the shared history
        history_messages = traj.agent.messages[:-1]
        intents = self.intent_extractor.extract_batch_with_history(
            candidates, history_messages
        )

        # Step 5: Cluster and compute entropy
        analysis = self.clusterer.analyze(
            intents, tau=self.tau, context=self.problem_statement[:500]
        )
        clusters = analysis["clusters"]
        entropy = analysis["entropy"]

        # Log full detail for inspection
        self._log_branching_decision(
            traj, candidates, intents, clusters, entropy, analysis["should_branch"]
        )

        # Step 6: Branch or proceed
        if analysis["should_branch"] and len(clusters) >= 2:
            # Branch! Create new trajectories for each cluster
            new_trajectories = self.manager.branch(
                parent=traj,
                clusters=clusters,
                candidates=candidates,
                entropy=entropy,
            )

            self._log_step(
                traj,
                greedy_content=greedy_content,
                branched=True,
                entropy=entropy,
                n_clusters=len(clusters),
                intents=intents,
                candidates=[c[:300] for c in candidates],
                cluster_indices=[c.indices for c in clusters],
                branch_ids=[t.trajectory_id for t in new_trajectories],
            )
        else:
            # Don't branch — execute greedy
            try:
                traj.agent.execute_response(greedy_message)
            except Submitted as e:
                traj.agent.add_messages(*e.messages)
                self._finish_trajectory(traj, e)
                return
            except InterruptAgentFlow as e:
                traj.agent.add_messages(*e.messages)

            self._log_step(
                traj,
                greedy_content=greedy_content,
                branched=False,
                entropy=entropy,
                n_clusters=len(clusters),
                intents=intents,
                candidates=[c[:300] for c in candidates],
                cluster_indices=[c.indices for c in clusters],
            )
            self._check_finished(traj)

    def _finish_trajectory(self, traj: Trajectory, exception: InterruptAgentFlow):
        """Handle trajectory completion."""
        traj.status = "completed"
        if isinstance(exception, Submitted):
            traj.submitted = True
            traj.patch = exception.messages[0].get("extra", {}).get("submission", "")
            logger.info(
                f"Trajectory {traj.trajectory_id} submitted "
                f"(patch: {len(traj.patch)} chars)"
            )
        elif isinstance(exception, LimitsExceeded):
            # Try to get the current patch even if limit was hit
            try:
                result = traj.env.execute({"command": "cd /testbed && git diff --no-color"})
                traj.patch = result.get("output", "").strip()
            except Exception:
                pass
            logger.info(f"Trajectory {traj.trajectory_id} hit limits")

    def _check_finished(self, traj: Trajectory):
        """Check if trajectory reached terminal state."""
        if traj.agent.is_finished():
            traj.status = "completed"
            traj.submitted = traj.agent.get_submission() != ""
            traj.patch = traj.agent.get_submission()

    def _prune_redundant_trajectories(self) -> None:
        """Prune active trajectories that are semantically equivalent.

        After each round of steps, compare the latest response (thought) of
        every active trajectory. If two trajectories bidirectionally entail
        each other, they're pursuing the same approach — keep the older one
        (more history invested) and prune the newer one.

        This prevents wasting the B=30 budget on converged trajectories.
        """
        active = self.manager.active_trajectories
        if len(active) <= 1:
            return

        # Summarize each active trajectory's overall approach via LLM
        traj_intents: list[tuple[Trajectory, str]] = []
        for traj in active:
            intent = self.intent_extractor.extract_trajectory_intent(traj.agent.messages)
            if intent:
                traj_intents.append((traj, intent))

        if len(traj_intents) <= 1:
            return

        # Cluster active trajectories by their latest intents
        intents = [intent for _, intent in traj_intents]
        clusters = self.clusterer.cluster(intents, context=self.problem_statement[:500])

        # For each cluster with >1 trajectory, keep highest-probability and prune rest
        pruned_ids = []
        for cluster in clusters:
            if len(cluster.indices) <= 1:
                continue

            # Sort by branch_prob descending (keep highest probability),
            # then by step count descending (more investment as tiebreaker)
            cluster_trajs = [traj_intents[i][0] for i in cluster.indices]
            cluster_trajs.sort(key=lambda t: (-t.branch_prob, -t.step))

            # Keep the first (highest probability), prune the rest
            keeper = cluster_trajs[0]
            for traj in cluster_trajs[1:]:
                traj.status = "pruned"
                traj.cleanup()
                pruned_ids.append(traj.trajectory_id)
                logger.info(
                    f"Pruned {traj.trajectory_id} (p={traj.branch_prob:.3f}) "
                    f"— redundant with {keeper.trajectory_id} (p={keeper.branch_prob:.3f})"
                )

        if pruned_ids:
            self.step_log.append({
                "timestamp": time.time(),
                "event": "prune",
                "pruned_ids": pruned_ids,
                "active_remaining": len(self.manager.active_trajectories),
            })
            self.manager.branching_log.append({
                "timestamp": time.time(),
                "event": "prune",
                "pruned_ids": pruned_ids,
                "reason": "cross-trajectory semantic redundancy",
            })

    def _log_branching_decision(
        self,
        traj: Trajectory,
        candidates: list[str],
        intents: list[str],
        clusters: list,
        entropy: float,
        should_branch: bool,
    ) -> None:
        """Log full branching decision detail to a readable file.

        This lets us inspect exactly what the model generated, what the
        intent summaries were, how they clustered, and why we did/didn't branch.
        """
        from src.diversity.intent import extract_thought

        log_path = os.path.join(self.manager.results_dir, "branching_decisions.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"TRAJECTORY: {traj.trajectory_id}  |  STEP: {traj.step}\n")
            f.write(f"ENTROPY: {entropy:.3f}  |  CLUSTERS: {len(clusters)}  |  BRANCH: {should_branch}\n")
            f.write(f"{'='*80}\n\n")

            for i, (cand, intent) in enumerate(zip(candidates, intents)):
                tag = "GREEDY" if i == 0 else f"SAMPLE-{i}"
                # Find which cluster this candidate belongs to
                cluster_id = "?"
                for ci, cl in enumerate(clusters):
                    if i in cl.indices:
                        cluster_id = str(ci)
                        break
                thought = extract_thought(cand)[:200]
                f.write(f"  [{i}] {tag}  (cluster {cluster_id})\n")
                f.write(f"      THOUGHT: {thought}\n")
                f.write(f"      INTENT:  {intent}\n\n")

            f.write(f"  CLUSTER SUMMARY:\n")
            for ci, cl in enumerate(clusters):
                f.write(f"    Cluster {ci}: indices={cl.indices} (size {len(cl.indices)})\n")

            if should_branch:
                f.write(f"\n  >>> BRANCHING into {len(clusters)} trajectories\n")
            else:
                f.write(f"\n  >>> NO BRANCH (entropy {entropy:.3f} <= tau {self.tau})\n")
            f.write("\n")

    def _log_step(self, traj: Trajectory, **kwargs):
        """Log a step for tree visualization."""
        entry = {
            "timestamp": time.time(),
            "trajectory_id": traj.trajectory_id,
            "step": traj.step,
            "status": traj.status,
            **kwargs,
        }
        self.step_log.append(entry)

    def _save_step_log(self):
        """Save the step log for analysis."""
        log_path = os.path.join(self.manager.results_dir, "step_log.json")
        with open(log_path, "w") as f:
            json.dump(self.step_log, f, indent=2)

    def _collect_results(self, elapsed: float, total_steps: int) -> dict:
        """Collect results from all trajectories."""
        completed = self.manager.completed_trajectories
        all_trajs = list(self.manager.trajectories.values())

        patches = []
        for traj in completed:
            if traj.patch:
                patches.append(
                    {
                        "trajectory_id": traj.trajectory_id,
                        "patch": traj.patch,
                        "submitted": traj.submitted,
                        "steps": traj.step,
                        "parent_id": traj.parent_id,
                        "branch_step": traj.branch_step,
                    }
                )

        results = {
            "instance_id": self.instance_id,
            "elapsed_seconds": elapsed,
            "total_steps": total_steps,
            "total_trajectories": len(all_trajs),
            "completed_trajectories": len(completed),
            "submitted_trajectories": sum(1 for t in completed if t.submitted),
            "patches": patches,
            "branching_events": len(self.manager.branching_log),
            "branching_log": self.manager.branching_log,
        }

        # Save summary
        summary_path = os.path.join(self.manager.results_dir, "metadata.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def _print_tree(self):
        """Print an ASCII tree visualization of the branching structure."""
        print(f"\n{'='*70}")
        print(f"  Branching Tree: {self.instance_id}")
        print(f"{'='*70}")

        if not self.manager.branching_log:
            print("  (no branching occurred — single trajectory)")
            root = self.manager.trajectories.get("t0")
            if root:
                status_icon = "✓" if root.submitted else "✗"
                print(f"  t0 [{root.step} steps] {status_icon} patch={len(root.patch)} chars")
            print()
            return

        # Build tree structure
        children: dict[str | None, list[str]] = {None: []}
        for traj in self.manager.trajectories.values():
            parent = traj.parent_id
            if parent not in children:
                children[parent] = []
            children[parent].append(traj.trajectory_id)

        # Print tree recursively
        roots = children.get(None, [])
        for root_id in sorted(roots):
            self._print_node(root_id, children, prefix="  ", is_last=True)

        # Print branching summary
        pruned = [t for t in self.manager.trajectories.values() if t.status == "pruned"]
        print(f"\n  Summary:")
        print(f"    Total trajectories: {self.manager.total_count}")
        print(f"    Pruned (redundant): {len(pruned)}")
        branch_events = [e for e in self.manager.branching_log if e.get("event") != "prune"]
        prune_events = [e for e in self.manager.branching_log if e.get("event") == "prune"]
        print(f"    Branching events: {len(branch_events)}")
        for event in branch_events:
            print(
                f"    Step {event['parent_step']}: "
                f"entropy={event['entropy']:.3f}, "
                f"{event['n_clusters']} clusters → "
                f"{event['n_branches_created']} branches"
            )
        if prune_events:
            print(f"    Prune events: {len(prune_events)}")
            for event in prune_events:
                print(f"      Pruned: {event['pruned_ids']}")

        # Print patches
        completed = self.manager.completed_trajectories
        if completed:
            print(f"\n  Completed trajectories:")
            for traj in completed:
                status = "SUBMITTED" if traj.submitted else "limit"
                print(
                    f"    {traj.trajectory_id}: {status}, "
                    f"{traj.step} steps, patch={len(traj.patch)} chars"
                )
        print()

    def _print_node(
        self,
        traj_id: str,
        children: dict,
        prefix: str = "",
        is_last: bool = True,
    ):
        """Recursively print a node in the branching tree."""
        traj = self.manager.trajectories.get(traj_id)
        if not traj:
            return

        # Connector characters
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "

        # Status indicator
        if traj.status == "completed":
            icon = "✓" if traj.submitted else "○"
        elif traj.status == "branched":
            icon = "⑂"
        elif traj.status == "pruned":
            icon = "✂"
        elif traj.status == "failed":
            icon = "✗"
        else:
            icon = "…"

        # Branch info
        branch_info = ""
        if traj.branch_step is not None:
            intent_preview = ""
            if traj.branch_info.get("cluster_intents"):
                intent_preview = f' "{traj.branch_info["cluster_intents"][0][:60]}"'
            branch_info = f" (branch@step{traj.branch_step}{intent_preview})"

        # Entropy info from branching log
        entropy_info = ""
        for event in self.manager.branching_log:
            if event.get("parent_id") == traj_id and "entropy" in event:
                entropy_info = f" H={event['entropy']:.2f}"
                break

        prob_info = f" p={traj.branch_prob:.2f}" if traj.branch_prob < 1.0 else ""
        patch_info = f" patch={len(traj.patch)}ch" if traj.patch else ""

        print(
            f"{prefix}{connector}{icon} {traj_id} "
            f"[{traj.step}steps]{prob_info}{entropy_info}{branch_info}{patch_info}"
        )

        # Print children
        child_ids = sorted(children.get(traj_id, []))
        for i, child_id in enumerate(child_ids):
            is_last_child = i == len(child_ids) - 1
            self._print_node(
                child_id,
                children,
                prefix=prefix + extension,
                is_last=is_last_child,
            )
