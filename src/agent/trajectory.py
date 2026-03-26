"""Trajectory management for branching agent.

A Trajectory represents one independent agent execution path. When branching
occurs, the parent trajectory is frozen and child trajectories are created
with cloned state.
"""

import copy
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

from src.agent.branching_agent import BranchingAgent
from src.diversity.clustering import SemanticCluster
from src.utils.docker_helpers import clone_container_state

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """Represents one agent trajectory (possibly branched)."""

    trajectory_id: str
    """Unique ID, e.g. 't0', 't0_b5_c0' (branched at step 5, cluster 0)."""

    agent: BranchingAgent
    """The mini-swe-agent DefaultAgent subclass for this trajectory."""

    env: DockerEnvironment
    """The Docker environment for this trajectory."""

    parent_id: str | None = None
    """ID of the parent trajectory (None for root)."""

    branch_step: int | None = None
    """Step at which this trajectory was created via branching."""

    cluster_id: int | None = None
    """Which cluster this trajectory represents."""

    status: str = "active"
    """One of: active, completed, branched, failed."""

    step: int = 0
    """Current step count for this trajectory."""

    last_branch_step: int = -100
    """Step at which this trajectory last branched (for cooldown)."""

    patch: str = ""
    """Final patch (set on completion)."""

    submitted: bool = False
    """Whether the agent submitted a solution."""

    branch_prob: float = 1.0
    """Probability of the candidate that created this branch.
    Greedy (candidates[0]) = 1.0, temperature samples = lower.
    Used for pruning: when trajectories converge, keep the highest-probability one."""

    branch_info: dict = field(default_factory=dict)
    """Metadata about the branching event that created this trajectory."""

    def save(self, output_dir: str) -> Path:
        """Save this trajectory to a file using mini-swe-agent's serialization."""
        path = Path(output_dir) / f"trajectory_{self.trajectory_id}.traj.json"
        self.agent.save(
            path,
            {
                "info": {
                    "trajectory_id": self.trajectory_id,
                    "parent_id": self.parent_id,
                    "branch_step": self.branch_step,
                    "cluster_id": self.cluster_id,
                    "status": self.status,
                    "total_steps": self.step,
                }
            },
        )
        return path

    def cleanup(self):
        """Stop and remove the Docker container.

        Overrides mini-swe-agent's cleanup which uses Unix shell syntax
        that fails on Windows. We use direct docker rm -f instead.
        """
        import subprocess
        container_id = getattr(self.env, "container_id", None)
        if container_id:
            try:
                subprocess.run(
                    ["docker", "rm", "-f", container_id],
                    capture_output=True, timeout=30,
                )
                self.env.container_id = None  # Prevent double cleanup in __del__
            except Exception as e:
                logger.warning(f"Failed to cleanup {self.trajectory_id}: {e}")


class TrajectoryManager:
    """Manages the lifecycle of multiple trajectories for one problem instance."""

    def __init__(
        self,
        instance_id: str,
        max_trajectories: int = 30,
        results_dir: str = "results/branching",
    ):
        self.instance_id = instance_id
        self.max_trajectories = max_trajectories
        self.results_dir = os.path.join(results_dir, instance_id)
        os.makedirs(self.results_dir, exist_ok=True)

        self.trajectories: dict[str, Trajectory] = {}
        self._next_branch_id = 0

        # Branching event log
        self.branching_log: list[dict] = []

    def create_root(
        self,
        agent_config: dict,
        model_config: dict,
        env_config: dict,
        task: str,
    ) -> Trajectory:
        """Create the initial (root) trajectory.

        Args:
            agent_config: Agent config dict (system_template, instance_template, etc.)
            model_config: Model config dict (model_name, model_kwargs, etc.)
            env_config: Environment config dict (image, cwd, timeout, etc.)
            task: The problem statement.

        Returns:
            The root Trajectory.
        """
        env = DockerEnvironment(**env_config)
        model = LitellmTextbasedModel(**model_config)
        agent = BranchingAgent(model=model, env=env, **agent_config)

        # Initialize the agent (set up messages with system + instance templates)
        agent.extra_template_vars |= {"task": task}
        agent.messages = []
        agent.add_messages(
            agent.model.format_message(
                role="system",
                content=agent._render_template(agent.config.system_template),
            ),
            agent.model.format_message(
                role="user",
                content=agent._render_template(agent.config.instance_template),
            ),
        )

        traj = Trajectory(
            trajectory_id="t0",
            agent=agent,
            env=env,
        )
        self.trajectories["t0"] = traj
        logger.info(f"Created root trajectory t0 for {self.instance_id}")
        return traj

    def branch(
        self,
        parent: Trajectory,
        clusters: list[SemanticCluster],
        candidates: list[str],
        entropy: float,
    ) -> list[Trajectory]:
        """Create one new trajectory per cluster from a parent trajectory.

        For each cluster:
        1. Start a new DockerEnvironment from the same image
        2. Clone modified files from parent container
        3. Create a new BranchingAgent with deep-copied messages
        4. Inject the cluster representative response

        Args:
            parent: The parent trajectory to branch from.
            clusters: List of semantic clusters.
            candidates: List of candidate response strings.
            entropy: The semantic entropy that triggered branching.

        Returns:
            List of newly created trajectories.
        """
        new_trajectories = []

        for cluster in clusters:
            if not self.can_branch(1):
                logger.warning(
                    f"Trajectory cap reached ({self.max_trajectories}), "
                    f"skipping remaining clusters"
                )
                break

            # Get the representative candidate for this cluster
            rep_idx = cluster.representative_idx
            rep_content = candidates[rep_idx]

            # Compute branch probability:
            # - Greedy response (candidates[0]) gets prob=1.0
            # - Temperature samples get prob proportional to cluster size / N
            # - Multiply by parent's branch_prob for cumulative probability
            is_greedy_cluster = 0 in cluster.indices
            cluster_prob = 1.0 if is_greedy_cluster else len(cluster.indices) / len(candidates)
            branch_prob = parent.branch_prob * cluster_prob

            # Generate trajectory ID
            traj_id = f"{parent.trajectory_id}_b{parent.step}_c{cluster.representative_idx}"

            try:
                traj = self._create_branch(
                    parent=parent,
                    traj_id=traj_id,
                    response_content=rep_content,
                    cluster=cluster,
                    branch_prob=branch_prob,
                )
                new_trajectories.append(traj)
                self.trajectories[traj_id] = traj
            except Exception as e:
                logger.error(f"Failed to create branch {traj_id}: {e}")
                continue

        # Mark parent as branched and release its container
        parent.status = "branched"
        parent.cleanup()

        # Log branching event
        self.branching_log.append(
            {
                "timestamp": time.time(),
                "parent_id": parent.trajectory_id,
                "parent_step": parent.step,
                "entropy": entropy,
                "n_clusters": len(clusters),
                "n_branches_created": len(new_trajectories),
                "branch_ids": [t.trajectory_id for t in new_trajectories],
                "cluster_sizes": [len(c.indices) for c in clusters],
                "cluster_intents": [
                    c.intents[0][:100] if c.intents else "" for c in clusters
                ],
            }
        )

        logger.info(
            f"Branched {parent.trajectory_id} at step {parent.step}: "
            f"{len(new_trajectories)} new trajectories "
            f"(entropy={entropy:.3f}, {len(clusters)} clusters)"
        )

        return new_trajectories

    def _create_branch(
        self,
        parent: Trajectory,
        traj_id: str,
        response_content: str,
        cluster: SemanticCluster,
        branch_prob: float = 1.0,
    ) -> Trajectory:
        """Create a single branched trajectory."""
        # Get parent's environment config to create a matching new environment
        env_config = parent.env.config.model_dump()
        new_env = DockerEnvironment(**env_config)

        # Clone container state (modified files) from parent to new container
        clone_container_state(
            source_container_id=parent.env.container_id,
            target_container_id=new_env.container_id,
        )

        # Create new model (each agent needs its own model instance)
        model_config = parent.agent.model.config.model_dump()
        new_model = LitellmTextbasedModel(**model_config)

        # Create new agent with copied messages
        agent_config = parent.agent.config.model_dump()
        new_agent = BranchingAgent(model=new_model, env=new_env, **agent_config)

        # Copy message history up to (but NOT including) the last assistant message
        # The last assistant message was the greedy response; we replace it with
        # the cluster representative
        messages = parent.agent.clone_messages()
        # Remove the last message (greedy response that was added by query_only)
        if messages and messages[-1].get("role") == "assistant":
            messages.pop()
        new_agent.set_messages(messages)
        new_agent.n_calls = parent.agent.n_calls
        new_agent.cost = parent.agent.cost
        new_agent.extra_template_vars = copy.deepcopy(parent.agent.extra_template_vars)

        # Inject the cluster representative response and execute it
        new_agent.inject_and_execute(response_content)

        traj = Trajectory(
            trajectory_id=traj_id,
            agent=new_agent,
            env=new_env,
            parent_id=parent.trajectory_id,
            branch_step=parent.step,
            cluster_id=cluster.representative_idx,
            status="active",
            step=parent.step,  # Continue from parent's step count
            last_branch_step=parent.step,
            branch_prob=branch_prob,
            branch_info={
                "cluster_size": len(cluster.indices),
                "cluster_intents": cluster.intents[:3],
                "response_preview": response_content[:200],
                "branch_prob": branch_prob,
            },
        )

        logger.info(f"Created branch {traj_id} from {parent.trajectory_id}")
        return traj

    @property
    def active_trajectories(self) -> list[Trajectory]:
        """Return trajectories with status == 'active'."""
        return [t for t in self.trajectories.values() if t.status == "active"]

    @property
    def completed_trajectories(self) -> list[Trajectory]:
        """Return trajectories that completed (submitted or hit limit)."""
        return [t for t in self.trajectories.values() if t.status == "completed"]

    @property
    def total_count(self) -> int:
        """Total number of trajectories (all statuses)."""
        return len(self.trajectories)

    def can_branch(self, n_new: int) -> bool:
        """Check if adding n_new trajectories would exceed the cap."""
        # Count active + completed (branched parents don't count against cap)
        active_or_completed = sum(
            1 for t in self.trajectories.values() if t.status in ("active", "completed")
        )
        return active_or_completed + n_new <= self.max_trajectories

    def save_all(self) -> None:
        """Save all trajectories and the branching log."""
        for traj in self.trajectories.values():
            try:
                traj.save(self.results_dir)
            except Exception as e:
                logger.error(f"Failed to save trajectory {traj.trajectory_id}: {e}")

        # Save branching log
        log_path = os.path.join(self.results_dir, "branching_log.json")
        with open(log_path, "w") as f:
            json.dump(self.branching_log, f, indent=2)

    def cleanup_all(self) -> None:
        """Stop and remove all containers."""
        for traj in self.trajectories.values():
            traj.cleanup()
