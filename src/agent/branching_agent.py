"""BranchingAgent: DefaultAgent subclass with single-step control for branching.

Exposes query_only() and execute_response() so the BranchingOrchestrator
can intercept between query and execution to decide whether to branch.
"""

import copy
import logging
import time

from minisweagent.agents.default import DefaultAgent, AgentConfig
from minisweagent.exceptions import LimitsExceeded, FormatError, InterruptAgentFlow

logger = logging.getLogger(__name__)


class BranchingAgent(DefaultAgent):
    """DefaultAgent with single-step control for the branching orchestrator.

    Key additions over DefaultAgent:
    - query_only(): Get the model response without executing actions
    - execute_response(): Execute a previously obtained response
    - inject_and_execute(): Inject a different response (from branching) and execute
    - Deep-copyable messages for trajectory cloning
    """

    def query_only(self) -> dict:
        """Query the model and return the response, but do NOT execute actions.

        This is the first half of DefaultAgent.step(). The orchestrator
        calls this to get the greedy response, then decides whether to branch.

        Returns:
            The model response message dict (with extra.actions).

        Raises:
            LimitsExceeded: If step/cost limits are exceeded.
            FormatError: If the model output is malformed.
        """
        # Same limit checking as DefaultAgent.query()
        if (
            0 < self.config.step_limit <= self.n_calls
            or 0 < self.config.cost_limit <= self.cost
        ):
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )

        self.n_calls += 1
        message = self.model.query(self.messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    def execute_response(self, message: dict) -> list[dict]:
        """Execute the actions from a previously obtained response message.

        This is the second half of DefaultAgent.step(). Called by the
        orchestrator after deciding not to branch.

        Args:
            message: The model response (from query_only or inject_and_execute).

        Returns:
            List of observation messages added to self.messages.
        """
        return self.execute_actions(message)

    def inject_and_execute(self, response_content: str) -> list[dict]:
        """Inject a non-greedy response (cluster representative) and execute it.

        Used when branching: this trajectory gets a different candidate than
        the greedy response. We need to:
        1. Parse the response into actions (using the model's parser)
        2. Build a proper message dict
        3. Add to messages
        4. Execute the actions

        Args:
            response_content: Raw text of the alternative response.

        Returns:
            List of observation messages.

        Raises:
            FormatError: If the response doesn't contain a valid action.
        """
        # Use the model's action parser to extract actions from the text
        # We create a minimal response-like object for _parse_actions
        from minisweagent.models.utils.actions_text import parse_regex_actions

        try:
            actions = parse_regex_actions(
                response_content,
                action_regex=self.model.config.action_regex,
                format_error_template=self.model.config.format_error_template,
            )
        except FormatError:
            # If parsing fails, the response doesn't have a valid action
            # Add it as-is with empty actions (orchestrator will handle)
            actions = []

        message = {
            "role": "assistant",
            "content": response_content,
            "extra": {
                "actions": actions,
                "cost": 0.0,  # No API cost for injected response
                "timestamp": time.time(),
                "injected": True,  # Flag that this was not the greedy response
            },
        }
        self.add_messages(message)

        if actions:
            return self.execute_actions(message)
        else:
            # No valid action — add an error observation
            error_msg = {
                "role": "user",
                "content": "No valid action found in the response. Please provide exactly one bash command.",
                "extra": {"injected_error": True},
            }
            return self.add_messages(error_msg)

    def clone_messages(self) -> list[dict]:
        """Deep copy the current message history for trajectory cloning."""
        return copy.deepcopy(self.messages)

    def set_messages(self, messages: list[dict]):
        """Replace the message history (used when creating branched trajectories)."""
        self.messages = messages

    def is_finished(self) -> bool:
        """Check if the agent has reached a terminal state."""
        if not self.messages:
            return False
        return self.messages[-1].get("role") == "exit"

    def get_submission(self) -> str:
        """Get the submitted patch if the agent has finished."""
        if not self.messages:
            return ""
        last = self.messages[-1]
        return last.get("extra", {}).get("submission", "")
