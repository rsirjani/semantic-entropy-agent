"""Core ReAct agent loop for SWE-bench."""

import re

from src.agent.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, OBSERVATION_TEMPLATE
from src.inference.vllm_client import VLLMClient
from src.utils.docker_helpers import SWEBenchContainer
from src.utils.logging import TrajectoryLogger


# Max chars for observation output before truncation
MAX_OBSERVATION_CHARS = 15000
TRUNCATION_MSG = "\n... [output truncated: {total} chars, showing first {head} and last {tail}] ...\n"


def truncate_output(output: str, max_chars: int = MAX_OBSERVATION_CHARS) -> str:
    """Truncate long command outputs, keeping head and tail."""
    if len(output) <= max_chars:
        return output
    head_size = max_chars // 2
    tail_size = max_chars // 2
    return (
        output[:head_size]
        + TRUNCATION_MSG.format(total=len(output), head=head_size, tail=tail_size)
        + output[-tail_size:]
    )


def parse_response(text: str) -> tuple[str, str]:
    """Parse a THOUGHT/ACTION response from the model.

    Returns (thought, action). If parsing fails, treats the whole
    response as thought with an empty action.
    """
    # Try to find THOUGHT: and ACTION: blocks
    thought_match = re.search(
        r"THOUGHT:\s*\n?(.*?)(?=\nACTION:|\Z)", text, re.DOTALL
    )
    action_match = re.search(
        r"ACTION:\s*\n?(.*?)(?=\nTHOUGHT:|\Z)", text, re.DOTALL
    )

    thought = thought_match.group(1).strip() if thought_match else text.strip()
    action = action_match.group(1).strip() if action_match else ""

    # Clean up action — remove markdown code fences if present
    action = re.sub(r"^```(?:bash|sh)?\s*\n?", "", action)
    action = re.sub(r"\n?```\s*$", "", action)
    action = action.strip()

    return thought, action


class ReactAgent:
    """Single-trajectory ReAct agent for SWE-bench."""

    def __init__(
        self,
        client: VLLMClient,
        container: SWEBenchContainer,
        logger: TrajectoryLogger,
        max_steps: int = 30,
        max_tokens_per_step: int = 4096,
        temperature: float = 0.0,
    ):
        self.client = client
        self.container = container
        self.logger = logger
        self.max_steps = max_steps
        self.max_tokens_per_step = max_tokens_per_step
        self.temperature = temperature

    def run(self, problem_statement: str) -> tuple[str, bool]:
        """Run the agent on a problem. Returns (patch, submitted)."""
        # Build initial messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                problem_statement=problem_statement
            )},
        ]

        self.logger.log_system(SYSTEM_PROMPT)
        self.logger.log_user(messages[1]["content"])

        submitted = False
        consecutive_errors = 0
        last_action = None

        for step in range(self.max_steps):
            # Get model response
            try:
                response = self.client.chat(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens_per_step,
                )
            except Exception as e:
                self.logger.log_observation(step, f"API error: {e}", 1)
                # Retry once after a brief message
                messages.append({"role": "user", "content": "The API returned an error. Please continue with your next step."})
                continue

            assistant_msg = response.choices[0].message.content or ""
            tokens_used = response.usage.completion_tokens if response.usage else 0

            # Parse thought and action
            thought, action = parse_response(assistant_msg)

            self.logger.log_assistant(
                step=step,
                thought=thought,
                action=action,
                tokens=tokens_used,
                raw_response=assistant_msg,
            )

            messages.append({"role": "assistant", "content": assistant_msg})

            # Check for submit
            if action.lower().strip() == "submit":
                submitted = True
                break

            # Check for empty action
            if not action:
                messages.append({"role": "user", "content": (
                    "OBSERVATION:\nNo action was detected in your response. "
                    "Please respond with the exact format:\n"
                    "THOUGHT:\n<your reasoning>\n\nACTION:\n<bash command or 'submit'>"
                )})
                self.logger.log_observation(step, "No action detected", 1)
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    break
                continue

            # Check for repeated action
            if action == last_action:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    messages.append({"role": "user", "content": (
                        "OBSERVATION:\nYou have repeated the same command multiple times. "
                        "Please try a different approach."
                    )})
                    self.logger.log_observation(step, "Repeated action detected", 1)
                    if consecutive_errors >= 5:
                        break
                    continue
            else:
                consecutive_errors = 0

            last_action = action

            # Execute bash command in container
            output, exit_code = self.container.exec_bash(action)

            # Truncate long output
            output = truncate_output(output)

            observation = OBSERVATION_TEMPLATE.format(
                exit_code=exit_code, output=output
            )

            self.logger.log_observation(step, output, exit_code)
            messages.append({"role": "user", "content": observation})

        # Extract patch
        patch = self.container.get_patch()

        token_usage = self.client.get_token_usage()
        self.logger.log_result(
            patch=patch,
            total_steps=step + 1 if 'step' in dir() else 0,
            total_tokens=token_usage["total_tokens"],
            submitted=submitted,
        )

        return patch, submitted
