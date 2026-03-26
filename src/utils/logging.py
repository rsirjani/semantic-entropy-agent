"""Trajectory logging for agent runs."""

import json
import os
from datetime import datetime, timezone


class TrajectoryLogger:
    """Logs agent trajectory events to a JSONL file."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.events = []

    def log(self, event: dict) -> None:
        """Log an event with a timestamp."""
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.events.append(event)
        # Append to file immediately for crash resilience
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def log_system(self, content: str) -> None:
        self.log({"type": "system", "content": content})

    def log_user(self, content: str) -> None:
        self.log({"type": "user", "content": content})

    def log_assistant(self, step: int, thought: str, action: str,
                      tokens: int = 0, raw_response: str = "") -> None:
        self.log({
            "type": "assistant",
            "step": step,
            "thought": thought,
            "action": action,
            "tokens": tokens,
            "raw_response": raw_response,
        })

    def log_observation(self, step: int, output: str, exit_code: int) -> None:
        self.log({
            "type": "observation",
            "step": step,
            "output": output,
            "exit_code": exit_code,
        })

    def log_result(self, patch: str, total_steps: int, total_tokens: int,
                   submitted: bool = False) -> None:
        self.log({
            "type": "result",
            "patch": patch,
            "total_steps": total_steps,
            "total_tokens": total_tokens,
            "submitted": submitted,
        })
