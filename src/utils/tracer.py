"""Pipeline Tracer — meticulous input/output logging for every operation.

Logs every input and output at every point in the branching pipeline as
structured JSONL. Each entry captures what went in, what came out, and
any scores or decisions made.

Usage:
    tracer = PipelineTracer("results/branching/instance_id/trace.jsonl")
    tracer.log("phase1.search_step", input={...}, output={...})
"""

import json
import os
import time
from typing import Any


def _truncate(val: Any, max_len: int = 2000) -> Any:
    """Truncate strings for logging while preserving structure."""
    if isinstance(val, str) and len(val) > max_len:
        return val[:max_len] + f"... [{len(val) - max_len} chars truncated]"
    if isinstance(val, dict):
        return {k: _truncate(v, max_len) for k, v in val.items()}
    if isinstance(val, list):
        return [_truncate(v, max_len) for v in val]
    return val


class PipelineTracer:
    """Meticulous structured logger for the branching pipeline."""

    def __init__(self, output_path: str, truncate_content: int = 3000):
        self.output_path = output_path
        self.truncate_content = truncate_content
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Clear any existing trace file
        with open(output_path, "w", encoding="utf-8") as f:
            pass
        self._seq = 0

    def log(
        self,
        operation: str,
        input: dict | None = None,
        output: dict | None = None,
        scores: dict | None = None,
        decision: str | None = None,
        trajectory_id: str | None = None,
        phase: str | None = None,
        step: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Log a single operation with its inputs, outputs, and scores.

        Args:
            operation: Dotted name like "phase1.search_step" or "phase2.nli_classify"
            input: What went into the operation
            output: What came out
            scores: Any numerical scores produced
            decision: Human-readable decision string (e.g., "branch", "prune")
            trajectory_id: Which trajectory this belongs to
            phase: Current phase name
            step: Step number within the trajectory
            metadata: Any additional context
        """
        self._seq += 1
        entry = {
            "seq": self._seq,
            "time": time.time(),
            "operation": operation,
            "phase": phase,
            "trajectory_id": trajectory_id,
            "step": step,
        }
        if input is not None:
            entry["input"] = _truncate(input, self.truncate_content)
        if output is not None:
            entry["output"] = _truncate(output, self.truncate_content)
        if scores is not None:
            entry["scores"] = scores
        if decision is not None:
            entry["decision"] = decision
        if metadata is not None:
            entry["metadata"] = metadata

        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
