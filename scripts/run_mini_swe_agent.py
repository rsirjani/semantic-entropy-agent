"""Run mini-swe-agent v2 on our 10 SWE-bench instances using local vLLM."""

import os
import sys

# Fix MSYS path conversion on Windows (Git Bash mangles /testbed, /bin/bash etc.)
os.environ["MSYS_NO_PATHCONV"] = "1"
os.environ["MSYS2_ARG_CONV_EXCL"] = "*"

# litellm needs this for local OpenAI-compatible endpoints
os.environ["OPENAI_API_KEY"] = "dummy"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Our 10 target instance IDs
TARGET_INSTANCES = [
    "sympy__sympy-12481",
    "sympy__sympy-16766",
    "sympy__sympy-18189",
    "sympy__sympy-12096",
    "sympy__sympy-15345",
    "sympy__sympy-23534",
    "sympy__sympy-22714",
    "sympy__sympy-19637",
    "sympy__sympy-18763",
    "sympy__sympy-19495",
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="*", default=None,
                        help="Specific instance IDs (default: all 10)")
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "results", "baseline_v2"),
                        help="Output directory")
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "swebench_local.yaml"),
                        help="Config file")
    parser.add_argument("--redo", action="store_true", help="Redo existing instances")
    args = parser.parse_args()

    instance_ids = args.instances or TARGET_INSTANCES
    filter_pattern = "|".join(instance_ids)

    # Build the command
    cmd = [
        sys.executable, "-m", "minisweagent.run.benchmarks.swebench",
        "--subset", "verified",
        "--split", "test",
        "--filter", filter_pattern,
        "-o", args.output,
        "-c", args.config,
    ]
    if args.redo:
        cmd.append("--redo-existing")

    print(f"Running mini-swe-agent v2 on {len(instance_ids)} instances")
    print(f"Output: {args.output}")
    print(f"Config: {args.config}")
    print(f"Command: {' '.join(cmd)}")

    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
