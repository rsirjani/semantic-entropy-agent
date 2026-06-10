r"""Budget-fairness audit (rubric R6.3).

Per-trajectory step distributions AND per-arm token/compute accounting, computed
purely from the artifacts a run leaves behind (no GPU/Docker). The specific
fairness concern (GOLD_STANDARD R6.3, VALIDATION_BRIEF change 3) is the step-limit
asymmetry: lazy strategy trajectories reset to step 0 and may use up to step_limit
(300) patch steps, vs the matched-k baseline's 250 total. This script SHOWS whether
passing branches actually exploited that headroom or submitted far earlier.

Per arm it reads, for each instance:
  - <results_dir>/<iid>/metadata.json   (patches[].steps, patches[].trajectory_id,
                                          total_steps, elapsed_seconds)
  - <results_dir>/<iid>/trajectory_*.traj.json  (per-message token usage from the
                                          litellm response: extra.response.usage)
  - <results_dir>/trajectory_eval_<iid>.json  (per-trajectory `resolved`)
joins steps to resolved by trajectory_id (NOT by index), and reports the step
distribution overall and FOR PASSING branches, plus how many passing branches
exceeded a reference cap (default 250 = the baseline's total budget).

Token accounting (R6.3 "per-arm token/compute accounting reported"): the
mini-swe-agent litellm model already stores the full provider response on every
assistant message (`extra.response.usage` with prompt/completion/total tokens), so
the per-arm token total is recoverable from the saved `.traj.json` transcripts with
NO change to the run loop. We sum tokens per trajectory and per arm; cost is left
out only because the local vLLM model is not registered for litellm cost
calculation (steps + tokens are the compute proxies).

Usage:
  python scripts/budget_audit.py --results-dir results/strategy_t0.7 \
      --eval results/strategy_t0.7 --reference-cap 250 \
      --out results/budget_audit_strategy_t0.7.json
"""

import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np


def _resolved_by_tid(eval_path: str) -> dict[str, dict[str, bool]]:
    files = ([eval_path] if eval_path.endswith(".json")
             else sorted(glob.glob(os.path.join(eval_path, "trajectory_eval_*.json"))))
    out: dict[str, dict[str, bool]] = {}
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        iid = d.get("instance_id")
        if not iid:
            continue
        for t in d.get("trajectories", []):
            tid = t.get("trajectory_id")
            if tid is None or tid == "primary":
                continue
            out.setdefault(iid, {})[tid] = bool(t.get("resolved"))
    return out


def _tid_from_traj_path(path: str) -> str:
    """`.../trajectory_t0_strategy_1.traj.json` -> `t0_strategy_1`."""
    base = os.path.basename(path)
    if base.endswith(".traj.json"):
        base = base[: -len(".traj.json")]
    if base.startswith("trajectory_"):
        base = base[len("trajectory_"):]
    return base


def _traj_tokens(traj_path: str) -> dict:
    """Sum prompt/completion/total tokens over a trajectory transcript.

    Tokens live on each assistant message at extra.response.usage (the full
    litellm response dump). Injected/branched responses carry no usage and are
    skipped, so we count only real model calls. Returns zeros if the transcript is
    unreadable or carries no usage (older runs).
    """
    prompt = completion = total = 0
    n_calls_with_usage = 0
    try:
        with open(traj_path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "n_calls_with_usage": 0}
    for m in (d.get("messages", []) if isinstance(d, dict) else []):
        usage = (m.get("extra", {}) or {}).get("response", {})
        usage = (usage or {}).get("usage") if isinstance(usage, dict) else None
        if not isinstance(usage, dict):
            continue
        pt = usage.get("prompt_tokens") or 0
        ct = usage.get("completion_tokens") or 0
        tt = usage.get("total_tokens")
        if tt is None:
            tt = pt + ct
        prompt += int(pt)
        completion += int(ct)
        total += int(tt)
        n_calls_with_usage += 1
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": total, "n_calls_with_usage": n_calls_with_usage}


def _tokens_by_tid(results_dir: str) -> dict[str, dict[str, dict]]:
    """{iid: {tid: token_dict}} over every trajectory transcript in the arm."""
    out: dict[str, dict[str, dict]] = {}
    for traj in sorted(glob.glob(os.path.join(results_dir, "*", "trajectory_*.traj.json"))):
        iid = os.path.basename(os.path.dirname(traj))
        tid = _tid_from_traj_path(traj)
        if tid == "primary":  # best-of duplicate, not an independent call set
            continue
        out.setdefault(iid, {})[tid] = _traj_tokens(traj)
    return out


def _dist(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "min": float(a.min()), "median": float(np.median(a)),
            "mean": round(float(a.mean()), 2), "max": float(a.max())}


def audit(results_dir: str, eval_path: str, reference_cap: int) -> dict:
    resolved = _resolved_by_tid(eval_path)
    tokens = _tokens_by_tid(results_dir)
    all_steps: list[float] = []
    passing_steps: list[float] = []
    total_steps_per_instance: list[float] = []
    elapsed_per_instance: list[float] = []
    over_cap_passing: list[dict] = []
    traj_total_tokens: list[float] = []
    passing_total_tokens: list[float] = []
    arm_prompt_tokens = arm_completion_tokens = arm_total_tokens = 0
    n_traj_with_tokens = 0
    n_instances = 0

    for meta in sorted(glob.glob(os.path.join(results_dir, "*", "metadata.json"))):
        try:
            with open(meta, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        iid = d.get("instance_id")
        if not iid:
            continue
        n_instances += 1
        if "total_steps" in d:
            total_steps_per_instance.append(float(d["total_steps"]))
        if "elapsed_seconds" in d:
            elapsed_per_instance.append(float(d["elapsed_seconds"]))
        res = resolved.get(iid, {})
        for p in d.get("patches", []):
            tid = p.get("trajectory_id")
            steps = p.get("steps")
            # Skip the best-of duplicate ("primary") so its steps are not counted
            # twice against the genuine trajectory it copies.
            if steps is None or tid == "primary":
                continue
            all_steps.append(float(steps))
            tok = tokens.get(iid, {}).get(tid)
            if tok and tok.get("n_calls_with_usage", 0) > 0:
                arm_prompt_tokens += tok["prompt_tokens"]
                arm_completion_tokens += tok["completion_tokens"]
                arm_total_tokens += tok["total_tokens"]
                traj_total_tokens.append(float(tok["total_tokens"]))
                n_traj_with_tokens += 1
            if res.get(tid):
                passing_steps.append(float(steps))
                if tok and tok.get("n_calls_with_usage", 0) > 0:
                    passing_total_tokens.append(float(tok["total_tokens"]))
                if steps > reference_cap:
                    over_cap_passing.append({"instance_id": iid, "trajectory_id": tid,
                                             "steps": int(steps)})

    return {
        "results_dir": results_dir,
        "n_instances": n_instances,
        "reference_cap": reference_cap,
        "steps_all_trajectories": _dist(all_steps),
        "steps_passing_trajectories": _dist(passing_steps),
        "total_steps_per_instance": _dist(total_steps_per_instance),
        "elapsed_seconds_per_instance": _dist(elapsed_per_instance),
        "passing_branches_over_reference_cap": over_cap_passing,
        "tokens_arm_total": {
            "prompt_tokens": arm_prompt_tokens,
            "completion_tokens": arm_completion_tokens,
            "total_tokens": arm_total_tokens,
            "n_trajectories_with_tokens": n_traj_with_tokens,
        },
        "tokens_per_trajectory": _dist(traj_total_tokens),
        "tokens_passing_trajectories": _dist(passing_total_tokens),
        "fairness_note": (
            "If passing_branches_over_reference_cap is empty, no passing branch used "
            "more than the baseline's total step budget — the step_limit asymmetry did "
            "NOT manufacture wins. tokens_arm_total is the per-arm token compute "
            "accounting (summed from each trajectory's litellm response usage); compare "
            "it across arms at matched k. KNOWN UNDERCOUNT on the treatment arm: the "
            "strategy-proposer call and intent-extraction sub-calls are not stored in "
            "the per-trajectory .traj.json transcripts, and DeBERTa-NLI forward passes "
            "are a different (0.4B) model — all excluded from these sums. The exclusion "
            "is bounded (one proposer call and O(N^2) NLI passes per instance vs k full "
            "agent trajectories) but means the treatment arm's true total is slightly "
            "HIGHER than reported; the structural claim that the matched-k vanilla arm "
            "pays at least as much (k full SEARCHes vs one shared SEARCH) rests on the "
            "trajectory sums, which dominate. Cost in $ is omitted only because the "
            "local vLLM model is not registered for litellm cost calculation."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Budget-fairness audit (R6.3).")
    p.add_argument("--results-dir", required=True, help="Arm results dir (contains <iid>/metadata.json).")
    p.add_argument("--eval", required=True, help="trajectory_eval dir (or single .json).")
    p.add_argument("--reference-cap", type=int, default=250,
                   help="Reference step budget to flag passing branches against (default: baseline 250).")
    p.add_argument("--out", default=None, help="Write the audit JSON here.")
    args = p.parse_args()

    report = audit(args.results_dir, args.eval, args.reference_cap)
    print(f"\n=== Budget audit: {args.results_dir} ({report['n_instances']} instances) ===")
    print(f"  steps (all traj):     {report['steps_all_trajectories']}")
    print(f"  steps (passing traj): {report['steps_passing_trajectories']}")
    print(f"  tokens (arm total):   {report['tokens_arm_total']}")
    print(f"  tokens (per traj):    {report['tokens_per_trajectory']}")
    over = report["passing_branches_over_reference_cap"]
    print(f"  passing branches over cap={args.reference_cap}: {len(over)} {over}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
