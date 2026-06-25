"""Experiment: WHEN does an agent's strategy lock in?

Hypothesis (to test, not assume): an SWE-agent commits to its overall strategy
early in the trajectory, then merely executes it. If true, the strategy-defining
content is concentrated in an early window — which justifies capturing/diversifying
strategy there rather than over the whole (un-embeddable) trajectory.

Method (zero new GPU; reuses completed trajectories + validated components):
  - At each agent step t, distill the strategy-SO-FAR via the focused-descriptor
    intent prompt over the history up to t (LLM, local vLLM).
  - Embed each step's strategy descriptor with an STS encoder (all-mpnet-base-v2).
  - Measure convergence to the FINAL strategy: sim_t = cos(e_t, e_final).
  - Lock-in step = first t where sim_t >= LOCK and stays >= LOCK to the end.
  - Report lock-in as a FRACTION of trajectory length, aggregated across
    trajectories (early-and-consistent fractions => the hypothesis holds).

Usage: python -X utf8 scripts/exp_strategy_lockin.py --arm-dir results/strategy_t0.7 [--per-instance 1]
"""
import argparse
import glob
import json
import os
import statistics
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sentence_transformers import SentenceTransformer, util
from src.diversity.intent import IntentExtractor

LOCK = 0.80  # cos>=0.80 to the final strategy == "on the final approach"


def thoughts_of(messages):
    """Indices of assistant messages (one per agent step)."""
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def lockin_fraction(sims):
    """First index (0-based) from which sims stay >= LOCK, as a fraction."""
    n = len(sims)
    lock = n - 1
    for t in range(n):
        if all(s >= LOCK for s in sims[t:]):
            lock = t
            break
    return lock / max(1, n - 1), lock, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", default="results/strategy_t0.7")
    ap.add_argument("--per-instance", type=int, default=1,
                    help="how many trajectories per instance to sample")
    ap.add_argument("--max-steps", type=int, default=40,
                    help="cap steps distilled per trajectory (cost guard)")
    args = ap.parse_args()

    ext = IntentExtractor(
        model_name="openai/qwen3-coder",
        model_kwargs={"api_base": "http://localhost:8001/v1", "api_key": "dummy"},
        method="llm",
    )
    st = SentenceTransformer("all-mpnet-base-v2", device="cpu")

    traj_files = []
    for inst in sorted(glob.glob(os.path.join(args.arm_dir, "sympy__*"))):
        fs = sorted(glob.glob(os.path.join(inst, "*.traj.json")))[: args.per_instance]
        traj_files += fs

    print(f"strategy lock-in over {len(traj_files)} trajectories (LOCK cos>={LOCK})\n")
    fractions = []
    for f in traj_files:
        try:
            msgs = json.load(open(f, encoding="utf-8")).get("messages", [])
        except Exception:
            continue
        idxs = thoughts_of(msgs)
        if len(idxs) < 4:
            continue
        if len(idxs) > args.max_steps:  # evenly subsample long trajectories
            keep = [idxs[round(k)] for k in
                    [i * (len(idxs) - 1) / (args.max_steps - 1) for i in range(args.max_steps)]]
            idxs = sorted(set(keep))
        # distill strategy-so-far at each kept step
        descs = [ext.extract_trajectory_intent(msgs[: i + 1]) for i in idxs]
        embs = st.encode(descs, normalize_embeddings=True)
        final = embs[-1]
        sims = [float(util.cos_sim(e, final)) for e in embs]
        frac, lock, n = lockin_fraction(sims)
        fractions.append(frac)
        name = f.replace("\\", "/").split("/")[-2] + "/" + f.replace("\\", "/").split("/")[-1].replace("trajectory_", "").replace(".traj.json", "")
        print(f"  {name:<34} lock@step {lock+1:>2}/{n:<2} (frac {frac:.2f})  "
              f"sims[0]={sims[0]:.2f} mid={sims[n//2]:.2f}")

    if fractions:
        print(f"\n=== {len(fractions)} trajectories ===")
        print(f"  lock-in fraction: median={statistics.median(fractions):.2f}  "
              f"mean={statistics.mean(fractions):.2f}  "
              f"min={min(fractions):.2f}  max={max(fractions):.2f}")
        early = sum(1 for x in fractions if x <= 0.5)
        print(f"  locked in by the FIRST HALF of the trajectory: {early}/{len(fractions)} "
              f"({100*early/len(fractions):.0f}%)")
        print(f"  => {'SUPPORTS early commitment' if statistics.median(fractions) <= 0.5 else 'does NOT support early commitment'}")


if __name__ == "__main__":
    main()
