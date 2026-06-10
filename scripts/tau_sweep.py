r"""Post-hoc tau sweep: evaluate the entropy gate at EVERY tau from one run (R3.3, R5.5).

Key design fact this script exploits: the headline run uses tau=0 ("always
branch"), which produces the SUPERSET of trajectories any tau > 0 would have
produced. When the gate declines to branch (entropy <= tau), the orchestrator
keeps exactly ONE trajectory — the dominant (largest, lowest-index-on-ties)
cluster's representative — and that trajectory ALREADY EXISTS in the tau=0 run:
trajectory i executes cluster i's representative strategy ("t0" for i=0, else
"t0_strategy_<i>"), and post-branch execution is greedy (temp 0), so the gated
trajectory is identical to what a real tau>0 run would have produced (modulo
vLLM nondeterminism). The SDLG arm's no-branch action keeps the greedy parent
"t0". The ENTIRE tau sweep is therefore computable post-hoc by trajectory
subsetting — no additional GPU runs.

Entropy quantization (stated plainly): with N candidates, discrete semantic
entropy takes only the values attainable by partitions of N — for N=5 exactly
{0, 0.500, 0.673, 0.950, 1.055, 1.332, 1.609} nats (7 values, one per partition
shape). The tau gate at small N is therefore a PARTITION-SHAPE RULE, not a
continuous dial: only taus straddling adjacent achievable values differ, and the
sweep grid below is exactly the achievable set. (tau=0 = "branch iff >=2
clusters".) Note also the plug-in entropy estimator is biased low at small N
(Miller–Madow ~ (K-1)/2N nats); the gate is DEFINED on the plug-in value, which
is consistent across instances only because N is held fixed (n_strategies =
sdlg_n_alternatives = 5) across arms and instances.

Per instance it reads:
  - <results_dir>/<iid>/phased_decisions.log  (STRATEGY PROPOSAL block: entropy +
    per-strategy cluster assignments -> cluster sizes in cluster order)
  - <iid>/branching_log.json                  (SDLG arm fallback: entropy only;
    gated no-branch subset = {"t0"})
  - trajectory_eval_<iid>.json                (per-trajectory resolved, by tid)

Usage:
  python scripts/tau_sweep.py --results-dir results/strategy_t0.7 \
      --eval results/strategy_t0.7 --out results/tau_sweep_strategy_t0.7.json
"""

import argparse
import glob
import json
import math
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compute_metrics import load_eval_by_tid


_HEADER_RE = re.compile(
    r"Proposed:\s*(\d+)\s*\|\s*Clusters:\s*(\d+)\s*\|\s*Unique:\s*(\d+)\s*\|\s*"
    r"Entropy:\s*([0-9]*\.?[0-9]+)")
_MEMBER_RE = re.compile(r"^\s*\[(\d+)\]\s+cluster=(\d+):", re.MULTILINE)


def partition_entropies(n: int) -> list[float]:
    """All achievable discrete-entropy values for N candidates (sorted, unique).

    One value per integer partition of n: H = -sum (m_i/n) log(m_i/n). This IS
    the admissible tau grid — taus between adjacent values are equivalent.
    """
    parts: list[list[int]] = []

    def rec(remaining: int, max_part: int, acc: list[int]) -> None:
        if remaining == 0:
            parts.append(list(acc))
            return
        for p in range(min(remaining, max_part), 0, -1):
            acc.append(p)
            rec(remaining - p, p, acc)
            acc.pop()

    rec(n, n, [])
    vals = set()
    for part in parts:
        h = -sum((m / n) * math.log(m / n) for m in part)
        vals.add(abs(round(h, 4)))  # abs() normalizes -0.0 from the [n] partition
    return sorted(vals)


def parse_instance(results_dir: str, iid: str) -> dict | None:
    """-> {"entropy": float, "cluster_sizes": [int] in cluster order, "arm": str}.

    Strategy arm: parsed from the STRATEGY PROPOSAL block. SDLG arm fallback:
    entropy from branching_log.json, cluster sizes unavailable (no-branch keeps
    "t0"). Returns None if neither artifact yields an entropy.
    """
    log = os.path.join(results_dir, iid, "phased_decisions.log")
    if os.path.isfile(log):
        try:
            with open(log, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            text = ""
        m = _HEADER_RE.search(text)
        if m:
            n_clusters = int(m.group(2))
            entropy = float(m.group(4))
            sizes = [0] * n_clusters
            block = text[m.end():]
            # Stop at the fork list so a second run's block can't bleed in.
            stop = block.find("Unique strategies to fork")
            if stop != -1:
                block = block[:stop]
            for sm in _MEMBER_RE.finditer(block):
                ci = int(sm.group(2))
                if 0 <= ci < n_clusters:
                    sizes[ci] += 1
            return {"entropy": entropy, "cluster_sizes": sizes,
                    "arm": "strategy_proposal"}
    blog = os.path.join(results_dir, iid, "branching_log.json")
    if os.path.isfile(blog):
        try:
            with open(blog, "r", encoding="utf-8") as f:
                events = json.load(f)
            ents = [e["entropy"] for e in events if "entropy" in e]
            if ents:
                return {"entropy": float(ents[0]), "cluster_sizes": None,
                        "arm": "sdlg"}
        except Exception:
            pass
    return None


def dominant_trajectory_id(cluster_sizes: list[int] | None) -> str:
    """Trajectory kept when the gate declines to branch.

    Strategy arm: the dominant cluster's representative = trajectory at the
    dominant cluster's index ("t0" for index 0). max() ties break to the lowest
    index, matching the orchestrator's max(clusters, key=len). SDLG arm (sizes
    unavailable): the greedy parent "t0".
    """
    if not cluster_sizes:
        return "t0"
    d = max(range(len(cluster_sizes)), key=lambda i: cluster_sizes[i])
    return "t0" if d == 0 else f"t0_strategy_{d}"


def sweep(results_dir: str, eval_path: str, taus: list[float] | None) -> dict:
    evals = load_eval_by_tid(eval_path)
    iids = sorted(evals)
    parsed: dict[str, dict] = {}
    for iid in iids:
        info = parse_instance(results_dir, iid)
        if info is not None:
            parsed[iid] = info

    usable = sorted(parsed)
    skipped = sorted(set(iids) - set(usable))
    n_candidates = None
    for iid in usable:
        sizes = parsed[iid]["cluster_sizes"]
        if sizes:
            n_candidates = sum(sizes)
            break

    if taus is None:
        taus = partition_entropies(n_candidates) if n_candidates else \
            [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

    rows = []
    for tau in taus:
        branched, used, passed = [], [], []
        missing_gated: list[str] = []
        for iid in usable:
            info = parsed[iid]
            outcomes = evals.get(iid, {})
            if info["entropy"] > tau:
                branched.append(1)
                used.append(len(outcomes))
                passed.append(1.0 if any(outcomes.values()) else 0.0)
            else:
                branched.append(0)
                used.append(1)
                tid = dominant_trajectory_id(info["cluster_sizes"])
                if tid not in outcomes:
                    missing_gated.append(f"{iid}:{tid}")
                passed.append(1.0 if outcomes.get(tid, False) else 0.0)
        n = len(usable)
        rows.append({
            "tau": tau,
            "branch_rate": round(sum(branched) / n, 4) if n else None,
            "mean_trajectories_used": round(sum(used) / n, 4) if n else None,
            "gated_pass_rate": round(sum(passed) / n, 4) if n else None,
            "missing_gated_trajectories": missing_gated,
        })

    return {
        "results_dir": results_dir,
        "n_instances": len(usable),
        "skipped_instances": skipped,
        "n_candidates": n_candidates,
        "achievable_entropies": (partition_entropies(n_candidates)
                                 if n_candidates else None),
        "per_instance": {iid: parsed[iid] for iid in usable},
        "sweep": rows,
        "validity_note": (
            "Computed post-hoc from the tau=0 superset run: the no-branch action "
            "keeps the dominant-cluster representative trajectory, which exists in "
            "the superset run and executed greedily (temp 0), so it is identical to "
            "a real tau>0 run modulo vLLM nondeterminism. gated_pass_rate at the "
            "smallest tau equals the full-branch (oracle) rate; at large tau it "
            "approaches the single-dominant-trajectory rate. The tau grid is the "
            "ACHIEVABLE entropy set for N candidates (entropy is partition-"
            "quantized at small N); intermediate taus are equivalent."),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Post-hoc tau sweep (R3.3/R5.5) from artifacts.")
    p.add_argument("--results-dir", required=True,
                   help="Arm results dir (contains <iid>/phased_decisions.log).")
    p.add_argument("--eval", required=True, help="trajectory_eval dir (or single .json).")
    p.add_argument("--taus", type=float, nargs="*", default=None,
                   help="Explicit tau grid (default: all achievable entropies for N).")
    p.add_argument("--out", default=None, help="Write the sweep JSON here.")
    args = p.parse_args()

    report = sweep(args.results_dir, args.eval, args.taus)
    print(f"\n=== tau sweep: {args.results_dir} ({report['n_instances']} instances, "
          f"N={report['n_candidates']}) ===")
    print(f"  achievable entropies (quantization at N): {report['achievable_entropies']}")
    print(f"  {'tau':>8} {'branch_rate':>12} {'mean_traj':>10} {'gated_pass':>11}")
    for r in report["sweep"]:
        print(f"  {r['tau']:>8} {r['branch_rate']:>12} "
              f"{r['mean_trajectories_used']:>10} {r['gated_pass_rate']:>11}")
    if report["skipped_instances"]:
        print(f"  skipped (no entropy artifact): {report['skipped_instances']}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
