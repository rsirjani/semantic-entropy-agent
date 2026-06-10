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

    Values are FULL precision (not display-rounded): the sweep's gate compares
    `entropy > tau` exactly like the orchestrator does, so rounding the grid
    would shift boundary decisions (e.g. the (2,2,1) partition of 5 has
    H = 1.054920…, which a 3-or-4-decimal rounding can flip across the gate).
    Dedup uses a 1e-9 tolerance only to merge float noise.
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
    vals: list[float] = []
    for part in parts:
        h = abs(-sum((m / n) * math.log(m / n) for m in part))  # abs: -0.0 -> 0.0
        if not any(abs(h - v) <= 1e-9 for v in vals):
            vals.append(h)
    return sorted(vals)


def exact_partition_entropy(cluster_sizes: list[int]) -> float | None:
    """Full-precision plug-in entropy from cluster sizes, or None if unusable.

    The discrete semantic entropy is a deterministic function of the cluster
    partition, so when the log yields the per-strategy cluster assignments we
    can recompute it EXACTLY instead of trusting the log's rounded `Entropy:`
    value. This matters at gate boundaries: the log prints few decimals, and
    e.g. the (2,2,1) partition of 5 (H = 1.054920…) rounds at 3 decimals to
    1.055 > 1.0549, which would make the post-hoc sweep branch at the exact
    achievable-grid tau where a real run gates.
    """
    if not cluster_sizes:
        return None
    n = sum(cluster_sizes)
    if n <= 0:
        return None
    return abs(-sum((m / n) * math.log(m / n) for m in cluster_sizes if m > 0))


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
        # The decisions log is APPEND-mode: a re-run adds a second STRATEGY
        # PROPOSAL block while predictions keep-last and metadata.json is
        # overwritten — so the LAST block is the one whose partition matches
        # the trajectories being subsetted; the first would be stale.
        m = None
        for m in _HEADER_RE.finditer(text):
            pass
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
            # Prefer the EXACT recomputed entropy over the log's rounded value
            # when they agree to within log-rounding tolerance (6e-4 covers a
            # 3-decimal log). Disagreement beyond that means the run did not use
            # the discrete partition entropy (e.g. kernel/von-Neumann, whose
            # value is NOT a function of cluster sizes) — keep the logged value
            # and say so, never silently overwrite it.
            exact = exact_partition_entropy(sizes)
            if exact is not None and abs(exact - entropy) <= 6e-4:
                return {"entropy": exact, "entropy_source": "recomputed_from_cluster_sizes",
                        "cluster_sizes": sizes, "arm": "strategy_proposal"}
            return {"entropy": entropy,
                    "entropy_source": ("parsed_log_disagrees_with_partition"
                                       if exact is not None else "parsed_log"),
                    "cluster_sizes": sizes, "arm": "strategy_proposal"}
    blog = os.path.join(results_dir, iid, "branching_log.json")
    if os.path.isfile(blog):
        try:
            with open(blog, "r", encoding="utf-8") as f:
                events = json.load(f)
            ents = [e["entropy"] for e in events if "entropy" in e]
            if ents:
                return {"entropy": float(ents[0]), "entropy_source": "branching_log",
                        "cluster_sizes": None, "arm": "sdlg"}
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

    # Realized N per instance. The config holds N fixed (n_strategies = 5), but
    # the proposer can under-deliver (parse failure, short rejection pass), and
    # entropy values from different N are NOT on the same quantization grid —
    # so the realized N must be REPORTED, not assumed. The grid uses the MODAL
    # N; instances at a different N are flagged, never silently pooled.
    n_by_instance = {iid: sum(parsed[iid]["cluster_sizes"])
                     for iid in usable
                     if parsed[iid]["cluster_sizes"]
                     and sum(parsed[iid]["cluster_sizes"]) > 0}
    n_candidates = None
    if n_by_instance:
        counts: dict[int, int] = {}
        for v in n_by_instance.values():
            counts[v] = counts.get(v, 0) + 1
        # Modal N; ties break to the LARGER N (the configured n_strategies is
        # an upper bound — under-delivery is the anomaly, not the target).
        n_candidates = max(sorted(counts), key=lambda v: (counts[v], v))
    non_modal = sorted(i for i, v in n_by_instance.items() if v != n_candidates)

    if taus is None:
        taus = partition_entropies(n_candidates) if n_candidates else \
            [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

    # Gate semantics replicate the orchestrator: branch iff entropy > tau, at
    # FULL precision. The 1e-9 epsilon only absorbs float noise so that a tau
    # set exactly at an achievable entropy level gates that level (a real run
    # replicates row tau=h_i by setting tau anywhere in [h_i, next level)).
    rows = []
    for tau in taus:
        branched, used, passed = [], [], []
        missing_gated: list[str] = []
        for iid in usable:
            info = parsed[iid]
            outcomes = evals.get(iid, {})
            if info["entropy"] > tau + 1e-9:
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
        "n_candidates_by_instance": n_by_instance,
        "non_modal_n_instances": non_modal,
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
            "ACHIEVABLE entropy set for the MODAL N (entropy is partition-"
            "quantized at small N); intermediate taus are equivalent. Entropies "
            "are recomputed at full precision from the logged cluster partition "
            "when consistent with the logged value (entropy_source per instance); "
            "instances whose realized N deviates from the modal N are listed in "
            "non_modal_n_instances — their entropies sit on a DIFFERENT "
            "quantization grid and must not be pooled silently."),
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
    ach = report["achievable_entropies"]
    print(f"  achievable entropies (quantization at N): "
          f"{[round(v, 4) for v in ach] if ach else ach}")
    print(f"  {'tau':>8} {'branch_rate':>12} {'mean_traj':>10} {'gated_pass':>11}")
    for r in report["sweep"]:
        print(f"  {round(r['tau'], 4):>8} {r['branch_rate']:>12} "
              f"{r['mean_trajectories_used']:>10} {r['gated_pass_rate']:>11}")
    if report["skipped_instances"]:
        print(f"  skipped (no entropy artifact): {report['skipped_instances']}")
    if report["non_modal_n_instances"]:
        print(f"  WARNING — realized N deviates from modal N={report['n_candidates']} "
              f"on: {report['non_modal_n_instances']} (different quantization grid)")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
