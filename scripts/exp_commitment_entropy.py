r"""Commitment-entropy experiment: H(S | tau_{:k}) vs prefix length k.

The causal question: after fixing the first k steps of an agent trajectory, how
much does the FINAL strategy still vary under the model's own resampling? If it
collapses to ~0 at small k, the strategy is committed early.

  H_k = semantic entropy of M final strategies obtained by:
    1. reconstruct the prefix state at step k (replay a reference trajectory's
       recorded bash actions into a fresh container -> exact filesystem state),
    2. set the agent's messages to tau_{:k},
    3. run M INDEPENDENT continuations to completion at temperature T,
    4. cluster the M final patches by STS (mpnet) cosine; H_k = -sum p_c log p_c.

Sweep k across the trajectory; plot H_k vs k. Non-circular (no reference to a
"final" anything) and interventional (divergence of counterfactual futures).
This same curve is the branching criterion: branch where H_k is still high.

One eval container alive at a time (VRAM-safe). Usage:
  python -X utf8 scripts/exp_commitment_entropy.py --instance sympy__sympy-22714 \
      --m 6 --fractions 0 0.25 0.5 0.75 0.9 --cont-temp 0.7 --out results/commitment
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ.setdefault("PYTHONUTF8", "1")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy
import logging

from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models.litellm_textbased import LitellmTextbasedModel
from minisweagent.exceptions import LimitsExceeded, FormatError, InterruptAgentFlow, Submitted
from src.agent.branching_agent import BranchingAgent
from src.evaluation.dataset import load_swebench_instances
from src.evaluation.behavioral_signature import (
    behavioral_entropy, structural_signature, cluster_counts, discrete_entropy,
)
from run_branching import load_config, build_env_config, find_eval_image, reset_litellm_clients

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("commitment")
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------- #
# Agent construction / running
# --------------------------------------------------------------------------- #

def make_agent(config, instance, temperature):
    env_config = build_env_config(config, instance["instance_id"])
    env = DockerEnvironment(**env_config)
    model_config = {k: v for k, v in config["model"].items() if k != "model_class"}
    model_config = copy.deepcopy(model_config)
    model_config.setdefault("model_kwargs", {})["temperature"] = temperature
    model = LitellmTextbasedModel(**model_config)
    agent = BranchingAgent(model=model, env=env, **config["agent"])
    agent.extra_template_vars |= {"task": instance["problem_statement"]}
    agent.messages = []
    agent.add_messages(
        agent.model.format_message(role="system",
                                   content=agent._render_template(agent.config.system_template)),
        agent.model.format_message(role="user",
                                   content=agent._render_template(agent.config.instance_template)),
    )
    return agent, env


def run_to_completion(agent, max_steps):
    """Greedy/sampled agent loop until submit / limit. Returns final patch."""
    steps = 0
    while steps < max_steps:
        try:
            msg = agent.query_only()
        except (LimitsExceeded, FormatError):
            break
        try:
            agent.execute_response(msg)
        except Submitted:
            break
        except InterruptAgentFlow:
            pass
        except Exception as e:
            logger.warning(f"step error: {e}")
            break
        steps += 1
        if agent.is_finished():
            break
    return capture_patch(agent)


def capture_patch(agent):
    sub = agent.get_submission()
    if sub and sub.strip():
        return sub
    # fallback: source-only diff from the container
    try:
        out = agent.env.execute("cd /testbed && git diff -- '*.py' ':(exclude)*/tests/*'")
        return (out.get("output", "") or "").strip()
    except Exception:
        return ""


def assistant_contents(agent):
    return [m["content"] for m in agent.messages if m.get("role") == "assistant"]


def replay_prefix(agent, ref_contents, k):
    """Replay the first k recorded assistant responses to reach prefix state k."""
    for content in ref_contents[:k]:
        try:
            agent.inject_and_execute(content)
        except Exception as e:
            logger.warning(f"replay step failed: {e}")


# --------------------------------------------------------------------------- #
# Outcome clustering -> semantic entropy
# --------------------------------------------------------------------------- #

_ST = None


def _embed(texts):
    global _ST
    if _ST is None:
        from sentence_transformers import SentenceTransformer
        _ST = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    return _ST.encode(texts, normalize_embeddings=True)


def semantic_entropy(patches, sim_thr=0.6):
    """Discrete semantic entropy of final strategies. Empty patches -> one
    'no-fix' cluster; non-empty clustered by mpnet cosine > sim_thr (union-find).
    H = -sum (n_c/M) log(n_c/M) over all clusters."""
    import numpy as np
    M = len(patches)
    if M == 0:
        return 0.0, []
    nonempty = [p for p in patches if p and p.strip()]
    n_empty = M - len(nonempty)
    clusters = []
    if nonempty:
        emb = _embed([p[:2000] for p in nonempty])
        n = len(nonempty)
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for i in range(n):
            for j in range(i + 1, n):
                if float(np.dot(emb[i], emb[j])) > sim_thr:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)
        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        clusters = [len(v) for v in groups.values()]
    if n_empty:
        clusters.append(n_empty)
    H = -sum((c / M) * math.log(c / M) for c in clusters if c > 0)
    return H, clusters


# --------------------------------------------------------------------------- #
# Behavioral (execution-grounded) clustering -> the paper's PRIMARY relation.
# Each continuation patch is evaluated; two patches share a class iff they induce
# the same per-test outcome vector. Unlike mpnet/NLI text-similarity this is
# checkable and model-free (Methods, contribution #2).
# --------------------------------------------------------------------------- #

def eval_patch_to_report(instance_id, model_name, patch, tid, run_id, timeout, temp_dir):
    """Evaluate one patch via the SWE-bench harness; return its report record
    (the inner ``{...}`` dict). Empty patches skip Docker and map to NO_PATCH;
    a missing report (apply-failure / timeout) maps to APPLY_FAILED, so every
    continuation lands in a behavioral class without a fabricated test verdict."""
    if not patch or not patch.strip():
        return {"patch_is_None": True, "patch_exists": False}
    # Heavy import deferred so the module stays unit-testable without swebench.
    from src.evaluation.run_eval import run_evaluation
    pred = {"instance_id": instance_id, "model_name_or_path": model_name,
            "model_patch": patch}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False,
                                     dir=temp_dir) as f:
        f.write(json.dumps(pred) + "\n")
        temp_path = f.name
    try:
        run_evaluation(predictions_path=temp_path, instance_ids=[instance_id],
                       run_id=run_id, timeout=timeout)
        report_path = os.path.join("logs", "run_evaluation", run_id,
                                   model_name.replace("/", "__"), instance_id,
                                   "report.json")
        if os.path.exists(report_path):
            with open(report_path) as rf:
                report = json.load(rf)
            return report.get(instance_id, report)
        # No report => harness swallowed a patch-attributable error (apply/timeout).
        return {"patch_exists": True, "patch_successfully_applied": False}
    finally:
        os.unlink(temp_path)


def behavioral_entropy_of_patches(patches, instance_id, model_name, run_id,
                                  timeout, temp_dir, miller_madow=True):
    """Eval each patch -> behavioral signature -> discrete entropy (Miller-Madow).
    Distinct eval run_id per patch so the harness's (run_id,model,instance) cache
    cannot collapse two different patches onto one stale report."""
    reports = []
    for i, p in enumerate(patches):
        reports.append(eval_patch_to_report(
            instance_id, model_name, p, f"cont{i}",
            f"{run_id}_c{i}", timeout, temp_dir))
    return behavioral_entropy(reports, instance_id, miller_madow=miller_madow)


def structural_entropy_of_patches(patches, miller_madow=True):
    """Secondary, zero-GPU lens: cluster by structural signature (files+hunks)."""
    sigs = [structural_signature(p) for p in patches]
    out = discrete_entropy(cluster_counts(sigs), miller_madow=miller_madow)
    out["clusters"] = cluster_counts(sigs)
    return out


# --------------------------------------------------------------------------- #
# Main sweep
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--dataset", default="verified")
    ap.add_argument("--config", default=None)
    ap.add_argument("--m", type=int, default=6, help="continuations per prefix")
    ap.add_argument("--fractions", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 0.9])
    ap.add_argument("--ref-temp", type=float, default=0.0, help="reference trajectory temp")
    ap.add_argument("--cont-temp", type=float, default=0.7, help="continuation temp")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--cluster", choices=["behavioral", "structural", "sts"],
                    default="behavioral",
                    help="meaning relation for outcome entropy. behavioral "
                         "(paper primary: per-test outcome vector, needs Docker "
                         "eval per continuation); structural (files+hunks, no "
                         "GPU); sts (mpnet cosine, text-similarity ablation).")
    ap.add_argument("--no-miller-madow", dest="miller_madow", action="store_false",
                    help="disable the (K-1)/2M plug-in bias correction (on by default)")
    ap.add_argument("--model-name", default=None,
                    help="model_name_or_path slug for eval reports/run_id "
                         "(default: config model.model_name)")
    ap.add_argument("--eval-timeout", type=int, default=1800)
    ap.set_defaults(miller_madow=True)
    args = ap.parse_args()

    reset_litellm_clients()
    config = load_config(args.config)
    inst = {i["instance_id"]: i for i in
            load_swebench_instances(dataset_name=args.dataset, instance_ids=[args.instance])}[args.instance]
    os.makedirs(os.path.join(PROJECT_ROOT, args.out), exist_ok=True)

    t0 = time.time()
    # 1) Reference trajectory (greedy) -> defines the prefixes.
    logger.info(f"[{args.instance}] reference run (T={args.ref_temp})...")
    ref_agent, ref_env = make_agent(config, inst, args.ref_temp)
    run_to_completion(ref_agent, args.max_steps)
    ref_contents = assistant_contents(ref_agent)
    L = len(ref_contents)
    try:
        ref_env.cleanup()
    except Exception:
        pass
    logger.info(f"[{args.instance}] reference length = {L} steps")
    if L < 4:
        logger.error("reference too short; aborting")
        return

    model_name = args.model_name or config["model"]["model_name"]
    temp_dir = os.path.join(PROJECT_ROOT, args.out, "_eval_tmp")
    os.makedirs(temp_dir, exist_ok=True)

    # 2) Sweep prefix lengths; M continuations each.
    curve = []
    for frac in args.fractions:
        k = int(round(frac * L))
        patches = []
        for m in range(args.m):
            agent, env = make_agent(config, inst, args.cont_temp)
            replay_prefix(agent, ref_contents, k)
            patch = run_to_completion(agent, args.max_steps)
            patches.append(patch)
            logger.info(f"[{args.instance}] k={k}({frac:.2f}) cont {m+1}/{args.m}: "
                        f"{len(patch)} chars")
            try:
                env.cleanup()
            except Exception:
                pass

        # Outcome entropy under the selected meaning relation.
        if args.cluster == "behavioral":
            ent = behavioral_entropy_of_patches(
                patches, args.instance, model_name, run_id=f"commit_{args.instance}_k{k}",
                timeout=args.eval_timeout, temp_dir=temp_dir,
                miller_madow=args.miller_madow)
        elif args.cluster == "structural":
            ent = structural_entropy_of_patches(patches, miller_madow=args.miller_madow)
        else:  # sts ablation
            H, clusters = semantic_entropy(patches)
            ent = discrete_entropy(clusters, miller_madow=args.miller_madow)
            ent["clusters"] = sorted(clusters, reverse=True)

        ent.update({"fraction": frac, "k": k, "cluster_method": args.cluster,
                    "n_nonempty": sum(1 for p in patches if p.strip())})
        curve.append(ent)
        logger.info(f"[{args.instance}] k={k}({frac:.2f})  H={ent['H']:.3f} "
                    f"H_mm={ent.get('H_mm', ent['H']):.3f}  clusters={ent.get('clusters')}")

    result = {"instance": args.instance, "ref_len": L, "m": args.m,
              "ref_temp": args.ref_temp, "cont_temp": args.cont_temp,
              "cluster_method": args.cluster, "miller_madow": args.miller_madow,
              "model_name": model_name,
              "elapsed_s": round(time.time() - t0, 1), "curve": curve}
    out_path = os.path.join(PROJECT_ROOT, args.out, f"Hk_{args.instance}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n=== H(S | prefix) for {args.instance} (ref_len={L}) ===")
    for c in curve:
        bar = "#" * int(round(c["H_norm"] * 30))
        print(f"  k={c['k']:>3} ({c['fraction']:.2f})  H={c['H']:.3f}  "
              f"H_norm={c['H_norm']:.2f} {bar}  clusters={c['clusters']}")
    print(f"\nWrote {out_path}  ({result['elapsed_s']}s)")


if __name__ == "__main__":
    main()
