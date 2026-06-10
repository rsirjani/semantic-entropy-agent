r"""Autonomous experiment campaign: confirmatory pilot -> analyst-driven refinement.

Executes the experiment plan the scrutiny loop converged on (scrutiny_03.md "minimal
sufficient experiment set"), then lets a headless Fable analyst choose follow-up
phases FROM ACTUAL DATA, within hard guardrails. Deterministic execution, model-
driven phase selection — the analyst only ever picks from a fixed menu; every
command that runs is constructed by this script.

Phase A (confirmatory, always first — the pre-registered primary endpoint):
  1. strategy arm  @ T=0.7, greedy clustering, tau=0 superset run
  2. per-trajectory SWE-bench eval of the treatment
  3. matched-k vanilla control @ T=0.7 (k read from the treatment's metadata)
  4. per-trajectory eval of the control
  5. compute_metrics (matched-k* + exact sign-flip primary endpoint),
     budget_audit (both arms), tau_sweep (treatment, post-hoc R3.3/R5.5)

Then up to --max-phases analyst-chosen phases from MENU (exploratory cells:
SDLG @0.7, temperature 0.2/1.0 cells, clustering variants, a repeat seed), each
with the same treatment->eval->control->eval->metrics pipeline. The analyst writes
campaign_decisions/decision_<N>.json; invalid or "stop" ends the campaign.

Guardrails: wall-clock cap, disk floor, one-run-per-spec, per-step retry-once,
a STOP file (campaign_decisions/STOP) aborts between steps. State persists in
results/campaign/campaign_state.json; --resume continues an interrupted campaign
(runs use --skip-existing; evals skip instances whose trajectory_eval exists).

Usage:
  python scripts/run_campaign.py                # dry-run: print the plan
  python scripts/run_campaign.py --go           # run Phase A + analyst phases
  python scripts/run_campaign.py --go --resume  # continue after interruption
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT_ROOT, "results")
CAMPAIGN_DIR = os.path.join(RESULTS, "campaign")
DECISIONS_DIR = os.path.join(PROJECT_ROOT, "campaign_decisions")
STATE_PATH = os.path.join(CAMPAIGN_DIR, "campaign_state.json")
STOP_FILE = os.path.join(DECISIONS_DIR, "STOP")

# Host port 8001: port 8000 on this machine is permanently shadowed by WSL's
# localhost relay forwarding to the pdf-reader backend (the PDF MCP server),
# so Docker cannot publish there. Container-internal vLLM stays on 8000.
VLLM_URL = "http://localhost:8001"
NLI_URL = "http://localhost:8100"
PRIMARY_TEMP = 0.7  # pre-registered primary endpoint (scrutiny_03.md, R6.5)

# --------------------------------------------------------------------------- #
# Spec menu — every runnable experiment cell. Phase A is fixed; the analyst
# chooses subsequent cells from MENU (minus what already ran).
# --------------------------------------------------------------------------- #

PHASE_A_KEY = "strategy_t0.7"

MENU: dict[str, dict] = {
    "strategy_t0.7":  {"arm": "strategy_proposal", "temperature": 0.7,
                       "clustering": "greedy",
                       "why": "CONFIRMATORY primary endpoint (pre-registered)"},
    "sdlg_t0.7":      {"arm": "sdlg", "temperature": 0.7, "clustering": "greedy",
                       "why": "exploratory mechanism contrast (R5.4 needs the "
                              "confidence-independent generator)"},
    "strategy_t0.2":  {"arm": "strategy_proposal", "temperature": 0.2,
                       "clustering": "greedy", "why": "exploratory low-T cell"},
    "strategy_t1.0":  {"arm": "strategy_proposal", "temperature": 1.0,
                       "clustering": "greedy", "why": "exploratory high-T cell"},
    "kernel_t0.7":    {"arm": "strategy_proposal", "temperature": 0.7,
                       "clustering": "kernel",
                       "why": "clustering ablation (KLE; tau recalibration applies)"},
    "connected_t0.7": {"arm": "strategy_proposal", "temperature": 0.7,
                       "clustering": "connected", "why": "clustering ablation"},
    "strategy_t0.7_seed2": {"arm": "strategy_proposal", "temperature": 0.7,
                            "clustering": "greedy", "repeat": True,
                            "why": "repeat draw of the primary cell — sampling "
                                   "variance of the headline numbers"},
}


def spec_dirs(key: str) -> dict[str, str]:
    """Result-dir layout for a spec: treatment dir, control base, control dir."""
    spec = MENU[key]
    t = spec["temperature"]
    treat = os.path.join(RESULTS, key)
    control_base = os.path.join(RESULTS, f"resample_{key}")
    return {
        "treatment": treat,
        "control_base": control_base,
        # run_resample_baseline appends _t<T> to its base dir:
        "control": f"{control_base}_t{t}",
        "metrics": os.path.join(RESULTS, f"metrics_{key}_vs_vanilla.json"),
    }


def build_steps(key: str) -> list[dict]:
    """The deterministic command sequence for one spec (treatment + control)."""
    spec = MENU[key]
    d = spec_dirs(key)
    t = spec["temperature"]
    py = sys.executable
    steps = [
        {"name": f"{key}/treatment_run",
         "cmd": [py, "scripts/run_branching.py", "--config", "configs/branching.yaml",
                 "--results-dir", d["treatment"],
                 "--clustering-strategy", spec["clustering"],
                 "--temperature", str(t),
                 "--diversity-method", spec["arm"],
                 "--skip-existing"],
         "timeout": 10 * 3600},
        {"name": f"{key}/treatment_eval", "eval_dir": d["treatment"],
         "timeout": 8 * 3600},
        {"name": f"{key}/control_run",
         "cmd": [py, "scripts/run_resample_baseline.py",
                 "--treatment-dir", d["treatment"],
                 "--results-dir", d["control_base"],
                 "--temperatures", str(t), "--skip-existing"],
         "timeout": 16 * 3600},
        {"name": f"{key}/control_eval", "eval_dir": d["control"],
         "timeout": 8 * 3600},
        {"name": f"{key}/metrics",
         "cmd": [py, "scripts/compute_metrics.py",
                 "--predictions", os.path.join(d["treatment"], "predictions_all_trajectories.jsonl"),
                 "--eval", d["treatment"], "--results-dir", d["treatment"],
                 "--compare-predictions", os.path.join(d["control"], "predictions_all_trajectories.jsonl"),
                 "--compare-eval", d["control"],
                 "--out", d["metrics"]],
         "timeout": 1800},
        {"name": f"{key}/budget_audit_treatment",
         "cmd": [py, "scripts/budget_audit.py", "--results-dir", d["treatment"],
                 "--eval", d["treatment"], "--reference-cap", "250",
                 "--out", os.path.join(RESULTS, f"budget_audit_{key}.json")],
         "timeout": 1800},
        {"name": f"{key}/tau_sweep_treatment",
         "cmd": [py, "scripts/tau_sweep.py", "--results-dir", d["treatment"],
                 "--eval", d["treatment"],
                 "--out", os.path.join(RESULTS, f"tau_sweep_{key}.json")],
         "timeout": 1800},
    ]
    return steps


# --------------------------------------------------------------------------- #
# Guardrails & state
# --------------------------------------------------------------------------- #

def disk_free_gb(path: str = RESULTS) -> float:
    usage = shutil.disk_usage(os.path.dirname(path) or path)
    return usage.free / 1e9


def guardrails_ok(state: dict, args) -> tuple[bool, str]:
    if os.path.exists(STOP_FILE):
        return False, f"STOP file present ({STOP_FILE})"
    elapsed_h = (time.time() - state["started_ts"]) / 3600
    if elapsed_h > args.max_hours:
        return False, f"wall-clock cap reached ({elapsed_h:.1f}h > {args.max_hours}h)"
    free = disk_free_gb()
    if free < args.min_disk_gb:
        return False, f"disk floor hit ({free:.0f}GB < {args.min_disk_gb}GB)"
    return True, ""


def load_state(resume: bool) -> dict:
    if resume and os.path.isfile(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        state.setdefault("completed_specs", [])
        state.setdefault("phase_log", [])
        return state
    return {"started_ts": time.time(),
            "started": datetime.now().isoformat(timespec="seconds"),
            "completed_specs": [], "phase_log": []}


def save_state(state: dict) -> None:
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def remaining_menu(state: dict) -> dict[str, dict]:
    return {k: v for k, v in MENU.items() if k not in state["completed_specs"]}


# --------------------------------------------------------------------------- #
# Analyst decision (headless Fable) — picks the next spec from the menu
# --------------------------------------------------------------------------- #

def validate_decision(decision: dict, state: dict) -> tuple[str | None, str]:
    """Return (choice, reason). choice None => stop. Raises ValueError if invalid."""
    if not isinstance(decision, dict):
        raise ValueError("decision is not a JSON object")
    choice = decision.get("choice")
    rationale = str(decision.get("rationale", ""))
    if choice == "stop":
        return None, rationale or "analyst chose stop"
    if choice not in MENU:
        raise ValueError(f"unknown menu choice: {choice!r}")
    if choice in state["completed_specs"]:
        raise ValueError(f"spec already ran: {choice}")
    if not rationale.strip():
        raise ValueError("decision missing a rationale")
    return choice, rationale


def analyst_prompt(state: dict, decision_path: str) -> str:
    menu_lines = "\n".join(
        f"  - {k}: arm={v['arm']}, T={v['temperature']}, clustering={v['clustering']}"
        f"  ({v['why']})"
        for k, v in remaining_menu(state).items()
    )
    completed = "\n".join(
        f"  - {p['spec']}: metrics={p.get('metrics_path', '?')}"
        for p in state["phase_log"] if p.get("status") == "completed"
    ) or "  (none)"
    return f"""You are the campaign analyst for a research experiment campaign
(semantic-entropy-gated branching vs matched-k vanilla resampling, SWE-bench SymPy).
Work in this repository (cwd is the project root). Read, in order:
 1. review_loop/scrutiny_03.md (the design: pre-registered primary endpoint,
    confirmatory vs exploratory cells, minimal sufficient experiment set);
 2. the metrics JSONs of every completed phase listed below (read the comparison
    block: matched-k* gain, exact sign-flip p, rarefied distinct counts,
    selected-pass@1, k-mismatch report);
 3. the budget_audit_*.json and tau_sweep_*.json companions in results/.

Completed phases:
{completed}

Decide the SINGLE most informative next experiment cell from this menu (or stop):
{menu_lines}

Decision principles: maximize information about the headline claim per GPU-hour.
Prefer the mechanism contrast (sdlg_t0.7) if the primary cell shows ANY signal
(positive or negative — both make the contrast informative); prefer a repeat seed
if the primary numbers look noise-dominated; prefer temperature cells to test
robustness only after the mechanism story is anchored; choose stop when another
cell would not change the paper's conclusions. A null result is a valid outcome —
do NOT chase a positive. You may NOT invent new specs, edit code, or launch runs
yourself.

Write EXACTLY one file, `{decision_path}`, valid JSON:
{{"choice": "<menu key or stop>", "rationale": "<3-8 sentences grounded in the
numbers you read>", "expectations": "<what result would mean what>"}}
The LAST thing you do must be writing that file. Make no other changes."""


def run_analyst(state: dict, n: int, args) -> tuple[str | None, str]:
    """Invoke headless Fable to write decision_<n>.json; validate; return choice."""
    os.makedirs(DECISIONS_DIR, exist_ok=True)
    decision_rel = f"campaign_decisions/decision_{n:02d}.json"
    decision_abs = os.path.join(PROJECT_ROOT, decision_rel)
    exe = shutil.which("claude")
    if not exe:
        return None, "claude CLI not found — stopping (campaign keeps Phase A results)"
    cmd = [exe, "-p", "--output-format", "json", "--model", args.analyst_model,
           "--max-turns", "40", "--dangerously-skip-permissions"]
    try:
        proc = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=1800,
            input=analyst_prompt(state, decision_rel),
        )
        log(f"analyst exit={proc.returncode}")
    except subprocess.TimeoutExpired:
        return None, "analyst timed out — stopping"
    if not os.path.isfile(decision_abs):
        return None, "analyst wrote no decision file — stopping"
    try:
        with open(decision_abs, "r", encoding="utf-8") as f:
            decision = json.load(f)
        return validate_decision(decision, state)
    except (ValueError, json.JSONDecodeError) as e:
        return None, f"invalid decision ({e}) — stopping"


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    with open(os.path.join(CAMPAIGN_DIR, "campaign.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def http_ok(url: str, timeout: int = 5) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def ensure_servers(args) -> None:
    """vLLM (docker) + NLI server must answer before any run step."""
    if not http_ok(f"{VLLM_URL}/v1/models"):
        log("vLLM not answering — (re)starting container...")
        subprocess.run(["docker", "start", "vllm-server"], capture_output=True)
        deadline = time.time() + 1200
        while time.time() < deadline and not http_ok(f"{VLLM_URL}/v1/models"):
            time.sleep(15)
        if not http_ok(f"{VLLM_URL}/v1/models"):
            raise RuntimeError("vLLM did not come up within 20 min (docker logs vllm-server)")
    log("vLLM OK")

    if not http_ok(f"{NLI_URL}/health"):
        log("NLI server not answering — starting...")
        nli_log = open(os.path.join(CAMPAIGN_DIR, "nli_server.log"), "a")
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                       | getattr(subprocess, "DETACHED_PROCESS", 0))
        subprocess.Popen(
            [sys.executable, "scripts/nli_server.py", "--port", "8100",
             "--device", args.nli_device],
            cwd=PROJECT_ROOT, stdout=nli_log, stderr=nli_log, **kwargs,
        )
        deadline = time.time() + 600
        while time.time() < deadline and not http_ok(f"{NLI_URL}/health"):
            time.sleep(10)
        if not http_ok(f"{NLI_URL}/health"):
            raise RuntimeError("NLI server did not come up within 10 min "
                               f"(see {CAMPAIGN_DIR}/nli_server.log)")
    log("NLI OK")


def discover_instances(treatment_dir: str) -> list[str]:
    """Instances that produced metadata in the treatment run (eval targets)."""
    out = []
    if not os.path.isdir(treatment_dir):
        return out
    for name in sorted(os.listdir(treatment_dir)):
        if os.path.isfile(os.path.join(treatment_dir, name, "metadata.json")):
            out.append(name)
    return out


def run_step(step: dict, args) -> None:
    """Run one step (subprocess or eval-loop), retry once, raise on failure."""
    name = step["name"]
    log_path = os.path.join(CAMPAIGN_DIR, "logs", name.replace("/", "__") + ".log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if "eval_dir" in step:
        # Per-instance eval loop with resume (skip existing trajectory_eval files).
        eval_dir = step["eval_dir"]
        instances = discover_instances(eval_dir)
        # The resample control writes runN/ subdirs per instance, not metadata.json
        # at the top — fall back to the predictions file for instance discovery.
        if not instances:
            pred = os.path.join(eval_dir, "predictions_all_trajectories.jsonl")
            if os.path.isfile(pred):
                with open(pred, "r", encoding="utf-8") as f:
                    instances = sorted({json.loads(l)["instance_id"]
                                        for l in f if l.strip()})
        if not instances:
            raise RuntimeError(f"{name}: no instances discovered in {eval_dir}")
        for iid in instances:
            marker = os.path.join(eval_dir, f"trajectory_eval_{iid}.json")
            if os.path.isfile(marker):
                log(f"  {name}: {iid} already evaluated, skipping")
                continue
            cmd = [sys.executable, "scripts/eval_all_trajectories.py",
                   "--results-dir", eval_dir, "--instance", iid]
            _run_logged(cmd, name=f"{name}:{iid}", log_path=log_path,
                        timeout=step["timeout"] // max(1, len(instances)) + 1800)
        return

    _run_logged(step["cmd"], name=name, log_path=log_path, timeout=step["timeout"])


def _run_logged(cmd: list[str], name: str, log_path: str, timeout: int) -> None:
    for attempt in (1, 2):
        log(f"  step {name} (attempt {attempt}): {' '.join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cmd))[:200]}")
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\n===== {datetime.now().isoformat()} attempt {attempt}: {cmd}\n")
            lf.flush()
            try:
                proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=lf, stderr=lf,
                                      timeout=timeout)
            except subprocess.TimeoutExpired:
                log(f"  step {name} TIMED OUT (attempt {attempt})")
                continue
        if proc.returncode == 0:
            return
        log(f"  step {name} failed rc={proc.returncode} (attempt {attempt}); log: {log_path}")
    raise RuntimeError(f"step {name} failed twice — see {log_path}")


def run_spec(key: str, state: dict, args) -> None:
    log(f"=== SPEC {key} ({MENU[key]['why']}) ===")
    entry = {"spec": key, "status": "running",
             "started": datetime.now().isoformat(timespec="seconds")}
    state["phase_log"].append(entry)
    save_state(state)
    for step in build_steps(key):
        ok, why = guardrails_ok(state, args)
        if not ok:
            entry["status"] = f"aborted ({why})"
            save_state(state)
            raise SystemExit(f"guardrail stop: {why}")
        ensure_servers(args)
        run_step(step, args)
    entry["status"] = "completed"
    entry["finished"] = datetime.now().isoformat(timespec="seconds")
    entry["metrics_path"] = spec_dirs(key)["metrics"]
    state["completed_specs"].append(key)
    save_state(state)
    log(f"=== SPEC {key} COMPLETE -> {entry['metrics_path']} ===")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--go", action="store_true", help="Run (default: print the plan)")
    p.add_argument("--resume", action="store_true", help="Continue interrupted campaign")
    p.add_argument("--max-phases", type=int, default=4,
                   help="Analyst-chosen phases after Phase A (default 4)")
    p.add_argument("--max-hours", type=float, default=48.0)
    p.add_argument("--min-disk-gb", type=float, default=150.0)
    p.add_argument("--analyst-model", default="claude-fable-5")
    # CPU is the known-good NLI device alongside vLLM at 0.85 GPU utilization:
    # measured 31.9/32.6 GB VRAM used with the model loaded, so deberta-large
    # + SDLG gradient backprop cannot fit on the GPU concurrently.
    p.add_argument("--nli-device", default="cpu")
    args = p.parse_args()

    state = load_state(args.resume)

    if not args.go:
        print("DRY-RUN. Phase A steps:")
        for s in build_steps(PHASE_A_KEY):
            print(f"  {s['name']}: " + (" ".join(s["cmd"]) if "cmd" in s
                                        else f"eval loop over {s['eval_dir']}"))
        print(f"\nMenu for analyst phases (max {args.max_phases}): {list(MENU)}")
        print("Pass --go to run.")
        return

    log(f"CAMPAIGN START (resume={args.resume}) state={STATE_PATH}")
    log(f"guardrails: max_hours={args.max_hours} min_disk_gb={args.min_disk_gb} "
        f"max_phases={args.max_phases}")

    # Phase A (confirmatory) — always first, exactly once.
    if PHASE_A_KEY not in state["completed_specs"]:
        run_spec(PHASE_A_KEY, state, args)
    else:
        log(f"Phase A ({PHASE_A_KEY}) already complete — continuing to analyst phases")

    # Analyst-driven phases.
    for n in range(1, args.max_phases + 1):
        ok, why = guardrails_ok(state, args)
        if not ok:
            log(f"guardrail stop before analyst phase {n}: {why}")
            break
        choice, rationale = run_analyst(state, n, args)
        log(f"analyst decision {n}: {choice or 'stop'} — {rationale[:300]}")
        if choice is None:
            break
        run_spec(choice, state, args)

    log("CAMPAIGN DONE. Completed specs: " + ", ".join(state["completed_specs"]))
    log("Next (human): read results/metrics_*.json, fill RESULTS.md §5 via "
        "compute_metrics output only, regenerate figures (scripts/make_figures.py).")


if __name__ == "__main__":
    main()
