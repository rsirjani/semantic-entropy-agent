r"""Autonomous experiment campaign: confirmatory pilot -> analyst-driven refinement.

Executes the experiment plan the scrutiny loop converged on (scrutiny_03.md
"minimal sufficient experiment set", as amended by iterations 4-7: hierarchical
H1 diversity -> H2 coverage confirmatory family, producer-level draw accounting,
both-arm budget audits), then lets a headless Fable analyst choose follow-up
phases FROM ACTUAL DATA, within hard guardrails. Deterministic execution, model-
driven phase selection — the analyst only ever picks from a fixed menu; every
command that runs is constructed by this script.

Pre-registration boundary: the analyst orders/stops EXPLORATORY cells only. The
confirmatory dataset is the FIRST completed Phase A run — a later repeat
(strategy_t0.7_seed2) estimates sampling variance and can never replace or pool
into the primary. Decision files (campaign_decisions/decision_*.json) are
checked-in artifacts; the adaptive exploratory selection is disclosed in
RESULTS.md §2.2.

Phase A (confirmatory, always first — the pre-registered primary endpoint):
  1. strategy arm  @ T=0.7, greedy clustering, tau=0 superset run (T and tau
     both passed EXPLICITLY on the command line — the confirmatory cell is
     defined by (T, tau) and neither may ride on a config default, R2.4)
  2. per-trajectory SWE-bench eval of the treatment
  3. matched-k vanilla control @ T=0.7 (k read from the treatment's metadata)
  4. per-trajectory eval of the control
  5. compute_metrics (H1/H2 matched-k* endpoints + exact sign-flip + power
     floors), budget_audit (BOTH arms — the fairness comparison needs both
     token totals), tau_sweep (treatment, post-hoc R3.3/R5.5)

Then up to --max-phases analyst-chosen phases from MENU (exploratory cells:
SDLG @0.7, temperature 0.2/1.0 cells, clustering variants, a repeat seed), each
with the same treatment->eval->control->eval->metrics pipeline. The analyst writes
campaign_decisions/decision_<N>.json; invalid or "stop" ends the campaign.

Guardrails: wall-clock cap, disk floor, one-run-per-spec, per-step retry-once,
a STOP file (campaign_decisions/STOP) aborts between steps; the analyst window
is integrity-guarded on BOTH planes (git porcelain for code/config, content
hashes for results/decision artifacts — adaptivity reads measured data, never
alters it). State persists in results/campaign/campaign_state.json; --resume
continues an interrupted campaign (runs use --skip-existing; evals skip
instances whose trajectory_eval exists; --max-phases counts completed analyst
phases globally, not per invocation).

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
                 # tau=0 superset run pinned EXPLICITLY (R2.4-class): the whole
                 # post-hoc tau ablation (R3.3) and the "superset" framing rest
                 # on this value, so it must not ride on a config default that
                 # an edit could silently change between cells.
                 "--entropy-threshold", "0",
                 "--diversity-method", spec["arm"],
                 "--skip-existing"],
         "timeout": 10 * 3600,
         "needs_servers": True},
        {"name": f"{key}/treatment_eval", "eval_dir": d["treatment"],
         "timeout": 8 * 3600},
        {"name": f"{key}/control_run",
         "cmd": [py, "scripts/run_resample_baseline.py",
                 "--treatment-dir", d["treatment"],
                 "--results-dir", d["control_base"],
                 "--temperatures", str(t), "--skip-existing"],
         "timeout": 16 * 3600,
         "needs_servers": True},
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
        # R6.3 is per-ARM accounting: the fairness claim ("the control received
        # at least as much compute") is only verifiable by comparing BOTH arms'
        # token totals at matched k, so the control is audited too.
        {"name": f"{key}/budget_audit_control",
         "cmd": [py, "scripts/budget_audit.py", "--results-dir", d["control"],
                 "--eval", d["control"], "--reference-cap", "250",
                 "--out", os.path.join(RESULTS, f"budget_audit_resample_{key}.json")],
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
 1. review_loop/scrutiny_07.md then scrutiny_04.md (the current design: the
    hierarchical confirmatory family — H1 = rarefied distinct gain at matched
    k*, H2 = diverse-pass@k* gain tested ONLY if H1 rejects — exact sign-flip
    inference with its tie-imposed power floor, producer-level draw
    accounting); scrutiny_03.md for the original endpoint pre-registration;
 2. RESULTS.md §2.2 (confirmatory family + matching rules) and §6 threats —
    especially threat 11 (gate-signal saturation): check the tau_sweep
    output's realized entropy distribution and realized-N flags BEFORE
    interpreting any branching numbers;
 3. the metrics JSONs of every completed phase listed below (read the
    comparison block: H1 rarefied distinct gain + exact sign-flip p +
    min_achievable_p beside it; the `confirmatory_family` field states
    whether the fixed-sequence gate is open — H2 is confirmatory ONLY if
    H1 rejected, and only in the pre-registered cell;
    nonempty_patch_fraction per arm — a productivity gap can masquerade as a
    diversity gap; k-mismatch report; selected-pass@1 with its
    degenerate-tiebreak count);
 4. the budget_audit_*.json (BOTH arms — confirm the control's token total is
    >= the treatment's at matched k) and tau_sweep_*.json companions in results/.

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
do NOT chase a positive. The confirmatory dataset is the FIRST completed Phase A
run; strategy_t0.7_seed2 estimates sampling variance and may never replace, pool
into, or re-litigate the primary. Your choices order EXPLORATORY cells only —
they cannot change what is confirmatory. You may NOT invent new specs, edit
code, or launch runs yourself.

Write EXACTLY one file, `{decision_path}`, valid JSON:
{{"choice": "<menu key or stop>", "rationale": "<3-8 sentences grounded in the
numbers you read>", "expectations": "<what result would mean what>"}}
The LAST thing you do must be writing that file. Make no other changes."""


def artifact_fingerprint(skip: set[str] | None = None) -> dict[str, str]:
    """sha256 of every file under results/ and campaign_decisions/.

    Excludes results/campaign/ (the campaign's OWN mutable area: campaign.log,
    nli_server.log, state, step logs — legitimately written while an analyst
    runs) and any absolute paths in `skip` (the decision file the current
    analyst call is expected to write).

    Why this exists: `unexpected_tree_changes` deliberately ignores results/
    and campaign_decisions/, because the campaign writes there itself. But
    those trees hold the DATA PLANE — the metrics/eval/predictions artifacts
    every later scheduling decision reads and the paper's results are filled
    from, plus the prior decision files that form the R6.5 audit chain. The
    analyst runs with permissions skipped, and prompts are not enforcement:
    an analyst edit to a metrics JSON (or to an earlier decision file) would
    otherwise pass unnoticed and silently steer every later phase. Hashing
    before/after the analyst call makes 'adaptivity reads data, never writes
    it' a checked invariant instead of a hope.
    """
    import hashlib
    skip = {os.path.abspath(p) for p in (skip or set())}
    campaign_abs = os.path.abspath(CAMPAIGN_DIR)
    out: dict[str, str] = {}
    for root in (RESULTS, DECISIONS_DIR):
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if os.path.abspath(os.path.join(dirpath, d)) != campaign_abs]
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                if os.path.abspath(fp) in skip:
                    continue
                h = hashlib.sha256()
                try:
                    with open(fp, "rb") as f:
                        for chunk in iter(lambda: f.read(1 << 20), b""):
                            h.update(chunk)
                except OSError:
                    continue
                try:
                    key = os.path.relpath(fp, PROJECT_ROOT)
                except ValueError:  # different drive (Windows) — absolute key
                    key = fp
                out[key.replace("\\", "/")] = h.hexdigest()
    return out


def changed_artifacts(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Paths added, removed, or modified between two artifact fingerprints."""
    return sorted(p for p in set(before) | set(after)
                  if before.get(p) != after.get(p))


def _tree_fingerprint() -> str:
    """`git status --porcelain` snapshot (empty string if git unavailable)."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=60)
        return out.stdout or ""
    except Exception:
        return ""


def unexpected_tree_changes(before: str, after: str) -> list[str]:
    """Status lines that appeared during the analyst run and touch code/config.

    The analyst is prompted to write ONLY its decision file, but it runs with
    permissions skipped — prompts are not enforcement. Any new change outside
    campaign_decisions/ or results/ (code, configs, tests, specs) would make
    every later phase run silently modified experiment code, so the campaign
    must stop loudly instead.
    """
    old = set(before.splitlines())
    flagged = []
    for line in after.splitlines():
        if line in old or not line.strip():
            continue
        path = line[3:] if len(line) > 3 else line
        p = path.split(" -> ")[-1].strip().strip('"').replace("\\", "/")
        if p.startswith(("campaign_decisions/", "results/")):
            continue
        flagged.append(line)
    return flagged


def run_analyst(state: dict, n: int, args) -> tuple[str | None, str]:
    """Invoke headless Fable to write decision_<n>.json; validate; return choice."""
    os.makedirs(DECISIONS_DIR, exist_ok=True)
    decision_rel = f"campaign_decisions/decision_{n:02d}.json"
    decision_abs = os.path.join(PROJECT_ROOT, decision_rel)
    if os.path.isfile(decision_abs):
        # Resume restarts numbering at 1, so a decision file from an earlier
        # (interrupted) campaign can already sit at this path. If THIS analyst
        # call then failed to write, the stale file would be read as its
        # output and the campaign would execute a choice nobody just made —
        # archive it first so only a freshly written file is ever validated.
        os.replace(decision_abs, decision_abs + ".superseded")
    exe = shutil.which("claude")
    if not exe:
        return None, "claude CLI not found — stopping (campaign keeps Phase A results)"
    cmd = [exe, "-p", "--output-format", "json", "--model", args.analyst_model,
           "--max-turns", "40", "--dangerously-skip-permissions"]
    tree_before = _tree_fingerprint()
    # Data-plane integrity (R6.5): snapshot AFTER archiving the stale decision
    # file (so the .superseded copy is part of the baseline) and excluding the
    # one file this analyst call is supposed to write.
    artifacts_before = artifact_fingerprint(skip={decision_abs})
    try:
        proc = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=1800,
            input=analyst_prompt(state, decision_rel),
        )
        log(f"analyst exit={proc.returncode}")
    except subprocess.TimeoutExpired:
        return None, "analyst timed out — stopping"
    flagged = unexpected_tree_changes(tree_before, _tree_fingerprint())
    if flagged:
        return None, ("analyst modified the working tree outside "
                      f"campaign_decisions/results ({flagged[:5]}) — stopping; "
                      "inspect `git status` before resuming")
    tampered = changed_artifacts(artifacts_before,
                                 artifact_fingerprint(skip={decision_abs}))
    if tampered:
        return None, ("analyst modified results/decision artifacts "
                      f"({tampered[:5]}) — stopping; adaptivity may read "
                      "measured data, never alter it. Restore the files (they "
                      "are regenerable from predictions via compute_metrics/"
                      "budget_audit/tau_sweep) before resuming")
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


def expected_model_id(config_path: str | None = None) -> str | None:
    """The served-model id the runs will request, from configs/branching.yaml.

    litellm routes "openai/<id>" to the local vLLM endpoint with model=<id>,
    so the id after the provider prefix must match what the container serves.
    """
    config_path = config_path or os.path.join(PROJECT_ROOT, "configs", "branching.yaml")
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        name = ((cfg or {}).get("model", {}) or {}).get("model_name", "") or ""
    except Exception:
        return None
    return (name.split("/", 1)[1] if name.startswith("openai/") else name) or None


def model_mismatch_error(expected: str | None, served: list[str]) -> str | None:
    """Error message iff the vLLM container demonstrably serves the wrong model.

    A standing `docker start vllm-server` can resurrect a container built for a
    DIFFERENT model than the config expects; every model call would then 404
    and the campaign would burn its retry budget on a misconfiguration. Only
    flags a *demonstrable* mismatch (both sides known) — an unreadable config
    or model list never blocks a run.
    """
    if not expected or not served:
        return None
    if expected in served:
        return None
    return (f"vLLM serves {served} but configs/branching.yaml expects "
            f"'{expected}' — wrong container/model for this campaign (R7.4). "
            f"Restart vllm-server with the configured model or fix the config.")


def served_model_ids(timeout: int = 10) -> list[str]:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{VLLM_URL}/v1/models", timeout=timeout) as r:
            data = json.load(r)
        return [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)]
    except Exception:
        return []


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
    err = model_mismatch_error(expected_model_id(), served_model_ids())
    if err:
        raise RuntimeError(err)
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
        # Only the agent runs need vLLM/NLI; evals need Docker only, and the
        # metrics/audit/sweep steps are pure post-processing. Requiring the
        # servers for those would let a dead vLLM container block metrics that
        # are computable from artifacts already on disk.
        if step.get("needs_servers"):
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

    # Analyst-driven phases. The cap counts COMPLETED analyst-chosen specs
    # across resumes — restarting the counter at 1 on --resume would (a) make
    # --max-phases a per-invocation budget a resume silently refills, and
    # (b) collide decision-file numbering with earlier phases' files, breaking
    # the 1:1 analyst-phase <-> decision_<n>.json mapping the R6.5 audit chain
    # relies on. An interrupted phase (decision written, spec incomplete)
    # correctly reuses its n: the stale file is archived and the decision
    # re-made.
    n_done = sum(1 for k in state["completed_specs"] if k != PHASE_A_KEY)
    for n in range(n_done + 1, args.max_phases + 1):
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
