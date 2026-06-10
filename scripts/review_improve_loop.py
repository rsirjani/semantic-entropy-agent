r"""Autonomous review -> improve -> assess loop, driven by GOLD_STANDARD.md.

Each iteration invokes the Claude Code CLI headlessly (`claude -p ... --output-format
json`) with a prompt that tells it to:
  1. AUDIT the repository against the rubric in GOLD_STANDARD.md (and the papers in
     PDFs/ it references),
  2. IMPLEMENT the highest-value in-scope fixes that move toward the gold standard,
  3. SELF-ASSESS and write review_loop/verdict_<NN>.json scoring every rubric item,
     with `gold_standard_met` and `blocking_gaps`.

The wrapper parses that verdict and decides whether to iterate again. It STOPS when
the gold standard is met and stable for --require-stable iterations, or on a cap
(max iterations / cost / stuck-with-no-progress).

SAFETY: this edits files and runs commands autonomously. By default it runs in
DRY-RUN (composes and prints the prompt + the exact claude command, invokes
nothing). Pass --go to actually run. When live, all work happens on an isolated
git branch (review-loop/<timestamp>) with a commit per iteration, so every step is
revertible. Full multi-hour GPU/SWE-bench runs are forbidden to the agent unless
--allow-experiments is given; otherwise it verifies wiring via smoke checks only.

Two charters (--charter):
  rubric   — the original mode: audit against GOLD_STANDARD.md, fix gaps, verdict.
  scrutiny — first-principles design review: re-derive the math, stress the
             philosophy/framing, steelman alternatives to every experimental
             choice, and refine the experiments BEFORE the GPU runs make design
             changes expensive. Value-level design changes still go through the
             spec ratchet (quarantined proposals); findings land in
             review_loop/scrutiny_<NN>.md. The loop stops when two consecutive
             iterations find no further substantive design flaw.

Examples:
  # See exactly what it would do, change nothing:
  python scripts/review_improve_loop.py

  # Original rubric loop (isolated branch, 12 iterations, $20 cap):
  python scripts/review_improve_loop.py --go --charter rubric --max-iterations 12 --max-cost-usd 20

  # First-principles scrutiny loop on Fable (resume numbering after iter 0-2):
  python scripts/review_improve_loop.py --go --charter scrutiny --model claude-fable-5 \
      --start-iteration 3 --max-iterations 6 --max-cost-usd 25
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.path.join(PROJECT_ROOT, "GOLD_STANDARD.md")
STATE_DIR = os.path.join(PROJECT_ROOT, "review_loop")
HISTORY_PATH = os.path.join(STATE_DIR, "history.jsonl")
SPEC_AMEND_DIR = os.path.join(STATE_DIR, "spec_amendments")


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

def compose_prompt(iteration: int, verdict_path_rel: str, allow_experiments: bool) -> str:
    experiments_clause = (
        "You MAY launch longer experiment runs if genuinely necessary."
        if allow_experiments else
        "You MUST NOT launch the multi-hour GPU / vLLM / full-SWE-bench runs. To "
        "judge readiness, verify the pipeline is WIRED and CORRECT via lightweight "
        "smoke checks only (py_compile, imports, clustering smoke tests, and "
        "`--help` / `--dry-run` of the driver scripts). Wiring + smoke proof is "
        "sufficient for the readiness verdict per the spec's DONE section."
    )
    return f"""\
You are the autonomous reviewer/engineer for a PhD research codebase. This is
iteration {iteration} of an improvement loop. Work entirely within this repository
(cwd is the project root).

YOUR RUBRIC IS THE FILE `GOLD_STANDARD.md` AT THE REPO ROOT. Read it first and in
full. It is the objective definition of "publication-ready" for this project and
references papers in `PDFs/` (read the relevant ones — e.g. proposal.pdf,
Aichberger_2025_SDLG.pdf, farquhar_nature.pdf, Farquhar_2024_Semantic_Entropy.pdf
— with the file-system-windows-python read-file MCP tool when you need to check
faithfulness). Also read `CLAUDE.md`, `VALIDATION_BRIEF.md`, and the prior
verdicts in `review_loop/` if present.

Do these three things, in order:

1. AUDIT the repository against every rubric item in GOLD_STANDARD.md. For each,
   determine pass / partial / fail with a concrete evidence pointer (file:line,
   command, or results artifact). Be adversarial and specific — find real gaps
   between the code and the spec / the papers, not cosmetic nits. Prefer
   correctness and evidence-validity issues over style.

2. IMPLEMENT the highest-value in-scope fixes that move the repo toward the gold
   standard THIS iteration. Make real edits. Keep changes focused and correct;
   match existing code style. {experiments_clause} TEST THE PIPELINE PART BY PART:
   prefer adding/running the stage-level tests in R8.5 (mocked, no GPU/Docker) so
   each component — search, proposal+clustering, entropy/branch gate, SDLG
   write-point, source-only patch capture, matched-k driver, metric scripts — is
   proven in isolation. Treat the §0.1 epistemic-vs-aleatoric uncertainty framing
   and its stratified analysis (§5), and model-swappability/ensembles (§9), as
   first-class rubric items, not afterthoughts. Verify your edits (py_compile,
   imports, smoke/stage tests) before claiming a rubric item passed. Do not weaken
   a BLOCKER, delete tests, fabricate results, or narrow the instance set to force
   a pass — if you think the spec itself is wrong, say so in `next_actions` instead.

SPEC EVOLUTION (the ratchet — read GOLD_STANDARD.md's "Spec evolution" rule): you
MAY edit `GOLD_STANDARD.md` ONLY to apply a *derivable, rigor-INCREASING correction*
(a biased estimator, an inconsistent definition, a broken identity), and you MUST
record the derivation + classification in
`review_loop/spec_amendments/proposal_{iteration:02d}_<slug>.md`. For ANYTHING that
relaxes a BLOCKER, rests on a value/convention judgment, or would make the current
artifact pass, DO NOT edit the spec — write it as a proposal file only. An
independent spec-critic reviews every edit to GOLD_STANDARD.md after this iteration
and the wrapper REVERTS it unless it is a confirmed rigor-increasing correction, so
a self-serving change is simply undone. Don't waste effort trying to sneak one past.

3. SELF-ASSESS honestly and WRITE the verdict file `{verdict_path_rel}` as JSON
   EXACTLY matching the schema at the bottom of GOLD_STANDARD.md (the
   `verdict_<NN>.json` block). Set `iteration` to {iteration}. Score every R1..R10
   key. Set `gold_standard_met` true ONLY if every BLOCKER item is `pass` and
   `blocking_gaps` is empty, per the spec's DONE section. List concrete remaining
   work (including human-only steps such as GPU runs you cannot perform here) in
   `next_actions`. A rigorous negative/null result is acceptable — do NOT bias
   toward a positive finding.

CRITICAL: the LAST thing you do must be writing `{verdict_path_rel}` as valid JSON.
The wrapper reads only that file to decide whether to loop again. If you cannot
complete all fixes, still write the verdict reflecting current state.
"""


def compose_scrutiny_prompt(iteration: int, verdict_path_rel: str,
                            allow_experiments: bool) -> str:
    experiments_clause = (
        "You MAY launch longer experiment runs if genuinely necessary."
        if allow_experiments else
        "You MUST NOT launch the multi-hour GPU / vLLM / full-SWE-bench runs; verify "
        "wiring and correctness via the mocked stage tests and driver smoke checks only."
    )
    return f"""\
You are the principal scientist doing a FIRST-PRINCIPLES design review of this PhD
research project, iteration {iteration} of a scrutiny loop. Work entirely within
this repository (cwd is the project root). The headline GPU experiments have NOT
been run yet — this is the last cheap moment to change the design. Your job is not
rubric compliance (a prior loop did that); it is to make these the best, most
truthful experiments this project can run, and to make sure everything makes sense.

ORIENT FIRST: read GOLD_STANDARD.md (the rubric + §0.1 framing), CLAUDE.md,
RESULTS.md (esp. §2.4 documented deviations and §6 threats), the prior verdicts and
any scrutiny_*.md in review_loop/, and the papers in PDFs/ as needed (use the
file-system-windows-python read-file MCP tool). Then do the following, in order:

1. ARTICULATE THE BIG PICTURE (write it down before judging anything). What is the
   end goal — what exact scientific claim should the final paper defend, what would
   the ideal evidence for it look like, and what is the minimal sufficient
   experiment set? State the claim in one falsifiable sentence. If the repo's
   current design serves a different (weaker, vaguer, or merely easier) claim than
   the one worth defending, say so explicitly.

2. SCRUTINIZE through three lenses, in writing, with evidence pointers:
   - TRUTHFULLY: does every claim in the docs/framing correspond to what the
     artifacts and design can actually show? Hunt overclaims, silent assumptions,
     and confounds (budget asymmetries, selection effects, eval gaps, circularity).
   - MATHEMATICALLY: re-derive, do not trust. The pass@k estimator and its use at
     matched k; the discrete-entropy estimator at small N (quantization, bias); the
     bootstrap's validity at n=10 instances; the KLE heat-kernel limits; whether
     trajectory-count matching is the right budget match given branched
     trajectories share a SEARCH prefix while vanilla resamples pay full cost
     (trajectory-matched vs token-matched — which comparison is fair, and for
     which claim?); statistical power and multiple-comparison exposure across the
     temperature sweep and ablations.
   - PHILOSOPHICALLY: is the §0.1 mode/diversity framing coherent and
     non-circular? Is "diversity" defined independently of the mechanism that
     produces it? Are the falsifiable predictions actually falsifiable at this n?
     What is the weakest joint a hostile reviewer would press, and does the design
     answer it or merely acknowledge it?

3. STEELMAN ALTERNATIVES: for each major design choice (clustering substrate =
   intent summaries; gate signal = discrete semantic entropy; matched-k definition;
   τ default; the two diversity arms; the independent diversity metric; the
   instance set), name the strongest alternative, and either justify the current
   choice against it or propose the change. Known open questions you should weigh
   (do NOT limit yourself to these): per-instance k-match is enforced at run time
   but not at metric time; τ=0 default means the headline config never exercises
   the gate the title claims; selected-pass@1 has no selector implementation;
   entropy from ~5 candidates takes few distinct values, so τ is effectively a
   cluster-count rule — say so plainly if true.

4. ACT on what you find, this iteration: implement in-scope improvements (code,
   tests, analysis scripts, docs) with real edits, verified (py_compile, pytest).
   {experiments_clause} Write the full scrutiny record — big picture, findings,
   decisions, rejected alternatives WITH reasons — to
   `review_loop/scrutiny_{iteration:02d}.md`. Design changes that rest on value
   judgments or would alter what the rubric demands go through the spec ratchet:
   write them as quarantined proposals in `review_loop/spec_amendments/` for human
   ratification, do NOT edit GOLD_STANDARD.md for them (derivable rigor-INCREASING
   corrections may be applied directly, per the spec's ratchet rule — an
   independent critic reviews and reverts anything else).

5. WRITE THE VERDICT `{verdict_path_rel}` as JSON exactly matching the schema at
   the bottom of GOLD_STANDARD.md, iteration {iteration}. In this charter,
   `gold_standard_met` means: every BLOCKER still passes AND this iteration found
   NO further substantive design flaw or unresolved scrutiny question (minor
   style/wording nits do not count). If you found and fixed real issues, set it
   false so the loop runs again; the loop should only go quiet when the design has
   nothing left to confess. Put anything requiring a human (GPU runs, value-level
   ratifications) in `next_actions`. A rigorous null result is as good as a
   positive one — do not bend the design toward producing a win.

CRITICAL: the LAST thing you do must be writing `{verdict_path_rel}` as valid JSON.
The wrapper reads only that file to decide whether to loop again.
"""


VERDICT_ONLY_PROMPT = (
    "You did NOT leave a valid verdict file. Without making further code changes, "
    "write `{verdict_path_rel}` now as valid JSON exactly matching the schema at "
    "the bottom of GOLD_STANDARD.md, reflecting the current state of the repo for "
    "iteration {iteration}. Output nothing else."
)


# --------------------------------------------------------------------------- #
# claude CLI
# --------------------------------------------------------------------------- #

def find_claude() -> str:
    exe = shutil.which("claude")
    if not exe:
        sys.exit("ERROR: `claude` CLI not found on PATH. Install Claude Code first.")
    return exe


def run_claude(
    exe: str, prompt: str, *, model: str, max_turns: int, permission_mode: str,
    timeout: int, resume_session: str | None,
) -> dict:
    """Invoke `claude -p` with JSON output. Returns a normalized result dict."""
    cmd = [exe, "-p", prompt, "--output-format", "json",
           "--model", model, "--max-turns", str(max_turns)]
    if permission_mode == "bypass":
        cmd.append("--dangerously-skip-permissions")
    else:
        cmd += ["--permission-mode", permission_mode]
    if resume_session:
        cmd += ["--resume", resume_session]

    try:
        proc = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, shell=False,
        )
    except subprocess.TimeoutExpired:
        return {"is_error": True, "subtype": "timeout", "result": "",
                "session_id": None, "total_cost_usd": 0.0, "num_turns": 0,
                "raw": "", "returncode": None}

    raw = proc.stdout or ""
    out = {"is_error": proc.returncode != 0, "subtype": "unknown", "result": "",
           "session_id": None, "total_cost_usd": 0.0, "num_turns": 0,
           "raw": raw, "returncode": proc.returncode, "stderr": proc.stderr}
    try:
        data = json.loads(raw)
        out.update(
            is_error=bool(data.get("is_error", out["is_error"])),
            subtype=data.get("subtype", "unknown"),
            result=data.get("result", ""),
            session_id=data.get("session_id"),
            total_cost_usd=float(data.get("total_cost_usd", 0.0) or 0.0),
            num_turns=int(data.get("num_turns", 0) or 0),
        )
    except json.JSONDecodeError:
        out["is_error"] = True
        out["subtype"] = "non_json_output"
    return out


# --------------------------------------------------------------------------- #
# Spec ratchet — an independent critic polices edits to GOLD_STANDARD.md
# --------------------------------------------------------------------------- #

CRITIC_PROMPT = """You are an INDEPENDENT adversarial reviewer of a change to a
research project's gold-standard rubric. Your ONLY job: decide whether the change is
a legitimate, mathematically/logically DERIVABLE correction that INCREASES rigor —
or a disguised relaxation / value-judgment / self-serving move that lowers the bar.

REJECT if the change does ANY of these:
- removes, weakens, or narrows a [BLOCKER] requirement;
- relaxes a definition, metric, threshold, or sufficiency bar;
- rests on a value / philosophical / "convention" judgment rather than a provable fact;
- would make it easier to declare gold_standard_met on the current artifact.
APPROVE ONLY a provable correction (e.g. fixing a biased estimator, an inconsistent
definition, a broken identity) that makes the rubric MORE rigorous and lowers no bar.
When uncertain, REJECT.

--- RUBRIC BEFORE ---
{before}
--- RUBRIC AFTER ---
{after}
--- END ---

Output EXACTLY one JSON object and nothing else:
{{"verdict": "approve" | "reject", "reason": "<one sentence>"}}"""


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _parse_critic_verdict(text: str) -> tuple[str, str]:
    """Return (verdict, reason). Defaults to ('reject', ...) on any ambiguity."""
    import re
    if text:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                v = str(obj.get("verdict", "reject")).strip().lower()
                return ("approve" if v == "approve" else "reject",
                        str(obj.get("reason", "")))
            except Exception:
                pass
    return "reject", "unparseable critic output -> conservative reject"


def enforce_spec_ratchet(exe: str, iteration: int, spec_before: str, args) -> float:
    """If the iteration edited GOLD_STANDARD.md, have an independent critic judge it.

    Reverts the change unless the critic confirms a derivable, rigor-increasing
    correction. Returns the critic's reported cost (0 if no spec change). The critic
    runs in read-only 'plan' mode so it cannot itself touch the repo.
    """
    spec_after = _read_text(SPEC_PATH)
    if spec_after == spec_before:
        return 0.0
    print("  Spec change detected in GOLD_STANDARD.md -> independent spec-critic review...")
    res = run_claude(
        exe, CRITIC_PROMPT.format(before=spec_before, after=spec_after),
        model=args.model, max_turns=4, permission_mode="plan",
        timeout=600, resume_session=None,
    )
    verdict, reason = _parse_critic_verdict(res["result"])
    record = {"iteration": iteration, "verdict": verdict, "reason": reason,
              "critic_subtype": res["subtype"]}
    os.makedirs(SPEC_AMEND_DIR, exist_ok=True)
    with open(os.path.join(SPEC_AMEND_DIR, f"critic_{iteration:02d}.json"), "w",
              encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    if verdict == "approve":
        print(f"  Spec-critic APPROVED (rigor-increasing correction): {reason}")
    else:
        with open(SPEC_PATH, "w", encoding="utf-8") as f:
            f.write(spec_before)
        print(f"  Spec-critic REJECTED -> reverted GOLD_STANDARD.md. Reason: {reason}")
    return res["total_cost_usd"]


# --------------------------------------------------------------------------- #
# git checkpointing
# --------------------------------------------------------------------------- #

def git(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", check=check)


def git_start_branch(no_git: bool) -> str | None:
    if no_git:
        return None
    if git("rev-parse", "--is-inside-work-tree").returncode != 0:
        print("  (not a git repo — skipping checkpointing)")
        return None
    branch = "review-loop/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    r = git("checkout", "-b", branch)
    if r.returncode != 0:
        print(f"  (could not create branch {branch}: {r.stderr.strip()})")
        return None
    print(f"  Working on isolated branch: {branch}")
    return branch


def git_checkpoint(branch: str | None, iteration: int, summary: str) -> None:
    if not branch:
        return
    git("add", "-A")
    if not git("diff", "--cached", "--quiet").returncode:
        return  # nothing staged
    msg = f"review-loop iter {iteration}: {summary[:120]}"
    git("commit", "-m", msg, "--no-verify")


# --------------------------------------------------------------------------- #
# verdict handling
# --------------------------------------------------------------------------- #

def verdict_path(iteration: int) -> str:
    return os.path.join(STATE_DIR, f"verdict_{iteration:02d}.json")


def load_verdict(iteration: int) -> dict | None:
    path = verdict_path(iteration)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  WARNING: verdict_{iteration:02d}.json is not valid JSON ({e})")
        return None


def summarize_verdict(v: dict) -> str:
    met = v.get("gold_standard_met")
    gaps = v.get("blocking_gaps") or []
    rub = v.get("rubric") or {}
    statuses = {k: (rub.get(k) or {}).get("status", "?") for k in sorted(rub)}
    n_changes = len(v.get("changes_made") or [])
    print(f"  gold_standard_met={met}  blocking_gaps={len(gaps)}  changes_made={n_changes}")
    print("  rubric: " + "  ".join(f"{k.split('_')[0]}:{s}" for k, s in statuses.items()))
    if gaps:
        for g in gaps[:6]:
            print(f"    - gap: {g}")
    return v.get("summary", "")


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--go", action="store_true",
                   help="Actually invoke claude (default is dry-run: print only).")
    p.add_argument("--model", default="claude-fable-5",
                   help="Model alias/id (default: claude-fable-5; 'opus'/'sonnet' aliases also work).")
    p.add_argument("--charter", default="rubric", choices=["rubric", "scrutiny"],
                   help="rubric = audit/fix against GOLD_STANDARD.md (original mode); "
                        "scrutiny = first-principles design review that re-derives the "
                        "math, stresses the framing, and refines the experiments before "
                        "the GPU runs (default: rubric).")
    p.add_argument("--max-iterations", type=int, default=12)
    p.add_argument("--max-turns", type=int, default=250,
                   help="Per-iteration agent turn cap. One thorough pass (read rubric + "
                        "PDFs, audit ~10 rubric sections, implement several fixes with "
                        "py_compile/pytest verification, write the verdict) needs ~100-150+ "
                        "turns; 250 leaves headroom so iterations COMPLETE rather than "
                        "truncate. The --timeout wall-clock cap is the real backstop.")
    p.add_argument("--timeout", type=int, default=5400, help="Per-iteration wall-clock seconds.")
    p.add_argument("--max-cost-usd", type=float, default=None,
                   help="Stop once cumulative reported cost exceeds this.")
    p.add_argument("--require-stable", type=int, default=2,
                   help="Consecutive 'gold_standard_met' verdicts required to stop (default 2).")
    p.add_argument("--stuck-limit", type=int, default=3,
                   help="Stop after this many consecutive zero-change, not-met iterations.")
    p.add_argument("--permission-mode", default="bypass",
                   choices=["bypass", "acceptEdits", "dontAsk", "plan"],
                   help="claude permission mode (default: bypass = --dangerously-skip-permissions).")
    p.add_argument("--allow-experiments", action="store_true",
                   help="Permit the agent to launch long experiment runs (default: forbidden).")
    p.add_argument("--no-git", action="store_true", help="Disable per-iteration branch/commit.")
    p.add_argument("--start-iteration", type=int, default=0,
                   help="Resume numbering from here (does not re-run earlier iterations).")
    args = p.parse_args()

    if not os.path.isfile(SPEC_PATH):
        sys.exit(f"ERROR: rubric not found at {SPEC_PATH}. Write GOLD_STANDARD.md first.")
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(SPEC_AMEND_DIR, exist_ok=True)
    exe = find_claude()

    print(f"Project: {PROJECT_ROOT}")
    print(f"Rubric:  {SPEC_PATH}")
    print(f"claude:  {exe}")
    print(f"Mode:    {'LIVE (--go)' if args.go else 'DRY-RUN (no claude invocation; pass --go to run)'}")
    print(f"Model={args.model}  charter={args.charter}  max_iters={args.max_iterations}  "
          f"max_turns={args.max_turns}  perm={args.permission_mode}  "
          f"experiments={args.allow_experiments}")

    branch = git_start_branch(args.no_git) if args.go else None

    cumulative_cost = 0.0
    consecutive_met = 0
    consecutive_stuck = 0

    for i in range(args.start_iteration, args.start_iteration + args.max_iterations):
        vpath_rel = os.path.relpath(verdict_path(i), PROJECT_ROOT).replace("\\", "/")
        composer = (compose_scrutiny_prompt if args.charter == "scrutiny"
                    else compose_prompt)
        prompt = composer(i, vpath_rel, args.allow_experiments)
        print(f"\n{'='*72}\n  ITERATION {i}\n{'='*72}")

        if not args.go:
            print("  [dry-run] Would invoke:")
            print(f"    claude -p <PROMPT> --output-format json --model {args.model} "
                  f"--max-turns {args.max_turns} "
                  + ("--dangerously-skip-permissions" if args.permission_mode == "bypass"
                     else f"--permission-mode {args.permission_mode}"))
            print("  [dry-run] PROMPT:\n" + "\n".join("    " + l for l in prompt.splitlines()))
            print("  [dry-run] stopping after one iteration preview. Pass --go to run for real.")
            return

        spec_before = _read_text(SPEC_PATH)   # ratchet: detect spec edits this iter
        res = run_claude(exe, prompt, model=args.model, max_turns=args.max_turns,
                         permission_mode=args.permission_mode, timeout=args.timeout,
                         resume_session=None)
        cumulative_cost += res["total_cost_usd"]
        print(f"  claude: subtype={res['subtype']} is_error={res['is_error']} "
              f"turns={res['num_turns']} cost=${res['total_cost_usd']:.4f} "
              f"cumulative=${cumulative_cost:.4f}")
        with open(os.path.join(STATE_DIR, f"raw_{i:02d}.json"), "w", encoding="utf-8") as f:
            f.write(res["raw"] or "")

        # Ensure a verdict exists; if not, ask once more (resume) to emit it.
        verdict = load_verdict(i)
        if verdict is None and res["session_id"]:
            print("  No verdict file — requesting verdict-only follow-up...")
            res2 = run_claude(
                exe, VERDICT_ONLY_PROMPT.format(verdict_path_rel=vpath_rel, iteration=i),
                model=args.model, max_turns=6, permission_mode=args.permission_mode,
                timeout=600, resume_session=res["session_id"],
            )
            cumulative_cost += res2["total_cost_usd"]
            verdict = load_verdict(i)

        # Spec ratchet: an independent critic polices any edit to GOLD_STANDARD.md
        # this iteration and reverts it unless it's a rigor-increasing correction.
        # Runs before git_checkpoint so a reverted change is never committed.
        cumulative_cost += enforce_spec_ratchet(exe, i, spec_before, args)

        if verdict is None:
            print("  ERROR: iteration produced no usable verdict. Recording and continuing.")
            record = {"iteration": i, "error": "no_verdict", "subtype": res["subtype"],
                      "cost_usd": res["total_cost_usd"], "cumulative_cost_usd": cumulative_cost}
            with open(HISTORY_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            consecutive_stuck += 1
            consecutive_met = 0
        else:
            summary = summarize_verdict(verdict)
            git_checkpoint(branch, i, summary or "iteration")
            record = {
                "iteration": i,
                "gold_standard_met": bool(verdict.get("gold_standard_met")),
                "blocking_gaps": verdict.get("blocking_gaps") or [],
                "n_changes": len(verdict.get("changes_made") or []),
                "confidence": verdict.get("confidence"),
                "cost_usd": res["total_cost_usd"],
                "cumulative_cost_usd": cumulative_cost,
                "session_id": res["session_id"],
            }
            with open(HISTORY_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            met = bool(verdict.get("gold_standard_met")) and not (verdict.get("blocking_gaps") or [])
            consecutive_met = consecutive_met + 1 if met else 0
            if record["n_changes"] == 0 and not met:
                consecutive_stuck += 1
            else:
                consecutive_stuck = 0

            if consecutive_met >= args.require_stable:
                print(f"\n✅ GOLD STANDARD MET and stable for {consecutive_met} iterations. Done.")
                _final(branch, cumulative_cost)
                return

        # Stop conditions
        if args.max_cost_usd is not None and cumulative_cost >= args.max_cost_usd:
            print(f"\n⛔ Cost cap reached (${cumulative_cost:.2f} ≥ ${args.max_cost_usd}). Stopping.")
            _final(branch, cumulative_cost)
            return
        if consecutive_stuck >= args.stuck_limit:
            print(f"\n⛔ Stuck: {consecutive_stuck} consecutive iterations with no progress and "
                  f"gold standard unmet. Human action needed (see latest next_actions). Stopping.")
            _final(branch, cumulative_cost)
            return

    print(f"\n⛔ Reached max iterations ({args.max_iterations}) without meeting the gold standard.")
    _final(branch, cumulative_cost)


def _final(branch: str | None, cost: float) -> None:
    print(f"  Total reported cost: ${cost:.4f}")
    print(f"  History: {HISTORY_PATH}")
    print(f"  Verdicts + raw outputs: {STATE_DIR}")
    if branch:
        print(f"  All changes are on branch '{branch}' (one commit per iteration). "
              f"Review with `git log {branch}`, merge or discard as you see fit.")


if __name__ == "__main__":
    main()
