# Applied amendment 14 — R8.1 strengthened: segment-aware write detection + enforced SEARCH read-only boundary

**Channel:** value-level design amendment (scrutiny charter, ratchet v2 channel 2) —
strengthening an existing requirement, argued from mechanism fidelity and a
measured defect, not from convenience.

## What changed in GOLD_STANDARD.md

R8.1 previously required only "no `echo`/submit/stderr false positives" in
write-command detection. It now additionally requires:

1. **No quoted-program false positives.** Comparison operators inside quoted
   awk/python programs (`awk 'NR>=350 {print}' file`) and heredoc body content
   must not read as redirects.
2. **No non-final-segment false negatives.** A write in any top-level
   `&&`/`;`/newline segment is a write (`sed -i … && pytest`).
3. **Enforced SEARCH read-only boundary.** The SEARCH phase must veto write
   commands with the same detector, because the prefix allowlist alone admits
   `echo … > file` (the `echo` prefix is an allowed read command).
4. **Submission decided before prefix matching** (VERIFY-only).
5. **Unenforced boundaries documented as such** (PATCH/VERIFY allowlists are
   prompt-level guidance; the orchestrator never checks them).

## Scientific argument

- **Branch-point fidelity (R1.1).** SDLG fires exactly once, at the first
  detected write. Measured on the 2,184 pilot actions, the old detector had 2
  false positives of the quoted-comparison class. In the SDLG arm, a false
  positive that precedes the first real write moves the *entire instance's*
  branch point to a view step — the mechanism the paper attributes branches to
  ("token substitution at the first implementation commitment") would be false
  for that instance. The pilot was the strategy arm; the SDLG arm has not run
  yet, so this was the last cheap moment to fix it.
- **Fork-state consistency.** `PhasedOrchestrator._create_lazy_trajectory`
  documents and depends on "search phase doesn't modify files": forks are
  fresh containers replaying the search *messages*, not filesystem clones. The
  old `is_command_allowed` let `echo fix > file.py`, `cat <<EOF > f.py`, and
  even the submit command through in SEARCH via prefix matches — a documented
  invariant with no guard, exactly the iteration-13 lesson ("a disclosed
  limitation is not yet a guarded one") one layer deeper: an *assumed*
  invariant is not yet a guarded one.
- **Direction of bias.** Pilot evidence shows 0 SEARCH-phase writes in 189
  search actions, so enforcement changes nothing on the observed distribution
  — it closes a tail risk. The detector changes alter 9 of 2,184 historical
  verdicts: 2 false positives removed; 7 chained patch-prep redirects
  (`git diff > patch.txt && cat …`) now classify identically to their
  unchained forms (which were already writes — the documented scratch-file
  limitation, unchanged in kind, now applied consistently).

## Tripwire check (clause ii)

R8.1 was `pass` before this amendment and the *strengthened* R8.1 is met by
the same iteration's code changes (`src/agent/phases.py`,
`tests/test_pipeline_correctness.py`); no rubric item flips from fail to pass,
and no evidence required for the headline claim is reduced. The amendment
raises the bar and the work to meet it was done in the same iteration.

## Rejected alternatives

- **Full shell-grammar parsing** (bashlex or a real tokenizer): correct in
  principle, but a heavyweight dependency for a detector whose residual gaps
  (programmatic writes via `python -c "open(...)"`) no command-string parser
  can close anyway. The quote/heredoc-stripping approach is pure-string,
  stage-testable, and was validated against the full pilot action log.
- **Enforcing VERIFY read-only too:** rejected — the submit protocol *requires*
  a redirect (`git diff > patch.txt`), a VERIFY-phase edit is still the
  agent's own work and lands in its captured patch, and the cost of blocking a
  legitimate submission flow mid-campaign far exceeds the value of the
  boundary. Documented as guidance instead (module docstring + R8.1 wording).
- **Special-casing `> patch.txt` as a non-write:** rejected — any filename
  special case invites its own false negatives, and the scratch-file
  limitation is already documented with the mitigation (callers needing
  precision diff the tree).
- **Leaving the spec unchanged** (the code is fixed either way): rejected —
  iteration 13's generalizing lesson is that guard behavior the project relies
  on must be named in the spec so a future regression is a spec violation, not
  a silent drift.
