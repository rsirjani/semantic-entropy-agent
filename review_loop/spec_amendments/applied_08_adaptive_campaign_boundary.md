# Applied amendment (iteration 8): R6.5 — adaptive-execution boundary for the campaign driver

**Channel:** value-level design amendment (scrutiny charter, ratchet v2 channel 2).

## What changed

R6.5 gains an **adaptive execution boundary** clause: if run scheduling is
automated by a data-reading agent (the campaign driver's LLM analyst), its
adaptivity may touch **exploratory cells only**; the confirmatory cell runs
first, exactly once, by a fixed command sequence; the FIRST completed
confirmatory run is the confirmatory dataset (a repeat draw is variance
estimation, never a replacement or pooling partner); every scheduling decision
is a checked-in artifact with a written rationale; and the writeup discloses
that the exploratory cell set is data-dependent.

## Why (argued from the science)

`scripts/run_campaign.py` (this worktree's purpose) introduces something the
pre-registration machinery of iterations 3–4 never contemplated: an **agent that
reads interim results and decides what runs next**. That is adaptive data
collection. For the *confirmatory* family it would be fatal (a scheduler that
peeks at H1 and decides whether to collect more primary data is optional
stopping); for *exploratory/descriptive* cells it is legitimate and
budget-efficient — but only if the boundary is explicit, enforced, and
disclosed. Three risks the clause closes:

1. **Optional stopping on the primary.** Without the first-run pin, the menu's
   repeat cell (`strategy_t0.7_seed2`) invites "the second draw looked better —
   report that one" or "pool the two draws" — classic garden-of-forking-paths.
   The pin makes the confirmatory dataset a function of the protocol, not the
   results.
2. **Silent adaptivity.** A reader of the paper must know that *which*
   exploratory cells exist is data-dependent (e.g. the SDLG contrast ran because
   the primary showed signal). Otherwise the exploratory table reads as a fixed
   design and invites survivorship misreadings.
3. **Unauditable scheduling.** The decision files
   (`campaign_decisions/decision_*.json`, rationale required, validated against
   the fixed menu) make the path reconstructable — the same artifacts-truthful-
   at-source principle as iterations 5–7, applied to the scheduler.

## Implementation (same iteration)

- `scripts/run_campaign.py`: docstring pins the pre-registration boundary; the
  analyst prompt now states the first-run-is-confirmatory rule and that analyst
  choices order exploratory cells only; Phase A runs first, exactly once
  (already enforced structurally: `PHASE_A_KEY` runs before any analyst is
  consulted, and `validate_decision` rejects re-running completed specs).
- The analyst prompt was also re-pointed from the stale iteration-3 design
  record to the current one (scrutiny_07/04: H1→H2 family, `min_achievable_p`,
  `nonempty_patch_fraction` productivity reading rule, threat-11 gate-saturation
  check, both-arm budget audits) — an analyst reading only scrutiny_03 would
  misinterpret the metric outputs the iteration-4–7 design produces.
- `RESULTS.md` §2.2: "Adaptive execution of the exploratory cells (disclosed)"
  paragraph with the three pinned consequences.
- Tests: `test_analyst_prompt_reflects_current_design` (needles for every
  load-bearing element), existing `test_phase_a_is_the_preregistered_primary`
  and `validate_decision` tests cover the structural enforcement.

## Tripwire check

Nothing flips to `pass`: the clause adds constraints on a component (the
campaign driver) that the rubric previously did not govern at all, and the
constraints are implemented in the same iteration. The confirmatory evidence
bar is untouched; the amendment only prevents a future weakening (optional
stopping / quiet primary replacement) before any data exist.

## Rejected alternatives

- **Forbid adaptive scheduling entirely (fixed cell order).** Rejected: the
  exploratory cells are descriptive by design (R6.5 already labels them so);
  spending the fixed GPU budget on the most informative descriptive cells is a
  pure efficiency gain with no inferential cost once the boundary is pinned.
  A fixed order would burn budget on uninformative cells (e.g. running both
  temperature cells before the mechanism contrast regardless of the primary's
  outcome).
- **Let the analyst also schedule confirmatory replications.** Rejected: any
  data-dependent decision about confirmatory data collection is optional
  stopping; if a replication policy is ever wanted it must be pre-registered
  with its own combining rule (e.g. fixed-n replication, Fisher combination),
  which is future work, not a scheduler option.
- **Disclose adaptivity only in the campaign README (not RESULTS/spec).**
  Rejected: the disclosure belongs where a reviewer reads the design (§2.2) and
  where the loop enforces it (R6.5); a README is neither.
