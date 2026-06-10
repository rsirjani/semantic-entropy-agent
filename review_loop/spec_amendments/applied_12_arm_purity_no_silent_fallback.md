# Applied amendment (iteration 12) — R3.1: arm attributability extends to the fallback layer

**Channel:** value-level design amendment (scrutiny charter, ratchet v2 channel 2).
**Spec edit:** `GOLD_STANDARD.md` R3.1 — added the clause that a diversity arm
which cannot produce candidates at a decision point must produce NO branch
there, never silently substitute a different generator.

## The defect that motivated it (verified in code, fixed same-iteration)

`src/diversity/sdlg.py::generate` ended with:

```python
if len(candidates) <= 1:
    return self._fallback_temperature(...)   # TemperatureSampler(temperature=0.7)
```

Two distinct corruptions, both invisible in the artifacts:

1. **Mechanism mis-attribution (R3.1).** When SDLG could not produce a
   substitution (thought < 5 words, ranking failure, NLI server hiccup), the
   arm silently switched to plain temperature sampling. The orchestrator
   (`phased_orchestrator._apply_sdlg`) has no way to see this — it clusters
   whatever `generate()` returns and records every fork as
   `event: "sdlg_fork"` in `branching_log.json`. A reader of the SDLG-arm
   results (the R3.1 generator ablation, and R5.4's "confidence-independent
   mechanism" claim) would attribute temperature-sampled diversity to
   gradient-guided off-mode substitution. The single warning line went to
   stderr, not to any artifact.
2. **Temperature-match break (R2.4).** The fallback hardcoded T=0.7. In the
   exploratory sweep cells (sdlg arm at T=0.2 / 1.0, available on the campaign
   MENU via the analyst), fallback-hit instances would generate candidates at
   0.7 while the matched vanilla control ran at the cell's T — an
   uncontrolled, undisclosed temperature asymmetry inside single instances.

## Why "no branch" rather than "fallback + flag"

- **Steelman of keeping a flagged fallback:** it preserves branch-rate parity
  between SDLG and strategy arms, and a `fallback: true` field in
  `branching_log.json` would make it auditable. Rejected because the arm's
  *scientific role* is a generator ablation: a mixture arm — even a disclosed
  one — answers "SDLG-or-temperature vs strategy", not "SDLG vs strategy".
  At n=10 instances, even one or two mixture instances materially blur an
  exploratory contrast that already has limited resolution; and the matched-k
  control matches whatever k the treatment realizes, so a no-branch instance
  costs nothing in fairness (it becomes a k=1 tie, the conservative outcome).
- **Steelman of threading the configured `sample_temperature` into the
  fallback (fixing only R2.4):** repairs the temperature match but leaves the
  attribution leak intact. Rejected — the leak is the deeper defect.
- **No-branch** is also exactly how the strategy arm already behaves when its
  proposer under-delivers and clustering collapses to one cluster (single
  dominant representative, no fork), so the two treatment arms now share one
  failure semantics: *generator produces nothing distinct → no branch →
  realized-N flags the instance*.

## Tripwire check (the one forbidden move)

- Flips no rubric item to `pass` for the existing artifact: R3 was already
  scored pass on the runnable-and-attributable basis; this edit ADDS an
  obligation, and the code change discharging it landed in the same commit
  (`sdlg.py` fallback removed; `tests/test_sdlg_importance.py::
  test_generate_without_alternatives_stays_pure_sdlg` proves the sampler can
  never fire — `litellm.completion` is monkeypatched to raise).
- Reduces no evidence required for the headline claim — the confirmatory cell
  (strategy arm) is untouched; this constrains an exploratory arm to be more
  honest, not less work.
- De-scopes nothing; discloses the removed fallback in RESULTS.md §2.1.
