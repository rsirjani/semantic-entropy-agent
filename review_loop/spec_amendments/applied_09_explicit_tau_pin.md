# Applied amendment (iteration 9): explicit τ pin in confirmatory/treatment commands

**Channel:** value-level design amendment (ratchet v2, scrutiny charter) — a
strengthened requirement. Arguably it is also channel-1 auto-applicable, since it
is the direct logical parallel of R2.4's existing temperature-explicitness clause
(an inconsistent standard: one of the two parameters defining the pre-registered
cell was required to be explicit, the other was not).

**The edit (GOLD_STANDARD.md R2.4):** added the sentence requiring the
entropy-gate τ to be passed explicitly (`--entropy-threshold 0`) in the
documented reproduction commands and the campaign-driver commands, never
inherited from a config default.

## Rationale (argued from the science)

The pre-registered confirmatory cell is *defined* by (arm, clustering, T, τ):
"strategy-proposal, greedy, **τ=0 superset run**, T=0.7" (R6.5, RESULTS §2.2).
Two design pillars rest specifically on τ=0:

1. **The post-hoc τ ablation (R3.3)** is valid only because the headline run is
   the *superset* of trajectories any τ>0 run would produce. If a config edit
   silently changed `entropy_threshold` between campaign cells, the sweep's
   subsetting logic would silently produce wrong gated rows — there is no
   loud failure mode.
2. **R1.5/R2** require the τ gate to be read from the same config key in every
   arm. That guarantees *consistency*, not *value*: the value that makes the
   run confirmatory was, before this amendment, visible nowhere in the
   commands that launch it.

R2.4 already encodes exactly this reasoning for temperature ("a treatment
command that silently inherits a different config default than the control's
CLI temperature is an R2.4 violation"). Temperature and τ have equal standing
in the cell definition; requiring explicitness for one but not the other was an
inconsistency, not a judgment call.

**Work discharged in the same iteration:** `scripts/run_campaign.py::build_steps`
now passes `--entropy-threshold 0` on every treatment run (all menu cells are
τ=0 superset runs — the ablation is post-hoc for every cell); RESULTS.md §5's
documented treatment command pins it likewise;
`tests/test_run_campaign.py::test_build_steps_pins_tau_superset_explicitly`
pins the requirement for every menu cell.

## Rejected alternatives

- **Test that the config default equals 0 instead of pinning the CLI flag.** A
  config-default test protects the value at test time, not at run time: the
  campaign runs for ~1–2 days, and the analyst subprocess (or any human edit)
  could alter the YAML between cells. The tree-integrity check would flag an
  analyst edit, but not a pre-launch human edit. Command-line pinning makes the
  cell definition part of the launched command itself — the same defense R2.4
  chose for temperature.
- **Pin τ only on the confirmatory cell, leave exploratory cells on the
  default.** Rejected: every exploratory cell's τ sweep also assumes the
  superset property, and a per-cell inconsistency would be a new foot-gun for
  zero savings.

## Tripwire check (anti-self-serving)

- Flips no rubric item to `pass` for the pre-existing artifact: before the
  edit, R2.4 passed under its old wording and the τ pin was satisfied by the
  config default; the amendment *adds* an obligation, discharged by
  same-iteration code/doc/test work.
- Reduces no evidence required for the headline claim; the GPU runs remain
  ungathered and are unaffected.
