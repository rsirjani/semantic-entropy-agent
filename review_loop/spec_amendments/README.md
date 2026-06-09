# Spec amendments — the ratchet's audit trail

This folder records every attempt to evolve `GOLD_STANDARD.md`, so spec changes are
auditable and reversible. It is written by `scripts/review_improve_loop.py`.

- `proposal_<NN>_<slug>.md` — written by the iteration agent. A *correction* it
  auto-applied (with the derivation) OR a *quarantined proposal* (a relaxation or
  value/convention change) it must NOT apply, left for you to ratify by hand.
- `critic_<NN>.json` — the independent spec-critic's verdict on any edit the agent
  made to `GOLD_STANDARD.md` that iteration: `{"verdict": "approve"|"reject", ...}`.
  On `reject`, the wrapper reverts `GOLD_STANDARD.md` before committing, so a
  self-serving or bar-lowering change never survives.

Rule (see GOLD_STANDARD.md "Spec evolution — the ratchet"): the bar may get *harder*
for mathematically/logically derivable reasons; it may not get *easier* for
convenient ones. To ratify a quarantined proposal, edit `GOLD_STANDARD.md` yourself.
