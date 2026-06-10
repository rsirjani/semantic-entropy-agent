# Applied amendment (iteration 4): realized-N reporting + full-precision gate reconstruction (R3.3)

**Channel:** mixed — the gate-precision half is an auto-applicable *correction*
(mathematically derivable); the realized-N half is a value-level strengthening.
**Spec section edited:** R3.3 (two added sentences).

## 1. Correction: gate reconstruction must compare at full precision

Derivation. The orchestrator branches iff `entropy > τ` at full float precision.
The post-hoc τ sweep previously parsed the 3-decimal `Entropy:` log line and
applied the same comparison. Rounding can cross the gate boundary: the (2,2,1)
partition of 5 has H = −(0.4·ln0.4·… ) = **1.054920…**, which a 3-decimal log
renders as **1.055 > 1.0549** — so at the achievable-grid τ=1.0549… the sweep
said "branch" where a real run gates. Conversely (3,1,1) = 0.950271 logs as
0.950, making the sweep gate at τ=0.950 where a real run branches. Since discrete
SE is a deterministic function of the cluster partition, and the partition IS in
the log, the exact value is recoverable: recompute from sizes, use it when it
agrees with the logged value within log-rounding tolerance (6e-4), otherwise keep
the logged value flagged `parsed_log_disagrees_with_partition` (kernel/von-Neumann
entropy is *not* a partition function — silently "correcting" it would corrupt
kernel-ablation sweeps). Grid values are now full precision; the gate comparison
carries a 1e-9 epsilon solely to absorb float noise at grid-boundary τ. One
existing test asserted the *unfaithful* rounded semantics (gating at τ=0.950 with
true entropy 0.950271) and was corrected to assert the orchestrator-faithful
behavior, with a new regression test pinning the (2,2,1) boundary case. The
orchestrator now also logs 6 decimals so future artifacts do not depend on the
recompute path.

This increases rigor and flips nothing to pass: it changes a script's numerical
faithfulness and tightens the spec's demand on it.

## 2. Strengthening: realized N reported, non-modal N flagged

R3.3 already required "N held fixed across arms and instances" — but N is only
fixed *by config*; the proposer demonstrably can under-deliver
(`StrategyProposer.propose` returns <N on parse failure, and its fallback returns
a single generic strategy). Nothing measured or reported the realized N, so the
fixed-N comparability premise was an unverified assumption. The amendment makes it
a measured, disclosed quantity: the τ sweep reports `n_candidates_by_instance`,
uses the **modal** N for the grid (ties → larger N, since the configured value is
an upper bound), flags `non_modal_n_instances`, and the spec + RESULTS threat 9
now require excluding flagged instances from pooled τ/strata analyses.

## Rejected alternatives

- **Hard-enforce N=5 at run time (retry until 5 parse).** Rejected: cannot be
  guaranteed against a stochastic LLM without unbounded retries, and silent
  retry-until-compliant changes the sampling distribution of the proposals
  themselves (selection pressure toward parse-friendly outputs). Measuring and
  disclosing is the honest fix.
- **Drop under-delivered instances at run time.** Rejected: data loss decided by
  an upstream parser; the metric-time flag keeps the trajectories (they still
  count for coverage/diversity at matched k) and only guards the τ/strata
  pooling, which is the analysis the quantization argument actually touches.
- **Snap parsed entropies to the nearest achievable grid value instead of
  recomputing.** Rejected: equivalent for the strategy arm but actively wrong for
  kernel runs (von Neumann entropy legitimately lies off the partition grid);
  recompute-with-tolerance handles both and never overwrites a disagreeing value.
