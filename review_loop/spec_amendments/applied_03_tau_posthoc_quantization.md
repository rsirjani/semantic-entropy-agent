# Applied amendment (iteration 3): τ sweep evaluated post-hoc from the τ=0 superset run; entropy quantization disclosed

**Rubric item amended:** R3.3 (redefined evidence path + added disclosure requirement).
**Channel:** mixed — the post-hoc equivalence is derivable (channel 1); requiring the
quantization disclosure is value-level (channel 2).

## The derivable equivalence

The previous R3.3 accepted "a documented plan or a run showing branch-rate vs τ",
implicitly suggesting separate GPU runs per τ. That is unnecessary, and the
equivalence is exact by construction of the orchestrator:

1. The τ=0 run branches every cluster, producing trajectory i = cluster i's
   representative strategy (`t0` for i=0, else `t0_strategy_<i>`), each executed
   greedily (temp 0) after the fork (`phased_orchestrator.py::_propose_strategies`,
   run loop at the fork site).
2. A hypothetical τ>0 run with the same SEARCH outcome takes the no-branch action
   iff entropy ≤ τ, keeping exactly ONE trajectory: the dominant (largest,
   lowest-index tie-break = `max(clusters, key=len)`) cluster's representative,
   executed greedily.
3. That trajectory is *the same computation* as the corresponding trajectory in
   the τ=0 superset run — same history prefix, same strategy text, same greedy
   decoding. The SDLG arm's no-branch action keeps the greedy parent `t0`,
   likewise present in the superset run.

Therefore every τ's outcome (branch decision, trajectories used, gated pass) is a
deterministic *subset selection* over the superset run's artifacts — implemented in
`scripts/tau_sweep.py`, unit-tested with synthetic logs
(`tests/test_tau_sweep.py`), and smoke-verified on the real `results/branching`
artifacts. Validity caveat recorded in the script output: equivalence holds modulo
vLLM nondeterminism (the same caveat any rerun carries).

This is NOT a reduction of evidence: the post-hoc sweep produces the same numbers
a per-τ rerun would, for every achievable τ at once, and the saved GPU budget is
redirected to the headline matched-k runs.

## The added disclosure requirement (quantization)

With N=5 candidates, discrete semantic entropy takes exactly the 7 values
attainable by integer partitions of 5: {0, 0.500, 0.673, 0.950, 1.055, 1.332,
1.609} nats. Consequences the spec now forces into the open:

- τ is a **cluster-partition-shape rule** at this N, not a continuous dial; only
  τ values straddling adjacent achievable entropies differ. The admissible τ grid
  IS the achievable set (the script computes and reports it for any N).
- τ=0 (the headline default) ≡ "branch iff ≥2 semantic clusters". The headline
  config's gate is real but degenerate-by-design: the τ=0 run intentionally
  collects the full diverse set, and the *gate* claims are carried by the post-hoc
  sweep (R3.3) and the R5.5 counter-analysis, not by the headline row.
- The plug-in entropy estimator is biased low (Miller–Madow ≈ (K−1)/2N nats); the
  gate is *defined* on the plug-in value, which stays comparable only because N is
  fixed (n_strategies = sdlg_n_alternatives = 5) across arms and instances — now a
  stated requirement.

## Rejected alternatives

- *Change the headline default to τ>0 so the title's gate visibly fires:* rejected.
  The τ=0 superset run strictly dominates — it contains every τ>0 run's outcome and
  enables the entire sweep post-hoc, whereas a τ>0 headline run would discard the
  branches needed for the coverage/diversity claim and for the τ ablation itself.
  The honest fix is disclosure (the paper must say the headline run branches on
  any non-trivial entropy) — applied in RESULTS.md §2.3 — not a weaker run design.
- *Continuous τ grid (e.g. 0.1 steps):* misleading at small N — adjacent grid
  points are mostly equivalent, inflating the apparent resolution of the ablation.
- *Larger N to smooth the quantization:* N=5 is a compute-bound design constant;
  raising it changes the budget and the matched-k control. Out of scope for v1;
  the quantization is disclosed instead.

**Tripwire check.** R3.3's previous bar ("documented plan or a run") was already
satisfiable by a plan; the amendment *raises* it to "runnable script + quantization
disclosure", and the script exists and is tested as of this iteration. No
ungathered evidence is excused: gated pass-rates on the real headline runs still
await the GPU runs, listed in next_actions.
