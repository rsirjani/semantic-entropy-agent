# Applied amendment 07 — draw accounting starts at the fork decision (R7.2)

**Channel:** 1 (auto-applicable correction — logically derivable from the
existing R4.1/R7.2 completeness contract and the iteration-6 "symmetric
counting" rule), with a small channel-2 component (the producer-side
re-run-safety rule for delimiter-less artifacts).

## What changed in GOLD_STANDARD.md

R7.2 gained one clause: (a) a fork that fails at **creation**
(container/clone/injection error) is a genuine draw and must be recorded as a
failed empty-patch trajectory, exactly as the resample driver records a
crashed resample; (b) a branch whose injected response **submits** during
creation is a completed draw whose patch must be captured, never discarded;
(c) artifacts without batch delimiters (the resample arm's all-trajectories
file) must be made re-run-safe at the producer via per-instance row
replacement, because no parser-side last-batch rule can isolate a smaller-k
re-run there.

## Derivation

R7.2 (iteration 6) already requires "one prediction row per genuine draw — a
trajectory that failed or produced no diff still consumed budget and must
appear as an empty-patch row," and iteration 6's standing decision is
"run-loop failures = failed draws in BOTH arms." The fork-creation case is the
same rule applied one step earlier in the lifecycle: the orchestrator decided
to draw (a cluster representative was chosen, a container/clone was paid for),
so the draw exists whether or not the trajectory object survived construction.
Before this iteration:

- `phased_orchestrator._create_lazy_trajectory` returned `None` on any
  exception and the run loop `continue`d — the strategy's draw vanished from
  `manager.trajectories`, hence from `total_trajectories`, the predictions
  rows, and the metric-time k (strategy arm).
- `_clone_for_sdlg` wrapped everything, **including the execution of the
  injected alternative**, in one `except Exception`. Since
  `inject_and_execute → execute_actions` raises `Submitted` when the
  alternative contains the submit command, a completed child carrying a real
  (possibly passing) patch was logged as "Failed to clone," discarded, and its
  container leaked (SDLG arm). This direction is silent data loss *against*
  the treatment; the generic-failure direction deflates the treatment's k
  *for* it. Both violate the completeness contract.
- The vanilla resample driver, by contrast, records an empty-patch row for
  every failed resample — so the asymmetry was exactly the one the iteration-6
  fix removed at the predictions producer, surviving one layer up.

The bias mechanism is iteration 6's M1 verbatim: dropped unproductive draws
shrink the treatment's recorded r below its true k, and the matched-k*
comparison then scores the treatment at pass@r(r, c) against vanilla's
pass@r(k, c_v).

(c) is the iteration-6 batch-aware-loader rule confronted with an artifact
that *cannot* carry the rule: the resample file has no "primary" delimiter
rows, so a shrinking-k re-run leaves stale surplus `runN` rows that
keep-last-per-tid resurrects into the vanilla arm's k, rarefaction
denominator, and diversity pool. Producer-side per-instance replacement
(mirroring the driver's own primary-file semantics) is the only place the
batch boundary is known.

## Implementation (same iteration, fully tested)

- `src/agent/trajectory.py`: `Trajectory.agent`/`env` may be `None` for
  failed-at-creation placeholders; `save()` no-ops on placeholders;
  `cleanup()` was already None-safe.
- `src/agent/phased_orchestrator.py`: `_register_failed_draw` (placeholder
  registration + immediate container reaping), `_inject_alternative`
  (classifies injection outcome: active / completed-Submitted-with-patch /
  failed), `_create_lazy_trajectory` and `_clone_for_sdlg` rewired; the SDLG
  fork index advances on every attempt so a failed draw's id is never
  overwritten.
- `scripts/run_resample_baseline.py`: `replace_instance_rows` — per-instance
  replace in `predictions_all_trajectories.jsonl` on re-run.
- `tests/test_end_to_end_mocked.py`: 6 new tests, including the mocked
  end-to-end chain queued by iteration 6 (synthetic trajectories →
  `collect_patch_entries` → `build_predictions` → eval loaders →
  `compute_metrics.compare`) and a shrinking-re-run batch test.

## Rejected alternatives

- **Synthesize missing fork rows at metric time** — rejected for the same
  reason iterations 5/6 rejected metric-side patching: the artifact would lie
  at source and the synthesized tids would be guesses.
- **Count failed creations nowhere in either arm (filter vanilla's infra
  failures out too)** — rejected: infra-vs-model failure is not
  machine-distinguishable from artifacts (iteration-6 standing decision);
  symmetric counting plus the `nonempty_patch_fraction` diagnostic plus the
  re-run-infra-failures runbook rule is the established policy.
- **Let `Submitted`-at-injection children re-enter the run loop** — rejected:
  the child is finished by definition (the agent submitted); recording it
  completed with its patch and reaping the container is the truthful record.
- **Parser-side batch rule for the resample file** (mirror of iteration 6's
  loader fix) — impossible: the file has no delimiter rows; adding synthetic
  delimiter rows would change the schema every consumer parses. Producer-side
  replacement is schema-preserving and matches the primary file's existing
  semantics.

## Tripwire check (the one forbidden effect)

Nothing flips to `pass`: the rubric items' statuses are unchanged by the spec
edit itself, the pilot artifacts remain smoke-only (they were produced by
older drivers and already trigger the mechanical mismatch warning), and the
amendment **raises** the evidence bar — the treatment must now carry even its
failed fork attempts in its own denominator, and a class of silent
treatment-patch loss is closed. No evidence requirement for the headline
claim is reduced; the corresponding work (code + tests) was done in the same
iteration, not waived.
