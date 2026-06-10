# Scrutiny record — iteration 12 (first-principles design review)

Charter: fresh-eyes principal-scientist pass over the campaign branch.
Iteration 11 closed the evaluation layer (classify-or-refuse verdicts,
content-keyed eval cache) and left a named worklist: the remaining unaudited
third-party seams — the swebench dataset loader, the NLI server protocol, the
litellm/vLLM response handling beyond the stage tests — plus verification of
the harness log-marker constants against the installed swebench version, and
the end-to-end doc chain. This pass executed exactly that worklist and found
the last unaudited seam was not clean: the **SDLG arm's generator-side HTTP
plumbing and its failure semantics** carried two real defects, one of which
would have silently disabled a documented method component in the live
configuration.

---

## 1. The big picture

**The claim the paper should defend (one falsifiable sentence):**

> On n=10 easy SymPy SWE-bench-Verified instances, at matched per-instance
> trajectory count and matched sampling temperature (T=0.7), semantic-entropy-
> gated branching (a) produces more structurally distinct final patches than
> resampling the identical phased agent (H1: rarefied distinct gain at matched
> k\*, exact paired sign-flip test), and (b) given (a), is more likely to
> contain a passing fix (H2: matched-k\* diverse-pass@k gain, exact test) —
> falsified if the matched-k vanilla control matches branching's rarefied
> diversity, or matches its coverage.

Unchanged from iterations 4–11; re-examined and still the right claim. The
repo serves this claim, not a weaker substitute.

**Ideal evidence:** the standing machinery (matched-k\* Chen coverage,
rarefied independent diversity, fixed-sequence H1→H2 exact tests, τ sweep from
the τ=0 superset, both-plane integrity-guarded campaign, classified-or-refused
eval verdicts) PLUS this iteration's closing clause: **every mechanism the
paper names must be the mechanism that actually ran.** An ablation arm whose
generator silently substitutes a different diversity source — or whose
documented scoring term silently evaluates to zero because an HTTP URL was
mis-derived — produces artifacts that are internally consistent and entirely
misleading. Mechanism fidelity is upstream of attribution, which is upstream
of the ablation reading.

**Minimal sufficient experiment set:** unchanged — (1) strategy arm T=0.7 τ=0
superset; (2) matched-k vanilla T=0.7; (3) exploratory T∈{0.2, 1.0}; (4) SDLG
arm T=0.7; τ ablation post-hoc at zero GPU. Campaign dry-run re-verified this
iteration (plan prints the full pinned command sequence, T and τ explicit).

---

## 2. Findings (with evidence pointers)

### TRUTHFULLY

- **T1 (major, fixed) — the SDLG importance term was dead in the live
  configuration: `rstrip("/v1")` ate the port.**
  `src/diversity/sdlg.py::_get_importance_scores` derived the generator's
  server root as `api_base.rstrip("/v1").rstrip("/")`. `str.rstrip` strips a
  **character set**, not a suffix: `"http://localhost:8001/v1"` →
  `"http://localhost:800"` (the port's trailing `1` is in the set `{/,v,1}`).
  The checked-in config (`configs/branching.yaml`) pins `api_base` to port
  **8001** — host port 8000 is owned by the PDF-reader relay on this machine —
  so in the actual planned runs **every** importance query (the top-k logprobs
  call AND the echo-scored fallback) would hit a dead port, every exception
  would be swallowed at `logger.debug`/`return None`, and every `I_ij` would
  be 0.0. Consequences: (i) RESULTS.md §2.4 deviation 1 documents a text-level
  vocabulary bridge ("I_ij computed under the generator's own tokenization …
  unit-tested") that would not actually have executed — a doc/artifact
  mismatch of exactly the kind this charter hunts; (ii) SDLG candidate ranking
  silently degrades from (A+S+I)/3 to (A+S)/3, i.e. the implemented method is
  no longer the documented method, with no trace in any artifact. The mocked
  tests passed because the fake `requests.post` ignores the URL. Port 8000
  would have worked **by luck** (`'0' ∉ {/,v,1}`) — which is presumably how
  the bug survived early development on the default port. *Fix:* suffix-safe
  derivation (strip trailing `/`, then remove a literal `/v1` suffix once) +
  regression test asserting every importance query URL starts with
  `http://localhost:8001/v1/` under the live config
  (`tests/test_sdlg_importance.py::test_importance_scores_hit_the_configured_port`).
- **T2 (major, fixed) — the SDLG arm silently switched diversity mechanisms
  on failure, mis-attributing branches and breaking the temperature match.**
  `sdlg.py::generate` ended with: if no alternatives were produced (thought
  < 5 words, ranking failure, NLI hiccup) → `_fallback_temperature(...)` →
  `TemperatureSampler(temperature=0.7)`. Two corruptions: (i) **R3.1
  attribution** — the orchestrator clusters whatever `generate()` returns and
  records every fork as `event: "sdlg_fork"` (`phased_orchestrator.py:1161`),
  so temperature-sampled branches would be read as gradient-guided off-mode
  substitution in the generator ablation and in R5.4's
  "confidence-independent mechanism" analysis; the only trace was a stderr
  warning, in no artifact. (ii) **R2.4 temperature match** — the fallback
  hardcoded T=0.7; in the MENU's exploratory cells (sdlg arm reachable at
  T=0.2/1.0) fallback-hit instances would generate at 0.7 against a vanilla
  control at the cell's T. *Fix:* fallback removed entirely — no alternatives
  → no fork; the instance stays a single greedy trajectory, flagged by the
  realized-N reporting. This also unifies failure semantics across the two
  treatment arms (the strategy arm already collapses to the dominant single
  representative when its proposer under-delivers). R3.1 amended to name the
  invariant (`spec_amendments/applied_12_arm_purity_no_silent_fallback.md`);
  RESULTS.md §2.1 disclosure added. Test: `litellm.completion` monkeypatched
  to raise — proving the sampler can never fire — while `generate()` returns
  exactly `[greedy]`.

### Seams audited CLEAN (the iteration-11 worklist, discharged)

- **Harness log-marker constants** verified against the installed swebench:
  `swebench/harness/constants/__init__.py:80` (`APPLY_PATCH_FAIL = ">>>>>
  Patch Apply Failed"`) and `run_evaluation.py:215/218` ("Timeout error: …",
  "Test timed out after …") match the inlined constants in
  `scripts/eval_all_trajectories.py:60-64` exactly.
- **Dataset loader** (`src/evaluation/dataset.py`): alias resolution, explicit
  instance filter with requested-order sort, and the 10-instance target list
  byte-identical to CLAUDE.md's table. `run_eval.py` pins
  SWE-bench/SWE-bench_Verified + namespace=None (local images). No defect.
- **NLI server/client protocol** (`scripts/nli_server.py`,
  `src/diversity/nli_client.py`): endpoint shapes consistent; the client's
  `get_embedding_matrix()` deliberately raises and `sdlg.py` routes ranking
  server-side via `/sdlg_rank` when `server_url` exists (sdlg.py:293-302), so
  the embedding matrix never crosses HTTP; label order
  (contradiction/neutral/entailment) consistent between `NLIModel.classify`
  and deberta-large-mnli's head. No defect.
- **NLI truncation seam:** clustering is context-conditioned with
  `problem_statement[:500]` (chars ≈ 125 tokens) at BOTH call sites
  (phased_orchestrator.py:718, 1091), so a (context+intent, context+intent)
  pair stays within DeBERTa's 512-token window with the intent text intact —
  a full-length problem statement would have truncated the intents away and
  collapsed every pair to entailment ≈ 1. Worth knowing it is the [:500] cap
  that protects this; documented here.
- **litellm/vLLM response handling:** every `litellm.completion` call site
  guards `content or ""`; the proposer/intent/SDLG-completion instrument
  calls are correctly temperature-0 (they are measurement/splice machinery,
  not the diversity source — the proposer's *diversity* call alone reads
  `sample_temperature`). The vLLM logprobs request (`logprobs=20`) sits at
  vLLM's default `max_logprobs` cap, not above it. `VLLMClient`/`react_agent`
  are imported only by the legacy `run_baseline.py`, which is outside the
  campaign path (R2.3).
- **Estimator stack, third spot-check** (not a full re-derivation — two stand
  from iterations 9–10): `pass_at_k` product form ≡ 1−C(n−c,k)/C(n,k);
  rarefaction as Σ_sig pass@k(n, m_sig, k) with empty patches in n but never
  contributing a signature; exact sign-flip enumerates all 2^n masks with the
  identity mask included (p ≥ 2^−n floor intact); seeded percentile bootstrap.
  The N=5 achievable-entropy grid {0, .500, .673, .950, 1.055, 1.332, 1.609}
  re-computed by hand from the 7 partitions of 5 — matches RESULTS.md §2.3.

### MATHEMATICALLY

- **M1 — no new defect in the decision-grade math.** This iteration's two
  fixes touch no estimator; T1 changes only the SDLG arm's candidate-ranking
  inputs (restoring the documented (A+S+I)/3), and T2 changes failure-mode
  branching semantics (no-fork instead of mechanism swap). Both leave the
  confirmatory cell (strategy arm) untouched.
- **M2 — fairness direction of T2 checked:** removing the fallback can only
  REDUCE the SDLG arm's realized k on failure instances (k=1 tie at matched
  k\*), i.e. it is conservative for the treatment, consistent with the
  project's standing rule that accounting fixes must never manufacture
  treatment wins.

### PHILOSOPHICALLY

- **P1 — framing re-audited, still coherent and non-circular.** H1's
  diversity metric remains mechanism-independent; τ quantization stated
  plainly; the τ=0 headline + post-hoc sweep answer the "gate never gates"
  objection. Nothing new to confess.
- **P2 — the weakest joint** remains gate saturation (threat 11) and n=10
  power — both pre-answered with pre-committed readings. After this iteration
  a hostile reviewer probing the SDLG arm ("how do you know those branches
  came from SDLG?") has an artifact-level answer: the arm has no other
  mechanism to produce them, by construction and by test.
- **P3 — the generalizing lesson, continuing the 5→11 series:** iteration 11
  showed a guarded pipeline can be poisoned by a third-party tool's silent
  failure semantics; iteration 12's instance is one layer closer to home —
  **a guarded pipeline can be poisoned by its own silent fallbacks.** A
  fallback written for development robustness (always produce candidates)
  is, in an attribution experiment, a confound generator. The audit chain now
  extends: … → third-party verdict semantics → **own-code failure semantics
  (fallbacks must fail closed, into the arm's null action, never sideways
  into a different mechanism)**.

---

## 3. Steelmanned alternatives (this iteration's decisions)

| Design choice | Strongest alternative | Decision |
|---|---|---|
| Remove the SDLG temperature fallback (no alternatives → no fork) | Keep a fallback but thread the configured `sample_temperature` and write `fallback: true` into `branching_log.json` | **Remove.** A flagged mixture still answers "SDLG-or-temperature vs strategy", not the ablation's question; at n=10 even 1–2 mixture instances blur an exploratory contrast with little resolution to spare; matched-k makes no-fork costless for fairness (k=1 tie); and the strategy arm already fails into its null action (dominant representative), so both arms now share one failure semantics. Rejected-alternative record in `applied_12_arm_purity_no_silent_fallback.md`. |
| Suffix-safe base_url derivation in sdlg.py | Pass the server root as its own config key (no derivation) | **Derive, safely.** A second config key for the same server invites the R8.2 divergence class (two keys describing one endpoint can disagree); the derivation is one line once it is suffix-correct, and the regression test pins the live-config URL. |
| Test the URL via recorded `requests.post` calls | Integration test against a real local server | **Mocked URL assertion.** R8.5's charter is GPU/server-free stage tests; the defect was in URL *construction*, which the mock observes exactly. |
| Strengthen R3.1 in the spec now | Leave the invariant in code+tests only | **Strengthen.** Every prior iteration's lesson got encoded where the next optimizer will trip over it; a future contributor adding a "robustness" fallback to either arm should hit a named spec clause, not re-discover the confound. Tripwire-checked: adds an obligation discharged same-iteration; flips nothing. |

Standing decisions re-examined and left in place: trajectory-matched budget
(conservative direction); τ=0 superset headline + command-pinned post-hoc
sweep; intent-summary clustering substrate (the [:500] context cap audited
this iteration is part of why it works); fixed-sequence H1→H2 family
(untouched — still no data); majority-signature selector with degeneracy
disclosure; 10-easy-SymPy scope; no-retune rule under gate saturation; R6.5
adaptive boundary + both-plane integrity guard; classify-or-refuse eval
verdicts; generation-layer symmetric failure counting.

Known limitations recorded, not fixed (disclosed): the echo-scoring path
(`max_tokens=0, echo=True`) is exercised against vLLM only at run time — if a
vLLM version rejects it, the disclosed degradation (top-k misses score 0.0)
applies and is visible in logs; harness log-marker constants remain inlined
strings coupled to the pinned swebench version (verified this iteration); the
iteration-10 porcelain caveat (campaign must start from a clean tree) stands.

---

## 4. Actions taken this iteration

All verified: **132 pytest pass** (130 → 132; +2 new, 0 removed), `py_compile`
clean on touched files, campaign dry-run prints the full pinned plan.

1. `src/diversity/sdlg.py` — suffix-safe `base_url` derivation in
   `_get_importance_scores` (T1); `_fallback_temperature` removed and
   `generate()` returns `[greedy]` with a logged no-fork warning when no
   alternatives exist (T2); stale comment updated.
2. `tests/test_sdlg_importance.py` — +2: live-config port preserved in every
   importance-query URL; no-alternatives path returns only greedy and can
   never call `litellm.completion`.
3. `GOLD_STANDARD.md` R3.1 — fallback-layer attributability clause
   (amendment record: `spec_amendments/applied_12_arm_purity_no_silent_fallback.md`,
   tripwire check inside).
4. `RESULTS.md` §2.1 — "Arm purity at the fallback layer" disclosure
   (fallback existed, what it broke, why removed, what happens instead).
5. `review_loop/scrutiny_12.md` — this record.

## 5. What still requires a human / GPU

1. Ratify the iteration-12 amendment
   (`applied_12_arm_purity_no_silent_fallback.md`) and any earlier unratified
   ones — independent spec-critic review per ratchet v2.
2. **Launch the campaign from this branch from a CLEAN working tree**:
   `python scripts/run_campaign.py --go`.
3. After the runs: check threat 11 first (realized entropy distribution);
   confirm `pred_eval_count_mismatch` / `k_mismatch_instances` empty; read H1
   against `nonempty_patch_fraction`; read H2 through `confirmatory_family`;
   verify both-arm budget audits; fill RESULTS §5 from script output only.
4. Git hygiene: decide whether this campaign branch becomes mainline.

## 6. Verdict logic

This iteration discharged the iteration-11 worklist and found the last
unaudited seam dirty: (T1, major) the SDLG importance term — a documented
method component — would have been silently zero in the live configuration
because of a character-set/suffix confusion in URL derivation; (T2, major)
the SDLG arm silently substituted temperature sampling when SDLG failed,
mis-attributing branches in the generator ablation and breaking the
temperature match in sweep cells. Both fixed, tested, and encoded (R3.1
clause). Per the charter, finding real fixable issues means
`gold_standard_met` = **false**; the loop should pass once more. The next
iteration arrives at a branch where every named seam — spec, metrics,
producers, artifacts, branch protocol, commands, state machine, data plane,
third-party verdict semantics, and now own-code fallback semantics — has had
a dedicated audit; if a fresh pass over the doc chain and any remaining
corner (e.g. `relevance.py` scoring internals, the figures script's reading
of partial artifacts) surfaces nothing substantive, the design has nothing
left to confess and the remaining actions are human-only (ratification + GPU
launch).
