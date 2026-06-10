# Applied amendments — iteration 15

Two GOLD_STANDARD.md edits, one per channel.

---

## A. R1.2 — clustering context policy (channel 1: derivable correction)

**Before:** "clustering is **context-conditioned** (problem statement
prepended) consistently at every call site."

**After:** "clustering applies a **consistent context policy at every call
site** — for this project the NLI context is deliberately **empty**, a
documented deviation from Kuhn's QA-style context-conditioning" (with the
measured saturation evidence and test pointer).

**Derivation (why this is a correction, not a preference):** the spec as
written was internally inconsistent with the repository's own evidence chain.
Phase A run-1 was archived (threat 11, RESULTS.md) precisely because
context-conditioning saturated DeBERTa entailment: all 10 pairs of five
structurally distinct strategies scored ≥0.94 entailment WITH the shared
500-char prefix vs ≤0.55 WITHOUT (reproducible via
`scripts/diagnose_context_saturation.py`) — the gate measured the prefix, not
the strategies (8/10 instances collapsed to one cluster, entropy 0, k=1).
The code was fixed accordingly (commit 107f60b, every call site `context=""`,
pinned by `tests/test_clustering_context.py`), but R1.2 still *mandated* the
defective instrument. Kuhn's conditioning exists to disambiguate
context-DEPENDENT short QA answers ("Paris" vs "France"); our clustering
substrate is self-contained imperative sentences, where the rationale does not
apply and the mechanism actively harms the measurement. An R1.2 audit under
the old wording would have had to grade the *correct* implementation as a
deviation and the *defective* one as compliant.

**Tripwire check:** R1 was `pass` before and after; no failing item flips; no
evidence requirement for the headline claim is reduced (the gate instrument is
*stronger* under the corrected wording — it can actually distinguish
strategies). The edit documents an already-implemented, already-disclosed,
measurement-backed deviation.

**Rejected alternatives:**
- *Keep context-conditioning with a shorter prefix (e.g. 100 chars).* Any
  shared prefix inflates pairwise entailment in the same direction; the
  saturation diagnostic showed the effect at the configured length, and no
  principled non-zero length exists. The substrate (self-contained sentences)
  removes the need entirely.
- *Condition on a per-pair-distinct context.* Contradicts the purpose
  (context is shared by construction) and has no support in Kuhn/Farquhar.
- *Delete the context parameter from the clusterer.* Rejected: the parameter
  documents the deviation at the call sites and keeps the ablation door open
  for QA-style substrates; the spec now requires a consistent *policy*, which
  the test pins.

---

## B. R2.3 — code-revision symmetry (channel 2: value-level strengthening)

**Edit:** "Scaffold-matched includes code-revision-matched" — all arms of a
compared cell must be produced by the same committed revision; the campaign
driver pins HEAD at start and refuses steps after a commit/checkout or with
tracked files modified.

**Scientific argument:** the scaffold IS part of the experimental apparatus;
two arms run at different code revisions are not the same scaffold even if no
diff line "should" matter. The need is measured, not hypothetical: Phase A
run-2's treatment ran at 07:13 and the anti-gaming guard set (command veto +
`--network none`) landed at 09:52 — before the control existed. Audit of the
archived run-2 artifacts: 209/1,539 treatment actions (13.6%) would have been
vetoed under the guard the control would have run with (and the treatment's
containers had real internet access — its 22 `pip install` attempts could
reach PyPI). The asymmetry was discovered and the cell archived while no
treatment-vs-control comparison existed (the control never ran), so the
archival cannot be outcome-chasing. The amendment converts the one-off fix
into a standing requirement with driver enforcement
(`run_campaign.guardrails_ok`) and a pinning test.

**Tripwire check:** this RAISES the bar (a new requirement), and the
corresponding work is done in the same iteration (driver enforcement + tests +
run-2 archived + RESULTS.md disclosure). It does not flip any item to pass for
the existing artifact — it makes the existing artifact *farther* from done
(the confirmatory cell must be re-run), which is the opposite of self-serving.

**Rejected alternatives:**
- *Salvage run-2 with disclosure instead of archiving.* Rejected: the control
  cannot be run without the guards (deliberately reintroducing a measured leak
  channel to "match" a flawed treatment is absurd), and a 13.6%-of-actions
  behavioral-envelope difference plus an open-network treatment is not a
  footnote-sized asymmetry. The R6.5 "first completed confirmatory run is the
  dataset" rule exists to prevent outcome-chasing; archiving for a
  protocol/scaffold defect decided before any paired outcome existed is the
  same legitimate exception run-1 used (instrument defect), and both archives
  are preserved and disclosed.
- *Pin only the confirmatory cell, let exploratory cells float across
  revisions.* Rejected: every cell is a treatment-vs-control comparison whose
  within-cell symmetry carries the inference; the cost of pinning the whole
  campaign is zero (don't edit code mid-campaign).
- *Tree-hash (porcelain) check only, no HEAD pin.* Rejected: a commit leaves
  the tree clean while changing the code — exactly the run-2 failure mode
  (guards were *committed* mid-campaign).
