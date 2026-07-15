# Frozen Experiment Status

The current controlled ABIDES experiment package is complete. It is not a
smoke test or a parameter screen. No result-triggered parameter search, seed
replacement, episode extension, or selective learner/ecology reporting was
used after the formal protocols were frozen.

## Decisions

| Experiment family | Current decision |
| --- | --- |
| Adaptive-v5 fresh panel | Risk-aware timing is better than matched random and circular-shift timing on all 20 policies; no-routing returns close to no intervention |
| Official full-policy MAPPO | Supported by policy-seed t and paired two-way bootstrap intervals; 8/10 policy seeds positive |
| Official IPPO | Positive point estimate but inconclusive intervals; no universal learner claim |
| E1 and E2 ecology holdouts | Supported in both pre-frozen N=10 perturbations, 10/10 policies positive in each |
| Dose curve | Mixed; matched random wins at dose 0.10 and the reversal is retained |
| Routing sensitivity | Locally favorable around the tested region, not uniformly favorable under every estimator/cell |
| Forced-undercut diagnostic | Supports high-spread coordination only; punishment/retaliation claim rejected |
| Brero/Banerjee adaptations | Executable but descriptive because frozen integrity or matching gates failed |

## Included Evidence

- frozen formal configurations and calibration artifacts;
- current ABIDES matching-engine adapter, economic environment, routing, gate,
  and learner implementations;
- paper-facing per-seed tables and figures;
- per-path, per-seed, and compressed per-step audit data for the holdout,
  sensitivity, diagnostic, and external-adaptation packages;
- policy-seed t intervals, exact sign tests, paired two-way bootstrap intervals,
  and secondary multiplicity checks where specified by protocol;
- release-local SHA-256 hashes and row counts;
- regression tests and a CI verifier for scope, integrity, and anonymity.

The fresh adaptive-v5 panel is committed at per-seed granularity. The other
retained audit packages expose finer per-path and per-step records. Internal
source-history provenance is withheld during double-blind review; third-party
ABIDES and official MAPPO pins remain visible for dependency reproducibility.

## Scope Boundary

The supported statement is restricted to controlled N=10 ABIDES dealer
markets. Use `high-spread coordination`, not punishment-supported or learned
collusion. Do not claim universal cross-learner performance, superiority over
the original external mechanisms, larger-N robustness, cross-matching-engine
validity, or real-market effectiveness.

Development archives, superseded result summaries, superseded engineering cells,
engineering iterations, local workstation outputs, author metadata, and source
repository history are deliberately outside this anonymous release. Negative
evidence that changes the scientific conclusion remains included in the current
package.
