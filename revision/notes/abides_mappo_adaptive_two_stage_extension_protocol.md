# ABIDES/MAPPO Adaptive Two-Stage Extension Protocol

Status: frozen after the 100-episode v2 result and before extension  
Date: 2026-07-12

## Trigger

The 100-episode two-stage adaptive pilot passed every integrity and efficacy
gate except convergence:

- matched-random minus learned low-risk spread: `0.12404`;
- 95% CI: `[0.03624, 0.21185]`;
- positive policy seeds: `5/5`;
- learned high-risk activation: `0.07054`;
- learned activation: `0.33540`;
- exact flow, CRN, dose, reward, and failure-mode checks: pass.

Only two of five seeds had every method satisfy the frozen convergence rule.
The failures were concentrated in always-on reward drift. Learned and matched
also marginally exceeded the 10% reward-window shift only on seed 440000.

The parent result therefore satisfies the preregistered condition for a
checkpoint extension. It is not a stable negative result, and no parameter,
method, or seed search is permitted.

## Extension

- Resume the exact model, optimizer, minibatch RNG, and history state saved at
  episode 100 for every method and training seed.
- Continue the same five training seeds to episode 150 using environment seeds
  100-149 in the existing deterministic namespace.
- Keep the two-stage gate, MAPPO, routing contract, reward, convergence rules,
  and all other settings unchanged.
- Evaluate only on fresh paths `451000-451009`; do not reuse the observed
  `450000-450009` outcomes.
- Write extension checkpoints and results to a new output directory with a new
  protocol fingerprint. Parent checkpoints remain immutable.

The runner validates that the parent failed only convergence, all other parent
checks passed, frozen fields match, and evaluation paths are fresh before it
loads any optimizer state.

## Decision

The extension passes only if all original adaptive-v2 integrity, timing,
high-risk, reward, volume, failure-mode, and convergence checks pass at episode
150. A complete pass permits the next paper-main gate: implementation of
reward-governance baselines and a second environment pilot. Any stable timing
or risk-control failure stops adaptive main. A second training extension is not
automatically allowed.
