# ABIDES Ecology Holdout v1 Summary

Date: 2026-07-14  
Status: both pre-frozen cells complete; formal and independent audit checks pass

## Decision

Both pre-frozen ecology holdouts preserve the favorable risk-aware timing
direction against equal-dose matched random. E1 changes risk persistence; E2
weakens informed flow. Each result is reported separately, with 10/10 positive
policy seeds. The evidence supports robustness within these two tested
ecologies, not arbitrary market ecologies or agent counts.

## Frozen Design

| Cell | Policy seeds | Frozen change from the v5 ecology |
| --- | --- | --- |
| E1 fast transition | `560000-560009` | `p_low_to_high=0.08`, `p_high_to_low=0.20` |
| E2 weaker informed flow | `570000-570009` | informed-flow probability `0.60` |

Each cell retrained base policies for 384 episodes and response policies for
96 episodes. Final evaluation used ten fresh paired paths and 100 decisions
per path. Both cells include no intervention, risk-aware, matched random,
spread-only threshold, always-on, and no-routing. The cells were frozen before
their outcomes were opened and are never pooled to conceal a reversal.

## Formal Source Results

The preregistered compact-runner endpoint pools low-risk steps within each
policy seed. Positive matched-random minus risk-aware values favor risk-aware
timing.

| Cell | Paired mean | 95% t-CI | Positive seeds | Risk/random activation | Risk/random high-risk activation |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1 | 0.03419 | [0.02860, 0.03978] | 10/10 | 0.1880 / 0.1880 | 0.0611 / 0.1832 |
| E2 | 0.03471 | [0.02909, 0.04033] | 10/10 | 0.2050 / 0.2050 | 0.0573 / 0.1527 |

Risk-aware and matched-random mean rewards are 10.6707 versus 10.6493 in E1
and 10.6016 versus 10.5997 in E2. Mean volume is exactly 20 in every method.
Both source runners pass the frozen convergence, dose, reward, routing,
failure-mode, common-path, and volume checks.

## Audit Estimand

The cross-group audit package gives each of the ten evaluation paths equal
weight within a policy seed. This differs slightly from the compact runner's
pooled-step estimator and is reported explicitly rather than treated as the
same number.

| Cell | Path-balanced paired mean | 95% t-CI | Exact sign p | Two-way bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: |
| E1 | 0.03328 | [0.02776, 0.03880] | 0.00195 | [0.02238, 0.04400] |
| E2 | 0.03984 | [0.03358, 0.04610] | 0.00195 | [0.02939, 0.05044] |

The two estimands agree in direction and seed stability. The verifier also
reconstructs the original pooled source metrics from committed per-path and
per-step records, so the difference cannot be mistaken for replay drift.

## Audit Replay Correction

The initial v1 replay failed closed because the compact runner exposed
realized `flow_shares` rather than latent routing weights. v2 adapted and
explicitly labeled realized-flow-share total variation, but a later audit
found that top-level `gate_active` was not bridged to the common path schema;
its exported activation values were incorrectly zero. No efficacy decision
used that field, but v2 is not accepted as the paper audit source.

The corrected v3 replay maps and stores `gate_active` explicitly, preserves
realized flow shares, and labels their provenance. It completed 20/20 tasks,
1,200 per-path rows, and 120,000 per-step rows with no failures. Its fingerprint
is `ad9b00acc5c36067da28b795acf138b3025c4340f3000cefebd2e1b9e456698c`.
A completed rerun started no containers and reproduced summary SHA-256
`6af81c1f0c9849f31a8ce0efde998c7d71935a8b9642129aeea352c05ee27675`.

The independent verifier replays activation and high-risk activation from
every step, reconstructs each original compact-runner aggregate, validates
model/path/action hashes, and checks conserved flow. Total variation for the
compact cells is explicitly realized-flow-share TV, not a latent platform
routing probability.

## Claim Boundary

The supported statement is:

> The matched-random timing advantage of risk-aware routing is robust in both
> pre-frozen N=10 ABIDES ecology perturbations tested here.

This does not establish robustness to larger N, other matching engines, or all
risk/flow processes. It also does not restore punishment-supported collusion
wording; the separate forced-deviation diagnostic limits the target to
high-spread coordination.
